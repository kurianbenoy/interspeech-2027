#!/usr/bin/env python3
"""
Create a reproducible, stratified sample of transcribed Malayalam VAANI audio.

The previous streaming implementation used `datasets` with a large shuffle
buffer. VAANI shards are ~250 MB each and the streaming parquet reader ended up
holding multiple shard-sized buffers in RAM, which reliably triggered the OOM
killer on a 7.6 GB machine.

This version processes one local shard at a time:

  OUT_DIR/
    manifest.jsonl
    audio/00000.wav ...
    shard_map.json          # district/gender bounds for every shard
    checkpoint.json         # resume state
    summary.json            # final distribution

Install:
  pip install -U huggingface_hub pyarrow

If access requires authentication:
  hf auth login
"""

import argparse
import io
import json
import math
import os
import random
import re
import shutil
import struct
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

# Must be set before huggingface_hub reads its constants.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import close_session, http_backoff


REPO_ID = "ARTPARK-IISc/VAANI"
CONFIG = "Malayalam"
DEFAULT_REVISION = "9e6ed4f94f7060e633cbc500480623ae7521522c"  # pinned main as of Nov 2026

# These are the districts with enough Malayalam VAANI data to matter for a
# 10k sample. Thiruvananthapuram lives in the boundary shards (146-153), so it
# is covered by the boundary pass rather than a single-district group.
MAIN_DISTRICTS = ["Kasaragod", "Kozhikode", "Wayanad", "Kollam"]
EXPECTED_CELLS = 2 * 5  # five Kerala districts x two genders

PARQUET_COLUMNS = [
    "audio",
    "language",
    "duration",
    "speakerID",
    "languagesKnown",
    "gender",
    "state",
    "district",
    "pincode",
    "isTranscriptionAvailable",
    "transcript",
    "referenceImage",
    "speakerImageHash",
    "UtteranceSequenceID",
]
AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".m4a", ".mp3", ".opus", ".webm"}


# ---------------------------------------------------------------------------
# parquet footer helpers (range requests, ~40 KB per shard)
# ---------------------------------------------------------------------------

class _RangeFile(io.RawIOBase):
    """Minimal seekable HTTP range reader so pyarrow only fetches the footer."""

    def __init__(self, url, headers, size, window=64 * 1024):
        self.url = url
        self.headers = headers
        self.size = size
        self.pos = 0
        self.window = window
        self._cache_start = 0
        self._cache = b""

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        elif whence == 2:
            self.pos = self.size + offset
        else:
            raise ValueError(whence)
        return self.pos

    def _fetch(self, start, end):
        r = http_backoff(
            "GET",
            self.url,
            headers={**self.headers, "Range": f"bytes={start}-{end - 1}"},
            timeout=120,
        )
        try:
            if r.status_code not in (200, 206):
                raise RuntimeError(f"range GET returned HTTP {r.status_code}")
            data = r.content
        finally:
            r.close()
        if r.status_code == 200:  # server ignored Range; be safe
            data = data[start:end]
        return data

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        n = min(len(b), self.size - self.pos)
        start = self.pos
        if self._cache_start <= start and start + n <= self._cache_start + len(self._cache):
            data = self._cache[start - self._cache_start : start - self._cache_start + n]
        else:
            window_start = max(0, (start // self.window) * self.window)
            window_end = min(self.size, max(start + n, window_start + self.window))
            self._cache = self._fetch(window_start, window_end)
            self._cache_start = window_start
            data = self._cache[start - window_start : start - window_start + n]
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


def _footer_metadata(url, headers, size):
    """Read a parquet footer with two range requests and return its metadata."""
    tail_start = size - 8
    r = http_backoff(
        "GET", url, headers={**headers, "Range": f"bytes={tail_start}-{size - 1}"}, timeout=120
    )
    try:
        if r.status_code not in (200, 206) or len(r.content) < 8:
            raise RuntimeError(f"tail range GET returned HTTP {r.status_code}")
        tail = r.content
    finally:
        r.close()

    footer_len = struct.unpack("<I", tail[-8:-4])[0]
    footer_start = size - 8 - footer_len
    r = http_backoff(
        "GET", url, headers={**headers, "Range": f"bytes={footer_start}-{size - 1}"}, timeout=120
    )
    try:
        if r.status_code not in (200, 206):
            raise RuntimeError(f"footer range GET returned HTTP {r.status_code}")
        footer = r.content[:-8]
    finally:
        r.close()

    buf = footer + struct.pack("<I", footer_len) + b"PAR1"
    return pq.read_metadata(pa.BufferReader(buf))


def _shard_stats(api_headers, base_url, shard):
    """Return {rows, district_min/max, gender_min/max} without downloading audio."""
    md = _footer_metadata(f"{base_url}/{shard['file']}", api_headers, shard["size"])
    out = {
        "file": shard["file"],
        "size": shard["size"],
        "rows": md.num_rows,
        "row_groups": md.num_row_groups,
    }
    for name in ("district", "gender"):
        lo = hi = None
        for rg in range(md.num_row_groups):
            for j in range(md.num_columns):
                col = md.row_group(rg).column(j)
                if col.path_in_schema != name or col.statistics is None:
                    continue
                try:
                    vmin, vmax = col.statistics.min, col.statistics.max
                except Exception:
                    continue
                lo = vmin if lo is None else min(lo, vmin)
                hi = vmax if hi is None else max(hi, vmax)
        out[f"{name}_min"] = lo
        out[f"{name}_max"] = hi
    return out


def build_shard_map(api, revision, out_dir, workers=8):
    """Fetch district/gender bounds for all shards; cache the result."""
    cache_path = out_dir / "shard_map.json"
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            if cached.get("repo") == REPO_ID and cached.get("revision") == revision:
                print(f"Using cached shard map ({len(cached['shards'])} shards)")
                return cached["shards"]
        except Exception:
            pass

    print("Building shard map from parquet footers (no audio download)...", flush=True)
    info = api.dataset_info(REPO_ID, files_metadata=True)
    shards = [
        {"file": s.rfilename, "size": s.size}
        for s in info.siblings
        if s.rfilename.startswith(f"audio/{CONFIG}/") and s.rfilename.endswith(".parquet")
    ]
    shards.sort(key=lambda s: s["file"])
    if not shards:
        raise RuntimeError(f"No parquet shards found for {REPO_ID}/{CONFIG}")

    headers = api._build_hf_headers()
    base_url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/{revision}"
    results = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_shard_stats, headers, base_url, s): s for s in shards}
        for i, fut in enumerate(as_completed(futures), 1):
            shard = futures[fut]
            try:
                results.append(fut.result())
            except Exception as exc:
                print(f"WARNING: could not inspect {shard['file']}: {exc}", flush=True)
            if i % 20 == 0:
                print(f"  inspected {i}/{len(shards)} shards ({time.time() - t0:.0f}s)", flush=True)

    results.sort(key=lambda s: s["file"])
    cache_path.write_text(
        json.dumps(
            {"repo": REPO_ID, "revision": revision, "built_at": time.time(), "shards": results},
            indent=1,
        )
    )
    print(f"Shard map: {len(results)} shards in {time.time() - t0:.0f}s -> {cache_path}")
    return results


# ---------------------------------------------------------------------------
# sampling state / checkpointing
# ---------------------------------------------------------------------------

@dataclass
class SampleState:
    saved: int = 0
    cell_counts: Counter = field(default_factory=Counter)
    speaker_counts: Counter = field(default_factory=Counter)
    district_counts: Counter = field(default_factory=Counter)
    gender_counts: Counter = field(default_factory=Counter)
    duration_bins: Counter = field(default_factory=Counter)
    seen: set = field(default_factory=set)
    processed_shards: set = field(default_factory=set)
    skipped_shards: list = field(default_factory=list)

    def to_json(self):
        return {
            "saved": self.saved,
            "cell_counts": {f"{d}|{g}": n for (d, g), n in self.cell_counts.items()},
            "speaker_counts": dict(self.speaker_counts),
            "district_counts": dict(self.district_counts),
            "gender_counts": dict(self.gender_counts),
            "duration_bins": dict(self.duration_bins),
            "seen": sorted(self.seen),
            "processed_shards": sorted(self.processed_shards),
            "skipped_shards": self.skipped_shards,
        }

    @classmethod
    def from_json(cls, data):
        st = cls()
        st.saved = int(data.get("saved", 0))
        st.cell_counts = Counter(
            {tuple(k.split("|", 1)): v for k, v in data.get("cell_counts", {}).items()}
        )
        st.speaker_counts = Counter(data.get("speaker_counts", {}))
        st.district_counts = Counter(data.get("district_counts", {}))
        st.gender_counts = Counter(data.get("gender_counts", {}))
        st.duration_bins = Counter(data.get("duration_bins", {}))
        st.seen = set(data.get("seen", []))
        st.processed_shards = set(data.get("processed_shards", []))
        st.skipped_shards = list(data.get("skipped_shards", []))
        return st


def save_checkpoint(path, state):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state.to_json(), ensure_ascii=False))
    os.replace(tmp, path)


def truncate_outputs(out_dir, saved):
    """Drop any samples written after the last checkpoint."""
    manifest = out_dir / "manifest.jsonl"
    if manifest.exists():
        lines = manifest.read_text(encoding="utf-8").splitlines()
        if len(lines) > saved:
            with manifest.open("w", encoding="utf-8") as f:
                for line in lines[:saved]:
                    f.write(line + "\n")
    audio_dir = out_dir / "audio"
    if audio_dir.exists():
        for path in audio_dir.iterdir():
            m = re.match(r"^(\d+)", path.stem)
            if m and int(m.group(1)) >= saved:
                path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# shard download / row processing
# ---------------------------------------------------------------------------

def download_shard(shard, revision, shards_dir, attempts=5):
    last = None
    for attempt in range(1, attempts + 1):
        try:
            path = hf_hub_download(
                repo_id=REPO_ID,
                filename=shard["file"],
                repo_type="dataset",
                revision=revision,
                local_dir=str(shards_dir),
            )
            path = Path(path)
            if not path.exists():
                raise RuntimeError("download returned a missing path")
            if path.stat().st_size != shard["size"]:
                raise RuntimeError(
                    f"size mismatch: {path.stat().st_size} != {shard['size']}"
                )
            return path
        except Exception as exc:
            last = exc
            wait = min(30, 2 ** attempt)
            print(
                f"  download failed ({attempt}/{attempts}) for {shard['file']}: {exc}; "
                f"retrying in {wait}s",
                flush=True,
            )
            time.sleep(wait)
    raise RuntimeError(f"could not download {shard['file']}: {last}")


def clean(value, fallback="UNKNOWN"):
    if value is None:
        return fallback
    value = str(value).strip()
    return value if value and value.lower() not in {"na", "none", "nan", "n/a"} else fallback


def missing(value):
    return value is None or str(value).strip().lower() in {"", "na", "none", "nan", "n/a"}


def is_transcribed(row):
    flag = str(row.get("isTranscriptionAvailable") or "").strip().lower()
    transcript = row.get("transcript")
    return flag in {"yes", "true", "1"} and isinstance(transcript, str) and transcript.strip()


def audio_extension(audio):
    path = audio.get("path")
    if path:
        suffix = Path(path).suffix.lower()
        if suffix in AUDIO_SUFFIXES:
            return suffix
    data = audio.get("bytes") or b""
    if data.startswith(b"RIFF"):
        return ".wav"
    if data.startswith(b"fLaC"):
        return ".flac"
    if data.startswith(b"OggS"):
        return ".ogg"
    if len(data) > 12 and data[4:8] == b"ftyp":
        return ".m4a"
    return ".audio"


def save_audio(audio, destination):
    data = audio.get("bytes")
    if data is None:
        raise RuntimeError("row has no audio bytes")
    if isinstance(data, memoryview):
        data = data.tobytes()
    destination.write_bytes(bytes(data))


def process_shard(
    path,
    shard,
    state,
    out_dir,
    per_cell_cap,
    max_per_speaker,
    target_n,
    rng,
    batch_size,
):
    """Process one local shard. Returns number of samples added."""
    manifest_path = out_dir / "manifest.jsonl"
    audio_dir = out_dir / "audio"
    added = 0
    pf = pq.ParquetFile(str(path))

    with manifest_path.open("a", encoding="utf-8") as manifest:
        for batch in pf.iter_batches(batch_size=batch_size, columns=PARQUET_COLUMNS):
            rows = batch.to_pylist()
            rng.shuffle(rows)
            for row in rows:
                if state.saved >= target_n:
                    break
                if not is_transcribed(row):
                    continue

                district = clean(row.get("district"))
                gender = clean(row.get("gender"))
                cell = (district, gender)
                if state.cell_counts[cell] >= per_cell_cap:
                    continue

                raw_speaker = row.get("speakerID")
                speaker = None if missing(raw_speaker) else str(raw_speaker).strip()
                if speaker is not None and state.speaker_counts[speaker] >= max_per_speaker:
                    continue

                audio = row.get("audio")
                if not isinstance(audio, dict) or audio.get("bytes") is None:
                    continue

                seq = row.get("UtteranceSequenceID")
                ref = row.get("referenceImage") or ""
                dedup_key = (speaker or "NA", str(ref), -1 if seq is None else int(seq))
                if dedup_key in state.seen:
                    continue

                filename = f"{state.saved:05d}{audio_extension(audio)}"
                try:
                    save_audio(audio, audio_dir / filename)
                except Exception:
                    continue

                duration = row.get("duration")
                try:
                    duration = float(duration)
                    duration_bin = (
                        "<=4s" if duration <= 4 else ("4–8s" if duration <= 8 else ">8s")
                    )
                except (TypeError, ValueError):
                    duration = None
                    duration_bin = "unknown"

                record = {
                    "sample_id": state.saved,
                    "audio_file": str(Path("audio") / filename),
                    "transcript": row["transcript"],
                    "duration": duration,
                    "speakerID": speaker if speaker is not None else "UNKNOWN",
                    "gender": gender,
                    "state": clean(row.get("state")),
                    "district": district,
                    "pincode": clean(row.get("pincode")),
                    "languagesKnown": row.get("languagesKnown"),
                    "isTranscriptionAvailable": row.get("isTranscriptionAvailable"),
                    "referenceImage": row.get("referenceImage"),
                    "speakerImageHash": row.get("speakerImageHash"),
                    "UtteranceSequenceID": row.get("UtteranceSequenceID"),
                    "duration_bin": duration_bin,
                    "source_file": shard["file"],
                }
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")

                state.seen.add(dedup_key)
                state.cell_counts[cell] += 1
                if speaker is not None:
                    state.speaker_counts[speaker] += 1
                state.district_counts[district] += 1
                state.gender_counts[gender] += 1
                state.duration_bins[duration_bin] += 1
                state.saved += 1
                added += 1

                if state.saved % 50 == 0:
                    manifest.flush()
                    print(
                        f"  saved {state.saved:,}/{target_n:,} "
                        f"(cell {district}/{gender}: {state.cell_counts[cell]})",
                        flush=True,
                    )

            if state.saved >= target_n:
                break
        manifest.flush()
    return added


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------

def plan_shards(shard_map, seed):
    """Boundary/other shards first, then single-district shards round-robin."""
    rng = random.Random(seed)
    boundary = []
    single = {}
    for shard in shard_map:
        dmin, dmax = shard.get("district_min"), shard.get("district_max")
        if dmin is None or dmin != dmax:
            boundary.append(shard)
        else:
            single.setdefault(dmin, []).append(shard)

    rng.shuffle(boundary)
    for district in single:
        rng.shuffle(single[district])

    order = list(MAIN_DISTRICTS)
    for district in sorted(single):
        if district not in order:
            order.append(district)

    round_robin = []
    while any(single.get(d) for d in order):
        for district in order:
            if single.get(district):
                round_robin.append(single[district].pop(0))

    return boundary + round_robin


def shard_is_saturated(shard, state, per_cell_cap):
    dmin, dmax = shard.get("district_min"), shard.get("district_max")
    if dmin is None or dmin != dmax:
        return False
    if dmin == "UNKNOWN":
        return False
    return all(
        state.cell_counts[(dmin, gender)] >= per_cell_cap
        for gender in ("Female", "Male")
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="vaani_malayalam_10k_v2")
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="Dataset revision (commit SHA recommended). Default: pinned Nov 2026 SHA.",
    )
    parser.add_argument("--max-per-speaker", type=int, default=60)
    parser.add_argument(
        "--expected-cells",
        type=int,
        default=EXPECTED_CELLS,
        help="Number of district/gender cells used to set the default per-cell cap.",
    )
    parser.add_argument(
        "--per-cell-cap",
        type=int,
        default=0,
        help="Hard cap per district/gender cell. 0 = 1.5 * n / expected-cells.",
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=None,
        help="Deprecated; kept so old command lines still work. Ignored.",
    )
    parser.add_argument("--keep-shards", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if args.buffer_size is not None:
        print("Note: --buffer-size is ignored by this shard-by-shard implementation.")

    out_dir = Path(args.out)
    audio_dir = out_dir / "audio"
    shards_dir = out_dir / "shards"
    checkpoint_path = out_dir / "checkpoint.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    shards_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.jsonl").touch()

    per_cell_cap = args.per_cell_cap or math.ceil(args.n / args.expected_cells * 1.5)

    api = HfApi()
    shard_map = build_shard_map(api, args.revision, out_dir, workers=args.workers)
    if not shard_map:
        raise RuntimeError("empty shard map")

    state = SampleState()
    if checkpoint_path.exists():
        state = SampleState.from_json(json.loads(checkpoint_path.read_text()))
        truncate_outputs(out_dir, state.saved)
        print(
            f"Resuming from checkpoint: {state.saved:,} samples, "
            f"{len(state.processed_shards)} shards already processed"
        )
    if state.saved >= args.n:
        print(f"Target already reached ({state.saved:,} samples). Nothing to do.")
        return

    plan = plan_shards(shard_map, args.seed)
    total = len(plan)
    t0 = time.time()
    rng = random.Random(args.seed ^ 0x5A17)
    print(
        f"Plan: {total} shards; target {args.n:,}; per-cell cap {per_cell_cap}; "
        f"max/speaker {args.max_per_speaker}"
    )
    print(f"Output: {out_dir.resolve()}", flush=True)

    for idx, shard in enumerate(plan, 1):
        if state.saved >= args.n:
            break
        if shard["file"] in state.processed_shards:
            continue
        if shard_is_saturated(shard, state, per_cell_cap):
            state.skipped_shards.append(shard["file"])
            continue

        print(
            f"\n[{idx}/{total}] {Path(shard['file']).name} "
            f"({shard.get('district_min')} -> {shard.get('district_max')}, "
            f"{shard['size'] / 1e6:.0f} MB)",
            flush=True,
        )
        shard_t0 = time.time()
        try:
            path = download_shard(shard, args.revision, shards_dir)
        except Exception as exc:
            print(f"  SKIP: {exc}", flush=True)
            state.skipped_shards.append(shard["file"])
            save_checkpoint(checkpoint_path, state)
            continue

        try:
            added = process_shard(
                path,
                shard,
                state,
                out_dir,
                per_cell_cap,
                args.max_per_speaker,
                args.n,
                rng,
                args.batch_size,
            )
        finally:
            if not args.keep_shards:
                try:
                    path.unlink(missing_ok=True)
                except Exception:
                    pass

        state.processed_shards.add(shard["file"])
        save_checkpoint(checkpoint_path, state)
        elapsed = time.time() - shard_t0
        print(
            f"  +{added} samples in {elapsed:.0f}s | total {state.saved:,}/{args.n:,} "
            f"| elapsed {(time.time() - t0) / 60:.1f} min",
            flush=True,
        )

    # Final checkpoint / summary
    save_checkpoint(checkpoint_path, state)
    summary = {
        "saved": state.saved,
        "target": args.n,
        "per_cell_cap": per_cell_cap,
        "expected_cells": args.expected_cells,
        "max_per_speaker": args.max_per_speaker,
        "revision": args.revision,
        "district_counts": dict(sorted(state.district_counts.items())),
        "gender_counts": dict(sorted(state.gender_counts.items())),
        "cell_counts": {f"{d}|{g}": n for (d, g), n in sorted(state.cell_counts.items())},
        "duration_bins": dict(sorted(state.duration_bins.items())),
        "unique_speakers": len(state.speaker_counts),
        "processed_shards": sorted(state.processed_shards),
        "skipped_shards": state.skipped_shards,
        "elapsed_seconds": round(time.time() - t0, 1),
        "manifest": str((out_dir / "manifest.jsonl").resolve()),
        "audio_dir": str(audio_dir.resolve()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))

    print(f"\nSaved {state.saved:,} samples")
    print(f"Manifest: {out_dir / 'manifest.jsonl'}")
    print(f"Audio directory: {audio_dir}")
    print(f"District counts: {dict(state.district_counts)}")
    print(f"Gender counts: {dict(state.gender_counts)}")
    print(f"Duration bins: {dict(state.duration_bins)}")
    print(f"Unique speakers: {len(state.speaker_counts):,}")
    print(f"Processed shards: {len(state.processed_shards):,}")
    print(f"Skipped shards: {len(state.skipped_shards):,}")
    print(f"Per-cell cap: {per_cell_cap}")
    print(f"Max per speaker: {args.max_per_speaker}")

    if state.saved < args.n:
        print(
            "\nNOTE: The dataset ended before reaching the target. "
            "Increase --per-cell-cap or inspect summary.json/skipped_shards."
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            close_session()
        except Exception:
            pass
