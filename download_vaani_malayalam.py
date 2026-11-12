#!/usr/bin/env python3
"""
Create a reproducible, stratified sample of transcribed Malayalam VAANI audio.

Output:
  OUT_DIR/
    manifest.jsonl
    audio/
      00000.wav
      ...

Install:
  pip install -U datasets

If access requires authentication:
  hf auth login
"""

import argparse
import json
import math
import shutil
from collections import Counter
from pathlib import Path

from datasets import Audio, load_dataset


REPO_ID = "ARTPARK-IISc/VAANI"
CONFIG = "Malayalam"


def clean(value, fallback="UNKNOWN"):
    if value is None:
        return fallback
    value = str(value).strip()
    return value if value and value.lower() not in {"na", "none", "nan"} else fallback


def is_transcribed(row):
    flag = clean(row.get("isTranscriptionAvailable"), "").lower()
    transcript = row.get("transcript")
    return flag in {"yes", "true", "1"} and isinstance(transcript, str) and transcript.strip()


def audio_extension(audio):
    """Infer a useful file extension from the source path or audio bytes."""
    path = audio.get("path")
    if path:
        suffix = Path(path).suffix.lower()
        if suffix:
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
    """Save raw streamed audio bytes, or copy a local audio path."""
    data = audio.get("bytes")
    source_path = audio.get("path")

    if data is not None:
        destination.write_bytes(data)
    elif source_path and Path(source_path).is_file():
        shutil.copyfile(source_path, destination)
    else:
        raise RuntimeError(
            f"Could not obtain audio bytes or a local audio file: {audio}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="vaani_malayalam_10k")
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--max-per-speaker", type=int, default=60)
    parser.add_argument(
        "--revision",
        default="main",
        help="Use a Hub commit SHA for a fixed dataset version; default: main.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    # VAANI's card documents the Malayalam config and train split.
    ds = load_dataset(
        REPO_ID,
        CONFIG,
        split="train",
        streaming=True,
        revision=args.revision,
    )

    # Don't decode to waveforms in Python; save the original audio bytes.
    ds = ds.cast_column("audio", Audio(decode=False))
    ds = ds.shuffle(seed=args.seed, buffer_size=args.buffer_size)

    # The viewer shows 13 districts and 2 gender categories. This per-cell cap
    # encourages district/gender coverage without imposing a population estimate.
    expected_cells = 13 * 2
    per_cell_cap = math.ceil(args.n / expected_cells)

    cell_counts = Counter()
    speaker_counts = Counter()
    duration_bins = Counter()
    district_counts = Counter()
    gender_counts = Counter()
    seen = set()
    saved = 0

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for row in ds:
            if not is_transcribed(row):
                continue

            district = clean(row.get("district"))
            gender = clean(row.get("gender"))
            speaker = clean(row.get("speakerID"))
            cell = (district, gender)

            if cell_counts[cell] >= per_cell_cap:
                continue
            if speaker_counts[speaker] >= args.max_per_speaker:
                continue

            # Avoid repeated rows for the same speaker/image/utterance sequence.
            # Keep distinct sequence IDs, which may represent separate utterances.
            dedup_key = (
                speaker,
                clean(row.get("referenceImage"), ""),
                clean(row.get("UtteranceSequenceID"), ""),
            )
            if dedup_key in seen:
                continue

            audio = row.get("audio")
            if not isinstance(audio, dict):
                continue

            filename = f"{saved:05d}{audio_extension(audio)}"
            audio_path = audio_dir / filename
            try:
                save_audio(audio, audio_path)
            except RuntimeError:
                continue

            duration = row.get("duration")
            try:
                duration = float(duration)
                duration_bin = "<=4s" if duration <= 4 else ("4–8s" if duration <= 8 else ">8s")
            except (TypeError, ValueError):
                duration = None
                duration_bin = "unknown"

            # Keep metadata, not the audio payload, in JSONL.
            record = {
                "sample_id": saved,
                "audio_file": str(Path("audio") / filename),
                "transcript": row["transcript"],
                "duration": duration,
                "speakerID": speaker,
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
            }
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")

            seen.add(dedup_key)
            cell_counts[cell] += 1
            speaker_counts[speaker] += 1
            duration_bins[duration_bin] += 1
            district_counts[district] += 1
            gender_counts[gender] += 1
            saved += 1

            if saved % 500 == 0:
                print(f"Saved {saved:,} samples")

            if saved >= args.n:
                break

    print(f"\nSaved {saved:,} samples")
    print(f"Manifest: {manifest_path}")
    print(f"Audio directory: {audio_dir}")
    print(f"District counts: {dict(district_counts)}")
    print(f"Gender counts: {dict(gender_counts)}")
    print(f"Duration bins: {dict(duration_bins)}")
    print(f"Unique speakers: {len(speaker_counts):,}")
    print(f"Max per district/gender cell: {per_cell_cap}")
    print(f"Max per speaker: {args.max_per_speaker}")

    if saved < args.n:
        print(
            "\nNOTE: The stream ended before reaching the target. "
            "The resulting sample has fewer labeled rows than requested; "
            "it did not silently add unlabeled audio."
        )


if __name__ == "__main__":
    main()

