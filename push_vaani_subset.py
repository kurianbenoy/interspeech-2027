#!/usr/bin/env python3
"""
Package `vaani_malayalam_10k_v2` as a Hugging Face dataset and push it.

The local build directory holds one WAV per row plus a JSONL manifest:

    vaani_malayalam_10k_v2/
      manifest.jsonl
      audio/00000.wav ...

Ten thousand loose WAVs make a poor hub dataset (slow clone, no viewer), so
this script rewrites the sample as parquet shards with the audio bytes inlined
in an HF `Audio`-compatible struct column. The result loads as:

    load_dataset("kurianbenoy/google-vaani-malayalam-subset", split="train")

Usage:
  python push_vaani_subset.py                 # build into build/ only
  python push_vaani_subset.py --push          # build, then upload
  python push_vaani_subset.py --push --private
"""

import argparse
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SRC_DIR = Path(__file__).resolve().parent / "vaani_malayalam_10k_v2"
BUILD_DIR = Path(__file__).resolve().parent / "build" / "hf-vaani-malayalam-subset"
DEFAULT_REPO_ID = "kurianbenoy/google-vaani-malayalam-subset"

ROWS_PER_SHARD = 1000

# String-ish fields are declared explicitly so an all-null column in a shard
# does not flip the schema between shards.
SCHEMA = pa.schema(
    [
        ("sample_id", pa.int32()),
        ("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
        ("transcript", pa.string()),
        ("duration", pa.float32()),
        ("speaker_id", pa.string()),
        ("gender", pa.string()),
        ("state", pa.string()),
        ("district", pa.string()),
        ("pincode", pa.string()),
        ("languages_known", pa.list_(pa.string())),
        ("duration_bin", pa.string()),
        ("utterance_sequence_id", pa.int64()),
        ("reference_image", pa.string()),
        ("speaker_image_hash", pa.string()),
        ("source_shard", pa.string()),
    ]
)

# The Hub only renders an audio player when the parquet schema carries the
# `huggingface` key-value metadata describing `audio` as an Audio feature.
# Without it the column is just struct<bytes, path> and shows up as base64.
# `Features.from_arrow_schema` accepts each entry only if the declared feature
# type matches the column's actual arrow type, so this must stay in sync with
# SCHEMA (asserted in build_shards).
FEATURES = {
    "sample_id": {"dtype": "int32", "_type": "Value"},
    "audio": {"_type": "Audio"},
    "transcript": {"dtype": "string", "_type": "Value"},
    "duration": {"dtype": "float32", "_type": "Value"},
    "speaker_id": {"dtype": "string", "_type": "Value"},
    "gender": {"dtype": "string", "_type": "Value"},
    "state": {"dtype": "string", "_type": "Value"},
    "district": {"dtype": "string", "_type": "Value"},
    "pincode": {"dtype": "string", "_type": "Value"},
    "languages_known": {"feature": {"dtype": "string", "_type": "Value"}, "_type": "List"},
    "duration_bin": {"dtype": "string", "_type": "Value"},
    "utterance_sequence_id": {"dtype": "int64", "_type": "Value"},
    "reference_image": {"dtype": "string", "_type": "Value"},
    "speaker_image_hash": {"dtype": "string", "_type": "Value"},
    "source_shard": {"dtype": "string", "_type": "Value"},
}

# Audio rows are ~170 KB each, so a small row group keeps the viewer's
# first-page fetch cheap.
ROWS_PER_ROW_GROUP = 50


def read_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def row_to_record(row: dict, audio_dir: Path) -> dict:
    wav = audio_dir / Path(row["audio_file"]).name
    return {
        "sample_id": row["sample_id"],
        "audio": {"bytes": wav.read_bytes(), "path": wav.name},
        "transcript": row.get("transcript"),
        "duration": row.get("duration"),
        "speaker_id": row.get("speakerID"),
        "gender": row.get("gender"),
        "state": row.get("state"),
        "district": row.get("district"),
        "pincode": row.get("pincode"),
        "languages_known": row.get("languagesKnown") or [],
        "duration_bin": row.get("duration_bin"),
        "utterance_sequence_id": row.get("UtteranceSequenceID"),
        "reference_image": row.get("referenceImage"),
        "speaker_image_hash": row.get("speakerImageHash"),
        "source_shard": row.get("source_file"),
    }


def hf_features_metadata(schema: pa.Schema) -> dict[bytes, bytes]:
    """Parquet key-value metadata that makes the Hub restore the Audio feature."""
    from datasets import Features

    declared = Features.from_dict(FEATURES).arrow_schema
    for name in schema.names:
        assert declared.field(name) == schema.field(name), (
            f"{name}: declared {declared.field(name).type} != written {schema.field(name).type}"
        )
    return {b"huggingface": json.dumps({"info": {"features": FEATURES}}).encode()}


def build_shards(rows: list[dict]) -> int:
    data_dir = BUILD_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for stale in data_dir.glob("*.parquet"):
        stale.unlink()

    chunks = [rows[i : i + ROWS_PER_SHARD] for i in range(0, len(rows), ROWS_PER_SHARD)]
    total = len(chunks)
    audio_dir = SRC_DIR / "audio"
    metadata = hf_features_metadata(SCHEMA)

    for idx, chunk in enumerate(chunks):
        records = [row_to_record(r, audio_dir) for r in chunk]
        table = pa.Table.from_pylist(records, schema=SCHEMA)
        table = table.replace_schema_metadata({**(table.schema.metadata or {}), **metadata})
        out = data_dir / f"train-{idx:05d}-of-{total:05d}.parquet"
        pq.write_table(table, out, row_group_size=ROWS_PER_ROW_GROUP)
        print(f"  {out.name}  {len(chunk)} rows  {out.stat().st_size / 1e6:.0f} MB")
    return total


def write_sidecars(rows: list[dict], summary: dict, repo_id: str) -> None:
    shutil.copyfile(SRC_DIR / "manifest.jsonl", BUILD_DIR / "manifest.jsonl")
    with (BUILD_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)

    (BUILD_DIR / ".gitattributes").write_text(
        "*.parquet filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8"
    )

    hours = sum(r.get("duration") or 0 for r in rows) / 3600
    card = DATASET_CARD
    for key, value in {
        "__REPO_ID__": repo_id,
        "__N__": f"{len(rows):,}",
        "__HOURS__": f"{hours:.1f}",
        "__SPEAKERS__": str(summary.get("unique_speakers", "n/a")),
        "__REVISION__": str(summary.get("revision", "n/a")),
        "__DISTRICTS__": "\n".join(
            f"- {d}: {c:,}"
            for d, c in sorted(summary["district_counts"].items(), key=lambda kv: -kv[1])
        ),
    }.items():
        card = card.replace(key, value)
    (BUILD_DIR / "README.md").write_text(card, encoding="utf-8")


DATASET_CARD = """---
license: cc-by-4.0
language:
- ml
task_categories:
- automatic-speech-recognition
- audio-classification
pretty_name: Google VAANI Malayalam Subset (10k)
size_categories:
- 10K<n<100K
source_datasets:
- ARTPARK-IISc/VAANI
tags:
- speech
- malayalam
- vaani
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---

# Google VAANI Malayalam Subset (10k)

A stratified 10,000-utterance subset of the Malayalam portion of
[`ARTPARK-IISc/VAANI`](https://huggingface.co/datasets/ARTPARK-IISc/VAANI)
(aka the Google/ARTPARK VAANI speech corpus), repackaged as parquet for fast
loading and viewer support.

- **Rows:** __N__ (~__HOURS__ hours of audio)
- **Audio:** 16 kHz, mono, 16-bit PCM WAV, embedded in the parquet files
- **Speakers:** 892 distinct `speaker_id` values (3,334 rows have `UNKNOWN` as the
  speaker id), at most 40 utterances per speaker
- **Source revision:** `__REVISION__`

## Sampling

Utterances were drawn from the transcribed Malayalam shards of VAANI,
stratified by district × gender (district/gender cells capped at 1,500 rows,
speakers at 60 rows) to keep the subset demographically balanced rather than
proportional to the corpus. Distribution in this subset:

__DISTRICTS__

Note that a handful of rows carry a Karnataka district label (Bangalore,
Belgaum, Bellary, Gulbarga, Nalgonda, Shimoga) even though the audio is
Malayalam — these are kept as-is from the source metadata.

## Fields

| Column | Description |
| --- | --- |
| `sample_id` | Row id, stable within this subset |
| `audio` | Audio feature (bytes + filename) |
| `transcript` | Malayalam transcript; `<static>`…`</static>` marks the spoken segment |
| `duration` | Duration in seconds |
| `speaker_id` | Source speaker identifier |
| `gender`, `state`, `district`, `pincode` | Speaker metadata from the source shard |
| `languages_known` | Languages the speaker reported |
| `duration_bin` | `<=4s`, `4–8s`, or `>8s` |
| `utterance_sequence_id` | Position of the utterance in the source recording |
| `reference_image`, `speaker_image_hash` | Source image metadata (images are **not** included) |
| `source_shard` | Original VAANI parquet shard |

`manifest.jsonl` and `summary.json` are included for provenance.

## Usage

```python
from datasets import load_dataset

ds = load_dataset("__REPO_ID__", split="train")
print(ds[0]["transcript"], ds[0]["audio"]["sampling_rate"])
```

## Attribution and license

The audio and transcripts come from VAANI, released by AI4Bharat / ARTPARK-IISc
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This subset is
redistributed under the same license; please cite the original work.

```bibtex
@inproceedings{vaani2025,
  title     = {{VAANI}: A Large-Scale Multi-District, Multi-Dialect Speech Dataset for Indic Languages},
  author    = {{AI4Bharat} and {ARTPARK-IISc}},
  year      = {2025}
}
```
"""


def push(repo_id: str, private: bool) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=BUILD_DIR,
        commit_message="Add 10k-utterance Malayalam VAANI subset",
    )
    print(f"pushed https://huggingface.co/datasets/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--push", action="store_true", help="upload after building")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(SRC_DIR / "manifest.jsonl")
    summary = json.loads((SRC_DIR / "summary.json").read_text(encoding="utf-8"))
    print(f"building {len(rows)} rows into {BUILD_DIR}")
    build_shards(rows)
    write_sidecars(rows, summary, args.repo_id)
    print("build complete")

    if args.push:
        push(args.repo_id, args.private)


if __name__ == "__main__":
    main()
