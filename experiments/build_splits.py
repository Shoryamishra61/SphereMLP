"""Build the immutable network-level Full development split."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

DEFAULT_REPO = "aicrowd/arc-whestbench-public-2026"
DEFAULT_REVISION = "v1-phase1"
PARTITION_SIZES = {
    "train": 700,
    "validation": 150,
    "calibration": 75,
    "internal_holdout": 75,
}


def deterministic_partitions(names: list[str], revision: str) -> dict[str, list[str]]:
    if len(names) != sum(PARTITION_SIZES.values()):
        raise ValueError(f"expected 1000 names, got {len(names)}")
    if len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError("MLP names must be non-empty and unique")
    ordered = sorted(
        names,
        key=lambda name: (sha256(f"{revision}\0{name}".encode()).hexdigest(), name),
    )
    result: dict[str, list[str]] = {}
    start = 0
    for partition, size in PARTITION_SIZES.items():
        result[partition] = ordered[start : start + size]
        start += size
    return result


def load_full_names(repo: str, revision: str) -> tuple[list[str], dict[str, object]]:
    """Project only names via HTTP range reads over the pinned parquet shards."""
    files = sorted(
        name
        for name in HfApi().list_repo_files(repo, repo_type="dataset", revision=revision)
        if name.startswith("data/full-") and name.endswith(".parquet")
    )
    if len(files) != 28:
        raise ValueError(f"expected 28 Full parquet shards, got {len(files)}")
    filesystem = HfFileSystem()
    names: list[str] = []
    for filename in files:
        path = f"datasets/{repo}@{revision}/{filename}"
        with filesystem.open(path, "rb") as handle:
            table = pq.read_table(handle, columns=["mlp_name"])
        names.extend(str(name) for name in table.column("mlp_name").to_pylist())
    metadata_path = f"datasets/{repo}@{revision}/metadata.json"
    with filesystem.open(metadata_path, "rb") as handle:
        dataset_metadata = json.loads(handle.read().decode("utf-8"))
    return names, dataset_metadata


def build_manifest(repo: str, revision: str) -> dict[str, object]:
    names, dataset_metadata = load_full_names(repo, revision)
    partitions = deterministic_partitions(names, revision)
    return {
        "schema_version": "1.0",
        "status": "FROZEN",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": f"hf://{repo}@{revision}",
        "dataset_revision": revision,
        "source_split": "full",
        "source_count": len(names),
        "seed_protocol": dataset_metadata.get("seed_protocol"),
        "name_projection": "pyarrow parquet mlp_name column via pinned HF range reads",
        "partition_method": "sort by sha256(revision + NUL + mlp_name), then slice",
        "partition_sizes": PARTITION_SIZES,
        "partitions": partitions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--output", type=Path, default=Path("manifests/dataset_splits.json"))
    args = parser.parse_args()
    manifest = build_manifest(args.repo, args.revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "source_count": manifest["source_count"],
                "partition_sizes": manifest["partition_sizes"],
            }
        )
    )


if __name__ == "__main__":
    main()
