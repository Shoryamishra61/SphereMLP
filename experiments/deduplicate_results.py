"""Canonicalize interrupted benchmark output without selecting on performance.

The first completed row for each (MLP, estimator) pair is retained.  This
order-based rule is independent of accuracy, runtime, and compute, and the
original append-only capture is preserved by the caller.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def canonicalize(input_path: Path, output_path: Path) -> tuple[int, int]:
    with input_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"missing CSV header: {input_path}")
        fields = reader.fieldnames
        rows: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        duplicate_count = 0
        for row in reader:
            key = (row["mlp_id"], row["estimator_name"])
            if key in seen:
                duplicate_count += 1
                continue
            seen.add(key)
            rows.append(row)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows), duplicate_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    retained, removed = canonicalize(args.input, args.output)
    print(f"retained={retained} removed_duplicates={removed}")


if __name__ == "__main__":
    main()
