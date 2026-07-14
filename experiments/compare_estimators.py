"""Paired comparison and deterministic MLP-bootstrap utilities."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def paired_comparison(
    rows: list[dict[str, str]], parent: str, candidate: str, *, seed: int, resamples: int
) -> dict[str, object]:
    """Compare candidate minus parent; negative deltas favor the candidate."""
    by_key = {(row["mlp_id"], row["estimator_name"]): row for row in rows}
    names = sorted(
        name
        for name, estimator in by_key
        if estimator == parent
        and (name, candidate) in by_key
        and by_key[(name, parent)]["status"] == "ok"
        and by_key[(name, candidate)]["status"] == "ok"
    )
    if not names:
        raise ValueError(f"no successful paired observations for {parent} and {candidate}")

    def values(field: str) -> np.ndarray:
        return np.array(
            [
                float(by_key[(name, candidate)][field]) - float(by_key[(name, parent)][field])
                for name in names
            ],
            dtype=np.float64,
        )

    score_delta = values("adjusted_score")
    raw_delta = values("raw_final_mse")
    compute_delta = values("compute_ratio")
    wall_delta = values("wall_ms")
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(names), size=(resamples, len(names)))
    boot_means = np.mean(score_delta[sampled], axis=1)
    return {
        "parent": parent,
        "candidate": candidate,
        "paired_mlp_count": len(names),
        "convention": "candidate_minus_parent; negative favors candidate",
        "mean_paired_adjusted_score_delta": float(np.mean(score_delta)),
        "bootstrap_95_ci": [
            float(np.quantile(boot_means, 0.025)),
            float(np.quantile(boot_means, 0.975)),
        ],
        "median_adjusted_score_delta": float(np.median(score_delta)),
        "candidate_win_fraction": float(np.mean(score_delta < 0.0)),
        "mean_raw_final_mse_delta": float(np.mean(raw_delta)),
        "mean_compute_ratio_delta": float(np.mean(compute_delta)),
        "mean_wall_ms_delta": float(np.mean(wall_delta)),
        "candidate_failures": sum(
            row["estimator_name"] == candidate and row["status"] != "ok" for row in rows
        ),
        "parent_failures": sum(
            row["estimator_name"] == parent and row["status"] != "ok" for row in rows
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--resamples", type=int, default=20_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.raw.open(newline="", encoding="utf-8") as handle:
        result = paired_comparison(
            list(csv.DictReader(handle)),
            args.parent,
            args.candidate,
            seed=args.seed,
            resamples=args.resamples,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
