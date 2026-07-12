"""Small paired Mini study for final-layer sampling compute profiles."""

from __future__ import annotations

import json
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import flopscope as flops
from whestbench import MLP, load_dataset

from whest_solution.sampling import spherical_propagation


PROFILES = (
    ("iid_4096", 4096, False, False),
    ("orthogonal_4096", 4096, True, False),
    ("lhs_rqmc_4096", 4096, False, True),
    ("iid_5120", 5120, False, False),
    ("iid_5632", 5632, False, False),
    ("iid_6144", 6144, False, False),
)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--profiles", nargs="*", default=None)
    parser.add_argument("--output", default="results/raw/t08_final_profile_mini10.json")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    profiles = (
        PROFILES
        if args.profiles is None
        else tuple(profile for profile in PROFILES if profile[0] in set(args.profiles))
    )
    if not profiles:
        raise ValueError("no known profiles selected")
    data = list(
        load_dataset("aicrowd/arc-whestbench-public-2026", revision="v1-phase1", split="mini")
    )[: args.limit]
    results: dict[str, dict[str, float]] = {}
    for name, samples, orthogonal_blocks, latin_hypercube in profiles:
        final_mse: list[float] = []
        ratios: list[float] = []
        walls: list[float] = []
        for row in data:
            mlp = MLP.from_row(row, seed_protocol_version="3.0")
            target = np.asarray(row["all_layer_means"], dtype=np.float64)[-1]
            start = time.perf_counter()
            with flops.BudgetContext(flop_budget=272_000_000_000, quiet=True) as context:
                prediction = spherical_propagation(
                    mlp,
                    samples=samples,
                    batch_size=512,
                    orthogonal_blocks=orthogonal_blocks,
                    latin_hypercube=latin_hypercube,
                ).predictions[-1]
            compute = context.flops_used + 1e11 * float(context.residual_wall_time_s or 0.0)
            final_mse.append(float(np.mean((np.asarray(prediction) - target) ** 2)))
            ratios.append(compute / 272_000_000_000)
            walls.append((time.perf_counter() - start) * 1000.0)
        results[name] = {
            "raw_final_mse_mean": float(np.mean(final_mse)),
            "adjusted_score_mean": float(np.mean(final_mse) * max(0.1, np.mean(ratios))),
            "mean_compute_ratio": float(np.mean(ratios)),
            "max_compute_ratio": float(np.max(ratios)),
            "p95_wall_ms": float(np.quantile(walls, 0.95)),
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"mlp_count": len(data), "profiles": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
