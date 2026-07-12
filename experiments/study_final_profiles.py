"""Small paired Mini study for final-layer sampling compute profiles."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import flopscope as flops
from whestbench import MLP, load_dataset

from whest_solution.sampling import spherical_propagation


PROFILES = (("iid_4096", 4096, False), ("orthogonal_4096", 4096, True), ("iid_5120", 5120, False), ("iid_5632", 5632, False), ("iid_6144", 6144, False))


def main() -> None:
    data = list(
        load_dataset("aicrowd/arc-whestbench-public-2026", revision="v1-phase1", split="mini")
    )[:10]
    results: dict[str, dict[str, float]] = {}
    for name, samples, orthogonal_blocks in PROFILES:
        final_mse: list[float] = []
        ratios: list[float] = []
        walls: list[float] = []
        for row in data:
            mlp = MLP.from_row(row, seed_protocol_version="3.0")
            target = np.asarray(row["all_layer_means"], dtype=np.float64)[-1]
            start = time.perf_counter()
            with flops.BudgetContext(flop_budget=272_000_000_000, quiet=True) as context:
                prediction = spherical_propagation(
                    mlp, samples=samples, batch_size=512, orthogonal_blocks=orthogonal_blocks
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
    output = Path("results/raw/t08_final_profile_mini10.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"mlp_count": len(data), "profiles": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
