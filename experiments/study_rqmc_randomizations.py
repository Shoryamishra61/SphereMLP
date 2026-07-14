"""Measure IID and randomized-LHS variability for one fixed Mini MLP."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from whestbench import MLP, load_dataset

import flopscope as flops
from whest_solution.sampling import spherical_propagation


def main() -> None:
    row = next(
        iter(load_dataset("aicrowd/arc-whestbench-public-2026", revision="v1-phase1", split="mini"))
    )
    mlp = MLP.from_row(row, seed_protocol_version="3.0")
    target = np.asarray(row["all_layer_means"], dtype=np.float64)[-1]
    results: dict[str, dict[str, float]] = {}
    for name, latin_hypercube in (("iid", False), ("lhs_rqmc", True)):
        errors: list[float] = []
        ratios: list[float] = []
        walls: list[float] = []
        for seed in (101, 202, 303, 404):
            started = time.perf_counter()
            with flops.BudgetContext(flop_budget=272_000_000_000, quiet=True) as context:
                prediction = spherical_propagation(
                    mlp,
                    samples=4096,
                    batch_size=512,
                    final_layer_only=True,
                    latin_hypercube=latin_hypercube,
                    seed=seed,
                ).predictions[-1]
            compute = context.flops_used + 1e11 * float(context.residual_wall_time_s or 0.0)
            errors.append(float(np.mean((np.asarray(prediction) - target) ** 2)))
            ratios.append(compute / 272_000_000_000)
            walls.append((time.perf_counter() - started) * 1000.0)
        results[name] = {
            "randomizations": len(errors),
            "raw_final_mse_mean": float(np.mean(errors)),
            "raw_final_mse_variance": float(np.var(errors, ddof=1)),
            "mean_compute_ratio": float(np.mean(ratios)),
            "max_compute_ratio": float(np.max(ratios)),
            "p95_wall_ms": float(np.quantile(walls, 0.95)),
        }
    output = Path("results/raw/t08_rqmc_randomizations.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
