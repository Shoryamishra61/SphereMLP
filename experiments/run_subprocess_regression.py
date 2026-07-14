"""Exercise every Mini MLP through the official isolated worker API.

The Whest CLI fixes its worker handshake at five seconds.  This host's
installed worker imports take longer before user code is loaded, so this
development regression makes that host-only startup allowance explicit while
leaving the official prediction timeout and budget unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from whestbench import MLP, SetupContext, load_dataset
from whestbench.runner import EstimatorEntrypoint, ResourceLimits, SubprocessRunner

from whest_solution.guards import valid_prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--setup-timeout", type=float, default=15.0)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    dataset = load_dataset("aicrowd/arc-whestbench-public-2026", revision="v1-phase1", split="mini")
    rows = list(dataset)[: args.limit]
    if len(rows) != args.limit:
        raise ValueError(f"expected {args.limit} Mini rows, found {len(rows)}")
    runner = SubprocessRunner()
    limits = ResourceLimits(
        setup_timeout_s=args.setup_timeout,
        predict_timeout_s=30.0,
        memory_limit_mb=65_536,
        flop_budget=272_000_000_000,
        wall_time_limit_s=60.0,
    )
    context = SetupContext(
        width=256,
        depth=32,
        flop_budget=272_000_000_000,
        api_version="1.0",
        submission_dir=str(Path("estimator.py").resolve().parent),
        seed=0,
    )
    failures: list[str] = []
    flops: list[int] = []
    try:
        runner.start(EstimatorEntrypoint(Path("estimator.py").resolve()), context, limits)
        for row in rows:
            mlp = MLP.from_row(row, seed_protocol_version="3.0")
            prediction = runner.predict(mlp, limits.flop_budget)
            if not valid_prediction(prediction, depth=mlp.depth, width=mlp.width):
                failures.append(f"{mlp.name}: output validation")
            stats = runner.last_predict_stats()
            if stats is not None:
                flops.append(stats.flops_used)
    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")
    finally:
        runner.close()
    result = {
        "runner": "official SubprocessRunner",
        "setup_timeout_seconds": args.setup_timeout,
        "predict_timeout_seconds": 30.0,
        "mlp_count": len(rows),
        "failure_count": len(failures),
        "failures": failures,
        "mean_analytical_flops": sum(flops) / len(flops) if flops else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
