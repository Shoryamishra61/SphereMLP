"""Resumable paired benchmark over a frozen network-level partition."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import flopscope.numpy as fnp
import numpy as np
import pyarrow.parquet as pq
from whestbench import MLP
from whestbench.budget import effective_compute, score_multiplier
from whestbench.estimators import MeanPropagationEstimator

import flopscope as flops
from estimator import Estimator
from whest_solution.guards import valid_prediction
from whest_solution.scalar import propagate_scalar

FLOP_BUDGET = 272_000_000_000
LAMBDA_FLOPS_PER_SECOND = 1e11
CSV_FIELDS = [
    "experiment_id",
    "mlp_id",
    "estimator_name",
    "raw_final_mse",
    "adjusted_score",
    "all_layer_mse",
    "per_layer_mse",
    "analytical_flops",
    "residual_time_charge",
    "effective_compute",
    "compute_ratio",
    "wall_ms",
    "peak_memory_mb",
    "status",
    "fallback_used",
    "seed",
]


@dataclass(frozen=True)
class Candidate:
    name: str
    predict: Callable[[MLP], fnp.ndarray]


def gaussian_monte_carlo(mlp: MLP, *, samples: int = 1536) -> fnp.ndarray:
    rng = fnp.random.default_rng(mlp.seed)
    x = fnp.asarray(rng.standard_normal((samples, mlp.width), dtype=fnp.float32))
    rows: list[fnp.ndarray] = []
    for weight in mlp.weights:
        x = fnp.maximum(x @ weight, 0.0)
        rows.append(fnp.mean(x, axis=0))
    return fnp.stack(rows, axis=0)


def candidates(names: list[str], gaussian_samples: int) -> list[Candidate]:
    available = {
        "zero": Candidate("zero", lambda mlp: fnp.zeros((mlp.depth, mlp.width), dtype=fnp.float32)),
        "official_scalar": Candidate(
            "official_scalar",
            lambda mlp: MeanPropagationEstimator().predict(mlp, FLOP_BUDGET),
        ),
        "exact_scalar": Candidate("exact_scalar", propagate_scalar),
        "runtime_covariance": Candidate(
            "runtime_covariance", lambda mlp: Estimator().predict(mlp, FLOP_BUDGET)
        ),
        "gaussian_mc": Candidate(
            "gaussian_mc",
            lambda mlp: gaussian_monte_carlo(mlp, samples=gaussian_samples),
        ),
    }
    unknown = sorted(set(names) - set(available))
    if unknown:
        raise ValueError(f"unknown estimator names: {unknown}")
    return [available[name] for name in names]


def load_selected_rows(snapshot: Path, selected_names: set[str]):
    for shard in sorted((snapshot / "data").glob("full-*.parquet")):
        table = pq.read_table(
            shard,
            filters=[("mlp_name", "in", sorted(selected_names))],
        )
        for row in table.to_pylist():
            if row["mlp_name"] in selected_names:
                yield row


def existing_pairs(output: Path) -> set[tuple[str, str]]:
    if not output.exists():
        return set()
    with output.open(newline="", encoding="utf-8") as handle:
        return {(row["mlp_id"], row["estimator_name"]) for row in csv.DictReader(handle)}


def append_result(output: Path, record: dict[str, object]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    exists = output.exists()
    with output.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(record)


def evaluate(
    experiment_id: str,
    mlp: MLP,
    target: np.ndarray,
    candidate: Candidate,
) -> dict[str, object]:
    status = "ok"
    fallback_used = False
    error_message = ""
    start = time.perf_counter()
    try:
        with flops.BudgetContext(flop_budget=FLOP_BUDGET, quiet=True) as context:
            prediction = candidate.predict(mlp)
            if not valid_prediction(prediction, depth=mlp.depth, width=mlp.width):
                raise ValueError("candidate failed output validation")
        predicted = np.asarray(prediction, dtype=np.float64)
        layer_mse = np.mean((predicted - target) ** 2, axis=1)
        analytical_flops = int(context.flops_used)
        residual_seconds = float(context.residual_wall_time_s or 0.0)
        compute = effective_compute(analytical_flops, residual_seconds, LAMBDA_FLOPS_PER_SECOND)
        if candidate.name == "runtime_covariance" and analytical_flops < 1_000_000_000:
            fallback_used = True
    except Exception as exc:
        status = "failed"
        error_message = f"{type(exc).__name__}: {exc}"
        layer_mse = np.full(mlp.depth, np.nan)
        analytical_flops = int(locals().get("context").flops_used) if "context" in locals() else 0
        residual_seconds = (
            float(locals().get("context").residual_wall_time_s or 0.0)
            if "context" in locals()
            else 0.0
        )
        compute = effective_compute(analytical_flops, residual_seconds, LAMBDA_FLOPS_PER_SECOND)
    wall_ms = (time.perf_counter() - start) * 1000.0
    failed = status != "ok"
    final_mse = float(layer_mse[-1]) if not failed else float("nan")
    multiplier = score_multiplier(compute, FLOP_BUDGET, failed=failed)
    return {
        "experiment_id": experiment_id,
        "mlp_id": mlp.name,
        "estimator_name": candidate.name,
        "raw_final_mse": final_mse,
        "adjusted_score": final_mse * multiplier,
        "all_layer_mse": float(np.mean(layer_mse)),
        "per_layer_mse": json.dumps(layer_mse.tolist(), separators=(",", ":")),
        "analytical_flops": analytical_flops,
        "residual_time_charge": residual_seconds * LAMBDA_FLOPS_PER_SECOND,
        "effective_compute": compute,
        "compute_ratio": compute / FLOP_BUDGET,
        "wall_ms": wall_ms,
        "peak_memory_mb": "",
        "status": status if not error_message else f"{status}: {error_message}",
        "fallback_used": fallback_used,
        "seed": mlp.seed,
    }


def write_summary(raw_path: Path, summary_path: Path) -> None:
    with raw_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_estimator: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_estimator.setdefault(row["estimator_name"], []).append(row)
    summary: dict[str, object] = {"raw_path": str(raw_path), "estimators": {}}
    for name, entries in sorted(by_estimator.items()):
        ok = [entry for entry in entries if entry["status"] == "ok"]
        if not ok:
            summary["estimators"][name] = {
                "mlp_count": len(entries),
                "failure_count": len(entries),
                "fallback_count": 0,
                "raw_final_mse_mean": None,
                "adjusted_score_mean": None,
                "all_layer_mse_mean": None,
                "per_layer_mse_mean": [],
            }
            continue

        def numeric(field: str) -> np.ndarray:
            return np.array([float(entry[field]) for entry in ok])

        layer_values = np.array([json.loads(entry["per_layer_mse"]) for entry in ok])
        summary["estimators"][name] = {
            "mlp_count": len(entries),
            "failure_count": len(entries) - len(ok),
            "fallback_count": sum(entry["fallback_used"].lower() == "true" for entry in ok),
            "raw_final_mse_mean": float(np.mean(numeric("raw_final_mse"))),
            "raw_final_mse_median": float(np.median(numeric("raw_final_mse"))),
            "adjusted_score_mean": float(np.mean(numeric("adjusted_score"))),
            "all_layer_mse_mean": float(np.mean(numeric("all_layer_mse"))),
            "p90_final_mse": float(np.quantile(numeric("raw_final_mse"), 0.90)),
            "p95_final_mse": float(np.quantile(numeric("raw_final_mse"), 0.95)),
            "mean_compute_ratio": float(np.mean(numeric("compute_ratio"))),
            "max_compute_ratio": float(np.max(numeric("compute_ratio"))),
            "mean_wall_ms": float(np.mean(numeric("wall_ms"))),
            "p95_wall_ms": float(np.quantile(numeric("wall_ms"), 0.95)),
            "max_wall_ms": float(np.max(numeric("wall_ms"))),
            "mean_analytical_flops": float(np.mean(numeric("analytical_flops"))),
            "per_layer_mse_mean": np.mean(layer_values, axis=0).tolist(),
        }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--partition", default="validation")
    parser.add_argument("--manifest", type=Path, default=Path("manifests/dataset_splits.json"))
    parser.add_argument(
        "--estimators", default="zero,official_scalar,exact_scalar,runtime_covariance,gaussian_mc"
    )
    parser.add_argument("--gaussian-samples", type=int, default=1536)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--experiment-id", default="t06_deterministic_validation")
    parser.add_argument("--output", type=Path, default=Path("results/raw/t06_validation.csv"))
    parser.add_argument(
        "--summary", type=Path, default=Path("results/summaries/t06_validation.json")
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    ordered_names = list(manifest["partitions"][args.partition])
    if args.limit is not None:
        ordered_names = ordered_names[: args.limit]
    selected = set(ordered_names)
    done = existing_pairs(args.output)
    chosen = candidates(args.estimators.split(","), args.gaussian_samples)
    expected_for_name = {name: {candidate.name for candidate in chosen} for name in selected}
    seen: set[str] = set()
    position = {name: index for index, name in enumerate(ordered_names, start=1)}
    for row in load_selected_rows(args.snapshot, selected):
        name = row["mlp_name"]
        seen.add(name)
        if all((name, candidate_name) in done for candidate_name in expected_for_name[name]):
            continue
        mlp = MLP.from_row(row, seed_protocol_version="3.0")
        target = np.asarray(row["all_layer_means"], dtype=np.float64)
        for candidate in chosen:
            if (name, candidate.name) in done:
                continue
            record = evaluate(args.experiment_id, mlp, target, candidate)
            append_result(args.output, record)
            print(
                json.dumps(
                    {
                        "network": f"{position[name]}/{len(ordered_names)}",
                        "mlp_id": name,
                        "estimator": candidate.name,
                        "status": record["status"],
                        "adjusted_score": record["adjusted_score"],
                    }
                ),
                flush=True,
            )
    missing = selected - seen
    if missing:
        raise ValueError(f"missing selected MLPs: {sorted(missing)}")
    write_summary(args.output, args.summary)


if __name__ == "__main__":
    main()
