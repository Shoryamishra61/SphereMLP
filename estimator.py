"""Official WhestBench estimator entry point.

T00 intentionally keeps the implementation minimal: later tasks replace this
validated scalar-safe result only after their own acceptance gates pass.
"""

from __future__ import annotations

import flopscope.numpy as fnp
from whestbench import MLP, BaseEstimator


def _scalar_budget_reserve(*, width: int, depth: int) -> int:
    return int(32 * depth * width * width + 128 * depth * width)


def _propagate_scalar(mlp: MLP) -> fnp.ndarray:
    from whest_solution.scalar import propagate_scalar

    return propagate_scalar(mlp)


def _valid_prediction(candidate: object, *, depth: int, width: int) -> bool:
    from whest_solution.guards import valid_prediction

    return valid_prediction(candidate, depth=depth, width=width)


def _covariance_budget_reserve(*, width: int, depth: int) -> int:
    return int(8_000 * depth * width * width)


def _propagate_covariance(mlp: MLP) -> fnp.ndarray:
    from whest_solution.covariance import propagate_covariance

    return propagate_covariance(mlp).predictions


class Estimator(BaseEstimator):
    """Return a finite, non-negative result satisfying the official contract."""

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        retained = fnp.zeros((mlp.depth, mlp.width), dtype=fnp.float32)
        if budget < _scalar_budget_reserve(width=mlp.width, depth=mlp.depth):
            return retained
        try:
            candidate = _propagate_scalar(mlp)
            if _valid_prediction(candidate, depth=mlp.depth, width=mlp.width):
                retained = candidate
        except Exception:
            # The free retained result is intentionally immune to optional
            # numerical or accounting failures. T15 expands failure telemetry.
            pass
        if budget < _covariance_budget_reserve(width=mlp.width, depth=mlp.depth):
            return retained
        try:
            candidate = _propagate_covariance(mlp)
            if _valid_prediction(candidate, depth=mlp.depth, width=mlp.width):
                retained = candidate
        except Exception:
            pass
        return retained
