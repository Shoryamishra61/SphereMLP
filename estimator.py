"""Official WhestBench estimator entry point.

T06 freezes exact-first-layer scalar propagation as the safe default.  The
full covariance candidate remains development-only because its validation
runtime tail exceeded the official per-call limit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from whestbench import BaseEstimator

if TYPE_CHECKING:
    import flopscope.numpy as fnp
    from whestbench import MLP


def _scalar_budget_reserve(*, width: int, depth: int) -> int:
    return int(32 * depth * width * width + 128 * depth * width)


def _propagate_scalar(mlp: "MLP") -> "fnp.ndarray":
    from whest_solution.scalar import propagate_scalar

    return propagate_scalar(mlp)


def _valid_prediction(candidate: object, *, depth: int, width: int) -> bool:
    from whest_solution.guards import valid_prediction

    return valid_prediction(candidate, depth=depth, width=width)


class Estimator(BaseEstimator):
    """Return a finite, non-negative result satisfying the official contract."""

    def predict(self, mlp: "MLP", budget: int) -> "fnp.ndarray":
        # Import only after the runner has completed its fixed setup handshake.
        # This is necessary for Windows subprocess workers with a short setup cap.
        import flopscope.numpy as fnp

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
        return retained
