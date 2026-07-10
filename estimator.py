"""Official WhestBench estimator entry point.

T00 intentionally keeps the implementation minimal: later tasks replace this
validated scalar-safe result only after their own acceptance gates pass.
"""

from __future__ import annotations

import flopscope.numpy as fnp
from whestbench import MLP, BaseEstimator


class Estimator(BaseEstimator):
    """Return a finite, non-negative result satisfying the official contract."""

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        _ = budget
        return fnp.zeros((mlp.depth, mlp.width), dtype=fnp.float32)
