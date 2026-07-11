"""Deterministic exact-first-layer diagonal Gaussian propagation."""

from __future__ import annotations

from math import pi

import flopscope.numpy as fnp
from whestbench import MLP

from .moments import relu_mean, relu_variance

_INV_SQRT_TWO_PI = 1.0 / (2.0 * pi) ** 0.5


def scalar_budget_reserve(*, width: int, depth: int) -> int:
    """Conservative preflight bound preventing tiny-budget probe exhaustion."""
    return int(32 * depth * width * width + 128 * depth * width)


def _finite_state(mean: fnp.ndarray, variance: fnp.ndarray) -> bool:
    return bool(fnp.all(fnp.isfinite(mean))) and bool(fnp.all(fnp.isfinite(variance)))


def propagate_scalar(
    mlp: MLP,
    *,
    sigma_epsilon: float = 1e-12,
    variance_clip_tolerance: float = 1e-10,
) -> fnp.ndarray:
    """Return all-layer means under a diagonal Gaussian closure.

    Weight matrices are input-by-output, as proven in T00. The first layer is
    handled directly from column norms and is therefore exact marginally.
    """
    width = int(mlp.width)
    rows: list[fnp.ndarray] = []

    first_weight = mlp.weights[0]
    pre_variance = fnp.sum(first_weight * first_weight, axis=0)
    pre_variance = fnp.maximum(pre_variance, 0.0)
    sigma = fnp.sqrt(pre_variance)
    mean = sigma * _INV_SQRT_TWO_PI
    second = pre_variance * 0.5
    variance = fnp.maximum(second - mean * mean, 0.0)
    if not _finite_state(mean, variance):
        mean = fnp.zeros(width, dtype=fnp.float64)
        variance = fnp.zeros(width, dtype=fnp.float64)
    rows.append(fnp.maximum(mean, 0.0))

    for weight in mlp.weights[1:]:
        pre_mean = weight.T @ mean
        pre_variance = (weight * weight).T @ variance
        pre_variance = fnp.where(
            pre_variance >= -variance_clip_tolerance,
            fnp.maximum(pre_variance, 0.0),
            fnp.asarray(float("nan")),
        )
        next_mean = relu_mean(pre_mean, pre_variance, sigma_epsilon=sigma_epsilon)
        next_variance = relu_variance(pre_mean, pre_variance, sigma_epsilon=sigma_epsilon)
        if _finite_state(next_mean, next_variance):
            mean = next_mean
            variance = next_variance
        else:
            mean = fnp.zeros(width, dtype=fnp.float64)
            variance = fnp.zeros(width, dtype=fnp.float64)
        rows.append(fnp.maximum(mean, 0.0))

    return fnp.stack(rows, axis=0)
