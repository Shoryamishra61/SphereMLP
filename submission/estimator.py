"""Self-contained Phase 1 submission estimator.

This file deliberately has no local-package imports.  It is packaged as a
single-file submission so the grader loads exactly the numerical code audited
locally, without archive subpackage resolution or path-separator ambiguity.
"""

from __future__ import annotations

from math import exp, lgamma, log, pi

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator


_SAMPLES = 17_408
_BATCH = 512
_ANALYTICAL_FRACTION = 0.30
_INV_SQRT_2PI = 1.0 / (2.0 * pi) ** 0.5


def _scalar_reserve(width: int, depth: int) -> int:
    return 32 * depth * width * width + 128 * depth * width


def _per_sample_flops(width: int, depth: int) -> int:
    return depth * 2 * width * width + 8 * depth * width


def _sample_count(budget: int, width: int, depth: int) -> int:
    available = int(budget * _ANALYTICAL_FRACTION) - _scalar_reserve(width, depth) - 2_500_000
    if available <= 0:
        return 0
    count = min(available // _per_sample_flops(width, depth), _SAMPLES)
    return (count // _BATCH) * _BATCH


def _valid(value: object, depth: int, width: int) -> bool:
    return (
        isinstance(value, fnp.ndarray)
        and tuple(value.shape) == (depth, width)
        and fnp.issubdtype(value.dtype, fnp.floating)
        and bool(fnp.all(fnp.isfinite(value)))
        and bool(fnp.all(value >= 0.0))
    )


def _scalar(mlp):
    """Exact first marginal and diagonal-Gaussian propagation fallback."""
    rows = []
    weight = mlp.weights[0]
    variance = fnp.maximum(fnp.sum(weight * weight, axis=0), 0.0)
    sigma = fnp.sqrt(variance)
    mean = sigma * _INV_SQRT_2PI
    activation_variance = fnp.maximum(0.5 * variance - mean * mean, 0.0)
    rows.append(mean)
    for weight in mlp.weights[1:]:
        pre_mean = weight.T @ mean
        pre_variance = fnp.maximum((weight * weight).T @ activation_variance, 0.0)
        sigma = fnp.sqrt(pre_variance)
        deterministic = sigma <= 1e-12
        safe_sigma = fnp.where(deterministic, 1.0, sigma)
        alpha = pre_mean / safe_sigma
        pdf = flops.stats.norm.pdf(alpha)
        cdf = flops.stats.norm.cdf(alpha)
        next_mean = sigma * pdf + pre_mean * cdf
        next_mean = fnp.where(deterministic, fnp.maximum(pre_mean, 0.0), next_mean)
        second = (pre_mean * pre_mean + pre_variance) * cdf + pre_mean * sigma * pdf
        second = fnp.where(deterministic, next_mean * next_mean, second)
        activation_variance = fnp.maximum(second - next_mean * next_mean, 0.0)
        mean = fnp.maximum(next_mean, 0.0)
        rows.append(mean)
    return fnp.stack(rows, axis=0)


def _chi_mean(dimension: int) -> float:
    return exp(0.5 * log(2.0) + lgamma((dimension + 1.0) / 2.0) - lgamma(dimension / 2.0))


def _spherical(mlp, samples: int):
    """Rao-Blackwellized Gaussian sampling: E[h(X)] = E[R] E[h(U)]."""
    width, depth = int(mlp.width), int(mlp.depth)
    rng = fnp.random.default_rng(int(mlp.seed))
    sums = [fnp.zeros((width,), dtype=fnp.float64) for _ in range(depth)]
    complete = 0
    while complete < samples:
        count = min(_BATCH, samples - complete)
        direction = fnp.asarray(rng.standard_normal((count, width), dtype=fnp.float64))
        norm = fnp.sqrt(fnp.sum(direction * direction, axis=1, keepdims=True))
        value = direction / fnp.where(norm > 0.0, norm, 1.0)
        for layer, weight in enumerate(mlp.weights):
            value = fnp.maximum(value @ weight, 0.0)
            sums[layer] = sums[layer] + fnp.sum(value, axis=0)
        complete += count
    # The chi-radius factor belongs outside the entire homogeneous network,
    # exactly once—not once per layer.
    return fnp.asarray(fnp.stack(sums, axis=0) * (_chi_mean(width) / samples), dtype=fnp.float32)


class Estimator(BaseEstimator):
    def predict(self, mlp, budget: int):
        depth, width = int(mlp.depth), int(mlp.width)
        retained = fnp.zeros((depth, width), dtype=fnp.float32)
        if budget < _scalar_reserve(width, depth):
            return retained
        try:
            candidate = _scalar(mlp)
            if _valid(candidate, depth, width):
                retained = candidate
        except Exception:
            pass
        samples = _sample_count(budget, width, depth)
        if samples >= _BATCH:
            try:
                candidate = _spherical(mlp, samples)
                if _valid(candidate, depth, width):
                    retained = candidate
            except Exception:
                pass
        return retained
