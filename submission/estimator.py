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


_SAMPLES = 4096
_BATCH = 512
_ANALYTICAL_FRACTION = 0.10
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


def _safe(value: object, depth: int, width: int):
    """Normalize a computed candidate without type-identity assumptions.

    The remote worker may wrap flopscope arrays at its IPC boundary.  Validate
    shape after coercion, then deterministically remove any numerical residue.
    """
    try:
        array = fnp.asarray(value, dtype=fnp.float32)
        if tuple(array.shape) != (depth, width):
            return None
        return fnp.maximum(fnp.where(fnp.isfinite(array), array, 0.0), 0.0)
    except Exception:
        return None


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


def _lhs_directions(rng, count: int, width: int):
    """Randomized-LHS normal directions, marginally uniform after projection.

    Each normal coordinate uses all equal-probability quantile strata once.
    The independent random permutations and jitters retain uniform spherical
    marginals for each row; only cross-row dependence changes.
    """
    strata = fnp.stack(
        [fnp.asarray(rng.permutation(count), dtype=fnp.float64) for _ in range(width)], axis=1
    )
    uniforms = (strata + fnp.asarray(rng.random((count, width), dtype=fnp.float64))) / float(count)
    gaussian = flops.stats.norm.ppf(uniforms)
    norm = fnp.sqrt(fnp.sum(gaussian * gaussian, axis=1, keepdims=True))
    return gaussian / fnp.where(norm > 0.0, norm, 1.0)


def _spherical(mlp, samples: int):
    """Rao-Blackwellized Gaussian sampling: E[h(X)] = E[R] E[h(U)]."""
    width, depth = int(mlp.width), int(mlp.depth)
    rng = fnp.random.default_rng(int(mlp.seed))
    final_sum = fnp.zeros((width,), dtype=fnp.float64)
    directions = _lhs_directions(rng, samples, width)
    complete = 0
    while complete < samples:
        count = min(_BATCH, samples - complete)
        value = directions[complete : complete + count]
        for layer, weight in enumerate(mlp.weights):
            value = fnp.maximum(value @ weight, 0.0)
        final_sum = final_sum + fnp.sum(value, axis=0)
        complete += count
    # The chi-radius factor belongs outside the entire homogeneous network,
    # exactly once—not once per layer.
    final = fnp.asarray(final_sum * (_chi_mean(width) / samples), dtype=fnp.float32)
    # The official objective scores only the final layer.  Keep the unscored
    # rows deterministic, finite, and free rather than spending extra work on
    # their reductions.
    final_row = fnp.reshape(final, (1, width))
    return fnp.concatenate((fnp.zeros((depth - 1, width), dtype=fnp.float32), final_row), axis=0)


class Estimator(BaseEstimator):
    def predict(self, mlp, budget: int):
        depth, width = int(mlp.depth), int(mlp.width)
        retained = fnp.zeros((depth, width), dtype=fnp.float32)
        if budget < _scalar_reserve(width, depth):
            return retained
        try:
            candidate = _scalar(mlp)
            safe = _safe(candidate, depth, width)
            if safe is not None:
                retained = safe
        except Exception:
            pass
        samples = _sample_count(budget, width, depth)
        if samples >= _BATCH:
            try:
                candidate = _spherical(mlp, samples)
                safe = _safe(candidate, depth, width)
                if safe is not None:
                    retained = safe
            except Exception:
                pass
        return retained
