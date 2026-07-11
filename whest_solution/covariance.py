"""Full-covariance Gaussian propagation with bounded pairwise integration."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from math import pi

import flopscope.numpy as fnp
from whestbench import MLP

import flopscope as flops

from .moments import (
    bivariate_relu_second_moment,
    pairwise_correlation,
    relu_mean,
    relu_second_moment,
)

_INV_SQRT_TWO_PI = 1.0 / (2.0 * pi) ** 0.5


class CovariancePropagationError(RuntimeError):
    """Raised when the approximation becomes unsafe to expose."""


@dataclass(frozen=True)
class LayerDiagnostics:
    trace: float
    mean_variance: float
    minimum_diagonal_variance: float
    maximum_absolute_correlation: float
    mean_absolute_off_diagonal_correlation: float
    frobenius_norm: float
    symmetry_error: float
    negative_diagonal_count: int


@dataclass(frozen=True)
class CovarianceResult:
    predictions: fnp.ndarray
    covariances: tuple[fnp.ndarray, ...]
    diagnostics: tuple[LayerDiagnostics, ...]


def covariance_budget_reserve(*, width: int, depth: int) -> int:
    """Conservative preflight reserve for the dense pairwise backend."""
    return int(8_000 * depth * width * width)


def _all_finite(*arrays: fnp.ndarray) -> bool:
    return all(bool(fnp.all(fnp.isfinite(array))) for array in arrays)


def _diagnostics(covariance: fnp.ndarray, correlation: fnp.ndarray) -> LayerDiagnostics:
    # Reducing/indexing a symmetric matrix intentionally consumes its symmetry
    # tag. Scope the warning suppression to diagnostic-only reductions.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", flops.SymmetryLossWarning)
        return _diagnostics_unwarned(covariance, correlation)


def _diagnostics_unwarned(covariance: fnp.ndarray, correlation: fnp.ndarray) -> LayerDiagnostics:
    diagonal = fnp.diag(covariance)
    width = int(covariance.shape[0])
    upper = fnp.triu_indices(width, k=1)
    upper_correlation = fnp.abs(correlation[upper])
    upper_covariance = covariance[upper]
    maximum_correlation = float(fnp.max(fnp.abs(fnp.diag(correlation))))
    if width > 1:
        maximum_correlation = max(maximum_correlation, float(fnp.max(upper_correlation)))
        mean_off_diagonal = float(fnp.mean(upper_correlation))
        symmetry_error = float(fnp.max(fnp.abs(covariance[upper] - covariance.T[upper])))
    else:
        mean_off_diagonal = 0.0
        symmetry_error = 0.0
    squared_frobenius = fnp.sum(diagonal * diagonal) + 2.0 * fnp.sum(
        upper_covariance * upper_covariance
    )
    return LayerDiagnostics(
        trace=float(fnp.sum(diagonal)),
        mean_variance=float(fnp.mean(diagonal)),
        minimum_diagonal_variance=float(fnp.min(diagonal)),
        maximum_absolute_correlation=maximum_correlation,
        mean_absolute_off_diagonal_correlation=mean_off_diagonal,
        frobenius_norm=float(fnp.sqrt(squared_frobenius)),
        symmetry_error=symmetry_error,
        negative_diagonal_count=int(float(fnp.sum(diagonal < 0.0))),
    )


def _validated_pre_variance(
    covariance: fnp.ndarray, *, variance_clip_tolerance: float
) -> fnp.ndarray:
    raw = fnp.diag(covariance)
    scale = max(1.0, float(fnp.max(fnp.abs(raw))))
    if float(fnp.min(raw)) < -variance_clip_tolerance * scale:
        raise CovariancePropagationError("pre-activation covariance has a negative diagonal")
    return fnp.maximum(raw, 0.0)


def propagate_covariance(
    mlp: MLP,
    *,
    sigma_epsilon: float = 1e-12,
    correlation_epsilon: float = 1e-7,
    variance_clip_tolerance: float = 1e-8,
) -> CovarianceResult:
    """Propagate a Gaussian mean/covariance closure through all ReLU layers."""
    rows: list[fnp.ndarray] = []
    covariances: list[fnp.ndarray] = []
    diagnostics: list[LayerDiagnostics] = []

    first_weight = mlp.weights[0]
    pre_covariance = fnp.einsum("ia,ib->ab", first_weight, first_weight)
    pre_covariance = 0.5 * (pre_covariance + pre_covariance.T)
    pre_variance = _validated_pre_variance(
        pre_covariance, variance_clip_tolerance=variance_clip_tolerance
    )
    sigma = fnp.sqrt(pre_variance)
    mean = sigma * _INV_SQRT_TWO_PI
    correlation = pairwise_correlation(pre_covariance, sigma_epsilon=sigma_epsilon)
    clipped_correlation = fnp.clip(correlation, -1.0, 1.0)
    kernel = fnp.sqrt(fnp.maximum(1.0 - clipped_correlation * clipped_correlation, 0.0))
    kernel = kernel + (pi - fnp.arccos(clipped_correlation)) * clipped_correlation
    kernel = flops.as_symmetric(kernel, symmetry=(0, 1))
    sigma_outer = flops.as_symmetric(fnp.outer(sigma, sigma), symmetry=(0, 1))
    second = sigma_outer * kernel / (2.0 * pi)
    fnp.fill_diagonal(second, pre_variance * 0.5)
    second = flops.as_symmetric(second, symmetry=(0, 1))
    mean_outer = flops.as_symmetric(fnp.outer(mean, mean), symmetry=(0, 1))
    covariance = second - mean_outer
    covariance = flops.as_symmetric(0.5 * (covariance + covariance.T), symmetry=(0, 1))
    if not _all_finite(mean, covariance):
        raise CovariancePropagationError("non-finite exact first-layer state")
    rows.append(fnp.maximum(mean, 0.0))
    covariances.append(covariance)
    diagnostics.append(_diagnostics(covariance, pairwise_correlation(covariance)))

    for weight in mlp.weights[1:]:
        pre_mean = weight.T @ mean
        pre_covariance = fnp.einsum("ij,ia,jb->ab", covariance, weight, weight)
        pre_covariance = 0.5 * (pre_covariance + pre_covariance.T)
        pre_variance = _validated_pre_variance(
            pre_covariance, variance_clip_tolerance=variance_clip_tolerance
        )
        correlation = pairwise_correlation(pre_covariance, sigma_epsilon=sigma_epsilon)
        mean = relu_mean(pre_mean, pre_variance, sigma_epsilon=sigma_epsilon)
        # Broadcasting symmetric pair matrices against quadrature nodes drops
        # only flopscope metadata, not numerical symmetry. Re-tag after the
        # explicit pair update and scope suppression to that known operation.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", flops.SymmetryLossWarning)
            second = bivariate_relu_second_moment(
                fnp.expand_dims(pre_mean, 1),
                fnp.expand_dims(pre_variance, 1),
                fnp.expand_dims(pre_mean, 0),
                fnp.expand_dims(pre_variance, 0),
                correlation,
                sigma_epsilon=sigma_epsilon,
                correlation_epsilon=correlation_epsilon,
            )
        diagonal_second = relu_second_moment(pre_mean, pre_variance, sigma_epsilon=sigma_epsilon)
        fnp.fill_diagonal(second, diagonal_second)
        second = flops.as_symmetric(second, symmetry=(0, 1))
        mean_outer = flops.as_symmetric(fnp.outer(mean, mean), symmetry=(0, 1))
        covariance = second - mean_outer
        covariance = flops.as_symmetric(0.5 * (covariance + covariance.T), symmetry=(0, 1))
        diagonal = fnp.diag(covariance)
        if float(fnp.min(diagonal)) < -variance_clip_tolerance * max(
            1.0, float(fnp.max(fnp.abs(diagonal)))
        ):
            raise CovariancePropagationError("post-ReLU covariance has a negative diagonal")
        fnp.fill_diagonal(covariance, fnp.maximum(diagonal, 0.0))
        if not _all_finite(mean, covariance):
            raise CovariancePropagationError("non-finite propagated covariance state")
        rows.append(fnp.maximum(mean, 0.0))
        covariances.append(covariance)
        diagnostics.append(_diagnostics(covariance, pairwise_correlation(covariance)))

    return CovarianceResult(
        predictions=fnp.stack(rows, axis=0),
        covariances=tuple(covariances),
        diagnostics=tuple(diagnostics),
    )
