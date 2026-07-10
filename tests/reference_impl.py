"""Slow, auditable NumPy reference routines for small development fixtures.

These functions are excluded from the runtime package. They favor explicit
loops and independent numerical integration over speed.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, pi, sqrt
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def forward_all_layers(weights: Sequence[FloatArray], inputs: FloatArray) -> list[FloatArray]:
    """Execute the official sample-row convention: ``x = relu(x @ w)``."""
    x = np.asarray(inputs, dtype=np.float64)
    outputs: list[FloatArray] = []
    for weight in weights:
        x = np.maximum(x @ np.asarray(weight, dtype=np.float64), 0.0)
        outputs.append(x.copy())
    return outputs


def monte_carlo_layer_means(
    weights: Sequence[FloatArray], n_samples: int, *, seed: int
) -> FloatArray:
    if not weights or n_samples <= 0:
        raise ValueError("weights must be non-empty and n_samples must be positive")
    width = int(np.asarray(weights[0]).shape[0])
    rng = np.random.default_rng(seed)
    inputs = rng.standard_normal((n_samples, width))
    return np.stack([layer.mean(axis=0) for layer in forward_all_layers(weights, inputs)])


def _normal_pdf(x: FloatArray) -> FloatArray:
    return np.exp(-0.5 * x * x) / sqrt(2.0 * pi)


def _normal_cdf(x: FloatArray) -> FloatArray:
    values = np.asarray(x, dtype=np.float64)
    return 0.5 * (1.0 + np.vectorize(erf, otypes=[float])(values / sqrt(2.0)))


def relu_moments(
    mu: FloatArray, variance: FloatArray, *, epsilon: float = 0.0
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return exact univariate Gaussian ReLU mean, second moment, and variance."""
    mu_arr, var_arr = np.broadcast_arrays(
        np.asarray(mu, dtype=np.float64), np.asarray(variance, dtype=np.float64)
    )
    if np.any(var_arr < 0.0):
        raise ValueError("variance must be non-negative")
    deterministic = var_arr <= epsilon
    sigma = np.sqrt(var_arr)
    safe_sigma = np.where(deterministic, 1.0, sigma)
    alpha = mu_arr / safe_sigma
    phi = _normal_pdf(alpha)
    cdf = _normal_cdf(alpha)
    mean = sigma * phi + mu_arr * cdf
    second = (mu_arr * mu_arr + var_arr) * cdf + mu_arr * sigma * phi
    deterministic_relu = np.maximum(mu_arr, 0.0)
    mean = np.where(deterministic, deterministic_relu, mean)
    second = np.where(deterministic, deterministic_relu * deterministic_relu, second)
    relu_variance = np.maximum(second - mean * mean, 0.0)
    return mean, second, relu_variance


def bivariate_relu_second_moment_quadrature(
    mu1: float,
    variance1: float,
    mu2: float,
    variance2: float,
    rho: float,
    *,
    order: int = 80,
) -> float:
    """Numerically integrate ``E[relu(Z1) relu(Z2)]`` with Gauss-Hermite nodes."""
    if variance1 < 0.0 or variance2 < 0.0:
        raise ValueError("variances must be non-negative")
    if order < 8:
        raise ValueError("quadrature order must be at least 8")
    sigma1, sigma2 = sqrt(variance1), sqrt(variance2)
    if sigma1 == 0.0 and sigma2 == 0.0:
        return max(mu1, 0.0) * max(mu2, 0.0)
    if sigma1 == 0.0:
        mean2 = float(relu_moments(np.array(mu2), np.array(variance2))[0])
        return max(mu1, 0.0) * mean2
    if sigma2 == 0.0:
        mean1 = float(relu_moments(np.array(mu1), np.array(variance1))[0])
        return mean1 * max(mu2, 0.0)

    clipped_rho = float(np.clip(rho, -1.0, 1.0))
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    g1 = sqrt(2.0) * nodes[:, None]
    g2 = sqrt(2.0) * nodes[None, :]
    z1 = mu1 + sigma1 * g1
    z2 = mu2 + sigma2 * (clipped_rho * g1 + sqrt(max(1.0 - clipped_rho * clipped_rho, 0.0)) * g2)
    integrand = np.maximum(z1, 0.0) * np.maximum(z2, 0.0)
    return float(np.sum(weights[:, None] * weights[None, :] * integrand) / pi)


def scalar_propagation(weights: Sequence[FloatArray]) -> tuple[FloatArray, FloatArray]:
    """Diagonal Gaussian closure using the official input-by-output orientation."""
    width = int(np.asarray(weights[0]).shape[0])
    mean = np.zeros(width, dtype=np.float64)
    variance = np.ones(width, dtype=np.float64)
    means: list[FloatArray] = []
    variances: list[FloatArray] = []
    for weight in weights:
        w = np.asarray(weight, dtype=np.float64)
        pre_mean = mean @ w
        pre_variance = variance @ (w * w)
        mean, _, variance = relu_moments(pre_mean, pre_variance)
        means.append(mean.copy())
        variances.append(variance.copy())
    return np.stack(means), np.stack(variances)


@dataclass(frozen=True)
class CovarianceReference:
    means: FloatArray
    covariances: tuple[FloatArray, ...]


def covariance_propagation(
    weights: Sequence[FloatArray], *, quadrature_order: int = 48
) -> CovarianceReference:
    """Full Gaussian covariance closure with independently integrated pair moments."""
    width = int(np.asarray(weights[0]).shape[0])
    mean = np.zeros(width, dtype=np.float64)
    covariance = np.eye(width, dtype=np.float64)
    means: list[FloatArray] = []
    covariances: list[FloatArray] = []

    for weight in weights:
        w = np.asarray(weight, dtype=np.float64)
        pre_mean = mean @ w
        pre_covariance = w.T @ covariance @ w
        pre_variance = np.maximum(np.diag(pre_covariance), 0.0)
        mean, diagonal_second, _ = relu_moments(pre_mean, pre_variance)
        second = np.empty((width, width), dtype=np.float64)
        np.fill_diagonal(second, diagonal_second)
        sigma = np.sqrt(pre_variance)
        for i in range(width):
            for j in range(i + 1, width):
                denominator = sigma[i] * sigma[j]
                rho = 0.0 if denominator == 0.0 else pre_covariance[i, j] / denominator
                pair = bivariate_relu_second_moment_quadrature(
                    float(pre_mean[i]),
                    float(pre_variance[i]),
                    float(pre_mean[j]),
                    float(pre_variance[j]),
                    float(rho),
                    order=quadrature_order,
                )
                second[i, j] = pair
                second[j, i] = pair
        covariance = second - np.outer(mean, mean)
        covariance = 0.5 * (covariance + covariance.T)
        means.append(mean.copy())
        covariances.append(covariance.copy())
    return CovarianceReference(np.stack(means), tuple(covariances))
