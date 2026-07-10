"""Metered Gaussian/ReLU moment primitives.

The general bivariate-normal CDF uses fixed 24-point Gauss-Legendre
integration of Plackett's identity because flopscope 0.8 has no multivariate
CDF. It is therefore a documented numerical approximation; the univariate and
zero-mean bivariate formulas are closed form.
"""

from __future__ import annotations

from math import exp, lgamma, log, pi

import flopscope.numpy as fnp

import flopscope as flops

_SQRT_TWO_PI = (2.0 * pi) ** 0.5
_GL_NODES = (
    -0.9951872199970213,
    -0.9747285559713095,
    -0.9382745520027328,
    -0.886415527004401,
    -0.820001985973903,
    -0.7401241915785544,
    -0.6480936519369755,
    -0.5454214713888396,
    -0.4337935076260451,
    -0.3150426796961634,
    -0.1911188674736163,
    -0.06405689286260563,
    0.06405689286260563,
    0.1911188674736163,
    0.3150426796961634,
    0.4337935076260451,
    0.5454214713888396,
    0.6480936519369755,
    0.7401241915785544,
    0.820001985973903,
    0.886415527004401,
    0.9382745520027328,
    0.9747285559713095,
    0.9951872199970213,
)
_GL_WEIGHTS = (
    0.012341229799987091,
    0.028531388628933743,
    0.04427743881741955,
    0.05929858491543674,
    0.07334648141108041,
    0.08619016153195329,
    0.09761865210411406,
    0.1074442701159656,
    0.11550566805372561,
    0.12167047292780342,
    0.1258374563468283,
    0.12793819534675221,
    0.12793819534675221,
    0.1258374563468283,
    0.12167047292780342,
    0.11550566805372561,
    0.1074442701159656,
    0.09761865210411406,
    0.08619016153195329,
    0.07334648141108041,
    0.05929858491543674,
    0.04427743881741955,
    0.028531388628933743,
    0.012341229799987091,
)


def normal_pdf(x: fnp.ndarray) -> fnp.ndarray:
    return flops.stats.norm.pdf(x)


def normal_cdf(x: fnp.ndarray) -> fnp.ndarray:
    return flops.stats.norm.cdf(x)


def _relu_components(
    mu: fnp.ndarray, variance: fnp.ndarray, sigma_epsilon: float
) -> tuple[fnp.ndarray, fnp.ndarray, fnp.ndarray]:
    variance = fnp.maximum(fnp.asarray(variance), 0.0)
    sigma = fnp.sqrt(variance)
    deterministic = sigma <= sigma_epsilon
    safe_sigma = fnp.where(deterministic, 1.0, sigma)
    alpha = fnp.asarray(mu) / safe_sigma
    pdf = normal_pdf(alpha)
    cdf = normal_cdf(alpha)
    return sigma, pdf, cdf


def relu_mean(
    mu: fnp.ndarray, variance: fnp.ndarray, *, sigma_epsilon: float = 1e-12
) -> fnp.ndarray:
    sigma, pdf, cdf = _relu_components(mu, variance, sigma_epsilon)
    deterministic = sigma <= sigma_epsilon
    value = sigma * pdf + mu * cdf
    return fnp.where(deterministic, fnp.maximum(mu, 0.0), value)


def relu_second_moment(
    mu: fnp.ndarray, variance: fnp.ndarray, *, sigma_epsilon: float = 1e-12
) -> fnp.ndarray:
    clipped_variance = fnp.maximum(fnp.asarray(variance), 0.0)
    sigma, pdf, cdf = _relu_components(mu, clipped_variance, sigma_epsilon)
    deterministic = sigma <= sigma_epsilon
    value = (mu * mu + clipped_variance) * cdf + mu * sigma * pdf
    deterministic_value = fnp.maximum(mu, 0.0)
    return fnp.where(deterministic, deterministic_value * deterministic_value, value)


def relu_variance(
    mu: fnp.ndarray, variance: fnp.ndarray, *, sigma_epsilon: float = 1e-12
) -> fnp.ndarray:
    mean = relu_mean(mu, variance, sigma_epsilon=sigma_epsilon)
    second = relu_second_moment(mu, variance, sigma_epsilon=sigma_epsilon)
    return fnp.maximum(second - mean * mean, 0.0)


def zero_mean_bivariate_relu_second_moment(
    sigma1: fnp.ndarray, sigma2: fnp.ndarray, rho: fnp.ndarray
) -> fnp.ndarray:
    clipped_rho = fnp.clip(rho, -1.0, 1.0)
    kernel = fnp.sqrt(fnp.maximum(1.0 - clipped_rho * clipped_rho, 0.0))
    kernel = kernel + (pi - fnp.arccos(clipped_rho)) * clipped_rho
    return fnp.maximum(sigma1, 0.0) * fnp.maximum(sigma2, 0.0) * kernel / (2.0 * pi)


def bivariate_normal_cdf_approx(
    a: fnp.ndarray,
    b: fnp.ndarray,
    rho: fnp.ndarray,
    *,
    correlation_epsilon: float = 1e-7,
) -> fnp.ndarray:
    """Approximate ``Phi_2(a, b; rho)`` by fixed Gauss-Legendre integration."""
    interior_rho = fnp.clip(rho, -1.0 + correlation_epsilon, 1.0 - correlation_epsilon)
    nodes = fnp.asarray(_GL_NODES)
    weights = fnp.asarray(_GL_WEIGHTS)
    t = fnp.expand_dims(interior_rho, -1) * (nodes + 1.0) * 0.5
    a_expanded = fnp.expand_dims(a, -1)
    b_expanded = fnp.expand_dims(b, -1)
    one_minus_t2 = fnp.maximum(1.0 - t * t, correlation_epsilon)
    exponent = -(
        a_expanded * a_expanded - 2.0 * t * a_expanded * b_expanded + b_expanded * b_expanded
    ) / (2.0 * one_minus_t2)
    integrand = fnp.exp(exponent) / fnp.sqrt(one_minus_t2)
    correction = interior_rho * 0.5 * fnp.sum(weights * integrand, axis=-1) / (2.0 * pi)
    return fnp.clip(normal_cdf(a) * normal_cdf(b) + correction, 0.0, 1.0)


def bivariate_relu_second_moment(
    mu1: fnp.ndarray,
    variance1: fnp.ndarray,
    mu2: fnp.ndarray,
    variance2: fnp.ndarray,
    rho: fnp.ndarray,
    *,
    sigma_epsilon: float = 1e-12,
    correlation_epsilon: float = 1e-7,
) -> fnp.ndarray:
    """Return the Gaussian pair ReLU moment using the documented CDF approximation."""
    variance1 = fnp.maximum(fnp.asarray(variance1), 0.0)
    variance2 = fnp.maximum(fnp.asarray(variance2), 0.0)
    sigma1 = fnp.sqrt(variance1)
    sigma2 = fnp.sqrt(variance2)
    deterministic1 = sigma1 <= sigma_epsilon
    deterministic2 = sigma2 <= sigma_epsilon
    safe_sigma1 = fnp.where(deterministic1, 1.0, sigma1)
    safe_sigma2 = fnp.where(deterministic2, 1.0, sigma2)
    a = mu1 / safe_sigma1
    b = mu2 / safe_sigma2
    interior_rho = fnp.clip(rho, -1.0 + correlation_epsilon, 1.0 - correlation_epsilon)
    one_minus_rho2 = fnp.maximum(1.0 - interior_rho * interior_rho, correlation_epsilon)
    q = fnp.sqrt(one_minus_rho2)
    joint_cdf = bivariate_normal_cdf_approx(
        a, b, interior_rho, correlation_epsilon=correlation_epsilon
    )
    density = fnp.exp(-(a * a - 2.0 * interior_rho * a * b + b * b) / (2.0 * one_minus_rho2)) / (
        2.0 * pi * q
    )
    value = (mu1 * mu2 + interior_rho * sigma1 * sigma2) * joint_cdf
    value = value + mu1 * sigma2 * normal_pdf(b) * normal_cdf((a - interior_rho * b) / q)
    value = value + mu2 * sigma1 * normal_pdf(a) * normal_cdf((b - interior_rho * a) / q)
    value = value + sigma1 * sigma2 * one_minus_rho2 * density

    mean1 = relu_mean(mu1, variance1, sigma_epsilon=sigma_epsilon)
    mean2 = relu_mean(mu2, variance2, sigma_epsilon=sigma_epsilon)
    deterministic_value1 = fnp.maximum(mu1, 0.0) * mean2
    deterministic_value2 = mean1 * fnp.maximum(mu2, 0.0)
    value = fnp.where(deterministic1, deterministic_value1, value)
    value = fnp.where(deterministic2, deterministic_value2, value)
    equal_variable = (mu1 == mu2) & (variance1 == variance2) & (rho >= 1.0 - correlation_epsilon)
    value = fnp.where(
        equal_variable,
        relu_second_moment(mu1, variance1, sigma_epsilon=sigma_epsilon),
        value,
    )
    return fnp.maximum(value, 0.0)


def pairwise_correlation(covariance: fnp.ndarray, *, sigma_epsilon: float = 1e-12) -> fnp.ndarray:
    variance = fnp.maximum(fnp.diag(covariance), 0.0)
    sigma = fnp.sqrt(variance)
    denominator = fnp.outer(sigma, sigma)
    safe_denominator = fnp.where(denominator > sigma_epsilon, denominator, 1.0)
    correlation = fnp.where(denominator > sigma_epsilon, covariance / safe_denominator, 0.0)
    correlation = fnp.clip(correlation, -1.0, 1.0)
    fnp.fill_diagonal(correlation, fnp.where(variance > sigma_epsilon**2, 1.0, 0.0))
    return correlation


def chi_mean(dimension: int) -> float:
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    log_mean = 0.5 * log(2.0) + lgamma((dimension + 1.0) / 2.0) - lgamma(dimension / 2.0)
    return exp(log_mean)
