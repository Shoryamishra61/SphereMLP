from __future__ import annotations

from math import acos, pi, sqrt

import flopscope.numpy as fnp
import numpy as np
from whestbench import MLP
from whestbench.estimators import MeanPropagationEstimator

import flopscope as flops
from tests.reference_impl import (
    bivariate_relu_second_moment_quadrature,
    covariance_propagation,
    forward_all_layers,
    monte_carlo_layer_means,
    relu_moments,
    scalar_propagation,
)


def test_plain_forward_uses_official_column_output_orientation() -> None:
    weight = np.array([[1.0, -2.0], [3.0, 4.0]])
    actual = forward_all_layers([weight], np.array([[1.0, 2.0]]))[0]
    np.testing.assert_array_equal(actual, [[7.0, 6.0]])


def test_exact_first_layer_means_match_fixed_seed_monte_carlo() -> None:
    weight = np.array([[1.0, -0.5, 0.2], [0.3, 0.8, -1.2], [-0.7, 0.1, 0.4]])
    expected = np.linalg.norm(weight, axis=0) / sqrt(2.0 * pi)
    actual = monte_carlo_layer_means([weight], 400_000, seed=20260711)[0]
    np.testing.assert_allclose(actual, expected, atol=3.5e-3, rtol=0.0)


def test_zero_mean_bivariate_quadrature_matches_arc_cosine_formula() -> None:
    sigma1, sigma2 = 1.3, 0.7
    for rho in (-0.8, 0.0, 0.65):
        expected = sigma1 * sigma2 / (2.0 * pi) * (sqrt(1.0 - rho * rho) + (pi - acos(rho)) * rho)
        actual = bivariate_relu_second_moment_quadrature(
            0.0, sigma1**2, 0.0, sigma2**2, rho, order=160
        )
        assert abs(actual - expected) < 1.2e-3


def test_general_bivariate_quadrature_matches_independence_identity() -> None:
    mean1 = float(relu_moments(np.array(0.4), np.array(1.2))[0])
    mean2 = float(relu_moments(np.array(-0.7), np.array(0.5))[0])
    actual = bivariate_relu_second_moment_quadrature(0.4, 1.2, -0.7, 0.5, 0.0, order=160)
    assert abs(actual - mean1 * mean2) < 1.2e-3


def test_scalar_reference_matches_installed_optimized_baseline() -> None:
    rng = np.random.default_rng(314159)
    weights = [rng.normal(scale=0.4, size=(4, 4)) for _ in range(3)]
    reference, _ = scalar_propagation(weights)
    mlp = MLP(
        width=4,
        depth=3,
        weights=[fnp.array(weight) for weight in weights],
        seed=9,
    )
    with flops.BudgetContext(flop_budget=1_000_000, quiet=True):
        optimized = MeanPropagationEstimator().predict(mlp, 1_000_000)
    np.testing.assert_allclose(np.asarray(optimized), reference, atol=1e-10, rtol=1e-10)


def test_first_layer_covariance_reference_matches_monte_carlo() -> None:
    weight = np.array([[0.9, -0.4, 0.2], [0.1, 0.7, -0.8], [-0.5, 0.3, 0.6]])
    reference = covariance_propagation([weight], quadrature_order=128)
    rng = np.random.default_rng(271828)
    samples = forward_all_layers([weight], rng.standard_normal((500_000, 3)))[0]
    np.testing.assert_allclose(samples.mean(axis=0), reference.means[0], atol=3e-3, rtol=0.0)
    np.testing.assert_allclose(
        np.cov(samples, rowvar=False, ddof=0),
        reference.covariances[0],
        atol=3e-3,
        rtol=0.0,
    )
