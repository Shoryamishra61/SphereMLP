from __future__ import annotations

from math import pi, sqrt

import flopscope.numpy as fnp
import numpy as np
import pytest

import flopscope as flops
from tests.reference_impl import relu_moments
from whest_solution.moments import chi_mean, relu_mean, relu_second_moment, relu_variance


def test_univariate_moments_match_float64_reference() -> None:
    mu = np.array([-20.0, -2.0, 0.0, 0.5, 3.0, 20.0])
    variance = np.array([1.0, 0.3, 2.0, 4.0, 0.2, 1.0])
    expected_mean, expected_second, expected_variance = relu_moments(mu, variance)
    with flops.BudgetContext(flop_budget=1_000_000, quiet=True):
        actual_mean = relu_mean(fnp.array(mu), fnp.array(variance))
        actual_second = relu_second_moment(fnp.array(mu), fnp.array(variance))
        actual_variance = relu_variance(fnp.array(mu), fnp.array(variance))
    np.testing.assert_allclose(np.asarray(actual_mean), expected_mean, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(actual_second), expected_second, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(actual_variance), expected_variance, atol=1e-12, rtol=1e-12
    )


def test_deterministic_branch_is_explicit_and_non_negative() -> None:
    mu = fnp.array([-3.0, 0.0, 2.5])
    tiny_variance = fnp.array([0.0, 1e-30, 0.0])
    with flops.BudgetContext(flop_budget=1_000_000, quiet=True):
        mean = relu_mean(mu, tiny_variance, sigma_epsilon=1e-12)
        second = relu_second_moment(mu, tiny_variance, sigma_epsilon=1e-12)
        variance = relu_variance(mu, tiny_variance, sigma_epsilon=1e-12)
    np.testing.assert_array_equal(np.asarray(mean), [0.0, 0.0, 2.5])
    np.testing.assert_array_equal(np.asarray(second), [0.0, 0.0, 6.25])
    np.testing.assert_array_equal(np.asarray(variance), [0.0, 0.0, 0.0])


def test_chi_mean_known_values_and_domain() -> None:
    assert chi_mean(1) == pytest.approx(sqrt(2.0 / pi), rel=1e-14)
    assert chi_mean(2) == pytest.approx(sqrt(pi / 2.0), rel=1e-14)
    assert sqrt(255.5) < chi_mean(256) < sqrt(256.0)
    with pytest.raises(ValueError):
        chi_mean(0)
