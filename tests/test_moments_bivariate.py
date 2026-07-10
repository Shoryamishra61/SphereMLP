from __future__ import annotations

from math import pi

import flopscope.numpy as fnp
import numpy as np
import pytest

import flopscope as flops
from tests.reference_impl import bivariate_relu_second_moment_quadrature, relu_moments
from whest_solution.moments import (
    bivariate_relu_second_moment,
    pairwise_correlation,
    zero_mean_bivariate_relu_second_moment,
)


def test_zero_mean_arc_cosine_boundaries() -> None:
    rho = fnp.array([-1.0, 0.0, 1.0])
    with flops.BudgetContext(flop_budget=1_000_000, quiet=True):
        actual = zero_mean_bivariate_relu_second_moment(2.0, 3.0, rho)
    np.testing.assert_allclose(np.asarray(actual), [0.0, 3.0 / pi, 3.0], atol=1e-14)


def test_general_pair_matches_independent_quadrature_fixtures_and_swap_symmetry() -> None:
    fixtures = [
        (0.4, 1.2, -0.7, 0.5, 0.0),
        (0.5, 0.8, 1.1, 1.5, 0.6),
        (-1.2, 2.0, 0.3, 0.7, -0.5),
    ]
    for mu1, var1, mu2, var2, rho in fixtures:
        reference = bivariate_relu_second_moment_quadrature(mu1, var1, mu2, var2, rho, order=200)
        with flops.BudgetContext(flop_budget=1_000_000, quiet=True):
            actual = bivariate_relu_second_moment(mu1, var1, mu2, var2, rho)
            swapped = bivariate_relu_second_moment(mu2, var2, mu1, var1, rho)
        assert float(actual) == pytest.approx(reference, abs=3.5e-3)
        assert float(actual) == pytest.approx(float(swapped), abs=1e-12)


def test_general_pair_deterministic_and_equal_variable_branches_are_finite() -> None:
    with flops.BudgetContext(flop_budget=2_000_000, quiet=True):
        deterministic = bivariate_relu_second_moment(2.0, 0.0, -0.2, 1.5, 0.9)
        equal = bivariate_relu_second_moment(0.3, 1.2, 0.3, 1.2, 1.0)
    other_mean = float(relu_moments(np.array(-0.2), np.array(1.5))[0])
    equal_second = float(relu_moments(np.array(0.3), np.array(1.2))[1])
    assert float(deterministic) == pytest.approx(2.0 * other_mean, abs=1e-12)
    assert np.isfinite(float(equal))
    assert float(equal) == pytest.approx(equal_second, abs=1e-6)


def test_pairwise_correlation_handles_zero_variance_and_clips_roundoff() -> None:
    covariance = fnp.array([[4.0, 2.0000001, 0.0], [2.0000001, 1.0, 0.0], [0.0, 0.0, 0.0]])
    with flops.BudgetContext(flop_budget=1_000_000, quiet=True):
        correlation = pairwise_correlation(covariance)
    expected = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    np.testing.assert_allclose(np.asarray(correlation), expected, atol=0.0)
