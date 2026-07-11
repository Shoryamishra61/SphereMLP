from __future__ import annotations

import flopscope.numpy as fnp
import numpy as np
from whestbench import MLP

import flopscope as flops
from tests.reference_impl import covariance_propagation, monte_carlo_layer_means, scalar_propagation
from whest_solution.covariance import covariance_budget_reserve, propagate_covariance


def _mlp(weights: list[np.ndarray]) -> MLP:
    width = weights[0].shape[0]
    return MLP(width=width, depth=len(weights), weights=[fnp.array(w) for w in weights], seed=4)


def _run(weights: list[np.ndarray]):
    with flops.BudgetContext(flop_budget=500_000_000, quiet=True):
        return propagate_covariance(_mlp(weights))


def test_first_layer_matches_independent_joint_reference() -> None:
    weight = np.array([[0.9, -0.4, 0.2], [0.1, 0.7, -0.8], [-0.5, 0.3, 0.6]])
    actual = _run([weight])
    reference = covariance_propagation([weight], quadrature_order=160)
    np.testing.assert_allclose(np.asarray(actual.predictions), reference.means, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(actual.covariances[0]), reference.covariances[0], atol=1.5e-3
    )


def test_covariance_is_symmetric_has_exact_diagonal_and_is_psd_on_small_fixture() -> None:
    rng = np.random.default_rng(505)
    weights = [rng.normal(scale=0.45, size=(4, 4)) for _ in range(3)]
    result = _run(weights)
    for covariance, diagnostics in zip(result.covariances, result.diagnostics):
        array = np.asarray(covariance)
        np.testing.assert_allclose(array, array.T, atol=1e-12)
        assert np.min(np.diag(array)) >= 0.0
        assert np.min(np.linalg.eigvalsh(array)) >= -4e-3
        assert diagnostics.symmetry_error <= 1e-12
        assert diagnostics.negative_diagonal_count == 0


def test_hidden_neuron_permutation_equivariance() -> None:
    rng = np.random.default_rng(606)
    first = rng.normal(scale=0.5, size=(4, 4))
    second = rng.normal(scale=0.5, size=(4, 4))
    permutation = np.array([2, 0, 3, 1])
    permuted_first = first[:, permutation]
    permuted_second = second[permutation, :]
    original = _run([first, second]).predictions
    permuted = _run([permuted_first, permuted_second]).predictions
    np.testing.assert_allclose(
        np.asarray(permuted[0]), np.asarray(original[0])[permutation], atol=1e-10
    )
    np.testing.assert_allclose(np.asarray(permuted[1]), np.asarray(original[1]), atol=1e-10)


def test_positive_scaling_of_first_layer_scales_all_states() -> None:
    rng = np.random.default_rng(707)
    weights = [rng.normal(scale=0.4, size=(3, 3)) for _ in range(2)]
    scale = 2.5
    base = _run(weights)
    scaled = _run([scale * weights[0], weights[1]])
    np.testing.assert_allclose(
        np.asarray(scaled.predictions), scale * np.asarray(base.predictions), rtol=1e-10
    )
    for scaled_cov, base_cov in zip(scaled.covariances, base.covariances):
        np.testing.assert_allclose(
            np.asarray(scaled_cov), scale**2 * np.asarray(base_cov), rtol=1e-9, atol=1e-12
        )


def test_extreme_correlations_and_zero_columns_remain_finite() -> None:
    weight = np.array(
        [[1.0, 1.0, -1.0, 0.0], [2.0, 2.0, -2.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.5, 0.5, -0.5, 0.0]]
    )
    result = _run([weight, np.eye(4)])
    assert bool(fnp.all(fnp.isfinite(result.predictions)))
    assert bool(fnp.all(result.predictions >= 0.0))


def test_small_deep_network_tracks_fixed_seed_high_sample_monte_carlo() -> None:
    rng = np.random.default_rng(808)
    width = 12
    weights = [rng.normal(scale=(2.0 / width) ** 0.5, size=(width, width)) for _ in range(3)]
    result = _run(weights)
    reference = monte_carlo_layer_means(weights, 200_000, seed=909)
    scalar, _ = scalar_propagation(weights)
    covariance_error = np.asarray(result.predictions) - reference
    scalar_error = scalar - reference
    assert np.max(np.abs(covariance_error)) < 0.15
    assert np.mean(covariance_error**2) < np.mean(scalar_error**2)


def test_covariance_reserve_skips_validator_and_fits_official_budget() -> None:
    assert covariance_budget_reserve(width=4, depth=2) > 100
    assert covariance_budget_reserve(width=256, depth=32) < 272_000_000_000
