from __future__ import annotations

import flopscope.numpy as fnp
import numpy as np
from whestbench import MLP

import flopscope as flops
from whest_solution.sampling import spherical_propagation


def _mlp(weights: list[np.ndarray], *, seed: int = 17) -> MLP:
    width = weights[0].shape[0]
    return MLP(
        width=width, depth=len(weights), weights=[fnp.asarray(w) for w in weights], seed=seed
    )


def _run(mlp: MLP, **kwargs):
    with flops.BudgetContext(flop_budget=5_000_000_000, quiet=True):
        return spherical_propagation(mlp, **kwargs)


def test_spherical_matches_one_layer_analytic_mean() -> None:
    weights = np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32)
    estimate = _run(_mlp([weights]), samples=32768, batch_size=1024)
    expected = np.linalg.norm(weights, axis=0) / np.sqrt(2.0 * np.pi)
    np.testing.assert_allclose(np.asarray(estimate.predictions[0]), expected, atol=0.015)


def test_radial_factor_is_applied_once_for_all_layers() -> None:
    """Repeated ReLUs with identity weights must retain the same mean.

    This catches the catastrophic error of multiplying the chi-radius mean
    inside every layer of the directional forward pass.
    """
    width = 2
    estimate = _run(_mlp([np.eye(width), np.eye(width)]), samples=32768, batch_size=1024)
    expected = np.full(width, 1.0 / np.sqrt(2.0 * np.pi))
    np.testing.assert_allclose(np.asarray(estimate.predictions[0]), expected, atol=0.015)
    np.testing.assert_allclose(np.asarray(estimate.predictions[1]), expected, atol=0.015)


def test_spherical_seed_determinism_and_batch_invariance() -> None:
    mlp = _mlp([np.eye(4), np.full((4, 4), 0.2)])
    first = _run(mlp, samples=1024, batch_size=128)
    second = _run(mlp, samples=1024, batch_size=128)
    alternate_batch = _run(mlp, samples=1024, batch_size=256)
    np.testing.assert_array_equal(np.asarray(first.predictions), np.asarray(second.predictions))
    np.testing.assert_allclose(
        np.asarray(first.predictions), np.asarray(alternate_batch.predictions), atol=1e-6
    )
    assert np.all(np.asarray(first.standard_error) >= 0.0)


def test_antithetic_has_correct_shape_and_finite_marginals() -> None:
    estimate = _run(_mlp([np.eye(3)]), samples=1024, batch_size=128, antithetic=True)
    assert estimate.predictions.shape == (1, 3)
    assert bool(fnp.all(fnp.isfinite(estimate.predictions)))
    assert bool(fnp.all(estimate.predictions >= 0.0))


def test_orthogonal_blocks_are_deterministic_and_finite() -> None:
    estimate = _run(_mlp([np.eye(4)]), samples=1024, batch_size=128, orthogonal_blocks=True)
    repeated = _run(_mlp([np.eye(4)]), samples=1024, batch_size=128, orthogonal_blocks=True)
    np.testing.assert_array_equal(
        np.asarray(estimate.predictions), np.asarray(repeated.predictions)
    )
    assert bool(fnp.all(fnp.isfinite(estimate.predictions)))


def test_randomized_latin_hypercube_is_deterministic_and_finite() -> None:
    kwargs = dict(samples=1024, batch_size=128, latin_hypercube=True, seed=91)
    estimate = _run(_mlp([np.eye(4)]), **kwargs)
    repeated = _run(_mlp([np.eye(4)]), **kwargs)
    np.testing.assert_array_equal(
        np.asarray(estimate.predictions), np.asarray(repeated.predictions)
    )
    assert bool(fnp.all(fnp.isfinite(estimate.predictions)))
    assert bool(fnp.all(estimate.predictions >= 0.0))


def test_final_layer_only_preserves_final_estimate() -> None:
    mlp = _mlp([np.eye(4), np.full((4, 4), 0.2)])
    full = _run(mlp, samples=1024, batch_size=128)
    final_only = _run(mlp, samples=1024, batch_size=128, final_layer_only=True)
    np.testing.assert_array_equal(
        np.asarray(final_only.predictions[-1]), np.asarray(full.predictions[-1])
    )
    np.testing.assert_array_equal(np.asarray(final_only.predictions[:-1]), 0.0)


def test_sampling_options_are_checked() -> None:
    mlp = _mlp([np.eye(2)])
    with np.testing.assert_raises(ValueError):
        _run(mlp, samples=3, batch_size=2, antithetic=True)
    with np.testing.assert_raises(ValueError):
        _run(mlp, samples=129, batch_size=128, orthogonal_blocks=True)
    with np.testing.assert_raises(ValueError):
        _run(mlp, samples=1024, batch_size=128, antithetic=True, latin_hypercube=True)
