from __future__ import annotations

from math import pi, sqrt

import flopscope.numpy as fnp
import numpy as np
from whestbench import MLP

import flopscope as flops
from tests.reference_impl import scalar_propagation
from whest_solution.scalar import propagate_scalar, scalar_budget_reserve


def _mlp(weights: list[np.ndarray], *, seed: int = 7) -> MLP:
    width = weights[0].shape[0]
    return MLP(
        width=width,
        depth=len(weights),
        weights=[fnp.array(weight) for weight in weights],
        seed=seed,
    )


def test_first_layer_is_exact_from_output_column_norms() -> None:
    weight = np.array([[1.0, -0.5, 0.2], [0.3, 0.8, -1.2], [-0.7, 0.1, 0.4]])
    with flops.BudgetContext(flop_budget=1_000_000, quiet=True):
        actual = propagate_scalar(_mlp([weight]))
    expected = np.linalg.norm(weight, axis=0) / sqrt(2.0 * pi)
    np.testing.assert_allclose(np.asarray(actual[0]), expected, atol=1e-12, rtol=1e-12)


def test_random_small_network_matches_independent_reference() -> None:
    rng = np.random.default_rng(404)
    weights = [rng.normal(scale=0.5, size=(4, 4)) for _ in range(3)]
    expected, _ = scalar_propagation(weights)
    with flops.BudgetContext(flop_budget=10_000_000, quiet=True):
        actual = propagate_scalar(_mlp(weights))
    np.testing.assert_allclose(np.asarray(actual), expected, atol=1e-11, rtol=1e-11)


def test_zero_columns_and_diagonal_weights_stay_finite() -> None:
    weights = [np.diag([0.0, 1.0, 2.0]), np.diag([1.0, 0.0, 0.5])]
    with flops.BudgetContext(flop_budget=10_000_000, quiet=True):
        actual = propagate_scalar(_mlp(weights))
    assert actual.shape == (2, 3)
    assert bool(fnp.all(fnp.isfinite(actual)))
    assert bool(fnp.all(actual >= 0.0))
    assert float(actual[0, 0]) == 0.0
    assert float(actual[1, 1]) == 0.0


def test_scalar_budget_reserve_rejects_validator_probe_budget() -> None:
    assert scalar_budget_reserve(width=4, depth=2) > 100
    assert scalar_budget_reserve(width=256, depth=32) < 272_000_000_000
