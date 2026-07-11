from __future__ import annotations

from unittest.mock import patch

import flopscope.numpy as fnp
from whestbench import MLP

import flopscope as flops
from estimator import Estimator


def _fixture() -> MLP:
    return MLP(width=2, depth=1, weights=[fnp.eye(2)], seed=1)


def test_low_budget_returns_free_valid_emergency_result() -> None:
    with flops.BudgetContext(flop_budget=100, quiet=True) as context:
        prediction = Estimator().predict(_fixture(), 100)
    assert prediction.shape == (1, 2)
    assert bool(fnp.all(prediction == 0.0))
    assert context.flops_used == 0


def test_scalar_exception_cannot_escape_or_replace_retained_result() -> None:
    with patch("estimator._propagate_scalar", side_effect=RuntimeError("injected")):
        with flops.BudgetContext(flop_budget=1_000_000, quiet=True):
            prediction = Estimator().predict(_fixture(), 1_000_000)
    assert bool(fnp.all(prediction == 0.0))


def test_invalid_scalar_candidate_cannot_replace_retained_result() -> None:
    invalid = fnp.array([[float("nan"), 1.0]])
    with patch("estimator._propagate_scalar", return_value=invalid):
        with flops.BudgetContext(flop_budget=1_000_000, quiet=True):
            prediction = Estimator().predict(_fixture(), 1_000_000)
    assert bool(fnp.all(prediction == 0.0))
