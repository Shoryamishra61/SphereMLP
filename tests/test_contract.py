"""Differential checks for the installed WhestBench estimator contract."""

from __future__ import annotations

import inspect

import flopscope.numpy as fnp
from whestbench import MLP
from whestbench.simulation import run_mlp_all_layers

import flopscope as flops
from estimator import Estimator


def _asymmetric_mlp() -> MLP:
    # Columns are output neurons under the official x @ w convention.
    weights = [fnp.array([[1.0, -2.0], [3.0, 4.0]], dtype=fnp.float32)]
    return MLP(width=2, depth=1, weights=weights, seed=1729, name="contract-fixture")


def test_entrypoint_signature_and_return_contract() -> None:
    mlp = _asymmetric_mlp()
    before = [w.copy() for w in mlp.weights]

    with flops.BudgetContext(flop_budget=100, quiet=True) as context:
        prediction = Estimator().predict(mlp, 100)

    assert list(inspect.signature(Estimator.predict).parameters) == ["self", "mlp", "budget"]
    assert isinstance(prediction, fnp.ndarray)
    assert prediction.shape == (mlp.depth, mlp.width)
    assert fnp.issubdtype(prediction.dtype, fnp.floating)
    assert bool(fnp.all(fnp.isfinite(prediction)))
    assert bool(fnp.all(prediction >= 0))
    assert context.flops_used == 0
    assert mlp.seed == 1729
    assert all(bool(fnp.array_equal(old, new)) for old, new in zip(before, mlp.weights))


def test_weight_orientation_is_samples_by_input_times_input_by_output() -> None:
    mlp = _asymmetric_mlp()
    inputs = fnp.array([[1.0, 2.0]], dtype=fnp.float32)

    with flops.BudgetContext(flop_budget=1_000, quiet=True):
        output = run_mlp_all_layers(mlp, inputs)[0]

    # [1, 2] @ [[1, -2], [3, 4]] = [7, 6], already non-negative.
    expected = fnp.array([[7.0, 6.0]], dtype=fnp.float32)
    assert bool(fnp.array_equal(output, expected))


def test_prediction_is_deterministic() -> None:
    mlp = _asymmetric_mlp()
    with flops.BudgetContext(flop_budget=100, quiet=True):
        first = Estimator().predict(mlp, 100)
    with flops.BudgetContext(flop_budget=100, quiet=True):
        second = Estimator().predict(mlp, 100)
    assert bool(fnp.array_equal(first, second))
