from __future__ import annotations

import flopscope.numpy as fnp
import numpy as np

import flopscope as flops
from whest_solution.guards import valid_prediction


def test_prediction_guard_accepts_only_safe_arrays() -> None:
    fixtures = [
        (fnp.zeros((2, 3), dtype=fnp.float32), True),
        (np.zeros((2, 3)), False),
        (fnp.zeros((3, 2)), False),
        (fnp.array([[0.0, -1.0, 0.0], [0.0, 0.0, 0.0]]), False),
        (fnp.array([[0.0, float("nan"), 0.0], [0.0, 0.0, 0.0]]), False),
    ]
    with flops.BudgetContext(flop_budget=1_000_000, quiet=True):
        actual = [valid_prediction(value, depth=2, width=3) for value, _ in fixtures]
    assert actual == [expected for _, expected in fixtures]
