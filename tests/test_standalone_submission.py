from __future__ import annotations

import importlib.util
from pathlib import Path

import flopscope.numpy as fnp
import numpy as np
from whestbench import MLP

import flopscope as flops
from estimator import Estimator as ModularEstimator


def _standalone_estimator_class():
    path = Path("submission/estimator.py").resolve()
    spec = importlib.util.spec_from_file_location("standalone_submission", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Estimator


def test_single_file_submission_matches_modular_runtime_on_fixed_mlp() -> None:
    weights = [fnp.asarray(np.array([[0.8, -0.2], [0.3, 0.7]], dtype=np.float32))]
    mlp = MLP(width=2, depth=1, weights=weights, seed=41)
    with flops.BudgetContext(flop_budget=272_000_000_000, quiet=True):
        modular = ModularEstimator().predict(mlp, 272_000_000_000)
    StandaloneEstimator = _standalone_estimator_class()
    with flops.BudgetContext(flop_budget=272_000_000_000, quiet=True):
        standalone = StandaloneEstimator().predict(mlp, 272_000_000_000)
    np.testing.assert_array_equal(np.asarray(standalone), np.asarray(modular))


def test_standalone_sphere_is_retained_after_sanitization() -> None:
    StandaloneEstimator = _standalone_estimator_class()
    weights = [fnp.asarray(np.eye(2, dtype=np.float32))]
    mlp = MLP(width=2, depth=1, weights=weights, seed=9)
    with flops.BudgetContext(flop_budget=272_000_000_000, quiet=True):
        prediction = StandaloneEstimator().predict(mlp, 272_000_000_000)
    assert float(fnp.sum(prediction)) > 0.0
