"""Official WhestBench estimator entry point.

T08 integrates spherical Rao-Blackwellized sampling as the primary estimator,
with the exact-first-layer scalar propagation retained as the validated
fallback.

Budget-adaptive strategy:
    Score = MSE × max(0.1, C/B).
    For unbiased estimators, MSE ∝ 1/N and C ∝ N, so
        Score ∝ max(0.1/N, 1/B)
    is always DECREASING in N.  We therefore maximise N within a safe
    fraction of the hard FLOP budget.  Batch size 512 minimises residual
    wall-time overhead.

Measured on the Mini split (100 MLPs):
    N=3,584  → adjusted score 1.98e-6  (13.4% compute ratio)
    N=49,152 → BUST: runner residual inflated effective compute to 189%
    Runner overhead factor: effective ≈ 2.4-2.5× analytical FLOPs

Conservative target: 30% analytical ratio → ~75% effective ratio (with 2.5× overhead)
    N=17,408 → analytical 73.6B (27%) → effective ~184B (67%)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from whestbench import BaseEstimator

if TYPE_CHECKING:
    import flopscope.numpy as fnp
    from whestbench import MLP

# ---------------------------------------------------------------------------
# Calibrated from actual Mini runner measurements:
#   - Per-sample analytical cost: ~4.23M FLOPs (width=256, depth=32)
#   - Runner overhead factor: effective ≈ 2.4-2.5× analytical
#   - At 3584 samples: 15.2B analytical → 36.4B effective (13.4% ratio)
#   - At 49152 samples: 208B analytical → 515B effective (189% BUST!)
#
# Target: 30% analytical ratio → ~75% effective ratio (safe margin)
# This gives N ≈ 0.30 × B / 4.23M ≈ 19,300 → round to 17,408 (34 × 512)
# ---------------------------------------------------------------------------
_DEFAULT_SPHERICAL_SAMPLES = 17408
_SPHERICAL_BATCH_SIZE = 512
_BUDGET_ANALYTICAL_FRACTION = 0.30


def _per_sample_flops(*, width: int, depth: int) -> int:
    """Analytical FLOPs consumed per spherical sample."""
    return int(depth * 2 * width * width + 8 * depth * width)


def _fixed_overhead_flops() -> int:
    """Fixed analytical FLOPs for RNG, normalization, stack/cast."""
    return 2_500_000


def _scalar_budget_reserve(*, width: int, depth: int) -> int:
    return int(32 * depth * width * width + 128 * depth * width)


def _compute_samples(budget: int, *, width: int, depth: int) -> int:
    """Compute the largest safe sample count that fits within budget."""
    scalar_cost = _scalar_budget_reserve(width=width, depth=depth)
    available = int(budget * _BUDGET_ANALYTICAL_FRACTION) - scalar_cost - _fixed_overhead_flops()
    if available <= 0:
        return 0
    per_sample = _per_sample_flops(width=width, depth=depth)
    n = available // per_sample
    # Round down to nearest multiple of batch size for clean batching
    n = (n // _SPHERICAL_BATCH_SIZE) * _SPHERICAL_BATCH_SIZE
    return max(0, min(n, _DEFAULT_SPHERICAL_SAMPLES))


def _propagate_scalar(mlp: "MLP") -> "fnp.ndarray":
    from whest_solution.scalar import propagate_scalar

    return propagate_scalar(mlp)


def _propagate_spherical(mlp: "MLP", *, samples: int, batch_size: int) -> "fnp.ndarray":
    from whest_solution.sampling import spherical_propagation

    result = spherical_propagation(mlp, samples=samples, batch_size=batch_size)
    return result.predictions


def _valid_prediction(candidate: object, *, depth: int, width: int) -> bool:
    from whest_solution.guards import valid_prediction

    return valid_prediction(candidate, depth=depth, width=width)


class Estimator(BaseEstimator):
    """Return a finite, non-negative result satisfying the official contract.

    Safety contract:
    1. Create a free valid zero result.
    2. Compute and validate the scalar fallback.
    3. Retain the latest valid result.
    4. Dynamically compute the maximum spherical sample count from budget.
    5. Attempt spherical sampling.
    6. Replace the retained result only after validation.
    7. Return the last validated result whenever an optional branch fails.
    8. Never return an unvalidated array.
    9. Never allow an optional exception to invalidate the estimator.
    """

    def predict(self, mlp: "MLP", budget: int) -> "fnp.ndarray":
        import flopscope.numpy as fnp

        depth = int(mlp.depth)
        width = int(mlp.width)

        # Step 1: free valid zero result
        retained = fnp.zeros((depth, width), dtype=fnp.float32)

        # Step 2: compute and validate the scalar fallback
        if budget < _scalar_budget_reserve(width=width, depth=depth):
            return retained

        try:
            candidate = _propagate_scalar(mlp)
            if _valid_prediction(candidate, depth=depth, width=width):
                retained = candidate
        except Exception:
            pass

        # Step 3: dynamic budget-adaptive spherical sampling
        samples = _compute_samples(budget, width=width, depth=depth)
        if samples >= 512:
            try:
                spherical_candidate = _propagate_spherical(
                    mlp, samples=samples, batch_size=_SPHERICAL_BATCH_SIZE
                )
                if _valid_prediction(spherical_candidate, depth=depth, width=width):
                    retained = spherical_candidate
            except Exception:
                pass

        return retained
