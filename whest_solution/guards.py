"""Runtime candidate validation and fallback guards."""

from __future__ import annotations

import flopscope.numpy as fnp


def valid_prediction(candidate: object, *, depth: int, width: int) -> bool:
    """Return whether a candidate is safe to expose to the grader."""
    if not isinstance(candidate, fnp.ndarray):
        return False
    if tuple(candidate.shape) != (depth, width):
        return False
    if not fnp.issubdtype(candidate.dtype, fnp.floating):
        return False
    if not bool(fnp.all(fnp.isfinite(candidate))):
        return False
    return bool(fnp.all(candidate >= 0.0))
