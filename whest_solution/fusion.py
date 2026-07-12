"""Bounded, permutation-equivariant convex fusion primitives."""

from __future__ import annotations

from collections.abc import Sequence

import flopscope.numpy as fnp


def _validated_weight(weight: float) -> float:
    value = float(weight)
    if not 0.0 <= value <= 1.0:
        raise ValueError("fusion weights must lie in [0, 1]")
    return value


def convex_fuse(primary: fnp.ndarray, secondary: fnp.ndarray, weight: float) -> fnp.ndarray:
    """Return ``(1-weight)*primary + weight*secondary`` with safe bounds.

    The same scalar is applied to every neuron, so neuron permutations commute
    with the operation.  Callers validate candidate estimates before adopting
    the fused result as the currently retained prediction.
    """
    if primary.shape != secondary.shape:
        raise ValueError("fusion parents must have equal shape")
    coefficient = _validated_weight(weight)
    result = (1.0 - coefficient) * primary + coefficient * secondary
    return fnp.maximum(result, 0.0)


def layerwise_convex_fuse(
    primary: fnp.ndarray, secondary: fnp.ndarray, weights: Sequence[float]
) -> fnp.ndarray:
    """Fuse an all-layer estimate with one bounded scalar per layer."""
    if primary.shape != secondary.shape or len(primary.shape) != 2:
        raise ValueError("fusion parents must be equally shaped rank-2 arrays")
    if len(weights) != int(primary.shape[0]):
        raise ValueError("one fusion weight is required per layer")
    columns = []
    # Weight rows rather than neurons: this keeps the operation equivariant.
    for layer, weight in enumerate(weights):
        columns.append(convex_fuse(primary[layer], secondary[layer], weight))
    return fnp.stack(columns, axis=0)
