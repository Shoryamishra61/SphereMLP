"""Adapters for the installed WhestBench contract proven in T00."""

from __future__ import annotations

from typing import Iterable

import flopscope.numpy as fnp
from whestbench import MLP

WEIGHT_CONVENTION = "samples_by_input @ input_by_output"


def output_shape(mlp: MLP) -> tuple[int, int]:
    return (int(mlp.depth), int(mlp.width))


def ordered_weights(mlp: MLP) -> Iterable[fnp.ndarray]:
    """Yield official input-by-output matrices without copying or mutation."""
    return iter(mlp.weights)


def estimator_seed(mlp: MLP) -> int:
    return int(mlp.seed)
