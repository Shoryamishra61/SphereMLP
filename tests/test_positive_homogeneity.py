from __future__ import annotations

import flopscope.numpy as fnp
import numpy as np


def test_relu_network_is_positive_homogeneous() -> None:
    weights = [fnp.asarray([[1.0, -0.4], [0.3, 0.8]]), fnp.asarray([[0.6, 0.2], [-0.1, 0.5]])]
    inputs = fnp.asarray([[0.7, -1.3], [-0.2, 0.4]])
    scale = 2.75
    base = inputs
    scaled = scale * inputs
    for weight in weights:
        base = fnp.maximum(base @ weight, 0.0)
        scaled = fnp.maximum(scaled @ weight, 0.0)
    np.testing.assert_allclose(np.asarray(scaled), scale * np.asarray(base), atol=1e-7)
