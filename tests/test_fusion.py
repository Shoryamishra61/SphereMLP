from __future__ import annotations

import flopscope.numpy as fnp
import numpy as np
import pytest

from whest_solution.fusion import convex_fuse, layerwise_convex_fuse


def test_convex_fusion_endpoints_and_nonnegativity() -> None:
    primary = fnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    secondary = fnp.asarray([[5.0, 6.0], [7.0, 8.0]])
    np.testing.assert_array_equal(np.asarray(convex_fuse(primary, secondary, 0.0)), primary)
    np.testing.assert_array_equal(np.asarray(convex_fuse(primary, secondary, 1.0)), secondary)
    assert np.all(np.asarray(convex_fuse(primary, secondary, 0.5)) >= 0.0)


def test_layerwise_fusion_and_permutation_equivariance() -> None:
    primary = fnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    secondary = fnp.asarray([[5.0, 6.0], [7.0, 8.0]])
    weights = (0.25, 0.75)
    fused = layerwise_convex_fuse(primary, secondary, weights)
    expected = np.asarray([[2.0, 3.0], [6.0, 7.0]])
    np.testing.assert_allclose(np.asarray(fused), expected)
    permutation = np.array([1, 0])
    permuted = layerwise_convex_fuse(primary[:, permutation], secondary[:, permutation], weights)
    np.testing.assert_allclose(np.asarray(permuted), expected[:, permutation])


def test_fusion_rejects_bad_weights_and_shapes() -> None:
    primary = fnp.asarray([[1.0, 2.0]])
    with pytest.raises(ValueError):
        convex_fuse(primary, primary, 1.1)
    with pytest.raises(ValueError):
        layerwise_convex_fuse(primary, primary, ())
    with pytest.raises(ValueError):
        convex_fuse(primary, fnp.asarray([[1.0]]), 0.5)
