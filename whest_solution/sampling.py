"""Batched spherical Rao-Blackwellized sampling for bias-free ReLU MLPs."""

from __future__ import annotations

from dataclasses import dataclass

import flopscope.numpy as fnp
from whestbench import MLP

from .contracts import estimator_seed
from .moments import chi_mean


@dataclass(frozen=True)
class SphericalEstimate:
    """All-layer directional estimate and independent-sample standard errors."""

    predictions: fnp.ndarray
    standard_error: fnp.ndarray
    samples: int


def _validate_options(
    *, samples: int, batch_size: int, antithetic: bool, orthogonal_blocks: bool, width: int
) -> None:
    if samples <= 1:
        raise ValueError("samples must exceed one to estimate uncertainty")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if antithetic and samples % 2:
        raise ValueError("antithetic sampling requires an even total sample count")
    if antithetic and batch_size % 2:
        raise ValueError("antithetic sampling requires an even batch_size")
    if orthogonal_blocks and antithetic:
        raise ValueError("orthogonal and antithetic direction modes are mutually exclusive")
    if orthogonal_blocks and (samples % width or batch_size % width):
        raise ValueError("orthogonal blocks require sample and batch counts divisible by width")


def _orthogonal_directions(rng, *, width: int, count: int) -> fnp.ndarray:
    """Return Haar-uniform orthogonal direction rows using metered QR blocks."""
    blocks: list[fnp.ndarray] = []
    for _ in range(count // width):
        gaussian = fnp.asarray(rng.standard_normal((width, width), dtype=fnp.float64))
        q, r = fnp.linalg.qr(gaussian)
        signs = fnp.sign(fnp.diag(r))
        signs = fnp.where(signs == 0.0, 1.0, signs)
        blocks.append(q * signs)
    return blocks[0] if len(blocks) == 1 else fnp.concatenate(blocks, axis=0)


def spherical_propagation(
    mlp: MLP,
    *,
    samples: int,
    batch_size: int,
    antithetic: bool = False,
    orthogonal_blocks: bool = False,
    final_layer_only: bool = False,
    seed: int | None = None,
) -> SphericalEstimate:
    """Estimate all ReLU-layer means by integrating the Gaussian radius exactly.

    For a zero-bias ReLU network ``h`` and ``X = R U`` with ``U`` uniform on
    the unit sphere, positive homogeneity gives ``E[h(X)] = E[R] E[h(U)]``.
    We propagate unit-sphere directions through the network and accumulate
    the raw direction-space activations.  The scalar ``E[R] = chi_mean(width)``
    is applied **once** to the final mean and standard error — not inside the
    per-layer loop (which would compound the factor by ``depth`` times).
    """
    _validate_options(
        samples=samples,
        batch_size=batch_size,
        antithetic=antithetic,
        orthogonal_blocks=orthogonal_blocks,
        width=int(mlp.width),
    )
    rng = fnp.random.default_rng(estimator_seed(mlp) if seed is None else int(seed))
    radial_mean = chi_mean(int(mlp.width))
    # Accumulators for the direction-space activations (no radial factor yet).
    totals = [fnp.zeros((mlp.width,), dtype=fnp.float64) for _ in range(mlp.depth)]
    squared_totals = [fnp.zeros((mlp.width,), dtype=fnp.float64) for _ in range(mlp.depth)]
    completed = 0
    while completed < samples:
        count = min(batch_size, samples - completed)
        if orthogonal_blocks:
            directions = _orthogonal_directions(rng, width=int(mlp.width), count=count)
        elif antithetic:
            # ``samples`` is the number of full forward evaluations, including
            # the negative directions.
            base_count = count // 2
            directions = fnp.asarray(
                rng.standard_normal((base_count, mlp.width), dtype=fnp.float64)
            )
            directions = fnp.concatenate((directions, -directions), axis=0)
        else:
            directions = fnp.asarray(rng.standard_normal((count, mlp.width), dtype=fnp.float64))
        norms = fnp.sqrt(fnp.sum(directions * directions, axis=1, keepdims=True))
        safe_norms = fnp.where(norms > 0.0, norms, 1.0)
        activations = directions / safe_norms
        for layer, weight in enumerate(mlp.weights):
            activations = fnp.maximum(activations @ weight, 0.0)
            if final_layer_only and layer != mlp.depth - 1:
                continue
            # Accumulate raw direction-space values — no radial factor here.
            totals[layer] = totals[layer] + fnp.sum(activations, axis=0)
            squared_totals[layer] = squared_totals[layer] + fnp.sum(
                activations * activations, axis=0
            )
        completed += count
    # Direction-space means: E_U[h(U)]
    dir_mean = fnp.stack(totals, axis=0) / samples
    squared = fnp.stack(squared_totals, axis=0)
    # Unbiased sample variance of E_U[h(U)]; clamp for numerical safety.
    variance = fnp.maximum((squared - samples * dir_mean * dir_mean) / (samples - 1), 0.0)
    dir_se = fnp.sqrt(variance / samples)
    # Apply the radial factor once: E[h(X)] = E[R] * E_U[h(U)]
    mean = dir_mean * radial_mean
    standard_error = dir_se * radial_mean
    if final_layer_only:
        mean = fnp.concatenate(
            (fnp.zeros((mlp.depth - 1, mlp.width), dtype=fnp.float64), mean[-1:]), axis=0
        )
        standard_error = fnp.concatenate(
            (fnp.zeros((mlp.depth - 1, mlp.width), dtype=fnp.float64), standard_error[-1:]),
            axis=0,
        )
    return SphericalEstimate(
        predictions=fnp.asarray(mean, dtype=fnp.float32),
        standard_error=fnp.asarray(standard_error, dtype=fnp.float32),
        samples=samples,
    )
