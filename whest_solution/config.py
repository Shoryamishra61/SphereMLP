"""Immutable runtime configuration and reproducible hashing."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256


@dataclass(frozen=True)
class RuntimeConfig:
    sigma_epsilon: float
    correlation_epsilon: float
    variance_clip_tolerance: float

    compute_profile: str
    target_compute_ratio: float
    hard_budget_headroom: float
    soft_time_limit_seconds: float

    use_covariance: bool
    use_spherical: bool
    spherical_samples: int
    spherical_batch_size: int
    antithetic: bool
    orthogonal_blocks: bool
    rqmc_randomizations: int

    use_k3: bool
    k3_rank: int
    k3_layers: tuple[int, ...]

    fusion_mode: str
    fixed_fusion_weight: float
    layerwise_fusion_weights: tuple[float, ...]

    use_calibrator: bool
    calibrator_artifact: str | None


def stable_config_hash(config: RuntimeConfig) -> str:
    """Return a deterministic SHA-256 over the complete frozen configuration."""
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(payload.encode("utf-8")).hexdigest()
