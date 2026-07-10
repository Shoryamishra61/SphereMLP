from __future__ import annotations

from dataclasses import replace

import pytest

from whest_solution.config import RuntimeConfig, stable_config_hash


@pytest.fixture
def config() -> RuntimeConfig:
    return RuntimeConfig(
        sigma_epsilon=1e-12,
        correlation_epsilon=1e-7,
        variance_clip_tolerance=1e-10,
        compute_profile="unfrozen_test_profile",
        target_compute_ratio=0.1,
        hard_budget_headroom=0.8,
        soft_time_limit_seconds=40.0,
        use_covariance=False,
        use_spherical=False,
        spherical_samples=0,
        spherical_batch_size=0,
        antithetic=False,
        orthogonal_blocks=False,
        rqmc_randomizations=0,
        use_k3=False,
        k3_rank=0,
        k3_layers=(),
        fusion_mode="none",
        fixed_fusion_weight=0.0,
        layerwise_fusion_weights=(),
        use_calibrator=False,
        calibrator_artifact=None,
    )


def test_config_is_frozen_and_hash_is_deterministic(config: RuntimeConfig) -> None:
    assert stable_config_hash(config) == stable_config_hash(config)
    assert len(stable_config_hash(config)) == 64
    with pytest.raises(AttributeError):
        config.compute_profile = "mutated"  # type: ignore[misc]


def test_hash_changes_when_any_field_changes(config: RuntimeConfig) -> None:
    assert stable_config_hash(config) != stable_config_hash(
        replace(config, target_compute_ratio=0.15)
    )
