from __future__ import annotations

import pytest

from experiments.build_splits import PARTITION_SIZES, deterministic_partitions


def test_partitions_are_deterministic_disjoint_and_complete() -> None:
    names = [f"network-{index:04d}" for index in range(1000)]
    first = deterministic_partitions(names, "v1-phase1")
    second = deterministic_partitions(list(reversed(names)), "v1-phase1")
    assert first == second
    assert {name for values in first.values() for name in values} == set(names)
    assert sum(len(values) for values in first.values()) == 1000
    assert {key: len(value) for key, value in first.items()} == PARTITION_SIZES


def test_partition_builder_rejects_wrong_size_or_duplicate_names() -> None:
    with pytest.raises(ValueError):
        deterministic_partitions(["too-small"], "v1-phase1")
    duplicates = [f"network-{index:04d}" for index in range(999)] + ["network-0000"]
    with pytest.raises(ValueError):
        deterministic_partitions(duplicates, "v1-phase1")
