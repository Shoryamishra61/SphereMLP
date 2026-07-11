from __future__ import annotations

import csv

from experiments.compare_estimators import paired_comparison
from experiments.deduplicate_results import canonicalize


def test_canonicalize_preserves_first_occurrence(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text(
        "mlp_id,estimator_name,adjusted_score\n"
        "a,scalar,1.0\n"
        "a,scalar,0.1\n"
        "a,covariance,0.2\n",
        encoding="utf-8",
    )
    output = tmp_path / "canonical.csv"
    retained, removed = canonicalize(source, output)
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert (retained, removed) == (2, 1)
    assert rows == [
        {"mlp_id": "a", "estimator_name": "scalar", "adjusted_score": "1.0"},
        {"mlp_id": "a", "estimator_name": "covariance", "adjusted_score": "0.2"},
    ]


def test_paired_comparison_is_order_independent_and_seeded():
    rows = [
        {"mlp_id": "a", "estimator_name": "base", "status": "ok", "adjusted_score": "2", "raw_final_mse": "4", "compute_ratio": "0.2", "wall_ms": "20"},
        {"mlp_id": "a", "estimator_name": "new", "status": "ok", "adjusted_score": "1", "raw_final_mse": "3", "compute_ratio": "0.3", "wall_ms": "30"},
        {"mlp_id": "b", "estimator_name": "base", "status": "ok", "adjusted_score": "4", "raw_final_mse": "8", "compute_ratio": "0.2", "wall_ms": "20"},
        {"mlp_id": "b", "estimator_name": "new", "status": "ok", "adjusted_score": "2", "raw_final_mse": "6", "compute_ratio": "0.3", "wall_ms": "30"},
    ]
    result = paired_comparison(rows[::-1], "base", "new", seed=7, resamples=100)
    assert result["paired_mlp_count"] == 2
    assert result["mean_paired_adjusted_score_delta"] == -1.5
    assert result["candidate_win_fraction"] == 1.0
