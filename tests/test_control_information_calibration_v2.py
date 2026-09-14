from __future__ import annotations

import json

import pytest

from scripts.calibrate_putback_control_information_selector_v2 import (
    MAX_UNITS,
    MIN_UNITS,
    REQUIRED_DEPENDENCY_KEYS_V2,
    THRESHOLDS,
    lock_candidate_v2,
    locked_split,
)


def _candidate() -> dict:
    return {
        "selector_config": {
            "threshold": 11.0,
            "drift": 1.0,
            "decay": 0.8,
            "detector_stride": 4,
            "min_units": 8,
            "max_units": 24,
            "initial_group_start": 0,
            "calibration_samples": 4,
            "adaptation_epsilon": 1e-6,
            "max_abs_log_bias": 4.1588830833596715,
        },
        "metrics": {
            "group_count_ratio": 1.0,
            "learned_boundary_fraction": 0.6,
            "captured_excess_information": 0.7,
            "forced_boundary_fraction": 0.3,
            "p95_group_length": 6.0,
        },
    }


def test_v2_grid_only_searches_the_dynamic_k8_memory_regime():
    assert THRESHOLDS == (8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0)
    assert MIN_UNITS == (8,)
    assert MAX_UNITS == (24,)


def test_v2_split_preserves_unopened_heldout_episodes():
    assert locked_split() == {
        "trace_episodes": list(range(40)),
        "statistics_episodes": list(range(30)),
        "calibration_episodes": list(range(30, 40)),
        "heldout_episodes": list(range(40, 50)),
    }


def test_v2_lock_hashes_base_score_and_episode_adapter_separately(tmp_path):
    assert "base_boundary_source" in REQUIRED_DEPENDENCY_KEYS_V2
    dependencies = {}
    for index, key in enumerate(sorted(REQUIRED_DEPENDENCY_KEYS_V2)):
        path = tmp_path / f"{key}.bin"
        path.write_bytes(f"v2-dependency-{index}".encode())
        dependencies[key] = path
    target = tmp_path / "locked_candidate_v2.json"

    payload = lock_candidate_v2(
        target, candidate=_candidate(), dependency_paths=dependencies
    )

    assert payload["schema_version"] == "putback_locked_control_information_candidate_v2"
    assert payload["method"] == "episode_calibrated_counterfactual_control_information"
    assert set(payload["hashes"]) == REQUIRED_DEPENDENCY_KEYS_V2
    assert json.loads(target.read_text()) == payload
    with pytest.raises(FileExistsError):
        lock_candidate_v2(
            target, candidate=_candidate(), dependency_paths=dependencies
        )
