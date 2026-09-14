from __future__ import annotations

import json

import pytest

from scripts.calibrate_putback_control_information_selector_v3 import (
    DECAYS,
    DRIFTS,
    REQUIRED_DEPENDENCY_KEYS_V3,
    SEGMENT_SCALE_FLOORS,
    THRESHOLDS,
    lock_candidate_v3,
    locked_split,
)


def _candidate() -> dict:
    return {
        "selector_config": {
            "threshold": 4.0,
            "drift": 0.25,
            "decay": 0.8,
            "detector_stride": 4,
            "min_units": 8,
            "max_units": 24,
            "initial_group_start": 0,
            "episode_calibration_samples": 4,
            "segment_calibration_samples": 4,
            "adaptation_epsilon": 1e-6,
            "max_abs_log_bias": 4.0,
            "segment_scale_floor": 1.0,
            "minimum_consecutive_evidence": 2,
        },
        "metrics": {
            "group_count_ratio": 1.0,
            "learned_boundary_fraction": 0.7,
            "captured_excess_information": 0.6,
            "forced_boundary_fraction": 0.2,
            "p95_group_length": 6.0,
        },
    }


def test_v3_search_space_is_locked_to_segment_relative_k8_regime():
    assert THRESHOLDS == (2.0, 4.0, 6.0, 8.0)
    assert DRIFTS == (0.25, 0.5, 1.0)
    assert DECAYS == (0.8, 0.9)
    assert SEGMENT_SCALE_FLOORS == (0.5, 1.0, 2.0)


def test_v3_split_does_not_select_on_heldout_episodes():
    assert locked_split() == {
        "trace_episodes": list(range(40)),
        "statistics_episodes": list(range(30)),
        "calibration_episodes": list(range(30, 40)),
        "heldout_episodes": list(range(40, 50)),
    }


def test_v3_candidate_locks_all_reused_and_new_boundary_sources(tmp_path):
    assert {
        "base_boundary_source",
        "episode_adapter_source",
        "segment_boundary_source",
    } <= REQUIRED_DEPENDENCY_KEYS_V3
    dependencies = {}
    for index, key in enumerate(sorted(REQUIRED_DEPENDENCY_KEYS_V3)):
        path = tmp_path / f"{key}.bin"
        path.write_bytes(f"v3-dependency-{index}".encode())
        dependencies[key] = path
    target = tmp_path / "locked_candidate_v3.json"

    payload = lock_candidate_v3(
        target, candidate=_candidate(), dependency_paths=dependencies
    )

    assert payload["schema_version"] == "putback_locked_control_information_candidate_v3"
    assert set(payload["hashes"]) == REQUIRED_DEPENDENCY_KEYS_V3
    assert json.loads(target.read_text()) == payload
    with pytest.raises(FileExistsError):
        lock_candidate_v3(
            target, candidate=_candidate(), dependency_paths=dependencies
        )
