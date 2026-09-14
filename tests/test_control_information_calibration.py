from __future__ import annotations

import json

import pytest

from scripts.calibrate_putback_control_information_selector import (
    REQUIRED_DEPENDENCY_KEYS,
    THRESHOLDS,
    lock_candidate,
    locked_split,
    select_calibrated_candidate,
)


def _candidate(
    *,
    threshold: float,
    group_count_ratio: float = 1.0,
    learned_fraction: float = 0.6,
    captured: float = 0.7,
    forced_fraction: float = 0.2,
    p95: float = 6.0,
):
    return {
        "selector_config": {
            "threshold": threshold,
            "drift": 0.5,
            "decay": 0.9,
            "detector_stride": 4,
            "min_units": 4,
            "max_units": 24,
            "initial_group_start": 0,
        },
        "metrics": {
            "group_count_ratio": group_count_ratio,
            "learned_boundary_fraction": learned_fraction,
            "captured_excess_information": captured,
            "forced_boundary_fraction": forced_fraction,
            "p95_group_length": p95,
        },
    }


def test_locked_split_never_uses_heldout_for_training_or_calibration():
    split = locked_split()

    assert split == {
        "trace_episodes": list(range(40)),
        "statistics_episodes": list(range(30)),
        "calibration_episodes": list(range(30, 40)),
        "heldout_episodes": list(range(40, 50)),
    }


def test_threshold_grid_reaches_matched_memory_rate_regime():
    assert THRESHOLDS == (2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0)


def test_candidate_selection_applies_rate_and_learned_event_gates():
    candidates = [
        _candidate(threshold=2.0, group_count_ratio=1.2, captured=0.99),
        _candidate(threshold=3.0, learned_fraction=0.49, captured=0.98),
        _candidate(threshold=4.0, captured=0.80, forced_fraction=0.3),
        _candidate(threshold=5.0, captured=0.80, forced_fraction=0.1),
    ]

    selected = select_calibrated_candidate(candidates)

    assert selected["selector_config"]["threshold"] == 5.0
    assert selected["selection_gates"] == {
        "group_count_within_10_percent": True,
        "learned_boundary_fraction_at_least_half": True,
    }


def test_candidate_selection_fails_closed_when_no_configuration_passes():
    with pytest.raises(RuntimeError, match="no selector candidate"):
        select_calibrated_candidate(
            [_candidate(threshold=2.0, group_count_ratio=1.5)]
        )


def test_locked_candidate_hashes_every_available_selector_dependency(tmp_path):
    dependencies = {}
    for index, key in enumerate(sorted(REQUIRED_DEPENDENCY_KEYS)):
        path = tmp_path / f"{key}.bin"
        path.write_bytes(f"dependency-{index}".encode())
        dependencies[key] = path
    target = tmp_path / "locked_candidate.json"

    payload = lock_candidate(
        target,
        candidate=_candidate(threshold=4.0),
        dependency_paths=dependencies,
    )

    assert set(payload["hashes"]) == REQUIRED_DEPENDENCY_KEYS
    assert payload["schema_version"] == "putback_locked_control_information_candidate_v1"
    assert json.loads(target.read_text()) == payload
    with pytest.raises(FileExistsError):
        lock_candidate(
            target,
            candidate=_candidate(threshold=4.0),
            dependency_paths=dependencies,
        )
