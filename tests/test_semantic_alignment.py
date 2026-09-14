from __future__ import annotations

import numpy as np
import pytest

from fastwam.memory.semantic_alignment import (
    analyze_episode_alignment,
    extract_gripper_events,
    match_confirmations,
)


def _literal_actions() -> np.ndarray:
    actions = np.zeros((348, 14), dtype=np.float32)
    actions[:, 6] = 1.0
    actions[:, 13] = 1.0
    actions[54:, 13] = 0.0
    actions[122:, 13] = 1.0
    actions[176:, 6] = 0.0
    actions[267:, 13] = 0.0
    actions[334:, 13] = 1.0
    return actions


def test_extracts_literal_gripper_crossings():
    events = extract_gripper_events(_literal_actions(), stride=16, threshold=0.5)

    assert events == [
        {"raw_frame": 54, "decision": 3.375, "gripper_dim": 13, "kind": "gripper_close"},
        {"raw_frame": 122, "decision": 7.625, "gripper_dim": 13, "kind": "gripper_open"},
        {"raw_frame": 176, "decision": 11.0, "gripper_dim": 6, "kind": "gripper_close"},
        {"raw_frame": 267, "decision": 16.6875, "gripper_dim": 13, "kind": "gripper_close"},
        {"raw_frame": 334, "decision": 20.875, "gripper_dim": 13, "kind": "gripper_open"},
    ]


def test_matching_is_one_to_one_and_reports_literal_metrics():
    events = extract_gripper_events(_literal_actions(), stride=16, threshold=0.5)

    report = match_confirmations([11, 14, 17, 20], events, tolerance=1.0)

    assert report["matches"] == [
        {"confirmation": 11.0, "event_index": 2, "event_decision": 11.0, "signed_error": 0.0, "absolute_error": 0.0},
        {"confirmation": 17.0, "event_index": 3, "event_decision": 16.6875, "signed_error": 0.3125, "absolute_error": 0.3125},
        {"confirmation": 20.0, "event_index": 4, "event_decision": 20.875, "signed_error": -0.875, "absolute_error": 0.875},
    ]
    assert report["unmatched_confirmations"] == [14.0]
    assert report["missed_event_indices"] == [0, 1]
    assert report["counts"] == {"confirmations": 4, "events": 5, "matched": 3}
    assert report["precision"] == 0.75
    assert report["recall"] == 0.6
    assert report["f1"] == pytest.approx(2.0 / 3.0)
    assert report["mean_absolute_error"] == pytest.approx(0.3958333333333333)


def test_alignment_exposes_retroactive_boundaries_and_online_partition():
    trace = {
        "decision_count": 18,
        "decision_frame_indices": list(range(0, 18 * 16, 16)),
        "peaks": [10, 13],
        "peak_confirmed_at": [11, 14],
        "boundaries": [0, 8, 10, 13, 18],
        "reasons": {"0": "start", "8": "max_length", "10": "surprise", "13": "surprise", "18": "end"},
        "segment_lengths": [8, 2, 3, 5],
    }

    report = analyze_episode_alignment(
        trace, _literal_actions(), stride=16, warmup_end=8, tolerance=1.0
    )

    assert report["retroactive_boundaries"] == [
        {"peak_boundary": 10, "available_at": 11, "lag": 1},
        {"peak_boundary": 13, "available_at": 14, "lag": 1},
    ]
    assert report["forced_boundaries"] == [8]
    assert report["online_counterfactual"]["boundaries"] == [0, 8, 11, 14, 18]
    assert report["online_counterfactual"]["segment_lengths"] == [8, 3, 3, 4]
    assert report["primary_metrics"]["counts"] == {
        "confirmations": 2,
        "events": 5,
        "matched": 1,
    }
    assert report["post_warmup_metrics"]["counts"] == {
        "confirmations": 2,
        "events": 3,
        "matched": 1,
    }
    assert len(report["joint_motion_by_decision"]) == 18
