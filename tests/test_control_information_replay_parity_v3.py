from __future__ import annotations

import torch

from scripts.freeze_putback_control_information_manifest_v3 import freeze_episode_v3


def _statistics() -> dict:
    return {
        "schema_version": "putback_control_information_statistics_v1",
        "episodes": list(range(30)),
        "phases": [0, 4, 8, 12],
        "depth_cap": 4,
        "median": torch.ones(4, 4),
        "mad_scale": torch.ones(4, 4),
    }


def _config() -> dict:
    return {
        "threshold": 1.0,
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
    }


def test_v3_freezer_uses_shared_replay_and_records_segment_baselines():
    frames = torch.tensor(list(range(16, 133, 4)))
    information = torch.tensor(
        [8.0 if int(frame) <= 28 else 16.0 for frame in frames]
    )

    payload = freeze_episode_v3(
        episode=3,
        trace={"frame_indices": frames, "information": information},
        episode_length=136,
        statistics=_statistics(),
        selector_config=_config(),
    )

    assert payload["episode"] == 3
    assert payload["retroactive_boundary_count"] == 0
    assert payload["episode_adaptation_factor"] > 0
    learned = [
        event
        for event in payload["detector_events"]
        if event["reason"] == "segment_relative_control_information"
    ]
    assert [event["confirmation_frame"] for event in learned] == [36]
    assert learned[0]["segment_baseline_scale"] == 1.0
    assert payload["boundaries"][0] == 0
    assert payload["boundaries"][-1] == payload["decision_count"]
