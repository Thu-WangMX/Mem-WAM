from __future__ import annotations

from dataclasses import asdict

import pytest
import torch

from fastwam.memory.control_information_boundary_v2 import (
    ControlInformationBoundaryStateV2,
    replay_control_information_trace_v2,
)
from scripts.freeze_putback_control_information_manifest_v2 import freeze_episode_v2


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
        "threshold": 2.0,
        "drift": 0.0,
        "decay": 1.0,
        "detector_stride": 4,
        "min_units": 8,
        "max_units": 24,
        "initial_group_start": 0,
        "calibration_samples": 4,
        "adaptation_epsilon": 1e-6,
        "max_abs_log_bias": 4.0,
    }


def test_v2_offline_helper_and_online_state_are_identical():
    frames = list(range(0, 68, 4))
    information = {
        frame: (32.0 if frame == 36 else 8.0)
        for frame in frames
        if frame >= 16
    }
    online_state = ControlInformationBoundaryStateV2(
        statistics=_statistics(), **_config()
    )
    online = []
    for frame in frames:
        event = online_state.update(
            frame=frame,
            information=None if frame < 16 else information[frame],
        )
        if event is not None:
            online.append(event)
    tail = online_state.finalize(frame=frames[-1])
    if tail is not None:
        online.append(tail)

    offline = replay_control_information_trace_v2(
        detector_frames=frames,
        information_by_frame=information,
        statistics=_statistics(),
        selector_config=_config(),
    )

    assert [asdict(event) for event in offline] == [
        asdict(event) for event in online
    ]
    assert online_state.adaptation_factor == pytest.approx(8.0, rel=1e-5)


def test_v2_frozen_episode_records_adaptation_and_zero_retroactivity():
    frames = torch.tensor(list(range(16, 68, 4)))
    information = torch.tensor(
        [32.0 if int(frame) == 36 else 8.0 for frame in frames]
    )

    payload = freeze_episode_v2(
        episode=3,
        trace={"frame_indices": frames, "information": information},
        episode_length=68,
        statistics=_statistics(),
        selector_config=_config(),
    )

    assert payload["episode"] == 3
    assert payload["retroactive_boundary_count"] == 0
    assert payload["adaptation_factor"] == pytest.approx(8.0, rel=1e-5)
    assert payload["calibration_frames"] == [16, 20, 24, 28]
    assert payload["boundaries"][0] == 0
    assert payload["boundaries"][-1] == payload["decision_count"]
