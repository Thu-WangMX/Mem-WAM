from __future__ import annotations

from dataclasses import asdict

import torch

from fastwam.memory.control_information_boundary import (
    ControlInformationBoundaryState,
    replay_control_information_trace,
)
from fastwam.memory.planning_aligned_manifest import align_detector_events


def _statistics():
    return {
        "schema_version": "putback_control_information_statistics_v1",
        "episodes": list(range(30)),
        "phases": [0, 4, 8, 12],
        "depth_cap": 4,
        "median": torch.zeros(4, 4),
        "mad_scale": torch.ones(4, 4),
    }


def _config():
    return {
        "threshold": 3.0,
        "drift": 0.5,
        "decay": 0.9,
        "detector_stride": 4,
        "min_units": 4,
        "max_units": 8,
        "initial_group_start": 0,
    }


def test_offline_helper_and_online_state_replay_are_identical():
    frames = list(range(0, 68, 4))
    information = {
        frame: (4.0 if frame in {20, 48} else 0.0)
        for frame in frames
        if frame >= 16
    }
    online_state = ControlInformationBoundaryState(
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

    offline = replay_control_information_trace(
        detector_frames=frames,
        information_by_frame=information,
        statistics=_statistics(),
        selector_config=_config(),
    )

    assert [asdict(event) for event in offline] == [asdict(event) for event in online]
    online_aligned = align_detector_events(
        [asdict(event) for event in online], episode_frames=68
    )
    offline_aligned = align_detector_events(
        [asdict(event) for event in offline], episode_frames=68
    )
    assert online_aligned == offline_aligned
