from __future__ import annotations

from dataclasses import asdict

import pytest
import torch

from fastwam.memory.control_information_boundary_v3 import (
    ControlInformationBoundaryStateV3,
    replay_control_information_trace_v3,
)


def _statistics() -> dict:
    return {
        "schema_version": "putback_control_information_statistics_v1",
        "episodes": list(range(30)),
        "phases": [0, 4, 8, 12],
        "depth_cap": 4,
        "median": torch.ones(4, 4),
        "mad_scale": torch.ones(4, 4),
    }


def _config(*, threshold: float = 1.0) -> dict:
    return {
        "threshold": threshold,
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


def _run(information_by_frame, *, last_frame=132, threshold=1.0):
    return replay_control_information_trace_v3(
        detector_frames=list(range(0, last_frame + 1, 4)),
        information_by_frame=information_by_frame,
        statistics=_statistics(),
        selector_config=_config(threshold=threshold),
    )


def test_segment_baseline_removes_persistent_residual_after_one_change():
    information = {
        frame: (8.0 if frame <= 28 else 16.0)
        for frame in range(16, 133, 4)
    }

    events = _run(information)

    learned = [
        event.confirmation_frame
        for event in events
        if event.reason == "segment_relative_control_information"
    ]
    assert learned == [36]
    assert all(event.confirmation_frame not in {68, 100} for event in events)
    assert any(
        event.confirmation_frame == 132 and event.reason == "forced_maximum"
        for event in events
    )


def test_two_consecutive_innovations_can_detect_a_later_relative_change():
    information = {}
    for frame in range(16, 101, 4):
        if frame <= 28:
            information[frame] = 8.0
        elif frame < 80:
            information[frame] = 32.0
        else:
            information[frame] = 64.0

    events = _run(information, last_frame=100, threshold=3.0)

    learned = [
        event.confirmation_frame
        for event in events
        if event.reason == "segment_relative_control_information"
    ]
    assert learned[:2] == [36, 84]


def test_v3_offline_replay_and_online_state_are_identical():
    frames = list(range(0, 101, 4))
    information = {
        frame: (8.0 if frame <= 28 else 16.0)
        for frame in frames
        if frame >= 16
    }
    online_state = ControlInformationBoundaryStateV3(
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

    offline = replay_control_information_trace_v3(
        detector_frames=frames,
        information_by_frame=information,
        statistics=_statistics(),
        selector_config=_config(),
    )

    assert [asdict(event) for event in offline] == [
        asdict(event) for event in online
    ]
    assert online_state.episode_adaptation_factor == pytest.approx(8.0, rel=1e-5)
    assert online_state.retroactive_boundary_count == 0


def test_v3_exposes_only_post_calibration_segment_innovation():
    state = ControlInformationBoundaryStateV3(
        statistics=_statistics(), **_config()
    )
    for frame in range(0, 33, 4):
        state.update(
            frame=frame,
            information=None if frame < 16 else (8.0 if frame <= 28 else 16.0),
        )
        if frame <= 28:
            assert state.last_segment_relative_innovation is None
    assert state.last_segment_relative_innovation == pytest.approx(1.0, rel=1e-5)
