from __future__ import annotations

import inspect

import pytest
import torch

from fastwam.memory.control_information_boundary import (
    ControlInformationBoundaryState,
    contextual_information_z,
    fit_control_information_statistics,
)


def _unit_statistics():
    return {
        "schema_version": "putback_control_information_statistics_v1",
        "episodes": list(range(3)),
        "phases": [0, 4, 8, 12],
        "depth_cap": 4,
        "median": torch.zeros(4, 4),
        "mad_scale": torch.ones(4, 4),
    }


def _state(**overrides):
    config = {
        "statistics": _unit_statistics(),
        "threshold": 2.0,
        "drift": 0.5,
        "decay": 1.0,
        "detector_stride": 4,
        "min_units": 4,
        "max_units": 24,
        "initial_group_start": 0,
    }
    config.update(overrides)
    return ControlInformationBoundaryState(**config)


def _warmup(state):
    for frame in (0, 4, 8, 12):
        assert state.update(frame=frame, information=None) is None


def test_cusum_confirms_at_current_frame_without_peak_backdating():
    state = _state()
    _warmup(state)

    event = state.update(frame=16, information=3.0)

    assert event is not None
    assert event.confirmation_frame == event.group_end == 16
    assert event.group_start == 0
    assert event.reason == "counterfactual_control_information"
    assert event.standardized_information == 3.0
    assert event.cusum == 2.5
    assert state.retroactive_boundary_count == 0


def test_update_has_no_proprio_or_gripper_argument():
    parameters = inspect.signature(ControlInformationBoundaryState.update).parameters
    assert "proprio" not in parameters
    assert all("gripper" not in name for name in parameters)


def test_maximum_and_terminal_events_have_separate_reasons():
    maximum = _state(threshold=1e9, max_units=4)
    _warmup(maximum)
    max_event = maximum.update(frame=16, information=0.0)

    terminal = _state(threshold=1e9)
    _warmup(terminal)
    tail_event = terminal.finalize(frame=12)

    assert max_event.reason == "forced_maximum"
    assert tail_event.reason == "terminal_tail"
    assert maximum.forced_maximum_boundary_count == 1
    assert terminal.terminal_tail_count == 1


def test_emission_resets_cusum_and_enforces_new_minimum_segment():
    state = _state()
    _warmup(state)
    assert state.update(frame=16, information=3.0) is not None
    assert state.cusum == 0.0

    for frame in (20, 24, 28):
        assert state.update(frame=frame, information=10.0) is None
    event = state.update(frame=32, information=10.0)
    assert event is not None
    assert (event.group_start, event.group_end) == (16, 32)


def test_state_requires_residual_free_warmup_and_post_warmup_information():
    state = _state()
    with pytest.raises(ValueError, match="warmup"):
        state.update(frame=0, information=1.0)

    state = _state()
    _warmup(state)
    with pytest.raises(ValueError, match="post-warmup"):
        state.update(frame=16, information=None)


def test_contextual_statistics_fit_phase_and_history_depth_without_heldout():
    frames = torch.arange(16, 80, 4)
    scores = {
        episode: {
            "frame_indices": frames,
            "information": torch.arange(len(frames), dtype=torch.float32)
            + float(episode),
        }
        for episode in range(3)
    }

    statistics = fit_control_information_statistics(
        scores, episodes=range(3), depth_cap=4
    )

    assert statistics["median"].shape == (4, 4)
    assert statistics["mad_scale"].shape == (4, 4)
    assert bool((statistics["mad_scale"] > 0).all())
    z = contextual_information_z(100.0, frame=16, statistics=statistics)
    assert z > 0
    with pytest.raises(ValueError, match="0-29"):
        fit_control_information_statistics(
            {30: scores[0]}, episodes=[30], depth_cap=4
        )
