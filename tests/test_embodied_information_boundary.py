from __future__ import annotations

import pytest
import torch

from fastwam.memory.embodied_information_boundary import (
    EmbodiedInformationBoundaryState,
    fit_contextual_residual_statistics,
    contextual_residual_z,
)


def test_contextual_stats_remove_phase_and_history_depth_warmup_bias():
    bank = {}
    for episode in range(30):
        rows = []
        frames = []
        for frame in range(16, 80, 4):
            phase = frame % 16
            depth = min((frame - phase) // 16, 4)
            rows.append(torch.tensor([phase + depth + episode * .1, 2 * depth + episode * .2]))
            frames.append(frame)
        bank[episode] = {"frame_indices": torch.tensor(frames), "residuals": torch.stack(rows)}
    stats = fit_contextual_residual_statistics(bank, episodes=range(30), depth_cap=4)
    assert stats["episodes"] == list(range(30))
    assert stats["median"].shape == stats["mad_scale"].shape == (4, 4, 2)
    z = contextual_residual_z(
        torch.tensor([20.0, 20.0]), frame=16, stats=stats
    )
    assert z.shape == (2,)
    assert torch.all(z >= 0)
    with pytest.raises(ValueError, match="0-29"):
        fit_contextual_residual_statistics(bank, episodes=range(1, 31), depth_cap=4)


def _state(**kwargs):
    base = dict(
        information_budget=6.0, top_k=2, clip_z=8.0,
        detector_stride=4, min_information_units=2, max_units=24,
        initial_group_start=0, gripper_dimensions=(6, 13), gripper_threshold=.5,
    )
    base.update(kwargs)
    return EmbodiedInformationBoundaryState(**base)


def test_four_warmups_then_gripper_anchor_has_priority_over_information():
    state = _state(information_budget=1.0)
    proprio = torch.zeros(14)
    for frame in (0, 4, 8, 12):
        assert state.update(frame=frame, proprio=proprio, standardized_residual=None) is None
    changed = proprio.clone(); changed[13] = 1
    event = state.update(
        frame=16, proprio=changed, standardized_residual=torch.full((4,), 8.0)
    )
    assert event is not None
    assert event.reason == "embodied_gripper_transition"
    assert (event.group_start, event.group_end) == (0, 16)
    assert event.confirmation_frame == 16
    assert event.changed_grippers == (13,)
    assert state.retroactive_boundary_count == 0


def test_wam_budget_dynamic_lengths_forced_max_and_terminal_tail():
    state = _state(information_budget=4.0, max_units=5)
    proprio = torch.zeros(14)
    events = []
    for frame in (0, 4, 8, 12):
        state.update(frame=frame, proprio=proprio, standardized_residual=None)
    for frame, value in ((16, 4.0), (20, 0.0), (24, 0.0), (28, 0.0),
                         (32, 0.0), (36, 0.0), (40, 0.0), (44, 0.0), (48, 0.0)):
        event = state.update(
            frame=frame, proprio=proprio,
            standardized_residual=torch.full((4,), value),
        )
        if event: events.append(event)
    tail = state.finalize(frame=52)
    if tail: events.append(tail)
    assert [event.reason for event in events] == [
        "wam_information_budget", "forced_maximum", "terminal_tail"
    ]
    assert [(e.group_start, e.group_end) for e in events] == [(0, 16), (16, 36), (36, 52)]
    assert state.wam_information_boundary_count == 1
    assert state.forced_maximum_boundary_count == 1
    assert state.terminal_tail_count == 1
    assert len({e.group_end - e.group_start for e in events}) >= 2


def test_prefix_invariance_and_no_observation_gaps():
    trace = [(frame, torch.zeros(14), torch.ones(4)) for frame in range(0, 52, 4)]
    def replay(count):
        state = _state(information_budget=5.0)
        events=[]
        for frame, proprio, residual in trace[:count]:
            event=state.update(
                frame=frame, proprio=proprio,
                standardized_residual=None if frame < 16 else residual,
            )
            if event: events.append(event)
        return events
    full=replay(len(trace))
    for count in range(1,len(trace)+1):
        assert replay(count)==[event for event in full if event.confirmation_frame <= trace[count-1][0]]
    state=_state()
    state.update(frame=0,proprio=torch.zeros(14),standardized_residual=None)
    with pytest.raises(ValueError,match="every four frames"):
        state.update(frame=8,proprio=torch.zeros(14),standardized_residual=None)
