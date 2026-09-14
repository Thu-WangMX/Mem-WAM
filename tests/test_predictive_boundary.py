from __future__ import annotations

import pytest
import torch

from fastwam.memory.predictive_boundary import (
    PredictiveBoundaryState,
    fit_residual_statistics,
    fuse_residual_score,
    robust_residual_z,
    select_sparse_simplex_weights,
)


def test_residual_statistics_are_training_only_and_reject_collapsed_streams():
    bank = {
        episode: torch.tensor(
            [[episode + step, 2 * episode + step * step] for step in range(4)],
            dtype=torch.float32,
        )
        for episode in range(30)
    }
    stats = fit_residual_statistics(bank, episodes=range(30))
    assert stats["episodes"] == list(range(30))
    assert stats["median"].shape == stats["mad_scale"].shape == (2,)
    assert torch.all(stats["mad_scale"] > 0)

    bank[30] = torch.full((4, 2), 1e9)
    unchanged = fit_residual_statistics(bank, episodes=range(30))
    torch.testing.assert_close(stats["median"], unchanged["median"])
    with pytest.raises(ValueError, match="0-29"):
        fit_residual_statistics(bank, episodes=range(1, 31))
    with pytest.raises(ValueError, match="collapsed"):
        fit_residual_statistics({0: torch.ones(8, 2)}, episodes=[0])


def test_sparse_simplex_weights_are_stable_and_preserve_local_wrist_event():
    # Stream 1 is consistently discriminative in every development episode;
    # stream 0 is a noisy global average and stream 2 is irrelevant.
    event = {
        episode: torch.tensor([[0.1 + episode % 2, 8.0 + 0.1 * episode, 0.2]])
        for episode in range(26, 30)
    }
    background = {
        episode: torch.tensor([[0.5, 0.5 + 0.02 * episode, 0.3]])
        for episode in range(26, 30)
    }
    selected = []
    for held_out in range(26, 30):
        weights = select_sparse_simplex_weights(
            event,
            background,
            episodes=[episode for episode in range(26, 30) if episode != held_out],
            top_k=1,
        )
        assert torch.all(weights >= 0)
        torch.testing.assert_close(weights.sum(), torch.tensor(1.0))
        selected.append(tuple(torch.nonzero(weights).flatten().tolist()))
    assert selected == [(1,), (1,), (1,), (1,)]

    stats = {"median": torch.zeros(3), "mad_scale": torch.ones(3)}
    score, active = fuse_residual_score(
        torch.tensor([0.0, 6.0, 0.0]), stats=stats, weights=torch.tensor([0.0, 1.0, 0.0])
    )
    assert score == pytest.approx(6.0)
    assert active == (1,)
    assert robust_residual_z(torch.tensor([0.0, 6.0, 0.0]), stats).mean() == 2.0


def _state(**overrides):
    kwargs = dict(
        stats={"median": torch.zeros(2), "mad_scale": torch.ones(2)},
        weights=torch.tensor([0.25, 0.75]),
        high_threshold=3.0,
        low_threshold=1.0,
        detector_stride=4,
        min_units=2,
        max_units=8,
        nms_units=2,
    )
    kwargs.update(overrides)
    return PredictiveBoundaryState(**kwargs)


def test_online_confirmation_emits_now_never_backdates_to_peak():
    state = _state(max_units=20)
    assert state.update(frame=20, residuals=torch.tensor([0.0, 0.0])) is None
    assert state.update(frame=24, residuals=torch.tensor([0.0, 5.0])) is None
    event = state.update(frame=28, residuals=torch.tensor([0.0, 2.0]))

    assert event is not None
    assert event.frame == 28
    assert event.peak_frame == 24
    assert event.reason == "predictive_surprise_confirmed"
    assert event.group_start == 16
    assert event.group_end == 28
    assert state.retroactive_boundary_count == 0


def test_hysteresis_nms_forced_max_and_terminal_tail_are_separate():
    state = _state(max_units=4)
    trace = [
        (4, [0.0, 0.0]),
        (8, [0.0, 5.0]),
        (12, [0.0, 2.0]),  # surprise boundary
        (16, [0.0, 5.0]),
        (20, [0.0, 2.0]),  # disarmed, cannot double fire
        (24, [0.0, 0.0]),  # rearm below low
        (28, [0.0, 0.0]),  # forced max: 12 -> 28
        (32, [0.0, 0.0]),
    ]
    events = []
    for frame, residuals in trace:
        event = state.update(frame=frame, residuals=torch.tensor(residuals))
        if event is not None:
            events.append(event)
    tail = state.finalize(frame=36)
    assert tail is not None
    events.append(tail)

    assert [event.reason for event in events] == [
        "predictive_surprise_confirmed",
        "forced_maximum",
        "terminal_tail",
    ]
    assert state.surprise_boundary_count == 1
    assert state.forced_maximum_boundary_count == 1
    assert state.terminal_tail_count == 1
    assert [(event.group_start, event.group_end) for event in events] == [
        (0, 12),
        (12, 28),
        (28, 36),
    ]
    assert all(left.group_end == right.group_start for left, right in zip(events, events[1:]))
    assert all(event.group_end > event.group_start for event in events)


def test_dynamic_nonforced_groups_can_have_multiple_lengths():
    state = _state(max_units=20)
    values = {
        4: 0.0,
        8: 5.0,
        12: 2.0,  # length 12
        16: 0.0,
        20: 0.0,
        24: 0.0,
        28: 5.0,
        32: 2.0,  # length 20
    }
    events = []
    for frame, value in values.items():
        event = state.update(frame=frame, residuals=torch.tensor([0.0, value]))
        if event is not None:
            events.append(event)
    assert [event.group_end - event.group_start for event in events] == [12, 20]
    assert all(event.reason == "predictive_surprise_confirmed" for event in events)


def test_explicit_episode_start_covers_four_phase_warmup_prefix():
    state = _state(max_units=20, initial_group_start=0)
    assert state.update(frame=16, residuals=torch.tensor([0.0, 5.0])) is None
    event = state.update(frame=20, residuals=torch.tensor([0.0, 2.0]))
    assert event is not None
    assert (event.group_start, event.group_end) == (0, 20)
