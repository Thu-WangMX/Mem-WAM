from __future__ import annotations

import math

import pytest
import torch

from fastwam.memory.control_information_boundary_v2 import (
    CausalEpisodeLogBiasAdapter,
    ControlInformationBoundaryStateV2,
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


def _state(*, threshold: float = 2.0) -> ControlInformationBoundaryStateV2:
    return ControlInformationBoundaryStateV2(
        statistics=_statistics(),
        threshold=threshold,
        drift=0.0,
        decay=1.0,
        detector_stride=4,
        min_units=8,
        max_units=24,
        initial_group_start=0,
        calibration_samples=4,
        adaptation_epsilon=1e-6,
        max_abs_log_bias=math.log(64.0),
    )


def _replay(scale: float):
    state = _state()
    for frame in (0, 4, 8, 12):
        assert state.update(frame=frame, information=None) is None
    values = (1.0, 1.0, 1.0, 1.0, 1.0, 4.0)
    events = []
    for frame, value in zip((16, 20, 24, 28, 32, 36), values):
        event = state.update(frame=frame, information=scale * value)
        if event is not None:
            events.append(event)
    return state, events


def test_episode_log_bias_makes_boundaries_invariant_to_multiplicative_rollout_shift():
    base, base_events = _replay(1.0)
    shifted, shifted_events = _replay(8.0)

    assert base.adaptation_factor == pytest.approx(1.0)
    assert shifted.adaptation_factor == pytest.approx(8.0)
    assert [(event.confirmation_frame, event.reason) for event in base_events] == [
        (36, "counterfactual_control_information")
    ]
    assert [(event.confirmation_frame, event.reason) for event in shifted_events] == [
        (36, "counterfactual_control_information")
    ]
    assert shifted_events[0].adapted_information == pytest.approx(
        base_events[0].adapted_information
    )
    assert shifted.retroactive_boundary_count == 0


def test_current_information_never_changes_the_frozen_causal_bias():
    adapter = CausalEpisodeLogBiasAdapter(
        statistics=_statistics(),
        calibration_samples=4,
        epsilon=1e-6,
        max_abs_log_bias=math.log(64.0),
    )
    for frame in (16, 20, 24, 28):
        assert adapter.observe(frame=frame, information=8.0) is None
    assert adapter.factor == pytest.approx(8.0)

    adapted = adapter.observe(frame=32, information=80.0)

    assert adapted == pytest.approx(10.0)
    assert adapter.factor == pytest.approx(8.0)
    assert adapter.calibration_frames == (16, 20, 24, 28)


def test_adapter_fails_closed_on_unbounded_domain_shift():
    adapter = CausalEpisodeLogBiasAdapter(
        statistics=_statistics(),
        calibration_samples=4,
        epsilon=1e-6,
        max_abs_log_bias=math.log(4.0),
    )
    for frame in (16, 20, 24):
        assert adapter.observe(frame=frame, information=8.0) is None
    with pytest.raises(ValueError, match="outside the locked bound"):
        adapter.observe(frame=28, information=8.0)


def test_calibration_never_emits_a_learned_boundary():
    state = _state(threshold=0.01)
    for frame in (0, 4, 8, 12):
        state.update(frame=frame, information=None)
    for frame, value in zip((16, 20, 24, 28), (1.0, 1.0, 1.0, 100.0)):
        assert state.update(frame=frame, information=value) is None
    assert state.control_information_boundary_count == 0
    assert state.cusum == 0.0
