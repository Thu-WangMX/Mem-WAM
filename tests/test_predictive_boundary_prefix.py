from __future__ import annotations

import torch

from fastwam.memory.predictive_boundary import PredictiveBoundaryState


TRACE = [0.0, 0.5, 4.0, 5.0, 2.0, 0.2, 0.0, 4.5, 2.5, 0.0, 0.0, 0.0]


def _replay(prefix: int):
    state = PredictiveBoundaryState(
        stats={"median": torch.zeros(2), "mad_scale": torch.ones(2)},
        weights=torch.tensor([0.0, 1.0]),
        high_threshold=3.0,
        low_threshold=1.0,
        detector_stride=4,
        min_units=2,
        max_units=8,
        nms_units=2,
    )
    events = []
    for index, value in enumerate(TRACE[:prefix], start=1):
        event = state.update(
            frame=4 * index,
            residuals=torch.tensor([1000.0, value]),
        )
        if event is not None:
            events.append(event)
    return events


def test_every_online_prefix_is_exactly_invariant_to_future_frames():
    full = _replay(len(TRACE))
    for prefix in range(1, len(TRACE) + 1):
        prefix_events = _replay(prefix)
        expected = [event for event in full if event.frame <= prefix * 4]
        assert prefix_events == expected

