from __future__ import annotations

import pytest
import torch

from fastwam.evaluation.predictive_selector_runtime import PredictiveSelectorRuntime


def _runtime():
    return PredictiveSelectorRuntime(
        selector_config={
            "stats": {"median": [0.0, 0.0], "mad_scale": [1.0, 1.0]},
            "weights": [0.0, 1.0], "high_threshold": 3.0, "low_threshold": 1.0,
            "detector_stride": 4, "min_units": 2, "max_units": 20, "nms_units": 2,
        }
    )


def test_runtime_advances_phases_and_emits_only_at_confirmation_arrival():
    runtime = _runtime()
    phases = []
    for frame, residual in [(0, None), (4, None), (8, None), (12, None),
                            (16, [0, 0]), (20, [0, 0]), (24, [0, 5]), (28, [0, 2])]:
        job = runtime.observe(frame=frame, observation=torch.tensor([frame]))
        phases.append(job.phase)
        event = runtime.complete(frame=frame, residual=None if residual is None else torch.tensor(residual))
    assert phases == [0, 4, 8, 12, 0, 4, 8, 12]
    assert event is not None
    assert event.frame == event.emitted_at_frame == 28
    assert event.peak_frame == 24
    assert event.group_start == 0
    assert runtime.maximum_queue_depth == 1
    assert runtime.retroactive_boundary_count == 0


def test_runtime_rejects_delayed_or_overlapping_detector_jobs():
    runtime = _runtime()
    runtime.observe(frame=0, observation=torch.zeros(1))
    with pytest.raises(RuntimeError, match="pending"):
        runtime.observe(frame=4, observation=torch.zeros(1))
    with pytest.raises(RuntimeError, match="completion frame"):
        runtime.complete(frame=4, residual=None)


def test_each_phase_warms_up_once_before_any_prediction_is_scored():
    runtime = _runtime()
    for frame in (0, 4, 8, 12):
        runtime.observe(frame=frame, observation=torch.zeros(1))
        assert runtime.complete(frame=frame, residual=None) is None
    runtime.observe(frame=16, observation=torch.zeros(1))
    with pytest.raises(ValueError, match="requires residuals"):
        runtime.complete(frame=16, residual=None)
