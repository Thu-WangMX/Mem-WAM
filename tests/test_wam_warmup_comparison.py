from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from fastwam.memory.wam_warmup_comparison import (
    adjacent_cosine_deltas,
    build_strategy_trace,
    calibrate_adjacent_threshold,
    evaluate_decision_rule,
    evaluate_strategy,
)


def _features_from_deltas(deltas: list[float]) -> torch.Tensor:
    angles = [0.0]
    for delta in deltas:
        angles.append(angles[-1] + math.acos(1.0 - delta))
    return torch.tensor([[math.cos(angle), math.sin(angle)] for angle in angles])


def test_adjacent_cosine_and_calibration_use_only_decisions_two_through_eight():
    episode0 = _features_from_deltas([1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.5])
    episode1 = _features_from_deltas([1.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 1.5])

    deltas = adjacent_cosine_deltas(episode0)
    calibration = calibrate_adjacent_threshold({0: episode0, 1: episode1}, [0, 1], gamma=1.0)

    assert deltas[0] is None
    assert deltas[1:] == pytest.approx([1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.5])
    expected_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7] + [0.2] * 7
    center = float(np.median(expected_values))
    mad = float(np.median(np.abs(np.asarray(expected_values) - center)))
    assert calibration["sample_count"] == 14
    assert calibration["eligible_decisions"] == [2, 3, 4, 5, 6, 7, 8]
    assert calibration["median"] == pytest.approx(center)
    assert calibration["mad"] == pytest.approx(mad, abs=1.0e-6)
    assert calibration["threshold"] == pytest.approx(center + 1.4826 * mad, abs=1.0e-6)


def test_adjacent_strategy_suppresses_startup_and_segments_at_confirmation():
    features = _features_from_deltas([1.9, 0.1, 0.2, 1.2, 0.1, 0.8, 0.1, 0.1])

    trace = build_strategy_trace(
        features,
        strategy="adjacent_cosine",
        adjacent_threshold=0.7,
        nms_distance=2,
        min_segment=2,
        max_segment=8,
    )

    assert 1 not in trace["peaks"]
    assert trace["peaks"] == [4, 6]
    assert trace["peak_confirmed_at"] == [5, 7]
    assert trace["boundaries"] == [0, 5, 7, 9]
    assert trace["retroactive_boundary_count"] == 0


def _actions(event_frames: list[int]) -> np.ndarray:
    values = np.zeros((160, 14), dtype=np.float32)
    values[:, 6] = 1.0
    values[:, 13] = 1.0
    state = 1.0
    for index, frame in enumerate(event_frames):
        state = 0.0 if state == 1.0 else 1.0
        values[frame:, 13] = state
    return values


def test_metrics_and_predeclared_decision_rule_are_literal():
    adjacent_traces = {
        40: {"peak_confirmed_at": [4, 7], "reasons": {"0": "start", "4": "surprise_confirmed", "7": "surprise_confirmed", "10": "end"}, "retroactive_boundary_count": 0},
        41: {"peak_confirmed_at": [4, 8], "reasons": {"0": "start", "4": "surprise_confirmed", "8": "surprise_confirmed", "10": "end"}, "retroactive_boundary_count": 0},
    }
    actions = {40: _actions([48, 112]), 41: _actions([48, 112])}

    metrics = evaluate_strategy(adjacent_traces, actions, stride=16, early_end=8, tolerance=1.0)
    verdict = evaluate_decision_rule(
        adjacent=metrics,
        current={"all": {"f1": 0.8}, "early": {"recall": 0.4}},
        short_history={"early": {"recall": 0.5}},
    )

    assert metrics["all"]["counts"] == {"confirmations": 4, "events": 4, "matched": 4}
    assert metrics["early"]["recall"] == 1.0
    assert metrics["forced_boundary_count"] == 0
    assert metrics["retroactive_boundary_count"] == 0
    assert verdict["clauses"] == {
        "early_recall_strictly_higher": True,
        "precision_at_least_0_60": True,
        "f1_within_0_05_of_current": True,
        "zero_retroactive_boundaries": True,
    }
    assert verdict["advance_to_training_boundary_experiment"] is True
