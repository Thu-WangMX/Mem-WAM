from __future__ import annotations

from fastwam.memory.phase_annotation import TRANSITIONS
from fastwam.memory.phase_boundary_metrics import (
    evaluate_phase_boundaries,
    reliability_gate,
)
from fastwam.memory.predictive_boundary import BoundaryEvent


def _annotation(episode: int, frames=(20, 40, 60, 80, 100)):
    return {
        "episode": episode,
        "transitions": [
            {
                "name": name,
                "frame": frame,
                "ambiguity_start": frame - 4,
                "ambiguity_end": frame + 4,
                "not_observed": False,
            }
            for name, frame in zip(TRANSITIONS, frames)
        ],
    }


def _event(frame: int, start: int, reason="predictive_surprise_confirmed"):
    return BoundaryEvent(
        frame=frame,
        peak_frame=frame - 4,
        reason=reason,
        group_start=start,
        group_end=frame,
        score=4.0,
        selected_streams=(1,),
    )


def test_one_to_one_matching_metrics_and_gate_cover_all_reliability_axes():
    annotations = {30: _annotation(30), 31: _annotation(31)}
    predictions = {
        30: [_event(20, 0), _event(44, 20), _event(60, 44), _event(84, 60), _event(100, 84)],
        31: [_event(24, 0), _event(40, 24), _event(64, 40), _event(80, 64), _event(104, 80)],
    }
    report = evaluate_phase_boundaries(
        predictions,
        annotations,
        tolerance=8,
        baseline_f1=0.80,
        retroactive_boundary_count=0,
        uniform_four_frame_budget=60,
    )

    assert report["micro"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert report["macro"]["f1"] == 1.0
    assert report["timing_mae"] == 2.0
    assert report["transition_recall"] == {name: 1.0 for name in TRANSITIONS}
    assert report["forced_maximum_count"] == 0
    assert report["memory_count"] == 10
    assert report["group_length_histogram"] == {16: 4, 20: 1, 24: 5}
    assert reliability_gate(report) == {
        "precision_at_least_0_60": True,
        "recall_at_least_0_60": True,
        "f1_at_least_0_65": True,
        "all_phase_recalls_nonzero": True,
        "f1_gain_at_least_0_10": True,
        "zero_retroactive": True,
        "within_uniform_four_frame_budget": True,
        "at_least_two_dynamic_lengths": True,
        "pass": True,
    }


def test_duplicate_predictions_cannot_match_one_transition_twice():
    report = evaluate_phase_boundaries(
        {30: [_event(20, 0), _event(24, 20)]},
        {30: _annotation(30, frames=(20, 40, 60, 80, 100))},
        tolerance=8,
        baseline_f1=0.0,
        retroactive_boundary_count=0,
        uniform_four_frame_budget=30,
    )
    assert report["counts"] == {"true_positive": 1, "false_positive": 1, "false_negative": 4}
    assert report["micro"]["precision"] == 0.5
    assert report["transition_recall"][TRANSITIONS[0]] == 1.0
    assert report["transition_recall"][TRANSITIONS[1]] == 0.0
