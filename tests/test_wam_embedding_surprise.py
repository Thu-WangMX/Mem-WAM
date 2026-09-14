from __future__ import annotations

import math

import pytest
import torch

from fastwam.memory.wam_embedding_surprise import (
    confirm_causal_peaks,
    score_causal_embeddings,
    segment_from_peaks,
    validate_causal_trace,
)


def test_future_features_cannot_change_prefix_scores_or_thresholds():
    features = torch.eye(12, dtype=torch.float32)
    changed = features.clone()
    changed[10:] = 1000.0

    left = score_causal_embeddings(features)
    right = score_causal_embeddings(changed)

    assert left["scores"][:10] == right["scores"][:10]
    assert left["thresholds"][:10] == right["thresholds"][:10]
    assert left["score_source_end"][9] == 8
    assert left["threshold_source_end"][9] == 8


def test_constant_features_have_finite_zero_surprise_after_warmup():
    trace = score_causal_embeddings(torch.ones(9, 5))

    assert trace["scores"][:4] == [None, None, None, None]
    assert trace["scores"][4:] == [0.0] * 5
    assert all(
        value is None or math.isfinite(value)
        for value in trace["thresholds"]
    )


def test_peak_is_confirmed_one_decision_later():
    scores = [None, None, None, None, 1.0, 4.0, 1.0]
    thresholds = [None, None, None, None, 2.0, 2.0, 2.0]

    assert confirm_causal_peaks(scores, thresholds) == (5,)


def test_nms_keeps_larger_peak_then_earlier_peak_on_tie():
    scores = [0.0, 4.0, 0.0, 5.0, 0.0, 5.0, 0.0]
    thresholds = [0.0] * len(scores)

    assert confirm_causal_peaks(scores, thresholds, nms_distance=3) == (3,)


@pytest.mark.parametrize(
    ("decision_count", "peaks", "expected_boundaries", "expected_lengths"),
    [
        (2, (), [0, 2], [2]),
        (9, (), [0, 7, 9], [7, 2]),
        (12, (1, 5), [0, 2, 5, 12], [2, 3, 7]),
        (22, (5, 13), [0, 5, 13, 20, 22], [5, 8, 7, 2]),
    ],
)
def test_segments_form_expected_bounded_partition(
    decision_count, peaks, expected_boundaries, expected_lengths
):
    segmentation = segment_from_peaks(decision_count, peaks)

    assert segmentation["boundaries"] == expected_boundaries
    assert segmentation["segment_lengths"] == expected_lengths
    assert sum(segmentation["segment_lengths"]) == decision_count


def test_terminal_length_one_is_avoided_when_boundary_can_shift():
    segmentation = segment_from_peaks(22, (5, 13))
    payload = {
        "decision_count": 22,
        "scores": [None] * 22,
        "thresholds": [None] * 22,
        "score_source_end": [None] * 22,
        "threshold_source_end": [None] * 22,
        "peaks": [5, 13],
        **segmentation,
    }

    validate_causal_trace(payload)
    assert all(2 <= length <= 8 for length in payload["segment_lengths"])


def test_validation_rejects_a_future_dependent_score():
    payload = {
        "decision_count": 4,
        "scores": [None, None, None, 1.0],
        "thresholds": [None] * 4,
        "score_source_end": [None, None, None, 3],
        "threshold_source_end": [None] * 4,
        "peaks": [],
        "boundaries": [0, 4],
        "reasons": {"0": "start", "4": "end"},
        "segment_lengths": [4],
    }

    with pytest.raises(ValueError, match="future-dependent score"):
        validate_causal_trace(payload)
