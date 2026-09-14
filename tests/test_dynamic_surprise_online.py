from __future__ import annotations

import pytest

from fastwam.memory.dynamic_surprise import OnlineSurpriseSegmenter


def _commit_scores(segmenter: OnlineSurpriseSegmenter, scores: list[float]):
    decisions = []
    for score in scores:
        decision = segmenter.preview(score)
        segmenter.commit(decision)
        decisions.append(decision)
    return decisions


def test_online_segmenter_closes_before_trigger_observation():
    segmenter = OnlineSurpriseSegmenter()

    decisions = _commit_scores(segmenter, [1.0, 1.0, 1.0, 3.0])

    assert decisions[-1].arrival == 4
    assert decisions[-1].threshold == pytest.approx(1.0)
    assert decisions[-1].close_range == (0, 4)
    assert decisions[-1].reason == "surprise"
    assert segmenter.open_start == 4


def test_online_segmenter_forces_max_length_at_arrival_eight():
    segmenter = OnlineSurpriseSegmenter()

    decisions = _commit_scores(segmenter, [1.0] * 8)

    assert all(decision.close_range is None for decision in decisions[:-1])
    assert decisions[-1].close_range == (0, 8)
    assert decisions[-1].reason == "max_length"


def test_online_segmenter_threshold_window_does_not_reset_at_boundary():
    segmenter = OnlineSurpriseSegmenter()
    _commit_scores(segmenter, [1.0, 1.0, 1.0, 3.0])

    after_boundary = segmenter.preview(2.0)

    # The prior window is still [1, 1, 1, 3]: mean=1.5, pop-std=sqrt(0.75).
    assert after_boundary.threshold == pytest.approx(2.799038105676658)
    assert after_boundary.close_range is None


def test_preview_is_immutable_until_exact_decision_is_committed():
    segmenter = OnlineSurpriseSegmenter()

    preview = segmenter.preview(1.25)

    assert segmenter.scores == ()
    assert segmenter.open_start == 0
    assert segmenter.next_arrival == 1
    segmenter.commit(preview)
    assert segmenter.scores == (1.25,)
    assert segmenter.next_arrival == 2
    with pytest.raises(ValueError, match="current preview"):
        segmenter.commit(preview)
