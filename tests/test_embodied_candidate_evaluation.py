from __future__ import annotations

from scripts.evaluate_putback_embodied_information_candidate import match_event_frames


def test_proxy_matching_is_one_to_one_and_reports_precision_recall_f1():
    report = match_event_frames([20, 24, 60], [20, 64], tolerance=4)
    assert report["counts"] == {"true_positive": 2, "false_positive": 1, "false_negative": 0}
    assert report["precision"] == 2 / 3
    assert report["recall"] == 1.0
    assert report["f1"] == 0.8

