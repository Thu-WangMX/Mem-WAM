from __future__ import annotations

import torch

from fastwam.memory.wam_embedding_causal_v2 import build_causal_v2_trace


def test_causal_v2_closes_only_when_peak_is_confirmed():
    features = torch.tensor([0, 1, 0, 1, 0, 10, 0, 0], dtype=torch.float32)[:, None]

    trace = build_causal_v2_trace(
        features,
        statistics_window=4,
        threshold_window=4,
        min_history=2,
        min_threshold_history=2,
        gamma=1.0,
        nms_distance=2,
        min_segment=2,
        max_segment=8,
    )

    assert trace["peaks"] == [5]
    assert trace["peak_confirmed_at"] == [6]
    assert trace["boundaries"] == [0, 6, 8]
    assert trace["reasons"] == {
        "0": "start",
        "6": "surprise_confirmed",
        "8": "end",
    }
    assert trace["segment_lengths"] == [6, 2]
    assert trace["retroactive_boundary_count"] == 0


def test_causal_v2_never_moves_confirmation_backward_to_avoid_terminal_one():
    features = torch.tensor([0, 1, 0, 1, 0, 10, 0], dtype=torch.float32)[:, None]

    trace = build_causal_v2_trace(
        features,
        statistics_window=4,
        threshold_window=4,
        min_history=2,
        min_threshold_history=2,
        gamma=1.0,
        nms_distance=2,
        min_segment=2,
        max_segment=8,
    )

    assert trace["peak_confirmed_at"] == [6]
    assert trace["boundaries"] == [0, 6, 7]
    assert trace["segment_lengths"] == [6, 1]
    assert trace["reasons"]["6"] == "surprise_confirmed"
    assert trace["reasons"]["7"] == "terminal_tail"
    assert trace["retroactive_boundary_count"] == 0
