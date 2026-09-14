"""Online-executable WAM embedding surprise traces."""

from __future__ import annotations

from typing import Any

import torch

from fastwam.memory.wam_embedding_surprise import (
    confirm_causal_peaks,
    score_causal_embeddings,
)


CAUSAL_V2_SCHEMA = "putback_wam_embedding_surprise_causal_v2"


def segment_from_confirmations(
    decision_count: int,
    confirmations: list[int],
    *,
    min_segment: int,
    max_segment: int,
) -> dict[str, Any]:
    """Apply arrivals in order without ever moving a boundary backward."""

    if decision_count < 1 or min_segment < 1 or max_segment < min_segment:
        raise ValueError("invalid decision/segment lengths")
    events = {int(value) for value in confirmations}
    if any(value <= 0 or value >= decision_count for value in events):
        raise ValueError("confirmations must be interior decision arrivals")
    boundaries = [0]
    reasons = {"0": "start"}
    start = 0
    for arrival in range(1, decision_count):
        length = arrival - start
        if arrival in events and length >= min_segment:
            boundaries.append(arrival)
            reasons[str(arrival)] = "surprise_confirmed"
            start = arrival
        elif length >= max_segment:
            boundaries.append(arrival)
            reasons[str(arrival)] = "max_length"
            start = arrival
    if start < decision_count:
        tail = decision_count - start
        boundaries.append(decision_count)
        reasons[str(decision_count)] = "end" if tail >= min_segment else "terminal_tail"
    return {
        "boundaries": boundaries,
        "reasons": reasons,
        "segment_lengths": [
            right - left for left, right in zip(boundaries, boundaries[1:])
        ],
    }


def build_causal_v2_trace(
    features: torch.Tensor,
    *,
    statistics_window: int = 8,
    threshold_window: int = 8,
    min_history: int = 4,
    min_threshold_history: int = 3,
    gamma: float = 1.0,
    nms_distance: int = 2,
    min_segment: int = 2,
    max_segment: int = 8,
) -> dict[str, Any]:
    """Score embeddings and close segments only at online confirmation time."""

    values = torch.as_tensor(features)
    statistics = score_causal_embeddings(
        values,
        statistics_window=statistics_window,
        threshold_window=threshold_window,
        min_history=min_history,
        min_threshold_history=min_threshold_history,
        gamma=gamma,
    )
    peaks = confirm_causal_peaks(
        statistics["scores"],
        statistics["thresholds"],
        nms_distance=nms_distance,
    )
    confirmations = [int(peak) + 1 for peak in peaks]
    partition = segment_from_confirmations(
        int(values.shape[0]),
        confirmations,
        min_segment=min_segment,
        max_segment=max_segment,
    )
    reasons = partition["reasons"]
    return {
        **statistics,
        "peaks": list(peaks),
        "peak_confirmed_at": confirmations,
        "boundaries": partition["boundaries"],
        "reasons": reasons,
        "segment_lengths": partition["segment_lengths"],
        "retroactive_boundary_count": 0,
    }
