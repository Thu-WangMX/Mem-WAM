"""Causal WAM-embedding surprise and bounded dynamic segmentation."""

from __future__ import annotations

import math
from statistics import median
from typing import Any, Mapping, Sequence

import torch


MAD_GAUSSIAN_SCALE = 1.4826


def _causal_threshold(
    prior_scores: Sequence[float],
    *,
    gamma: float,
) -> float:
    center = float(median(prior_scores))
    deviation = float(median(abs(value - center) for value in prior_scores))
    return center + float(gamma) * MAD_GAUSSIAN_SCALE * deviation


def score_causal_embeddings(
    features: torch.Tensor,
    *,
    statistics_window: int = 8,
    threshold_window: int = 8,
    min_history: int = 4,
    min_threshold_history: int = 3,
    gamma: float = 1.0,
    absolute_floor: float = 1.0e-4,
    relative_floor: float = 0.1,
) -> dict[str, list[Any]]:
    """Score normalized embeddings without reading the current/future baseline."""

    values = torch.as_tensor(features, dtype=torch.float32, device="cpu")
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("features must have shape [T,D] with nonzero dimensions")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("features must be finite")
    if statistics_window < 1 or threshold_window < 1:
        raise ValueError("window lengths must be positive")
    if min_history < 2 or min_threshold_history < 1:
        raise ValueError("history lengths are invalid")
    if absolute_floor <= 0.0 or relative_floor < 0.0:
        raise ValueError("standard-deviation floors are invalid")

    scores: list[float | None] = []
    thresholds: list[float | None] = []
    score_source_end: list[int | None] = []
    threshold_source_end: list[int | None] = []
    sigma_floors: list[float | None] = []

    for index in range(int(values.shape[0])):
        if index < min_history:
            score = None
            sigma_floor = None
            score_end = None
        else:
            prior = values[max(0, index - statistics_window) : index]
            mean = prior.mean(dim=0)
            sigma = prior.std(dim=0, unbiased=False)
            nonzero = sigma[sigma > 0]
            relative = (
                float(nonzero.median().item()) * float(relative_floor)
                if nonzero.numel()
                else 0.0
            )
            sigma_floor = max(float(absolute_floor), relative)
            standardized = (values[index] - mean).abs() / sigma.clamp_min(
                sigma_floor
            )
            score = float(standardized.mean().item())
            score_end = index - 1

        valid_prior_scores = [
            float(item)
            for item in scores[max(0, len(scores) - threshold_window) :]
            if item is not None
        ]
        if len(valid_prior_scores) >= min_threshold_history:
            threshold = _causal_threshold(valid_prior_scores, gamma=gamma)
            threshold_end = index - 1
        else:
            threshold = None
            threshold_end = None

        scores.append(score)
        thresholds.append(threshold)
        score_source_end.append(score_end)
        threshold_source_end.append(threshold_end)
        sigma_floors.append(sigma_floor)

    return {
        "scores": scores,
        "thresholds": thresholds,
        "score_source_end": score_source_end,
        "threshold_source_end": threshold_source_end,
        "sigma_floors": sigma_floors,
    }


def confirm_causal_peaks(
    scores: Sequence[float | None],
    thresholds: Sequence[float | None],
    *,
    nms_distance: int = 2,
) -> tuple[int, ...]:
    """Confirm local peaks at index ``p`` only after score ``p+1`` exists."""

    if len(scores) != len(thresholds):
        raise ValueError("scores and thresholds must have equal length")
    if nms_distance < 1:
        raise ValueError("nms_distance must be positive")

    candidates: list[int] = []
    for peak in range(1, len(scores) - 1):
        left, value, right = scores[peak - 1], scores[peak], scores[peak + 1]
        threshold = thresholds[peak]
        if None in (left, value, right, threshold):
            continue
        if (
            float(value) > float(left)
            and float(value) >= float(right)
            and float(value) > float(threshold)
        ):
            candidates.append(peak)

    kept: list[int] = []
    for peak in candidates:
        if not kept or peak - kept[-1] >= nms_distance:
            kept.append(peak)
            continue
        previous = kept[-1]
        if float(scores[peak]) > float(scores[previous]):
            kept[-1] = peak
    return tuple(kept)


def segment_from_peaks(
    decision_count: int,
    peaks: Sequence[int],
    *,
    min_segment: int = 2,
    max_segment: int = 8,
) -> dict[str, object]:
    """Convert causal event starts into a complete bounded episode partition."""

    decision_count = int(decision_count)
    min_segment = int(min_segment)
    max_segment = int(max_segment)
    if decision_count < min_segment:
        raise ValueError("decision_count is shorter than min_segment")
    if min_segment < 1 or max_segment < min_segment:
        raise ValueError("invalid segment bounds")
    candidates = sorted({int(peak) for peak in peaks if 0 < int(peak) < decision_count})

    boundaries = [0]
    reasons = {"0": "start"}
    start = 0
    while start < decision_count:
        remaining = decision_count - start
        if remaining <= max_segment:
            end_limit = decision_count
        else:
            end_limit = start + max_segment

        upcoming = next((peak for peak in candidates if peak > start), None)
        reason = "end" if end_limit == decision_count else "max_length"
        end = end_limit
        if upcoming is not None and upcoming <= end_limit:
            end = max(start + min_segment, upcoming)
            if end <= end_limit:
                reason = "surprise" if end == upcoming else "surprise_delayed"
            else:
                end = end_limit

        if decision_count - end == 1:
            if end - start > min_segment:
                end -= 1
                if reason == "surprise":
                    reason = "surprise_shifted_for_tail"
                elif reason == "end":
                    reason = "tail_rebalanced"
            elif end != decision_count:
                end = decision_count
                reason = "end"

        if end <= start or end - start > max_segment:
            raise RuntimeError(
                f"failed to construct bounded segment from {start} to {end}"
            )
        boundaries.append(end)
        reasons[str(end)] = "end" if end == decision_count else reason
        start = end

    lengths = [right - left for left, right in zip(boundaries, boundaries[1:])]
    return {
        "boundaries": boundaries,
        "reasons": reasons,
        "segment_lengths": lengths,
    }


def validate_causal_trace(payload: Mapping[str, Any]) -> None:
    """Validate causality and partition invariants for one persisted episode."""

    decision_count = int(payload["decision_count"])
    if decision_count < 2:
        raise ValueError("decision_count must be at least two")
    for field in (
        "scores",
        "thresholds",
        "score_source_end",
        "threshold_source_end",
    ):
        if len(payload[field]) != decision_count:
            raise ValueError(f"{field} length does not match decision_count")

    for index, source_end in enumerate(payload["score_source_end"]):
        if source_end is not None and int(source_end) > index - 1:
            raise ValueError(f"future-dependent score at decision {index}")
    for index, source_end in enumerate(payload["threshold_source_end"]):
        if source_end is not None and int(source_end) > index - 1:
            raise ValueError(f"future-dependent threshold at decision {index}")

    peaks = [int(value) for value in payload["peaks"]]
    if peaks != sorted(set(peaks)) or any(
        peak <= 0 or peak >= decision_count for peak in peaks
    ):
        raise ValueError("peaks are invalid")

    boundaries = [int(value) for value in payload["boundaries"]]
    if boundaries[0] != 0 or boundaries[-1] != decision_count:
        raise ValueError("boundaries must span the complete episode")
    if any(right <= left for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError("boundaries must be strictly increasing")
    lengths = [right - left for left, right in zip(boundaries, boundaries[1:])]
    if lengths != [int(value) for value in payload["segment_lengths"]]:
        raise ValueError("segment lengths do not match boundaries")
    if any(length < 2 or length > 8 for length in lengths):
        raise ValueError("segments must have length 2 through 8")
    reasons = payload["reasons"]
    if reasons.get("0") != "start" or reasons.get(str(decision_count)) != "end":
        raise ValueError("start/end reasons are invalid")
