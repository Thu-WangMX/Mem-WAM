"""Predeclared warmup strategies for frozen initialization-WAM features."""

from __future__ import annotations

from statistics import median
from typing import Any, Mapping, Sequence

import torch

from fastwam.memory.semantic_alignment import (
    extract_gripper_events,
    match_confirmations,
)
from fastwam.memory.wam_embedding_causal_v2 import (
    build_causal_v2_trace,
    segment_from_confirmations,
)
from fastwam.memory.wam_embedding_surprise import MAD_GAUSSIAN_SCALE


STRATEGIES = ("current", "short_history", "adjacent_cosine")


def adjacent_cosine_deltas(features: torch.Tensor) -> list[float | None]:
    values = torch.as_tensor(features, dtype=torch.float32, device="cpu")
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("features must have shape [T,D] with T >= 2")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("features must be finite")
    normalized = torch.nn.functional.normalize(values, dim=1)
    deltas = 1.0 - (normalized[1:] * normalized[:-1]).sum(dim=1)
    return [None, *[float(value) for value in deltas.tolist()]]


def calibrate_adjacent_threshold(
    feature_bank: Mapping[int, torch.Tensor],
    calibration_episodes: Sequence[int],
    *,
    gamma: float = 1.0,
) -> dict[str, Any]:
    episodes = [int(value) for value in calibration_episodes]
    if not episodes or len(episodes) != len(set(episodes)):
        raise ValueError("calibration episodes must be nonempty and unique")
    eligible = list(range(2, 9))
    values = []
    for episode in episodes:
        if episode not in feature_bank:
            raise KeyError(f"calibration feature missing for episode {episode}")
        deltas = adjacent_cosine_deltas(feature_bank[episode])
        if len(deltas) <= eligible[-1]:
            raise ValueError(f"episode {episode} is shorter than decision 8")
        values.extend(float(deltas[index]) for index in eligible)
    center = float(median(values))
    mad = float(median(abs(value - center) for value in values))
    return {
        "episodes": episodes,
        "eligible_decisions": eligible,
        "sample_count": len(values),
        "median": center,
        "mad": mad,
        "gamma": float(gamma),
        "threshold": center + float(gamma) * MAD_GAUSSIAN_SCALE * mad,
        "startup_suppressed_decision": 1,
    }


def _fixed_threshold_peaks(
    scores: Sequence[float | None],
    threshold: float,
    *,
    nms_distance: int,
) -> list[int]:
    if nms_distance < 1:
        raise ValueError("nms_distance must be positive")
    candidates = []
    for peak in range(2, len(scores) - 1):
        left, value, right = scores[peak - 1], scores[peak], scores[peak + 1]
        if None in (left, value, right):
            continue
        if float(value) > float(left) and float(value) >= float(right) and float(value) > float(threshold):
            candidates.append(peak)
    kept: list[int] = []
    for peak in candidates:
        if not kept or peak - kept[-1] >= nms_distance:
            kept.append(peak)
        elif float(scores[peak]) > float(scores[kept[-1]]):
            kept[-1] = peak
    return kept


def build_strategy_trace(
    features: torch.Tensor,
    *,
    strategy: str,
    adjacent_threshold: float | None = None,
    nms_distance: int = 2,
    min_segment: int = 2,
    max_segment: int = 8,
) -> dict[str, Any]:
    if strategy not in STRATEGIES:
        raise ValueError(f"unsupported strategy: {strategy}")
    if strategy in ("current", "short_history"):
        return {
            "strategy": strategy,
            **build_causal_v2_trace(
                features,
                statistics_window=8,
                threshold_window=8,
                min_history=4 if strategy == "current" else 2,
                min_threshold_history=3 if strategy == "current" else 2,
                gamma=1.0,
                nms_distance=nms_distance,
                min_segment=min_segment,
                max_segment=max_segment,
            ),
        }
    if adjacent_threshold is None:
        raise ValueError("adjacent_threshold is required for adjacent_cosine")
    scores = adjacent_cosine_deltas(features)
    peaks = _fixed_threshold_peaks(
        scores, float(adjacent_threshold), nms_distance=nms_distance
    )
    confirmations = [peak + 1 for peak in peaks]
    partition = segment_from_confirmations(
        len(scores),
        confirmations,
        min_segment=min_segment,
        max_segment=max_segment,
    )
    return {
        "strategy": strategy,
        "scores": scores,
        "thresholds": [None, *([float(adjacent_threshold)] * (len(scores) - 1))],
        "peaks": peaks,
        "peak_confirmed_at": confirmations,
        **partition,
        "retroactive_boundary_count": 0,
        "startup_suppressed_decision": 1,
    }


def _micro(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    confirmations = sum(int(row["counts"]["confirmations"]) for row in reports)
    events = sum(int(row["counts"]["events"]) for row in reports)
    matched = sum(int(row["counts"]["matched"]) for row in reports)
    precision = matched / confirmations if confirmations else 0.0
    recall = matched / events if events else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    errors = [
        float(match["absolute_error"])
        for row in reports
        for match in row["matches"]
    ]
    return {
        "counts": {"confirmations": confirmations, "events": events, "matched": matched},
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_absolute_error": sum(errors) / len(errors) if errors else None,
    }


def evaluate_strategy(
    traces: Mapping[int, Mapping[str, Any]],
    actions_by_episode: Mapping[int, Any],
    *,
    stride: int = 16,
    early_end: float = 8.0,
    tolerance: float = 1.0,
) -> dict[str, Any]:
    if set(traces) != set(actions_by_episode):
        raise ValueError("trace/action episode sets differ")
    per_episode = []
    all_reports = []
    early_reports = []
    for episode in sorted(traces):
        trace = traces[episode]
        events = extract_gripper_events(actions_by_episode[episode], stride=stride)
        confirmations = [float(value) for value in trace["peak_confirmed_at"]]
        all_report = match_confirmations(confirmations, events, tolerance=tolerance)
        early_indices = [
            index for index, event in enumerate(events) if float(event["decision"]) <= early_end
        ]
        early_events = [events[index] for index in early_indices]
        early_confirmations = [
            value for value in confirmations if value <= early_end + tolerance
        ]
        early_report = match_confirmations(
            early_confirmations, early_events, tolerance=tolerance
        )
        all_reports.append(all_report)
        early_reports.append(early_report)
        per_episode.append(
            {
                "episode": episode,
                "events": events,
                "all": all_report,
                "early": early_report,
                "confirmations": confirmations,
                "peaks": list(trace.get("peaks", [])),
                "boundaries": list(trace.get("boundaries", [])),
                "segment_lengths": list(trace.get("segment_lengths", [])),
            }
        )
    forced = sum(
        1
        for trace in traces.values()
        for reason in trace["reasons"].values()
        if str(reason) == "max_length"
    )
    retroactive = sum(int(trace["retroactive_boundary_count"]) for trace in traces.values())
    return {
        "all": _micro(all_reports),
        "early": _micro(early_reports),
        "forced_boundary_count": forced,
        "retroactive_boundary_count": retroactive,
        "per_episode": per_episode,
    }


def evaluate_decision_rule(
    *,
    adjacent: Mapping[str, Any],
    current: Mapping[str, Any],
    short_history: Mapping[str, Any],
) -> dict[str, Any]:
    clauses = {
        "early_recall_strictly_higher": (
            float(adjacent["early"]["recall"]) > float(current["early"]["recall"])
            and float(adjacent["early"]["recall"])
            > float(short_history["early"]["recall"])
        ),
        "precision_at_least_0_60": float(adjacent["all"]["precision"]) >= 0.60,
        "f1_within_0_05_of_current": float(adjacent["all"]["f1"])
        >= float(current["all"]["f1"]) - 0.05,
        "zero_retroactive_boundaries": int(adjacent["retroactive_boundary_count"]) == 0,
    }
    return {
        "clauses": clauses,
        "advance_to_training_boundary_experiment": all(clauses.values()),
    }
