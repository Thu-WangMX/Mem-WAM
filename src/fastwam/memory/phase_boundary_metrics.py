"""Semantic PutBack phase-boundary metrics and publication reliability gate."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from fastwam.memory.phase_annotation import TRANSITIONS


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _match_episode(events, annotation: dict[str, Any], tolerance: int):
    truths = [row for row in annotation["transitions"] if not row.get("not_observed", False)]
    pairs = sorted(
        (
            (abs(int(event.frame) - int(truth["frame"])), event_index, truth_index)
            for event_index, event in enumerate(events)
            for truth_index, truth in enumerate(truths)
            if abs(int(event.frame) - int(truth["frame"])) <= tolerance
        ),
        key=lambda row: (row[0], int(events[row[1]].frame), row[2]),
    )
    used_events: set[int] = set()
    used_truths: set[int] = set()
    matches = []
    for distance, event_index, truth_index in pairs:
        if event_index in used_events or truth_index in used_truths:
            continue
        used_events.add(event_index)
        used_truths.add(truth_index)
        matches.append((events[event_index], truths[truth_index], distance))
    return matches, len(events) - len(matches), len(truths) - len(matches), truths


def evaluate_phase_boundaries(
    predictions_by_episode: Mapping[int, Sequence[Any]],
    annotations_by_episode: Mapping[int, dict[str, Any]],
    *,
    tolerance: int = 8,
    baseline_f1: float,
    retroactive_boundary_count: int,
    uniform_four_frame_budget: int,
) -> dict[str, Any]:
    if set(predictions_by_episode) != set(annotations_by_episode):
        raise ValueError("prediction and annotation episode sets differ")
    totals = Counter()
    transition_totals = Counter()
    transition_hits = Counter()
    timing_errors: list[int] = []
    per_episode = {}
    group_lengths = Counter()
    forced = 0
    memory_count = 0
    for episode in sorted(predictions_by_episode):
        events = list(predictions_by_episode[episode])
        matches, fp, fn, truths = _match_episode(events, annotations_by_episode[episode], tolerance)
        tp = len(matches)
        totals.update(true_positive=tp, false_positive=fp, false_negative=fn)
        per_episode[int(episode)] = _prf(tp, fp, fn)
        for truth in truths:
            transition_totals[truth["name"]] += 1
        for event, truth, distance in matches:
            transition_hits[truth["name"]] += 1
            timing_errors.append(distance)
        for event in events:
            forced += int(event.reason == "forced_maximum")
            memory_count += 1
            group_lengths[int(event.group_end) - int(event.group_start)] += 1
    micro = _prf(totals["true_positive"], totals["false_positive"], totals["false_negative"])
    macro = {
        key: sum(row[key] for row in per_episode.values()) / len(per_episode)
        for key in ("precision", "recall", "f1")
    }
    return {
        "counts": dict(totals),
        "micro": micro,
        "macro": macro,
        "per_episode": per_episode,
        "transition_recall": {
            name: _safe_ratio(transition_hits[name], transition_totals[name])
            for name in TRANSITIONS
        },
        "timing_mae": sum(timing_errors) / len(timing_errors) if timing_errors else None,
        "forced_maximum_count": forced,
        "memory_count": memory_count,
        "group_length_histogram": dict(sorted(group_lengths.items())),
        "baseline_f1": float(baseline_f1),
        "retroactive_boundary_count": int(retroactive_boundary_count),
        "uniform_four_frame_budget": int(uniform_four_frame_budget),
    }


def reliability_gate(report: Mapping[str, Any]) -> dict[str, bool]:
    checks = {
        "precision_at_least_0_60": report["micro"]["precision"] >= 0.60,
        "recall_at_least_0_60": report["micro"]["recall"] >= 0.60,
        "f1_at_least_0_65": report["micro"]["f1"] >= 0.65,
        "all_phase_recalls_nonzero": all(
            report["transition_recall"].get(name, 0.0) > 0 for name in TRANSITIONS
        ),
        "f1_gain_at_least_0_10": report["micro"]["f1"] - report["baseline_f1"] >= 0.10 - 1e-12,
        "zero_retroactive": report["retroactive_boundary_count"] == 0,
        "within_uniform_four_frame_budget": report["memory_count"]
        <= report["uniform_four_frame_budget"],
        "at_least_two_dynamic_lengths": len(report["group_length_histogram"]) >= 2,
    }
    return {**checks, "pass": all(checks.values())}

