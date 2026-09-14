"""Semantic alignment diagnostics for WAM-surprise event boundaries."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from fastwam.memory.wam_embedding_surprise import segment_from_peaks


GRIPPER_DIMS = (6, 13)


def extract_gripper_events(
    actions: np.ndarray,
    *,
    stride: int = 16,
    threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Extract open/close threshold crossings in continuous decision time."""

    values = np.asarray(actions, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != 14:
        raise ValueError("actions must have shape [F,14] with at least two frames")
    if not np.isfinite(values).all():
        raise ValueError("actions must be finite")
    if int(stride) <= 0:
        raise ValueError("stride must be positive")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must lie in [0,1]")

    events: list[dict[str, Any]] = []
    for dimension in GRIPPER_DIMS:
        state = values[:, dimension] >= float(threshold)
        for frame in (np.flatnonzero(state[1:] != state[:-1]) + 1).tolist():
            events.append(
                {
                    "raw_frame": int(frame),
                    "decision": float(frame) / int(stride),
                    "gripper_dim": int(dimension),
                    "kind": "gripper_open" if bool(state[frame]) else "gripper_close",
                }
            )
    return sorted(events, key=lambda row: (row["raw_frame"], row["gripper_dim"]))


def match_confirmations(
    confirmations: Sequence[float],
    events: Sequence[Mapping[str, Any]],
    *,
    tolerance: float = 1.0,
) -> dict[str, Any]:
    """Greedily match nearest event/confirmation pairs one-to-one."""

    arrivals = [float(value) for value in confirmations]
    if not all(math.isfinite(value) for value in arrivals):
        raise ValueError("confirmations must be finite")
    if len(set(arrivals)) != len(arrivals):
        raise ValueError("confirmations must be unique")
    if float(tolerance) < 0.0:
        raise ValueError("tolerance must be nonnegative")
    event_times = [float(row["decision"]) for row in events]
    if not all(math.isfinite(value) for value in event_times):
        raise ValueError("event decision coordinates must be finite")

    candidates = sorted(
        (
            abs(confirmation - event_time),
            confirmation,
            event_index,
            confirmation_index,
        )
        for confirmation_index, confirmation in enumerate(arrivals)
        for event_index, event_time in enumerate(event_times)
        if abs(confirmation - event_time) <= float(tolerance)
    )
    used_confirmations: set[int] = set()
    used_events: set[int] = set()
    matches: list[dict[str, Any]] = []
    for distance, confirmation, event_index, confirmation_index in candidates:
        if confirmation_index in used_confirmations or event_index in used_events:
            continue
        used_confirmations.add(confirmation_index)
        used_events.add(event_index)
        signed = confirmation - event_times[event_index]
        matches.append(
            {
                "confirmation": confirmation,
                "event_index": int(event_index),
                "event_decision": event_times[event_index],
                "signed_error": signed,
                "absolute_error": distance,
            }
        )
    matches.sort(key=lambda row: row["confirmation"])

    matched = len(matches)
    precision = matched / len(arrivals) if arrivals else 0.0
    recall = matched / len(events) if events else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "matches": matches,
        "unmatched_confirmations": [
            value for index, value in enumerate(arrivals) if index not in used_confirmations
        ],
        "missed_event_indices": [
            index for index in range(len(events)) if index not in used_events
        ],
        "counts": {
            "confirmations": len(arrivals),
            "events": len(events),
            "matched": matched,
        },
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_signed_error": (
            sum(float(row["signed_error"]) for row in matches) / matched
            if matched
            else None
        ),
        "mean_absolute_error": (
            sum(float(row["absolute_error"]) for row in matches) / matched
            if matched
            else None
        ),
    }


def _joint_motion_by_decision(
    actions: np.ndarray, decision_frames: Sequence[int]
) -> list[float]:
    joints = np.concatenate((actions[:, :6], actions[:, 7:13]), axis=1)
    frame_motion = np.linalg.norm(np.diff(joints, axis=0), axis=1)
    result: list[float] = []
    for index, start in enumerate(decision_frames):
        end = (
            int(decision_frames[index + 1])
            if index + 1 < len(decision_frames)
            else int(actions.shape[0])
        )
        window = frame_motion[int(start) : max(int(start), end - 1)]
        result.append(float(window.mean()) if window.size else 0.0)
    return result


def analyze_episode_alignment(
    trace: Mapping[str, Any],
    actions: np.ndarray,
    *,
    stride: int = 16,
    warmup_end: float = 8.0,
    tolerance: float = 1.0,
) -> dict[str, Any]:
    """Join one persisted WAM trace with strong action-derived events."""

    decision_count = int(trace["decision_count"])
    decision_frames = [int(value) for value in trace["decision_frame_indices"]]
    if len(decision_frames) != decision_count:
        raise ValueError("decision frame count does not match trace")
    if decision_frames != [index * int(stride) for index in range(decision_count)]:
        raise ValueError("trace decision frames do not match the requested stride")

    values = np.asarray(actions, dtype=np.float32)
    events = extract_gripper_events(values, stride=stride)
    confirmations = [float(value) for value in trace["peak_confirmed_at"]]
    primary = match_confirmations(confirmations, events, tolerance=tolerance)

    post_event_indices = [
        index for index, row in enumerate(events) if float(row["decision"]) > warmup_end
    ]
    post_events = [events[index] for index in post_event_indices]
    post_confirmations = [value for value in confirmations if value > warmup_end]
    post = match_confirmations(post_confirmations, post_events, tolerance=tolerance)
    for row in post["matches"]:
        row["event_index"] = post_event_indices[int(row["event_index"])]
    post["missed_event_indices"] = [
        post_event_indices[index] for index in post["missed_event_indices"]
    ]

    reasons = {str(key): str(value) for key, value in trace["reasons"].items()}
    forced = [
        int(boundary)
        for boundary in trace["boundaries"]
        if reasons.get(str(boundary), "").startswith("max_length")
    ]
    retroactive = []
    for peak, available_at in zip(trace["peaks"], trace["peak_confirmed_at"]):
        if int(peak) in trace["boundaries"] and int(available_at) > int(peak):
            retroactive.append(
                {
                    "peak_boundary": int(peak),
                    "available_at": int(available_at),
                    "lag": int(available_at) - int(peak),
                }
            )
    online_partition = segment_from_peaks(
        decision_count,
        [int(value) for value in trace["peak_confirmed_at"]],
        min_segment=2,
        max_segment=8,
    )
    return {
        "strong_events": events,
        "confirmations": confirmations,
        "primary_metrics": primary,
        "post_warmup_metrics": post,
        "warmup_end": float(warmup_end),
        "forced_boundaries": forced,
        "retroactive_boundaries": retroactive,
        "original_partition": {
            "boundaries": [int(value) for value in trace["boundaries"]],
            "segment_lengths": [int(value) for value in trace["segment_lengths"]],
            "reasons": reasons,
        },
        "online_counterfactual": online_partition,
        "joint_motion_by_decision": _joint_motion_by_decision(values, decision_frames),
    }
