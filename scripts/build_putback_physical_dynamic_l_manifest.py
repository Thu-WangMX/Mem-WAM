#!/usr/bin/env python3
"""Freeze the training manifest for the train-free physical dynamic-L selector."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from fastwam.memory.physical_dynamic_l import SCHEMA_VERSION, validate_segments


ARM_DIMS = tuple(range(6)) + tuple(range(7, 13))
GRIPPER_DIMS = (6, 13)


def load_states(root: Path, episode: int) -> np.ndarray:
    path = root / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
    table = pq.read_table(path, columns=["observation.state"])
    value = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != 14:
        raise ValueError(f"episode {episode} has invalid proprio shape {value.shape}")
    return value


def trailing_mean(value: np.ndarray, window: int) -> np.ndarray:
    output = np.empty_like(value)
    for index in range(len(value)):
        output[index] = value[max(0, index - window + 1) : index + 1].mean()
    return output


def motion_statistics(episodes: list[np.ndarray]) -> tuple[np.ndarray, float]:
    deltas = [np.diff(states[:, ARM_DIMS], axis=0) for states in episodes]
    all_deltas = np.concatenate(deltas, axis=0)
    scale = np.std(all_deltas, axis=0)
    scale = np.maximum(scale, np.quantile(scale, 0.15)).astype(np.float32)
    normalized = all_deltas / np.maximum(scale, 1e-5)
    motion = np.sqrt(np.mean(normalized * normalized, axis=1))
    return scale, float(np.median(motion))


def physical_events(
    states: np.ndarray,
    scale: np.ndarray,
    motion_gate: float,
    *,
    stride: int,
    window: int = 8,
    peak_radius: int = 4,
    refractory: int = 16,
) -> list[dict]:
    arm = states[:, ARM_DIMS]
    delta = np.diff(arm, axis=0, prepend=arm[:1]) / np.maximum(scale, 1e-5)
    motion = np.sqrt(np.mean(delta * delta, axis=1))
    smooth = trailing_mean(motion, window)
    settled = 1.0 / (1.0 + smooth)
    candidates = []
    for index in range(peak_radius, len(states) - peak_radius):
        neighborhood = settled[index - peak_radius : index + peak_radius + 1]
        prior_motion = motion[max(1, index - 2 * window) : index + 1]
        if settled[index] < float(neighborhood.max()) or not prior_motion.size:
            continue
        if float(prior_motion.max()) <= motion_gate:
            continue
        candidates.append(
            {
                "raw_frame": index,
                "planning_index": (index + stride - 1) // stride,
                "known_at": (index + peak_radius + stride - 1) // stride,
                "family": "settled_peak",
                "score": float(settled[index] - min(neighborhood[0], neighborhood[-1])),
            }
        )
    selected = []
    for candidate in sorted(candidates, key=lambda row: row["score"], reverse=True):
        if all(abs(candidate["raw_frame"] - row["raw_frame"]) >= refractory for row in selected):
            selected.append(candidate)

    for dimension, side in zip(GRIPPER_DIMS, ("left", "right")):
        value = states[:, dimension]
        stable = 0 if value[0] <= 0.2 else (1 if value[0] >= 0.8 else None)
        for index in range(1, len(value)):
            reached = 0 if value[index] <= 0.2 else (1 if value[index] >= 0.8 else None)
            if reached is None:
                continue
            if stable is not None and reached != stable:
                selected.append(
                    {
                        "raw_frame": index,
                        "planning_index": (index + stride - 1) // stride,
                        "known_at": (index + stride - 1) // stride,
                        "family": "gripper",
                        "kind": f"gripper_{side}_{'closed' if reached else 'open'}",
                        "score": 1.0,
                    }
                )
            stable = reached
    return sorted(selected, key=lambda row: (row["planning_index"], row["family"]))


def causal_segments(
    decision_count: int,
    events: list[dict],
    *,
    anchor_frames: int = 2,
    min_l: int = 4,
    nominal_l: int = 6,
    max_l: int = 8,
) -> list[dict]:
    """Wait for eight source frames, then retroactively snap near nominal L6."""
    segments = []
    start = int(anchor_frames)
    while start + max_l < decision_count:
        confirmed_at = start + max_l
        nominal_endpoint = start + nominal_l - 1
        candidates = [
            event
            for event in events
            if start + min_l - 1 <= int(event["planning_index"]) <= start + max_l - 1
            and int(event["known_at"]) <= confirmed_at
        ]
        if candidates:
            selected = min(
                candidates,
                key=lambda event: (
                    0 if event["family"] == "gripper" else 1,
                    abs(int(event["planning_index"]) - nominal_endpoint),
                    -float(event["score"]),
                ),
            )
            endpoint = int(selected["planning_index"])
            reason = f"snap_{selected['family']}"
        else:
            endpoint = nominal_endpoint
            reason = "nominal_L0"
        segments.append(
            {
                "start": start,
                "end": endpoint + 1,
                "confirmed_at": confirmed_at,
                "reason": reason,
            }
        )
        start = endpoint + 1
    return segments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lerobot-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()
    root = Path(args.lerobot_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite physical manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    all_states = [load_states(root, episode) for episode in range(args.episodes)]
    scale, motion_gate = motion_statistics(all_states)
    with tempfile.TemporaryDirectory(
        dir=output.parent, prefix=f".{output.name}.building."
    ) as temporary:
        staging = Path(temporary)
        (staging / "episodes").mkdir()
        mapping = {}
        histogram = Counter()
        reason_histogram = Counter()
        for episode, states in enumerate(all_states):
            decision_count = len(states[::16])
            events = physical_events(states, scale, motion_gate, stride=16)
            segments = causal_segments(decision_count, events)
            payload = {
                "episode": episode,
                "decision_count": decision_count,
                "segments": segments,
            }
            validate_segments(payload)
            relative = f"episodes/episode_{episode:03d}.json"
            (staging / relative).write_text(json.dumps(payload, indent=2) + "\n")
            mapping[str(episode)] = relative
            histogram.update(row["end"] - row["start"] for row in segments)
            reason_histogram.update(row["reason"] for row in segments)
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "task": "put_back_block",
            "episode_count": args.episodes,
            "replan_stride": 16,
            "anchor_frames": 2,
            "recent_frames": 4,
            "min_segment": 4,
            "nominal_segment": 6,
            "max_segment": 8,
            "memory_tokens": 8,
            "uses_robotwin_pretrained": False,
            "selector": "full_rate_joint_slowdown_plus_completed_gripper_snap_to_L6",
            "joint_dimensions": list(ARM_DIMS),
            "gripper_dimensions": list(GRIPPER_DIMS),
            "joint_delta_scale": scale.tolist(),
            "motion_gate": motion_gate,
            "settledness_window": 8,
            "peak_radius": 4,
            "refractory": 16,
            "gripper_hysteresis": [0.2, 0.8],
            "segment_length_histogram": dict(sorted(histogram.items())),
            "reason_histogram": dict(sorted(reason_histogram.items())),
        }
        (staging / "manifest.json").write_text(
            json.dumps({"metadata": metadata, "episodes": mapping}, indent=2) + "\n"
        )
        staging.rename(output)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
