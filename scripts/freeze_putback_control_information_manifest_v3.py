from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch

from fastwam.evaluation.control_information_online import sha256_file
from fastwam.evaluation.control_information_online_v3 import RUNTIME_LOCK_SCHEMA_V3
from fastwam.memory.control_information_boundary_v3 import (
    replay_control_information_trace_v3,
)
from fastwam.memory.dynamic_surprise import SCHEMA_VERSION, validate_episode_segments
from fastwam.memory.planning_aligned_manifest import align_detector_events
from scripts.trace_putback_control_information import TRACE_SCHEMA


def freeze_episode_v3(
    *,
    episode: int,
    trace: Mapping[str, torch.Tensor],
    episode_length: int,
    statistics: Mapping[str, Any],
    selector_config: Mapping[str, Any],
) -> dict[str, Any]:
    episode = int(episode)
    episode_length = int(episode_length)
    detector_frames = list(range(0, episode_length, 4))
    frames = torch.as_tensor(trace["frame_indices"], dtype=torch.int64)
    information = torch.as_tensor(trace["information"], dtype=torch.float32)
    if frames.ndim != 1 or information.shape != frames.shape:
        raise ValueError("v3 counterfactual-control trace shape mismatch")
    information_by_frame = {
        int(frame): float(value)
        for frame, value in zip(frames.tolist(), information.tolist())
    }
    events = replay_control_information_trace_v3(
        detector_frames=detector_frames,
        information_by_frame=information_by_frame,
        statistics=statistics,
        selector_config=selector_config,
    )
    if not events:
        raise RuntimeError("v3 episode did not produce a terminal partition")
    factors = {float(event.episode_adaptation_factor) for event in events}
    if len(factors) != 1:
        raise RuntimeError("v3 episode adaptation changed after calibration")
    aligned = align_detector_events(
        [asdict(event) for event in events],
        episode_frames=episode_length,
        replan_stride=16,
        minimum_segment_decisions=2,
        maximum_segment_decisions=8,
    )
    payload = {
        "episode": episode,
        "decision_count": aligned["decision_count"],
        "boundaries": aligned["boundaries"],
        "reasons": aligned["reasons"],
        "alignment": aligned["confirmation_to_decision"],
        "retroactive_boundary_count": aligned["retroactive_boundary_count"],
        "episode_adaptation_factor": factors.pop(),
        "episode_calibration_frames": [16, 20, 24, 28],
        "segment_calibration_samples": int(
            selector_config["segment_calibration_samples"]
        ),
        "minimum_consecutive_evidence": int(
            selector_config["minimum_consecutive_evidence"]
        ),
        "detector_events": [asdict(event) for event in events],
    }
    validate_episode_segments(payload)
    return payload


def _validated_trace(path: Path, expected_episodes: list[int]) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("schema_version") != TRACE_SCHEMA
        or payload.get("complete") is not True
        or payload.get("traced_episodes") != expected_episodes
    ):
        raise ValueError(f"v3 trace {path} does not contain episodes {expected_episodes}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces-0-39", required=True)
    parser.add_argument("--traces-40-49", required=True)
    parser.add_argument("--contextual-statistics", required=True)
    parser.add_argument("--runtime-lock", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite v3 planning manifest: {output}")
    runtime_lock_path = Path(args.runtime_lock).resolve()
    runtime_lock = json.loads(runtime_lock_path.read_text())
    if runtime_lock.get("schema_version") != RUNTIME_LOCK_SCHEMA_V3:
        raise ValueError("v3 control-information runtime lock schema is incompatible")
    if runtime_lock["runtime_hashes"]["offline_freezer_source"] != sha256_file(
        Path(__file__).resolve()
    ):
        raise ValueError("v3 offline freezer source differs from runtime lock")
    statistics_path = Path(args.contextual_statistics).resolve()
    candidate_lock = runtime_lock["candidate"]
    if candidate_lock["hashes"]["contextual_statistics"] != sha256_file(
        statistics_path
    ):
        raise ValueError("v3 contextual statistics differ from locked candidate")
    statistics = torch.load(statistics_path, map_location="cpu", weights_only=True)
    selector_config = candidate_lock["candidate"]["selector_config"]

    early_path = Path(args.traces_0_39).resolve()
    late_path = Path(args.traces_40_49).resolve()
    early = _validated_trace(early_path, list(range(40)))
    late = _validated_trace(late_path, list(range(40, 50)))
    if late["hashes"].get("runtime_lock") != sha256_file(runtime_lock_path):
        raise ValueError("v3 heldout traces were not opened by this runtime lock")

    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale v3 planning-manifest temporary: {temporary}")
    temporary.mkdir(parents=True)
    (temporary / "episodes").mkdir()
    episode_paths = {}
    segment_lengths = Counter()
    reason_counts = Counter()
    status_counts = Counter()
    detector_reason_counts = Counter()
    adaptation_factors = []
    baseline_scales = []
    for episode in range(50):
        source = early if episode < 40 else late
        traces = {int(key): value for key, value in source["episodes"].items()}
        lengths = {
            int(key): int(value) for key, value in source["episode_lengths"].items()
        }
        payload = freeze_episode_v3(
            episode=episode,
            trace=traces[episode],
            episode_length=lengths[episode],
            statistics=statistics,
            selector_config=selector_config,
        )
        adaptation_factors.append(float(payload["episode_adaptation_factor"]))
        for event in payload["detector_events"]:
            detector_reason_counts[event["reason"]] += 1
            if event["reason"] != "terminal_tail":
                baseline_scales.append(float(event["segment_baseline_scale"]))
        for left, right in validate_episode_segments(payload):
            segment_lengths[right - left] += 1
        for row in payload["alignment"]:
            status_counts[row["status"]] += 1
            if row["status"] == "kept":
                reason_counts[row["reason"]] += 1
        relative = f"episodes/episode_{episode:03d}.json"
        (temporary / relative).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        episode_paths[str(episode)] = relative

    factor_tensor = torch.tensor(adaptation_factors, dtype=torch.float32)
    scale_tensor = torch.tensor(baseline_scales, dtype=torch.float32)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "task": "putback",
        "episode_count": 50,
        "boundary_step": -1,
        "replan_stride": 16,
        "detector_stride": 4,
        "selector": "locked_segment_relative_control_information_v3",
        "runtime_lock_sha256": sha256_file(runtime_lock_path),
        "traces_0_39_sha256": sha256_file(early_path),
        "traces_40_49_sha256": sha256_file(late_path),
        "contextual_statistics_sha256": sha256_file(statistics_path),
        "alignment_policy": "ceil_confirmation_to_next_stride16_min2_max8",
        "memory_tokens_per_group": 8,
        "episode_calibration_samples": int(
            selector_config["episode_calibration_samples"]
        ),
        "segment_calibration_samples": int(
            selector_config["segment_calibration_samples"]
        ),
        "minimum_consecutive_evidence": int(
            selector_config["minimum_consecutive_evidence"]
        ),
        "adaptation_factor_min": float(factor_tensor.min()),
        "adaptation_factor_median": float(torch.quantile(factor_tensor, 0.5)),
        "adaptation_factor_max": float(factor_tensor.max()),
        "segment_baseline_scale_min": float(scale_tensor.min()),
        "segment_baseline_scale_median": float(torch.quantile(scale_tensor, 0.5)),
        "segment_baseline_scale_max": float(scale_tensor.max()),
        "segment_length_histogram": dict(sorted(segment_lengths.items())),
        "kept_reason_counts": dict(reason_counts),
        "alignment_status_counts": dict(status_counts),
        "detector_reason_counts": dict(detector_reason_counts),
        "retroactive_boundary_count": 0,
    }
    (temporary / "manifest.json").write_text(
        json.dumps(
            {"metadata": metadata, "episodes": episode_paths},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    temporary.replace(output)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
