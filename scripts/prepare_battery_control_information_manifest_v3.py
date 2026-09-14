from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from fastwam.evaluation.control_information_online import sha256_file
from fastwam.evaluation.control_information_online_v3 import RUNTIME_LOCK_SCHEMA_V3
from fastwam.evaluation.embodied_information_online import load_predictor
from fastwam.memory.control_information_probe import load_compact_control_probe
from fastwam.memory.dynamic_surprise import SCHEMA_VERSION, validate_episode_segments
from scripts.build_putback_control_information_dataset import DATASET_SCHEMA
from scripts.build_putback_feature_predictor_dataset import build_projected_episode
from scripts.freeze_putback_control_information_manifest_v3 import freeze_episode_v3
from scripts.prepare_putback_phase_annotation import load_episode_control
from scripts.trace_putback_control_information import TRACE_SCHEMA, trace_episode


def _require_locked_hash(
    candidate: dict[str, Any], name: str, path: Path
) -> None:
    expected = candidate["hashes"].get(name)
    actual = sha256_file(path)
    if expected != actual:
        raise ValueError(
            f"frozen selector dependency {name} changed: {actual} != {expected}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-bank", required=True)
    parser.add_argument("--pca", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--probe-dataset", required=True)
    parser.add_argument("--feature-predictor", required=True)
    parser.add_argument("--control-probe", required=True)
    parser.add_argument("--contextual-statistics", required=True)
    parser.add_argument("--runtime-lock", required=True)
    parser.add_argument("--analysis-output", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-history", type=int, default=8)
    args = parser.parse_args()

    if args.max_history != 8:
        raise ValueError("v3 feature-predictor history is locked to eight")

    analysis_output = Path(args.analysis_output).expanduser().resolve()
    manifest_output = Path(args.manifest_output).expanduser().resolve()
    for output in (analysis_output, manifest_output):
        if output.exists() or output.with_name(output.name + ".tmp").exists():
            raise FileExistsError(f"refusing existing output or temporary: {output}")

    runtime_lock_path = Path(args.runtime_lock).resolve()
    runtime_lock = json.loads(runtime_lock_path.read_text())
    if runtime_lock.get("schema_version") != RUNTIME_LOCK_SCHEMA_V3:
        raise ValueError("runtime lock is not the frozen v3 selector")
    candidate = runtime_lock["candidate"]

    pca_path = Path(args.pca).resolve()
    predictor_path = Path(args.feature_predictor).resolve()
    probe_path = Path(args.control_probe).resolve()
    statistics_path = Path(args.contextual_statistics).resolve()
    _require_locked_hash(candidate, "pca", pca_path)
    _require_locked_hash(candidate, "feature_predictor", predictor_path)
    _require_locked_hash(candidate, "control_probe", probe_path)
    _require_locked_hash(candidate, "contextual_statistics", statistics_path)

    feature_root = Path(args.feature_bank).resolve()
    feature_manifest_path = feature_root / "bank_manifest.json"
    feature_manifest = json.loads(feature_manifest_path.read_text())
    if feature_manifest.get("complete") is not True:
        raise ValueError("Battery initialization-WAM feature bank is incomplete")
    if int(feature_manifest.get("episode_count", -1)) != 50:
        raise ValueError("Battery feature bank must contain exactly 50 episodes")

    probe_dataset_path = Path(args.probe_dataset).resolve()
    probe_payload = torch.load(
        probe_dataset_path, map_location="cpu", weights_only=False
    )
    if probe_payload.get("schema_version") != DATASET_SCHEMA:
        raise ValueError("Battery control-probe dataset schema is incompatible")
    probe_rows_by_episode = {
        episode: [
            row
            for row in probe_payload["rows"]
            if int(row["episode"]) == episode
        ]
        for episode in range(50)
    }

    device = torch.device(args.device)
    predictor = load_predictor(predictor_path, device=device)
    control_probe, control_metadata = load_compact_control_probe(
        probe_path, device=device
    )
    pca = torch.load(pca_path, map_location="cpu", weights_only=True)
    statistics = torch.load(
        statistics_path, map_location="cpu", weights_only=True
    )
    selector_config = candidate["candidate"]["selector_config"]
    dataset_root = Path(args.dataset_root).resolve()

    traces: dict[int, dict[str, Any]] = {}
    episode_lengths: dict[int, int] = {}
    frozen_episodes: dict[int, dict[str, Any]] = {}
    for episode in range(50):
        feature = torch.load(
            feature_root / "features" / f"episode_{episode:03d}.pt",
            map_location="cpu",
            weights_only=True,
        )
        maximum_frame = max(
            int(stream["frame_indices"].max())
            for stream in feature["phases"].values()
        )
        actions, proprio = load_episode_control(
            dataset_root, episode, list(range(maximum_frame + 1))
        )
        dynamics = build_projected_episode(
            episode=episode,
            feature_payload=feature,
            pca_artifact=pca,
            actions=torch.from_numpy(actions),
            proprio=torch.from_numpy(proprio),
            max_history=args.max_history,
        )["visual_action"]
        trace = trace_episode(
            episode=episode,
            dynamics_examples=dynamics,
            control_probe_examples=probe_rows_by_episode[episode],
            feature_predictor=predictor,
            control_probe=control_probe,
        )
        traces[episode] = trace
        episode_lengths[episode] = len(actions)
        frozen_episodes[episode] = freeze_episode_v3(
            episode=episode,
            trace=trace,
            episode_length=len(actions),
            statistics=statistics,
            selector_config=selector_config,
        )
        print(
            json.dumps(
                {
                    "episode": episode,
                    "scores": len(trace["frame_indices"]),
                    "mean_information": float(trace["information"].mean()),
                    "segments": len(frozen_episodes[episode]["boundaries"]) - 1,
                }
            ),
            flush=True,
        )

    analysis_tmp = analysis_output.with_name(analysis_output.name + ".tmp")
    analysis_tmp.mkdir(parents=True)
    trace_path = analysis_tmp / "scores_0_49.pt"
    torch.save(
        {
            "schema_version": TRACE_SCHEMA,
            "trace_authorization": "cross_task_frozen_control_information_runtime_v3",
            "complete": True,
            "episodes": traces,
            "episode_lengths": episode_lengths,
            "traced_episodes": list(range(50)),
            "feature_predictor_schema": "putback_feature_predictor_model_v1",
            "control_probe_schema": control_metadata["schema_version"],
            "hashes": {
                "battery_feature_bank_manifest": sha256_file(feature_manifest_path),
                "battery_probe_dataset": sha256_file(probe_dataset_path),
                "pca": sha256_file(pca_path),
                "feature_predictor": sha256_file(predictor_path),
                "control_probe": sha256_file(probe_path),
                "contextual_statistics": sha256_file(statistics_path),
                "runtime_lock": sha256_file(runtime_lock_path),
            },
        },
        trace_path,
    )

    manifest_tmp = manifest_output.with_name(manifest_output.name + ".tmp")
    manifest_tmp.mkdir(parents=True)
    (manifest_tmp / "episodes").mkdir()
    episode_paths: dict[str, str] = {}
    segment_lengths: Counter[int] = Counter()
    reason_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    detector_reason_counts: Counter[str] = Counter()
    adaptation_factors: list[float] = []
    baseline_scales: list[float] = []

    for episode, payload in frozen_episodes.items():
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
        (manifest_tmp / relative).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        episode_paths[str(episode)] = relative

    factor_tensor = torch.tensor(adaptation_factors, dtype=torch.float32)
    scale_tensor = torch.tensor(baseline_scales, dtype=torch.float32)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "task": "battery_try",
        "episode_count": 50,
        "boundary_step": -1,
        "replan_stride": 16,
        "detector_stride": 4,
        "selector": "locked_segment_relative_control_information_v3",
        "selector_transfer": "putback_locked_v3_to_battery_zero_refit",
        "runtime_lock_sha256": sha256_file(runtime_lock_path),
        "trace_sha256": sha256_file(trace_path),
        "battery_feature_bank_sha256": sha256_file(feature_manifest_path),
        "battery_probe_dataset_sha256": sha256_file(probe_dataset_path),
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
    (manifest_tmp / "manifest.json").write_text(
        json.dumps(
            {"metadata": metadata, "episodes": episode_paths},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    analysis_tmp.replace(analysis_output)
    manifest_tmp.replace(manifest_output)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
