from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from fastwam.memory.control_information_boundary import (
    fit_control_information_statistics,
)
from fastwam.memory.control_information_boundary_v3 import (
    ControlInformationBoundaryStateV3,
)
from fastwam.memory.planning_aligned_manifest import align_detector_events
from scripts.calibrate_putback_control_information_selector import (
    _sha256,
    locked_split,
)
from scripts.trace_putback_control_information import TRACE_SCHEMA


LOCK_SCHEMA_V3 = "putback_locked_control_information_candidate_v3"
REQUIRED_DEPENDENCY_KEYS_V3 = {
    "initialization_manifest",
    "feature_bank_manifest",
    "pca",
    "feature_predictor",
    "control_probe",
    "contextual_statistics",
    "normalization",
    "base_score_source",
    "base_boundary_source",
    "episode_adapter_source",
    "segment_boundary_source",
}
THRESHOLDS = (2.0, 4.0, 6.0, 8.0)
DRIFTS = (0.25, 0.5, 1.0)
DECAYS = (0.8, 0.9)
SEGMENT_SCALE_FLOORS = (0.5, 1.0, 2.0)
MIN_UNITS = 8
MAX_UNITS = 24
EPISODE_CALIBRATION_SAMPLES = 4
SEGMENT_CALIBRATION_SAMPLES = 4
ADAPTATION_EPSILON = 1e-6
MAX_ABS_LOG_BIAS = math.log(64.0)
MINIMUM_CONSECUTIVE_EVIDENCE = 2


def lock_candidate_v3(
    path: str | Path,
    *,
    candidate: Mapping[str, Any],
    dependency_paths: Mapping[str, str | Path],
) -> dict[str, Any]:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite locked v3 candidate: {path}")
    if set(dependency_paths) != REQUIRED_DEPENDENCY_KEYS_V3:
        raise ValueError(
            "v3 candidate dependencies must be exactly "
            f"{sorted(REQUIRED_DEPENDENCY_KEYS_V3)}"
        )
    payload = {
        "schema_version": LOCK_SCHEMA_V3,
        "method": "segment_relative_counterfactual_control_information",
        "split": locked_split(),
        "candidate": json.loads(json.dumps(candidate)),
        "hashes": {
            key: _sha256(Path(dependency_paths[key]))
            for key in sorted(REQUIRED_DEPENDENCY_KEYS_V3)
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return payload


def _evaluate_candidate_v3(
    *,
    traces: Mapping[int, Mapping[str, torch.Tensor]],
    episode_lengths: Mapping[int, int],
    statistics: Mapping[str, Any],
    episodes: Sequence[int],
    selector_config: Mapping[str, Any],
) -> dict[str, Any]:
    group_lengths: list[int] = []
    adaptation_factors: list[float] = []
    learned = 0
    forced = 0
    total_groups = 0
    target_groups = 0
    captured_excess = 0.0
    total_excess = 0.0
    episode_reports = {}
    for episode in episodes:
        trace = traces[int(episode)]
        frame_indices = torch.as_tensor(trace["frame_indices"], dtype=torch.int64)
        information = torch.as_tensor(trace["information"], dtype=torch.float32)
        by_frame = {
            int(frame): float(score)
            for frame, score in zip(frame_indices.tolist(), information.tolist())
        }
        state = ControlInformationBoundaryStateV3(
            statistics=statistics, **dict(selector_config)
        )
        detector_frames = list(range(0, int(episode_lengths[int(episode)]), 4))
        events = []
        excess_by_frame = {}
        for frame in detector_frames:
            event = state.update(
                frame=frame,
                information=None if frame < 16 else by_frame[frame],
            )
            innovation = state.last_segment_relative_innovation
            if innovation is not None:
                excess_by_frame[frame] = max(
                    0.0, float(innovation) - float(selector_config["drift"])
                )
            if event is not None:
                events.append(asdict(event))
        tail = state.finalize(frame=detector_frames[-1])
        if tail is not None:
            events.append(asdict(tail))
        if state.episode_adaptation_factor is None:
            raise RuntimeError("v3 episode did not freeze episode adaptation")
        adaptation_factors.append(float(state.episode_adaptation_factor))

        aligned = align_detector_events(
            events,
            episode_frames=int(episode_lengths[int(episode)]),
            replan_stride=16,
            minimum_segment_decisions=2,
            maximum_segment_decisions=8,
        )
        lengths = [
            right - left
            for left, right in zip(
                aligned["boundaries"], aligned["boundaries"][1:]
            )
        ]
        group_lengths.extend(lengths)
        total_groups += len(lengths)
        target_groups += (aligned["decision_count"] + 3) // 4
        learned_frames = []
        kept_reasons = []
        for row in aligned["confirmation_to_decision"]:
            if row["status"] != "kept":
                continue
            reason = row["reason"]
            kept_reasons.append(reason)
            if reason == "segment_relative_control_information":
                learned += 1
                learned_frames.append(int(row["confirmation_frame"]))
            elif reason == "forced_maximum":
                forced += 1
        total_excess += sum(excess_by_frame.values())
        captured_frames = {
            frame
            for confirmation in learned_frames
            for frame in (confirmation - 4, confirmation, confirmation + 4)
        }
        captured_excess += sum(
            value
            for frame, value in excess_by_frame.items()
            if frame in captured_frames
        )
        episode_reports[str(episode)] = {
            "episode_adaptation_factor": float(state.episode_adaptation_factor),
            "boundaries": aligned["boundaries"],
            "segment_lengths": lengths,
            "kept_reasons": kept_reasons,
        }

    boundary_total = learned + forced
    lengths_tensor = torch.tensor(sorted(group_lengths), dtype=torch.float32)
    factor_tensor = torch.tensor(sorted(adaptation_factors), dtype=torch.float32)
    return {
        "group_count_ratio": total_groups / target_groups,
        "learned_boundary_fraction": learned / max(boundary_total, 1),
        "captured_excess_information": captured_excess / max(total_excess, 1e-12),
        "forced_boundary_fraction": forced / max(boundary_total, 1),
        "p95_group_length": float(torch.quantile(lengths_tensor, 0.95)),
        "group_count": total_groups,
        "fixed_group_size_four_count": target_groups,
        "learned_boundary_count": learned,
        "forced_boundary_count": forced,
        "adaptation_factor_min": float(factor_tensor.min()),
        "adaptation_factor_median": float(torch.quantile(factor_tensor, 0.5)),
        "adaptation_factor_max": float(factor_tensor.max()),
        "episodes": episode_reports,
    }


def _select_candidate_v3(candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    passing = []
    for original in candidates:
        candidate = json.loads(json.dumps(original))
        metrics = candidate["metrics"]
        gates = {
            "group_count_within_10_percent": 0.9
            <= float(metrics["group_count_ratio"])
            <= 1.1,
            "learned_boundary_fraction_at_least_half": float(
                metrics["learned_boundary_fraction"]
            )
            >= 0.5,
        }
        candidate["selection_gates"] = gates
        if all(gates.values()):
            passing.append(candidate)
    if not passing:
        raise RuntimeError("no v3 candidate passes rate and learned-event gates")
    passing.sort(
        key=lambda row: (
            -float(row["metrics"]["captured_excess_information"]),
            float(row["metrics"]["forced_boundary_fraction"]),
            abs(float(row["metrics"]["group_count_ratio"]) - 1.0),
            float(row["metrics"]["p95_group_length"]),
            tuple(
                row["selector_config"][key]
                for key in (
                    "threshold",
                    "drift",
                    "decay",
                    "segment_scale_floor",
                )
            ),
        )
    )
    return passing[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", required=True)
    parser.add_argument("--initialization-manifest", required=True)
    parser.add_argument("--feature-bank-manifest", required=True)
    parser.add_argument("--pca", required=True)
    parser.add_argument("--feature-predictor", required=True)
    parser.add_argument("--control-probe", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--base-score-source", required=True)
    parser.add_argument("--base-boundary-source", required=True)
    parser.add_argument("--episode-adapter-source", required=True)
    parser.add_argument("--segment-boundary-source", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    output = Path(args.output_root).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite v3 selector calibration: {output}")
    trace_path = Path(args.traces).resolve()
    trace_payload = torch.load(trace_path, map_location="cpu", weights_only=False)
    split = locked_split()
    if (
        trace_payload.get("schema_version") != TRACE_SCHEMA
        or trace_payload.get("complete") is not True
        or trace_payload.get("traced_episodes") != split["trace_episodes"]
    ):
        raise ValueError("v3 candidate traces must contain exactly episodes 0-39")
    traces = {int(key): value for key, value in trace_payload["episodes"].items()}
    episode_lengths = {
        int(key): int(value)
        for key, value in trace_payload["episode_lengths"].items()
    }
    statistics = fit_control_information_statistics(
        traces, episodes=split["statistics_episodes"], depth_cap=4
    )

    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale v3 calibration temporary: {temporary}")
    temporary.mkdir(parents=True)
    statistics_path = temporary / "contextual_statistics.pt"
    torch.save(statistics, statistics_path)
    candidates = []
    for threshold, drift, decay, scale_floor in itertools.product(
        THRESHOLDS, DRIFTS, DECAYS, SEGMENT_SCALE_FLOORS
    ):
        config = {
            "threshold": threshold,
            "drift": drift,
            "decay": decay,
            "detector_stride": 4,
            "min_units": MIN_UNITS,
            "max_units": MAX_UNITS,
            "initial_group_start": 0,
            "episode_calibration_samples": EPISODE_CALIBRATION_SAMPLES,
            "segment_calibration_samples": SEGMENT_CALIBRATION_SAMPLES,
            "adaptation_epsilon": ADAPTATION_EPSILON,
            "max_abs_log_bias": MAX_ABS_LOG_BIAS,
            "segment_scale_floor": scale_floor,
            "minimum_consecutive_evidence": MINIMUM_CONSECUTIVE_EVIDENCE,
        }
        candidates.append(
            {
                "selector_config": config,
                "metrics": _evaluate_candidate_v3(
                    traces=traces,
                    episode_lengths=episode_lengths,
                    statistics=statistics,
                    episodes=split["calibration_episodes"],
                    selector_config=config,
                ),
            }
        )
    selected = _select_candidate_v3(candidates)
    report = {
        "schema_version": "putback_control_information_calibration_v3",
        "split": split,
        "grid": {
            "thresholds": list(THRESHOLDS),
            "drifts": list(DRIFTS),
            "decays": list(DECAYS),
            "segment_scale_floors": list(SEGMENT_SCALE_FLOORS),
            "min_units": MIN_UNITS,
            "max_units": MAX_UNITS,
            "episode_calibration_samples": EPISODE_CALIBRATION_SAMPLES,
            "segment_calibration_samples": SEGMENT_CALIBRATION_SAMPLES,
            "minimum_consecutive_evidence": MINIMUM_CONSECUTIVE_EVIDENCE,
        },
        "candidate_count": len(candidates),
        "selected": selected,
        "candidates": candidates,
        "trace_sha256": _sha256(trace_path),
    }
    (temporary / "calibration_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    lock_candidate_v3(
        temporary / "locked_candidate.json",
        candidate=selected,
        dependency_paths={
            "initialization_manifest": args.initialization_manifest,
            "feature_bank_manifest": args.feature_bank_manifest,
            "pca": args.pca,
            "feature_predictor": args.feature_predictor,
            "control_probe": args.control_probe,
            "contextual_statistics": statistics_path,
            "normalization": args.normalization,
            "base_score_source": args.base_score_source,
            "base_boundary_source": args.base_boundary_source,
            "episode_adapter_source": args.episode_adapter_source,
            "segment_boundary_source": args.segment_boundary_source,
        },
    )
    temporary.replace(output)
    print(json.dumps(selected, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
