from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from fastwam.memory.control_information_boundary import (
    ControlInformationBoundaryState,
    contextual_information_z,
    fit_control_information_statistics,
)
from fastwam.memory.planning_aligned_manifest import align_detector_events
from scripts.trace_putback_control_information import TRACE_SCHEMA


LOCK_SCHEMA = "putback_locked_control_information_candidate_v1"
REQUIRED_DEPENDENCY_KEYS = {
    "initialization_manifest",
    "feature_bank_manifest",
    "pca",
    "feature_predictor",
    "control_probe",
    "contextual_statistics",
    "normalization",
    "boundary_source",
}
THRESHOLDS = (2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0)
DRIFTS = (0.25, 0.5, 1.0)
DECAYS = (0.8, 0.9, 1.0)
MIN_UNITS = (4, 8)
MAX_UNITS = (16, 24)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locked_split() -> dict[str, list[int]]:
    return {
        "trace_episodes": list(range(40)),
        "statistics_episodes": list(range(30)),
        "calibration_episodes": list(range(30, 40)),
        "heldout_episodes": list(range(40, 50)),
    }


def select_calibrated_candidate(
    candidates: Sequence[dict[str, Any]],
) -> dict[str, Any]:
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
        raise RuntimeError("no selector candidate passes rate and learned-event gates")
    passing.sort(
        key=lambda row: (
            -float(row["metrics"]["captured_excess_information"]),
            float(row["metrics"]["forced_boundary_fraction"]),
            float(row["metrics"]["p95_group_length"]),
            tuple(
                row["selector_config"][key]
                for key in (
                    "threshold",
                    "drift",
                    "decay",
                    "min_units",
                    "max_units",
                )
            ),
        )
    )
    return passing[0]


def lock_candidate(
    path: str | Path,
    *,
    candidate: Mapping[str, Any],
    dependency_paths: Mapping[str, str | Path],
) -> dict[str, Any]:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite locked candidate: {path}")
    if set(dependency_paths) != REQUIRED_DEPENDENCY_KEYS:
        raise ValueError(
            f"candidate dependencies must be exactly {sorted(REQUIRED_DEPENDENCY_KEYS)}"
        )
    hashes = {
        key: _sha256(Path(dependency_paths[key]))
        for key in sorted(REQUIRED_DEPENDENCY_KEYS)
    }
    payload = {
        "schema_version": LOCK_SCHEMA,
        "method": "counterfactual_control_information",
        "split": locked_split(),
        "candidate": json.loads(json.dumps(candidate)),
        "hashes": hashes,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return payload


def _evaluate_candidate(
    *,
    traces: Mapping[int, Mapping[str, torch.Tensor]],
    episode_lengths: Mapping[int, int],
    statistics: Mapping[str, Any],
    episodes: Sequence[int],
    selector_config: Mapping[str, Any],
) -> dict[str, Any]:
    group_lengths: list[int] = []
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
        state = ControlInformationBoundaryState(
            statistics=statistics, **dict(selector_config)
        )
        detector_frames = list(range(0, int(episode_lengths[int(episode)]), 4))
        events = []
        z_by_frame = {}
        for frame in detector_frames:
            value = None if frame < 16 else by_frame[frame]
            if value is not None:
                z_by_frame[frame] = contextual_information_z(
                    value, frame=frame, statistics=statistics
                )
            event = state.update(frame=frame, information=value)
            if event is not None:
                events.append(asdict(event))
        tail = state.finalize(frame=detector_frames[-1])
        if tail is not None:
            events.append(asdict(tail))
        aligned = align_detector_events(
            events,
            episode_frames=int(episode_lengths[int(episode)]),
            replan_stride=16,
            minimum_segment_decisions=2,
            maximum_segment_decisions=8,
        )
        lengths = [
            right - left
            for left, right in zip(aligned["boundaries"], aligned["boundaries"][1:])
        ]
        group_lengths.extend(lengths)
        total_groups += len(lengths)
        target_groups += (aligned["decision_count"] + 3) // 4
        kept_learned_frames = []
        kept_reasons = []
        for row in aligned["confirmation_to_decision"]:
            if row["status"] != "kept":
                continue
            reason = row["reason"]
            kept_reasons.append(reason)
            if reason == "counterfactual_control_information":
                learned += 1
                kept_learned_frames.append(int(row["confirmation_frame"]))
            elif reason == "forced_maximum":
                forced += 1
        excess = {
            frame: max(0.0, z - float(selector_config["drift"]))
            for frame, z in z_by_frame.items()
        }
        total_excess += sum(excess.values())
        captured_frames = {
            frame
            for boundary in kept_learned_frames
            for frame in (boundary - 4, boundary, boundary + 4)
        }
        captured_excess += sum(
            value for frame, value in excess.items() if frame in captured_frames
        )
        episode_reports[str(episode)] = {
            "boundaries": aligned["boundaries"],
            "segment_lengths": lengths,
            "kept_reasons": kept_reasons,
        }
    boundary_total = learned + forced
    sorted_lengths = torch.tensor(sorted(group_lengths), dtype=torch.float32)
    return {
        "group_count_ratio": total_groups / target_groups,
        "learned_boundary_fraction": learned / max(boundary_total, 1),
        "captured_excess_information": captured_excess / max(total_excess, 1e-12),
        "forced_boundary_fraction": forced / max(boundary_total, 1),
        "p95_group_length": float(torch.quantile(sorted_lengths, 0.95)),
        "group_count": total_groups,
        "fixed_group_size_four_count": target_groups,
        "learned_boundary_count": learned,
        "forced_boundary_count": forced,
        "episodes": episode_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", required=True)
    parser.add_argument("--initialization-manifest", required=True)
    parser.add_argument("--feature-bank-manifest", required=True)
    parser.add_argument("--pca", required=True)
    parser.add_argument("--feature-predictor", required=True)
    parser.add_argument("--control-probe", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--boundary-source", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    output = Path(args.output_root).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite selector calibration: {output}")
    trace_path = Path(args.traces).resolve()
    trace_payload = torch.load(trace_path, map_location="cpu", weights_only=False)
    split = locked_split()
    if (
        trace_payload.get("schema_version") != TRACE_SCHEMA
        or trace_payload.get("complete") is not True
        or trace_payload.get("traced_episodes") != split["trace_episodes"]
    ):
        raise ValueError("candidate traces must contain exactly episodes 0-39")
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
        raise FileExistsError(f"stale selector calibration temporary: {temporary}")
    temporary.mkdir(parents=True)
    statistics_path = temporary / "contextual_statistics.pt"
    torch.save(statistics, statistics_path)
    candidates = []
    for threshold, drift, decay, min_units, max_units in itertools.product(
        THRESHOLDS, DRIFTS, DECAYS, MIN_UNITS, MAX_UNITS
    ):
        config = {
            "threshold": threshold,
            "drift": drift,
            "decay": decay,
            "detector_stride": 4,
            "min_units": min_units,
            "max_units": max_units,
            "initial_group_start": 0,
        }
        metrics = _evaluate_candidate(
            traces=traces,
            episode_lengths=episode_lengths,
            statistics=statistics,
            episodes=split["calibration_episodes"],
            selector_config=config,
        )
        candidates.append({"selector_config": config, "metrics": metrics})
    selected = select_calibrated_candidate(candidates)
    report = {
        "schema_version": "putback_control_information_calibration_v1",
        "split": split,
        "grid": {
            "thresholds": list(THRESHOLDS),
            "drifts": list(DRIFTS),
            "decays": list(DECAYS),
            "min_units": list(MIN_UNITS),
            "max_units": list(MAX_UNITS),
        },
        "candidate_count": len(candidates),
        "selected": selected,
        "candidates": candidates,
        "trace_sha256": _sha256(trace_path),
    }
    report_path = temporary / "calibration_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    dependencies = {
        "initialization_manifest": args.initialization_manifest,
        "feature_bank_manifest": args.feature_bank_manifest,
        "pca": args.pca,
        "feature_predictor": args.feature_predictor,
        "control_probe": args.control_probe,
        "contextual_statistics": statistics_path,
        "normalization": args.normalization,
        "boundary_source": args.boundary_source,
    }
    lock_candidate(
        temporary / "locked_candidate.json",
        candidate=selected,
        dependency_paths=dependencies,
    )
    temporary.replace(output)
    print(json.dumps(selected, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
