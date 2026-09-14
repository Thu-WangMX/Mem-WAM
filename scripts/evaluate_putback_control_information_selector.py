"""Fail-closed paper gate for the locked PutBack control-information selector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


GATE_SCHEMA = "putback_control_information_selector_gate_v1"


def _number(section: Mapping[str, Any], key: str) -> float:
    try:
        value = float(section[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"quality report lacks numeric field {key}") from error
    return value


def evaluate_gate(report: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate every predeclared selector gate without relaxing thresholds."""

    try:
        shortcut = report["shortcut"]
        segments = report["segments"]
        replay = report["replay"]
        semantic = report["semantic"]
        visual = report["visual"]
        deployment = report["deployment"]
        latency = report["latency"]
    except (KeyError, TypeError) as error:
        raise ValueError("quality report is missing a required section") from error

    wam_mae = _number(shortcut, "wam_probe_mae")
    proprio_mae = _number(shortcut, "proprio_only_mae")
    schedule_mae = _number(shortcut, "schedule_baseline_mae")
    learned_fraction = _number(segments, "learned_boundary_fraction")
    semantic_f1 = _number(semantic, "control_information_f1")
    raw_f1 = _number(semantic, "raw_residual_f1")
    gripper_f1 = _number(semantic, "gripper_selector_f1")
    latency_p95 = _number(latency, "selector_latency_p95_ms")
    latency_limit = _number(latency, "locked_latency_limit_ms")
    peak_bytes = _number(latency, "peak_allocated_bytes")
    total_bytes = _number(latency, "device_total_bytes")
    clauses = {
        "probe_beats_proprio": wam_mae < 0.95 * proprio_mae,
        "probe_beats_schedule": wam_mae < 0.95 * schedule_mae,
        "dynamic_patterns": int(segments["distinct_segment_patterns"]) >= 5,
        "learned_majority": learned_fraction >= 0.50,
        "no_retroactive": int(replay["retroactive_boundary_count"]) == 0,
        "replay_all_episodes": int(replay["matching_episodes"]) == 50,
        "semantic_review_exists": int(semantic["reviewed_transition_count"]) > 0,
        "semantic_better_than_raw": semantic_f1 >= raw_f1 + 0.05,
        "semantic_not_worse_than_gripper": semantic_f1 >= gripper_f1 - 0.05,
        "development_video_reviewed": bool(visual["development_reviewed"]),
        "four_heldout_videos": int(visual["heldout_rendered"]) == 4,
        "strict_online_exercised": int(deployment["selector_observations"]) > 0,
        "no_fallback": int(deployment["fallback_count"]) == 0,
        "no_old_selector": int(deployment["old_selector_count"]) == 0,
        "online_latency": latency_p95 <= latency_limit,
        "gpu_memory_headroom": total_bytes > 0 and peak_bytes <= 0.95 * total_bytes,
    }
    return {
        "schema_version": GATE_SCHEMA,
        "pass": all(clauses.values()),
        "clauses": clauses,
        "measurements": dict(report),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurements", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    measurements_path = Path(args.measurements).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite selector gate: {output}")
    result = evaluate_gate(json.loads(measurements_path.read_text()))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
