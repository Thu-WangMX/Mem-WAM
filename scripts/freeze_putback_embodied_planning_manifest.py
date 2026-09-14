from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch

from fastwam.memory.dynamic_surprise import SCHEMA_VERSION, validate_episode_segments
from fastwam.memory.embodied_information_boundary import (
    EmbodiedInformationBoundaryState,
    contextual_residual_z,
)
from fastwam.memory.planning_aligned_manifest import align_detector_events
from scripts.evaluate_putback_embodied_information_candidate import _control


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _trace_map(payload):
    return {
        int(frame): residual
        for frame, residual in zip(payload["frame_indices"], payload["residuals"])
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual-traces-0-39", required=True)
    parser.add_argument("--residual-traces-40-49", required=True)
    parser.add_argument("--contextual-stats", required=True)
    parser.add_argument("--locked-candidate", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen manifest: {output}")

    lock_path = Path(args.locked_candidate).resolve()
    lock = json.loads(lock_path.read_text())
    if lock.get("schema_version") != "putback_locked_embodied_information_candidate_v1":
        raise ValueError("locked embodied candidate schema mismatch")
    stats_path = Path(args.contextual_stats).resolve()
    if _sha(stats_path) != lock["hashes"]["contextual_stats_sha256"]:
        raise ValueError("contextual statistics do not match candidate lock")
    stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    early_path = Path(args.residual_traces_0_39).resolve()
    late_path = Path(args.residual_traces_40_49).resolve()
    early = torch.load(early_path, map_location="cpu", weights_only=False)["episodes"]
    late_payload = torch.load(late_path, map_location="cpu", weights_only=False)
    late = late_payload.get("episodes", late_payload)
    if sorted(int(key) for key in early) != list(range(40)):
        raise ValueError("early residual artifact must contain episodes 0-39")
    if sorted(int(key) for key in late) != list(range(40, 50)):
        raise ValueError("late residual artifact must contain episodes 40-49")

    dataset_root = Path(args.dataset_root).resolve()
    temporary = output.with_name(output.name + ".tmp")
    temporary.mkdir(parents=True)
    episodes_dir = temporary / "episodes"
    episodes_dir.mkdir()
    episode_paths = {}
    segment_lengths = Counter()
    status_counts = Counter()
    reason_counts = Counter()
    for episode in range(50):
        trace = early[episode] if episode < 40 else late[episode]
        residuals = _trace_map(trace)
        _, proprio = _control(dataset_root, episode)
        detector_frames = list(range(0, len(proprio), 4))
        state = EmbodiedInformationBoundaryState(**lock["selector_config"])
        events = []
        for frame in detector_frames:
            standardized = None
            if frame >= 16:
                if frame not in residuals:
                    raise KeyError(f"episode {episode} lacks residual frame {frame}")
                standardized = contextual_residual_z(
                    residuals[frame], frame=frame, stats=stats
                )
            event = state.update(
                frame=frame,
                proprio=proprio[frame],
                standardized_residual=standardized,
            )
            if event is not None:
                events.append(event.__dict__)
        tail = state.finalize(frame=detector_frames[-1])
        if tail is not None:
            events.append(tail.__dict__)
        aligned = align_detector_events(events, episode_frames=len(proprio))
        payload = {
            "episode": episode,
            "decision_count": aligned["decision_count"],
            "boundaries": aligned["boundaries"],
            "reasons": aligned["reasons"],
            "alignment": aligned["confirmation_to_decision"],
            "retroactive_boundary_count": aligned["retroactive_boundary_count"],
        }
        segments = validate_episode_segments(payload)
        for left, right in segments:
            segment_lengths[right - left] += 1
        for row in aligned["confirmation_to_decision"]:
            status_counts[row["status"]] += 1
            if row["status"] == "kept":
                reason_counts[row["reason"]] += 1
        relative = f"episodes/episode_{episode:03d}.json"
        (temporary / relative).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        episode_paths[str(episode)] = relative

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "task": "putback",
        "episode_count": 50,
        "boundary_step": -1,
        "replan_stride": 16,
        "selector": "locked_embodied_information_v1",
        "locked_candidate_sha256": _sha(lock_path),
        "residual_traces_0_39_sha256": _sha(early_path),
        "residual_traces_40_49_sha256": _sha(late_path),
        "contextual_stats_sha256": _sha(stats_path),
        "alignment_policy": "ceil_confirmation_to_next_stride16_min2_max8",
        "segment_length_histogram": dict(sorted(segment_lengths.items())),
        "alignment_status_counts": dict(status_counts),
        "kept_reason_counts": dict(reason_counts),
        "retroactive_boundary_count": 0,
    }
    (temporary / "manifest.json").write_text(
        json.dumps({"metadata": metadata, "episodes": episode_paths}, indent=2,
                   sort_keys=True) + "\n"
    )
    temporary.replace(output)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
