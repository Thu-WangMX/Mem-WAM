from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import torch

from fastwam.memory.embodied_information_boundary import (
    EmbodiedInformationBoundaryState,
    contextual_residual_z,
    fit_contextual_residual_statistics,
)
from scripts.prepare_putback_phase_annotation import propose_gripper_events


def match_event_frames(predicted, target, *, tolerance: int):
    pairs = sorted(
        (abs(int(left) - int(right)), i, j)
        for i, left in enumerate(predicted)
        for j, right in enumerate(target)
        if abs(int(left) - int(right)) <= tolerance
    )
    used_pred, used_target = set(), set()
    for _, i, j in pairs:
        if i not in used_pred and j not in used_target:
            used_pred.add(i); used_target.add(j)
    tp = len(used_pred); fp = len(predicted) - tp; fn = len(target) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "counts": {"true_positive": tp, "false_positive": fp, "false_negative": fn},
        "precision": precision, "recall": recall, "f1": f1,
    }


def _control(dataset_root: Path, episode: int):
    table = pq.read_table(
        dataset_root / "data" / "chunk-000" / f"episode_{episode:06d}.parquet",
        columns=["action", "observation.state"],
    )
    return (
        torch.tensor(table["action"].to_pylist(), dtype=torch.float32),
        torch.tensor(table["observation.state"].to_pylist(), dtype=torch.float32),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual-artifact", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", default="30-39")
    parser.add_argument("--information-budget", type=float, default=120.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--clip-z", type=float, default=8.0)
    parser.add_argument("--min-information-units", type=int, default=4)
    parser.add_argument("--max-units", type=int, default=24)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite candidate evaluation: {output}")
    left, right = (int(value) for value in args.episodes.split("-", 1))
    episodes = list(range(left, right + 1))
    if episodes != list(range(30, 40)):
        raise ValueError("Candidate 3 development is locked to episodes 30-39")
    artifact = torch.load(args.residual_artifact, map_location="cpu", weights_only=False)
    if artifact.get("mode") != "visual_action" or sorted(artifact["episodes"]) != list(range(40)):
        raise ValueError("Candidate 3 requires V+A residual traces for episodes 0-39")
    stats = fit_contextual_residual_statistics(
        artifact["episodes"], episodes=range(30), depth_cap=4
    )
    config = {
        "information_budget": args.information_budget, "top_k": args.top_k,
        "clip_z": args.clip_z, "detector_stride": 4,
        "min_information_units": args.min_information_units,
        "max_units": args.max_units, "initial_group_start": 0,
        "gripper_dimensions": [6, 13], "gripper_threshold": .5,
    }
    dataset_root = Path(args.dataset_root).resolve()
    all_pred, all_target = [], []
    episode_reports = {}; reason_counts = Counter(); length_histogram = Counter()
    for episode in episodes:
        trace = artifact["episodes"][episode]
        residual_by_frame = {
            int(frame): row for frame, row in zip(trace["frame_indices"], trace["residuals"])
        }
        actions, proprio = _control(dataset_root, episode)
        detector_frames = list(range(0, len(proprio), 4))
        target = [
            row["frame"] for row in propose_gripper_events(
                actions[detector_frames].numpy(), detector_frames
            )
        ]
        state = EmbodiedInformationBoundaryState(**config)
        events = []
        for frame in detector_frames:
            standardized = None
            if frame >= 16:
                if frame not in residual_by_frame:
                    raise RuntimeError(f"episode {episode} lacks residual frame {frame}")
                standardized = contextual_residual_z(
                    residual_by_frame[frame], frame=frame, stats=stats
                )
            event = state.update(
                frame=frame, proprio=proprio[frame], standardized_residual=standardized
            )
            if event is not None:
                events.append(event)
        tail = state.finalize(frame=detector_frames[-1])
        if tail is not None:
            events.append(tail)
        predicted = [
            event.confirmation_frame for event in events
            if event.reason in {"embodied_gripper_transition", "wam_information_budget"}
        ]
        metrics = match_event_frames(predicted, target, tolerance=4)
        all_pred.extend((episode, frame) for frame in predicted)
        all_target.extend((episode, frame) for frame in target)
        for event in events:
            reason_counts[event.reason] += 1
            length_histogram[event.group_end - event.group_start] += 1
        episode_reports[str(episode)] = {
            "proxy_metrics": metrics, "target_control_transitions": target,
            "events": [event.__dict__ for event in events],
        }
    totals = Counter()
    for row in episode_reports.values(): totals.update(row["proxy_metrics"]["counts"])
    tp, fp, fn = totals["true_positive"], totals["false_positive"], totals["false_negative"]
    precision = tp/(tp+fp); recall=tp/(tp+fn); f1=2*precision*recall/(precision+recall)
    report = {
        "schema_version": "putback_embodied_information_candidate_dev_v1",
        "split": "development_30_39", "label_type": "control_transition_proxy_not_human_semantics",
        "selector_config": config,
        "proxy_micro": {"precision": precision, "recall": recall, "f1": f1,
                        "counts": dict(totals)},
        "reason_counts": dict(reason_counts),
        "group_length_histogram": dict(sorted(length_histogram.items())),
        "dynamic_length_count": len(length_histogram),
        "retroactive_boundary_count": 0,
        "episodes": episode_reports,
    }
    output.mkdir(parents=True)
    torch.save(stats, output / "contextual_statistics.pt")
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print(json.dumps({key: report[key] for key in (
        "proxy_micro", "reason_counts", "group_length_histogram",
        "dynamic_length_count", "retroactive_boundary_count")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

