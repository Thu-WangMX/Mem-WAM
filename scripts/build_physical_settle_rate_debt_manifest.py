#!/usr/bin/env python3
"""Build a frozen train-free physical settle/rate-debt L4--8/K8 manifest."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from fastwam.memory.physical_settle_rate_debt import SCHEMA_VERSION
from fastwam.memory.physical_settle_rate_debt_online import (
    OnlinePhysicalSettleRateSegmenter,
)


def load_states(root: Path, episode: int) -> np.ndarray:
    path = root / "data" / f"chunk-{episode // 1000:03d}" / f"episode_{episode:06d}.parquet"
    table = pq.read_table(path, columns=["observation.state"])
    return np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lerobot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--replan-stride", type=int, default=16)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    lengths: Counter[int] = Counter()
    reasons: Counter[str] = Counter()
    per_episode: Counter[int] = Counter()
    token_exposure = 0
    with tempfile.TemporaryDirectory(
        dir=args.output.parent, prefix=f".{args.output.name}.building."
    ) as temporary:
        stage = Path(temporary)
        (stage / "episodes").mkdir()
        mapping: dict[str, str] = {}
        for episode in range(args.episodes):
            states = load_states(args.lerobot_root, episode)
            decision_states = states[:: args.replan_stride]
            runtime = OnlinePhysicalSettleRateSegmenter()
            segments = []
            for decision, state in enumerate(decision_states):
                segment = runtime.arrive_planning(decision=decision, state=state)
                if segment is not None:
                    segments.append(segment)
            lengths.update(segment.length for segment in segments)
            reasons.update(segment.reason for segment in segments)
            per_episode[len(segments)] += 1
            token_exposure += sum(
                8 * max(0, len(decision_states) - segment.confirmed_at)
                for segment in segments
            )
            payload = {
                "schema_version": SCHEMA_VERSION,
                "episode": episode,
                "decision_count": len(decision_states),
                "segments": [asdict(segment) for segment in segments],
                "uncompressed_tail_start": segments[-1].end if segments else 2,
            }
            relative = f"episodes/episode_{episode:03d}.json"
            (stage / relative).write_text(json.dumps(payload, indent=2) + "\n")
            mapping[str(episode)] = relative

        metadata = {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "task": args.task,
            "episode_count": args.episodes,
            "replan_stride": args.replan_stride,
            "anchor_frames": 2,
            "recent_frames": 4,
            "reader_raw_tail": "complete_open_segment_suffix",
            "fixed_reader_recent_frames": False,
            "maximum_raw_tail_frames": 8,
            "maximum_confirmation_delay_frames": 4,
            "min_segment": 4,
            "max_segment": 8,
            "memory_tokens": 8,
            "history_window": 8,
            "event_threshold": 0.8,
            "target_mean_segment": 7.0,
            "maximum_rate_debt": 5.0,
            "selector": "causal_selfcal_move_to_settle_or_gripper",
            "selector_inputs": ["observation.state"],
            "uses_action": False,
            "uses_reward": False,
            "uses_task_checkpoint": False,
            "uses_cross_episode_fit": False,
            "segment_length_histogram": {
                str(key): value for key, value in sorted(lengths.items())
            },
            "reason_histogram": dict(sorted(reasons.items())),
            "segments_per_episode_histogram": {
                str(key): value for key, value in sorted(per_episode.items())
            },
            "compressed_k8_token_exposure": token_exposure,
        }
        (stage / "manifest.json").write_text(
            json.dumps({"metadata": metadata, "episodes": mapping}, indent=2) + "\n"
        )
        stage.rename(args.output)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
