#!/usr/bin/env python3
"""Replay qpos and prove physical settle/rate-debt online/manifest parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fastwam.memory.physical_settle_rate_debt import PhysicalSettleRateManifestStore
from fastwam.memory.physical_settle_rate_debt_online import (
    OnlinePhysicalSettleRateSegmenter,
)

from build_physical_settle_rate_debt_manifest import load_states


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lerobot-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()
    store = PhysicalSettleRateManifestStore(
        args.manifest,
        expected_episode_count=args.episodes,
        expected_task=args.task,
    )
    stride = int(store.metadata["replan_stride"])
    total_segments = 0
    for episode in range(args.episodes):
        states = load_states(args.lerobot_root, episode)[::stride]
        runtime = OnlinePhysicalSettleRateSegmenter()
        actual = []
        for decision, state in enumerate(states):
            segment = runtime.arrive_planning(decision=decision, state=state)
            if segment is not None:
                actual.append(segment)
        expected = list(store.segments_for_episode(episode))
        if actual != expected:
            raise RuntimeError(
                f"episode {episode} online mismatch:\nexpected={expected}\nactual={actual}"
            )
        total_segments += len(actual)
    print(
        json.dumps(
            {
                "status": "ok",
                "task": args.task,
                "episodes": args.episodes,
                "segments": total_segments,
            }
        )
    )


if __name__ == "__main__":
    main()
