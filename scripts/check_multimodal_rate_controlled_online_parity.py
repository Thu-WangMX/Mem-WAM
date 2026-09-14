#!/usr/bin/env python3
"""Replay cached observations and prove online selector/manifest parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from fastwam.memory.multimodal_rate_controlled import MultimodalRateManifestStore
from fastwam.memory.multimodal_rate_controlled_online import (
    OnlineMultimodalRateSegmenter,
)


def load_states(root: Path, episode: int) -> np.ndarray:
    path = root / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
    table = pq.read_table(path, columns=["observation.state"])
    return np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--lerobot-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()

    store = MultimodalRateManifestStore(
        args.manifest,
        expected_episode_count=args.episodes,
        expected_task=args.task,
    )
    source_manifest = json.loads((args.latent_root / "manifest.json").read_text())
    source_episodes = source_manifest["episodes"]
    scale = store.metadata["joint_delta_scale"]
    gate = float(store.metadata["motion_gate"])
    total_segments = 0
    for episode in range(args.episodes):
        relative = source_episodes.get(f"{args.task}/{episode}")
        if not isinstance(relative, str):
            raise KeyError(f"missing latent source for {args.task}/{episode}")
        payload = torch.load(args.latent_root / relative, map_location="cpu", weights_only=False)
        latents = torch.as_tensor(payload["latents"])
        frames = [int(value) for value in payload["frame_indices"].tolist()]
        states = load_states(args.lerobot_root, episode)
        runtime = OnlineMultimodalRateSegmenter(
            joint_delta_scale=scale,
            motion_gate=gate,
        )
        actual = []
        next_frame = 0
        for decision, (frame, latent) in enumerate(zip(frames, latents)):
            while next_frame <= frame:
                runtime.observe_frame(frame=next_frame, state=states[next_frame])
                next_frame += 1
            closed = runtime.arrive_planning(decision=decision, latent=latent)
            if closed is not None:
                actual.append(closed)
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
