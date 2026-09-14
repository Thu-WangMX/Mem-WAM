#!/usr/bin/env python3
"""Build a task-generic train-free latent-kernel manifest from frozen VAE latents."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path

import torch

from fastwam.memory.latent_kernel_regime import SCHEMA_VERSION, validate_segments
from fastwam.memory.latent_kernel_regime_online import OnlineLatentKernelRegimeSegmenter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()

    latent_root = Path(args.latent_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite latent-kernel manifest: {output}")
    source_manifest = json.loads((latent_root / "manifest.json").read_text())
    metadata = source_manifest.get("metadata", {})
    expected_source = {
        "schema_version": "fastwam_full_kv_continuous_episode_vae_latents_v4",
        "complete": True,
        "replan_stride": 16,
        "episode_count": int(args.episodes),
        "latent_shape": [48, 1, 24, 20],
    }
    for key, want in expected_source.items():
        if metadata.get(key) != want:
            raise ValueError(f"source latent {key}: expected {want!r}, got {metadata.get(key)!r}")

    episode_mapping = source_manifest.get("episodes", {})
    output.parent.mkdir(parents=True, exist_ok=True)
    length_histogram: Counter[int] = Counter()
    reason_histogram: Counter[str] = Counter()
    with tempfile.TemporaryDirectory(
        dir=output.parent, prefix=f".{output.name}.building."
    ) as temporary:
        staging = Path(temporary)
        (staging / "episodes").mkdir()
        mapping: dict[str, str] = {}
        for episode in range(args.episodes):
            source_relative = episode_mapping.get(f"{args.task}/{episode}")
            if not isinstance(source_relative, str):
                raise KeyError(f"missing frozen latent for {args.task}/{episode}")
            source = torch.load(
                latent_root / source_relative, map_location="cpu", weights_only=False
            )
            latents = torch.as_tensor(source["latents"])
            frame_indices = [int(value) for value in source["frame_indices"].tolist()]
            if latents.ndim != 5 or tuple(latents.shape[1:]) != (48, 1, 24, 20):
                raise ValueError(f"episode {episode} latent shape is {tuple(latents.shape)}")
            if len(frame_indices) != len(latents):
                raise ValueError(f"episode {episode} frame/latent count mismatch")

            runtime = OnlineLatentKernelRegimeSegmenter()
            closed: list[tuple[int, int, int]] = []
            for decision, latent in enumerate(latents):
                runtime.observe(decision=decision, latent=latent)
                close_range = runtime.arrive_planning(decision=decision)
                if close_range is not None:
                    closed.append((close_range[0], close_range[1], decision))
            scores, ranks = runtime._scores_and_ranks()
            source_ends = [None if score is None else index + 1 for index, score in enumerate(scores)]
            segments = []
            for start, end, confirmed_at in closed:
                reason = (
                    "stable_max_length"
                    if end - start == runtime.max_segment
                    else "multires_latent_kernel_regime"
                )
                row = {
                    "start": start,
                    "end": end,
                    "confirmed_at": confirmed_at,
                    "reason": reason,
                    "score": scores[end],
                    "causal_rank": ranks[end],
                }
                segments.append(row)
                length_histogram.update([end - start])
                reason_histogram.update([reason])
            payload = {
                "schema_version": SCHEMA_VERSION,
                "episode": episode,
                "decision_frame_indices": frame_indices,
                "decision_count": len(latents),
                "scores": scores,
                "score_source_end": source_ends,
                "causal_ranks": ranks,
                "segments": segments,
                "uncompressed_tail_start": closed[-1][1] if closed else 2,
            }
            validate_segments(payload)
            relative = f"episodes/episode_{episode:03d}.json"
            (staging / relative).write_text(json.dumps(payload, indent=2) + "\n")
            mapping[str(episode)] = relative

        frozen_metadata = {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "task": str(args.task),
            "episode_count": int(args.episodes),
            "selector_input": "frozen_continuous_causal_vae_latents_only",
            "uses_action": False,
            "uses_proprioception": False,
            "uses_reward": False,
            "uses_task_checkpoint": False,
            "fitted_parameters": None,
            "replan_stride": 16,
            "anchor_frames": 2,
            "recent_frames": 4,
            "min_segment": 4,
            "max_segment": 8,
            "memory_tokens": 8,
            "kernel_window_each_side": 2,
            "views": [
                "appearance_flat",
                "scene_channel_mean",
                "local_channel_direction",
            ],
            "fusion": "unweighted_arithmetic_mean",
            "rank_threshold": 0.75,
            "history_window": 8,
            "segment_length_histogram": dict(sorted(length_histogram.items())),
            "reason_histogram": dict(sorted(reason_histogram.items())),
        }
        (staging / "manifest.json").write_text(
            json.dumps({"metadata": frozen_metadata, "episodes": mapping}, indent=2)
            + "\n"
        )
        staging.rename(output)
    print(json.dumps(frozen_metadata, indent=2))


if __name__ == "__main__":
    main()
