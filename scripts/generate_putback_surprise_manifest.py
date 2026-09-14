"""Generate a complete phase-pinned PutBack surprise manifest on 1..8 GPUs."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from fastwam.datasets.lerobot.dynamic_surprise_dataset import (
    attach_dynamic_surprise_groups,
)
from fastwam.memory.dynamic_surprise import (
    SCHEMA_VERSION,
    DynamicSurpriseManifestStore,
    causal_boundaries,
    validate_episode_segments,
)
from fastwam.memory.dynamic_surprise_scorer import score_transition
from fastwam.memory.dynamic_surprise_scorer import transition_noise_seed
from fastwam.models.wan22.fastwam import _decode_dynamic_memory_groups
from fastwam.utils import misc
from fastwam.utils.config_resolvers import register_default_resolvers


def _dump_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


@torch.no_grad()
def _score_transition(model, sample, next_sample, *, sigma: float, seed: int):
    batch = default_collate([sample])
    memory_groups = _decode_dynamic_memory_groups(batch)
    inputs = model.build_inputs(batch)
    history = inputs["history_latents"].squeeze(3).permute(0, 2, 1, 3, 4).contiguous()
    actual = next_sample["history_latents"][-1].unsqueeze(0)
    return score_transition(
        model,
        history_latents=history,
        actual_latent=actual,
        action=inputs["action"],
        context=inputs["context"],
        context_mask=inputs["context_mask"],
        video_context=inputs["video_context"],
        video_context_mask=inputs["video_context_mask"],
        memory_groups=memory_groups,
        sigma=sigma,
        noise_seed=seed,
    )


def _distributed_context():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--boundary-step", required=True, type=int)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--initialization-only", action="store_true")
    source.add_argument("--checkpoint")
    parser.add_argument("--conditioning-manifest")
    parser.add_argument("--conditioning-boundary-step", type=int)
    parser.add_argument("--episode-count", type=int, default=50)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--min-segment", type=int, default=2)
    parser.add_argument("--max-segment", type=int, default=8)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument(
        "--noise-mode",
        choices=("per_transition", "per_episode"),
        default="per_transition",
    )
    parser.add_argument("--task-config", default="rmbench_putback_layerwise_k8_25k")
    args = parser.parse_args()

    if args.initialization_only != (args.boundary_step == -1):
        raise ValueError("initialization-only requires boundary-step=-1 and vice versa")
    if args.checkpoint and (
        args.conditioning_manifest is None or args.conditioning_boundary_step is None
    ):
        raise ValueError("checkpoint scoring requires its phase conditioning manifest")
    if not 0.0 < args.sigma <= 1.0:
        raise ValueError("sigma must lie in (0,1]")

    rank, world_size, local_rank = _distributed_context()
    repo = Path(args.repo).resolve()
    output = Path(args.output).resolve()
    episodes = list(range(args.episode_count))
    assigned = episodes[rank::world_size]
    if rank == 0:
        if output.exists():
            raise FileExistsError(f"refusing to overwrite manifest directory: {output}")
        (output / "episodes").mkdir(parents=True)
    if world_size > 1:
        dist.barrier()

    conditioning_store = None
    if args.conditioning_manifest:
        conditioning_store = DynamicSurpriseManifestStore(
            args.conditioning_manifest,
            expected_boundary_step=args.conditioning_boundary_step,
            expected_episode_count=args.episode_count,
        )

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    register_default_resolvers()
    misc.register_work_dir(str(output / ".work" / f"rank_{rank}"))
    with initialize_config_dir(version_base="1.3", config_dir=str(repo / "configs")):
        cfg = compose(
            config_name="train",
            overrides=[
                f"task={args.task_config}",
                f"output_dir={output / '.work' / f'rank_{rank}'}",
                "+data.train.episode_indices="
                + json.dumps(assigned, separators=(",", ":")),
            ],
        )
    dataset = instantiate(cfg.data.train)
    grouped = defaultdict(list)
    for index in range(len(dataset)):
        sample = dataset[index]
        episode = int(sample["episode_index"])
        if conditioning_store is not None:
            attach_dynamic_surprise_groups(
                sample, conditioning_store, replan_stride=16
            )
        grouped[episode].append(sample)

    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    model_cfg.load_text_encoder = False
    if args.initialization_only:
        model_cfg.skip_dit_load_from_pretrain = False
    else:
        model_cfg.skip_dit_load_from_pretrain = True
        model_cfg.action_dit_pretrained_path = None
    device = f"cuda:{local_rank}"
    model = instantiate(model_cfg, model_dtype=torch.bfloat16, device=device).eval()
    if args.checkpoint:
        model.load_checkpoint(str(Path(args.checkpoint).resolve()))

    for episode in assigned:
        started = time.perf_counter()
        samples = grouped[episode]
        rows = []
        for transition in range(len(samples) - 1):
            metrics = _score_transition(
                model,
                samples[transition],
                samples[transition + 1],
                sigma=args.sigma,
                seed=transition_noise_seed(
                    episode,
                    transition,
                    mode=args.noise_mode,
                ),
            )
            rows.append(
                {
                    "source_decision": transition,
                    "arrival_decision": transition + 1,
                    **metrics,
                }
            )
        segmentation = causal_boundaries(
            [row["score"] for row in rows],
            min_segment=args.min_segment,
            max_segment=args.max_segment,
            gamma=args.gamma,
            window=args.window,
        )
        for row, threshold in zip(rows, segmentation["thresholds"]):
            row["threshold"] = threshold
        payload = {
            "episode": episode,
            "decision_count": len(samples),
            "boundaries": segmentation["boundaries"],
            "reasons": segmentation["reasons"],
            "segment_lengths": segmentation["segment_lengths"],
            "scores": rows,
            "elapsed_seconds": time.perf_counter() - started,
        }
        validate_episode_segments(payload, decision_count=len(samples))
        _dump_json(output / "episodes" / f"episode_{episode:03d}.json", payload)
        print(
            json.dumps(
                {
                    "rank": rank,
                    "episode": episode,
                    "decisions": len(samples),
                    "segments": len(segmentation["segment_lengths"]),
                    "elapsed_seconds": payload["elapsed_seconds"],
                }
            ),
            flush=True,
        )

    if world_size > 1:
        dist.barrier()
    if rank == 0:
        episode_mapping = {
            str(episode): f"episodes/episode_{episode:03d}.json"
            for episode in episodes
        }
        missing = [
            relative for relative in episode_mapping.values() if not (output / relative).is_file()
        ]
        if missing:
            raise RuntimeError(f"manifest generation incomplete; missing {missing}")
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "task": "putback",
            "episode_count": args.episode_count,
            "boundary_step": args.boundary_step,
            "replan_stride": 16,
            "model_source": "initialization" if args.initialization_only else "checkpoint",
            "checkpoint": args.checkpoint,
            "conditioning_manifest": args.conditioning_manifest,
            "conditioning_boundary_step": args.conditioning_boundary_step,
            "seed": seed,
            "noise_mode": args.noise_mode,
            "sigma": args.sigma,
            "score": "0.7*normalized_l1+0.3*cosine_distance",
            "gamma": args.gamma,
            "min_segment": args.min_segment,
            "max_segment": args.max_segment,
            "window": args.window,
            "minimum_threshold_priors": min(3, args.window),
            "threshold_window_resets_at_boundary": False,
        }
        _dump_json(output / "manifest.json", {"metadata": metadata, "episodes": episode_mapping})
        DynamicSurpriseManifestStore(
            output,
            expected_boundary_step=args.boundary_step,
            expected_episode_count=args.episode_count,
        )
        print(json.dumps({"complete": True, "output": str(output)}), flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
