from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms.functional as transforms_F

from fastwam.datasets.lerobot.full_kv_dataset import SCHEMA_VERSION
from fastwam.models.wan22.helpers.loader import (
    _load_registered_model,
    _resolve_configs,
)


CAMERAS = ("head_camera", "left_camera", "right_camera")


def _distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank


def _decode_rgb(encoded) -> np.ndarray:
    if isinstance(encoded, np.ndarray) and encoded.ndim == 3:
        image = encoded
    else:
        image = cv2.imdecode(
            np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if image is None:
            raise ValueError("OpenCV failed to decode an RMBench image")
        # RMBench passes the simulator RGB array directly to cv2.imencode.
        # The encode/decode round trip therefore preserves the simulator's
        # numeric RGB channel order. Applying BGR2RGB here would introduce an
        # extra red/blue swap and make cached training latents disagree with
        # online simulator observations.
    return image


def _mosaic(handle: h5py.File, frame_index: int) -> torch.Tensor:
    images = []
    for camera in CAMERAS:
        image = _decode_rgb(
            handle[f"observation/{camera}/rgb"][frame_index]
        )
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
        tensor = tensor.float().div_(255.0)
        # Match configs/data/rmbench_putback_full_kv.yaml exactly: the
        # official FastWAM processor first resizes every camera to 240x320.
        tensor = transforms_F.resize(
            tensor,
            size=[240, 320],
            interpolation=transforms_F.InterpolationMode.BILINEAR,
            antialias=True,
        )
        images.append(tensor)
    head = transforms_F.resize(
        images[0],
        size=[256, 320],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    left = transforms_F.resize(
        images[1],
        size=[128, 160],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    right = transforms_F.resize(
        images[2],
        size=[128, 160],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    wrists = torch.cat([left, right], dim=2)
    # MemoryWAM layout: wrists on top, head camera on the bottom.
    mosaic = torch.cat([wrists, head], dim=1)
    return mosaic.mul_(2.0).sub_(1.0)


def _encode_batch(vae, videos: list[torch.Tensor], device: torch.device):
    """Encode RGB videos shaped [3,T,H,W] with Wan's causal temporal VAE."""
    videos = [
        video.to(device=device, dtype=torch.bfloat16)
        for video in videos
    ]
    encoded = vae.encode(videos, device=device, tiled=False)
    if isinstance(encoded, torch.Tensor):
        encoded = [item for item in encoded]
    if len(encoded) != len(videos):
        raise RuntimeError(
            f"VAE returned {len(encoded)} samples for batch {len(videos)}"
        )
    return [
        latent.detach().to(device="cpu", dtype=torch.bfloat16)
        for latent in encoded
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lerobot-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--replan-stride", type=int, default=16)
    parser.add_argument("--temporal-subframes", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional deterministic prefix for a smoke test.",
    )
    args = parser.parse_args()
    if args.temporal_subframes <= 0:
        raise ValueError("`--temporal-subframes` must be positive")
    if args.replan_stride % args.temporal_subframes != 0:
        raise ValueError(
            "`--replan-stride` must be divisible by `--temporal-subframes`"
        )
    temporal_step = args.replan_stride // args.temporal_subframes

    rank, world_size, local_rank = _distributed()
    device = torch.device(f"cuda:{local_rank}")
    root = Path(args.lerobot_root).resolve()
    output = Path(args.output).resolve()
    shard_root = output / "shards" / f"shard_{rank}"
    shard_root.mkdir(parents=True, exist_ok=True)

    _, _, vae_config, _ = _resolve_configs(
        model_id="Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id="Wan-AI/Wan2.1-T2V-1.3B",
        redirect_common_files=True,
    )
    vae_config.download_if_necessary()
    vae = _load_registered_model(
        vae_config.path,
        "wan_video_vae",
        torch_dtype=torch.bfloat16,
        device=str(device),
    ).eval()

    episodes = [
        json.loads(line)
        for line in (root / "meta" / "episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if args.max_episodes is not None:
        if args.max_episodes <= 0:
            raise ValueError("`--max-episodes` must be positive")
        episodes = episodes[: args.max_episodes]
    local_manifest: dict[str, str] = {}
    for row in episodes[rank::world_size]:
        episode_index = int(row["episode_index"])
        length = int(row["length"])
        raw_file = Path(row["raw_file_name"])
        frame_indices = list(range(0, length, int(args.replan_stride)))
        if not frame_indices:
            raise RuntimeError(f"episode {episode_index}: no decision frames")
        with h5py.File(raw_file, "r") as handle:
            # Match LingBotVA's temporal contract: encode the complete episode
            # as one causal VAE stream. Sampling every four simulator frames
            # and Wan's 4x temporal compression yields exactly one latent at
            # each 16-frame policy decision point, while preserving VAE state
            # across decision boundaries.
            sampled_frame_indices = list(
                range(0, frame_indices[-1] + 1, temporal_step)
            )
            episode_video = torch.stack(
                [_mosaic(handle, index) for index in sampled_frame_indices],
                dim=1,
            )
            encoded_episode = _encode_batch(
                vae,
                [episode_video],
                device,
            )[0]
            if encoded_episode.shape[1] != len(frame_indices):
                raise RuntimeError(
                    f"episode {episode_index}: expected {len(frame_indices)} "
                    "continuous causal latents for "
                    f"{len(sampled_frame_indices)} sampled RGB frames, got "
                    f"{tuple(encoded_episode.shape)}"
                )
        payload = {
            "frame_indices": torch.tensor(frame_indices, dtype=torch.int64),
            "latents": encoded_episode.permute(1, 0, 2, 3).unsqueeze(2),
        }
        expected = (len(frame_indices), 48, 1, 24, 20)
        if tuple(payload["latents"].shape) != expected:
            raise RuntimeError(
                f"episode {episode_index}: expected {expected}, "
                f"got {tuple(payload['latents'].shape)}"
            )
        relative = Path("shards") / f"shard_{rank}" / (
            f"episode_{episode_index:06d}.pt"
        )
        torch.save(payload, output / relative)
        local_manifest[f"{root.name}/{episode_index}"] = str(relative)
        print(
            f"rank={rank} episode={episode_index} observations={len(frame_indices)}",
            flush=True,
        )

    if world_size > 1:
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local_manifest)
        dist.barrier()
    else:
        gathered = [local_manifest]
    if rank == 0:
        merged: dict[str, str] = {}
        for shard in gathered:
            overlap = set(merged).intersection(shard)
            if overlap:
                raise RuntimeError(f"Duplicate episodes across shards: {overlap}")
            merged.update(shard)
        vae_path = Path(vae_config.path)
        if not vae_path.is_file():
            raise RuntimeError(f"Resolved official VAE is not a file: {vae_path}")
        manifest = {
            "metadata": {
                "schema_version": SCHEMA_VERSION,
                "complete": True,
                "replan_stride": int(args.replan_stride),
                "temporal_subframes": int(args.temporal_subframes),
                "temporal_subframe_stride": int(temporal_step),
                "latent_shape": [48, 1, 24, 20],
                "latent_dtype": "bfloat16",
                "episode_count": len(merged),
                "vae_model_sha256": _sha256(vae_path),
                "mosaic": "wrists_top_head_bottom_384x320",
                "color_contract": "simulator_rgb_preserved",
                "encoding": "continuous_episode_stride4_causal_vae",
            },
            "episodes": dict(sorted(merged.items())),
        }
        with (output / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        print(json.dumps(manifest["metadata"], indent=2), flush=True)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
