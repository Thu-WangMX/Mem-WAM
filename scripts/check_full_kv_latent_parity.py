from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import h5py
import numpy as np
import torch

from experiments.robotwin.fastwam_policy.deploy_policy import (
    WorldActionRobotWinPolicy,
)
from fastwam.models.wan22.fastwam import FastWAM
from fastwam.models.wan22.helpers.loader import (
    _load_registered_model,
    _resolve_configs,
)


CAMERAS = ("head_camera", "left_camera", "right_camera")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode_rgb(encoded) -> np.ndarray:
    if isinstance(encoded, np.ndarray) and encoded.ndim == 3:
        return encoded
    image = cv2.imdecode(
        np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if image is None:
        raise ValueError("OpenCV failed to decode an RMBench image")
    return image


def _observation(handle: h5py.File, frame_index: int) -> dict:
    return {
        "observation": {
            camera: {
                "rgb": _decode_rgb(
                    handle[f"observation/{camera}/rgb"][frame_index]
                )
            }
            for camera in CAMERAS
        }
    }


def _episode_row(root: Path, episode_index: int) -> dict:
    rows = [
        json.loads(line)
        for line in (root / "meta" / "episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    matches = [row for row in rows if int(row["episode_index"]) == episode_index]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one metadata row for episode {episode_index}, got {len(matches)}"
        )
    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lerobot-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--decisions", type=int, nargs="+", default=[0, 16, 32])
    parser.add_argument("--replan-stride", type=int, default=16)
    parser.add_argument("--temporal-subframes", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-abs-tol", type=float, default=0.02)
    parser.add_argument("--mean-abs-tol", type=float, default=0.002)
    args = parser.parse_args()

    if args.replan_stride % args.temporal_subframes != 0:
        raise ValueError("replan stride must be divisible by temporal subframes")
    temporal_step = args.replan_stride // args.temporal_subframes
    device = torch.device(args.device)
    root = Path(args.lerobot_root).resolve()
    cache_root = Path(args.cache_root).resolve()
    manifest = json.loads((cache_root / "manifest.json").read_text())
    episode_key = f"{root.name}/{args.episode}"
    payload = torch.load(
        cache_root / manifest["episodes"][episode_key],
        map_location="cpu",
        weights_only=True,
    )
    frame_to_position = {
        int(frame): index
        for index, frame in enumerate(payload["frame_indices"].tolist())
    }

    _, _, vae_config, _ = _resolve_configs(
        model_id="Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id="Wan-AI/Wan2.1-T2V-1.3B",
        redirect_common_files=True,
    )
    vae_config.download_if_necessary()
    vae_path = Path(vae_config.path)
    actual_sha = _sha256(vae_path)
    expected_sha = manifest["metadata"]["vae_model_sha256"]
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"VAE checkpoint mismatch: expected {expected_sha}, got {actual_sha}"
        )
    vae = _load_registered_model(
        vae_config.path,
        "wan_video_vae",
        torch_dtype=torch.bfloat16,
        device=str(device),
    ).eval()

    policy = object.__new__(WorldActionRobotWinPolicy)
    policy.model = SimpleNamespace(
        device=device,
        torch_dtype=torch.bfloat16,
    )
    model = object.__new__(FastWAM)
    torch.nn.Module.__init__(model)
    model.vae = vae
    model.device = device

    row = _episode_row(root, args.episode)
    results = []
    with h5py.File(Path(row["raw_file_name"]), "r") as handle, torch.no_grad():
        for decision in args.decisions:
            if decision not in frame_to_position:
                raise KeyError(f"Decision frame {decision} is absent from the cache")
            if decision == 0:
                window = [0]
            else:
                window = [
                    decision - args.replan_stride + i * temporal_step
                    for i in range(args.temporal_subframes + 1)
                ]
            online_frames = [
                policy._build_robotwin_image_tensor(
                    _observation(handle, frame_index)
                )
                for frame_index in window
            ]
            online_video = torch.stack(online_frames, dim=2)
            online_all = model._encode_input_image_latents_tensor(
                online_video,
                tiled=False,
            )
            online = online_all[0, :, -1:].cpu().to(torch.float32)
            cached = payload["latents"][frame_to_position[decision]].to(
                torch.float32
            )
            diff = (online - cached).abs()
            result = {
                "decision": decision,
                "window": window,
                "max_abs": float(diff.max().item()),
                "mean_abs": float(diff.mean().item()),
                "exact_bf16": bool(
                    torch.equal(online.to(torch.bfloat16), cached.to(torch.bfloat16))
                ),
            }
            results.append(result)
            if (
                result["max_abs"] > args.max_abs_tol
                or result["mean_abs"] > args.mean_abs_tol
            ):
                raise RuntimeError(f"Latent parity failed: {result}")
    print(
        json.dumps(
            {
                "episode": args.episode,
                "vae_sha256": actual_sha,
                "results": results,
                "passed": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
