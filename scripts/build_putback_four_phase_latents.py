from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import h5py
import torch
import torch.distributed as dist

from fastwam.memory.four_phase_temporal import (
    DETECTOR_FRAME_STRIDE,
    PHASE_OFFSETS,
    VIDEO_EXPERT_FRAME_STRIDE,
    phase_frame_indices,
)
from fastwam.models.wan22.helpers.loader import _load_registered_model
from scripts.precompute_full_kv_observation_latents import _encode_batch, _mosaic


LATENT_SCHEMA = "putback_four_phase_latents_v1"
MOSAIC_CONTRACT = "wrists_top_head_bottom_384x320"


def load_registered_vae(
    vae_path: str | Path,
    *,
    device: torch.device,
    loader: Callable[..., Any] = _load_registered_model,
):
    """Bridge pathlib callers to the legacy FastWAM string-path loader."""

    return loader(
        str(Path(vae_path)),
        "wan_video_vae",
        torch_dtype=torch.bfloat16,
        device=str(device),
    ).eval()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def episodes_for_rank(
    episodes: Sequence[int], rank: int, world_size: int
) -> list[int]:
    rank = int(rank)
    world_size = int(world_size)
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank {rank} is outside world_size {world_size}")
    return [int(value) for index, value in enumerate(episodes) if index % world_size == rank]


def _parse_episodes(text: str) -> list[int]:
    text = str(text).strip()
    if "-" in text and "," not in text:
        left, right = (int(value) for value in text.split("-", 1))
        episodes = list(range(left, right + 1))
    else:
        episodes = [int(value) for value in text.split(",") if value.strip()]
    if not episodes or len(episodes) != len(set(episodes)):
        raise ValueError("episodes must be a nonempty range or unique list")
    return episodes


def build_four_phase_episode(
    *,
    episode: int,
    length: int,
    mosaic_at: Callable[[int], torch.Tensor],
    encode_video: Callable[[torch.Tensor], torch.Tensor],
    source_hdf5_sha256: str,
    vae_sha256: str,
) -> dict[str, Any]:
    """Encode four overlapping causal-VAE streams without changing native rate."""

    episode = int(episode)
    length = int(length)
    if length <= max(PHASE_OFFSETS):
        raise ValueError("episode is too short to initialize all four phases")
    sampled_frames = list(range(0, length, DETECTOR_FRAME_STRIDE))
    mosaics = {frame: mosaic_at(frame) for frame in sampled_frames}
    phases: dict[str, dict[str, torch.Tensor]] = {}
    for phase in PHASE_OFFSETS:
        input_frames = list(range(phase, length, DETECTOR_FRAME_STRIDE))
        video = torch.stack([mosaics[frame] for frame in input_frames], dim=1)
        encoded = torch.as_tensor(encode_video(video), dtype=torch.bfloat16)
        if encoded.ndim != 4 or encoded.shape[0] != 48:
            raise ValueError(
                f"phase {phase}: expected encoded [48,T,H,W], got {tuple(encoded.shape)}"
            )
        output_frames = phase_frame_indices(length, phase)
        if int(encoded.shape[1]) != len(output_frames):
            raise ValueError(
                f"phase {phase}: expected {len(output_frames)} causal latents "
                f"for {len(input_frames)} RGB mosaics, got {encoded.shape[1]}"
            )
        latents = encoded.permute(1, 0, 2, 3).unsqueeze(2).contiguous()
        if tuple(latents.shape[1:]) != (48, 1, 24, 20):
            raise ValueError(
                f"phase {phase}: expected latent tail (48,1,24,20), got {tuple(latents.shape)}"
            )
        phases[str(phase)] = {
            "frame_indices": torch.tensor(output_frames, dtype=torch.int64),
            "input_frame_indices": torch.tensor(input_frames, dtype=torch.int64),
            "latents": latents,
        }
    return {
        "schema_version": LATENT_SCHEMA,
        "episode": episode,
        "length": length,
        "phase_offsets": list(PHASE_OFFSETS),
        "source_rgb_stride": DETECTOR_FRAME_STRIDE,
        "video_expert_frame_stride": VIDEO_EXPERT_FRAME_STRIDE,
        "latent_shape": [48, 1, 24, 20],
        "latent_dtype": "bfloat16",
        "mosaic": MOSAIC_CONTRACT,
        "source_hdf5_sha256": str(source_hdf5_sha256),
        "vae_sha256": str(vae_sha256),
        "phase0_equivalence_pending": True,
        "phases": phases,
    }


def validate_existing_episode(
    path: str | Path,
    *,
    episode: int,
    length: int,
    source_hdf5_sha256: str,
    vae_sha256: str,
    expected_sha256: str,
) -> bool:
    try:
        path = Path(path)
        if _sha256(path) != str(expected_sha256):
            return False
        payload = torch.load(path, map_location="cpu", weights_only=True)
        expected_identity = {
            "schema_version": LATENT_SCHEMA,
            "episode": int(episode),
            "length": int(length),
            "phase_offsets": list(PHASE_OFFSETS),
            "source_rgb_stride": DETECTOR_FRAME_STRIDE,
            "video_expert_frame_stride": VIDEO_EXPERT_FRAME_STRIDE,
            "latent_shape": [48, 1, 24, 20],
            "latent_dtype": "bfloat16",
            "mosaic": MOSAIC_CONTRACT,
            "source_hdf5_sha256": str(source_hdf5_sha256),
            "vae_sha256": str(vae_sha256),
            "phase0_equivalence_pending": True,
        }
        if any(payload.get(key) != value for key, value in expected_identity.items()):
            return False
        phases = payload.get("phases", {})
        if set(phases) != {str(value) for value in PHASE_OFFSETS}:
            return False
        for phase in PHASE_OFFSETS:
            row = phases[str(phase)]
            frames = torch.as_tensor(row["frame_indices"], dtype=torch.int64)
            latents = torch.as_tensor(row["latents"])
            expected_frames = phase_frame_indices(int(length), phase)
            if frames.tolist() != expected_frames:
                return False
            if tuple(latents.shape) != (len(expected_frames), 48, 1, 24, 20):
                return False
            if latents.dtype != torch.bfloat16 or not bool(torch.isfinite(latents).all()):
                return False
        return True
    except (FileNotFoundError, KeyError, TypeError, ValueError, RuntimeError):
        return False


def _atomic_torch(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary output exists: {temporary}")
    torch.save(payload, temporary)
    temporary.replace(path)


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary output exists: {temporary}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lerobot-root", required=True)
    parser.add_argument("--vae-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", default="0-49")
    args = parser.parse_args()

    root = Path(args.lerobot_root).expanduser().resolve()
    vae_path = Path(args.vae_path).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    episodes = _parse_episodes(args.episodes)
    if episodes != list(range(50)):
        raise ValueError("production four-phase bank requires exactly episodes 0-49")
    if not vae_path.is_file():
        raise FileNotFoundError(f"missing VAE: {vae_path}")
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()).get("complete") is True:
        raise FileExistsError(f"refusing completed output root: {output}")

    rank, world_size, local_rank = _distributed()
    output.mkdir(parents=True, exist_ok=True)
    shard_root = output / "shards" / f"shard_{rank}"
    shard_root.mkdir(parents=True, exist_ok=True)
    vae_sha256 = _sha256(vae_path)
    device = torch.device(f"cuda:{local_rank}")
    vae = load_registered_vae(vae_path, device=device)

    rows = [
        json.loads(line)
        for line in (root / "meta" / "episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    by_episode = {int(row["episode_index"]): row for row in rows}
    assigned = episodes_for_rank(episodes, rank, world_size)
    local_entries: dict[int, dict[str, Any]] = {}
    for episode in assigned:
        if episode not in by_episode:
            raise KeyError(f"episode {episode} is absent from dataset metadata")
        row = by_episode[episode]
        length = int(row["length"])
        raw_path = Path(row["raw_file_name"]).expanduser().resolve()
        source_sha256 = _sha256(raw_path)
        feature_path = shard_root / f"episode_{episode:06d}.pt"
        metadata_path = shard_root / f"episode_{episode:06d}.json"
        if feature_path.exists() or metadata_path.exists():
            if not (feature_path.exists() and metadata_path.exists()):
                raise RuntimeError(f"incomplete existing episode output: {episode}")
            metadata = json.loads(metadata_path.read_text())
            if not validate_existing_episode(
                feature_path,
                episode=episode,
                length=length,
                source_hdf5_sha256=source_sha256,
                vae_sha256=vae_sha256,
                expected_sha256=metadata.get("file_sha256", ""),
            ):
                raise RuntimeError(f"refusing incompatible episode output: {episode}")
            local_entries[episode] = metadata
            continue
        with h5py.File(raw_path, "r") as handle:
            payload = build_four_phase_episode(
                episode=episode,
                length=length,
                mosaic_at=lambda frame, source=handle: _mosaic(source, frame),
                encode_video=lambda video: _encode_batch(vae, [video], device)[0],
                source_hdf5_sha256=source_sha256,
                vae_sha256=vae_sha256,
            )
        _atomic_torch(feature_path, payload)
        metadata = {
            "schema_version": LATENT_SCHEMA,
            "episode": episode,
            "length": length,
            "relative_path": str(feature_path.relative_to(output)),
            "file_sha256": _sha256(feature_path),
            "source_hdf5_sha256": source_sha256,
            "vae_sha256": vae_sha256,
            "phase_counts": {
                str(phase): len(phase_frame_indices(length, phase))
                for phase in PHASE_OFFSETS
            },
        }
        _atomic_json(metadata_path, metadata)
        local_entries[episode] = metadata
        print(json.dumps({"rank": rank, "episode": episode, "status": "written"}), flush=True)

    gathered: list[dict[int, dict[str, Any]] | None] = [None] * world_size
    if world_size > 1:
        dist.all_gather_object(gathered, local_entries)
        dist.barrier()
    else:
        gathered = [local_entries]
    if rank == 0:
        merged: dict[int, dict[str, Any]] = {}
        for shard in gathered:
            if shard is None or set(merged).intersection(shard):
                raise RuntimeError("invalid or overlapping distributed episode results")
            merged.update(shard)
        if sorted(merged) != episodes:
            raise RuntimeError(f"incomplete four-phase bank: {sorted(merged)}")
        _atomic_json(
            manifest_path,
            {
                "schema_version": LATENT_SCHEMA,
                "complete": True,
                "episodes": [merged[index] for index in episodes],
                "episode_count": len(episodes),
                "phase_offsets": list(PHASE_OFFSETS),
                "source_rgb_stride": DETECTOR_FRAME_STRIDE,
                "video_expert_frame_stride": VIDEO_EXPERT_FRAME_STRIDE,
                "mosaic": MOSAIC_CONTRACT,
                "vae_sha256": vae_sha256,
                "phase0_equivalence_pending": True,
                "policy_checkpoint": None,
            },
        )
        print(json.dumps({"bank_status": "complete", "episodes": len(episodes)}), flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
