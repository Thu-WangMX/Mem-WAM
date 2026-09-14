"""Extract a complete PutBack feature bank from the exact initialization WAM."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.distributed as dist

from fastwam.memory.wam_embedding_feature import (
    capture_last_frame_feature,
    load_episode_text_context,
)
from scripts.diagnose_putback_wam_embedding_surprise import (
    DEFAULT_ACTION_INIT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_LATENT_CACHE,
    DEFAULT_MODEL_BASE,
    DEFAULT_TEXT_CACHE,
    _fresh_fingerprint,
    _instantiate_initialization_model,
    _load_episode_latents,
)


BANK_SCHEMA = "putback_init_wam_embedding_bank_v1"
DEFAULT_OUTPUT = (
    "/mnt/vepfs02/output/kevin_wang/memorywam/analysis/"
    "putback_init_wam_embedding_bank_v1"
)
DEFAULT_INIT_REFERENCE = (
    "/mnt/vepfs02/output/kevin_wang/memorywam/analysis/"
    "putback_wam_embedding_surprise_init_v1/initialization_manifest.json"
)


def episodes_for_rank(
    episodes: Sequence[int], rank: int, world_size: int
) -> list[int]:
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError("rank/world_size are invalid")
    values = [int(value) for value in episodes]
    if len(values) != len(set(values)):
        raise ValueError("episodes must be unique")
    return [value for value in values if value % world_size == rank]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_existing_entry(
    feature_path: Path,
    metadata_path: Path,
    *,
    episode: int,
    initialization_fingerprint: str,
    expected_frame_indices: Sequence[int],
    expected_feature_dim: int,
) -> bool:
    try:
        metadata = json.loads(metadata_path.read_text())
        if metadata != {
            **metadata,
            "schema_version": BANK_SCHEMA,
            "episode": int(episode),
            "initialization_fingerprint": initialization_fingerprint,
            "decision_frame_indices": [int(value) for value in expected_frame_indices],
            "feature_dim": int(expected_feature_dim),
        }:
            return False
        if metadata.get("feature_sha256") != _sha256(feature_path):
            return False
        saved = torch.load(feature_path, map_location="cpu", weights_only=True)
        features = torch.as_tensor(saved["features"])
        frames = torch.as_tensor(saved["decision_frame_indices"], dtype=torch.int64)
        return (
            features.ndim == 2
            and int(features.shape[0]) == len(expected_frame_indices)
            and int(features.shape[1]) == int(expected_feature_dim)
            and frames.tolist() == [int(value) for value in expected_frame_indices]
        )
    except (FileNotFoundError, KeyError, ValueError, TypeError, RuntimeError, json.JSONDecodeError):
        return False


def _parse_episodes(text: str) -> list[int]:
    text = str(text).strip()
    if "-" in text and "," not in text:
        left, right = (int(value) for value in text.split("-", 1))
        values = list(range(left, right + 1))
    else:
        values = [int(value) for value in text.split(",") if value.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("episodes must be a nonempty range or unique list")
    return values


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_feature(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    torch.save(payload, temporary)
    temporary.replace(path)


def _contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": BANK_SCHEMA,
        "episodes": _parse_episodes(args.episodes),
        "feature_layer": int(args.feature_layer),
        "feature_window": int(args.feature_window),
        "replan_stride": 16,
        "seed": int(args.seed),
        "model_source": "initialization",
        "policy_checkpoint": None,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--latent-cache", default=DEFAULT_LATENT_CACHE)
    parser.add_argument("--text-cache", default=DEFAULT_TEXT_CACHE)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--action-init", default=DEFAULT_ACTION_INIT)
    parser.add_argument("--model-base", default=DEFAULT_MODEL_BASE)
    parser.add_argument("--initialization-reference", default=DEFAULT_INIT_REFERENCE)
    parser.add_argument("--episodes", default="0-49")
    parser.add_argument("--feature-window", type=int, default=8)
    parser.add_argument("--feature-layer", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-contract", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.print_contract:
        print(json.dumps(_contract(args), sort_keys=True))
        return
    episodes = _parse_episodes(args.episodes)
    if episodes != list(range(50)):
        raise ValueError("the immutable bank contract requires episodes 0 through 49")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != 8:
        raise ValueError("feature bank extraction requires exactly eight ranks")
    assigned = episodes_for_rank(episodes, rank, world_size)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    output = Path(args.output).expanduser().resolve()
    complete_manifest = output / "bank_manifest.json"
    if rank == 0:
        if complete_manifest.exists():
            raise FileExistsError(f"refusing to overwrite complete feature bank: {output}")
        (output / "features").mkdir(parents=True, exist_ok=True)
        (output / "episodes").mkdir(exist_ok=True)
    dist.barrier()

    reference_path = Path(args.initialization_reference).expanduser().resolve()
    reference = json.loads(reference_path.read_text())
    if (
        reference.get("model_source") != "initialization"
        or reference.get("policy_checkpoint") is not None
    ):
        raise ValueError("initialization reference is incompatible")
    expected_fingerprint = reference.get("fresh_action_io_proprio_fingerprint")
    if not isinstance(expected_fingerprint, str) or not expected_fingerprint:
        raise ValueError("initialization reference fingerprint is missing")

    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(Path(args.model_base).resolve())
    os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
    os.environ["FASTWAM_ACTION_DIT_INIT"] = str(Path(args.action_init).resolve())
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = f"cuda:{local_rank}"
    model, resolved_model_cfg = _instantiate_initialization_model(
        repo=Path(args.repo).resolve(), output=output, device=device
    )
    fingerprint = _fresh_fingerprint(model)
    fingerprints: list[str | None] = [None] * world_size
    dist.all_gather_object(fingerprints, fingerprint)
    if len(set(fingerprints)) != 1 or fingerprint != expected_fingerprint:
        raise RuntimeError(
            f"initialization fingerprint mismatch: expected={expected_fingerprint} got={fingerprints}"
        )
    if rank == 0:
        bank_initialization = {
            "schema_version": BANK_SCHEMA,
            "source_manifest": str(reference_path),
            "source_manifest_sha256": _sha256(reference_path),
            "initialization_fingerprint": fingerprint,
            "model_source": "initialization",
            "policy_checkpoint": None,
            "model_config": resolved_model_cfg,
            "contract": _contract(args),
        }
        bank_initialization_path = output / "initialization_manifest.json"
        if bank_initialization_path.exists():
            if json.loads(bank_initialization_path.read_text()) != bank_initialization:
                raise RuntimeError("refusing to overwrite incompatible bank initialization manifest")
        else:
            _atomic_json(bank_initialization_path, bank_initialization)
    dist.barrier()

    for episode in assigned:
        episode_data = _load_episode_latents(Path(args.latent_cache), episode)
        frame_indices = episode_data["frame_indices"].tolist()
        feature_path = output / "features" / f"episode_{episode:03d}.pt"
        metadata_path = output / "episodes" / f"episode_{episode:03d}.json"
        if feature_path.exists() or metadata_path.exists():
            if validate_existing_entry(
                feature_path,
                metadata_path,
                episode=episode,
                initialization_fingerprint=fingerprint,
                expected_frame_indices=frame_indices,
                expected_feature_dim=3072,
            ):
                print(json.dumps({"rank": rank, "episode": episode, "status": "validated_skip"}), flush=True)
                continue
            raise RuntimeError(f"refusing to overwrite incompatible bank entry: episode {episode}")
        text = load_episode_text_context(
            Path(args.text_cache), Path(args.dataset_root), episode
        )
        episode_latents = episode_data["latents"]
        features = []
        for decision in range(int(episode_latents.shape[0])):
            start = max(0, decision + 1 - int(args.feature_window))
            causal = episode_latents[start : decision + 1, :, 0].permute(1, 0, 2, 3)
            features.append(
                capture_last_frame_feature(
                    model,
                    latents=causal,
                    video_context=text["context"],
                    video_context_mask=text["mask"],
                    layer_index=args.feature_layer,
                )
            )
        feature_tensor = torch.stack(features)
        _atomic_feature(
            feature_path,
            {
                "features": feature_tensor,
                "decision_frame_indices": episode_data["frame_indices"],
            },
        )
        _atomic_json(
            metadata_path,
            {
                "schema_version": BANK_SCHEMA,
                "episode": episode,
                "initialization_fingerprint": fingerprint,
                "decision_frame_indices": frame_indices,
                "feature_dim": int(feature_tensor.shape[1]),
                "feature_sha256": _sha256(feature_path),
                "feature_layer": int(args.feature_layer),
                "feature_window": int(args.feature_window),
                "prompt": text["prompt"],
                "task_index": text["task_index"],
            },
        )
        print(json.dumps({"rank": rank, "episode": episode, "decisions": len(features), "status": "written"}), flush=True)

    del model
    torch.cuda.empty_cache()
    dist.barrier()
    if rank == 0:
        rows = []
        for episode in episodes:
            metadata_path = output / "episodes" / f"episode_{episode:03d}.json"
            feature_path = output / "features" / f"episode_{episode:03d}.pt"
            metadata = json.loads(metadata_path.read_text())
            if not validate_existing_entry(
                feature_path,
                metadata_path,
                episode=episode,
                initialization_fingerprint=fingerprint,
                expected_frame_indices=metadata["decision_frame_indices"],
                expected_feature_dim=3072,
            ):
                raise RuntimeError(f"bank completeness validation failed: episode {episode}")
            rows.append(
                {
                    "episode": episode,
                    "feature_sha256": metadata["feature_sha256"],
                    "decisions": len(metadata["decision_frame_indices"]),
                }
            )
        _atomic_json(
            complete_manifest,
            {
                "schema_version": BANK_SCHEMA,
                "complete": True,
                "episodes": rows,
                "episode_count": len(rows),
                "initialization_fingerprint": fingerprint,
                "contract": _contract(args),
            },
        )
        print(json.dumps({"bank_status": "complete", "episodes": len(rows), "output": str(output)}), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
