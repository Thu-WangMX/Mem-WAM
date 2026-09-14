"""Extract auditable four-phase multi-layer spatial initialization-WAM features."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from fastwam.memory.four_phase_temporal import PHASE_OFFSETS
from fastwam.memory.multilayer_spatial_feature import (
    FEATURE_LAYERS,
    FEATURE_REGIONS,
    capture_spatial_features,
)
from fastwam.memory.wam_embedding_feature import load_episode_text_context
from scripts.build_putback_four_phase_latents import LATENT_SCHEMA, episodes_for_rank
from scripts.diagnose_putback_wam_embedding_surprise import (
    _fresh_fingerprint,
    _instantiate_initialization_model,
)


FEATURE_BANK_SCHEMA = "putback_four_phase_multilayer_wam_features_v1"
EXPECTED_INITIALIZATION_FINGERPRINT = (
    "51652b699bc50c5e1b0fa582a996788ea041d341f013043cf7174f52de679bc8"
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stack_features(features: dict[int, dict[str, torch.Tensor]]) -> torch.Tensor:
    if tuple(features) != FEATURE_LAYERS:
        raise ValueError(f"expected layers {FEATURE_LAYERS}, got {tuple(features)}")
    rows = []
    for layer in FEATURE_LAYERS:
        if tuple(features[layer]) != FEATURE_REGIONS:
            raise ValueError(
                f"layer {layer}: expected regions {FEATURE_REGIONS}, got {tuple(features[layer])}"
            )
        rows.append(torch.stack([features[layer][region] for region in FEATURE_REGIONS]))
    tensor = torch.stack(rows).float()
    if tensor.ndim != 3 or not bool(torch.isfinite(tensor).all()):
        raise ValueError("captured layer-region features are malformed")
    return tensor


def extract_episode_features(
    latent_payload: dict[str, Any],
    *,
    capture: Callable[..., dict[int, dict[str, torch.Tensor]]],
    initialization_fingerprint: str,
    feature_window: int = 8,
) -> dict[str, Any]:
    if latent_payload.get("schema_version") != LATENT_SCHEMA:
        raise ValueError("four-phase latent schema mismatch")
    if int(feature_window) <= 0:
        raise ValueError("feature_window must be positive")
    source_phases = latent_payload.get("phases", {})
    if set(source_phases) != {str(value) for value in PHASE_OFFSETS}:
        raise ValueError("four-phase latent payload is incomplete")
    phases: dict[str, dict[str, torch.Tensor]] = {}
    feature_dim: int | None = None
    for phase in PHASE_OFFSETS:
        source = source_phases[str(phase)]
        frames = torch.as_tensor(source["frame_indices"], dtype=torch.int64)
        latents = torch.as_tensor(source["latents"], dtype=torch.bfloat16)
        if tuple(latents.shape) != (len(frames), 48, 1, 24, 20):
            raise ValueError(f"phase {phase} latent shape mismatch")
        captured = []
        history_lengths = []
        for index in range(len(frames)):
            start = max(0, index + 1 - int(feature_window))
            causal = latents[start : index + 1, :, 0].permute(1, 0, 2, 3)
            tensor = _stack_features(capture(latents=causal))
            if feature_dim is None:
                feature_dim = int(tensor.shape[-1])
            elif int(tensor.shape[-1]) != feature_dim:
                raise ValueError("captured feature dimensions changed within episode")
            captured.append(tensor)
            history_lengths.append(index + 1 - start)
        phases[str(phase)] = {
            "frame_indices": frames.clone(),
            "features": torch.stack(captured),
            "warmup": torch.tensor([True, *([False] * (len(frames) - 1))]),
            "history_lengths": torch.tensor(history_lengths, dtype=torch.int64),
        }
    return {
        "schema_version": FEATURE_BANK_SCHEMA,
        "episode": int(latent_payload["episode"]),
        "initialization_fingerprint": str(initialization_fingerprint),
        "feature_layers": list(FEATURE_LAYERS),
        "feature_regions": list(FEATURE_REGIONS),
        "feature_dim": int(feature_dim or 0),
        "feature_window": int(feature_window),
        "phase_offsets": list(PHASE_OFFSETS),
        "source_latent_schema": LATENT_SCHEMA,
        "phases": phases,
    }


def compare_phase0_features(
    new_payload: dict[str, Any],
    existing_payload: dict[str, Any],
    *,
    atol: float = 2e-5,
    rtol: float = 2e-4,
) -> dict[str, Any]:
    phase0 = new_payload["phases"]["0"]
    new_frames = torch.as_tensor(phase0["frame_indices"], dtype=torch.int64)
    old_frames = torch.as_tensor(
        existing_payload["decision_frame_indices"], dtype=torch.int64
    )
    if not torch.equal(new_frames, old_frames):
        raise ValueError("phase-0 frame indices do not match existing feature bank")
    new = torch.as_tensor(phase0["features"])[:, 4, 0].float()
    old = torch.as_tensor(existing_payload["features"]).float()
    if new.shape != old.shape:
        raise ValueError(f"phase-0 feature shape mismatch: {new.shape} vs {old.shape}")
    difference = (new - old).abs()
    passed = bool(torch.allclose(new, old, atol=float(atol), rtol=float(rtol)))
    return {
        "pass": passed,
        "compared": int(new.shape[0]),
        "atol": float(atol),
        "rtol": float(rtol),
        "max_absolute_error": float(difference.max().item()) if difference.numel() else 0.0,
        "mean_absolute_error": float(difference.mean().item()) if difference.numel() else 0.0,
    }


def validate_existing_feature_episode(
    path: str | Path,
    *,
    episode: int,
    initialization_fingerprint: str,
    expected_sha256: str,
    expected_feature_dim: int = 3072,
) -> bool:
    try:
        path = Path(path)
        if _sha256(path) != str(expected_sha256):
            return False
        payload = torch.load(path, map_location="cpu", weights_only=True)
        identity = {
            "schema_version": FEATURE_BANK_SCHEMA,
            "episode": int(episode),
            "initialization_fingerprint": str(initialization_fingerprint),
            "feature_layers": list(FEATURE_LAYERS),
            "feature_regions": list(FEATURE_REGIONS),
            "feature_dim": int(expected_feature_dim),
            "phase_offsets": list(PHASE_OFFSETS),
            "source_latent_schema": LATENT_SCHEMA,
        }
        if any(payload.get(key) != value for key, value in identity.items()):
            return False
        for phase in PHASE_OFFSETS:
            row = payload["phases"][str(phase)]
            frames = torch.as_tensor(row["frame_indices"])
            features = torch.as_tensor(row["features"])
            warmup = torch.as_tensor(row["warmup"])
            if tuple(features.shape) != (
                len(frames),
                len(FEATURE_LAYERS),
                len(FEATURE_REGIONS),
                int(expected_feature_dim),
            ):
                return False
            if warmup.tolist() != [True, *([False] * (len(frames) - 1))]:
                return False
            if not bool(torch.isfinite(features).all()):
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


def _parse_episodes(text: str) -> list[int]:
    left, right = (int(value) for value in str(text).split("-", 1))
    return list(range(left, right + 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--latent-cache", required=True)
    parser.add_argument("--text-cache", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--action-init", required=True)
    parser.add_argument("--model-base", required=True)
    parser.add_argument("--initialization-reference", required=True)
    parser.add_argument("--phase0-bank", required=True)
    parser.add_argument("--episodes", default="0-49")
    parser.add_argument("--feature-window", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    episodes = _parse_episodes(args.episodes)
    if episodes != list(range(50)):
        raise ValueError("feature bank requires exactly episodes 0-49")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != 8:
        raise ValueError("feature extraction requires exactly eight ranks")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    output = Path(args.output).resolve()
    complete_manifest = output / "bank_manifest.json"
    if rank == 0:
        if complete_manifest.exists():
            raise FileExistsError(f"refusing complete feature bank: {output}")
        (output / "features").mkdir(parents=True, exist_ok=True)
        (output / "episodes").mkdir(exist_ok=True)
    dist.barrier()

    reference_path = Path(args.initialization_reference).resolve()
    reference = json.loads(reference_path.read_text())
    if reference.get("model_source") != "initialization" or reference.get("policy_checkpoint") is not None:
        raise ValueError("initialization reference is incompatible")
    expected_fingerprint = reference.get(
        "fresh_action_io_proprio_fingerprint",
        reference.get("initialization_fingerprint", EXPECTED_INITIALIZATION_FINGERPRINT),
    )
    if expected_fingerprint != EXPECTED_INITIALIZATION_FINGERPRINT:
        raise ValueError("unexpected initialization fingerprint")
    latent_root = Path(args.latent_cache).resolve()
    latent_manifest = json.loads((latent_root / "manifest.json").read_text())
    if latent_manifest.get("complete") is not True or latent_manifest.get("phase0_equivalence_pending") is not True:
        raise ValueError("four-phase latent bank is incomplete or incompatible")
    latent_entries = {
        int(row["episode"]): row for row in latent_manifest.get("episodes", [])
    }

    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(Path(args.model_base).resolve())
    os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
    os.environ["FASTWAM_ACTION_DIT_INIT"] = str(Path(args.action_init).resolve())
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model, model_config = _instantiate_initialization_model(
        repo=Path(args.repo).resolve(), output=output, device=f"cuda:{local_rank}"
    )
    fingerprint = _fresh_fingerprint(model)
    fingerprints: list[str | None] = [None] * world_size
    dist.all_gather_object(fingerprints, fingerprint)
    if set(fingerprints) != {expected_fingerprint}:
        raise RuntimeError(f"initialization fingerprint mismatch: {fingerprints}")

    local_rows: dict[int, dict[str, Any]] = {}
    for episode in episodes_for_rank(episodes, rank, world_size):
        latent_entry = latent_entries[episode]
        latent_path = latent_root / latent_entry["relative_path"]
        latent_payload = torch.load(latent_path, map_location="cpu", weights_only=True)
        text = load_episode_text_context(args.text_cache, args.dataset_root, episode)
        feature_path = output / "features" / f"episode_{episode:03d}.pt"
        metadata_path = output / "episodes" / f"episode_{episode:03d}.json"
        old_path = Path(args.phase0_bank) / "features" / f"episode_{episode:03d}.pt"
        if feature_path.exists() or metadata_path.exists():
            if not (feature_path.exists() and metadata_path.exists()):
                raise RuntimeError(f"incomplete feature entry: {episode}")
            metadata = json.loads(metadata_path.read_text())
            if not validate_existing_feature_episode(
                feature_path,
                episode=episode,
                initialization_fingerprint=fingerprint,
                expected_sha256=metadata.get("feature_sha256", ""),
            ) or metadata.get("phase0_equivalence", {}).get("pass") is not True:
                raise RuntimeError(f"refusing incompatible feature entry: {episode}")
            local_rows[episode] = metadata
            continue
        payload = extract_episode_features(
            latent_payload,
            capture=lambda *, latents: capture_spatial_features(
                model,
                latents=latents,
                video_context=text["context"],
                video_context_mask=text["mask"],
            ),
            initialization_fingerprint=fingerprint,
            feature_window=args.feature_window,
        )
        old_payload = torch.load(old_path, map_location="cpu", weights_only=True)
        equivalence = compare_phase0_features(payload, old_payload)
        if equivalence["pass"] is not True:
            raise RuntimeError(f"phase-0 equivalence failed for episode {episode}: {equivalence}")
        _atomic_torch(feature_path, payload)
        metadata = {
            "schema_version": FEATURE_BANK_SCHEMA,
            "episode": episode,
            "feature_sha256": _sha256(feature_path),
            "source_latent_sha256": _sha256(latent_path),
            "initialization_fingerprint": fingerprint,
            "feature_layers": list(FEATURE_LAYERS),
            "feature_regions": list(FEATURE_REGIONS),
            "feature_dim": 3072,
            "phase0_equivalence": equivalence,
        }
        _atomic_json(metadata_path, metadata)
        local_rows[episode] = metadata
        print(json.dumps({"rank": rank, "episode": episode, "status": "written"}), flush=True)

    gathered: list[dict[int, dict[str, Any]] | None] = [None] * world_size
    dist.all_gather_object(gathered, local_rows)
    dist.barrier()
    if rank == 0:
        merged: dict[int, dict[str, Any]] = {}
        for shard in gathered:
            if shard is None or set(merged).intersection(shard):
                raise RuntimeError("invalid distributed feature rows")
            merged.update(shard)
        if sorted(merged) != episodes or not all(
            row["phase0_equivalence"]["pass"] for row in merged.values()
        ):
            raise RuntimeError("feature bank completeness/equivalence failed")
        _atomic_json(
            output / "initialization_manifest.json",
            {
                "schema_version": FEATURE_BANK_SCHEMA,
                "initialization_fingerprint": fingerprint,
                "model_source": "initialization",
                "policy_checkpoint": None,
                "model_config": model_config,
                "source_reference": str(reference_path),
                "source_reference_sha256": _sha256(reference_path),
            },
        )
        _atomic_json(
            complete_manifest,
            {
                "schema_version": FEATURE_BANK_SCHEMA,
                "complete": True,
                "episode_count": 50,
                "episodes": [merged[index] for index in episodes],
                "initialization_fingerprint": fingerprint,
                "feature_layers": list(FEATURE_LAYERS),
                "feature_regions": list(FEATURE_REGIONS),
                "feature_dim": 3072,
                "feature_window": args.feature_window,
                "phase_offsets": list(PHASE_OFFSETS),
                "phase0_equivalence": "pass",
                "policy_checkpoint": None,
            },
        )
    del model
    torch.cuda.empty_cache()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

