"""Diagnose PutBack segmentation from frozen initialization WAM embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from fastwam.memory.wam_embedding_feature import (
    capture_last_frame_feature,
    load_episode_text_context,
)
from fastwam.memory.wam_embedding_surprise import (
    confirm_causal_peaks,
    score_causal_embeddings,
    segment_from_peaks,
    validate_causal_trace,
)
from fastwam.utils import misc
from fastwam.utils.config_resolvers import register_default_resolvers


SCHEMA_VERSION = "putback_wam_embedding_surprise_init_v1"
DEFAULT_LATENT_CACHE = (
    "/mnt/vepfs02/output/kevin_wang/memorywam/data/"
    "fastwam_putback_continuous_episode_stride16_v4"
)
DEFAULT_TEXT_CACHE = (
    "/mnt/vepfs02/output/kevin_wang/memorywam/data/"
    "fastwam_putback_text_cache_rgb_v3"
)
DEFAULT_DATASET_ROOT = (
    "/mnt/vepfs02/output/kevin_wang/memorywam/data/"
    "rmbench_lerobot_v21_rgb_v3/put_back_block"
)
DEFAULT_ACTION_INIT = (
    "/mnt/vepfs02/output/kevin_wang/memorywam/model_assets/"
    "fastwam_fullkv_official_init/"
    "ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
)
DEFAULT_MODEL_BASE = "/mnt/vepfs02/datasets/kevin_wang/code/projects/FastWAM/checkpoints"
DEFAULT_OUTPUT = (
    "/mnt/vepfs02/output/kevin_wang/memorywam/analysis/"
    "putback_wam_embedding_surprise_init_v1"
)


def _parse_episodes(text: str) -> list[int]:
    episodes = [int(value) for value in str(text).split(",") if value.strip()]
    if not episodes or len(set(episodes)) != len(episodes):
        raise ValueError("episodes must be a nonempty unique comma-separated list")
    return episodes


def _contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "episodes": _parse_episodes(args.episodes),
        "feature_layer": int(args.feature_layer),
        "feature_window": int(args.feature_window),
        "gamma": float(args.gamma),
        "max_segment": int(args.max_segment),
        "min_history": int(args.min_history),
        "min_segment": int(args.min_segment),
        "nms_distance": int(args.nms_distance),
        "policy_checkpoint": None,
        "seed": int(args.seed),
        "statistics_window": int(args.statistics_window),
        "threshold_window": int(args.threshold_window),
    }


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _atomic_torch(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _asset_row(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise FileNotFoundError(f"missing initialization asset: {resolved}")
    return {
        "path": str(resolved),
        "size": int(resolved.stat().st_size),
        "sha256": _sha256(resolved),
    }


def _fresh_fingerprint(model) -> str:
    modules = [
        ("action_encoder", model.action_expert.action_encoder),
        ("action_head", model.action_expert.head),
    ]
    if getattr(model, "proprio_encoder", None) is not None:
        modules.append(("proprio_encoder", model.proprio_encoder))
    digest = hashlib.sha256()
    for module_name, module in modules:
        for parameter_name, tensor in module.state_dict().items():
            value = tensor.detach().contiguous()
            digest.update(f"{module_name}.{parameter_name}".encode("utf-8"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def _load_episode_latents(root: Path, episode: int) -> dict[str, torch.Tensor]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    metadata = manifest.get("metadata", {})
    if metadata.get("complete") is not True or metadata.get("replan_stride") != 16:
        raise ValueError("continuous latent manifest is incomplete or incompatible")
    episodes = manifest.get("episodes", {})
    if not isinstance(episodes, dict):
        raise ValueError("continuous latent manifest episodes must be a mapping")
    # Continuous caches are namespaced by the RM-Bench task, for example
    # ``put_back_block/0`` and ``battery_try/0``.  Feature extraction is
    # task-agnostic, so resolve the numeric episode suffix rather than baking a
    # PutBack namespace into the loader.  Reject ambiguity instead of silently
    # selecting from a multi-task manifest.
    suffix = f"/{int(episode)}"
    matches = [
        value
        for key, value in episodes.items()
        if (str(key) == str(int(episode)) or str(key).endswith(suffix))
        and isinstance(value, str)
    ]
    relative = matches[0] if len(matches) == 1 else None
    if not isinstance(relative, str):
        available_namespaces = sorted(
            {str(key).rsplit("/", 1)[0] for key in episodes if "/" in str(key)}
        )
        raise KeyError(
            f"episode {episode} has {len(matches)} matches in the latent manifest; "
            f"namespaces={available_namespaces}"
        )
    payload = torch.load(root / relative, map_location="cpu", weights_only=True)
    latents = torch.as_tensor(payload["latents"], dtype=torch.bfloat16)
    frame_indices = torch.as_tensor(payload["frame_indices"], dtype=torch.int64)
    if latents.ndim != 5 or tuple(latents.shape[1:]) != (48, 1, 24, 20):
        raise ValueError(f"episode {episode} has invalid latents {tuple(latents.shape)}")
    if frame_indices.shape != (latents.shape[0],):
        raise ValueError("frame indices do not match episode latents")
    return {"latents": latents, "frame_indices": frame_indices}


def _trace_from_features(features: torch.Tensor, args: argparse.Namespace) -> dict[str, Any]:
    statistics = score_causal_embeddings(
        features,
        statistics_window=args.statistics_window,
        threshold_window=args.threshold_window,
        min_history=args.min_history,
        gamma=args.gamma,
    )
    peaks = confirm_causal_peaks(
        statistics["scores"],
        statistics["thresholds"],
        nms_distance=args.nms_distance,
    )
    segmentation = segment_from_peaks(
        int(features.shape[0]),
        peaks,
        min_segment=args.min_segment,
        max_segment=args.max_segment,
    )
    return {
        **statistics,
        "peaks": list(peaks),
        "peak_confirmed_at": [peak + 1 for peak in peaks],
        **segmentation,
    }


def _instantiate_initialization_model(
    *, repo: Path, output: Path, device: str
):
    register_default_resolvers()
    misc.register_work_dir(str(output / ".work" / device.replace(":", "_")))
    with initialize_config_dir(version_base="1.3", config_dir=str(repo / "configs")):
        cfg = compose(
            config_name="train",
            overrides=[
                "task=rmbench_putback_layerwise_k8_25k",
                f"output_dir={output / '.work'}",
                "resume=null",
            ],
        )
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    model_cfg.load_text_encoder = False
    model_cfg.skip_dit_load_from_pretrain = False
    model = instantiate(
        model_cfg,
        model_dtype=torch.bfloat16,
        device=device,
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, OmegaConf.to_container(model_cfg, resolve=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--latent-cache", default=DEFAULT_LATENT_CACHE)
    parser.add_argument("--text-cache", default=DEFAULT_TEXT_CACHE)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--action-init", default=DEFAULT_ACTION_INIT)
    parser.add_argument("--model-base", default=DEFAULT_MODEL_BASE)
    parser.add_argument("--episodes", default="40,41")
    parser.add_argument("--feature-window", type=int, default=8)
    parser.add_argument("--feature-layer", type=int, default=-1)
    parser.add_argument("--statistics-window", type=int, default=8)
    parser.add_argument("--threshold-window", type=int, default=8)
    parser.add_argument("--min-history", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--nms-distance", type=int, default=2)
    parser.add_argument("--min-segment", type=int, default=2)
    parser.add_argument("--max-segment", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-contract", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.print_contract:
        print(json.dumps(_contract(args), sort_keys=True))
        return
    if args.feature_window < 1 or args.min_segment < 2 or args.max_segment < args.min_segment:
        raise ValueError("invalid feature/segment lengths")

    episodes = _parse_episodes(args.episodes)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != len(episodes):
        raise ValueError(
            f"world size {world_size} must equal episode count {len(episodes)}"
        )
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl")

    output = Path(args.output).expanduser().resolve()
    if rank == 0:
        if output.exists():
            raise FileExistsError(f"refusing to overwrite output: {output}")
        (output / "episodes").mkdir(parents=True)
        (output / "features").mkdir()
    if world_size > 1:
        dist.barrier()

    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(Path(args.model_base).resolve())
    os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
    os.environ["FASTWAM_ACTION_DIT_INIT"] = str(Path(args.action_init).resolve())
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = f"cuda:{local_rank}"
    model, resolved_model_cfg = _instantiate_initialization_model(
        repo=Path(args.repo).resolve(), output=output, device=device
    )
    fingerprint = _fresh_fingerprint(model)
    fingerprints = [fingerprint]
    if world_size > 1:
        fingerprints = [None] * world_size
        dist.all_gather_object(fingerprints, fingerprint)
    if len(set(fingerprints)) != 1:
        raise RuntimeError(f"fresh initialization differs across ranks: {fingerprints}")

    if rank == 0:
        video_assets = sorted(
            (Path(args.model_base) / "Wan-AI" / "Wan2.2-TI2V-5B").glob(
                "diffusion_pytorch_model*.safetensors"
            )
        )
        assets = [_asset_row(path) for path in video_assets]
        assets.append(_asset_row(Path(args.action_init)))
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "model_source": "initialization",
            "policy_checkpoint": None,
            "resume": None,
            "seed": seed,
            "fresh_action_io_proprio_fingerprint": fingerprint,
            "assets": assets,
            "model_config": resolved_model_cfg,
            "contract": _contract(args),
        }
        _atomic_json(output / "initialization_manifest.json", manifest)

    episode = episodes[rank]
    episode_data = _load_episode_latents(Path(args.latent_cache), episode)
    text = load_episode_text_context(
        Path(args.text_cache), Path(args.dataset_root), episode
    )
    episode_latents = episode_data["latents"]
    features = []
    started = time.perf_counter()
    for decision in range(int(episode_latents.shape[0])):
        start = max(0, decision + 1 - int(args.feature_window))
        causal = episode_latents[start : decision + 1, :, 0].permute(1, 0, 2, 3)
        feature = capture_last_frame_feature(
            model,
            latents=causal,
            video_context=text["context"],
            video_context_mask=text["mask"],
            layer_index=args.feature_layer,
        )
        features.append(feature)
        print(
            json.dumps(
                {
                    "rank": rank,
                    "episode": episode,
                    "decision": decision,
                    "window_start": start,
                    "window_end": decision,
                }
            ),
            flush=True,
        )
    feature_tensor = torch.stack(features)
    trace = _trace_from_features(feature_tensor, args)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "episode": episode,
        "decision_count": int(feature_tensor.shape[0]),
        "decision_frame_indices": episode_data["frame_indices"].tolist(),
        "feature_dim": int(feature_tensor.shape[1]),
        "feature_window": int(args.feature_window),
        "feature_layer": int(args.feature_layer),
        "text_context_source": str(text["source"]),
        "prompt": text["prompt"],
        "task_index": text["task_index"],
        "elapsed_seconds": time.perf_counter() - started,
        **trace,
    }
    validate_causal_trace(payload)
    _atomic_torch(
        output / "features" / f"episode_{episode:03d}.pt",
        {
            "features": feature_tensor,
            "decision_frame_indices": episode_data["frame_indices"],
        },
    )
    _atomic_json(output / "episodes" / f"episode_{episode:03d}.json", payload)

    del model
    torch.cuda.empty_cache()
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        rows = []
        patterns = []
        for current in episodes:
            episode_path = output / "episodes" / f"episode_{current:03d}.json"
            feature_path = output / "features" / f"episode_{current:03d}.pt"
            recorded = json.loads(episode_path.read_text(encoding="utf-8"))
            saved = torch.load(feature_path, map_location="cpu", weights_only=True)
            replay = _trace_from_features(saved["features"], args)
            for field in (
                "scores",
                "thresholds",
                "score_source_end",
                "threshold_source_end",
                "peaks",
                "peak_confirmed_at",
                "boundaries",
                "reasons",
                "segment_lengths",
            ):
                if recorded[field] != replay[field]:
                    raise RuntimeError(
                        f"episode {current} replay mismatch for {field}"
                    )
            validate_causal_trace(recorded)
            patterns.append(tuple(recorded["segment_lengths"]))
            rows.append(
                {
                    "episode": current,
                    "decisions": recorded["decision_count"],
                    "peaks": recorded["peaks"],
                    "boundaries": recorded["boundaries"],
                    "segment_lengths": recorded["segment_lengths"],
                    "elapsed_seconds": recorded["elapsed_seconds"],
                }
            )
        _atomic_json(
            output / "aggregate.json",
            {
                "schema_version": SCHEMA_VERSION,
                "initialization_manifest": "initialization_manifest.json",
                "episodes": rows,
                "distinct_segment_patterns": len(set(patterns)),
                "not_both_fixed_length_eight": any(
                    any(length != 8 for length in pattern) for pattern in patterns
                ),
                "replay_validation": "pass",
            },
        )
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
