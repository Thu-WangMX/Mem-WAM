"""Strict train/inference contract for PutBack wrist-event checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from omegaconf import DictConfig, OmegaConf
import torch


EXPECTED_NATIVE = {
    "enabled": True,
    "mode": "layerwise",
    "memory_tokens": 8,
    "group_size": 4,
    "anchor_frames": 2,
    "recent_frames": 4,
    "recursive": False,
}
EXPECTED_INFERENCE_CONTRACT = {
    "video_scheduler_shift": 5.0,
    "action_scheduler_shift": 1.0,
    "video_num_train_timesteps": 1000,
    "action_num_train_timesteps": 1000,
    "video_attention_mask_mode": "first_frame_causal",
    "full_kv_cache": True,
}
EXPECTED_DECODER = "causal_first_threshold_crossing_with_two_decision_cooldown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _select(config: DictConfig, path: str) -> Any:
    value = OmegaConf.select(config, path)
    if value is None:
        raise ValueError(f"missing contract field: {path}")
    return value


def _expect(config: DictConfig, path: str, expected: object) -> None:
    actual = _select(config, path)
    if actual != expected:
        raise ValueError(
            f"contract mismatch for {path}: expected {expected!r}, found {actual!r}"
        )


def _load_config(path: str | Path) -> DictConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"missing training config: {config_path}")
    config = OmegaConf.load(config_path)
    if not isinstance(config, DictConfig):
        raise ValueError(f"training config must be a mapping: {config_path}")
    return config


def validate_wrist_event_eval_contract(
    *,
    training_config_path: str | Path,
    evaluation_config: DictConfig,
    manifest_path: str | Path,
    predictor_checkpoint_path: str | Path,
    policy_checkpoint_path: str | Path,
    expected_step: int = 5000,
) -> dict[str, object]:
    """Validate every frozen boundary and K8 runtime choice before GPU use."""

    training = _load_config(training_config_path)
    if not isinstance(evaluation_config, DictConfig):
        evaluation_config = OmegaConf.create(evaluation_config)

    manifest_file = Path(manifest_path).expanduser().resolve()
    predictor_file = Path(predictor_checkpoint_path).expanduser().resolve()
    policy_file = Path(policy_checkpoint_path).expanduser().resolve()
    for label, path in (
        ("manifest", manifest_file),
        ("predictor checkpoint", predictor_file),
        ("policy checkpoint", policy_file),
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"missing or empty {label}: {path}")

    _expect(
        training,
        "data.train._target_",
        "fastwam.datasets.lerobot.wrist_event_dataset.WristEventRobotVideoDataset",
    )
    _expect(training, "data.train.concat_multi_camera", "robotwin")
    _expect(training, "data.train.robotwin_head_position", "bottom")
    _expect(training, "data.train.replan_stride", 16)
    training_manifest = Path(
        str(_select(training, "data.train.wrist_event_manifest_path"))
    ).expanduser().resolve()
    if training_manifest != manifest_file.parent:
        raise ValueError(
            "training wrist_event_manifest_path does not match evaluated manifest: "
            f"{training_manifest} != {manifest_file.parent}"
        )

    for key, expected in EXPECTED_NATIVE.items():
        _expect(training, f"model.native_cache.{key}", expected)
        _expect(evaluation_config, f"model.native_cache.{key}", expected)

    manifest_payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    metadata = manifest_payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("boundary manifest metadata must be a mapping")
    expected_metadata = {
        "schema_version": "putback_wrist_latent_event_segments_v1",
        "complete": True,
        "task": "put_back_block",
        "episode_count": 50,
        "decoder": EXPECTED_DECODER,
        "history": 3,
        "threshold": 0.5,
        "min_segment": 2,
        "max_segment": 8,
        "replan_stride": 16,
        "uses_vlm": False,
        "uses_absolute_decision_index": False,
    }
    for key, expected in expected_metadata.items():
        actual = metadata.get(key)
        if actual != expected:
            raise ValueError(
                f"manifest contract mismatch for {key}: "
                f"expected {expected!r}, found {actual!r}"
            )

    predictor_sha = _sha256(predictor_file)
    if metadata.get("checkpoint_sha256") != predictor_sha:
        raise ValueError(
            "predictor checkpoint SHA-256 does not match boundary manifest: "
            f"{predictor_sha} != {metadata.get('checkpoint_sha256')}"
        )
    predictor_payload = torch.load(
        predictor_file,
        map_location="cpu",
        weights_only=False,
    )
    predictor_config = predictor_payload.get("model_config", {})
    if predictor_config.get("history") != 3:
        raise ValueError("predictor history must equal 3")
    if predictor_config.get("include_proprio") is not False:
        raise ValueError("predictor must be wrist-latent-only")

    eval_expected = {
        "EVALUATION.dynamic_surprise_online": False,
        "EVALUATION.wrist_event_online": True,
        "EVALUATION.wrist_event_threshold": 0.5,
        "EVALUATION.wrist_event_history": 3,
        "EVALUATION.wrist_event_min_segment": 2,
        "EVALUATION.wrist_event_max_segment": 8,
        "EVALUATION.replan_steps": 16,
    }
    for path, expected in eval_expected.items():
        _expect(evaluation_config, path, expected)
    eval_predictor = Path(
        str(_select(evaluation_config, "EVALUATION.wrist_event_checkpoint"))
    ).expanduser().resolve()
    if eval_predictor != predictor_file:
        raise ValueError(
            "evaluation wrist_event_checkpoint does not match validated predictor: "
            f"{eval_predictor} != {predictor_file}"
        )

    checkpoint = torch.load(
        policy_file,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    actual_step = checkpoint.get("step")
    if actual_step != int(expected_step):
        raise ValueError(
            f"policy checkpoint step mismatch: expected {expected_step}, "
            f"found {actual_step}"
        )
    mot = checkpoint.get("mot")
    if not isinstance(mot, Mapping) or not mot:
        raise ValueError("policy checkpoint has no non-empty mot weights")
    layerwise = checkpoint.get("layerwise_block_memory")
    if not isinstance(layerwise, Mapping) or not layerwise:
        raise ValueError(
            "policy checkpoint has no non-empty layerwise_block_memory weights"
        )
    checkpoint_contract = checkpoint.get("inference_contract")
    if not isinstance(checkpoint_contract, Mapping):
        raise ValueError("policy checkpoint has no inference_contract")
    for key, expected in EXPECTED_INFERENCE_CONTRACT.items():
        actual = checkpoint_contract.get(key)
        if actual != expected:
            raise ValueError(
                f"policy inference contract mismatch for {key}: "
                f"expected {expected!r}, found {actual!r}"
            )

    return {
        "train_infer_consistent": True,
        "task": "put_back_block",
        "checkpoint_step": int(actual_step),
        "checkpoint_path": str(policy_file),
        "memory_mode": "layerwise",
        "memory_tokens": 8,
        "predictor_sha256": predictor_sha,
        "threshold": 0.5,
        "history": 3,
        "min_segment": 2,
        "max_segment": 8,
        "uses_vlm": False,
    }
