from __future__ import annotations

import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf
import pytest
import torch

from fastwam.evaluation.wrist_event_contract import (
    validate_wrist_event_eval_contract,
)


INFERENCE_CONTRACT = {
    "video_scheduler_shift": 5.0,
    "action_scheduler_shift": 1.0,
    "video_num_train_timesteps": 1000,
    "action_num_train_timesteps": 1000,
    "video_attention_mask_mode": "first_frame_causal",
    "full_kv_cache": True,
}


def _write_contract_fixture(tmp_path: Path) -> dict[str, Path]:
    predictor = tmp_path / "best.pt"
    torch.save(
        {
            "model_config": {
                "history": 3,
                "include_proprio": False,
            },
            "model_state": {"weight": torch.ones(1)},
        },
        predictor,
    )
    predictor_sha = hashlib.sha256(predictor.read_bytes()).hexdigest()

    manifest_root = tmp_path / "boundary_manifest"
    manifest_root.mkdir()
    manifest = manifest_root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "metadata": {
                    "schema_version": "putback_wrist_latent_event_segments_v1",
                    "complete": True,
                    "task": "put_back_block",
                    "episode_count": 50,
                    "checkpoint_sha256": predictor_sha,
                    "decoder": "causal_first_threshold_crossing_with_two_decision_cooldown",
                    "history": 3,
                    "threshold": 0.5,
                    "min_segment": 2,
                    "max_segment": 8,
                    "replan_stride": 16,
                    "uses_vlm": False,
                    "uses_absolute_decision_index": False,
                },
                "episodes": {str(index): f"episode_{index:03d}.json" for index in range(50)},
            }
        ),
        encoding="utf-8",
    )

    native = {
        "enabled": True,
        "mode": "layerwise",
        "memory_tokens": 8,
        "group_size": 4,
        "anchor_frames": 2,
        "recent_frames": 4,
        "recursive": False,
    }
    training = tmp_path / "config.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "max_steps": 40000,
                "data": {
                    "train": {
                        "_target_": "fastwam.datasets.lerobot.wrist_event_dataset.WristEventRobotVideoDataset",
                        "wrist_event_manifest_path": str(manifest_root),
                        "concat_multi_camera": "robotwin",
                        "robotwin_head_position": "bottom",
                        "replan_stride": 16,
                    }
                },
                "model": {"native_cache": native},
            }
        ),
        training,
    )
    evaluation = OmegaConf.create(
        {
            "model": {"native_cache": native},
            "EVALUATION": {
                "dynamic_surprise_online": False,
                "wrist_event_online": True,
                "wrist_event_checkpoint": str(predictor),
                "wrist_event_threshold": 0.5,
                "wrist_event_history": 3,
                "wrist_event_min_segment": 2,
                "wrist_event_max_segment": 8,
                "replan_steps": 16,
            },
        }
    )
    policy = tmp_path / "step_005000.pt"
    torch.save(
        {
            "mot": {"weight": torch.ones(1, dtype=torch.bfloat16)},
            "layerwise_block_memory": {
                "slots": torch.ones(1, dtype=torch.bfloat16)
            },
            "step": 5000,
            "torch_dtype": "torch.bfloat16",
            "inference_contract": INFERENCE_CONTRACT,
        },
        policy,
    )
    return {
        "training": training,
        "manifest": manifest,
        "predictor": predictor,
        "policy": policy,
        "evaluation": evaluation,
    }


def _validate(paths: dict[str, object]) -> dict[str, object]:
    return validate_wrist_event_eval_contract(
        training_config_path=paths["training"],
        evaluation_config=paths["evaluation"],
        manifest_path=paths["manifest"],
        predictor_checkpoint_path=paths["predictor"],
        policy_checkpoint_path=paths["policy"],
        expected_step=5000,
    )


def test_valid_contract_reports_exact_5k_k8_predictor_identity(tmp_path):
    paths = _write_contract_fixture(tmp_path)

    report = _validate(paths)

    assert report["checkpoint_step"] == 5000
    assert report["memory_tokens"] == 8
    assert report["predictor_sha256"] == json.loads(
        paths["manifest"].read_text()
    )["metadata"]["checkpoint_sha256"]
    assert report["train_infer_consistent"] is True


def test_contract_rejects_predictor_hash_mismatch(tmp_path):
    paths = _write_contract_fixture(tmp_path)
    payload = json.loads(paths["manifest"].read_text())
    payload["metadata"]["checkpoint_sha256"] = "0" * 64
    paths["manifest"].write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="predictor checkpoint SHA-256"):
        _validate(paths)


def test_contract_rejects_online_threshold_mismatch(tmp_path):
    paths = _write_contract_fixture(tmp_path)
    paths["evaluation"].EVALUATION.wrist_event_threshold = 0.6

    with pytest.raises(ValueError, match="wrist_event_threshold"):
        _validate(paths)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(step=4999), "checkpoint step"),
        (
            lambda payload: payload.pop("layerwise_block_memory"),
            "layerwise_block_memory",
        ),
    ],
)
def test_contract_rejects_invalid_portable_checkpoint(tmp_path, mutation, message):
    paths = _write_contract_fixture(tmp_path)
    payload = torch.load(paths["policy"], map_location="cpu", weights_only=False)
    mutation(payload)
    torch.save(payload, paths["policy"])

    with pytest.raises(ValueError, match=message):
        _validate(paths)
