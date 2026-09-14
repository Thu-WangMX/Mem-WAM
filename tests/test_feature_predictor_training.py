from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from scripts.train_putback_feature_predictor import (
    PredictorTrainingConfig,
    evaluate_loss,
    make_split_manifest,
    save_compact_model,
    serialize_split_manifest,
    train_predictor,
)


def _dataset(count: int = 64, feature_dim: int = 4, condition_dim: int = 3):
    generator = torch.Generator().manual_seed(13)
    examples = []
    weight = torch.tensor(
        [[1.0, -0.5, 0.2], [0.3, 0.8, -0.4], [-0.7, 0.1, 0.5], [0.2, 0.2, 0.2]]
    )
    for index in range(count):
        history = torch.randn((3, feature_dim), generator=generator)
        condition = torch.randn((condition_dim,), generator=generator)
        target = 0.35 * history[-1] + weight @ condition
        examples.append(
            {
                "episode": index % 10,
                "phase": (0, 4, 8, 12)[index % 4],
                "target_frame": 16 + index,
                "history": history,
                "condition": condition,
                "target": target,
            }
        )
    return examples


def _config(epochs: int) -> PredictorTrainingConfig:
    return PredictorTrainingConfig(
        feature_dim=4,
        condition_dim=3,
        hidden_dim=24,
        batch_size=16,
        epochs=epochs,
        learning_rate=2e-2,
        weight_decay=0.0,
        patience=100,
        seed=42,
    )


def test_split_manifest_is_mode_independent_and_byte_identical():
    visual = make_split_manifest(train_episodes=range(26), val_episodes=range(26, 30))
    action = make_split_manifest(train_episodes=range(26), val_episodes=range(26, 30))

    assert serialize_split_manifest(visual) == serialize_split_manifest(action)
    assert visual["train_episodes"] == list(range(26))
    assert visual["val_episodes"] == [26, 27, 28, 29]
    assert max(visual["val_episodes"]) < 30


def test_deterministic_training_reduces_predictable_loss_and_model_bytes(tmp_path: Path):
    train = _dataset()
    validation = _dataset(count=32)
    config = _config(epochs=60)

    first = train_predictor(train, validation, config=config)
    second = train_predictor(train, validation, config=config)

    initial = first.report["initial_validation_loss"]
    final = evaluate_loss(first.model, validation, batch_size=16)
    assert final < initial * 0.20
    assert first.report == second.report
    for left, right in zip(first.model.state_dict().values(), second.model.state_dict().values()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)

    left_path = tmp_path / "left.safetensors"
    right_path = tmp_path / "right.safetensors"
    save_compact_model(left_path, first.model, config=config, report=first.report)
    save_compact_model(right_path, second.model, config=config, report=second.report)
    assert hashlib.sha256(left_path.read_bytes()).digest() == hashlib.sha256(
        right_path.read_bytes()
    ).digest()


def test_resume_is_exactly_equivalent_to_uninterrupted_training():
    train = _dataset()
    validation = _dataset(count=32)
    full = train_predictor(train, validation, config=_config(epochs=20))
    first_half = train_predictor(train, validation, config=_config(epochs=7))
    resumed = train_predictor(
        train,
        validation,
        config=_config(epochs=20),
        resume_state=first_half.resume_state,
    )

    assert set(resumed.resume_state) == {
        "schema_version",
        "epoch",
        "model",
        "optimizer",
        "best_model",
        "best_validation_loss",
        "epochs_without_improvement",
        "report",
        "config",
    }
    for left, right in zip(full.model.state_dict().values(), resumed.model.state_dict().values()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert full.report == resumed.report


def test_launcher_runs_matched_modes_in_disjoint_gpu_groups():
    launcher = (
        Path(__file__).parents[1] / "ops" / "train_putback_feature_predictors_8gpu.sh"
    ).read_text()

    assert "CUDA_VISIBLE_DEVICES=0,1,2,3" in launcher
    assert "CUDA_VISIBLE_DEVICES=4,5,6,7" in launcher
    assert 'export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"' in launcher
    assert "--mode visual_only" in launcher
    assert "--mode visual_action" in launcher
    assert "cmp \"${OUTPUT_ROOT}/visual_only/split_manifest.json\"" in launcher
    assert "refusing to overwrite OUTPUT_ROOT" in launcher
