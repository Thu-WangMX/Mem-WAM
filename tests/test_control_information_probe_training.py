from __future__ import annotations

import pytest
import torch

from scripts.train_putback_control_information_probe import (
    ControlProbeTrainingConfig,
    collate_rows,
    ensure_deterministic_cuda_environment,
    split_rows,
    train_control_probe,
)


def _row(*, episode: int, index: int):
    feature = torch.tensor(
        [float(index % 5), float((index * 2) % 7), float(index % 3), 1.0]
    )
    target = torch.zeros(2, 2)
    target[:, 0] = feature[0] - 0.5 * feature[1]
    target[:, 1] = feature[2] + feature[3]
    return {
        "episode": episode,
        "phase": 0,
        "frame": index * 4,
        "feature": feature,
        "proprio": torch.zeros(3),
        "target_actions": target,
        "target_mask": torch.ones(2, dtype=torch.bool),
    }


def _config():
    return ControlProbeTrainingConfig(
        feature_dim=4,
        proprio_dim=3,
        hidden_dim=16,
        proprio_hidden_dim=4,
        horizon=2,
        action_dim=2,
        batch_size=16,
        epochs=80,
        learning_rate=1e-2,
        weight_decay=0.0,
        patience=20,
        seed=42,
    )


def test_split_rows_keeps_calibration_and_heldout_out_of_training():
    rows = [_row(episode=episode, index=episode) for episode in range(50)]

    train, validation = split_rows(rows)

    assert [row["episode"] for row in train] == list(range(26))
    assert [row["episode"] for row in validation] == list(range(26, 30))
    assert not ({row["episode"] for row in train + validation} & set(range(30, 50)))


def test_proprio_only_batch_zeroes_wam_feature_without_changing_targets():
    rows = [_row(episode=0, index=1), _row(episode=0, index=2)]

    wam = collate_rows(rows, mode="wam_proprio", device=torch.device("cpu"))
    proprio = collate_rows(rows, mode="proprio_only", device=torch.device("cpu"))

    assert torch.count_nonzero(wam.feature).item() > 0
    assert torch.count_nonzero(proprio.feature).item() == 0
    torch.testing.assert_close(wam.target, proprio.target)
    torch.testing.assert_close(wam.mask, proprio.mask)


def test_synthetic_wam_probe_training_reduces_validation_loss():
    train = [_row(episode=index % 26, index=index) for index in range(128)]
    validation = [_row(episode=26 + index % 4, index=200 + index) for index in range(32)]

    result = train_control_probe(
        train,
        validation,
        config=_config(),
        mode="wam_proprio",
        device=torch.device("cpu"),
    )

    assert result.report["best_validation_loss"] < 0.25 * result.report[
        "initial_validation_loss"
    ]
    assert result.report["validation_mae"] < 0.5


def test_training_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        collate_rows([_row(episode=0, index=0)], mode="schedule", device="cpu")


def test_cuda_training_fails_fast_without_cublas_workspace_config(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG"):
        ensure_deterministic_cuda_environment(torch.device("cuda:0"))

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    ensure_deterministic_cuda_environment(torch.device("cuda:0"))
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")
    ensure_deterministic_cuda_environment(torch.device("cpu"))
