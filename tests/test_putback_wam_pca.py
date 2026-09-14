from __future__ import annotations

import pytest
import torch

from scripts.fit_putback_wam_pca import fit_pca_artifact, transform_feature_tensor


def _episode(episode: int, feature_dim: int = 4) -> dict:
    phases = {}
    for phase in (0, 4, 8, 12):
        values = []
        for step in range(5):
            base = torch.arange(5 * 4 * feature_dim, dtype=torch.float32)
            varied = (
                base
                + episode * (base.remainder(7) + 1)
                + step * (base.remainder(5) + 1).square()
                + phase * base.remainder(3)
            )
            values.append(varied.reshape(5, 4, feature_dim))
        phases[str(phase)] = {
            "features": torch.stack(values),
            "warmup": torch.tensor([True, False, False, False, False]),
            "frame_indices": torch.tensor([phase + 16 * step for step in range(5)]),
        }
    return {
        "schema_version": "putback_four_phase_multilayer_wam_features_v1",
        "episode": episode,
        "phases": phases,
    }


def test_pca_fits_only_declared_training_episodes_and_all_streams():
    bank = {episode: _episode(episode) for episode in range(6)}

    artifact = fit_pca_artifact(
        bank,
        episodes=[0, 1, 2, 3, 4],
        component_dim=2,
        source_bank_sha256="bank-sha",
    )

    assert artifact["schema_version"] == "putback_wam_stream_pca_v1"
    assert artifact["episodes"] == [0, 1, 2, 3, 4]
    assert artifact["source_bank_sha256"] == "bank-sha"
    assert artifact["mean"].shape == (5, 4, 4)
    assert artifact["components"].shape == (5, 4, 2, 4)
    assert artifact["projected_scale"].shape == (5, 4, 2)
    assert torch.isfinite(artifact["components"]).all()
    transformed = transform_feature_tensor(
        bank[5]["phases"]["0"]["features"], artifact
    )
    assert transformed.shape == (5, 5, 4, 2)


def test_pca_rejects_nontraining_episode_and_rank_collapse():
    with pytest.raises(ValueError, match="0-29"):
        fit_pca_artifact(
            {30: _episode(30)},
            episodes=[30],
            component_dim=2,
            source_bank_sha256="bank-sha",
        )
    collapsed = _episode(0)
    for phase in collapsed["phases"].values():
        phase["features"].zero_()
    with pytest.raises(ValueError, match="rank"):
        fit_pca_artifact(
            {0: collapsed},
            episodes=[0],
            component_dim=2,
            source_bank_sha256="bank-sha",
        )
