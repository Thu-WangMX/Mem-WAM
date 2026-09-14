from __future__ import annotations

import torch

from scripts.build_putback_control_information_dataset import (
    build_projected_episode,
    normalization_from_stats,
)
from scripts.fit_putback_wam_pca import fit_pca_artifact
from tests.test_putback_wam_pca import _episode


def test_artifact_builder_projects_flattens_and_preserves_detector_frames():
    bank = {episode: _episode(episode, feature_dim=6) for episode in range(4)}
    pca = fit_pca_artifact(
        bank, episodes=range(4), component_dim=2, source_bank_sha256="a" * 64
    )
    actions = torch.randn(80, 14, generator=torch.Generator().manual_seed(8))
    proprio = torch.randn(80, 14, generator=torch.Generator().manual_seed(9))
    normalization = {
        "action_mean": torch.zeros(14),
        "action_std": torch.ones(14),
        "proprio_mean": torch.zeros(14),
        "proprio_std": torch.ones(14),
    }

    rows = build_projected_episode(
        episode=2,
        feature_payload=bank[2],
        pca_artifact=pca,
        actions=actions,
        proprio=proprio,
        normalization=normalization,
        horizon=16,
    )

    assert [row["frame"] for row in rows] == list(range(0, 80, 4))
    assert rows[0]["feature"].shape == (5 * 4 * 2,)
    assert rows[-1]["target_mask"].sum().item() == 4


def test_normalization_uses_global_z_score_statistics():
    payload = {
        "action": {
            "default": {
                "global_mean": [[float(value) for value in range(14)]],
                "global_std": [[2.0] * 14],
            }
        },
        "state": {
            "default": {
                "global_mean": [[3.0] * 14],
                "global_std": [[4.0] * 14],
            }
        },
    }

    normalization = normalization_from_stats(payload)

    assert normalization["action_mean"].shape == (14,)
    assert normalization["action_mean"].tolist() == [float(value) for value in range(14)]
    assert normalization["action_std"].tolist() == [2.0] * 14
    assert normalization["proprio_mean"].tolist() == [3.0] * 14
    assert normalization["proprio_std"].tolist() == [4.0] * 14
