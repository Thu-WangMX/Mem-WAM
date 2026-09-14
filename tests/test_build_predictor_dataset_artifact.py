from __future__ import annotations

import torch

from scripts.build_putback_feature_predictor_dataset import build_projected_episode
from scripts.fit_putback_wam_pca import fit_pca_artifact
from tests.test_putback_wam_pca import _episode


def test_real_artifact_builder_projects_flattens_and_builds_matched_modes():
    bank = {episode: _episode(episode, feature_dim=6) for episode in range(4)}
    pca = fit_pca_artifact(
        bank, episodes=range(4), component_dim=2, source_bank_sha256="a" * 64
    )
    actions = torch.randn(80, 14, generator=torch.Generator().manual_seed(8))
    proprio = torch.randn(80, 14, generator=torch.Generator().manual_seed(9))

    payload = build_projected_episode(
        episode=2, feature_payload=bank[2], pca_artifact=pca,
        actions=actions, proprio=proprio, max_history=8,
    )

    assert set(payload) == {"visual_only", "visual_action"}
    assert len(payload["visual_only"]) == len(payload["visual_action"])
    assert payload["visual_action"][0]["target"].shape == (5 * 4 * 2,)
    assert payload["visual_action"][0]["history"].shape[-1] == 40
    for visual, action in zip(payload["visual_only"], payload["visual_action"]):
        assert visual["episode"] == action["episode"] == 2
        assert visual["target_frame"] == action["target_frame"]
        torch.testing.assert_close(visual["target"], action["target"])
        assert torch.count_nonzero(visual["condition"]) == 0
