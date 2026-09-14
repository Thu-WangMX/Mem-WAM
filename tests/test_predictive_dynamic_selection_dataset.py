from __future__ import annotations

import torch
import pytest

from fastwam.datasets.lerobot.predictive_dynamic_selection_dataset import (
    DynamicSelectionManifestStore,
    attach_predictive_dynamic_groups,
)


def _manifest():
    hashes = {"predictor_sha256": "a" * 64}
    return {
        "schema_version": "putback_dynamic_selection_manifest_v1",
        "source_hashes": hashes,
        "episodes": {
            "0": {
                "terminal_frame": 24,
                "groups": [
                    {"start_frame": 0, "end_frame": 8, "detector_units": 2, "confirmation_frame": 8,
                     "peak_frame": 4, "reason": "predictive_surprise_confirmed", "score": 4.0,
                     "selected_streams": [1], "source_hashes": hashes},
                    {"start_frame": 8, "end_frame": 24, "detector_units": 4, "confirmation_frame": 24,
                     "peak_frame": 24, "reason": "terminal_tail", "score": 0.0,
                     "selected_streams": [], "source_hashes": hashes},
                ],
            }
        },
    }


def test_manifest_backed_adapter_preserves_every_unit_and_action_target():
    manifest = _manifest()
    store = DynamicSelectionManifestStore(
        manifest=manifest,
        verification={"exact": True, "manifest_sha256": "verified"},
        expected_source_hashes=manifest["source_hashes"],
    )
    latents = torch.arange(6 * 3, dtype=torch.float32).reshape(6, 3)
    action = torch.randn(16, 14)
    sample = {
        "episode_index": 0,
        "frame_index": 24,
        "observation_frame_indices": torch.tensor([4, 8, 12, 16, 20, 24]),
        "history_latents": latents,
        "action": action.clone(),
    }
    result = attach_predictive_dynamic_groups(sample, store, tokens_per_group=8)
    assert result["history_memory_groups"].tolist() == [[0, 2], [2, 6]]
    assert result["history_memory_group_lengths"].tolist() == [2, 4]
    assert result["history_memory_group_reasons"] == ["predictive_surprise_confirmed", "terminal_tail"]
    assert [chunk.shape[0] for chunk in result["compressor_inputs"]] == [2, 4]
    assert result["memory_token_count"] == 16
    assert result["memory_attention_mask"].tolist() == [1] * 16
    torch.testing.assert_close(result["action"], action)
    assert "score" not in result["action_model_metadata"]


def test_adapter_rejects_unverified_hash_mismatch_and_frame_gaps():
    manifest = _manifest()
    with pytest.raises(ValueError, match="verified"):
        DynamicSelectionManifestStore(
            manifest=manifest, verification={"exact": False},
            expected_source_hashes=manifest["source_hashes"],
        )
    with pytest.raises(ValueError, match="hash"):
        DynamicSelectionManifestStore(
            manifest=manifest, verification={"exact": True},
            expected_source_hashes={"predictor_sha256": "b" * 64},
        )
    store = DynamicSelectionManifestStore(
        manifest=manifest, verification={"exact": True},
        expected_source_hashes=manifest["source_hashes"],
    )
    with pytest.raises(ValueError, match="coverage"):
        attach_predictive_dynamic_groups(
            {"episode_index": 0, "frame_index": 24,
             "observation_frame_indices": torch.tensor([4, 8, 16, 20, 24]),
             "history_latents": torch.zeros(5, 2), "action": torch.zeros(16, 14)},
            store, tokens_per_group=8,
        )

