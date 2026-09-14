import json

import pytest

from fastwam.memory.latent_kernel_regime import (
    SCHEMA_VERSION,
    LatentKernelManifestStore,
    validate_segments,
)


def _episode():
    return {
        "schema_version": SCHEMA_VERSION,
        "episode": 0,
        "decision_count": 22,
        "score_source_end": [None, None, None, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, None],
        "segments": [
            {"start": 2, "end": 6, "confirmed_at": 10, "reason": "multires_latent_kernel_regime"},
            {"start": 6, "end": 14, "confirmed_at": 14, "reason": "stable_max_length"},
        ],
    }


def test_segment_contract_accepts_delayed_visual_boundary():
    records = validate_segments(_episode())
    assert [row.length for row in records] == [4, 8]


def test_segment_contract_rejects_future_dependent_boundary():
    payload = _episode()
    payload["score_source_end"][6] = 11
    with pytest.raises(ValueError, match="unavailable"):
        validate_segments(payload)


def test_store_rejects_any_physical_signal(tmp_path):
    episode_dir = tmp_path / "episodes"
    episode_dir.mkdir()
    (episode_dir / "episode_000.json").write_text(json.dumps(_episode()))
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "task": "put_back_block",
        "episode_count": 1,
        "selector_input": "frozen_continuous_causal_vae_latents_only",
        "uses_action": False,
        "uses_proprioception": True,
        "uses_reward": False,
        "uses_task_checkpoint": False,
        "fitted_parameters": None,
        "replan_stride": 16,
        "anchor_frames": 2,
        "recent_frames": 4,
        "min_segment": 4,
        "max_segment": 8,
        "memory_tokens": 8,
        "kernel_window_each_side": 2,
        "views": ["appearance_flat", "scene_channel_mean", "local_channel_direction"],
        "fusion": "unweighted_arithmetic_mean",
        "rank_threshold": 0.75,
        "history_window": 8,
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps({"metadata": metadata, "episodes": {"0": "episodes/episode_000.json"}})
    )
    with pytest.raises(ValueError, match="uses_proprioception"):
        LatentKernelManifestStore(tmp_path, expected_episode_count=1)
