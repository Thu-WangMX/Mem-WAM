import json

import pytest

from fastwam.memory.physical_dynamic_l import (
    SCHEMA_VERSION,
    PhysicalDynamicLManifestStore,
    validate_segments,
)


def test_validate_segments_accepts_delayed_confirmation_and_variable_l():
    records = validate_segments(
        {
            "decision_count": 24,
            "segments": [
                {"start": 2, "end": 6, "confirmed_at": 10, "reason": "snap_gripper"},
                {"start": 6, "end": 12, "confirmed_at": 14, "reason": "nominal_L0"},
                {"start": 12, "end": 20, "confirmed_at": 20, "reason": "snap_settled_peak"},
            ],
        }
    )

    assert [record.length for record in records] == [4, 6, 8]
    assert [record.confirmed_at for record in records] == [10, 14, 20]


@pytest.mark.parametrize(
    "segments",
    [
        [{"start": 0, "end": 4, "confirmed_at": 8, "reason": "bad_anchor"}],
        [{"start": 2, "end": 5, "confirmed_at": 10, "reason": "too_short"}],
        [{"start": 2, "end": 6, "confirmed_at": 5, "reason": "noncausal"}],
    ],
)
def test_validate_segments_rejects_contract_violations(segments):
    with pytest.raises(ValueError):
        validate_segments({"decision_count": 20, "segments": segments})


def test_manifest_store_rejects_robotwin_pretraining(tmp_path):
    episode_dir = tmp_path / "episodes"
    episode_dir.mkdir()
    (episode_dir / "episode_000.json").write_text(
        json.dumps({"episode": 0, "decision_count": 12, "segments": []})
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "task": "put_back_block",
        "episode_count": 1,
        "replan_stride": 16,
        "anchor_frames": 2,
        "recent_frames": 4,
        "min_segment": 4,
        "nominal_segment": 6,
        "max_segment": 8,
        "memory_tokens": 8,
        "uses_robotwin_pretrained": True,
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps({"metadata": metadata, "episodes": {"0": "episodes/episode_000.json"}})
    )

    with pytest.raises(ValueError, match="uses_robotwin_pretrained"):
        PhysicalDynamicLManifestStore(tmp_path, expected_episode_count=1)
