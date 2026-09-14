import json

import pytest

from fastwam.memory.wrist_event import WristEventManifestStore


def _write_manifest(tmp_path, *, complete=True, schema="putback_wrist_latent_event_segments_v1"):
    episodes = tmp_path / "episodes"
    episodes.mkdir()
    (episodes / "episode_000.json").write_text(
        json.dumps(
            {
                "episode": 0,
                "decision_count": 10,
                "boundaries": [0, 5, 10],
                "reasons": {"0": "start", "5": "wrist_event", "10": "end"},
                "segment_lengths": [5, 5],
            }
        )
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "schema_version": schema,
                    "complete": complete,
                    "task": "put_back_block",
                    "episode_count": 1,
                    "replan_stride": 16,
                    "uses_vlm": False,
                },
                "episodes": {"0": "episodes/episode_000.json"},
            }
        )
    )


def test_store_reads_validated_putback_segments(tmp_path):
    _write_manifest(tmp_path)

    store = WristEventManifestStore(tmp_path, expected_episode_count=1)

    assert store.metadata["uses_vlm"] is False
    assert store.segments_for_episode(0) == ((0, 5), (5, 10))


def test_store_rejects_incomplete_or_wrong_schema(tmp_path):
    _write_manifest(tmp_path, complete=False)
    with pytest.raises(ValueError, match="complete"):
        WristEventManifestStore(tmp_path)

    (tmp_path / "manifest.json").unlink()
    for child in (tmp_path / "episodes").iterdir():
        child.unlink()
    (tmp_path / "episodes").rmdir()
    _write_manifest(tmp_path, schema="wrong")
    with pytest.raises(ValueError, match="schema"):
        WristEventManifestStore(tmp_path)
