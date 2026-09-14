import json

import pytest

from fastwam.memory.dynamic_surprise import (
    DynamicSurpriseManifestStore,
    causal_boundaries,
    closed_segments_for_history,
    validate_episode_segments,
)


def _episode(boundaries=(0, 4, 8, 11), reasons=None):
    if reasons is None:
        reasons = {"0": "start", "4": "surprise", "8": "max_length", "11": "end"}
    return {
        "episode": 0,
        "decision_count": 11,
        "boundaries": list(boundaries),
        "reasons": reasons,
    }


def test_validate_episode_segments_accepts_terminal_one_frame_tail():
    payload = _episode(
        boundaries=(0, 4, 10, 11),
        reasons={"0": "start", "4": "surprise", "10": "surprise", "11": "end"},
    )

    assert validate_episode_segments(payload, decision_count=11) == (
        (0, 4),
        (4, 10),
        (10, 11),
    )


@pytest.mark.parametrize(
    ("boundaries", "message"),
    [
        ((1, 4, 8, 11), "start at zero"),
        ((0, 4, 4, 11), "strictly increasing"),
        ((0, 1, 8, 11), "length"),
        ((0, 4, 13), "decision count"),
        ((0, 9, 11), "length"),
    ],
)
def test_validate_episode_segments_rejects_invalid_partitions(boundaries, message):
    with pytest.raises(ValueError, match=message):
        validate_episode_segments(_episode(boundaries=boundaries), decision_count=11)


def test_closed_segments_leave_boundary_observation_raw():
    segments = ((0, 4), (4, 8), (8, 11))

    assert closed_segments_for_history(segments, history_frames=4) == ()
    assert closed_segments_for_history(segments, history_frames=5) == ((0, 1, 2, 3),)
    assert closed_segments_for_history(segments, history_frames=9) == (
        (0, 1, 2, 3),
        (4, 5, 6, 7),
    )


def test_manifest_store_rejects_boundary_phase_mismatch(tmp_path):
    episode_dir = tmp_path / "episodes"
    episode_dir.mkdir()
    (episode_dir / "episode_000.json").write_text(json.dumps(_episode()), encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "schema_version": "putback_dynamic_surprise_segments_v1",
                    "complete": True,
                    "task": "putback",
                    "episode_count": 1,
                    "boundary_step": 5000,
                    "replan_stride": 16,
                },
                "episodes": {"0": "episodes/episode_000.json"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="boundary_step"):
        DynamicSurpriseManifestStore(tmp_path, expected_boundary_step=10000)


def test_manifest_store_loads_and_validates_episode(tmp_path):
    episode_dir = tmp_path / "episodes"
    episode_dir.mkdir()
    (episode_dir / "episode_000.json").write_text(json.dumps(_episode()), encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "schema_version": "putback_dynamic_surprise_segments_v1",
                    "complete": True,
                    "task": "putback",
                    "episode_count": 1,
                    "boundary_step": -1,
                    "replan_stride": 16,
                },
                "episodes": {"0": "episodes/episode_000.json"},
            }
        ),
        encoding="utf-8",
    )

    store = DynamicSurpriseManifestStore(tmp_path, expected_boundary_step=-1)

    assert store.segments_for_episode(0) == ((0, 4), (4, 8), (8, 11))


def test_manifest_store_accepts_battery_when_explicitly_expected(tmp_path):
    episode_dir = tmp_path / "episodes"
    episode_dir.mkdir()
    (episode_dir / "episode_000.json").write_text(
        json.dumps(_episode()), encoding="utf-8"
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "schema_version": "putback_dynamic_surprise_segments_v1",
                    "complete": True,
                    "task": "battery_try",
                    "episode_count": 1,
                    "boundary_step": -1,
                    "replan_stride": 16,
                },
                "episodes": {"0": "episodes/episode_000.json"},
            }
        ),
        encoding="utf-8",
    )

    store = DynamicSurpriseManifestStore(
        tmp_path,
        expected_boundary_step=-1,
        expected_task="battery_try",
    )
    assert store.metadata["task"] == "battery_try"


def test_manifest_store_rejects_cross_task_manifest(tmp_path):
    episode_dir = tmp_path / "episodes"
    episode_dir.mkdir()
    (episode_dir / "episode_000.json").write_text(
        json.dumps(_episode()), encoding="utf-8"
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "schema_version": "putback_dynamic_surprise_segments_v1",
                    "complete": True,
                    "task": "battery_try",
                    "episode_count": 1,
                    "boundary_step": -1,
                    "replan_stride": 16,
                },
                "episodes": {"0": "episodes/episode_000.json"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be 'putback'"):
        DynamicSurpriseManifestStore(tmp_path, expected_boundary_step=-1)


def test_causal_threshold_uses_global_rolling_history_without_segment_reset():
    result = causal_boundaries(
        [1.0, 1.0, 1.0, 4.0, 1.0, 1.0, 1.0, 4.0],
        min_segment=2,
        max_segment=8,
        gamma=1.5,
        window=5,
    )

    assert result["boundaries"] == [0, 4, 8, 9]
    assert result["reasons"]["4"] == "surprise"
    assert result["reasons"]["8"] == "surprise"
    assert result["reasons"]["9"] == "end"
    assert result["thresholds"][4] is not None
