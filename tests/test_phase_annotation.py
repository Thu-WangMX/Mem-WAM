from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import numpy as np

from fastwam.memory.phase_annotation import (
    ANNOTATION_SCHEMA,
    TRANSITIONS,
    AnnotationStore,
    validate_episode_annotation,
    write_reviewed_manifest,
)
from scripts.prepare_putback_phase_annotation import (
    build_annotation_draft,
    load_episode_control,
    propose_gripper_events,
)


def _episode(episode: int = 30) -> dict:
    return {
        "episode": episode,
        "split": "development" if episode < 40 else "heldout",
        "reviewed": True,
        "reviewer": "human",
        "transitions": [
            {
                "name": name,
                "frame": 20 + index * 20,
                "ambiguity_start": 16 + index * 20,
                "ambiguity_end": 24 + index * 20,
                "not_observed": False,
                "reason": None,
            }
            for index, name in enumerate(TRANSITIONS)
        ],
    }


def _manifest() -> dict:
    return {
        "schema_version": ANNOTATION_SCHEMA,
        "frame_stride": 4,
        "episodes": [_episode(episode) for episode in range(30, 50)],
    }


def test_annotation_requires_ordered_reviewed_transitions():
    validate_episode_annotation(_episode())

    reversed_episode = _episode()
    reversed_episode["transitions"][1]["frame"] = 8
    reversed_episode["transitions"][1]["ambiguity_start"] = 4
    reversed_episode["transitions"][1]["ambiguity_end"] = 12
    with pytest.raises(ValueError, match="ordered"):
        validate_episode_annotation(reversed_episode)

    wrong_name = _episode()
    wrong_name["transitions"][0]["name"] = "unknown"
    with pytest.raises(ValueError, match="transition names"):
        validate_episode_annotation(wrong_name)


def test_annotation_enforces_grid_interval_and_explicit_missing_reason():
    off_grid = _episode()
    off_grid["transitions"][0]["frame"] = 21
    with pytest.raises(ValueError, match="divisible by four"):
        validate_episode_annotation(off_grid)

    outside = _episode()
    outside["transitions"][0]["frame"] = 28
    with pytest.raises(ValueError, match="ambiguity"):
        validate_episode_annotation(outside)

    missing = _episode()
    missing["transitions"][2] = {
        "name": TRANSITIONS[2],
        "frame": None,
        "ambiguity_start": None,
        "ambiguity_end": None,
        "not_observed": True,
        "reason": "lift is occluded in every camera",
    }
    validate_episode_annotation(missing)
    missing["transitions"][2]["reason"] = ""
    with pytest.raises(ValueError, match="reason"):
        validate_episode_annotation(missing)


def test_development_store_cannot_open_heldout_annotations(tmp_path):
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(_manifest()))
    store = AnnotationStore(path, access="development")

    assert store.load_episode(30)["episode"] == 30
    with pytest.raises(PermissionError, match="held-out"):
        store.load_episode(40)


def test_final_store_burns_locked_selector_before_heldout_access(tmp_path):
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(_manifest()))
    marker = tmp_path / "heldout_opened.json"

    with pytest.raises(ValueError, match="selector"):
        AnnotationStore(path, access="final", heldout_marker=marker)
    store = AnnotationStore(
        path,
        access="final",
        locked_selector_hash="selector-sha",
        heldout_marker=marker,
    )
    assert json.loads(marker.read_text())["locked_selector_hash"] == "selector-sha"
    assert store.load_episode(40)["episode"] == 40
    with pytest.raises(RuntimeError, match="different selector"):
        AnnotationStore(
            path,
            access="final",
            locked_selector_hash="different",
            heldout_marker=marker,
        )


def test_reviewed_manifest_is_immutable_and_hashed(tmp_path):
    output = tmp_path / "reviewed.json"
    result = write_reviewed_manifest(output, _manifest())

    assert output.is_file()
    assert len(result["sha256"]) == 64
    assert result["episode_count"] == 20
    with pytest.raises(FileExistsError, match="reviewed"):
        write_reviewed_manifest(output, _manifest())


def test_unreviewed_draft_keeps_heuristics_separate_from_labels():
    actions = [[0.0] * 14 for _ in range(12)]
    for index in range(3, 9):
        actions[index][6] = 1.0
        actions[index][13] = 1.0
    frames = [index * 4 for index in range(12)]

    proposals = propose_gripper_events(actions, frames)
    draft = build_annotation_draft(episode=30, proposals=proposals)

    assert proposals == [
        {"kind": "gripper_close", "frame": 12, "dimensions": [6, 13]},
        {"kind": "gripper_open", "frame": 36, "dimensions": [6, 13]},
    ]
    assert draft["reviewed"] is False
    assert draft["proposals"] == proposals
    assert all(row["status"] == "pending" for row in draft["transitions"])
    assert all(row["frame"] is None for row in draft["transitions"])
    validate_episode_annotation(draft, require_reviewed=False)
    with pytest.raises(ValueError, match="reviewed"):
        validate_episode_annotation(draft)


def test_control_data_comes_from_lerobot_parquet_not_raw_hdf5(tmp_path):
    root = tmp_path / "dataset"
    parquet = root / "data" / "chunk-000" / "episode_000030.parquet"
    parquet.parent.mkdir(parents=True)
    actions = np.arange(6 * 14, dtype=np.float32).reshape(6, 14)
    states = actions + 1000
    table = pa.table(
        {
            "frame_index": pa.array(range(6), type=pa.int64()),
            "action": pa.array(actions.tolist(), type=pa.list_(pa.float32(), 14)),
            "observation.state": pa.array(
                states.tolist(), type=pa.list_(pa.float32(), 14)
            ),
        }
    )
    pq.write_table(table, parquet)

    sampled_actions, sampled_states = load_episode_control(root, 30, [0, 4])

    np.testing.assert_array_equal(sampled_actions, actions[[0, 4]])
    np.testing.assert_array_equal(sampled_states, states[[0, 4]])
