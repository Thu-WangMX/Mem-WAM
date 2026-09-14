from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from fastwam.memory.phase_annotation import ANNOTATION_SCHEMA, TRANSITIONS
from scripts.evaluate_putback_predictive_selectors import (
    LockedEvaluationSession,
    lock_candidate,
    select_passing_candidate,
)


REQUIRED_HASHES = {
    "predictor_sha256": "1" * 64,
    "pca_sha256": "2" * 64,
    "stats_sha256": "3" * 64,
    "config_sha256": "4" * 64,
    "code_sha256": "5" * 64,
}


def _annotation_manifest(path: Path):
    episodes = []
    for episode in range(30, 50):
        episodes.append(
            {
                "episode": episode,
                "split": "development" if episode < 40 else "heldout",
                "reviewed": True,
                "reviewer": "unit-test",
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
        )
    path.write_text(
        json.dumps(
            {"schema_version": ANNOTATION_SCHEMA, "frame_stride": 4, "episodes": episodes}
        )
    )


def _passing(name: str, f1: float, precision: float, memory: int):
    return {
        "name": name,
        "report": {
            "micro": {"precision": precision, "recall": 0.8, "f1": f1},
            "macro": {"precision": precision, "recall": 0.8, "f1": f1},
            "transition_recall": {name: 0.5 for name in TRANSITIONS},
            "baseline_f1": 0.5,
            "retroactive_boundary_count": 0,
            "memory_count": memory,
            "uniform_four_frame_budget": 100,
            "group_length_histogram": {8: 2, 12: 2},
        },
        "stable_streams": [1, 6],
        "fold_streams": [[1, 6], [1], [1, 6], [1], [1], [1], [1], [1], [1], [1]],
    }


def test_candidate_selection_requires_gate_and_stable_streams():
    winner = select_passing_candidate(
        [_passing("v_only", 0.72, 0.82, 30), _passing("v_action", 0.75, 0.70, 35)]
    )
    assert winner["name"] == "v_action"
    assert winner["stable_streams"] == [1]


def test_development_cannot_read_final_and_final_burns_hash_bound_marker(tmp_path):
    annotations = tmp_path / "annotations.json"
    _annotation_manifest(annotations)
    candidate_path = tmp_path / "locked_candidate.json"
    lock_candidate(candidate_path, candidate=_passing("v_action", 0.75, 0.7, 35), hashes=REQUIRED_HASHES)

    development = LockedEvaluationSession(
        annotation_path=annotations,
        mode="development",
    )
    assert development.load_episode(30)["episode"] == 30
    with pytest.raises(PermissionError):
        development.load_episode(40)

    marker = tmp_path / "heldout_opened.json"
    final = LockedEvaluationSession(
        annotation_path=annotations,
        mode="final",
        locked_candidate_path=candidate_path,
        heldout_marker=marker,
    )
    assert marker.exists()
    marker_payload = json.loads(marker.read_text())
    assert "heldout_opened_at" in marker_payload
    assert marker_payload["locked_candidate_sha256"] == hashlib.sha256(
        candidate_path.read_bytes()
    ).hexdigest()
    assert final.load_episode(40)["episode"] == 40

    changed = json.loads(candidate_path.read_text())
    changed["hashes"]["code_sha256"] = "9" * 64
    candidate_path.write_text(json.dumps(changed, sort_keys=True))
    with pytest.raises(RuntimeError, match="different locked candidate"):
        LockedEvaluationSession(
            annotation_path=annotations,
            mode="final",
            locked_candidate_path=candidate_path,
            heldout_marker=marker,
        )


def test_cloud_evaluation_launcher_is_immutable_contract():
    launcher = (
        Path(__file__).parents[1] / "ops" / "evaluate_putback_predictive_selectors.sh"
    ).read_text()
    assert "set -euo pipefail" in launcher
    assert "COMPARISON_JSON=\"${COMPARISON_JSON:?" in launcher
    assert "LOCKED_CANDIDATE=\"${LOCKED_CANDIDATE:?" in launcher
    assert "scripts/evaluate_putback_predictive_selectors.py" in launcher
