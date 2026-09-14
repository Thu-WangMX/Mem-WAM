from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "replay_putback_wam_embedding_causal_v2.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_replay_is_provenanced_deterministic_and_immutable(tmp_path):
    source = tmp_path / "v1"
    output = tmp_path / "v2"
    (source / "features").mkdir(parents=True)
    (source / "episodes").mkdir()
    manifest = {
        "schema_version": "putback_wam_embedding_surprise_init_v1",
        "model_source": "initialization",
        "policy_checkpoint": None,
        "fresh_action_io_proprio_fingerprint": "fingerprint-42",
    }
    (source / "initialization_manifest.json").write_text(json.dumps(manifest))
    features = torch.tensor([0, 1, 0, 1, 0, 10, 0, 0], dtype=torch.float32)[:, None]
    frame_indices = torch.arange(8, dtype=torch.int64) * 16
    feature_path = source / "features" / "episode_040.pt"
    torch.save({"features": features, "decision_frame_indices": frame_indices}, feature_path)
    (source / "episodes" / "episode_040.json").write_text(
        json.dumps(
            {
                "schema_version": "putback_wam_embedding_surprise_init_v1",
                "episode": 40,
                "decision_count": 8,
                "decision_frame_indices": frame_indices.tolist(),
            }
        )
    )
    command = [sys.executable, str(SCRIPT), "--source", str(source), "--output", str(output), "--episodes", "40", "--statistics-window", "4", "--threshold-window", "4", "--min-history", "2", "--min-threshold-history", "2"]

    first = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)

    assert first.returncode == 0, first.stderr
    v2_manifest = json.loads((output / "manifest.json").read_text())
    assert v2_manifest["schema_version"] == "putback_wam_embedding_surprise_causal_v2"
    assert v2_manifest["policy_checkpoint"] is None
    assert v2_manifest["source_initialization_fingerprint"] == "fingerprint-42"
    assert v2_manifest["source_initialization_manifest_sha256"] == _sha256(source / "initialization_manifest.json")
    trace = json.loads((output / "episodes" / "episode_040.json").read_text())
    assert trace["boundaries"] == [0, 6, 8]
    assert trace["source_feature_sha256"] == _sha256(feature_path)
    assert trace["replay_validation"] == "pass"
    aggregate = json.loads((output / "aggregate.json").read_text())
    assert aggregate["retroactive_boundary_count"] == 0
    assert aggregate["replay_validation"] == "pass"
    assert _sha256(output / "features" / "episode_040.pt") == _sha256(feature_path)

    second = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert second.returncode != 0
    assert "refusing to overwrite" in second.stderr
