from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_putback_wam_warmup_comparison.py"


def _features() -> torch.Tensor:
    return torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.98, 0.2], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.98, 0.2], [1.0, 0.0]],
        dtype=torch.float32,
    )


def _actions() -> list[list[float]]:
    values = [[0.0] * 14 for _ in range(160)]
    for row in values:
        row[6] = 1.0
        row[13] = 1.0
    for frame in range(48, 160):
        values[frame][13] = 0.0
    for frame in range(112, 160):
        values[frame][13] = 1.0
    return values


def test_cli_never_loads_calibration_actions_and_freezes_heldout_report(tmp_path):
    bank = tmp_path / "bank"
    dataset = tmp_path / "dataset"
    output = tmp_path / "comparison"
    (bank / "features").mkdir(parents=True)
    (bank / "episodes").mkdir()
    rows = []
    for episode in range(50):
        feature_path = bank / "features" / f"episode_{episode:03d}.pt"
        frames = torch.arange(10, dtype=torch.int64) * 16
        torch.save({"features": _features(), "decision_frame_indices": frames}, feature_path)
        digest = hashlib.sha256(feature_path.read_bytes()).hexdigest()
        metadata = {
            "schema_version": "putback_init_wam_embedding_bank_v1",
            "episode": episode,
            "initialization_fingerprint": "init-42",
            "decision_frame_indices": frames.tolist(),
            "feature_dim": 2,
            "feature_sha256": digest,
        }
        (bank / "episodes" / f"episode_{episode:03d}.json").write_text(json.dumps(metadata))
        rows.append({"episode": episode, "feature_sha256": digest, "decisions": 10})
    (bank / "bank_manifest.json").write_text(json.dumps({"schema_version": "putback_init_wam_embedding_bank_v1", "complete": True, "episode_count": 50, "initialization_fingerprint": "init-42", "episodes": rows}))
    for episode in range(40, 50):
        path = dataset / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"episode_index": [episode] * 160, "frame_index": list(range(160)), "action": _actions()}), path)
    command = [sys.executable, str(SCRIPT), "--bank", str(bank), "--dataset-root", str(dataset), "--output", str(output)]

    first = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)

    assert first.returncode == 0, first.stderr
    report = json.loads((output / "report.json").read_text())
    assert report["schema_version"] == "putback_wam_warmup_comparison_v1"
    assert report["contract"]["calibration_episodes"] == list(range(40))
    assert report["contract"]["heldout_episodes"] == list(range(40, 50))
    assert report["action_loaded_episodes"] == list(range(40, 50))
    assert report["calibration"]["sample_count"] == 280
    assert set(report["strategies"]) == {"current", "short_history", "adjacent_cosine"}
    assert isinstance(report["decision_rule"]["advance_to_training_boundary_experiment"], bool)
    assert report["bank_manifest_sha256"] == hashlib.sha256((bank / "bank_manifest.json").read_bytes()).hexdigest()

    second = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert second.returncode != 0
    assert "refusing to overwrite" in second.stderr
