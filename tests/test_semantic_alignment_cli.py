from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_putback_wam_semantic_alignment.py"


def _actions() -> list[list[float]]:
    values = [[0.0] * 14 for _ in range(348)]
    for row in values:
        row[6] = 1.0
        row[13] = 1.0
    for frame in range(54, 348):
        values[frame][13] = 0.0
    for frame in range(122, 348):
        values[frame][13] = 1.0
    for frame in range(176, 348):
        values[frame][6] = 0.0
    for frame in range(267, 348):
        values[frame][13] = 0.0
    for frame in range(334, 348):
        values[frame][13] = 1.0
    return values


def _write_fixture(root: Path, episode: int) -> None:
    dataset_file = root / "dataset" / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
    dataset_file.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": [episode] * 348,
                "frame_index": list(range(348)),
                "action": _actions(),
            }
        ),
        dataset_file,
    )
    episodes = root / "analysis" / "episodes"
    episodes.mkdir(parents=True, exist_ok=True)
    trace = {
        "schema_version": "putback_wam_embedding_surprise_init_v1",
        "episode": episode,
        "decision_count": 22,
        "decision_frame_indices": list(range(0, 22 * 16, 16)),
        "scores": [None] * 22,
        "thresholds": [None] * 22,
        "score_source_end": [None] * 22,
        "threshold_source_end": [None] * 22,
        "peaks": [10, 13],
        "peak_confirmed_at": [11, 14],
        "boundaries": [0, 8, 10, 13, 20, 22],
        "reasons": {"0": "start", "8": "max_length", "10": "surprise", "13": "surprise", "20": "max_length", "22": "end"},
        "segment_lengths": [8, 2, 3, 7, 2],
    }
    (episodes / f"episode_{episode:03d}.json").write_text(json.dumps(trace))


def test_cli_writes_provenance_micro_metrics_and_refuses_overwrite(tmp_path):
    for episode in (40, 41):
        _write_fixture(tmp_path, episode)
    command = [
        sys.executable,
        str(SCRIPT),
        "--analysis-root",
        str(tmp_path / "analysis"),
        "--dataset-root",
        str(tmp_path / "dataset"),
        "--episodes",
        "40,41",
    ]

    first = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)

    assert first.returncode == 0, first.stderr
    report = json.loads((tmp_path / "analysis" / "semantic_alignment.json").read_text())
    assert report["schema_version"] == "putback_wam_surprise_semantic_alignment_v1"
    assert [row["episode"] for row in report["episodes"]] == [40, 41]
    assert all(len(row["source_trace_sha256"]) == 64 for row in report["episodes"])
    assert report["aggregate"]["primary"]["counts"] == {
        "confirmations": 4,
        "events": 10,
        "matched": 2,
    }
    assert report["aggregate"]["post_warmup"]["counts"] == {
        "confirmations": 4,
        "events": 6,
        "matched": 2,
    }
    assert report["aggregate"]["retroactive_boundary_count"] == 4

    second = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    assert second.returncode != 0
    assert "refusing to overwrite" in second.stderr
