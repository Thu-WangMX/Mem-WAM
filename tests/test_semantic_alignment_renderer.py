from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "render_putback_wam_semantic_alignment.py"


def _video(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    assert writer.isOpened()
    for index in range(8):
        frame = np.full((48, 64, 3), color, dtype=np.uint8)
        cv2.putText(frame, str(index), (2, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        writer.write(frame)
    writer.release()


def test_renderer_emits_semantic_alignment_h264_video(tmp_path):
    analysis = tmp_path / "analysis"
    dataset = tmp_path / "dataset"
    (analysis / "episodes").mkdir(parents=True)
    for camera, color in {
        "observation.images.cam_high": (20, 40, 80),
        "observation.images.cam_left_wrist": (40, 80, 20),
        "observation.images.cam_right_wrist": (80, 20, 40),
    }.items():
        _video(dataset / "videos" / "chunk-000" / camera / "episode_000040.mp4", color)
    trace = {
        "episode": 40, "decision_count": 4, "decision_frame_indices": [0, 2, 4, 6],
        "scores": [None, 0.5, 2.0, 0.7], "thresholds": [None, None, 1.0, 1.0],
        "peaks": [2], "peak_confirmed_at": [3], "boundaries": [0, 2, 4],
    }
    (analysis / "episodes" / "episode_040.json").write_text(json.dumps(trace))
    report = {
        "schema_version": "putback_wam_surprise_semantic_alignment_v1",
        "episodes": [{
            "episode": 40,
            "strong_events": [{"raw_frame": 4, "decision": 2.0, "gripper_dim": 13, "kind": "gripper_close"}],
            "confirmations": [3.0],
            "primary_metrics": {"matches": [{"confirmation": 3.0, "event_index": 0, "event_decision": 2.0}], "counts": {"matched": 1, "events": 1, "confirmations": 1}},
            "forced_boundaries": [],
            "retroactive_boundaries": [{"peak_boundary": 2, "available_at": 3, "lag": 1}],
            "original_partition": {"boundaries": [0, 2, 4]},
            "online_counterfactual": {"boundaries": [0, 3, 4]},
            "joint_motion_by_decision": [0.1, 0.2, 0.8, 0.1],
        }],
    }
    (analysis / "semantic_alignment.json").write_text(json.dumps(report))
    output = tmp_path / "semantic.mp4"

    result = subprocess.run([sys.executable, str(SCRIPT), "--analysis-root", str(analysis), "--dataset-root", str(dataset), "--episode", "40", "--output", str(output)], cwd=ROOT, text=True, capture_output=True)

    assert result.returncode == 0, result.stderr
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height:format=duration", "-of", "json", str(output)], text=True, capture_output=True, check=True)
    metadata = json.loads(probe.stdout)
    assert metadata["streams"][0]["codec_name"] == "h264"
    assert metadata["streams"][0]["width"] == 640
    assert metadata["streams"][0]["height"] == 900
    assert float(metadata["format"]["duration"]) > 0
