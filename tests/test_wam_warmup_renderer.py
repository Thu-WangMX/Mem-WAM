from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "render_putback_wam_warmup_comparison.py"


def _video(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    assert writer.isOpened()
    for index in range(8):
        frame = np.full((48, 64, 3), color, dtype=np.uint8)
        cv2.putText(frame, str(index), (2, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        writer.write(frame)
    writer.release()


def test_renderer_emits_three_strategy_h264_video(tmp_path):
    comparison = tmp_path / "comparison"
    dataset = tmp_path / "dataset"
    for camera, color in {
        "observation.images.cam_high": (20, 40, 80),
        "observation.images.cam_left_wrist": (40, 80, 20),
        "observation.images.cam_right_wrist": (80, 20, 40),
    }.items():
        _video(dataset / "videos" / "chunk-000" / camera / "episode_000040.mp4", color)
    strategies = {}
    for name, confirmations in {"current": [3], "short_history": [2, 3], "adjacent_cosine": [2]}.items():
        path = comparison / "traces" / name
        path.mkdir(parents=True, exist_ok=True)
        (path / "episode_040.json").write_text(json.dumps({"episode": 40, "decision_frame_indices": [0, 2, 4, 6], "scores": [None, 0.2, 1.0, 0.3], "thresholds": [None, None, 0.5, 0.5], "peaks": [value - 1 for value in confirmations], "peak_confirmed_at": confirmations, "boundaries": [0, 2, 4]}))
        strategies[name] = {"per_episode": [{"episode": 40, "events": [{"decision": 2.0, "kind": "gripper_close", "gripper_dim": 13}], "all": {"counts": {"matched": 1, "events": 1, "confirmations": len(confirmations)}}}]}
    comparison.mkdir(exist_ok=True)
    (comparison / "report.json").write_text(json.dumps({"schema_version": "putback_wam_warmup_comparison_v1", "calibration": {"threshold": 0.5}, "strategies": strategies, "decision_rule": {"advance_to_training_boundary_experiment": False}}))
    output = tmp_path / "warmup.mp4"

    result = subprocess.run([sys.executable, str(SCRIPT), "--comparison-root", str(comparison), "--dataset-root", str(dataset), "--episode", "40", "--output", str(output)], cwd=ROOT, text=True, capture_output=True)

    assert result.returncode == 0, result.stderr
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height:format=duration", "-of", "json", str(output)], text=True, capture_output=True, check=True)
    metadata = json.loads(probe.stdout)
    assert metadata["streams"][0]["codec_name"] == "h264"
    assert metadata["streams"][0]["width"] == 640
    assert metadata["streams"][0]["height"] == 900
    assert float(metadata["format"]["duration"]) > 0
