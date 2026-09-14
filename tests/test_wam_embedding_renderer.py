from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "render_putback_wam_embedding_surprise.py"


def _write_video(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48)
    )
    assert writer.isOpened()
    for index in range(8):
        frame = np.full((48, 64, 3), color, dtype=np.uint8)
        cv2.putText(frame, str(index), (3, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        writer.write(frame)
    writer.release()


def test_renderer_emits_annotated_h264_three_camera_video(tmp_path):
    dataset = tmp_path / "dataset"
    analysis = tmp_path / "analysis"
    (analysis / "episodes").mkdir(parents=True)
    cameras = {
        "observation.images.cam_high": (20, 40, 80),
        "observation.images.cam_left_wrist": (40, 80, 20),
        "observation.images.cam_right_wrist": (80, 20, 40),
    }
    for camera, color in cameras.items():
        _write_video(
            dataset / "videos" / "chunk-000" / camera / "episode_000040.mp4",
            color,
        )
    payload = {
        "episode": 40,
        "decision_count": 4,
        "decision_frame_indices": [0, 2, 4, 6],
        "scores": [None, None, 1.0, 3.0],
        "thresholds": [None, None, None, 2.0],
        "peaks": [2],
        "peak_confirmed_at": [3],
        "boundaries": [0, 2, 4],
        "reasons": {"0": "start", "2": "surprise", "4": "end"},
        "segment_lengths": [2, 2],
    }
    (analysis / "episodes" / "episode_040.json").write_text(json.dumps(payload))
    output = tmp_path / "annotated.mp4"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--analysis-root",
            str(analysis),
            "--dataset-root",
            str(dataset),
            "--episode",
            "40",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height:format=duration",
            "-of",
            "json",
            str(output),
        ],
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    )
    metadata = json.loads(probe.stdout)
    assert metadata["streams"][0]["codec_name"] == "h264"
    assert metadata["streams"][0]["width"] == 640
    assert metadata["streams"][0]["height"] == 840
    assert float(metadata["format"]["duration"]) > 0
