from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

from scripts.render_putback_predictive_selector import render_predictive_selector_video


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe is required")
def test_renderer_produces_valid_h264_three_camera_diagnostic(tmp_path):
    frame_count = 25
    frames = []
    for index in range(frame_count):
        mosaic = np.zeros((120, 480, 3), dtype=np.uint8)
        mosaic[:, :160, 0] = index * 5
        mosaic[:, 160:320, 1] = index * 5
        mosaic[:, 320:, 2] = index * 5
        frames.append(mosaic)
    output = tmp_path / "selector.mp4"
    render_predictive_selector_video(
        frames=frames,
        frame_indices=[index * 4 for index in range(frame_count)],
        scores=np.linspace(0, 5, frame_count, dtype=np.float32),
        high_threshold=3.0,
        low_threshold=1.0,
        transitions=[{"name": "contact", "frame": 32, "ambiguity_start": 28, "ambiguity_end": 36}],
        events=[
            {
                "frame": 36,
                "peak_frame": 32,
                "group_start": 0,
                "group_end": 36,
                "reason": "predictive_surprise_confirmed",
            },
            {
                "frame": 64,
                "peak_frame": 64,
                "group_start": 36,
                "group_end": 64,
                "reason": "forced_maximum",
            },
        ],
        output=output,
        fps=10,
    )
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,pix_fmt", "-of", "csv=p=0", str(output),
        ],
        text=True,
        capture_output=True,
    )
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip().startswith("h264,")
    assert "yuv420p" in probe.stdout

