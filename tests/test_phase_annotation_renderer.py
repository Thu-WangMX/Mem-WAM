from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

from scripts.render_putback_phase_annotation import render_annotation_video
from tests.test_phase_annotation import _episode


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe is required")
def test_annotation_renderer_produces_decodable_h264(tmp_path):
    frames = []
    for index in range(31):
        image = np.zeros((120, 480, 3), dtype=np.uint8)
        image[:, :160, 0] = index * 5
        image[:, 160:320, 1] = index * 5
        image[:, 320:, 2] = index * 5
        frames.append(image)
    actions = np.zeros((31, 14), dtype=np.float32)
    actions[5:, 6] = 1.0
    actions[20:, 13] = 1.0
    proprio = np.linspace(0, 1, 31 * 14, dtype=np.float32).reshape(31, 14)
    output = tmp_path / "episode_030_annotation.mp4"

    render_annotation_video(
        frames=frames,
        frame_indices=[index * 4 for index in range(31)],
        actions=actions,
        proprio=proprio,
        annotation=_episode(),
        output=output,
        fps=10,
    )

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,pix_fmt",
            "-of",
            "csv=p=0",
            str(output),
        ],
        text=True,
        capture_output=True,
    )
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip().startswith("h264,")
