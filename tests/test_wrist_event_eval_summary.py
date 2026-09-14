from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from fastwam.evaluation.wrist_event_summary import summarize_wrist_event_run


PAIRS = ((100000, 1000), (200000, 1002), (1300000, 1008), (1400000, 1010))
REPO = Path(__file__).resolve().parents[1]
SUMMARY_CLI = REPO / "scripts" / "summarize_putback_wrist_event_eval.py"


def _write_scene(root: Path, scene: int, policy: int, *, success: bool) -> None:
    success_count = 1 if success else 0
    log = root / f"scene{scene}_policy{policy}.log"
    log.write_text(
        "\n".join(
            [
                'FASTWAM_WRIST_EVENT_ARRIVAL {"arrival": 0, "close_range": null, "reason": null, "scheduled_probability": null}',
                'FASTWAM_WRIST_EVENT_FORECAST {"probability": 0.61, "source_decision": 0, "target_decision": 1}',
                'FASTWAM_WRIST_EVENT_ARRIVAL {"arrival": 2, "close_range": [0, 2], "reason": "wrist_event", "scheduled_probability": 0.61}',
                'FASTWAM_NATIVE_CACHE_METRICS {"frame_index": 2, "memory_tokens": 8, "replan": 3}',
                f"Success rate: [96m{success_count}/1[0m => 100.0%",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    video = (
        root
        / "artifacts"
        / "step_005000"
        / f"scene{scene}_policy{policy}"
        / "put_back_block"
        / f"episode0_randomized-false_success-{str(success).lower()}.mp4"
    )
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"fake-mp4")


def test_summary_uses_official_success_and_collects_event_metrics_and_videos(tmp_path):
    for index, (scene, policy) in enumerate(PAIRS):
        _write_scene(tmp_path, scene, policy, success=index < 3)

    summary = summarize_wrist_event_run(tmp_path, PAIRS)

    assert summary["complete"] is True
    assert summary["successes"] == 3
    assert summary["episodes"] == 4
    assert summary["success_rate"] == 0.75
    assert summary["total_event_closures"] == 4
    assert summary["total_forecasts"] == 4
    assert len(summary["videos"]) == 4
    assert summary["scenes"][3]["official_success"] is False
    assert summary["scenes"][3]["video_success"] is False


def test_summary_marks_traceback_or_missing_official_result_incomplete(tmp_path):
    scene, policy = PAIRS[0]
    (tmp_path / f"scene{scene}_policy{policy}.log").write_text(
        "Traceback (most recent call last):\nRuntimeError: boom\n",
        encoding="utf-8",
    )

    summary = summarize_wrist_event_run(tmp_path, (PAIRS[0],))

    assert summary["complete"] is False
    assert summary["scenes"][0]["traceback"] is True
    assert summary["scenes"][0]["official_success"] is None


def test_summary_is_json_serializable(tmp_path):
    _write_scene(tmp_path, *PAIRS[0], success=True)

    payload = summarize_wrist_event_run(tmp_path, (PAIRS[0],))

    json.dumps(payload)


def test_summary_cli_writes_json_atomically_and_fails_incomplete_run(tmp_path):
    scene, policy = PAIRS[0]
    output = tmp_path / "summary.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SUMMARY_CLI),
            "--run-root",
            str(tmp_path),
            "--scene-pair",
            f"{scene}:{policy}",
            "--output",
            str(output),
        ],
        env=os.environ | {"PYTHONPATH": f"{REPO / 'src'}:{REPO}"},
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert json.loads(output.read_text(encoding="utf-8"))["complete"] is False
    assert not output.with_suffix(".json.tmp").exists()
