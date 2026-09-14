from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "ops" / "eval_putback_wrist_event_5k_four_rmbench.sh"


def test_launcher_preflight_accepts_future_5k_without_writing_outputs(tmp_path):
    predictor_repo = tmp_path / "predictor"
    (predictor_repo / "src").mkdir(parents=True)
    predictor_checkpoint = tmp_path / "best.pt"
    predictor_checkpoint.write_bytes(b"future-test-predictor")
    manifest = tmp_path / "boundary_manifest" / "manifest.json"
    manifest.parent.mkdir()
    manifest.write_text("{}\n", encoding="utf-8")
    future_training_root = tmp_path / "future_training"
    log_root = tmp_path / "must_not_exist"

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        env=os.environ
        | {
            "REPO_ROOT": str(REPO),
            "PYTHON_BIN": sys.executable,
            "TRAINING_ROOT": str(future_training_root),
            "PREDICTOR_REPO": str(predictor_repo),
            "PREDICTOR_CHECKPOINT": str(predictor_checkpoint),
            "BOUNDARY_MANIFEST": str(manifest),
            "LOG_ROOT": str(log_root),
            "PREFLIGHT_ONLY": "1",
        },
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "checkpoint_state=waiting" in result.stdout
    assert str(
        future_training_root / "checkpoints" / "weights" / "step_005000.pt"
    ) in result.stdout
    assert (
        "scene_policy_pairs="
        "100000:1000,200000:1002,1300000:1008,1400000:1010"
    ) in result.stdout
    assert "preflight_status=ok" in result.stdout
    assert not log_root.exists()


def test_launcher_has_valid_shell_syntax():
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)


def test_launcher_passes_distinct_physical_gpu_to_robotwin_entrypoint():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert '"gpu_id=${gpu}"' in text
    assert 'CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}"' not in text
