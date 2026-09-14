from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "ops" / "diagnose_putback_wam_embedding_surprise_two.sh"


def test_launcher_preflight_reports_locked_two_episode_contract(tmp_path):
    environment = os.environ.copy()
    environment.update(
        {
            "PREFLIGHT_ONLY": "1",
            "REPO_ROOT": str(ROOT),
            "OUTPUT_DIR": str(tmp_path / "result"),
            "CUDA_VISIBLE_DEVICES": "0,1",
        }
    )
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "episodes=40,41" in result.stdout
    assert "model_source=initialization" in result.stdout
    assert "policy_checkpoint=null" in result.stdout
    assert "feature_window=8" in result.stdout
    assert "statistics_window=8" in result.stdout
    assert "threshold_window=8" in result.stdout
    assert "gamma=1.0" in result.stdout
    assert "segment_range=2..8" in result.stdout
    assert "gpus=0,1" in result.stdout
    assert "preflight_status=ok" in result.stdout
    assert not (tmp_path / "result").exists()
