from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "diagnose_putback_wam_embedding_surprise.py"


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_runner_prints_locked_initialization_only_contract():
    result = _run("--print-contract")

    assert result.returncode == 0, result.stderr
    contract = json.loads(result.stdout)
    assert contract == {
        "episodes": [40, 41],
        "feature_layer": -1,
        "feature_window": 8,
        "gamma": 1.0,
        "max_segment": 8,
        "min_history": 4,
        "min_segment": 2,
        "nms_distance": 2,
        "policy_checkpoint": None,
        "seed": 42,
        "statistics_window": 8,
        "threshold_window": 8,
    }


def test_runner_has_no_policy_checkpoint_cli_escape_hatch():
    result = _run("--checkpoint", "/tmp/trained.pt", "--print-contract")

    assert result.returncode == 2
    assert "unrecognized arguments: --checkpoint" in result.stderr
