from pathlib import Path
import os
import subprocess


REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "ops" / "diagnose_putback_30k_frozen_init_scorer_four.sh"


def test_frozen_init_launcher_preflight_has_exact_paired_contract(tmp_path) -> None:
    checkpoint = tmp_path / "step_030000.pt"
    checkpoint.write_bytes(b"checkpoint")

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        env=os.environ
        | {
            "REPO_ROOT": str(REPO),
            "CHECKPOINT": str(checkpoint),
            "PREFLIGHT_ONLY": "1",
        },
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "scorer_source=initialization" in result.stdout
    assert "checkpoint_step=30000" in result.stdout
    assert "pairs=100000:1000 200000:1002 300000:1004 400000:1006" in result.stdout


def test_frozen_init_launcher_has_valid_shell_syntax() -> None:
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
