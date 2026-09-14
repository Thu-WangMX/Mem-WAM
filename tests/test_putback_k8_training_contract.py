from __future__ import annotations

import subprocess
from pathlib import Path

from test_putback_training_contract import REPO_ROOT, _preflight_env


LAUNCHER = REPO_ROOT / "ops" / "train_putback_layerwise_k8_25k.sh"


def _run_launcher(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_putback_k8_fresh_preflight_reports_exact_training_contract(tmp_path: Path):
    """Catches accidental K=32 reuse or any drift from the formal PutBack run."""
    env, output_dir = _preflight_env(tmp_path)
    output_dir = tmp_path / "layerwise_k8_output"
    env["OUTPUT_DIR"] = str(output_dir)

    result = _run_launcher(env)

    assert result.returncode == 0, result.stdout + result.stderr
    for expected in (
        "task=rmbench_putback_layerwise_k8_25k",
        "episodes=50",
        "max_steps=25000",
        "memory_mode=layerwise",
        "memory_tokens=8",
        "group_size=4",
        "anchor_frames=2",
        "recent_frames=4",
        "recursive=false",
        "minimum_history_frames=1",
        "global_batch_size=8",
        "checkpoint_steps=1000,5000,10000,15000,20000,25000",
        "resume=null",
        f"output_dir={output_dir}",
        "preflight_status=ok",
    ):
        assert expected in result.stdout
    assert not output_dir.exists()


def test_putback_k8_launcher_has_a_distinct_default_output():
    """Catches a K=8 launch that could overwrite the completed K=32 experiment."""
    source = LAUNCHER.read_text(encoding="utf-8")

    assert "layerwise_block_memory_k8_putback_e2e_25k_seed42" in source
    assert "layerwise_block_memory_putback_e2e_25k_seed42}" not in source

