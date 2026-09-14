from __future__ import annotations

import json
import subprocess
from pathlib import Path

from test_putback_training_contract import REPO_ROOT, _preflight_env


LAUNCHER = REPO_ROOT / "ops" / "train_putback_layerwise_k8_continue_40k.sh"


def _complete_state(output_dir: Path, *, global_step: int = 25000) -> Path:
    state = output_dir / "checkpoints" / "state" / "step_025000"
    state.mkdir(parents=True)
    (state / "trainer_state.json").write_text(
        json.dumps({"global_step": global_step, "epoch": 178, "batch_in_epoch": 80}),
        encoding="utf-8",
    )
    (state / "scheduler.bin").write_bytes(b"scheduler")
    for rank in range(8):
        model = state / "pytorch_model_fsdp_0" / f"__{rank}_0.distcp"
        optimizer = state / "optimizer_0" / f"__{rank}_0.distcp"
        random_state = state / f"random_states_{rank}.pkl"
        model.parent.mkdir(exist_ok=True)
        optimizer.parent.mkdir(exist_ok=True)
        model.write_bytes(b"model")
        optimizer.write_bytes(b"optimizer")
        random_state.write_bytes(b"random")
    return state


def _run(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_k8_continue_preflight_resumes_25k_and_targets_40k(tmp_path: Path):
    """Catches a continuation that restarts weights or counts 40k extra steps."""
    env, output_dir = _preflight_env(tmp_path)
    state = _complete_state(output_dir)
    env["RESUME_STATE"] = str(state)

    result = _run(env)

    assert result.returncode == 0, result.stdout + result.stderr
    for expected in (
        "task=rmbench_putback_layerwise_k8_continue_40k",
        "episodes=50",
        "resume_step=25000",
        f"resume={state}",
        "max_steps=40000",
        "additional_steps=15000",
        "checkpoint_steps=30000,35000,40000",
        "memory_mode=layerwise",
        "memory_tokens=8",
        "group_size=4",
        "minimum_history_frames=1",
        "global_batch_size=8",
        f"output_dir={output_dir}",
        "preflight_status=ok",
    ):
        assert expected in result.stdout
    assert not (output_dir / "checkpoints" / "state" / "step_030000").exists()


def test_k8_continue_rejects_state_whose_global_step_is_not_25k(tmp_path: Path):
    """Catches accidentally resuming from a stale or later checkpoint."""
    env, output_dir = _preflight_env(tmp_path)
    state = _complete_state(output_dir, global_step=24999)
    env["RESUME_STATE"] = str(state)

    result = _run(env)

    assert result.returncode == 2
    assert "expected global_step=25000" in result.stderr
