from __future__ import annotations

import json
from pathlib import Path
import subprocess

from test_putback_training_contract import REPO_ROOT, _preflight_env


LAUNCHER = REPO_ROOT / "ops" / "resume_putback_wrist_event_k8_5k_to40k.sh"


def _write_manifest(root: Path) -> Path:
    episodes = {}
    for episode in range(50):
        relative = f"episodes/episode_{episode:03d}.json"
        episodes[str(episode)] = relative
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "episode": episode,
                    "decision_count": 2,
                    "boundaries": [0, 2],
                    "reasons": {"0": "start", "2": "end"},
                    "segment_lengths": [2],
                    "threshold": 0.5,
                }
            ),
            encoding="utf-8",
        )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "schema_version": "putback_wrist_latent_event_segments_v1",
                    "complete": True,
                    "task": "put_back_block",
                    "episode_count": 50,
                    "replan_stride": 16,
                    "uses_vlm": False,
                },
                "episodes": episodes,
            }
        ),
        encoding="utf-8",
    )
    return root


def _complete_state(output_dir: Path, *, global_step: int = 5000) -> Path:
    state = output_dir / "checkpoints" / "state" / "step_005000"
    state.mkdir(parents=True)
    (state / "trainer_state.json").write_text(
        json.dumps({"global_step": global_step, "epoch": 35, "batch_in_epoch": 100}),
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


def _environment(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    env, output = _preflight_env(tmp_path)
    manifest = _write_manifest(tmp_path / "wrist_manifest")
    state = _complete_state(output)
    env["WRIST_EVENT_MANIFEST"] = str(manifest)
    env["RESUME_STATE"] = str(state)
    return env, output, state


def _run(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_resume_preflight_restores_complete_5k_state_to_absolute_40k(tmp_path):
    env, output, state = _environment(tmp_path)

    result = _run(env)

    assert result.returncode == 0, result.stdout + result.stderr
    for expected in (
        "task=rmbench_putback_wrist_event_k8_40k",
        "episodes=50",
        "resume_step=5000",
        f"resume={state.resolve()}",
        "max_steps=40000",
        "additional_steps=35000",
        "checkpoint_steps=10000,15000,20000,25000,30000,35000,40000",
        "process_group_timeout_seconds=7200",
        "memory_mode=layerwise",
        "memory_tokens=8",
        "minimum_history_frames=1",
        "global_batch_size=8",
        f"output_dir={output}",
        "preflight_status=ok",
    ):
        assert expected in result.stdout


def test_resume_preflight_rejects_wrong_global_step(tmp_path):
    env, output, _ = _environment(tmp_path)
    state = output / "checkpoints" / "state" / "step_005000"
    (state / "trainer_state.json").write_text(
        json.dumps({"global_step": 4999, "epoch": 35, "batch_in_epoch": 100}),
        encoding="utf-8",
    )

    result = _run(env)

    assert result.returncode == 2
    assert "expected global_step=5000" in result.stderr


def test_resume_preflight_rejects_missing_model_shard(tmp_path):
    env, output, _ = _environment(tmp_path)
    missing = (
        output
        / "checkpoints"
        / "state"
        / "step_005000"
        / "pytorch_model_fsdp_0"
        / "__7_0.distcp"
    )
    missing.unlink()

    result = _run(env)

    assert result.returncode == 2
    assert "Incomplete resume state" in result.stderr


def test_resume_launcher_has_valid_shell_syntax():
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
