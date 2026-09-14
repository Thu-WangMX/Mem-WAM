from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "ops" / "train_battery_native_cache_25k.sh"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _preflight_env(tmp_path: Path) -> tuple[dict[str, str], Path]:
    asset_root = tmp_path / "battery_try"
    dataset = asset_root / "lerobot" / "battery_try"
    stats = asset_root / "stats" / "dataset_stats.json"
    text_cache = asset_root / "text_cache"
    latents = asset_root / "temporal_fullkv_continuous_episode_stride16_v4"
    action_init = tmp_path / "ActionDiT.pt"
    model_base = tmp_path / "model_base"
    output_dir = tmp_path / "formal_output"

    _write_json(
        asset_root / "_build_manifest.json",
        {
            "color_contract": "simulator_rgb_preserved",
            "episodes_per_task": 50,
        },
    )
    _write_json(asset_root / "_build_logs" / "rgb_contract.json", {"passed": True})
    _write_json(dataset / "meta" / "info.json", {"total_episodes": 50})
    _write_json(stats, {"action": {"default": {}}})
    _write_json(
        latents / "manifest.json",
        {
            "metadata": {
                "schema_version": "fastwam_full_kv_continuous_episode_vae_latents_v4",
                "complete": True,
                "episode_count": 50,
                "replan_stride": 16,
                "color_contract": "simulator_rgb_preserved",
            }
        },
    )
    text_cache.mkdir(parents=True)
    text_cache_backing = tmp_path / "text_cache_backing" / "prompt.pt"
    text_cache_backing.parent.mkdir()
    text_cache_backing.write_bytes(b"text-embedding")
    (text_cache / "prompt.pt").symlink_to(text_cache_backing)
    action_init.write_bytes(b"action-dit")
    model_base.mkdir()

    env = os.environ.copy()
    env.update(
        {
            "REPO_ROOT": str(REPO_ROOT),
            "RUNTIME_BIN": str(Path(sys.executable).parent),
            "OUTPUT_DIR": str(output_dir),
            "DIFFSYNTH_MODEL_BASE_PATH": str(model_base),
            "RMBENCH_BATTERY_LEROBOT": str(dataset),
            "MEMORYWAM_BATTERY_STATS": str(stats),
            "MEMORYWAM_BATTERY_TEXT_CACHE": str(text_cache),
            "MEMORYWAM_BATTERY_CONTINUOUS_LATENTS": str(latents),
            "FASTWAM_ACTION_DIT_INIT": str(action_init),
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            "PREFLIGHT_ONLY": "1",
        }
    )
    return env, output_dir


def _run_launcher(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_battery_launcher_preflight_validates_contract_without_creating_output(tmp_path: Path):
    """Catches launchers that skip Battery asset/config validation or start a run."""
    env, output_dir = _preflight_env(tmp_path)

    result = _run_launcher(env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "task=rmbench_battery_native_cache_25k" in result.stdout
    assert "episodes=50" in result.stdout
    assert "max_steps=25000" in result.stdout
    assert "memory_mode=layerwise" in result.stdout
    assert "memory_tokens=32" in result.stdout
    assert "group_size=4" in result.stdout
    assert "anchor_frames=2" in result.stdout
    assert "recent_frames=4" in result.stdout
    assert "recursive=false" in result.stdout
    assert "minimum_history_frames=1" in result.stdout
    assert "global_batch_size=8" in result.stdout
    assert "gradient_checkpointing=true" in result.stdout
    assert "resume=null" in result.stdout
    assert "checkpoint_steps=1000,5000,10000,15000,20000,25000" in result.stdout
    assert "preflight_status=ok" in result.stdout
    assert not output_dir.exists()


def test_battery_launcher_refuses_to_overwrite_formal_output(tmp_path: Path):
    """Catches accidental reuse of a completed or in-progress formal run directory."""
    env, output_dir = _preflight_env(tmp_path)
    output_dir.mkdir()

    result = _run_launcher(env)

    assert result.returncode == 2
    assert "Refusing to overwrite existing formal output" in result.stderr


def test_battery_resume_preflight_accepts_complete_state_in_matching_output(tmp_path: Path):
    """Catches resume launchers that reject a valid FSDP state or start fresh."""
    env, output_dir = _preflight_env(tmp_path)
    state_dir = output_dir / "checkpoints" / "state" / "step_005000"
    required = ["trainer_state.json", "scheduler.bin"]
    required += [f"pytorch_model_fsdp_0/__{rank}_0.distcp" for rank in range(8)]
    required += [f"optimizer_0/__{rank}_0.distcp" for rank in range(8)]
    required += [f"random_states_{rank}.pkl" for rank in range(8)]
    for relative in required:
        path = state_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"state")
    env["RESUME_STATE"] = str(state_dir)

    result = _run_launcher(env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"resume={state_dir}" in result.stdout
    assert "preflight_status=ok" in result.stdout


def test_battery_resume_preflight_rejects_incomplete_state(tmp_path: Path):
    """Catches attempts to resume from a partially written distributed checkpoint."""
    env, output_dir = _preflight_env(tmp_path)
    state_dir = output_dir / "checkpoints" / "state" / "step_005000"
    state_dir.mkdir(parents=True)
    (state_dir / "trainer_state.json").write_bytes(b"state")
    env["RESUME_STATE"] = str(state_dir)

    result = _run_launcher(env)

    assert result.returncode == 2
    assert "Incomplete resume state" in result.stderr
