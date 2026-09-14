from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "ops" / "build_putback_four_phase_latents_8gpu.sh"


def test_launcher_enforces_eight_gpu_immutable_preflight(tmp_path):
    dataset = tmp_path / "dataset"
    (dataset / "meta").mkdir(parents=True)
    raw = tmp_path / "episode.hdf5"
    raw.write_bytes(b"hdf5")
    (dataset / "meta" / "episodes.jsonl").write_text(
        json.dumps(
            {
                "episode_index": 0,
                "length": 37,
                "raw_file_name": str(raw),
            }
        )
        + "\n"
    )
    vae = tmp_path / "wan_vae.pt"
    vae.write_bytes(b"vae")
    runtime = Path(os.sys.executable).parent
    env = {
        **os.environ,
        "REPO_ROOT": str(ROOT),
        "OUTPUT_DIR": str(tmp_path / "four_phase"),
        "LEROBOT_ROOT": str(dataset),
        "VAE_PATH": str(vae),
        "RUNTIME_BIN": str(runtime),
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "MIN_FREE_GIB": "0",
        "PREFLIGHT_ONLY": "1",
    }

    result = subprocess.run(
        ["bash", str(LAUNCHER)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert result.returncode == 0, result.stderr
    assert "preflight_status=ok" in result.stdout
    assert "phase_offsets=0,4,8,12" in result.stdout
    assert "video_expert_frame_stride=16" in result.stdout
    assert "policy_checkpoint=null" in result.stdout
    assert "MIN_FREE_GIB=${MIN_FREE_GIB:-100}" in LAUNCHER.read_text()

    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    rejected = subprocess.run(
        ["bash", str(LAUNCHER)], cwd=ROOT, env=env, text=True, capture_output=True
    )
    assert rejected.returncode != 0
    assert "exactly eight" in rejected.stderr


def test_launcher_refuses_a_completed_output_root(tmp_path):
    dataset = tmp_path / "dataset"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta" / "episodes.jsonl").write_text("{}\n")
    vae = tmp_path / "wan_vae.pt"
    vae.write_bytes(b"vae")
    output = tmp_path / "four_phase"
    output.mkdir()
    (output / "manifest.json").write_text(json.dumps({"complete": True}))
    env = {
        **os.environ,
        "REPO_ROOT": str(ROOT),
        "OUTPUT_DIR": str(output),
        "LEROBOT_ROOT": str(dataset),
        "VAE_PATH": str(vae),
        "RUNTIME_BIN": str(Path(os.sys.executable).parent),
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "MIN_FREE_GIB": "0",
        "PREFLIGHT_ONLY": "1",
    }

    result = subprocess.run(
        ["bash", str(LAUNCHER)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert result.returncode != 0
    assert "completed output" in result.stderr
