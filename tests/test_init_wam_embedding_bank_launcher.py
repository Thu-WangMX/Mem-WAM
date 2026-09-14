from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "ops" / "extract_putback_init_wam_embedding_bank_8gpu.sh"


def test_bank_launcher_preflight_requires_exact_eight_gpu_contract(tmp_path):
    latent = tmp_path / "latent"
    text = tmp_path / "text"
    dataset = tmp_path / "dataset"
    model_base = tmp_path / "models"
    latent.mkdir()
    text.mkdir()
    dataset.mkdir()
    (model_base / "Wan-AI" / "Wan2.2-TI2V-5B").mkdir(parents=True)
    (latent / "manifest.json").write_text(json.dumps({"complete": True}))
    action_init = tmp_path / "action.pt"
    action_init.write_bytes(b"action")
    init_ref = tmp_path / "initialization_manifest.json"
    init_ref.write_text(json.dumps({"model_source": "initialization", "policy_checkpoint": None}))
    runtime = Path(os.sys.executable).parent
    env = {
        **os.environ,
        "REPO_ROOT": str(ROOT),
        "OUTPUT_DIR": str(tmp_path / "bank"),
        "LATENT_CACHE": str(latent),
        "TEXT_CACHE": str(text),
        "DATASET_ROOT": str(dataset),
        "ACTION_INIT": str(action_init),
        "MODEL_BASE": str(model_base),
        "INIT_REFERENCE": str(init_ref),
        "RUNTIME_BIN": str(runtime),
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "PREFLIGHT_ONLY": "1",
    }

    result = subprocess.run(["bash", str(LAUNCHER)], cwd=ROOT, env=env, text=True, capture_output=True)

    assert result.returncode == 0, result.stderr
    assert "preflight_status=ok" in result.stdout
    assert "gpus=0,1,2,3,4,5,6,7" in result.stdout
    assert "policy_checkpoint=null" in result.stdout

    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    rejected = subprocess.run(["bash", str(LAUNCHER)], cwd=ROOT, env=env, text=True, capture_output=True)
    assert rejected.returncode != 0
    assert "exactly eight" in rejected.stderr
