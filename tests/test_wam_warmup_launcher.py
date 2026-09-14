from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "ops" / "analyze_putback_wam_warmup_comparison.sh"


def test_warmup_launcher_preflight_is_cpu_only_and_frozen(tmp_path):
    bank = tmp_path / "bank"
    dataset = tmp_path / "dataset"
    bank.mkdir()
    (bank / "bank_manifest.json").write_text(json.dumps({"complete": True}))
    for episode in range(40, 50):
        path = dataset / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        for camera in (
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        ):
            video = dataset / "videos" / "chunk-000" / camera / f"episode_{episode:06d}.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(b"x")
    env = {
        **os.environ,
        "REPO_ROOT": str(ROOT),
        "BANK_ROOT": str(bank),
        "DATASET_ROOT": str(dataset),
        "OUTPUT_DIR": str(tmp_path / "comparison"),
        "RUNTIME_BIN": str(Path(os.sys.executable).parent),
        "PREFLIGHT_ONLY": "1",
    }

    result = subprocess.run(["bash", str(LAUNCHER)], cwd=ROOT, env=env, text=True, capture_output=True)

    assert result.returncode == 0, result.stderr
    assert "preflight_status=ok" in result.stdout
    assert "calibration_episodes=0-39" in result.stdout
    assert "heldout_episodes=40-49" in result.stdout
    assert "policy_checkpoint=null" in result.stdout
    assert "gpu" not in result.stdout.lower()
