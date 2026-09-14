from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "ops" / "train_putback_control_information_anchor_recent_k8_40k.sh"
MANIFEST = Path(
    "/mnt/vepfs02/output/kevin.wang/memorywam/data/"
    "putback_control_information_planning_segments_v3"
)


def test_anchor_recent_launcher_preflight_locks_v4_contract(tmp_path):
    assert LAUNCHER.is_file(), "formal v4 anchor/recent launcher is missing"
    output = tmp_path / "must_not_be_created"
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env={
            **os.environ,
            "REPO_ROOT": str(ROOT),
            "OUTPUT_DIR": str(output),
            "MANIFEST_ROOT": str(MANIFEST),
            "PREFLIGHT_ONLY": "1",
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            "DIFFSYNTH_MODEL_BASE_PATH": (
                "/mnt/vepfs02/output/kevin.wang/memorywam/model_assets/"
                "diffsynth_official"
            ),
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    for expected in (
        "task=rmbench_putback_control_information_anchor_recent_k8_40k",
        "implementation=dynamic_k8_anchor2_recent4_v4",
        "selector=locked_segment_relative_control_information_v3",
        "runtime_lock_sha256=c27e63084f3b579f1d04d6ca62126ca8caa8f6032ae94286cab196402c6cfb96",
        "episodes=50",
        "max_steps=40000",
        "memory_tokens=8 anchor_frames=2 recent_frames=4",
        "checkpoint_steps=5000,10000,15000,20000,25000,30000,35000,40000",
        "save_training_state_every=10000 keep_training_state_checkpoints=1",
        f"output_dir={output}",
        "preflight_status=ok",
    ):
        assert expected in result.stdout
    assert "checkpoint_steps=1000," not in result.stdout
    assert not output.exists()


def test_anchor_recent_launcher_uses_isolated_new_user_paths_and_is_shell_valid():
    assert LAUNCHER.is_file(), "formal v4 anchor/recent launcher is missing"
    source = LAUNCHER.read_text(encoding="utf-8")

    assert (
        "/mnt/vepfs02/output/kevin.wang/memorywam/code/"
        "fastwam_control_information_anchor_recent"
    ) in source
    assert (
        "/mnt/vepfs02/output/kevin.wang/memorywam/train/"
        "control_information_v4_anchor_recent_k8_putback_e2e_40k_seed42"
    ) in source
    assert "control_information_v3_k8_putback_e2e_40k_seed42" not in source
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
