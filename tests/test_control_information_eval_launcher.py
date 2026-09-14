from __future__ import annotations

import os
from pathlib import Path
import subprocess
import zipfile

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "ops" / "eval_putback_control_information_v3_rmbench.sh"
MANIFEST = Path(
    "/mnt/vepfs02/output/kevin.wang/memorywam/data/"
    "putback_control_information_planning_segments_v3"
)
RUNTIME_LOCK = Path(
    "/mnt/vepfs02/output/kevin.wang/memorywam/analysis/"
    "putback_control_information_selector_v3/selector_runtime/locked_selector.json"
)


def _write_portable_checkpoint(path: Path) -> None:
    path.parent.mkdir(parents=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{path.stem}/data.pkl", b"portable checkpoint fixture")
        archive.writestr(f"{path.stem}/version", b"3\n")
        archive.writestr(f"{path.stem}/data/0", b"tensor fixture")


def _write_training_config(path: Path) -> None:
    cfg = OmegaConf.create(
        {
            "max_steps": 40000,
            "save_every": 5000,
            "save_steps": [],
            "seed": 42,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 2e-4,
            "mixed_precision": "bf16",
            "native_cache_train_mode": "full",
            "data": {
                "train": {
                    "minimum_history_frames": 1,
                    "surprise_manifest_path": str(MANIFEST),
                }
            },
            "model": {
                "native_cache": {
                    "enabled": True,
                    "mode": "layerwise",
                    "memory_tokens": 8,
                    "group_size": 4,
                    "anchor_frames": 2,
                    "recent_frames": 4,
                    "recursive": False,
                }
            },
        }
    )
    OmegaConf.save(cfg, path)


def test_v3_eval_preflight_rejects_mismatch_and_locks_online_contract(tmp_path):
    assert LAUNCHER.is_file(), "formal v3 RM-Bench launcher is missing"
    train_root = tmp_path / "control_information_v3_k8_putback_e2e_40k_seed42"
    checkpoint = train_root / "checkpoints" / "weights" / "step_005000.pt"
    _write_portable_checkpoint(checkpoint)
    _write_training_config(train_root / "config.yaml")
    log_root = tmp_path / "must_not_be_created"

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env={
            **os.environ,
            "REPO_ROOT": str(ROOT),
            "CHECKPOINT": str(checkpoint),
            "MANIFEST_ROOT": str(MANIFEST),
            "RUNTIME_LOCK": str(RUNTIME_LOCK),
            "LOG_ROOT": str(log_root),
            "GPUS": "0,1,2,3",
            "PAIR_INDICES": "0,1,2,3",
            "PREFLIGHT_ONLY": "1",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    for expected in (
        "selector=segment_relative_counterfactual_control_information",
        "selector_runtime=v3",
        "selector_source=frozen_initialization_wam",
        "strict_online=true forward_alignment=true",
        "old_dynamic_surprise=false gripper_hard_trigger=false",
        "runtime_lock_sha256=c27e63084f3b579f1d04d6ca62126ca8caa8f6032ae94286cab196402c6cfb96",
        "memory_tokens=8 detector_stride=4 replan_steps=16",
        "segment_frame_range=32..96",
        "checkpoint_step=5000 portable_checkpoint=true",
        "scene_policy_pairs=100000:1000,100001:1001,100002:1002,100003:1003",
        "preflight_status=ok",
    ):
        assert expected in result.stdout
    assert not log_root.exists()


def test_v3_eval_launcher_uses_new_user_root_and_is_shell_valid():
    assert LAUNCHER.is_file(), "formal v3 RM-Bench launcher is missing"
    source = LAUNCHER.read_text(encoding="utf-8")

    assert (
        "/mnt/vepfs02/output/kevin.wang/memorywam/train/"
        "control_information_v3_k8_putback_e2e_40k_seed42"
    ) in source
    assert "--config-name sim_robotwin_control_information_v3" in source
    assert "dynamic_surprise_online: false" not in source
    assert '} | tee "${summary}"' not in source
    assert '} >"${summary}"' in source
    assert 'cat "${summary}"' in source
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
