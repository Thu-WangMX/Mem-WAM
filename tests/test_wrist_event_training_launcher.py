import os
from pathlib import Path
import subprocess


def test_formal_launcher_preflight_validates_fixed_putback_manifest(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    environment = {
        **os.environ,
        "REPO_ROOT": str(repo),
        "OUTPUT_DIR": str(tmp_path / "must_not_be_created"),
        "PREFLIGHT_ONLY": "1",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "MIN_FREE_GIB": "1",
    }

    completed = subprocess.run(
        ["bash", str(repo / "ops" / "train_putback_wrist_event_k8_40k.sh")],
        env=environment,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "preflight_ok" in completed.stdout
    assert "episode_count=50" in completed.stdout
    assert (
        "checkpoint_steps=5000,10000,15000,20000,25000,30000,35000,40000"
        in completed.stdout
    )
    assert "checkpoint_steps=1000," not in completed.stdout
    assert not (tmp_path / "must_not_be_created").exists()


def test_formal_launcher_rejects_output_filesystem_without_checkpoint_budget(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    environment = {
        **os.environ,
        "REPO_ROOT": str(repo),
        "OUTPUT_DIR": str(tmp_path / "must_not_be_created"),
        "PREFLIGHT_ONLY": "1",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "MIN_FREE_GIB": "999999999",
    }

    completed = subprocess.run(
        ["bash", str(repo / "ops" / "train_putback_wrist_event_k8_40k.sh")],
        env=environment,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 2
    assert "checkpoint budget" in completed.stderr
    assert not (tmp_path / "must_not_be_created").exists()


def test_formal_launcher_uses_vepfs01_mirror_output():
    repo = Path(__file__).resolve().parents[1]
    environment = {
        **os.environ,
        "REPO_ROOT": str(repo),
        "PREFLIGHT_ONLY": "1",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "MIN_FREE_GIB": "1",
    }
    environment.pop("OUTPUT_DIR", None)

    completed = subprocess.run(
        ["bash", str(repo / "ops" / "train_putback_wrist_event_k8_40k.sh")],
        env=environment,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert (
        "output_dir=/mnt/vepfs01/output/kevin_wang/memorywam/train/"
        "wrist_latent_event_memory_putback_k8_40k_seed42"
        in completed.stdout
    )
