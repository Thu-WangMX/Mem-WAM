from pathlib import Path
import os
import subprocess


REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "ops" / "train_putback_dynamic_surprise_k8_40k.sh"
PHASE_LAUNCHER = REPO / "ops" / "train_putback_dynamic_surprise_phase.sh"
RESUME_LAUNCHER = REPO / "ops" / "train_putback_dynamic_surprise_k8_resume25k_to40k.sh"


def test_40k_launcher_is_resumable_and_uses_real_checkpoint_layout():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "5000 10000 15000 20000 25000 30000 35000 40000" in text
    assert 'checkpoints/weights/step_' in text
    assert 'checkpoints/state/step_' in text
    assert '${output_dir}/weights/' not in text
    assert '${output_dir}/training_state/' not in text
    assert "highest_complete_state" in text
    assert "PREFLIGHT_ONLY" in text
    assert "manifest_is_complete" in text
    assert ".incomplete." in text
    assert "policy_step=40000 boundary_step=35000" in text


def test_40k_launcher_has_valid_shell_syntax():
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)


def test_phase_launcher_accepts_30k_with_25k_boundary(tmp_path):
    runtime_bin = tmp_path / "runtime" / "bin"
    runtime_bin.mkdir(parents=True)
    (runtime_bin / "python").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    (runtime_bin / "accelerate").write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$FAKE_ACCELERATE_ARGS\"\n",
        encoding="utf-8",
    )
    for executable in runtime_bin.iterdir():
        executable.chmod(0o755)

    repo = tmp_path / "repo"
    (repo / "scripts" / "accelerate_configs").mkdir(parents=True)
    manifest = tmp_path / "boundary25"
    manifest.mkdir()
    (manifest / "manifest.json").write_text("{}\n", encoding="utf-8")
    resume = tmp_path / "state25"
    resume.mkdir()
    (resume / "trainer_state.json").write_text("{}\n", encoding="utf-8")
    output = tmp_path / "output"
    args_file = tmp_path / "accelerate.args"

    env = os.environ | {
        "REPO_ROOT": str(repo),
        "RUNTIME_BIN": str(runtime_bin),
        "OUTPUT_DIR": str(output),
        "PHASE_END_STEP": "30000",
        "DYNAMIC_SURPRISE_MANIFEST": str(manifest),
        "DYNAMIC_SURPRISE_BOUNDARY_STEP": "25000",
        "RESUME_STATE": str(resume),
        "SAVE_STEPS": "[]",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "FAKE_ACCELERATE_ARGS": str(args_file),
    }
    result = subprocess.run(
        ["bash", str(PHASE_LAUNCHER)], env=env, text=True, capture_output=True
    )

    assert result.returncode == 0, result.stderr
    args = args_file.read_text(encoding="utf-8")
    assert "max_steps=30000\n" in args
    assert f"resume={resume.resolve()}\n" in args


def test_resume25_launcher_delegates_to_resumable_40k_launcher(tmp_path):
    repo = tmp_path / "repo"
    ops = repo / "ops"
    ops.mkdir(parents=True)
    delegated = ops / LAUNCHER.name
    delegated.write_text("#!/usr/bin/env bash\necho delegated-40k\n", encoding="utf-8")
    delegated.chmod(0o755)

    result = subprocess.run(
        ["bash", str(RESUME_LAUNCHER)],
        env=os.environ | {"REPO_ROOT": str(repo)},
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "delegated-40k"
