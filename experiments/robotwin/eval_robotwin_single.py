"""
RobotWin single-task evaluation entrypoint (Hydra).

Features:
- Read `configs/sim_robotwin.yaml`.
- Check or create the symlink:
  `RoboTwin/policy/fastwam -> experiments/robotwin/fastwam`.
- Forward config overrides to the official RoboTwin entrypoint
  `script/eval_policy.py` and save logs.

Common arguments:
- `ckpt`: path to the FastWAM checkpoint (required).
- `EVALUATION.task_name`: task name to evaluate (required).
- `gpu_id`: sets `CUDA_VISIBLE_DEVICES`.

Examples:
1) Minimal run
   python experiments/robotwin/eval_robotwin_single.py \
     ckpt=/path/to/ckpt.pt \
     EVALUATION.task_name=click_alarmclock

2) Run with more evaluation overrides
   python experiments/robotwin/eval_robotwin_single.py \
     ckpt=/path/to/ckpt.pt \
     EVALUATION.task_name=click_alarmclock \
     EVALUATION.task_config=demo_randomized \
     EVALUATION.replan_steps=4 \
     EVALUATION.num_inference_steps=4 \
     gpu_id=0
"""

import os
import errno
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY_SOURCE_NAME = "fastwam_policy"

MEMORY_POLICY_FORWARD_KEYS = (
    "physical_settle_rate_debt_online",
    "latent_kernel_regime_online",
    "dynamic_surprise_online",
    "dynamic_surprise_sigma",
    "dynamic_surprise_gamma",
    "dynamic_surprise_window",
    "dynamic_surprise_max_segment",
    "dynamic_surprise_scorer_source",
    "dynamic_surprise_init_action_dit_path",
    "dynamic_surprise_noise_mode",
    "wrist_event_online",
    "wrist_event_predictor_repo",
    "wrist_event_checkpoint",
    "wrist_event_threshold",
    "wrist_event_history",
    "wrist_event_min_segment",
    "wrist_event_max_segment",
    "embodied_information_online",
    "embodied_information_lock",
    "embodied_information_pca",
    "embodied_information_predictor",
    "embodied_information_statistics",
    "embodied_information_init_action_dit_path",
    "control_information_online",
    "control_information_lock",
    "control_information_pca",
    "control_information_feature_predictor",
    "control_information_control_probe",
    "control_information_statistics",
    "control_information_normalization",
    "control_information_init_action_dit_path",
)


def _resolve_path(path_str: str, *, base: Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_str))))
    if not path.is_absolute():
        path = (base / path).resolve()
    return path.resolve()


def _resolve_optional_path(path_value: Any, *, base: Path) -> Path | None:
    if path_value is None:
        return None
    text = str(path_value).strip()
    if text == "" or text.lower() in {"none", "null"}:
        return None
    return _resolve_path(text, base=base)


def resolve_robotwin_artifact_root(
    configured_value: Any,
    project_root: Path,
) -> Path:
    """Resolve an optional external evaluation root without changing legacy runs."""

    explicit = _resolve_optional_path(configured_value, base=project_root)
    if explicit is not None:
        return explicit
    return (project_root / "evaluate_results" / "robotwin").resolve()


def _resolve_dataset_stats_path(cfg: DictConfig, ckpt_path: Path) -> Path:
    explicit = _resolve_optional_path(cfg.EVALUATION.dataset_stats_path, base=PROJECT_ROOT)
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)

    for parent in list(ckpt_path.parents)[:4]:
        candidates.append((parent / "dataset_stats.json").resolve())

    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    raise FileNotFoundError(
        "Failed to locate dataset_stats.json. Tried explicit "
        "EVALUATION.dataset_stats_path and checkpoint parent directories. "
        "Please pass EVALUATION.dataset_stats_path=/path/to/dataset_stats.json."
    )


def _resolve_ckpt_tag(ckpt_path: Path) -> str:
    parts = ckpt_path.resolve().parts
    if "runs" in parts:
        runs_idx = parts.index("runs")
        if runs_idx + 2 >= len(parts):
            raise ValueError(
                f"`ckpt` under runs must follow .../runs/<task>/<date_dir>/..., got: {ckpt_path}"
            )
        task_name = parts[runs_idx + 1]
        date_dir = parts[runs_idx + 2]
        if task_name == "" or date_dir == "":
            raise ValueError(
                f"`ckpt` under runs must follow .../runs/<task>/<date_dir>/..., got: {ckpt_path}"
            )
        return f"{task_name}_{date_dir}"
    return ckpt_path.stem


def _ensure_policy_symlink(
    robotwin_root: Path,
    policy_source_dir: Path,
    policy_name: str = POLICY_SOURCE_NAME,
) -> Path:
    policy_root = robotwin_root / "policy"
    if not policy_root.is_dir():
        raise FileNotFoundError(f"RoboTwin policy directory not found: {policy_root}")

    policy_target = policy_root / str(policy_name)
    source_resolved = policy_source_dir.resolve()

    if not policy_target.exists() and not policy_target.is_symlink():
        try:
            policy_target.symlink_to(source_resolved, target_is_directory=True)
            return policy_target
        except FileExistsError:
            # Concurrent per-GPU evaluators may both observe the missing link.
            # Let the loser revalidate the link created by the winner.
            pass
        except OSError as exc:
            if exc.errno not in {errno.EROFS, errno.EACCES, errno.EPERM}:
                raise
            # Dataset-hosted RoboTwin/RMBench checkouts can be read-only. The
            # subprocess receives the policy package through PYTHONPATH below.
            return source_resolved

    if policy_target.is_symlink():
        target_resolved = policy_target.resolve()
        if target_resolved != source_resolved:
            raise RuntimeError(
                f"Policy symlink conflict: {policy_target} -> {target_resolved}, "
                f"expected -> {source_resolved}"
            )
        return policy_target

    raise RuntimeError(
        f"Path already exists and is not a symlink: {policy_target}. "
        "Please handle it manually to avoid overriding existing policy files."
    )


def _format_override_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, (int, float)):
        return str(value)
    return repr(str(value))


def _append_override(overrides: list[str], key: str, value: Any, *, skip_none: bool = True) -> None:
    if skip_none and value is None:
        return
    overrides.extend([f"--{key}", _format_override_value(value)])


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_robotwin.yaml")
def main(cfg: DictConfig):
    if cfg.ckpt is None:
        raise ValueError("`ckpt` must not be None.")
    if cfg.EVALUATION.task_name is None:
        raise ValueError("`EVALUATION.task_name` must not be None.")

    ckpt_path = _resolve_path(str(cfg.ckpt), base=PROJECT_ROOT)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt_tag = _resolve_ckpt_tag(ckpt_path)

    robotwin_root = _resolve_path(str(cfg.EVALUATION.robotwin_root), base=PROJECT_ROOT)
    if not robotwin_root.exists():
        raise FileNotFoundError(f"RoboTwin root not found: {robotwin_root}")

    policy_source_dir = (
        PROJECT_ROOT / "experiments" / "robotwin" / POLICY_SOURCE_NAME
    ).resolve()
    if not policy_source_dir.is_dir():
        raise FileNotFoundError(f"Policy source directory not found: {policy_source_dir}")

    policy_binding = _ensure_policy_symlink(
        robotwin_root=robotwin_root,
        policy_source_dir=policy_source_dir,
        policy_name=str(cfg.EVALUATION.policy_name),
    )
    # On read-only RMBench mounts the requested alias cannot be created.  The
    # source package is still first on PYTHONPATH, so import its real name.
    policy_import_name = str(cfg.EVALUATION.policy_name)
    if policy_binding.resolve() == policy_source_dir:
        policy_import_name = POLICY_SOURCE_NAME

    output_dir = _resolve_path(str(cfg.EVALUATION.output_dir), base=PROJECT_ROOT)
    run_ts = output_dir.name
    if run_ts == "":
        raise ValueError(f"Invalid EVALUATION.output_dir (missing run_ts): {output_dir}")
    artifact_root = resolve_robotwin_artifact_root(
        cfg.EVALUATION.get("artifact_root"),
        PROJECT_ROOT,
    )
    run_output_dir = artifact_root / ckpt_tag / run_ts
    run_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_output_dir / (
        f"eval_{str(cfg.EVALUATION.task_name)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    robotwin_eval_base = (
        artifact_root / ckpt_tag / run_ts / str(cfg.EVALUATION.task_name)
    )

    sim_cfg_name = str(cfg.get("sim_cfg_name", "sim_robotwin.yaml"))
    sim_cfg_path = (PROJECT_ROOT / "configs" / sim_cfg_name).resolve()
    if not sim_cfg_path.is_file():
        raise FileNotFoundError(
            f"Simulation config selected by `sim_cfg_name` was not found: "
            f"{sim_cfg_path}"
        )
    sim_task = HydraConfig.get().runtime.choices.get("task")

    dataset_stats_path = _resolve_dataset_stats_path(cfg, ckpt_path)

    overrides: list[str] = []
    _append_override(overrides, "task_name", cfg.EVALUATION.task_name)
    _append_override(overrides, "task_config", cfg.EVALUATION.task_config)
    _append_override(overrides, "ckpt_setting", str(ckpt_path))
    _append_override(overrides, "seed", cfg.seed)
    _append_override(overrides, "policy_name", policy_import_name)
    _append_override(overrides, "instruction_type", cfg.EVALUATION.instruction_type)
    _append_override(overrides, "eval_num_episodes", cfg.EVALUATION.eval_num_episodes)
    _append_override(
        overrides,
        "start_scene_seed",
        cfg.EVALUATION.get("start_scene_seed"),
    )

    _append_override(overrides, "sim_cfg_path", str(sim_cfg_path))
    _append_override(overrides, "sim_task", sim_task)
    _append_override(overrides, "output_root", str(robotwin_eval_base))
    _append_override(overrides, "eval_output_dir", str(robotwin_eval_base))
    _append_override(overrides, "mixed_precision", cfg.mixed_precision)
    _append_override(overrides, "device", cfg.EVALUATION.device)
    _append_override(overrides, "dataset_stats_path", str(dataset_stats_path))
    _append_override(overrides, "action_horizon", cfg.EVALUATION.action_horizon)
    _append_override(overrides, "replan_steps", cfg.EVALUATION.replan_steps)
    _append_override(overrides, "num_inference_steps", cfg.EVALUATION.num_inference_steps)
    _append_override(overrides, "sigma_shift", cfg.EVALUATION.sigma_shift)
    _append_override(overrides, "text_cfg_scale", cfg.EVALUATION.text_cfg_scale)
    _append_override(overrides, "negative_prompt", cfg.EVALUATION.negative_prompt)
    _append_override(overrides, "rand_device", cfg.EVALUATION.rand_device)
    _append_override(overrides, "tiled", cfg.EVALUATION.tiled)
    _append_override(overrides, "timing_enabled", cfg.EVALUATION.timing_enabled)
    for key in MEMORY_POLICY_FORWARD_KEYS:
        _append_override(overrides, key, cfg.EVALUATION.get(key))
    _append_override(
        overrides,
        "action_delta_fraction",
        cfg.EVALUATION.action_delta_fraction,
    )
    _append_override(
        overrides,
        "skip_get_obs_within_replan",
        cfg.EVALUATION.skip_get_obs_within_replan,
    )
    _append_override(
        overrides,
        "diagnostic_max_steps",
        cfg.EVALUATION.get("diagnostic_max_steps"),
    )

    cmd = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "third_party" / "RoboTwin" / "script" / "eval_policy.py"),
        "--config",
        str(policy_source_dir / "deploy_policy.yml"),
        "--overrides",
        *overrides,
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    # SAPIEN reads URDF files with Python's locale-dependent text encoding.
    # Non-interactive SSH jobs may otherwise fall back to ASCII and fail on
    # valid UTF-8 comments in RMBench assets.
    env["PYTHONUTF8"] = "1"
    env["LANG"] = "C.UTF-8"
    env["LC_ALL"] = "C.UTF-8"
    # Some heavyweight model loaders leave the process cwd changed.  SAPIEN's
    # URDF resolver calls getcwd even for mesh paths, so the policy restores
    # this persistent benchmark root immediately after construction.
    env["FASTWAM_ROBOTWIN_ROOT"] = str(robotwin_root)
    existing_pythonpath = env.get("PYTHONPATH", "")
    pythonpath_parts = [str(policy_source_dir.parent)]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    with open(log_file, "w", encoding="utf-8") as log_f:
        process = subprocess.Popen(
            cmd,
            cwd=str(robotwin_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_f.write(line)
            log_f.flush()
        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(f"RoboTwin evaluation failed with return code {return_code}. Log: {log_file}")

    print(f"Evaluation finished successfully. Log saved to: {log_file}")
    OmegaConf.save(
        config=cfg,
        f=str(run_output_dir / f"eval_config_{str(cfg.EVALUATION.task_name)}.yaml"),
    )


if __name__ == "__main__":
    main()
