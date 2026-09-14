import json
import logging
import os
import sys
import time
import inspect
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torchvision.transforms.functional as transforms_F

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.memory.native_cache import summarize_native_cache_state
from fastwam.memory.dynamic_surprise import OnlineSurpriseSegmenter
from fastwam.memory.event_conditioned_rate import online_memory_tokens_for_segment
from fastwam.memory.latent_kernel_regime_online import OnlineLatentKernelRegimeSegmenter
from fastwam.memory.physical_settle_rate_debt_online import (
    OnlinePhysicalSettleRateSegmenter,
)
from fastwam.memory.dynamic_surprise_scorer import (
    score_transition,
    transition_noise_seed,
)
from fastwam.memory.wrist_event import (
    OnlineWristEventSegmenter,
    causal_latent_window,
)
from fastwam.evaluation.embodied_information_online import (
    EXPECTED_INITIALIZATION_FINGERPRINT,
    EmbodiedInformationOnlineRuntime,
    initialization_fingerprint,
    load_locked_artifacts,
)
from fastwam.evaluation.control_information_online import (
    RUNTIME_LOCK_SCHEMA as CONTROL_INFORMATION_RUNTIME_LOCK_SCHEMA_V1,
    ControlInformationOnlineRuntime,
    load_locked_control_information_artifacts,
)
from fastwam.evaluation.control_information_online_v2 import (
    RUNTIME_LOCK_SCHEMA_V2 as CONTROL_INFORMATION_RUNTIME_LOCK_SCHEMA_V2,
    ControlInformationOnlineRuntimeV2,
    load_locked_control_information_artifacts_v2,
)
from fastwam.evaluation.control_information_online_v3 import (
    RUNTIME_LOCK_SCHEMA_V3 as CONTROL_INFORMATION_RUNTIME_LOCK_SCHEMA_V3,
    ControlInformationOnlineRuntimeV3,
    load_locked_control_information_artifacts_v3,
)

logger = logging.getLogger(__name__)


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse bool value: {value}")


def _parse_optional_int(value: Any) -> Optional[int]:
    if _is_none_like(value):
        return None
    return int(value)


def _parse_optional_float(value: Any) -> Optional[float]:
    if _is_none_like(value):
        return None
    return float(value)


def _normalize_surprise_scorer_source(value: Any) -> str:
    source = str(value).strip().lower()
    if source not in {"policy", "initialization"}:
        raise ValueError(
            "dynamic surprise scorer source must be 'policy' or "
            f"'initialization', got {value!r}"
        )
    return source


def _frozen_init_scorer_model_cfg(
    model_cfg: DictConfig,
    *,
    official_action_dit_path: str,
) -> DictConfig:
    if _is_none_like(official_action_dit_path):
        raise ValueError(
            "initialization surprise scorer requires an official ActionDiT path"
        )
    configured = OmegaConf.create(
        OmegaConf.to_container(model_cfg, resolve=True)
    )
    configured.load_text_encoder = False
    configured.skip_dit_load_from_pretrain = False
    configured.action_dit_pretrained_path = str(official_action_dit_path)
    return configured


def _validate_online_selector_modes(
    *,
    dynamic_surprise_online: bool,
    wrist_event_online: bool,
    embodied_information_online: bool,
    control_information_online: bool,
    latent_kernel_regime_online: bool,
    physical_settle_rate_debt_online: bool,
) -> None:
    enabled = (
        bool(dynamic_surprise_online),
        bool(wrist_event_online),
        bool(embodied_information_online),
        bool(control_information_online),
        bool(latent_kernel_regime_online),
        bool(physical_settle_rate_debt_online),
    )
    if sum(enabled) > 1:
        raise ValueError("exactly one online selector may be enabled at a time")


def _resolve_control_information_paths(
    *,
    runtime_lock: str | Path,
    pca: str | Path,
    feature_predictor: str | Path,
    control_probe: str | Path,
    statistics: str | Path,
    initialization_action_dit: str | Path,
    normalization: str | Path,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Resolve every locked selector dependency, failing before rollout."""

    project_root = Path(project_root).expanduser().resolve()
    raw = {
        "runtime_lock": runtime_lock,
        "pca": pca,
        "feature_predictor": feature_predictor,
        "control_probe": control_probe,
        "contextual_statistics": statistics,
        "initialization_action_dit": initialization_action_dit,
        "normalization": normalization,
    }
    resolved: dict[str, Path] = {}
    for name, value in raw.items():
        if _is_none_like(value):
            raise ValueError(f"control-information {name} is required")
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = project_root / path
        path = path.resolve()
        if not path.is_file():
            label = "lock" if name == "runtime_lock" else name
            raise FileNotFoundError(
                f"control-information {label} not found: {path}"
            )
        resolved[name] = path

    runtime_schema = json.loads(resolved["runtime_lock"].read_text()).get(
        "schema_version"
    )
    if runtime_schema == CONTROL_INFORMATION_RUNTIME_LOCK_SCHEMA_V1:
        resolved["runtime_version"] = "v1"
    elif runtime_schema == CONTROL_INFORMATION_RUNTIME_LOCK_SCHEMA_V2:
        resolved["runtime_version"] = "v2"
    elif runtime_schema == CONTROL_INFORMATION_RUNTIME_LOCK_SCHEMA_V3:
        resolved["runtime_version"] = "v3"
    else:
        raise ValueError(
            "control-information runtime lock schema is incompatible: "
            f"{runtime_schema!r}"
        )

    selector_root = resolved["runtime_lock"].parent.parent
    analysis_root = resolved["pca"].parent
    candidate_lock = selector_root / "selector_candidate" / "locked_candidate.json"
    initialization_manifest = (
        analysis_root
        / "putback_four_phase_multilayer_wam_features_v1"
        / "initialization_manifest.json"
    )
    feature_bank_manifest = (
        analysis_root
        / "putback_four_phase_multilayer_wam_features_v1"
        / "bank_manifest.json"
    )
    derived = {
        "candidate_lock": candidate_lock,
        "initialization_manifest": initialization_manifest,
        "feature_bank_manifest": feature_bank_manifest,
    }
    for name, path in derived.items():
        if not path.is_file():
            raise FileNotFoundError(f"control-information {name} not found: {path}")
        resolved[name] = path.resolve()

    resolved["artifact_paths"] = {
        "initialization_manifest": resolved["initialization_manifest"],
        "feature_bank_manifest": resolved["feature_bank_manifest"],
        "pca": resolved["pca"],
        "feature_predictor": resolved["feature_predictor"],
        "control_probe": resolved["control_probe"],
        "contextual_statistics": resolved["contextual_statistics"],
        "normalization": resolved["normalization"],
    }
    if resolved["runtime_version"] in {"v2", "v3"}:
        resolved["artifact_paths"].update(
            {
                "base_score_source": SRC_ROOT
                / "fastwam/memory/control_information_probe.py",
                "base_boundary_source": SRC_ROOT
                / "fastwam/memory/control_information_boundary.py",
                "episode_adapter_source": SRC_ROOT
                / "fastwam/memory/control_information_boundary_v2.py",
            }
        )
        if resolved["runtime_version"] == "v3":
            resolved["artifact_paths"]["segment_boundary_source"] = (
                SRC_ROOT / "fastwam/memory/control_information_boundary_v3.py"
            )
            online_source = (
                SRC_ROOT / "fastwam/evaluation/control_information_online_v3.py"
            )
            offline_freezer_source = (
                PROJECT_ROOT
                / "scripts/freeze_putback_control_information_manifest_v3.py"
            )
        else:
            online_source = (
                SRC_ROOT / "fastwam/evaluation/control_information_online_v2.py"
            )
            offline_freezer_source = (
                PROJECT_ROOT
                / "scripts/freeze_putback_control_information_manifest_v2.py"
            )
        resolved["source_paths"] = {
            "base_online_source": SRC_ROOT
            / "fastwam/evaluation/control_information_online.py",
            "online_source": online_source,
            "planning_aligner_source": SRC_ROOT
            / "fastwam/memory/planning_aligned_manifest.py",
            "offline_freezer_source": offline_freezer_source,
        }
    else:
        resolved["source_paths"] = {
            "online_source": SRC_ROOT
            / "fastwam/evaluation/control_information_online.py",
            "planning_aligner_source": SRC_ROOT
            / "fastwam/memory/planning_aligned_manifest.py",
            "offline_freezer_source": PROJECT_ROOT
            / "scripts/freeze_putback_control_information_manifest.py",
        }
    return resolved


def _fresh_control_information_runtime(
    previous: ControlInformationOnlineRuntime,
) -> ControlInformationOnlineRuntime:
    """Reset all causal feature/CUSUM/alignment state while reusing frozen weights."""

    runtime_class = type(previous)
    if runtime_class not in {
        ControlInformationOnlineRuntime,
        ControlInformationOnlineRuntimeV2,
        ControlInformationOnlineRuntimeV3,
    }:
        raise TypeError(
            f"unsupported control-information runtime: {runtime_class.__name__}"
        )
    return runtime_class(
        feature_predictor=previous.feature_predictor,
        control_probe=previous.control_probe,
        normalization=previous.normalization,
        contextual_statistics=previous.contextual_statistics,
        selector_config=previous.selector_config,
        feature_window=previous.feature_window,
        policy_model=previous.policy_model,
        initialization_model=previous.initialization_model,
        pca_artifact=previous.pca_artifact,
        capture_fn=previous.capture_fn,
    )


def _control_information_diagnostic_metrics(
    *,
    score: Any,
    observed_feature: torch.Tensor,
    raw_proprio: torch.Tensor,
    executed_actions: list[torch.Tensor],
) -> dict[str, float]:
    """Expose scale diagnostics without changing selector state or decisions."""

    def rms(value: torch.Tensor) -> float:
        tensor = torch.as_tensor(value, dtype=torch.float32)
        if tensor.numel() == 0 or not bool(torch.isfinite(tensor).all()):
            raise ValueError("control-information diagnostic tensor must be finite")
        return float(torch.sqrt(torch.mean(tensor.square())).item())

    observed = torch.as_tensor(observed_feature, dtype=torch.float32).reshape(-1)
    predicted = torch.as_tensor(score.predicted_feature, dtype=torch.float32).reshape(-1)
    if observed.shape != predicted.shape:
        raise ValueError("observed and predicted diagnostic features must align")
    actions = torch.stack(
        [torch.as_tensor(value, dtype=torch.float32).reshape(-1) for value in executed_actions]
    )
    prior_rms = rms(score.prior_action)
    posterior_rms = rms(score.posterior_action)
    return {
        "observed_feature_rms": rms(observed),
        "observed_feature_abs_max": float(observed.abs().max().item()),
        "predicted_feature_rms": rms(predicted),
        "feature_prediction_residual_rms": rms(observed - predicted),
        "prior_action_rms": prior_rms,
        "posterior_action_rms": posterior_rms,
        "relative_control_information": float(score.information)
        / max(prior_rms, posterior_rms, 1e-8),
        "raw_proprio_rms": rms(raw_proprio),
        "executed_action_rms": rms(actions),
        "executed_action_delta_rms": rms(actions[-1] - actions[0]),
    }


def _control_information_event_payload(event: Any) -> dict[str, Any]:
    """Serialize v1/v2/v3 boundary events without changing their semantics."""

    standardized = getattr(event, "standardized_information", None)
    contextual_standardized = getattr(
        event, "contextual_standardized_information", None
    )
    if standardized is None:
        standardized = contextual_standardized
    if standardized is None:
        raise AttributeError(
            "control-information event has no standardized score field"
        )
    return {
        "confirmation_frame": event.confirmation_frame,
        "group_start": event.group_start,
        "group_end": event.group_end,
        "reason": event.reason,
        "information": event.information,
        "adapted_information": getattr(event, "adapted_information", None),
        "adaptation_factor": getattr(
            event,
            "adaptation_factor",
            getattr(event, "episode_adaptation_factor", None),
        ),
        "segment_relative_innovation": getattr(
            event, "segment_relative_innovation", None
        ),
        "segment_baseline_location": getattr(
            event, "segment_baseline_location", None
        ),
        "segment_baseline_scale": getattr(
            event, "segment_baseline_scale", None
        ),
        "standardized_information": standardized,
        "contextual_standardized_information": contextual_standardized,
        "cusum": event.cusum,
    }


def _surprise_scorer_memory_groups(
    source: str,
    policy_memory_groups: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...] | None:
    return None if _normalize_surprise_scorer_source(source) == "initialization" else policy_memory_groups


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _resolve_sim_cfg_name(sim_cfg_path: Optional[str], sim_cfg_name: Optional[str]) -> str:
    configs_root = (PROJECT_ROOT / "configs").resolve()
    if not _is_none_like(sim_cfg_path):
        cfg_path = Path(str(sim_cfg_path)).expanduser().resolve()
        try:
            relative = cfg_path.relative_to(configs_root)
        except ValueError as exc:
            raise ValueError(
                f"`sim_cfg_path` must be under {configs_root}, got: {cfg_path}"
            ) from exc
        return relative.as_posix()

    if _is_none_like(sim_cfg_name):
        return "sim_robotwin.yaml"
    return str(sim_cfg_name)


def _compose_sim_cfg(
    sim_cfg_path: Optional[str],
    sim_cfg_name: Optional[str],
    sim_task: Optional[str],
) -> DictConfig:
    config_name = _resolve_sim_cfg_name(sim_cfg_path=sim_cfg_path, sim_cfg_name=sim_cfg_name)
    configs_root = (PROJECT_ROOT / "configs").resolve()
    overrides = []
    if not _is_none_like(sim_task):
        overrides.append(f"task={str(sim_task)}")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base="1.3", config_dir=str(configs_root)):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def _resolve_dataset_stats_path(dataset_stats_path: Optional[str]) -> Path:
    if _is_none_like(dataset_stats_path):
        raise FileNotFoundError(
            "`dataset_stats_path` is required. "
            "Please pass it from eval entrypoint overrides."
        )
    resolved = Path(str(dataset_stats_path)).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset stats path not found: {resolved}")
    return resolved


class WorldActionRobotWinPolicy:
    def __init__(
        self,
        model_cfg: DictConfig,
        processor_cfg: DictConfig,
        checkpoint_path: str,
        dataset_stats_path: Path,
        device: str,
        model_dtype: torch.dtype,
        action_horizon: int,
        replan_steps: int,
        num_inference_steps: int,
        sigma_shift: Optional[float],
        seed: Optional[int],
        text_cfg_scale: float,
        negative_prompt: str,
        rand_device: str,
        tiled: bool,
        timing_enabled: bool,
        num_video_frames: int,
        action_delta_fraction: Optional[float],
        dynamic_surprise_online: bool = False,
        dynamic_surprise_sigma: float = 1.0,
        dynamic_surprise_gamma: float = 1.5,
        dynamic_surprise_window: int = 5,
        dynamic_surprise_max_segment: int = 8,
        dynamic_surprise_scorer_source: str = "policy",
        dynamic_surprise_init_action_dit_path: Optional[str] = None,
        dynamic_surprise_noise_mode: str = "per_transition",
        wrist_event_online: bool = False,
        wrist_event_predictor_repo: Optional[str] = None,
        wrist_event_checkpoint: Optional[str] = None,
        wrist_event_threshold: float = 0.5,
        wrist_event_history: int = 3,
        wrist_event_min_segment: int = 2,
        wrist_event_max_segment: int = 8,
        embodied_information_online: bool = False,
        embodied_information_lock: Optional[str] = None,
        embodied_information_pca: Optional[str] = None,
        embodied_information_predictor: Optional[str] = None,
        embodied_information_statistics: Optional[str] = None,
        embodied_information_init_action_dit_path: Optional[str] = None,
        control_information_online: bool = False,
        latent_kernel_regime_online: bool = False,
        physical_settle_rate_debt_online: bool = False,
        control_information_lock: Optional[str] = None,
        control_information_pca: Optional[str] = None,
        control_information_feature_predictor: Optional[str] = None,
        control_information_control_probe: Optional[str] = None,
        control_information_statistics: Optional[str] = None,
        control_information_normalization: Optional[str] = None,
        control_information_init_action_dit_path: Optional[str] = None,
    ) -> None:
        model_cfg_copy = OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=True))
        model_cfg_copy.load_text_encoder = True

        self.model = instantiate(model_cfg_copy, model_dtype=model_dtype, device=device)
        self.model.load_checkpoint(checkpoint_path)
        self.model = self.model.to(device).eval()
        self.dynamic_surprise_scorer_source = _normalize_surprise_scorer_source(
            dynamic_surprise_scorer_source
        )
        if self.dynamic_surprise_scorer_source == "initialization":
            scorer_cfg = _frozen_init_scorer_model_cfg(
                model_cfg,
                official_action_dit_path=str(
                    dynamic_surprise_init_action_dit_path or ""
                ),
            )
            self._surprise_scoring_model = instantiate(
                scorer_cfg,
                model_dtype=model_dtype,
                device=device,
            ).to(device).eval()
            for parameter in self._surprise_scoring_model.parameters():
                parameter.requires_grad_(False)
        else:
            self._surprise_scoring_model = self.model
        print(
            "FASTWAM_DYNAMIC_SURPRISE_SCORER "
            + json.dumps(
                {"source": self.dynamic_surprise_scorer_source},
                sort_keys=True,
            ),
            flush=True,
        )

        self.processor: FastWAMProcessor = instantiate(processor_cfg).eval()
        dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
        self.processor.set_normalizer_from_stats(dataset_stats)
        action_meta = self.processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("Expected one action key for trajectory projection")
        action_key = action_meta[0]["key"]
        action_stats = dataset_stats["action"][action_key]
        self._action_min = (
            action_stats["global_min"].reshape(-1).float().cpu().numpy()
        )
        self._action_max = (
            action_stats["global_max"].reshape(-1).float().cpu().numpy()
        )
        self.action_delta_fraction = (
            None
            if action_delta_fraction is None
            else float(action_delta_fraction)
        )
        if (
            self.action_delta_fraction is not None
            and not 0.0 < self.action_delta_fraction <= 1.0
        ):
            raise ValueError(
                "`action_delta_fraction` must be in (0, 1], got "
                f"{self.action_delta_fraction}"
            )
        self._action_delta_limit = (
            None
            if self.action_delta_fraction is None
            else (
                self._action_max - self._action_min
            ) * self.action_delta_fraction
        )

        self.action_horizon = int(action_horizon)
        self.replan_steps = int(max(1, min(replan_steps, action_horizon)))
        self._temporal_subframes = 4
        if self.replan_steps % self._temporal_subframes != 0:
            raise ValueError(
                "Temporal full-KV requires replan_steps divisible by four, "
                f"got {self.replan_steps}"
            )
        self._temporal_sample_stride = (
            self.replan_steps // self._temporal_subframes
        )
        self.num_inference_steps = int(num_inference_steps)
        self.sigma_shift = sigma_shift
        self.seed = seed
        self.text_cfg_scale = float(text_cfg_scale)
        self.negative_prompt = str(negative_prompt)
        self.rand_device = str(rand_device)
        self.tiled = bool(tiled)
        self.timing_enabled = bool(timing_enabled)
        self._num_video_frames = int(num_video_frames)
        self.dynamic_surprise_online = bool(dynamic_surprise_online)
        self.wrist_event_online = bool(wrist_event_online)
        self.embodied_information_online = bool(embodied_information_online)
        self.control_information_online = bool(control_information_online)
        self.latent_kernel_regime_online = bool(latent_kernel_regime_online)
        self.physical_settle_rate_debt_online = bool(
            physical_settle_rate_debt_online
        )
        _validate_online_selector_modes(
            dynamic_surprise_online=self.dynamic_surprise_online,
            wrist_event_online=self.wrist_event_online,
            embodied_information_online=self.embodied_information_online,
            control_information_online=self.control_information_online,
            latent_kernel_regime_online=self.latent_kernel_regime_online,
            physical_settle_rate_debt_online=(
                self.physical_settle_rate_debt_online
            ),
        )
        self._physical_settle_rate_runtime = OnlinePhysicalSettleRateSegmenter()
        if self.physical_settle_rate_debt_online:
            print(
                "FASTWAM_PHYSICAL_SETTLE_RATE_SELECTOR "
                + json.dumps(
                    {
                        "source": "raw_qpos",
                        "strict_online": True,
                        "anchor_frames": 2,
                        "min_segment": 4,
                        "max_segment": 8,
                        "history_window": 8,
                        "event_threshold": 0.8,
                        "target_mean_segment": 7.0,
                        "maximum_rate_debt": 5.0,
                        "memory_tokens": 8,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        self._latent_kernel_runtime = OnlineLatentKernelRegimeSegmenter()
        if self.latent_kernel_regime_online:
            print(
                "FASTWAM_LATENT_KERNEL_SELECTOR "
                + json.dumps(
                    {
                        "source": "frozen_causal_vae_only",
                        "online": True,
                        "anchor_frames": 2,
                        "recent_frames": 4,
                        "min_segment": 4,
                        "max_segment": 8,
                        "memory_tokens": 8,
                        "rank_threshold": 0.75,
                        "history_window": 8,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        self.dynamic_surprise_sigma = float(dynamic_surprise_sigma)
        self.dynamic_surprise_noise_mode = str(dynamic_surprise_noise_mode)
        transition_noise_seed(0, 0, mode=self.dynamic_surprise_noise_mode)
        self._surprise_segmenter = OnlineSurpriseSegmenter(
            min_segment=2,
            max_segment=int(dynamic_surprise_max_segment),
            gamma=float(dynamic_surprise_gamma),
            window=int(dynamic_surprise_window),
        )
        self._surprise_latents: list[torch.Tensor] = []
        self._surprise_previous_action: Optional[torch.Tensor] = None
        self._surprise_previous_conditioning: Optional[dict[str, torch.Tensor]] = None
        self._wrist_event_predictor = None
        self._wrist_innovation_fn = None
        self._wrist_event_latents: list[torch.Tensor] = []
        self._wrist_event_history = int(wrist_event_history)
        if self.wrist_event_online:
            if _is_none_like(wrist_event_predictor_repo) or _is_none_like(
                wrist_event_checkpoint
            ):
                raise ValueError(
                    "wrist_event_online requires predictor repo and checkpoint"
                )
            predictor_src = (
                Path(str(wrist_event_predictor_repo)).expanduser().resolve() / "src"
            )
            predictor_checkpoint = Path(
                str(wrist_event_checkpoint)
            ).expanduser().resolve()
            if not predictor_src.is_dir() or not predictor_checkpoint.is_file():
                raise FileNotFoundError(
                    "missing wrist-event predictor dependency: "
                    f"src={predictor_src}, checkpoint={predictor_checkpoint}"
                )
            if str(predictor_src) not in sys.path:
                sys.path.insert(0, str(predictor_src))
            from wrist_event_predictor.training import load_predictor_checkpoint
            from wrist_event_predictor.latent_innovation import wrist_relative_rmse

            predictor, predictor_payload = load_predictor_checkpoint(
                predictor_checkpoint,
                device=self.model.device,
            )
            predictor_config = predictor_payload["model_config"]
            if bool(predictor_config.get("include_proprio", False)):
                raise ValueError("primary wrist-event policy requires a wrist-only predictor")
            if int(predictor_config["history"]) != self._wrist_event_history:
                raise ValueError(
                    "predictor history does not match deployment history: "
                    f"{predictor_config['history']} != {self._wrist_event_history}"
                )
            self._wrist_event_predictor = predictor.eval()
            self._wrist_innovation_fn = wrist_relative_rmse
            for parameter in self._wrist_event_predictor.parameters():
                parameter.requires_grad_(False)
            self._wrist_event_segmenter = OnlineWristEventSegmenter(
                threshold=float(wrist_event_threshold),
                min_segment=int(wrist_event_min_segment),
                max_segment=int(wrist_event_max_segment),
            )
            print(
                "FASTWAM_WRIST_EVENT_PREDICTOR "
                + json.dumps(
                    {
                        "checkpoint": str(predictor_checkpoint),
                        "threshold": float(wrist_event_threshold),
                        "history": self._wrist_event_history,
                        "min_segment": int(wrist_event_min_segment),
                        "max_segment": int(wrist_event_max_segment),
                        "uses_vlm": False,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        else:
            self._wrist_event_segmenter = None

        self._embodied_runtime = None
        self._embodied_executed_actions: list[torch.Tensor] = []
        if self.embodied_information_online:
            required = {
                "lock": embodied_information_lock,
                "pca": embodied_information_pca,
                "predictor": embodied_information_predictor,
                "statistics": embodied_information_statistics,
                "initialization_action_dit": embodied_information_init_action_dit_path,
            }
            missing = [name for name, value in required.items() if _is_none_like(value)]
            if missing:
                raise ValueError(
                    "embodied_information_online lacks artifacts: " + ", ".join(missing)
                )
            init_cfg = _frozen_init_scorer_model_cfg(
                model_cfg,
                official_action_dit_path=str(embodied_information_init_action_dit_path),
            )
            fork_devices = []
            if self.model.device.type == "cuda":
                fork_devices = [self.model.device.index or torch.cuda.current_device()]
            # The locked offline selector was built from the completely fresh
            # model under seed 42.  Isolate this RNG use so per-scene policy
            # seeds cannot silently change the selector or the action rollout.
            with torch.random.fork_rng(devices=fork_devices):
                torch.manual_seed(42)
                if self.model.device.type == "cuda":
                    torch.cuda.manual_seed_all(42)
                initialization_model = instantiate(
                    init_cfg, model_dtype=model_dtype, device=device
                ).to(device).eval()
            for parameter in initialization_model.parameters():
                parameter.requires_grad_(False)
            fingerprint = initialization_fingerprint(initialization_model)
            if fingerprint != EXPECTED_INITIALIZATION_FINGERPRINT:
                raise RuntimeError(
                    "initialization-WAM fingerprint mismatch: "
                    f"{fingerprint} != {EXPECTED_INITIALIZATION_FINGERPRINT}"
                )
            # Feature capture uses only the initialized Video Expert.  Drop the
            # Action Expert, VAE and cache compressor before rollout; keeping
            # them would waste several GiB and risks an avoidable inference OOM.
            del initialization_model.mot.mixtures["action"]
            initialization_model.vae = None
            initialization_model.proprio_encoder = None
            initialization_model.layerwise_block_memory = None
            initialization_model.native_cache_compressor = None
            if self.model.device.type == "cuda":
                torch.cuda.empty_cache()
            lock, pca, predictor, statistics = load_locked_artifacts(
                lock_path=str(embodied_information_lock),
                pca_path=str(embodied_information_pca),
                predictor_path=str(embodied_information_predictor),
                statistics_path=str(embodied_information_statistics),
                device=self.model.device,
            )
            self._embodied_runtime = EmbodiedInformationOnlineRuntime(
                policy_model=self.model,
                initialization_model=initialization_model,
                pca_artifact=pca,
                predictor=predictor,
                contextual_statistics=statistics,
                selector_config=lock["selector_config"],
                feature_window=8,
            )
            print(
                "FASTWAM_EMBODIED_INFORMATION_SELECTOR "
                + json.dumps(
                    {
                        "source": "initialization_wam",
                        "strict_online": True,
                        "detector_stride": 4,
                        "planning_stride": 16,
                        "forward_alignment": True,
                        "uses_old_dynamic_surprise": False,
                        "initialization_fingerprint": fingerprint,
                        "selector_config": lock["selector_config"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        self._control_information_runtime = None
        self._control_information_executed_actions: list[torch.Tensor] = []
        self._control_information_latencies_ms: list[float] = []
        if self.control_information_online:
            control_information_normalization_path = (
                dataset_stats_path
                if _is_none_like(control_information_normalization)
                else str(control_information_normalization)
            )
            control_paths = _resolve_control_information_paths(
                runtime_lock=str(control_information_lock or ""),
                pca=str(control_information_pca or ""),
                feature_predictor=str(
                    control_information_feature_predictor or ""
                ),
                control_probe=str(control_information_control_probe or ""),
                statistics=str(control_information_statistics or ""),
                initialization_action_dit=str(
                    control_information_init_action_dit_path or ""
                ),
                normalization=control_information_normalization_path,
            )
            init_cfg = _frozen_init_scorer_model_cfg(
                model_cfg,
                official_action_dit_path=str(
                    control_paths["initialization_action_dit"]
                ),
            )
            fork_devices = []
            if self.model.device.type == "cuda":
                fork_devices = [self.model.device.index or torch.cuda.current_device()]
            with torch.random.fork_rng(devices=fork_devices):
                torch.manual_seed(42)
                if self.model.device.type == "cuda":
                    torch.cuda.manual_seed_all(42)
                initialization_model = instantiate(
                    init_cfg, model_dtype=model_dtype, device=device
                ).to(device).eval()
            for parameter in initialization_model.parameters():
                parameter.requires_grad_(False)
            fingerprint = initialization_fingerprint(initialization_model)
            if fingerprint != EXPECTED_INITIALIZATION_FINGERPRINT:
                raise RuntimeError(
                    "initialization-WAM fingerprint mismatch: "
                    f"{fingerprint} != {EXPECTED_INITIALIZATION_FINGERPRINT}"
                )
            del initialization_model.mot.mixtures["action"]
            initialization_model.vae = None
            initialization_model.proprio_encoder = None
            initialization_model.layerwise_block_memory = None
            initialization_model.native_cache_compressor = None
            if self.model.device.type == "cuda":
                torch.cuda.empty_cache()
            if control_paths["runtime_version"] == "v3":
                loaded = load_locked_control_information_artifacts_v3(
                    runtime_lock_path=control_paths["runtime_lock"],
                    candidate_lock_path=control_paths["candidate_lock"],
                    artifact_paths=control_paths["artifact_paths"],
                    source_paths=control_paths["source_paths"],
                    device=self.model.device,
                )
                runtime_class = ControlInformationOnlineRuntimeV3
            elif control_paths["runtime_version"] == "v2":
                loaded = load_locked_control_information_artifacts_v2(
                    runtime_lock_path=control_paths["runtime_lock"],
                    candidate_lock_path=control_paths["candidate_lock"],
                    artifact_paths=control_paths["artifact_paths"],
                    source_paths=control_paths["source_paths"],
                    device=self.model.device,
                )
                runtime_class = ControlInformationOnlineRuntimeV2
            else:
                loaded = load_locked_control_information_artifacts(
                    runtime_lock_path=control_paths["runtime_lock"],
                    candidate_lock_path=control_paths["candidate_lock"],
                    artifact_paths=control_paths["artifact_paths"],
                    source_paths=control_paths["source_paths"],
                    device=self.model.device,
                )
                runtime_class = ControlInformationOnlineRuntime
            self._control_information_runtime = runtime_class(
                feature_predictor=loaded["feature_predictor"],
                control_probe=loaded["control_probe"],
                normalization=loaded["normalization"],
                contextual_statistics=loaded["contextual_statistics"],
                selector_config=loaded["selector_config"],
                feature_window=8,
                policy_model=self.model,
                initialization_model=initialization_model,
                pca_artifact=loaded["pca"],
            )
            print(
                "FASTWAM_CONTROL_INFORMATION_SELECTOR "
                + json.dumps(
                    {
                        "source": "frozen_initialization_wam",
                        "method": "counterfactual_control_information",
                        "runtime_version": control_paths["runtime_version"],
                        "strict_online": True,
                        "detector_stride": 4,
                        "planning_stride": 16,
                        "forward_alignment": True,
                        "uses_old_dynamic_surprise": False,
                        "uses_gripper_hard_trigger": False,
                        "initialization_fingerprint": fingerprint,
                        "selector_config": loaded["selector_config"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        self.pending_actions: deque[np.ndarray] = deque()
        self._full_kv_cache = None
        self._native_cache_state = None
        self._full_kv_frame_index = 0
        self._temporal_frames: list[torch.Tensor] = []
        self.episode_count = 0
        self.step_count = 0
        self._timing_rollout = {"infer_s": 0.0, "sim_s": 0.0}
        if self.model.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.model.device)
        self._log_action_trajectory = _parse_bool(
            os.environ.get("FASTWAM_LOG_ACTION_TRAJECTORY", "false")
        )
        self._advance_policy_seed = _parse_bool(
            os.environ.get("FASTWAM_ADVANCE_POLICY_SEED", "false")
        )
        self._freeze_inactive_arm = _parse_bool(
            os.environ.get("FASTWAM_FREEZE_INACTIVE_ARM", "false")
        )
        self._active_arm_energy_ratio = float(
            os.environ.get("FASTWAM_ACTIVE_ARM_ENERGY_RATIO", "1.25")
        )
        if self._active_arm_energy_ratio < 1.0:
            raise ValueError("FASTWAM_ACTIVE_ARM_ENERGY_RATIO must be >= 1.0")
        self._action_start_blend_steps = int(
            os.environ.get("FASTWAM_ACTION_START_BLEND_STEPS", "0")
        )
        if not 0 <= self._action_start_blend_steps <= self.action_horizon:
            raise ValueError(
                "FASTWAM_ACTION_START_BLEND_STEPS must be between 0 and "
                f"action_horizon={self.action_horizon}, got "
                f"{self._action_start_blend_steps}"
            )
        self._replan_count = 0

        logger.info(
            "Initialized WorldActionRobotWinPolicy | ckpt=%s | stats=%s | horizon=%d | replan=%d",
            checkpoint_path,
            dataset_stats_path,
            self.action_horizon,
            self.replan_steps,
        )
        print(
            "FASTWAM_EVAL_CONTRACT "
            + json.dumps(
                {
                    "checkpoint": str(Path(checkpoint_path).resolve()),
                    "dataset_stats": str(dataset_stats_path.resolve()),
                    "action_horizon": self.action_horizon,
                    "replan_steps": self.replan_steps,
                    "policy_seed": self.seed,
                    "advance_policy_seed": self._advance_policy_seed,
                    "freeze_inactive_arm": self._freeze_inactive_arm,
                    "active_arm_energy_ratio": self._active_arm_energy_ratio,
                    "action_start_blend_steps": self._action_start_blend_steps,
                    "num_inference_steps": self.num_inference_steps,
                    "sigma_shift": self.sigma_shift,
                    "text_cfg_scale": self.text_cfg_scale,
                    "rand_device": self.rand_device,
                    "action_delta_fraction": self.action_delta_fraction,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("Expected exactly one merged state key in shape_meta['state'].")
        state_key = state_meta[0]["key"]

        state_batch = {"state": {state_key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)}}
        state_batch = self.processor.action_state_transform(state_batch)
        state_batch = self.processor.normalizer.forward(state_batch)
        return state_batch["state"][state_key]

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"Expected action tensor [B,T,D], got {tuple(action.shape)}")

        action_meta = self.processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("Expected exactly one merged action key in shape_meta['action'].")

        action_key = action_meta[0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][action_key]
        denorm = normalizer.backward(action.to(dtype=torch.float32, device="cpu"))
        return denorm.numpy()

    def _build_robotwin_image_tensor(self, observation: Dict[str, Any]) -> torch.Tensor:
        obs_data = observation["observation"]
        camera_tensors = []
        for camera in ("head_camera", "left_camera", "right_camera"):
            image = np.asarray(obs_data[camera]["rgb"], dtype=np.uint8)
            tensor = (
                torch.from_numpy(np.ascontiguousarray(image))
                .permute(2, 0, 1)
                .float()
                .div_(255.0)
            )
            camera_tensors.append(
                transforms_F.resize(
                    tensor,
                    size=[240, 320],
                    interpolation=transforms_F.InterpolationMode.BILINEAR,
                    antialias=True,
                )
            )
        head = transforms_F.resize(
            camera_tensors[0],
            size=[256, 320],
            interpolation=transforms_F.InterpolationMode.BILINEAR,
            antialias=True,
        )
        left = transforms_F.resize(
            camera_tensors[1],
            size=[128, 160],
            interpolation=transforms_F.InterpolationMode.BILINEAR,
            antialias=True,
        )
        right = transforms_F.resize(
            camera_tensors[2],
            size=[128, 160],
            interpolation=transforms_F.InterpolationMode.BILINEAR,
            antialias=True,
        )
        wrists = torch.cat([left, right], dim=2)
        # Match cache precomputation exactly: normalize while still float32,
        # then cast to the VAE dtype. Casting before normalization changes
        # many 8-bit RGB values after bf16 rounding and measurably shifts the
        # encoded latent.
        image_tensor = torch.cat([wrists, head], dim=1).unsqueeze(0)
        image_tensor = image_tensor * 2.0 - 1.0
        return image_tensor.to(
            device=self.model.device,
            dtype=self.model.torch_dtype,
        )

    def _infer_action_chunk(self, observation: Dict[str, Any], instruction: str) -> np.ndarray:
        image_tensor = self._build_robotwin_image_tensor(observation)
        temporal_frames = [*self._temporal_frames, image_tensor]
        if (len(temporal_frames) - 1) % self._temporal_subframes != 0:
            raise RuntimeError(
                "Continuous temporal memory must contain 1 + 4k frames, "
                f"got {len(temporal_frames)}"
            )
        temporal_video = torch.stack(temporal_frames, dim=2)
        state_vector = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        proprio = self._normalize_state(state_vector)

        boundary_decision = None
        wrist_arrival_decision = None
        embodied_close_range = None
        control_information_close_range = None
        control_information_close_token_count = None
        control_information_reason = None
        latent_kernel_close_range = None
        physical_settle_segment = None
        allocation_mode = None
        if getattr(self, "physical_settle_rate_debt_online", False):
            physical_settle_segment = (
                self._physical_settle_rate_runtime.arrive_planning(
                    decision=self._full_kv_frame_index,
                    state=state_vector,
                )
            )
            print(
                "FASTWAM_PHYSICAL_SETTLE_RATE_ARRIVAL "
                + json.dumps(
                    {
                        "decision": self._full_kv_frame_index,
                        "close_range": (
                            None
                            if physical_settle_segment is None
                            else [
                                physical_settle_segment.start,
                                physical_settle_segment.end,
                            ]
                        ),
                        "confirmed_at": (
                            None
                            if physical_settle_segment is None
                            else physical_settle_segment.confirmed_at
                        ),
                        "reason": (
                            None
                            if physical_settle_segment is None
                            else physical_settle_segment.reason
                        ),
                        "rate_debt": (
                            None
                            if physical_settle_segment is None
                            else physical_settle_segment.rate_debt
                        ),
                        "close_token_count": (
                            None if physical_settle_segment is None else 8
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if getattr(self, "latent_kernel_regime_online", False):
            with torch.no_grad():
                encoded = self.model._encode_input_image_latents_tensor(
                    input_image=temporal_video, tiled=self.tiled
                )
            current_latent = encoded[:, :, -1:].detach()
            self._latent_kernel_runtime.observe(
                decision=self._full_kv_frame_index,
                latent=current_latent,
            )
            latent_kernel_close_range = self._latent_kernel_runtime.arrive_planning(
                decision=self._full_kv_frame_index
            )
            print(
                "FASTWAM_LATENT_KERNEL_ARRIVAL "
                + json.dumps(
                    {
                        "decision": self._full_kv_frame_index,
                        "close_range": latent_kernel_close_range,
                        "close_token_count": (
                            None if latent_kernel_close_range is None else 8
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if getattr(self, "embodied_information_online", False):
            embodied_close_range = self._embodied_runtime.arrive_planning(
                frame=self.step_count
            )
            print(
                "FASTWAM_EMBODIED_PLANNING_ARRIVAL "
                + json.dumps(
                    {
                        "frame": self.step_count,
                        "decision": self.step_count // 16,
                        "close_range": embodied_close_range,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if getattr(self, "control_information_online", False):
            control_information_aligner = self._control_information_runtime.aligner
            control_information_decision_index = self.step_count // 16
            control_information_reason = getattr(
                control_information_aligner, "_pending", {}
            ).get(control_information_decision_index)
            control_information_close_range = (
                self._control_information_runtime.arrive_planning(
                    frame=self.step_count
                )
            )
            if control_information_close_range is not None:
                allocation_mode = str(
                    getattr(
                        getattr(self.model, "layerwise_block_memory", None),
                        "allocation_mode",
                        "span_full",
                    )
                )
                control_information_close_token_count = online_memory_tokens_for_segment(
                    control_information_close_range[1]
                    - control_information_close_range[0],
                    allocation_mode=allocation_mode,
                    reason=control_information_reason,
                )
            print(
                "FASTWAM_CONTROL_INFORMATION_PLANNING_ARRIVAL "
                + json.dumps(
                    {
                        "frame": self.step_count,
                        "decision": self.step_count // 16,
                        "close_range": control_information_close_range,
                        "close_reason": control_information_reason,
                        "close_token_count": control_information_close_token_count,
                        "allocation_mode": allocation_mode,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if getattr(self, "wrist_event_online", False):
            wrist_arrival_decision = self._wrist_event_segmenter.preview_arrival()
            if wrist_arrival_decision.arrival != self._full_kv_frame_index:
                raise RuntimeError(
                    "wrist-event arrival/frame mismatch: "
                    f"{wrist_arrival_decision.arrival} != {self._full_kv_frame_index}"
                )
            print(
                "FASTWAM_WRIST_EVENT_ARRIVAL "
                + json.dumps(
                    {
                        "arrival": wrist_arrival_decision.arrival,
                        "scheduled_probability": wrist_arrival_decision.probability,
                        "close_range": wrist_arrival_decision.close_range,
                        "reason": wrist_arrival_decision.reason,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if getattr(self, "dynamic_surprise_online", False):
            with torch.no_grad():
                encoded = self.model._encode_input_image_latents_tensor(
                    input_image=temporal_video, tiled=self.tiled
                )
            actual_latent = encoded[:, :, -1:].detach()
            if self._surprise_latents:
                if self._surprise_previous_action is None or self._surprise_previous_conditioning is None:
                    raise RuntimeError("dynamic surprise state is missing source conditioning")
                previous_groups = tuple(
                    tuple(range(unit.start, unit.endpoint + 1))
                    for unit in self._native_cache_state.units
                    if unit.kind == "memory"
                ) if self._native_cache_state is not None else ()
                score = score_transition(
                    self._surprise_scoring_model,
                    history_latents=torch.cat(self._surprise_latents, dim=2),
                    actual_latent=actual_latent,
                    action=self._surprise_previous_action,
                    context=self._surprise_previous_conditioning["context"],
                    context_mask=self._surprise_previous_conditioning["context_mask"],
                    video_context=self._surprise_previous_conditioning["video_context"],
                    video_context_mask=self._surprise_previous_conditioning["video_context_mask"],
                    memory_groups=_surprise_scorer_memory_groups(
                        self.dynamic_surprise_scorer_source,
                        previous_groups,
                    ),
                    sigma=self.dynamic_surprise_sigma,
                    noise_seed=transition_noise_seed(
                        max(0, self.episode_count - 1),
                        self._full_kv_frame_index - 1,
                        mode=self.dynamic_surprise_noise_mode,
                    ),
                )
                boundary_decision = self._surprise_segmenter.preview(score["score"])
                print(
                    "FASTWAM_DYNAMIC_SURPRISE "
                    + json.dumps(
                        {
                            "arrival": boundary_decision.arrival,
                            "score": boundary_decision.score,
                            "threshold": boundary_decision.threshold,
                            "close_range": boundary_decision.close_range,
                            "reason": boundary_decision.reason,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        prompt = DEFAULT_PROMPT.format(task=instruction)
        inference_seed = (
            None
            if self.seed is None
            else int(self.seed) + (self._replan_count if self._advance_policy_seed else 0)
        )
        infer_kwargs = {
            "prompt": prompt,
            "input_image": temporal_video,
            "action_horizon": self.action_horizon,
            "proprio": proprio,
            "negative_prompt": self.negative_prompt,
            "text_cfg_scale": self.text_cfg_scale,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": self.sigma_shift,
            "seed": inference_seed,
            "rand_device": self.rand_device,
            "tiled": self.tiled,
            "full_kv_frame_index": self._full_kv_frame_index,
        }
        if (
            getattr(self, "dynamic_surprise_online", False)
            or getattr(self, "wrist_event_online", False)
            or getattr(self, "embodied_information_online", False)
            or getattr(self, "control_information_online", False)
            or getattr(self, "latent_kernel_regime_online", False)
            or getattr(self, "physical_settle_rate_debt_online", False)
        ):
            infer_kwargs["dynamic_surprise_online"] = True
            if getattr(self, "physical_settle_rate_debt_online", False):
                infer_kwargs["dynamic_close_range"] = (
                    None
                    if physical_settle_segment is None
                    else (
                        physical_settle_segment.start,
                        physical_settle_segment.end,
                    )
                )
                infer_kwargs["dynamic_close_token_count"] = (
                    None if physical_settle_segment is None else 8
                )
            elif getattr(self, "latent_kernel_regime_online", False):
                infer_kwargs["dynamic_close_range"] = latent_kernel_close_range
                infer_kwargs["dynamic_close_token_count"] = (
                    None if latent_kernel_close_range is None else 8
                )
            elif getattr(self, "wrist_event_online", False):
                infer_kwargs["dynamic_close_range"] = wrist_arrival_decision.close_range
            elif getattr(self, "embodied_information_online", False):
                infer_kwargs["dynamic_close_range"] = embodied_close_range
            elif getattr(self, "control_information_online", False):
                infer_kwargs["dynamic_close_range"] = (
                    control_information_close_range
                )
                infer_kwargs["dynamic_close_token_count"] = (
                    control_information_close_token_count
                )
            else:
                infer_kwargs["dynamic_close_range"] = (
                    None if boundary_decision is None else boundary_decision.close_range
                )
        if (
            getattr(self.model, "native_cache_compressor", None) is not None
            or getattr(self.model, "layerwise_block_memory", None) is not None
        ):
            infer_kwargs["native_cache_state"] = self._native_cache_state
        else:
            infer_kwargs["full_kv_cache"] = self._full_kv_cache
        if "num_video_frames" in inspect.signature(self.model.infer_action).parameters:
            infer_kwargs["num_video_frames"] = int(self._num_video_frames)
        infer_t0 = time.perf_counter() if self.timing_enabled else 0.0
        with torch.no_grad():
            pred = self.model.infer_action(**infer_kwargs)
        wrist_probability = None
        current_wrist_latent = None
        source_wrist_innovation = None
        source_left_innovation = None
        source_right_innovation = None
        if getattr(self, "wrist_event_online", False):
            current_wrist_latent = pred.get("current_observation_latent")
            if not isinstance(current_wrist_latent, torch.Tensor):
                raise RuntimeError(
                    "wrist-event policy requires infer_action to return the current latent"
                )
            if self._wrist_event_latents:
                innovation = self._wrist_innovation_fn(
                    self._wrist_event_latents[-1], current_wrist_latent
                )
                source_left_innovation = float(innovation["left"].detach().cpu())
                source_right_innovation = float(innovation["right"].detach().cpu())
                source_wrist_innovation = float(
                    innovation["combined"].detach().cpu()
                )
            proposed_history = [*self._wrist_event_latents, current_wrist_latent]
            predictor_input = causal_latent_window(
                proposed_history,
                history=self._wrist_event_history,
            ).to(device=self.model.device, dtype=torch.float32)
            with torch.no_grad():
                wrist_probability = float(
                    torch.sigmoid(self._wrist_event_predictor(predictor_input))[0]
                    .detach()
                    .cpu()
                )
        # Commit memory state only after the whole inference call succeeds.
        self._full_kv_cache = pred["full_kv_cache"]
        self._native_cache_state = pred.get("native_cache_state")
        self._full_kv_frame_index += 1
        # Preserve every stride-4 observation so the next decision is encoded
        # with one continuous causal VAE stream instead of restarting the VAE
        # at each action-chunk boundary.
        self._temporal_frames = temporal_frames
        if getattr(self, "dynamic_surprise_online", False):
            if boundary_decision is not None:
                self._surprise_segmenter.commit(boundary_decision)
            self._surprise_latents.append(actual_latent)
            self._surprise_previous_action = pred["action"].unsqueeze(0).to(
                device=self.model.device, dtype=self.model.torch_dtype
            )
            self._surprise_previous_conditioning = {
                "context": pred["action_context"],
                "context_mask": pred["action_context_mask"],
                "video_context": pred["video_context"],
                "video_context_mask": pred["video_context_mask"],
            }
        if getattr(self, "wrist_event_online", False):
            self._wrist_event_segmenter.commit_arrival(wrist_arrival_decision)
            self._wrist_event_latents.append(current_wrist_latent.detach())
            self._wrist_event_latents = self._wrist_event_latents[
                -self._wrist_event_history :
            ]
            self._wrist_event_segmenter.schedule_next(wrist_probability)
            print(
                "FASTWAM_WRIST_EVENT_FORECAST "
                + json.dumps(
                    {
                        "source_decision": self._full_kv_frame_index - 1,
                        "target_decision": self._full_kv_frame_index,
                        "probability": wrist_probability,
                        "source_wrist_innovation": source_wrist_innovation,
                        "source_left_innovation": source_left_innovation,
                        "source_right_innovation": source_right_innovation,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if self.timing_enabled:
            self._timing_rollout["infer_s"] += time.perf_counter() - infer_t0
        if self._native_cache_state is not None:
            cache_metrics = summarize_native_cache_state(self._native_cache_state)
            cache_metrics.update(
                {
                    "episode": self.episode_count,
                    "replan": self._replan_count,
                    "frame_index": self._full_kv_frame_index - 1,
                    "cuda_peak_allocated_bytes": (
                        int(torch.cuda.max_memory_allocated(self.model.device))
                        if self.model.device.type == "cuda"
                        else 0
                    ),
                    "cuda_peak_reserved_bytes": (
                        int(torch.cuda.max_memory_reserved(self.model.device))
                        if self.model.device.type == "cuda"
                        else 0
                    ),
                }
            )
            print(
                "FASTWAM_NATIVE_CACHE_METRICS "
                + json.dumps(cache_metrics, sort_keys=True),
                flush=True,
            )

        action_tensor = pred["action"]  # [T, D]
        action_chunk = self._denormalize_action(action_tensor)[0]  # [T, D]
        return action_chunk

    def _fill_action_queue(self, observation: Dict[str, Any], instruction: str) -> None:
        action_chunk = self._infer_action_chunk(observation=observation, instruction=instruction)
        current_qpos = np.asarray(
            observation["joint_action"]["vector"], dtype=np.float32
        )
        active_arm = None
        arm_motion_energy = None
        if self._freeze_inactive_arm:
            arm_motion_energy = np.asarray(
                [
                    np.sqrt(np.mean((action_chunk[:, 0:6] - current_qpos[0:6]) ** 2)),
                    np.sqrt(np.mean((action_chunk[:, 7:13] - current_qpos[7:13]) ** 2)),
                ],
                dtype=np.float32,
            )
            dominant = int(np.argmax(arm_motion_energy))
            subordinate = 1 - dominant
            if arm_motion_energy[dominant] >= (
                self._active_arm_energy_ratio
                * max(float(arm_motion_energy[subordinate]), 1e-8)
            ):
                active_arm = "left" if dominant == 0 else "right"
                inactive_slice = slice(7, 14) if dominant == 0 else slice(0, 7)
                action_chunk[:, inactive_slice] = current_qpos[inactive_slice]
        # A freshly denoised absolute-qpos chunk can start away from the
        # measured state even though the expert trajectory is continuous.
        # Optionally ramp only the first few targets from the current qpos;
        # the switch defaults to zero so the paper-aligned baseline is intact.
        if self._action_start_blend_steps > 0:
            blend_steps = min(
                self._action_start_blend_steps, action_chunk.shape[0]
            )
            alpha = (
                np.arange(1, blend_steps + 1, dtype=np.float32)
                / float(blend_steps + 1)
            )[:, None]
            action_chunk[:blend_steps] = (
                current_qpos[None] * (1.0 - alpha)
                + action_chunk[:blend_steps] * alpha
            )
        if self._log_action_trajectory:
            trace = {
                "episode": self.episode_count,
                "replan": self._replan_count,
                "step": self.step_count,
                "inference_seed": (
                    None
                    if self.seed is None
                    else int(self.seed)
                    + (self._replan_count if self._advance_policy_seed else 0)
                ),
                "current_qpos": current_qpos.tolist(),
                "predicted_action_qpos": action_chunk.tolist(),
                "current_gripper_qpos": [
                    float(current_qpos[6]),
                    float(current_qpos[13]),
                ],
                "predicted_left_gripper_qpos": action_chunk[:, 6].tolist(),
                "predicted_right_gripper_qpos": action_chunk[:, 13].tolist(),
                "active_arm_filter": active_arm,
                "arm_motion_energy": (
                    None if arm_motion_energy is None else arm_motion_energy.tolist()
                ),
            }
            # RoboTwin does not install a handler for this module's logger.
            print(f"FASTWAM_ACTION_TRAJECTORY {json.dumps(trace)}", flush=True)
        self._replan_count += 1
        if self._action_delta_limit is not None:
            current = current_qpos.copy()
            projected = np.empty_like(action_chunk)
            for index, target in enumerate(action_chunk):
                bounded = np.clip(
                    target,
                    self._action_min,
                    self._action_max,
                )
                delta = np.clip(
                    bounded - current,
                    -self._action_delta_limit,
                    self._action_delta_limit,
                )
                current = np.clip(
                    current + delta,
                    self._action_min,
                    self._action_max,
                )
                projected[index] = current
            action_chunk = projected
        n_exec = min(self.replan_steps, action_chunk.shape[0])
        for i in range(n_exec):
            self.pending_actions.append(np.asarray(action_chunk[i], dtype=np.float32))

    def should_request_observation(self) -> bool:
        return (
            not self.pending_actions
            or (
                self.step_count > 0
                and self.step_count % self._temporal_sample_stride == 0
            )
        )

    def step(self, task_env, observation: Optional[Dict[str, Any]]) -> None:
        instruction = None
        if (
            getattr(self, "embodied_information_online", False)
            and observation is not None
            and self.step_count % 4 == 0
        ):
            instruction = task_env.get_instruction()
            if self._embodied_runtime._context is None:
                prompt = DEFAULT_PROMPT.format(task=instruction)
                context, context_mask = self.model.encode_prompt(prompt)
                self._embodied_runtime.set_prompt(context, context_mask)
            image = self._build_robotwin_image_tensor(observation)
            proprio = torch.as_tensor(
                observation["joint_action"]["vector"], dtype=torch.float32
            )
            event = self._embodied_runtime.observe(
                frame=self.step_count,
                image=image,
                proprio=proprio,
                executed_actions=self._embodied_executed_actions,
            )
            print(
                "FASTWAM_EMBODIED_DETECTOR "
                + json.dumps(
                    {
                        "frame": self.step_count,
                        "event": None if event is None else {
                            "confirmation_frame": event.confirmation_frame,
                            "reason": event.reason,
                            "information_score": event.information_score,
                            "accumulated_information": event.accumulated_information,
                            "changed_grippers": event.changed_grippers,
                        },
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if (
            getattr(self, "control_information_online", False)
            and observation is not None
            and self.step_count % 4 == 0
        ):
            instruction = task_env.get_instruction()
            runtime = self._control_information_runtime
            if runtime._context is None:
                prompt = DEFAULT_PROMPT.format(task=instruction)
                context, context_mask = self.model.encode_prompt(prompt)
                runtime.set_prompt(context, context_mask)
            image = self._build_robotwin_image_tensor(observation)
            proprio = torch.as_tensor(
                observation["joint_action"]["vector"], dtype=torch.float32
            )
            if self.model.device.type == "cuda":
                torch.cuda.synchronize(self.model.device)
            selector_started = time.perf_counter()
            event, score = runtime.observe_image(
                frame=self.step_count,
                image=image,
                raw_proprio=proprio,
                executed_actions=self._control_information_executed_actions,
            )
            diagnostics = None
            if score is not None:
                diagnostics = _control_information_diagnostic_metrics(
                    score=score,
                    observed_feature=runtime._features[self.step_count % 16][-1],
                    raw_proprio=proprio,
                    executed_actions=self._control_information_executed_actions[-16:],
                )
            if self.model.device.type == "cuda":
                torch.cuda.synchronize(self.model.device)
            selector_latency_ms = 1000.0 * (
                time.perf_counter() - selector_started
            )
            self._control_information_latencies_ms.append(selector_latency_ms)
            print(
                "FASTWAM_CONTROL_INFORMATION_DETECTOR "
                + json.dumps(
                    {
                        "frame": self.step_count,
                        "information": (
                            None if score is None else score.information
                        ),
                        "adaptation_factor": getattr(
                            runtime.boundary_state,
                            "adaptation_factor",
                            getattr(
                                runtime.boundary_state,
                                "episode_adaptation_factor",
                                None,
                            ),
                        ),
                        "calibration_frames": list(
                            getattr(
                                getattr(
                                    runtime.boundary_state,
                                    "adapter",
                                    getattr(
                                        runtime.boundary_state,
                                        "episode_adapter",
                                        None,
                                    ),
                                ),
                                "calibration_frames",
                                (),
                            )
                        ),
                        "segment_baseline_location": getattr(
                            getattr(
                                runtime.boundary_state, "segment_baseline", None
                            ),
                            "location",
                            None,
                        ),
                        "segment_baseline_scale": getattr(
                            getattr(
                                runtime.boundary_state, "segment_baseline", None
                            ),
                            "scale",
                            None,
                        ),
                        "segment_relative_innovation": getattr(
                            runtime.boundary_state,
                            "last_segment_relative_innovation",
                            None,
                        ),
                        "consecutive_evidence": getattr(
                            runtime.boundary_state, "consecutive_evidence", None
                        ),
                        "boundary_cusum": float(runtime.boundary_state.cusum),
                        "diagnostics": diagnostics,
                        "selector_latency_ms": selector_latency_ms,
                        "cuda_allocated_bytes": (
                            int(torch.cuda.memory_allocated(self.model.device))
                            if self.model.device.type == "cuda"
                            else 0
                        ),
                        "cuda_peak_allocated_bytes": (
                            int(torch.cuda.max_memory_allocated(self.model.device))
                            if self.model.device.type == "cuda"
                            else 0
                        ),
                        "event": None
                        if event is None
                        else _control_information_event_payload(event),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if (
            self.pending_actions
            and observation is not None
            and self.step_count > 0
            and self.step_count % self._temporal_sample_stride == 0
        ):
            self._temporal_frames.append(
                self._build_robotwin_image_tensor(observation)
            )
        if not self.pending_actions:
            if observation is None:
                raise ValueError(
                    "Observation is required when action queue is empty "
                    "(replan step for fastwam)."
                )
            if instruction is None:
                instruction = task_env.get_instruction()
            self._fill_action_queue(observation=observation, instruction=instruction)

        if not self.pending_actions:
            logger.warning("No action generated; skip current eval step.")
            return

        action = self.pending_actions.popleft()
        sim_t0 = time.perf_counter() if self.timing_enabled else 0.0
        task_env.take_action(action, action_type="qpos")
        if getattr(self, "embodied_information_online", False):
            self._embodied_executed_actions.append(
                torch.as_tensor(action, dtype=torch.float32).clone()
            )
        if getattr(self, "control_information_online", False):
            self._control_information_executed_actions.append(
                torch.as_tensor(action, dtype=torch.float32).clone()
            )
        if self.timing_enabled:
            self._timing_rollout["sim_s"] += time.perf_counter() - sim_t0
        self.step_count += 1

    def reset_timing_rollout(self) -> None:
        self._timing_rollout["infer_s"] = 0.0
        self._timing_rollout["sim_s"] = 0.0

    def get_timing_rollout(self) -> Dict[str, float]:
        return {
            "infer_s": float(self._timing_rollout["infer_s"]),
            "sim_s": float(self._timing_rollout["sim_s"]),
        }

    def reset(self) -> None:
        self.pending_actions.clear()
        self._full_kv_cache = None
        self._native_cache_state = None
        self._full_kv_frame_index = 0
        self._temporal_frames.clear()
        self._surprise_segmenter = OnlineSurpriseSegmenter(
            min_segment=2,
            max_segment=self._surprise_segmenter.max_segment,
            gamma=self._surprise_segmenter.gamma,
            window=self._surprise_segmenter.window,
        )
        self._surprise_latents.clear()
        self._surprise_previous_action = None
        self._surprise_previous_conditioning = None
        if getattr(self, "wrist_event_online", False):
            previous = self._wrist_event_segmenter
            self._wrist_event_segmenter = OnlineWristEventSegmenter(
                threshold=previous.threshold,
                min_segment=previous.min_segment,
                max_segment=previous.max_segment,
            )
            self._wrist_event_latents.clear()
        if getattr(self, "embodied_information_online", False):
            previous = self._embodied_runtime
            self._embodied_runtime = EmbodiedInformationOnlineRuntime(
                policy_model=previous.policy_model,
                initialization_model=previous.initialization_model,
                pca_artifact=previous.pca_artifact,
                predictor=previous.predictor,
                contextual_statistics=previous.contextual_statistics,
                selector_config=previous.selector_config,
                capture_fn=previous.capture_fn,
                feature_window=previous.feature_window,
            )
            self._embodied_executed_actions.clear()
        if getattr(self, "control_information_online", False):
            self._control_information_runtime = _fresh_control_information_runtime(
                self._control_information_runtime
            )
            self._control_information_executed_actions.clear()
            self._control_information_latencies_ms.clear()
        if getattr(self, "latent_kernel_regime_online", False):
            self._latent_kernel_runtime.reset()
        if getattr(self, "physical_settle_rate_debt_online", False):
            self._physical_settle_rate_runtime.reset()
        self.episode_count += 1
        self.step_count = 0
        self._replan_count = 0
        self.reset_timing_rollout()
        if self.model.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.model.device)


def encode_obs(observation: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return observation


def get_model(usr_args: Dict[str, Any]):
    sim_cfg_path = usr_args.get("sim_cfg_path")
    sim_cfg_name = usr_args.get("sim_cfg_name")
    sim_task = usr_args.get("sim_task")
    cfg = _compose_sim_cfg(
        sim_cfg_path=sim_cfg_path,
        sim_cfg_name=sim_cfg_name,
        sim_task=sim_task,
    )
    model_cfg = cfg.model
    if not _is_none_like(usr_args.get("wan_model_id")):
        model_cfg.model_id = str(usr_args["wan_model_id"])
    if not _is_none_like(usr_args.get("tokenizer_model_id")):
        model_cfg.tokenizer_model_id = str(usr_args["tokenizer_model_id"])
    if not _is_none_like(usr_args.get("redirect_common_files")):
        model_cfg.redirect_common_files = _parse_bool(usr_args["redirect_common_files"])
    if not _is_none_like(usr_args.get("skip_dit_load_from_pretrain")):
        model_cfg.skip_dit_load_from_pretrain = _parse_bool(
            usr_args["skip_dit_load_from_pretrain"]
        )

    checkpoint_path = usr_args.get("ckpt_setting")
    if _is_none_like(checkpoint_path):
        raise ValueError("`ckpt_setting` is required and must be a valid checkpoint path.")

    device = str(usr_args.get("device") or cfg.EVALUATION.get("device") or "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; fallback device to cpu.")
        device = "cpu"

    mixed_precision = str(usr_args.get("mixed_precision") or cfg.get("mixed_precision", "bf16"))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)

    dataset_stats_path = _resolve_dataset_stats_path(
        dataset_stats_path=usr_args.get("dataset_stats_path"),
    )

    action_horizon = _parse_optional_int(usr_args.get("action_horizon"))
    if action_horizon is None:
        eval_horizon = _parse_optional_int(cfg.EVALUATION.get("action_horizon"))
        action_horizon = eval_horizon if eval_horizon is not None else int(cfg.data.train.num_frames) - 1
    if action_horizon <= 0:
        raise ValueError(f"`action_horizon` must be positive, got {action_horizon}")

    replan_steps = _parse_optional_int(usr_args.get("replan_steps"))
    if replan_steps is None:
        replan_steps = int(cfg.EVALUATION.get("replan_steps", 8))

    num_inference_steps = _parse_optional_int(usr_args.get("num_inference_steps"))
    if num_inference_steps is None:
        num_inference_steps = int(cfg.EVALUATION.get("num_inference_steps", cfg.eval_num_inference_steps))

    sigma_shift = _parse_optional_float(usr_args.get("sigma_shift"))
    if sigma_shift is None:
        sigma_shift = _parse_optional_float(cfg.EVALUATION.get("sigma_shift"))

    seed = _parse_optional_int(usr_args.get("seed"))
    text_cfg_scale = float(usr_args.get("text_cfg_scale", cfg.EVALUATION.get("text_cfg_scale", 1.0)))
    negative_prompt = str(usr_args.get("negative_prompt", cfg.EVALUATION.get("negative_prompt", "")))
    rand_device = str(usr_args.get("rand_device", cfg.EVALUATION.get("rand_device", "cpu")))
    tiled = _parse_bool(usr_args.get("tiled", cfg.EVALUATION.get("tiled", False)))
    timing_enabled = _parse_bool(
        usr_args.get("timing_enabled", cfg.EVALUATION.get("timing_enabled", False))
    )
    action_delta_fraction = _parse_optional_float(
        usr_args.get(
            "action_delta_fraction",
            cfg.EVALUATION.get("action_delta_fraction"),
        )
    )
    dynamic_surprise_online = _parse_bool(
        usr_args.get("dynamic_surprise_online", False)
    )
    wrist_event_online = _parse_bool(usr_args.get("wrist_event_online", False))
    embodied_information_online = _parse_bool(
        usr_args.get("embodied_information_online", False)
    )
    control_information_online = _parse_bool(
        usr_args.get("control_information_online", False)
    )
    latent_kernel_regime_online = _parse_bool(
        usr_args.get("latent_kernel_regime_online", False)
    )
    physical_settle_rate_debt_online = _parse_bool(
        usr_args.get("physical_settle_rate_debt_online", False)
    )

    policy = WorldActionRobotWinPolicy(
        model_cfg=model_cfg,
        processor_cfg=cfg.data.train.processor,
        checkpoint_path=str(checkpoint_path),
        dataset_stats_path=dataset_stats_path,
        device=device,
        model_dtype=model_dtype,
        action_horizon=action_horizon,
        replan_steps=replan_steps,
        num_inference_steps=num_inference_steps,
        sigma_shift=sigma_shift,
        seed=seed,
        text_cfg_scale=text_cfg_scale,
        negative_prompt=negative_prompt,
        rand_device=rand_device,
        tiled=tiled,
        timing_enabled=timing_enabled,
        num_video_frames=(int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1,
        action_delta_fraction=action_delta_fraction,
        dynamic_surprise_online=dynamic_surprise_online,
        dynamic_surprise_sigma=float(usr_args.get("dynamic_surprise_sigma", 1.0)),
        dynamic_surprise_gamma=float(usr_args.get("dynamic_surprise_gamma", 1.5)),
        dynamic_surprise_window=int(usr_args.get("dynamic_surprise_window", 5)),
        dynamic_surprise_max_segment=int(
            usr_args.get("dynamic_surprise_max_segment", 8)
        ),
        dynamic_surprise_scorer_source=str(
            usr_args.get("dynamic_surprise_scorer_source", "policy")
        ),
        dynamic_surprise_init_action_dit_path=usr_args.get(
            "dynamic_surprise_init_action_dit_path"
        ),
        dynamic_surprise_noise_mode=str(
            usr_args.get("dynamic_surprise_noise_mode", "per_transition")
        ),
        wrist_event_online=wrist_event_online,
        wrist_event_predictor_repo=usr_args.get("wrist_event_predictor_repo"),
        wrist_event_checkpoint=usr_args.get("wrist_event_checkpoint"),
        wrist_event_threshold=float(usr_args.get("wrist_event_threshold", 0.5)),
        wrist_event_history=int(usr_args.get("wrist_event_history", 3)),
        wrist_event_min_segment=int(usr_args.get("wrist_event_min_segment", 2)),
        wrist_event_max_segment=int(usr_args.get("wrist_event_max_segment", 8)),
        embodied_information_online=embodied_information_online,
        embodied_information_lock=usr_args.get("embodied_information_lock"),
        embodied_information_pca=usr_args.get("embodied_information_pca"),
        embodied_information_predictor=usr_args.get(
            "embodied_information_predictor"
        ),
        embodied_information_statistics=usr_args.get(
            "embodied_information_statistics"
        ),
        embodied_information_init_action_dit_path=usr_args.get(
            "embodied_information_init_action_dit_path"
        ),
        control_information_online=control_information_online,
        latent_kernel_regime_online=latent_kernel_regime_online,
        physical_settle_rate_debt_online=physical_settle_rate_debt_online,
        control_information_lock=usr_args.get("control_information_lock"),
        control_information_pca=usr_args.get("control_information_pca"),
        control_information_feature_predictor=usr_args.get(
            "control_information_feature_predictor"
        ),
        control_information_control_probe=usr_args.get(
            "control_information_control_probe"
        ),
        control_information_statistics=usr_args.get(
            "control_information_statistics"
        ),
        control_information_normalization=usr_args.get(
            "control_information_normalization"
        ),
        control_information_init_action_dit_path=usr_args.get(
            "control_information_init_action_dit_path"
        ),
    )
    robotwin_root = os.environ.get("FASTWAM_ROBOTWIN_ROOT")
    if robotwin_root:
        resolved_root = Path(robotwin_root).expanduser().resolve()
        if not resolved_root.is_dir():
            raise FileNotFoundError(
                f"FASTWAM_ROBOTWIN_ROOT is not a directory: {resolved_root}"
            )
        os.chdir(resolved_root)
        print(
            "FASTWAM_ROBOTWIN_CWD "
            + json.dumps({"cwd": str(Path.cwd())}, sort_keys=True),
            flush=True,
        )
    return policy


def eval(TASK_ENV, model, observation: Optional[Dict[str, Any]]):
    obs = encode_obs(observation)
    model.step(TASK_ENV, obs)


def reset_model(model):
    model.reset()
