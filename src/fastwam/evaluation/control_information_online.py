"""Strict-online frozen-WAM counterfactual control-information selector."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from fastwam.memory.control_information_boundary import (
    ControlInformationBoundaryState,
)
from fastwam.memory.control_information_dataset import (
    normalize_control_tensor,
    normalization_from_policy_stats,
)
from fastwam.memory.control_information_probe import (
    CounterfactualControlScore,
    load_compact_control_probe,
    score_counterfactual_control_information,
)
from fastwam.memory.multilayer_spatial_feature import capture_spatial_features
from fastwam.memory.planning_aligned_manifest import OnlinePlanningBoundaryAligner
from fastwam.evaluation.embodied_information_online import (
    load_predictor,
    project_captured_features,
)


CANDIDATE_LOCK_SCHEMA = "putback_locked_control_information_candidate_v1"
RUNTIME_LOCK_SCHEMA = "putback_locked_control_information_runtime_v1"
RUNTIME_SOURCE_KEYS = {
    "online_source",
    "planning_aligner_source",
    "offline_freezer_source",
}
LOCKED_ARTIFACT_KEYS = {
    "initialization_manifest",
    "feature_bank_manifest",
    "pca",
    "feature_predictor",
    "control_probe",
    "contextual_statistics",
    "normalization",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finalize_runtime_lock(
    path: str | Path,
    *,
    candidate_lock_path: str | Path,
    source_paths: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Extend an immutable selector candidate with exact runtime source hashes."""

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite runtime lock: {path}")
    if set(source_paths) != RUNTIME_SOURCE_KEYS:
        raise ValueError(f"runtime sources must be exactly {sorted(RUNTIME_SOURCE_KEYS)}")
    candidate_path = Path(candidate_lock_path)
    candidate = json.loads(candidate_path.read_text())
    if candidate.get("schema_version") != CANDIDATE_LOCK_SCHEMA:
        raise ValueError("control-information candidate lock schema is incompatible")
    payload = {
        "schema_version": RUNTIME_LOCK_SCHEMA,
        "method": "counterfactual_control_information",
        "candidate_sha256": sha256_file(candidate_path),
        "candidate": candidate,
        "runtime_hashes": {
            key: sha256_file(source_paths[key]) for key in sorted(RUNTIME_SOURCE_KEYS)
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return payload


def verify_locked_runtime_dependencies(
    *,
    runtime_lock_path: str | Path,
    candidate_lock_path: str | Path,
    artifact_paths: Mapping[str, str | Path],
    source_paths: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Reject a runtime whose selector, artifacts, or executable sources drifted."""

    runtime_path = Path(runtime_lock_path).resolve()
    candidate_path = Path(candidate_lock_path).resolve()
    runtime = json.loads(runtime_path.read_text())
    candidate = json.loads(candidate_path.read_text())
    if runtime.get("schema_version") != RUNTIME_LOCK_SCHEMA:
        raise ValueError("control-information runtime lock schema is incompatible")
    if candidate.get("schema_version") != CANDIDATE_LOCK_SCHEMA:
        raise ValueError("control-information candidate lock schema is incompatible")
    candidate_sha256 = sha256_file(candidate_path)
    if runtime.get("candidate_sha256") != candidate_sha256:
        raise ValueError("candidate lock sha256 mismatch")
    if runtime.get("candidate") != candidate:
        raise ValueError("runtime lock does not embed the candidate lock verbatim")
    if set(artifact_paths) != LOCKED_ARTIFACT_KEYS:
        raise ValueError(
            f"locked artifacts must be exactly {sorted(LOCKED_ARTIFACT_KEYS)}"
        )
    if set(source_paths) != RUNTIME_SOURCE_KEYS:
        raise ValueError(f"runtime sources must be exactly {sorted(RUNTIME_SOURCE_KEYS)}")
    candidate_hashes = candidate.get("hashes", {})
    for name in sorted(LOCKED_ARTIFACT_KEYS):
        actual = sha256_file(artifact_paths[name])
        expected = candidate_hashes.get(name)
        if actual != expected:
            raise ValueError(f"locked {name} sha256 mismatch: {actual} != {expected}")
    runtime_hashes = runtime.get("runtime_hashes", {})
    for name in sorted(RUNTIME_SOURCE_KEYS):
        actual = sha256_file(source_paths[name])
        expected = runtime_hashes.get(name)
        if actual != expected:
            raise ValueError(f"locked {name} sha256 mismatch: {actual} != {expected}")
    return runtime


def load_locked_control_information_artifacts(
    *,
    runtime_lock_path: str | Path,
    candidate_lock_path: str | Path,
    artifact_paths: Mapping[str, str | Path],
    source_paths: Mapping[str, str | Path],
    device: torch.device,
) -> dict[str, Any]:
    """Verify and load every frozen selector dependency used by online inference."""

    runtime = verify_locked_runtime_dependencies(
        runtime_lock_path=runtime_lock_path,
        candidate_lock_path=candidate_lock_path,
        artifact_paths=artifact_paths,
        source_paths=source_paths,
    )
    paths = {name: Path(path).resolve() for name, path in artifact_paths.items()}
    pca = torch.load(paths["pca"], map_location="cpu", weights_only=True)
    statistics = torch.load(
        paths["contextual_statistics"], map_location="cpu", weights_only=True
    )
    feature_predictor = load_predictor(paths["feature_predictor"], device=device)
    control_probe, control_probe_metadata = load_compact_control_probe(
        paths["control_probe"], device=device
    )
    normalization = normalization_from_policy_stats(
        json.loads(paths["normalization"].read_text())
    )
    candidate = runtime["candidate"]
    return {
        "runtime_lock": runtime,
        "pca": pca,
        "feature_predictor": feature_predictor,
        "control_probe": control_probe,
        "control_probe_metadata": control_probe_metadata,
        "contextual_statistics": statistics,
        "normalization": normalization,
        "selector_config": dict(candidate["candidate"]["selector_config"]),
    }


class ControlInformationOnlineRuntime:
    """Observe stride-four states and queue only forward planning boundaries."""

    def __init__(
        self,
        *,
        feature_predictor: torch.nn.Module,
        control_probe: torch.nn.Module,
        normalization: Mapping[str, torch.Tensor],
        contextual_statistics: Mapping[str, Any],
        selector_config: Mapping[str, Any],
        feature_window: int = 8,
        policy_model: Any | None = None,
        initialization_model: Any | None = None,
        pca_artifact: Mapping[str, Any] | None = None,
        capture_fn: Callable[..., Mapping[int, Mapping[str, torch.Tensor]]] = capture_spatial_features,
    ) -> None:
        self.feature_predictor = feature_predictor
        self.control_probe = control_probe
        self.normalization = {
            key: torch.as_tensor(value, dtype=torch.float32).clone()
            for key, value in normalization.items()
        }
        expected_normalization = {
            "action_mean",
            "action_std",
            "proprio_mean",
            "proprio_std",
        }
        if set(self.normalization) != expected_normalization:
            raise ValueError("online control normalization fields are incomplete")
        self.contextual_statistics = dict(contextual_statistics)
        self.selector_config = dict(selector_config)
        self.feature_window = int(feature_window)
        if self.feature_window <= 0:
            raise ValueError("feature_window must be positive")
        self.boundary_state = ControlInformationBoundaryState(
            statistics=self.contextual_statistics, **self.selector_config
        )
        self.aligner = OnlinePlanningBoundaryAligner(
            replan_stride=16, minimum_segment_decisions=2
        )
        self._features: dict[int, list[torch.Tensor]] = {
            phase: [] for phase in (0, 4, 8, 12)
        }
        self.policy_model = policy_model
        self.initialization_model = initialization_model
        self.pca_artifact = None if pca_artifact is None else dict(pca_artifact)
        self.capture_fn = capture_fn
        self._images: list[torch.Tensor] = []
        self._latents: dict[int, list[torch.Tensor]] = {
            phase: [] for phase in (0, 4, 8, 12)
        }
        self._context: torch.Tensor | None = None
        self._context_mask: torch.Tensor | None = None

    def set_prompt(self, context: torch.Tensor, context_mask: torch.Tensor) -> None:
        if self._context is not None:
            return
        self._context = torch.as_tensor(context).detach()
        self._context_mask = torch.as_tensor(context_mask, dtype=torch.bool).detach()

    @torch.inference_mode()
    def observe_projected(
        self,
        *,
        frame: int,
        projected_feature: torch.Tensor,
        raw_proprio: torch.Tensor,
        executed_actions: Sequence[torch.Tensor | Any],
    ) -> tuple[Any | None, CounterfactualControlScore | None]:
        frame = int(frame)
        if frame % 4:
            raise ValueError("online control-information frame must be stride-four aligned")
        feature = torch.as_tensor(projected_feature, dtype=torch.float32).reshape(-1)
        proprio = torch.as_tensor(raw_proprio, dtype=torch.float32).reshape(-1)
        if not bool(torch.isfinite(feature).all()) or not bool(torch.isfinite(proprio).all()):
            raise ValueError("online control-information state must be finite")
        if self.normalization["proprio_mean"].shape != proprio.shape:
            raise ValueError("online proprio dimension differs from locked normalization")
        normalized_proprio = normalize_control_tensor(
            proprio[None],
            mean=self.normalization["proprio_mean"],
            std=self.normalization["proprio_std"],
            name="proprio",
        )[0]
        phase = frame % 16
        score = None
        information = None
        if frame >= 16:
            history_values = self._features[phase][-self.feature_window :]
            if not history_values:
                raise RuntimeError("online same-phase WAM history is empty")
            if len(executed_actions) < 16:
                raise RuntimeError("online selector lacks sixteen executed actions")
            action = torch.stack(
                [
                    torch.as_tensor(value, dtype=torch.float32).reshape(-1)
                    for value in executed_actions[-16:]
                ]
            )
            if action.shape[1:] != (len(self.normalization["action_mean"]),):
                raise ValueError("executed action dimension differs from normalization")
            history = torch.stack(history_values).reshape(len(history_values), -1)
            dynamics_condition = torch.cat([action.reshape(-1), proprio])
            score = score_counterfactual_control_information(
                feature_predictor=self.feature_predictor,
                control_probe=self.control_probe,
                history=history,
                dynamics_condition=dynamics_condition,
                observed_feature=feature,
                normalized_proprio=normalized_proprio,
            )
            information = score.information
        self._features[phase].append(feature.detach().cpu())
        event = self.boundary_state.update(frame=frame, information=information)
        if event is not None:
            self.aligner.observe_confirmation(
                confirmation_frame=event.confirmation_frame, reason=event.reason
            )
        return event, score

    @torch.inference_mode()
    def observe_image(
        self,
        *,
        frame: int,
        image: torch.Tensor,
        raw_proprio: torch.Tensor,
        executed_actions: Sequence[torch.Tensor | Any],
    ) -> tuple[Any | None, CounterfactualControlScore | None]:
        if (
            self.policy_model is None
            or self.initialization_model is None
            or self.pca_artifact is None
        ):
            raise RuntimeError("online image path lacks frozen WAM runtime components")
        if self._context is None or self._context_mask is None:
            raise RuntimeError("online image path requires a frozen prompt")
        frame = int(frame)
        if frame != 4 * len(self._images):
            raise ValueError("online images must arrive contiguously every four frames")
        self._images.append(torch.as_tensor(image).detach())
        phase = frame % 16
        phase_start = phase // 4
        phase_video = torch.stack(self._images[phase_start:], dim=2)
        encoded = self.policy_model._encode_input_image_latents_tensor(
            input_image=phase_video, tiled=False
        )
        latest = encoded[:, :, -1:].detach()
        self._latents[phase].append(latest)
        causal = torch.cat(self._latents[phase][-self.feature_window :], dim=2)[0]
        captured = self.capture_fn(
            self.initialization_model,
            latents=causal,
            video_context=self._context,
            video_context_mask=self._context_mask,
        )
        projected = project_captured_features(captured, self.pca_artifact).reshape(-1)
        return self.observe_projected(
            frame=frame,
            projected_feature=projected,
            raw_proprio=raw_proprio,
            executed_actions=executed_actions,
        )

    def arrive_planning(self, *, frame: int) -> tuple[int, int] | None:
        return self.aligner.arrive_decision(int(frame))
