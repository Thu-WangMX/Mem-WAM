"""Strict-online initialization-WAM information selector for RoboTwin rollout."""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import numpy as np

from fastwam.memory.embodied_information_boundary import (
    EmbodiedInformationBoundaryState,
    contextual_residual_z,
)
from fastwam.memory.multilayer_spatial_feature import (
    FEATURE_LAYERS,
    FEATURE_REGIONS,
    capture_spatial_features,
)
from fastwam.memory.planning_aligned_manifest import OnlinePlanningBoundaryAligner
from fastwam.memory.predictive_feature_model import FeaturePredictor


EXPECTED_INITIALIZATION_FINGERPRINT = (
    "51652b699bc50c5e1b0fa582a996788ea041d341f013043cf7174f52de679bc8"
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def initialization_fingerprint(model: Any) -> str:
    modules = [
        ("action_encoder", model.action_expert.action_encoder),
        ("action_head", model.action_expert.head),
    ]
    if getattr(model, "proprio_encoder", None) is not None:
        modules.append(("proprio_encoder", model.proprio_encoder))
    digest = hashlib.sha256()
    for module_name, module in modules:
        for parameter_name, tensor in module.state_dict().items():
            value = tensor.detach().contiguous()
            digest.update(f"{module_name}.{parameter_name}".encode())
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def _stack_captured(captured: Mapping[int, Mapping[str, torch.Tensor]]) -> torch.Tensor:
    if tuple(captured) != FEATURE_LAYERS:
        raise ValueError(f"feature layers differ from locked order: {tuple(captured)}")
    rows = []
    for layer in FEATURE_LAYERS:
        if tuple(captured[layer]) != FEATURE_REGIONS:
            raise ValueError(f"feature regions differ from locked order at layer {layer}")
        rows.append(torch.stack([
            torch.as_tensor(captured[layer][region], dtype=torch.float32)
            for region in FEATURE_REGIONS
        ]))
    values = torch.stack(rows)
    if values.ndim != 3 or not bool(torch.isfinite(values).all()):
        raise ValueError("captured WAM feature tensor is malformed")
    return values


def project_captured_features(
    captured: Mapping[int, Mapping[str, torch.Tensor]],
    pca_artifact: Mapping[str, Any],
) -> torch.Tensor:
    values = _stack_captured(captured)
    mean = torch.as_tensor(pca_artifact["mean"], dtype=torch.float32)
    components = torch.as_tensor(pca_artifact["components"], dtype=torch.float32)
    scale = torch.as_tensor(pca_artifact["projected_scale"], dtype=torch.float32)
    if values.shape != mean.shape or components.shape[:2] != values.shape[:2]:
        raise ValueError("PCA artifact does not match captured feature streams")
    projected = torch.einsum("lrd,lrcd->lrc", values - mean, components)
    if projected.shape != scale.shape or bool((scale <= 0).any()):
        raise ValueError("PCA projected scale is incompatible")
    return projected / scale


def load_predictor(path: str | Path, *, device: torch.device) -> FeaturePredictor:
    path = Path(path)
    if path.suffix == ".fpbin":
        with path.open("rb") as stream:
            if stream.readline() != b"FASTWAM_PREDICTOR_V1\n":
                raise ValueError("compact predictor magic is invalid")
            header_size = struct.unpack("<Q", stream.read(8))[0]
            metadata = json.loads(stream.read(header_size))
            state = {}
            for row in metadata["tensors"]:
                if row["dtype"] != "float32":
                    raise ValueError(f"unsupported compact predictor dtype: {row['dtype']}")
                raw = stream.read(int(row["nbytes"]))
                if len(raw) != int(row["nbytes"]):
                    raise ValueError("compact predictor tensor payload is truncated")
                array = np.frombuffer(raw, dtype=np.float32).reshape(row["shape"]).copy()
                state[row["name"]] = torch.from_numpy(array)
            if stream.read(1):
                raise ValueError("compact predictor has trailing bytes")
        config = metadata["config"]
    else:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        config = payload["config"]
        state = payload["best_model"]
    model = FeaturePredictor(
        feature_dim=int(config["feature_dim"]),
        condition_dim=int(config["condition_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        condition_embedding_dim=int(config["condition_embedding_dim"]),
    ).to(device).eval()
    model.load_state_dict(state)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_locked_artifacts(
    *, lock_path: str | Path, pca_path: str | Path, predictor_path: str | Path,
    statistics_path: str | Path, device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any], FeaturePredictor, dict[str, Any]]:
    lock_path = Path(lock_path).resolve()
    lock = json.loads(lock_path.read_text())
    if lock.get("schema_version") != "putback_locked_embodied_information_candidate_v1":
        raise ValueError("selector lock schema is incompatible")
    paths = {
        "pca": Path(pca_path).resolve(),
        "predictor": Path(predictor_path).resolve(),
        "contextual_statistics": Path(statistics_path).resolve(),
    }
    hash_keys = {
        "pca": "pca_sha256",
        "predictor": "predictor_sha256",
        "contextual_statistics": "contextual_stats_sha256",
    }
    for key, path in paths.items():
        expected = lock["hashes"][hash_keys[key]]
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"locked {key} sha256 mismatch: {actual} != {expected}")
    pca = torch.load(paths["pca"], map_location="cpu", weights_only=True)
    statistics = torch.load(
        paths["contextual_statistics"], map_location="cpu", weights_only=True
    )
    predictor = load_predictor(paths["predictor"], device=device)
    return lock, pca, predictor, statistics


class EmbodiedInformationOnlineRuntime:
    """Observe every fourth simulator frame, then align events forward to planning."""

    def __init__(
        self, *, policy_model: Any, initialization_model: Any,
        pca_artifact: Mapping[str, Any], predictor: FeaturePredictor,
        contextual_statistics: Mapping[str, Any], selector_config: Mapping[str, Any],
        capture_fn: Callable[..., Mapping[int, Mapping[str, torch.Tensor]]] = capture_spatial_features,
        feature_window: int = 8,
    ) -> None:
        self.policy_model = policy_model
        self.initialization_model = initialization_model
        self.pca_artifact = dict(pca_artifact)
        self.predictor = predictor
        self.contextual_statistics = dict(contextual_statistics)
        self.selector_config = dict(selector_config)
        self.capture_fn = capture_fn
        self.feature_window = int(feature_window)
        if self.feature_window <= 0:
            raise ValueError("feature_window must be positive")
        self.boundary_state = EmbodiedInformationBoundaryState(**self.selector_config)
        self.aligner = OnlinePlanningBoundaryAligner(
            replan_stride=16, minimum_segment_decisions=2
        )
        self._images: list[torch.Tensor] = []
        self._latents: dict[int, list[torch.Tensor]] = {phase: [] for phase in (0, 4, 8, 12)}
        self._features: dict[int, list[torch.Tensor]] = {phase: [] for phase in (0, 4, 8, 12)}
        self._context: torch.Tensor | None = None
        self._context_mask: torch.Tensor | None = None

    def set_prompt(self, context: torch.Tensor, context_mask: torch.Tensor) -> None:
        if self._context is not None:
            return
        self._context = context.detach()
        self._context_mask = context_mask.detach()

    @torch.inference_mode()
    def observe(
        self, *, frame: int, image: torch.Tensor, proprio: torch.Tensor,
        executed_actions: Sequence[torch.Tensor | Any],
    ):
        frame = int(frame)
        if frame % 4 or frame != 4 * len(self._images):
            raise ValueError("embodied selector observations must be contiguous stride-4")
        if self._context is None or self._context_mask is None:
            raise RuntimeError("prompt context must be frozen before the first observation")
        self._images.append(image.detach())
        phase = frame % 16
        phase_start = phase // 4
        phase_video = torch.stack(self._images[phase_start:], dim=2)
        encoded = self.policy_model._encode_input_image_latents_tensor(
            input_image=phase_video, tiled=False
        )
        latest = encoded[:, :, -1:].detach()
        self._latents[phase].append(latest)
        causal = torch.cat(self._latents[phase][-self.feature_window:], dim=2)[0]
        captured = self.capture_fn(
            self.initialization_model,
            latents=causal,
            video_context=self._context,
            video_context_mask=self._context_mask,
        )
        projected = project_captured_features(captured, self.pca_artifact)
        standardized = None
        if frame >= 16:
            if len(executed_actions) < 16:
                raise RuntimeError("post-warmup selector lacks 16 executed actions")
            history_values = self._features[phase][-self.feature_window:]
            if not history_values:
                raise RuntimeError("same-phase feature history is empty")
            history = torch.stack(history_values).reshape(len(history_values), -1)
            action = torch.stack([
                torch.as_tensor(value, dtype=torch.float32)
                for value in executed_actions[-16:]
            ]).reshape(-1)
            state = torch.as_tensor(proprio, dtype=torch.float32).reshape(-1)
            condition = torch.cat([action, state])
            first_parameter = next(iter(self.predictor.parameters()), None)
            device = (
                self.policy_model.device
                if first_parameter is None
                else first_parameter.device
            )
            prediction = self.predictor(
                history[None].to(device),
                condition[None].to(device),
                lengths=torch.tensor([len(history)], dtype=torch.int64),
            )[0].cpu().reshape_as(projected)
            residual = (prediction - projected).square().mean(dim=-1).sqrt().flatten()
            standardized = contextual_residual_z(
                residual, frame=frame, stats=self.contextual_statistics
            )
        self._features[phase].append(projected.cpu())
        event = self.boundary_state.update(
            frame=frame,
            proprio=torch.as_tensor(proprio, dtype=torch.float32),
            standardized_residual=standardized,
        )
        if event is not None:
            self.aligner.observe_confirmation(
                confirmation_frame=event.confirmation_frame, reason=event.reason
            )
        return event

    def arrive_planning(self, *, frame: int) -> tuple[int, int] | None:
        return self.aligner.arrive_decision(int(frame))
