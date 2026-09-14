"""Frozen-WAM future-action probe and compact immutable artifact format."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


MAGIC = b"FASTWAM_CONTROL_INFORMATION_PROBE_V1\n"
SCHEMA_VERSION = "putback_control_information_probe_v1"


@dataclass(frozen=True)
class CounterfactualControlScore:
    information: float
    predicted_feature: torch.Tensor
    prior_action: torch.Tensor
    posterior_action: torch.Tensor


class FutureActionProbe(nn.Module):
    """Predict a normalized future action chunk from frozen WAM state."""

    def __init__(
        self,
        *,
        feature_dim: int = 1280,
        proprio_dim: int = 14,
        hidden_dim: int = 512,
        proprio_hidden_dim: int = 128,
        horizon: int = 16,
        action_dim: int = 14,
    ) -> None:
        super().__init__()
        dimensions = {
            "feature_dim": feature_dim,
            "proprio_dim": proprio_dim,
            "hidden_dim": hidden_dim,
            "proprio_hidden_dim": proprio_hidden_dim,
            "horizon": horizon,
            "action_dim": action_dim,
        }
        if any(int(value) <= 0 for value in dimensions.values()):
            raise ValueError("all future-action probe dimensions must be positive")
        self.feature_dim = int(feature_dim)
        self.proprio_dim = int(proprio_dim)
        self.hidden_dim = int(hidden_dim)
        self.proprio_hidden_dim = int(proprio_hidden_dim)
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.feature_encoder = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim), nn.SiLU()
        )
        self.proprio_encoder = nn.Sequential(
            nn.Linear(self.proprio_dim, self.proprio_hidden_dim), nn.SiLU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim + self.proprio_hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.horizon * self.action_dim),
        )

    def forward(self, feature: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 2 or feature.shape[1] != self.feature_dim:
            raise ValueError("feature must have shape [B,feature_dim]")
        if proprio.shape != (len(feature), self.proprio_dim):
            raise ValueError("proprio must have shape [B,proprio_dim]")
        if not bool(torch.isfinite(feature).all()) or not bool(
            torch.isfinite(proprio).all()
        ):
            raise ValueError("future-action probe inputs must be finite")
        fused = torch.cat(
            [self.feature_encoder(feature), self.proprio_encoder(proprio)], dim=-1
        )
        output = self.fusion(fused).reshape(
            len(feature), self.horizon, self.action_dim
        )
        if not bool(torch.isfinite(output).all()):
            raise RuntimeError("future-action probe produced non-finite output")
        return output


def masked_action_huber_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Average Huber loss over valid time steps and every action dimension."""

    prediction = torch.as_tensor(prediction)
    target = torch.as_tensor(target, device=prediction.device, dtype=prediction.dtype)
    mask = torch.as_tensor(mask, device=prediction.device, dtype=torch.bool)
    if prediction.ndim != 3 or target.shape != prediction.shape:
        raise ValueError("prediction and target must share shape [B,H,D]")
    if mask.shape != prediction.shape[:2]:
        raise ValueError("mask must have shape [B,H]")
    expanded = mask.unsqueeze(-1).expand_as(prediction)
    if not bool(expanded.any()):
        raise ValueError("masked action loss requires at least one valid target")
    return F.huber_loss(
        prediction[expanded], target[expanded], delta=1.0, reduction="mean"
    )


def _config_dict(config: Mapping[str, Any]) -> dict[str, int]:
    keys = (
        "feature_dim",
        "proprio_dim",
        "hidden_dim",
        "proprio_hidden_dim",
        "horizon",
        "action_dim",
    )
    if set(config) != set(keys):
        raise ValueError(f"control-probe config keys must be exactly {keys}")
    values = {key: int(config[key]) for key in keys}
    if any(value <= 0 for value in values.values()):
        raise ValueError("control-probe config dimensions must be positive")
    return values


def save_compact_control_probe(
    path: str | Path,
    model: FutureActionProbe,
    *,
    config: Mapping[str, Any],
    report: Mapping[str, Any],
) -> None:
    """Atomically save only frozen model tensors and audit metadata."""

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite control probe: {path}")
    configuration = _config_dict(config)
    tensors = []
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().float().contiguous()
        tensors.append((name, tensor))
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "config": configuration,
        "report": dict(report),
        "tensors": [
            {
                "name": name,
                "dtype": "float32",
                "shape": list(tensor.shape),
                "nbytes": tensor.numel() * tensor.element_size(),
            }
            for name, tensor in tensors
        ],
    }
    header = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(MAGIC)
        stream.write(struct.pack("<Q", len(header)))
        stream.write(header)
        for _, tensor in tensors:
            stream.write(tensor.numpy().tobytes(order="C"))
    temporary.replace(path)


def load_compact_control_probe(
    path: str | Path,
    *,
    device: torch.device,
) -> tuple[FutureActionProbe, dict[str, Any]]:
    """Load, validate, freeze, and return one compact control probe."""

    path = Path(path)
    with path.open("rb") as stream:
        if stream.readline() != MAGIC:
            raise ValueError("compact control-probe magic is invalid")
        raw_size = stream.read(8)
        if len(raw_size) != 8:
            raise ValueError("compact control-probe header size is truncated")
        header_size = struct.unpack("<Q", raw_size)[0]
        metadata = json.loads(stream.read(header_size))
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("compact control-probe schema is incompatible")
        state: dict[str, torch.Tensor] = {}
        for row in metadata["tensors"]:
            if row["dtype"] != "float32":
                raise ValueError("compact control-probe tensor dtype is unsupported")
            raw = stream.read(int(row["nbytes"]))
            if len(raw) != int(row["nbytes"]):
                raise ValueError("compact control-probe tensor is truncated")
            array = np.frombuffer(raw, dtype=np.float32).reshape(row["shape"]).copy()
            state[str(row["name"])] = torch.from_numpy(array)
        if stream.read(1):
            raise ValueError("compact control-probe artifact has trailing bytes")

    model = FutureActionProbe(**_config_dict(metadata["config"]))
    model.load_state_dict(state, strict=True)
    model.to(device=device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, metadata


def _require_frozen(module: nn.Module, *, name: str) -> None:
    if module.training or any(parameter.requires_grad for parameter in module.parameters()):
        raise ValueError(f"{name} must be frozen and in evaluation mode")


def _module_device(module: nn.Module, fallback: torch.device) -> torch.device:
    parameter = next(iter(module.parameters()), None)
    if parameter is not None:
        return parameter.device
    buffer = next(iter(module.buffers()), None)
    return fallback if buffer is None else buffer.device


@torch.inference_mode()
def score_counterfactual_control_information(
    *,
    feature_predictor: nn.Module,
    control_probe: nn.Module,
    history: torch.Tensor,
    dynamics_condition: torch.Tensor,
    observed_feature: torch.Tensor,
    normalized_proprio: torch.Tensor,
) -> CounterfactualControlScore:
    """Measure the future-control change caused by the unpredictable WAM state."""

    _require_frozen(feature_predictor, name="feature predictor")
    _require_frozen(control_probe, name="control probe")
    history = torch.as_tensor(history, dtype=torch.float32)
    condition = torch.as_tensor(dynamics_condition, dtype=torch.float32)
    observed = torch.as_tensor(observed_feature, dtype=torch.float32)
    proprio = torch.as_tensor(normalized_proprio, dtype=torch.float32)
    values = (history, condition, observed, proprio)
    if history.ndim != 2 or len(history) <= 0:
        raise ValueError("history must have shape [T,feature_dim] with T positive")
    if condition.ndim != 1 or observed.shape != (history.shape[1],) or proprio.ndim != 1:
        raise ValueError("counterfactual-control input dimensions are incompatible")
    if not all(bool(torch.isfinite(value).all()) for value in values):
        raise ValueError("counterfactual-control inputs must be finite")

    dynamics_device = _module_device(feature_predictor, history.device)
    predicted = feature_predictor(
        history[None].to(dynamics_device),
        condition[None].to(dynamics_device),
        lengths=torch.tensor([len(history)], dtype=torch.int64),
    )[0]
    if predicted.shape != observed.shape or not bool(torch.isfinite(predicted).all()):
        raise RuntimeError("feature predictor produced an incompatible or non-finite state")
    control_device = _module_device(control_probe, predicted.device)
    prior = control_probe(
        predicted[None].to(control_device), proprio[None].to(control_device)
    )[0]
    posterior = control_probe(
        observed[None].to(control_device), proprio[None].to(control_device)
    )[0]
    if prior.shape != posterior.shape or prior.ndim != 2:
        raise RuntimeError("control probe outputs must share shape [H,action_dim]")
    if not bool(torch.isfinite(prior).all()) or not bool(torch.isfinite(posterior).all()):
        raise RuntimeError("control probe produced non-finite actions")
    information = torch.sqrt(torch.mean((posterior - prior).square()))
    return CounterfactualControlScore(
        information=float(information.item()),
        predicted_feature=predicted.detach().float().cpu(),
        prior_action=prior.detach().float().cpu(),
        posterior_action=posterior.detach().float().cpu(),
    )
