"""Strict-causal examples for a frozen-WAM future-action probe."""

from __future__ import annotations

from typing import Any, Mapping

import torch


PHASE_OFFSETS = (0, 4, 8, 12)
ACTION_DIM = 14
PROPRIO_DIM = 14


def _vector(value: torch.Tensor, *, size: int, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.shape != (size,) or not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be one finite vector of length {size}")
    return tensor


def normalize_control_tensor(
    value: torch.Tensor,
    *,
    mean: torch.Tensor,
    std: torch.Tensor,
    name: str,
) -> torch.Tensor:
    """Normalize a `[T, D]` control tensor with a locked finite scale."""

    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim != 2 or not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be a finite rank-two tensor")
    center = _vector(mean, size=tensor.shape[1], name=f"{name}_mean")
    scale = _vector(std, size=tensor.shape[1], name=f"{name}_std")
    if bool((scale <= 0).any()):
        singular = name[:-1] if name.endswith("s") else name
        raise ValueError(f"{singular}_std must be finite and positive")
    return (tensor - center) / scale


def normalization_from_policy_stats(
    payload: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Extract the policy's global z-score tensors for action and state."""

    try:
        action = payload["action"]["default"]
        state = payload["state"]["default"]
        result = {
            "action_mean": torch.as_tensor(
                action["global_mean"], dtype=torch.float32
            ).reshape(-1),
            "action_std": torch.as_tensor(
                action["global_std"], dtype=torch.float32
            ).reshape(-1),
            "proprio_mean": torch.as_tensor(
                state["global_mean"], dtype=torch.float32
            ).reshape(-1),
            "proprio_std": torch.as_tensor(
                state["global_std"], dtype=torch.float32
            ).reshape(-1),
        }
    except (KeyError, TypeError) as error:
        raise ValueError("normalization statistics lack global z-score fields") from error
    for name, value in result.items():
        if value.shape != (14,) or not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be one finite 14-vector")
        if name.endswith("_std") and bool((value <= 0).any()):
            raise ValueError(f"{name} must be positive")
    return result


def _validate_projected_phases(
    projected_phases: Mapping[str, Mapping[str, torch.Tensor]],
) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
    if set(projected_phases) != {str(value) for value in PHASE_OFFSETS}:
        raise ValueError("projected features must contain all four phase streams")
    rows: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    feature_dim: int | None = None
    for phase in PHASE_OFFSETS:
        stream = projected_phases[str(phase)]
        frames = torch.as_tensor(stream["frame_indices"], dtype=torch.int64)
        features = torch.as_tensor(stream["features"], dtype=torch.float32)
        warmup = torch.as_tensor(stream["warmup"], dtype=torch.bool)
        if frames.ndim != 1 or features.ndim != 2 or warmup.shape != frames.shape:
            raise ValueError(f"phase {phase} feature stream shape mismatch")
        if len(frames) != len(features) or not len(frames):
            raise ValueError(f"phase {phase} feature stream is empty or incomplete")
        if warmup.tolist() != [True, *([False] * (len(frames) - 1))]:
            raise ValueError(f"phase {phase} warmup contract mismatch")
        if any(int(frame) % 16 != phase for frame in frames.tolist()):
            raise ValueError(f"phase {phase} frame alignment mismatch")
        if len(frames) > 1 and not bool((frames[1:] > frames[:-1]).all()):
            raise ValueError(f"phase {phase} frames must increase strictly")
        if not bool(torch.isfinite(features).all()):
            raise ValueError(f"phase {phase} features are non-finite")
        if feature_dim is None:
            feature_dim = int(features.shape[1])
        elif int(features.shape[1]) != feature_dim:
            raise ValueError("projected feature dimensions differ between phases")
        rows.extend(
            (phase, frame, feature)
            for frame, feature in zip(frames.tolist(), features)
        )
    return rows


def build_control_probe_examples(
    *,
    episode: int,
    projected_phases: Mapping[str, Mapping[str, torch.Tensor]],
    actions: torch.Tensor,
    proprio: torch.Tensor,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    proprio_mean: torch.Tensor,
    proprio_std: torch.Tensor,
    horizon: int = 16,
) -> list[dict[str, Any]]:
    """Build samples whose future action chunk is a label, never an input."""

    episode = int(episode)
    horizon = int(horizon)
    if episode < 0 or horizon <= 0:
        raise ValueError("episode must be non-negative and horizon must be positive")
    raw_actions = torch.as_tensor(actions, dtype=torch.float32)
    raw_proprio = torch.as_tensor(proprio, dtype=torch.float32)
    if raw_actions.ndim != 2 or raw_actions.shape[1] != ACTION_DIM:
        raise ValueError("actions must have shape [T,14]")
    if raw_proprio.shape != (len(raw_actions), PROPRIO_DIM):
        raise ValueError("proprio must have the same T and shape [T,14]")
    normalized_actions = normalize_control_tensor(
        raw_actions, mean=action_mean, std=action_std, name="actions"
    )
    normalized_proprio = normalize_control_tensor(
        raw_proprio, mean=proprio_mean, std=proprio_std, name="proprio"
    )

    examples: list[dict[str, Any]] = []
    for phase, raw_frame, raw_feature in _validate_projected_phases(projected_phases):
        frame = int(raw_frame)
        if frame < 0 or frame >= len(normalized_actions):
            raise IndexError(f"feature frame {frame} is outside episode length")
        valid = min(horizon, len(normalized_actions) - frame)
        target = torch.zeros((horizon, ACTION_DIM), dtype=torch.float32)
        target[:valid] = normalized_actions[frame : frame + valid]
        examples.append(
            {
                "episode": episode,
                "phase": phase,
                "frame": frame,
                "feature": torch.as_tensor(raw_feature, dtype=torch.float32).clone(),
                "proprio": normalized_proprio[frame].clone(),
                "target_actions": target,
                "target_mask": torch.arange(horizon) < valid,
            }
        )
    examples.sort(key=lambda row: int(row["frame"]))
    if [int(row["frame"]) for row in examples] != sorted(
        int(row["frame"]) for row in examples
    ):
        raise RuntimeError("control-probe examples are not in causal order")
    return examples
