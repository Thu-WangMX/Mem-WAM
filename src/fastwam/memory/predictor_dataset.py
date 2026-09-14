"""Strict-causal same-phase examples for WAM feature prediction."""

from __future__ import annotations

from typing import Any

import torch

from fastwam.memory.four_phase_temporal import PHASE_OFFSETS


CONDITION_ACTION_FRAMES = 16
ACTION_DIM = 14
PROPRIO_DIM = 14
CONDITION_DIM = CONDITION_ACTION_FRAMES * ACTION_DIM + PROPRIO_DIM


def build_predictor_examples(
    *,
    episode: int,
    projected_phases: dict[str, dict[str, torch.Tensor]],
    actions: torch.Tensor,
    proprio: torch.Tensor,
    mode: str,
    max_history: int = 8,
) -> list[dict[str, Any]]:
    if mode not in {"visual_only", "visual_action"}:
        raise ValueError("mode must be visual_only or visual_action")
    actions = torch.as_tensor(actions, dtype=torch.float32)
    proprio = torch.as_tensor(proprio, dtype=torch.float32)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError("actions must have shape [T,14]")
    if proprio.shape != actions.shape:
        raise ValueError("proprio must match action shape [T,14]")
    if int(max_history) <= 0:
        raise ValueError("max_history must be positive")
    if set(projected_phases) != {str(value) for value in PHASE_OFFSETS}:
        raise ValueError("projected phase streams are incomplete")
    examples = []
    for phase in PHASE_OFFSETS:
        row = projected_phases[str(phase)]
        frames = torch.as_tensor(row["frame_indices"], dtype=torch.int64)
        features = torch.as_tensor(row["features"], dtype=torch.float32)
        warmup = torch.as_tensor(row["warmup"], dtype=torch.bool)
        if features.ndim != 2 or frames.shape != warmup.shape or len(frames) != len(features):
            raise ValueError(f"phase {phase} projected feature shape mismatch")
        if warmup.tolist() != [True, *([False] * (len(frames) - 1))]:
            raise ValueError(f"phase {phase} warmup contract mismatch")
        if any(int(frame) % 16 != phase for frame in frames.tolist()):
            raise ValueError(f"phase {phase} frame contract mismatch")
        for index in range(1, len(frames)):
            target_frame = int(frames[index])
            action_start = target_frame - CONDITION_ACTION_FRAMES
            if action_start < 0 or target_frame >= len(actions):
                raise IndexError("predictor target lacks a complete causal control interval")
            history_start = max(0, index - int(max_history))
            history_frames = frames[history_start:index].tolist()
            condition_action_frames = list(range(action_start, target_frame))
            condition = torch.cat(
                [actions[action_start:target_frame].reshape(-1), proprio[target_frame]]
            )
            if condition.shape != (CONDITION_DIM,):
                raise RuntimeError("predictor condition dimension mismatch")
            if mode == "visual_only":
                condition = torch.zeros_like(condition)
            examples.append(
                {
                    "episode": int(episode),
                    "phase": phase,
                    "target_frame": target_frame,
                    "history_frames": [int(value) for value in history_frames],
                    "history": features[history_start:index].clone(),
                    "target": features[index].clone(),
                    "condition_action_frames": condition_action_frames,
                    "proprio_frame": target_frame,
                    "condition": condition,
                    "mode": mode,
                }
            )
    examples.sort(key=lambda row: int(row["target_frame"]))
    return examples
