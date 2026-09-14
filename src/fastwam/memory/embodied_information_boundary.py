"""Causal embodied anchors plus accumulated WAM information boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch


PHASES = (0, 4, 8, 12)
ROBUST_SCALE = 1.4826


def _context(frame: int, depth_cap: int) -> tuple[int, int]:
    frame = int(frame)
    phase = frame % 16
    if phase not in PHASES or frame < 16:
        raise ValueError("predictive residual frame must be >=16 and four-frame aligned")
    depth = min((frame - phase) // 16, int(depth_cap))
    return PHASES.index(phase), depth - 1


def fit_contextual_residual_statistics(
    residuals_by_episode: Mapping[int, Mapping[str, torch.Tensor]],
    *, episodes: Iterable[int], depth_cap: int = 4,
) -> dict[str, Any]:
    selected = [int(value) for value in episodes]
    if not selected or len(selected) != len(set(selected)) or any(
        episode < 0 or episode > 29 for episode in selected
    ):
        raise ValueError("contextual statistics require unique episodes from 0-29")
    depth_cap = int(depth_cap)
    if depth_cap <= 0:
        raise ValueError("depth_cap must be positive")
    stream_count = None
    buckets = [[[] for _ in range(depth_cap)] for _ in PHASES]
    for episode in selected:
        if episode not in residuals_by_episode:
            raise KeyError(f"missing residual episode {episode}")
        payload = residuals_by_episode[episode]
        frames = torch.as_tensor(payload["frame_indices"], dtype=torch.int64)
        residuals = torch.as_tensor(payload["residuals"], dtype=torch.float32)
        if residuals.ndim != 2 or frames.shape != (len(residuals),):
            raise ValueError("contextual residual trace shape mismatch")
        stream_count = int(residuals.shape[1])
        for frame, row in zip(frames.tolist(), residuals):
            phase_index, depth_index = _context(frame, depth_cap)
            buckets[phase_index][depth_index].append(row)
    median = torch.empty((4, depth_cap, int(stream_count)), dtype=torch.float32)
    scale = torch.empty_like(median)
    for phase_index in range(4):
        for depth_index in range(depth_cap):
            if not buckets[phase_index][depth_index]:
                raise ValueError("a phase/history-depth residual context is empty")
            values = torch.stack(buckets[phase_index][depth_index])
            med = values.median(dim=0).values
            mad = (values - med).abs().median(dim=0).values * ROBUST_SCALE
            if bool((mad <= 1e-8).any()):
                raise ValueError("a contextual residual stream has collapsed MAD")
            median[phase_index, depth_index] = med
            scale[phase_index, depth_index] = mad
    return {
        "schema_version": "putback_contextual_residual_statistics_v1",
        "episodes": selected, "phases": list(PHASES), "depth_cap": depth_cap,
        "median": median, "mad_scale": scale,
    }


def contextual_residual_z(
    residual: torch.Tensor, *, frame: int, stats: Mapping[str, Any]
) -> torch.Tensor:
    phase_index, depth_index = _context(frame, int(stats["depth_cap"]))
    value = torch.as_tensor(residual, dtype=torch.float32)
    median = torch.as_tensor(stats["median"])[phase_index, depth_index]
    scale = torch.as_tensor(stats["mad_scale"])[phase_index, depth_index]
    if value.shape != median.shape or bool((scale <= 1e-8).any()):
        raise ValueError("contextual residual dimensions or scale are invalid")
    return ((value - median) / scale).clamp_min(0.0)


@dataclass(frozen=True)
class EmbodiedBoundaryEvent:
    confirmation_frame: int
    group_start: int
    group_end: int
    reason: str
    information_score: float
    accumulated_information: float
    changed_grippers: tuple[int, ...]


class EmbodiedInformationBoundaryState:
    def __init__(
        self, *, information_budget: float, top_k: int, clip_z: float,
        detector_stride: int = 4, min_information_units: int = 4,
        max_units: int = 24, initial_group_start: int = 0,
        gripper_dimensions: Sequence[int] = (6, 13), gripper_threshold: float = .5,
    ) -> None:
        self.information_budget = float(information_budget)
        self.top_k = int(top_k)
        self.clip_z = float(clip_z)
        self.detector_stride = int(detector_stride)
        self.min_information_units = int(min_information_units)
        self.max_units = int(max_units)
        self.group_start = int(initial_group_start)
        self.gripper_dimensions = tuple(int(value) for value in gripper_dimensions)
        self.gripper_threshold = float(gripper_threshold)
        if min(self.information_budget, self.clip_z) <= 0 or self.top_k <= 0:
            raise ValueError("information configuration must be positive")
        if self.detector_stride != 4 or not 0 < self.min_information_units <= self.max_units:
            raise ValueError("invalid detector unit configuration")
        self.last_frame: int | None = None
        self.last_grippers: torch.Tensor | None = None
        self.accumulated_information = 0.0
        self.retroactive_boundary_count = 0
        self.embodied_boundary_count = 0
        self.wam_information_boundary_count = 0
        self.forced_maximum_boundary_count = 0
        self.terminal_tail_count = 0

    def _emit(self, *, frame, reason, score, changed):
        event = EmbodiedBoundaryEvent(
            confirmation_frame=int(frame), group_start=self.group_start,
            group_end=int(frame), reason=reason, information_score=float(score),
            accumulated_information=float(self.accumulated_information),
            changed_grippers=tuple(changed),
        )
        if event.group_end <= event.group_start:
            raise RuntimeError("boundary creates an empty group")
        self.group_start = int(frame)
        self.accumulated_information = 0.0
        return event

    def update(self, *, frame: int, proprio: torch.Tensor,
               standardized_residual: torch.Tensor | None):
        frame = int(frame)
        if self.last_frame is None:
            if frame != self.group_start:
                raise ValueError("episode must begin at its initial group start")
        elif frame != self.last_frame + self.detector_stride:
            raise ValueError("observations must arrive every four frames")
        self.last_frame = frame
        state = torch.as_tensor(proprio, dtype=torch.float32)
        if state.ndim != 1 or max(self.gripper_dimensions) >= len(state):
            raise ValueError("proprio does not contain configured gripper dimensions")
        grippers = state[list(self.gripper_dimensions)] >= self.gripper_threshold
        changed = () if self.last_grippers is None else tuple(
            self.gripper_dimensions[index]
            for index in torch.nonzero(grippers != self.last_grippers).flatten().tolist()
        )
        self.last_grippers = grippers
        if frame < 16:
            if standardized_residual is not None:
                raise ValueError("four phase anchors are residual-free warmups")
            return None
        if standardized_residual is None:
            raise ValueError("post-warmup observations require a WAM residual")
        z = torch.as_tensor(standardized_residual, dtype=torch.float32).clamp(0, self.clip_z)
        if z.ndim != 1 or self.top_k > len(z) or not bool(torch.isfinite(z).all()):
            raise ValueError("standardized WAM residual is invalid")
        score = float(torch.topk(z, self.top_k).values.mean().item())
        self.accumulated_information += score
        units = (frame - self.group_start) // self.detector_stride
        if changed:
            event = self._emit(
                frame=frame, reason="embodied_gripper_transition", score=score, changed=changed
            )
            self.embodied_boundary_count += 1
            return event
        if units >= self.max_units:
            event = self._emit(
                frame=frame, reason="forced_maximum", score=score, changed=()
            )
            self.forced_maximum_boundary_count += 1
            return event
        if units >= self.min_information_units and self.accumulated_information >= self.information_budget:
            event = self._emit(
                frame=frame, reason="wam_information_budget", score=score, changed=()
            )
            self.wam_information_boundary_count += 1
            return event
        return None

    def finalize(self, *, frame: int):
        frame = int(frame)
        if frame < self.group_start or (frame - self.group_start) % self.detector_stride:
            raise ValueError("terminal frame is not aligned after current group")
        if frame == self.group_start:
            return None
        event = self._emit(
            frame=frame, reason="terminal_tail", score=0.0, changed=()
        )
        self.terminal_tail_count += 1
        return event

