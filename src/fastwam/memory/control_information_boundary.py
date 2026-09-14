"""Strict-causal CUSUM boundaries over counterfactual control information."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch


PHASES = (0, 4, 8, 12)
ROBUST_SCALE = 1.4826
STATISTICS_SCHEMA = "putback_control_information_statistics_v1"


def _context(frame: int, depth_cap: int) -> tuple[int, int]:
    frame = int(frame)
    phase = frame % 16
    if frame < 16 or phase not in PHASES:
        raise ValueError("control-information frame must be >=16 and stride-four aligned")
    depth = min((frame - phase) // 16, int(depth_cap))
    return PHASES.index(phase), depth - 1


def fit_control_information_statistics(
    scores_by_episode: Mapping[int, Mapping[str, torch.Tensor]],
    *,
    episodes: Iterable[int],
    depth_cap: int = 4,
) -> dict[str, Any]:
    """Fit robust phase/history-depth statistics using training episodes only."""

    selected = [int(value) for value in episodes]
    if (
        not selected
        or len(selected) != len(set(selected))
        or any(episode < 0 or episode > 29 for episode in selected)
    ):
        raise ValueError("control-information statistics require unique episodes 0-29")
    depth_cap = int(depth_cap)
    if depth_cap <= 0:
        raise ValueError("depth_cap must be positive")
    buckets: list[list[list[torch.Tensor]]] = [
        [[] for _ in range(depth_cap)] for _ in PHASES
    ]
    for episode in selected:
        if episode not in scores_by_episode:
            raise KeyError(f"missing control-information episode {episode}")
        payload = scores_by_episode[episode]
        frames = torch.as_tensor(payload["frame_indices"], dtype=torch.int64)
        information = torch.as_tensor(payload["information"], dtype=torch.float32)
        if frames.ndim != 1 or information.shape != frames.shape:
            raise ValueError("control-information trace must have matching scalar rows")
        if not bool(torch.isfinite(information).all()) or bool((information < 0).any()):
            raise ValueError("control-information trace must be finite and non-negative")
        for frame, score in zip(frames.tolist(), information):
            phase_index, depth_index = _context(frame, depth_cap)
            buckets[phase_index][depth_index].append(score)
    median = torch.empty((len(PHASES), depth_cap), dtype=torch.float32)
    mad_scale = torch.empty_like(median)
    for phase_index in range(len(PHASES)):
        for depth_index in range(depth_cap):
            if not buckets[phase_index][depth_index]:
                raise ValueError("a control-information calibration context is empty")
            values = torch.stack(buckets[phase_index][depth_index])
            center = values.median()
            scale = (values - center).abs().median() * ROBUST_SCALE
            if not bool(torch.isfinite(scale)) or float(scale) <= 1e-8:
                raise ValueError("a control-information calibration MAD collapsed")
            median[phase_index, depth_index] = center
            mad_scale[phase_index, depth_index] = scale
    return {
        "schema_version": STATISTICS_SCHEMA,
        "episodes": selected,
        "phases": list(PHASES),
        "depth_cap": depth_cap,
        "median": median,
        "mad_scale": mad_scale,
    }


def _validated_statistics(
    statistics: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if statistics.get("schema_version") != STATISTICS_SCHEMA:
        raise ValueError("control-information statistics schema is incompatible")
    depth_cap = int(statistics["depth_cap"])
    median = torch.as_tensor(statistics["median"], dtype=torch.float32)
    scale = torch.as_tensor(statistics["mad_scale"], dtype=torch.float32)
    expected = (len(PHASES), depth_cap)
    if median.shape != expected or scale.shape != expected:
        raise ValueError("control-information statistics shape is incompatible")
    if not bool(torch.isfinite(median).all()) or not bool(torch.isfinite(scale).all()):
        raise ValueError("control-information statistics must be finite")
    if bool((scale <= 1e-8).any()):
        raise ValueError("control-information statistics scale must be positive")
    return median, scale, depth_cap


def contextual_information_z(
    information: float,
    *,
    frame: int,
    statistics: Mapping[str, Any],
) -> float:
    median, scale, depth_cap = _validated_statistics(statistics)
    value = float(information)
    if not torch.isfinite(torch.tensor(value)) or value < 0:
        raise ValueError("control information must be finite and non-negative")
    phase_index, depth_index = _context(int(frame), depth_cap)
    return max(
        0.0,
        (value - float(median[phase_index, depth_index]))
        / float(scale[phase_index, depth_index]),
    )


@dataclass(frozen=True)
class ControlInformationBoundaryEvent:
    confirmation_frame: int
    group_start: int
    group_end: int
    reason: str
    information: float
    standardized_information: float
    cusum: float


class ControlInformationBoundaryState:
    """Online-only event state with no gripper or future-frame interface."""

    def __init__(
        self,
        *,
        statistics: Mapping[str, Any],
        threshold: float,
        drift: float,
        decay: float,
        detector_stride: int = 4,
        min_units: int = 4,
        max_units: int = 24,
        initial_group_start: int = 0,
    ) -> None:
        _validated_statistics(statistics)
        self.statistics = dict(statistics)
        self.threshold = float(threshold)
        self.drift = float(drift)
        self.decay = float(decay)
        self.detector_stride = int(detector_stride)
        self.min_units = int(min_units)
        self.max_units = int(max_units)
        self.group_start = int(initial_group_start)
        if self.threshold <= 0 or self.drift < 0 or not 0 < self.decay <= 1:
            raise ValueError("CUSUM threshold/drift/decay configuration is invalid")
        if (
            self.detector_stride != 4
            or self.min_units <= 0
            or self.max_units < self.min_units
            or self.group_start % self.detector_stride
        ):
            raise ValueError("control-information detector/group configuration is invalid")
        self.last_frame: int | None = None
        self.cusum = 0.0
        self.retroactive_boundary_count = 0
        self.control_information_boundary_count = 0
        self.forced_maximum_boundary_count = 0
        self.terminal_tail_count = 0

    def _emit(
        self,
        *,
        frame: int,
        reason: str,
        information: float,
        standardized_information: float,
    ) -> ControlInformationBoundaryEvent:
        if int(frame) <= self.group_start:
            raise RuntimeError("control-information boundary creates an empty group")
        event = ControlInformationBoundaryEvent(
            confirmation_frame=int(frame),
            group_start=self.group_start,
            group_end=int(frame),
            reason=str(reason),
            information=float(information),
            standardized_information=float(standardized_information),
            cusum=float(self.cusum),
        )
        self.group_start = int(frame)
        self.cusum = 0.0
        return event

    def update(
        self,
        *,
        frame: int,
        information: float | None,
    ) -> ControlInformationBoundaryEvent | None:
        frame = int(frame)
        if self.last_frame is None:
            if frame != self.group_start:
                raise ValueError("episode must begin at the configured group start")
        elif frame != self.last_frame + self.detector_stride:
            raise ValueError("control-information observations must be contiguous stride-four")
        self.last_frame = frame
        if frame < 16:
            if information is not None:
                raise ValueError("control-information warmup must be residual-free")
            return None
        if information is None:
            raise ValueError("post-warmup control-information observation is missing")
        standardized = contextual_information_z(
            information, frame=frame, statistics=self.statistics
        )
        self.cusum = max(
            0.0,
            self.decay * self.cusum + max(0.0, standardized - self.drift),
        )
        units = (frame - self.group_start) // self.detector_stride
        if units >= self.max_units:
            event = self._emit(
                frame=frame,
                reason="forced_maximum",
                information=float(information),
                standardized_information=standardized,
            )
            self.forced_maximum_boundary_count += 1
            return event
        if units >= self.min_units and self.cusum >= self.threshold:
            event = self._emit(
                frame=frame,
                reason="counterfactual_control_information",
                information=float(information),
                standardized_information=standardized,
            )
            self.control_information_boundary_count += 1
            return event
        return None

    def finalize(self, *, frame: int) -> ControlInformationBoundaryEvent | None:
        frame = int(frame)
        if frame < self.group_start or (frame - self.group_start) % self.detector_stride:
            raise ValueError("terminal control-information frame is not group-aligned")
        if self.last_frame is not None and frame < self.last_frame:
            raise ValueError("terminal control-information frame precedes the trace")
        if frame == self.group_start:
            return None
        event = self._emit(
            frame=frame,
            reason="terminal_tail",
            information=0.0,
            standardized_information=0.0,
        )
        self.terminal_tail_count += 1
        return event


def replay_control_information_trace(
    *,
    detector_frames: Iterable[int],
    information_by_frame: Mapping[int, float],
    statistics: Mapping[str, Any],
    selector_config: Mapping[str, Any],
) -> list[ControlInformationBoundaryEvent]:
    """Replay the exact online state machine over one immutable scalar trace."""

    frames = [int(frame) for frame in detector_frames]
    if not frames:
        raise ValueError("control-information replay frames are empty")
    state = ControlInformationBoundaryState(
        statistics=statistics, **dict(selector_config)
    )
    events = []
    for frame in frames:
        if frame < 16:
            information = None
        else:
            if frame not in information_by_frame:
                raise KeyError(f"control-information replay lacks frame {frame}")
            information = float(information_by_frame[frame])
        event = state.update(frame=frame, information=information)
        if event is not None:
            events.append(event)
    tail = state.finalize(frame=frames[-1])
    if tail is not None:
        events.append(tail)
    return events
