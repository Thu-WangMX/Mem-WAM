"""Strict-causal episode-calibrated control-information boundaries."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch

from fastwam.memory.control_information_boundary import (
    PHASES,
    STATISTICS_SCHEMA,
    _validated_statistics,
    contextual_information_z,
)


def _context_median(frame: int, statistics: Mapping[str, Any]) -> float:
    median, _, depth_cap = _validated_statistics(statistics)
    frame = int(frame)
    phase = frame % 16
    if frame < 16 or phase not in PHASES:
        raise ValueError("adaptation frame must be >=16 and stride-four aligned")
    depth = min((frame - phase) // 16, int(depth_cap))
    return float(median[PHASES.index(phase), depth - 1])


class CausalEpisodeLogBiasAdapter:
    """Freeze one robust multiplicative score correction from an online prefix."""

    def __init__(
        self,
        *,
        statistics: Mapping[str, Any],
        calibration_samples: int = 4,
        epsilon: float = 1e-6,
        max_abs_log_bias: float = math.log(64.0),
    ) -> None:
        _validated_statistics(statistics)
        self.statistics = dict(statistics)
        self.calibration_samples = int(calibration_samples)
        self.epsilon = float(epsilon)
        self.max_abs_log_bias = float(max_abs_log_bias)
        if self.calibration_samples <= 0:
            raise ValueError("calibration_samples must be positive")
        if self.epsilon <= 0 or not math.isfinite(self.epsilon):
            raise ValueError("adaptation epsilon must be finite and positive")
        if self.max_abs_log_bias <= 0 or not math.isfinite(self.max_abs_log_bias):
            raise ValueError("max_abs_log_bias must be finite and positive")
        self._log_ratios: list[float] = []
        self._calibration_frames: list[int] = []
        self._last_frame: int | None = None
        self._log_bias: float | None = None

    @property
    def factor(self) -> float | None:
        return None if self._log_bias is None else math.exp(self._log_bias)

    @property
    def calibration_frames(self) -> tuple[int, ...]:
        return tuple(self._calibration_frames)

    def observe(self, *, frame: int, information: float) -> float | None:
        frame = int(frame)
        value = float(information)
        if self._last_frame is None:
            if frame != 16:
                raise ValueError("episode adaptation must begin at frame 16")
        elif frame != self._last_frame + 4:
            raise ValueError("episode adaptation observations must be contiguous stride-four")
        if value < 0 or not math.isfinite(value):
            raise ValueError("control information must be finite and non-negative")
        self._last_frame = frame

        if self._log_bias is None:
            reference = _context_median(frame, self.statistics)
            if reference < 0 or not math.isfinite(reference):
                raise ValueError("contextual information median must be finite and non-negative")
            self._log_ratios.append(
                math.log((value + self.epsilon) / (reference + self.epsilon))
            )
            self._calibration_frames.append(frame)
            if len(self._log_ratios) == self.calibration_samples:
                bias = float(torch.quantile(torch.tensor(self._log_ratios), 0.5))
                if abs(bias) > self.max_abs_log_bias:
                    raise ValueError(
                        "episode score adaptation is outside the locked bound: "
                        f"abs(log_bias)={abs(bias):.6f} > {self.max_abs_log_bias:.6f}"
                    )
                self._log_bias = bias
            return None

        factor = self.factor
        assert factor is not None
        return max(0.0, (value + self.epsilon) / factor - self.epsilon)


@dataclass(frozen=True)
class ControlInformationBoundaryEventV2:
    confirmation_frame: int
    group_start: int
    group_end: int
    reason: str
    information: float
    adapted_information: float
    adaptation_factor: float
    standardized_information: float
    cusum: float


class ControlInformationBoundaryStateV2:
    """Close segments from adapted information without future or backdating."""

    def __init__(
        self,
        *,
        statistics: Mapping[str, Any],
        threshold: float,
        drift: float,
        decay: float,
        detector_stride: int = 4,
        min_units: int = 8,
        max_units: int = 24,
        initial_group_start: int = 0,
        calibration_samples: int = 4,
        adaptation_epsilon: float = 1e-6,
        max_abs_log_bias: float = math.log(64.0),
    ) -> None:
        if statistics.get("schema_version") != STATISTICS_SCHEMA:
            raise ValueError("control-information statistics schema is incompatible")
        _validated_statistics(statistics)
        self.statistics = dict(statistics)
        self.threshold = float(threshold)
        self.drift = float(drift)
        self.decay = float(decay)
        self.detector_stride = int(detector_stride)
        self.min_units = int(min_units)
        self.max_units = int(max_units)
        self.group_start = int(initial_group_start)
        self.calibration_samples = int(calibration_samples)
        if self.threshold <= 0 or self.drift < 0 or not 0 < self.decay <= 1:
            raise ValueError("CUSUM threshold/drift/decay configuration is invalid")
        if (
            self.detector_stride != 4
            or self.min_units < 4 + self.calibration_samples
            or self.max_units < self.min_units
            or self.group_start != 0
        ):
            raise ValueError("episode-calibrated detector/group configuration is invalid")
        self.adapter = CausalEpisodeLogBiasAdapter(
            statistics=self.statistics,
            calibration_samples=self.calibration_samples,
            epsilon=adaptation_epsilon,
            max_abs_log_bias=max_abs_log_bias,
        )
        self.last_frame: int | None = None
        self.cusum = 0.0
        self.retroactive_boundary_count = 0
        self.control_information_boundary_count = 0
        self.forced_maximum_boundary_count = 0
        self.terminal_tail_count = 0

    @property
    def adaptation_factor(self) -> float | None:
        return self.adapter.factor

    def _emit(
        self,
        *,
        frame: int,
        reason: str,
        information: float,
        adapted_information: float,
        standardized_information: float,
    ) -> ControlInformationBoundaryEventV2:
        factor = self.adaptation_factor
        if factor is None:
            raise RuntimeError("cannot emit before episode adaptation is frozen")
        if int(frame) <= self.group_start:
            raise RuntimeError("control-information boundary creates an empty group")
        event = ControlInformationBoundaryEventV2(
            confirmation_frame=int(frame),
            group_start=self.group_start,
            group_end=int(frame),
            reason=str(reason),
            information=float(information),
            adapted_information=float(adapted_information),
            adaptation_factor=float(factor),
            standardized_information=float(standardized_information),
            cusum=float(self.cusum),
        )
        self.group_start = int(frame)
        self.cusum = 0.0
        return event

    def update(
        self, *, frame: int, information: float | None
    ) -> ControlInformationBoundaryEventV2 | None:
        frame = int(frame)
        if self.last_frame is None:
            if frame != 0:
                raise ValueError("episode must begin at frame zero")
        elif frame != self.last_frame + self.detector_stride:
            raise ValueError("control-information observations must be contiguous stride-four")
        self.last_frame = frame
        if frame < 16:
            if information is not None:
                raise ValueError("control-information warmup must be residual-free")
            return None
        if information is None:
            raise ValueError("post-warmup control-information observation is missing")

        adapted = self.adapter.observe(frame=frame, information=float(information))
        if adapted is None:
            self.cusum = 0.0
            return None
        standardized = contextual_information_z(
            adapted, frame=frame, statistics=self.statistics
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
                adapted_information=adapted,
                standardized_information=standardized,
            )
            self.forced_maximum_boundary_count += 1
            return event
        if units >= self.min_units and self.cusum >= self.threshold:
            event = self._emit(
                frame=frame,
                reason="counterfactual_control_information",
                information=float(information),
                adapted_information=adapted,
                standardized_information=standardized,
            )
            self.control_information_boundary_count += 1
            return event
        return None

    def finalize(self, *, frame: int) -> ControlInformationBoundaryEventV2 | None:
        frame = int(frame)
        if frame < self.group_start or (frame - self.group_start) % 4:
            raise ValueError("terminal control-information frame is not group-aligned")
        if frame == self.group_start:
            return None
        event = self._emit(
            frame=frame,
            reason="terminal_tail",
            information=0.0,
            adapted_information=0.0,
            standardized_information=0.0,
        )
        self.terminal_tail_count += 1
        return event


def replay_control_information_trace_v2(
    *,
    detector_frames: Iterable[int],
    information_by_frame: Mapping[int, float],
    statistics: Mapping[str, Any],
    selector_config: Mapping[str, Any],
) -> list[ControlInformationBoundaryEventV2]:
    """Replay the exact v2 online state over one immutable scalar trace."""

    frames = [int(frame) for frame in detector_frames]
    if not frames:
        raise ValueError("v2 control-information replay frames are empty")
    state = ControlInformationBoundaryStateV2(
        statistics=statistics, **dict(selector_config)
    )
    events = []
    for frame in frames:
        if frame < 16:
            information = None
        else:
            if frame not in information_by_frame:
                raise KeyError(f"v2 control-information replay lacks frame {frame}")
            information = float(information_by_frame[frame])
        event = state.update(frame=frame, information=information)
        if event is not None:
            events.append(event)
    tail = state.finalize(frame=frames[-1])
    if tail is not None:
        events.append(tail)
    return events
