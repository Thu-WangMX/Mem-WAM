"""Strict-causal segment-relative counterfactual control information."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch

from fastwam.memory.control_information_boundary import (
    STATISTICS_SCHEMA,
    _validated_statistics,
    contextual_information_z,
)
from fastwam.memory.control_information_boundary_v2 import (
    CausalEpisodeLogBiasAdapter,
)


class CausalSegmentRobustBaseline:
    """Freeze a robust local location/scale from the start of one segment."""

    def __init__(self, *, calibration_samples: int, scale_floor: float) -> None:
        self.calibration_samples = int(calibration_samples)
        self.scale_floor = float(scale_floor)
        if self.calibration_samples <= 0:
            raise ValueError("segment calibration_samples must be positive")
        if self.scale_floor <= 0 or not math.isfinite(self.scale_floor):
            raise ValueError("segment scale_floor must be finite and positive")
        self.reset()

    def reset(self) -> None:
        self._values: list[float] = []
        self.location: float | None = None
        self.scale: float | None = None

    @property
    def ready(self) -> bool:
        return self.location is not None and self.scale is not None

    @property
    def calibration_values(self) -> tuple[float, ...]:
        return tuple(self._values)

    def observe(self, value: float) -> bool:
        if self.ready:
            raise RuntimeError("segment baseline is already frozen")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("segment baseline value must be finite")
        self._values.append(value)
        if len(self._values) > self.calibration_samples:
            raise RuntimeError("segment baseline received too many calibration samples")
        if len(self._values) == self.calibration_samples:
            tensor = torch.tensor(self._values, dtype=torch.float32)
            location = float(torch.quantile(tensor, 0.5))
            mad = float(torch.quantile((tensor - location).abs(), 0.5))
            self.location = location
            self.scale = max(1.4826 * mad, self.scale_floor)
        return self.ready

    def standardized_innovation(self, value: float) -> float:
        if not self.ready:
            raise RuntimeError("segment baseline is not frozen")
        assert self.location is not None and self.scale is not None
        return (float(value) - self.location) / self.scale


@dataclass(frozen=True)
class ControlInformationBoundaryEventV3:
    confirmation_frame: int
    group_start: int
    group_end: int
    reason: str
    information: float
    adapted_information: float
    episode_adaptation_factor: float
    contextual_standardized_information: float
    segment_baseline_location: float
    segment_baseline_scale: float
    segment_relative_innovation: float
    consecutive_evidence: int
    cusum: float


class ControlInformationBoundaryStateV3:
    """Detect changes relative to a causally frozen baseline for each segment."""

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
        episode_calibration_samples: int = 4,
        segment_calibration_samples: int = 4,
        adaptation_epsilon: float = 1e-6,
        max_abs_log_bias: float = math.log(64.0),
        segment_scale_floor: float = 1.0,
        minimum_consecutive_evidence: int = 2,
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
        self.episode_calibration_samples = int(episode_calibration_samples)
        self.segment_calibration_samples = int(segment_calibration_samples)
        self.adaptation_epsilon = float(adaptation_epsilon)
        self.minimum_consecutive_evidence = int(minimum_consecutive_evidence)
        if self.threshold <= 0 or self.drift < 0 or not 0 < self.decay <= 1:
            raise ValueError("segment-relative CUSUM configuration is invalid")
        if (
            self.detector_stride != 4
            or self.episode_calibration_samples != 4
            or self.segment_calibration_samples <= 0
            or self.min_units < self.episode_calibration_samples + 4
            or self.max_units < self.min_units
            or self.group_start != 0
            or self.minimum_consecutive_evidence <= 0
        ):
            raise ValueError("segment-relative detector/group configuration is invalid")
        self.episode_adapter = CausalEpisodeLogBiasAdapter(
            statistics=self.statistics,
            calibration_samples=self.episode_calibration_samples,
            epsilon=self.adaptation_epsilon,
            max_abs_log_bias=float(max_abs_log_bias),
        )
        self.segment_baseline = CausalSegmentRobustBaseline(
            calibration_samples=self.segment_calibration_samples,
            scale_floor=float(segment_scale_floor),
        )
        self._episode_calibration_rows: list[tuple[int, float]] = []
        self.last_frame: int | None = None
        self.cusum = 0.0
        self.consecutive_evidence = 0
        self.retroactive_boundary_count = 0
        self.segment_relative_boundary_count = 0
        self.forced_maximum_boundary_count = 0
        self.terminal_tail_count = 0
        self.last_adapted_information: float | None = None
        self.last_contextual_standardized_information: float | None = None
        self.last_segment_relative_innovation: float | None = None

    @property
    def episode_adaptation_factor(self) -> float | None:
        return self.episode_adapter.factor

    def _adapt(self, information: float) -> float:
        factor = self.episode_adaptation_factor
        if factor is None:
            raise RuntimeError("episode adaptation factor is not frozen")
        return max(
            0.0,
            (float(information) + self.adaptation_epsilon) / factor
            - self.adaptation_epsilon,
        )

    def _reset_segment_detector(self) -> None:
        self.segment_baseline.reset()
        self.cusum = 0.0
        self.consecutive_evidence = 0

    def _emit(
        self,
        *,
        frame: int,
        reason: str,
        information: float,
        adapted_information: float,
        contextual_z: float,
        innovation: float,
    ) -> ControlInformationBoundaryEventV3:
        factor = self.episode_adaptation_factor
        if factor is None:
            raise RuntimeError("cannot emit before episode adaptation is frozen")
        if int(frame) <= self.group_start:
            raise RuntimeError("segment-relative boundary creates an empty group")
        location = self.segment_baseline.location
        scale = self.segment_baseline.scale
        event = ControlInformationBoundaryEventV3(
            confirmation_frame=int(frame),
            group_start=self.group_start,
            group_end=int(frame),
            reason=str(reason),
            information=float(information),
            adapted_information=float(adapted_information),
            episode_adaptation_factor=float(factor),
            contextual_standardized_information=float(contextual_z),
            segment_baseline_location=0.0 if location is None else float(location),
            segment_baseline_scale=1.0 if scale is None else float(scale),
            segment_relative_innovation=float(innovation),
            consecutive_evidence=int(self.consecutive_evidence),
            cusum=float(self.cusum),
        )
        self.group_start = int(frame)
        self._reset_segment_detector()
        return event

    def update(
        self, *, frame: int, information: float | None
    ) -> ControlInformationBoundaryEventV3 | None:
        self.last_adapted_information = None
        self.last_contextual_standardized_information = None
        self.last_segment_relative_innovation = None
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
        raw = float(information)
        if self.episode_adaptation_factor is None:
            self._episode_calibration_rows.append((frame, raw))
        adapted = self.episode_adapter.observe(frame=frame, information=raw)
        if adapted is None:
            if self.episode_adaptation_factor is not None:
                for calibration_frame, calibration_raw in self._episode_calibration_rows:
                    calibration_adapted = self._adapt(calibration_raw)
                    self.segment_baseline.observe(
                        contextual_information_z(
                            calibration_adapted,
                            frame=calibration_frame,
                            statistics=self.statistics,
                        )
                    )
                if not self.segment_baseline.ready:
                    raise RuntimeError("initial segment baseline did not freeze")
            return None

        contextual_z = contextual_information_z(
            adapted, frame=frame, statistics=self.statistics
        )
        self.last_adapted_information = float(adapted)
        self.last_contextual_standardized_information = float(contextual_z)
        if not self.segment_baseline.ready:
            self.segment_baseline.observe(contextual_z)
            self.cusum = 0.0
            self.consecutive_evidence = 0
            return None

        innovation = self.segment_baseline.standardized_innovation(contextual_z)
        self.last_segment_relative_innovation = float(innovation)
        positive = max(0.0, innovation - self.drift)
        self.consecutive_evidence = (
            self.consecutive_evidence + 1 if positive > 0 else 0
        )
        self.cusum = max(0.0, self.decay * self.cusum + positive)
        units = (frame - self.group_start) // self.detector_stride
        if units >= self.max_units:
            event = self._emit(
                frame=frame,
                reason="forced_maximum",
                information=raw,
                adapted_information=adapted,
                contextual_z=contextual_z,
                innovation=innovation,
            )
            self.forced_maximum_boundary_count += 1
            return event
        if (
            units >= self.min_units
            and self.consecutive_evidence >= self.minimum_consecutive_evidence
            and self.cusum >= self.threshold
        ):
            event = self._emit(
                frame=frame,
                reason="segment_relative_control_information",
                information=raw,
                adapted_information=adapted,
                contextual_z=contextual_z,
                innovation=innovation,
            )
            self.segment_relative_boundary_count += 1
            return event
        return None

    def finalize(self, *, frame: int) -> ControlInformationBoundaryEventV3 | None:
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
            contextual_z=0.0,
            innovation=0.0,
        )
        self.terminal_tail_count += 1
        return event


def replay_control_information_trace_v3(
    *,
    detector_frames: Iterable[int],
    information_by_frame: Mapping[int, float],
    statistics: Mapping[str, Any],
    selector_config: Mapping[str, Any],
) -> list[ControlInformationBoundaryEventV3]:
    frames = [int(frame) for frame in detector_frames]
    if not frames:
        raise ValueError("v3 replay requires detector frames")
    state = ControlInformationBoundaryStateV3(
        statistics=statistics, **dict(selector_config)
    )
    events = []
    for frame in frames:
        event = state.update(
            frame=frame,
            information=None if frame < 16 else float(information_by_frame[frame]),
        )
        if event is not None:
            events.append(event)
    tail = state.finalize(frame=frames[-1])
    if tail is not None:
        events.append(tail)
    return events
