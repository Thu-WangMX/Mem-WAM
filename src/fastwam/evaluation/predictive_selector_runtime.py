"""One-job-at-a-time online four-phase predictive selector scheduler."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch

from fastwam.memory.predictive_boundary import BoundaryEvent, PredictiveBoundaryState


@dataclass(frozen=True)
class DetectorJob:
    frame: int
    phase: int
    submitted_at: float


@dataclass(frozen=True)
class RuntimeBoundaryEvent:
    frame: int
    emitted_at_frame: int
    peak_frame: int
    reason: str
    group_start: int
    group_end: int
    score: float
    selected_streams: tuple[int, ...]


class PredictiveSelectorRuntime:
    def __init__(self, *, selector_config: dict[str, Any], clock=time.perf_counter):
        config = selector_config
        self.state = PredictiveBoundaryState(
            stats={"median": torch.tensor(config["stats"]["median"]),
                   "mad_scale": torch.tensor(config["stats"]["mad_scale"])},
            weights=torch.tensor(config["weights"]),
            high_threshold=config["high_threshold"], low_threshold=config["low_threshold"],
            detector_stride=config["detector_stride"], min_units=config["min_units"],
            max_units=config["max_units"], nms_units=config["nms_units"],
            initial_group_start=0,
        )
        self.clock = clock
        self.pending: DetectorJob | None = None
        self.last_arrival: int | None = None
        self.maximum_queue_depth = 0
        self.records = []

    @property
    def retroactive_boundary_count(self):
        return self.state.retroactive_boundary_count

    def observe(self, *, frame: int, observation: torch.Tensor) -> DetectorJob:
        frame = int(frame)
        if self.pending is not None:
            raise RuntimeError("detector job is still pending at the next arrival")
        if self.last_arrival is not None and frame != self.last_arrival + 4:
            raise ValueError("four-phase observations must arrive every four frames")
        if self.last_arrival is None and frame != 0:
            raise ValueError("online runtime must begin at frame zero")
        self.last_arrival = frame
        self.pending = DetectorJob(frame=frame, phase=frame % 16, submitted_at=self.clock())
        self.maximum_queue_depth = max(self.maximum_queue_depth, 1)
        return self.pending

    def complete(self, *, frame: int, residual: torch.Tensor | None):
        frame = int(frame)
        if self.pending is None:
            raise RuntimeError("no pending detector job")
        if frame != self.pending.frame:
            raise RuntimeError("detector completion frame differs from its actual arrival")
        job = self.pending
        self.pending = None
        event: BoundaryEvent | None = None
        if frame < 16:
            if residual is not None:
                raise ValueError("four-phase warmup anchors have no predictive residual")
        else:
            if residual is None:
                raise ValueError("non-warmup detector completion requires residuals")
            event = self.state.update(frame=frame, residuals=residual)
        self.records.append({
            "phase": job.phase, "input_frame": frame, "completion_frame": frame,
            "wall_time_seconds": self.clock() - job.submitted_at, "queue_depth": 1,
            "emitted_boundary": None if event is None else event.frame,
        })
        if event is None:
            return None
        return RuntimeBoundaryEvent(
            frame=event.frame, emitted_at_frame=frame, peak_frame=event.peak_frame,
            reason=event.reason, group_start=event.group_start, group_end=event.group_end,
            score=event.score, selected_streams=event.selected_streams,
        )
