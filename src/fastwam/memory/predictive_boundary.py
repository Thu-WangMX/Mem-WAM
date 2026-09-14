"""Strict-causal predictive-surprise fusion and dynamic boundary state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch


ROBUST_SCALE = 1.4826


def fit_residual_statistics(
    residuals_by_episode: Mapping[int, torch.Tensor],
    *,
    episodes: Iterable[int],
) -> dict[str, Any]:
    selected = [int(value) for value in episodes]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("residual-stat episodes must be nonempty and unique")
    if any(episode < 0 or episode > 29 for episode in selected):
        raise ValueError("residual statistics may use only episodes 0-29")
    if any(episode not in residuals_by_episode for episode in selected):
        raise KeyError("residual bank is missing a selected episode")
    rows = [
        torch.as_tensor(residuals_by_episode[episode], dtype=torch.float32)
        for episode in selected
    ]
    if any(row.ndim != 2 or row.shape[1] != rows[0].shape[1] for row in rows):
        raise ValueError("residual tensors must share shape [T,streams]")
    values = torch.cat(rows, dim=0)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("residual statistics contain non-finite values")
    median = values.median(dim=0).values
    mad_scale = (values - median).abs().median(dim=0).values * ROBUST_SCALE
    if bool((mad_scale <= 1e-8).any()):
        collapsed = torch.nonzero(mad_scale <= 1e-8).flatten().tolist()
        raise ValueError(f"collapsed zero-MAD residual streams: {collapsed}")
    return {
        "schema_version": "putback_predictive_residual_stats_v1",
        "episodes": selected,
        "median": median,
        "mad_scale": mad_scale,
    }


def robust_residual_z(
    residuals: torch.Tensor, stats: Mapping[str, Any]
) -> torch.Tensor:
    values = torch.as_tensor(residuals, dtype=torch.float32)
    median = torch.as_tensor(stats["median"], dtype=torch.float32)
    scale = torch.as_tensor(stats["mad_scale"], dtype=torch.float32)
    if values.shape[-1:] != median.shape or scale.shape != median.shape:
        raise ValueError("residual/stat stream dimensions differ")
    if bool((scale <= 1e-8).any()):
        raise ValueError("residual scale is collapsed")
    return ((values - median) / scale).clamp_min(0.0)


def _validate_simplex_weights(weights: torch.Tensor, stream_count: int) -> torch.Tensor:
    values = torch.as_tensor(weights, dtype=torch.float32)
    if values.shape != (stream_count,):
        raise ValueError("fusion weight dimension mismatch")
    if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
        raise ValueError("fusion weights must be finite and non-negative")
    if not torch.isclose(values.sum(), torch.tensor(1.0), rtol=0, atol=1e-6):
        raise ValueError("fusion weights must sum to one")
    return values


def select_sparse_simplex_weights(
    event_z_by_episode: Mapping[int, torch.Tensor],
    background_z_by_episode: Mapping[int, torch.Tensor],
    *,
    episodes: Sequence[int],
    top_k: int,
) -> torch.Tensor:
    selected = [int(value) for value in episodes]
    if not selected or any(
        episode not in event_z_by_episode or episode not in background_z_by_episode
        for episode in selected
    ):
        raise ValueError("development episode scores are incomplete")
    event = torch.cat(
        [torch.as_tensor(event_z_by_episode[episode], dtype=torch.float32) for episode in selected]
    )
    background = torch.cat(
        [
            torch.as_tensor(background_z_by_episode[episode], dtype=torch.float32)
            for episode in selected
        ]
    )
    if event.ndim != 2 or background.ndim != 2 or event.shape[1] != background.shape[1]:
        raise ValueError("development scores must have shape [T,streams]")
    top_k = int(top_k)
    if top_k <= 0 or top_k > event.shape[1]:
        raise ValueError("top_k is outside the stream count")
    separation = (
        event.median(dim=0).values - background.median(dim=0).values
    ).clamp_min(0.0)
    indices = torch.topk(separation, k=top_k, largest=True, sorted=True).indices
    selected_separation = separation[indices]
    if float(selected_separation.sum()) <= 1e-8:
        raise ValueError("no development stream separates events from background")
    weights = torch.zeros_like(separation)
    weights[indices] = selected_separation / selected_separation.sum()
    return weights


def fuse_residual_score(
    residuals: torch.Tensor,
    *,
    stats: Mapping[str, Any],
    weights: torch.Tensor,
) -> tuple[float, tuple[int, ...]]:
    z = robust_residual_z(residuals, stats)
    if z.ndim != 1:
        raise ValueError("online residuals must be one-dimensional")
    simplex = _validate_simplex_weights(weights, len(z))
    contribution = z * simplex
    selected = tuple(
        int(index)
        for index in torch.nonzero(contribution > 0, as_tuple=False).flatten().tolist()
    )
    return float(contribution.sum().item()), selected


@dataclass(frozen=True)
class BoundaryEvent:
    frame: int
    peak_frame: int
    reason: str
    group_start: int
    group_end: int
    score: float
    selected_streams: tuple[int, ...]


class PredictiveBoundaryState:
    """Online-only hysteretic selector over interleaved four-phase observations."""

    def __init__(
        self,
        *,
        stats: Mapping[str, Any],
        weights: torch.Tensor,
        high_threshold: float,
        low_threshold: float,
        detector_stride: int = 4,
        min_units: int = 2,
        max_units: int = 8,
        nms_units: int = 2,
        initial_group_start: int | None = None,
    ) -> None:
        stream_count = len(torch.as_tensor(stats["median"]))
        self.stats = {
            "median": torch.as_tensor(stats["median"], dtype=torch.float32).clone(),
            "mad_scale": torch.as_tensor(stats["mad_scale"], dtype=torch.float32).clone(),
        }
        self.weights = _validate_simplex_weights(weights, stream_count).clone()
        self.high_threshold = float(high_threshold)
        self.low_threshold = float(low_threshold)
        self.detector_stride = int(detector_stride)
        self.min_units = int(min_units)
        self.max_units = int(max_units)
        self.nms_units = int(nms_units)
        if not 0 <= self.low_threshold < self.high_threshold:
            raise ValueError("thresholds must satisfy 0 <= low < high")
        if (
            self.detector_stride <= 0
            or self.min_units <= 0
            or self.max_units < self.min_units
            or self.nms_units <= 0
        ):
            raise ValueError("invalid dynamic-boundary unit configuration")
        self.group_start: int | None = (
            None if initial_group_start is None else int(initial_group_start)
        )
        if self.group_start is not None and self.group_start % self.detector_stride:
            raise ValueError("initial group start must be detector-unit aligned")
        self.last_frame: int | None = None
        self.last_boundary_frame: int | None = None
        self.armed = True
        self._candidate: tuple[int, float, tuple[int, ...]] | None = None
        self.retroactive_boundary_count = 0
        self.surprise_boundary_count = 0
        self.forced_maximum_boundary_count = 0
        self.terminal_tail_count = 0

    def _emit(
        self,
        *,
        frame: int,
        peak_frame: int,
        reason: str,
        score: float,
        selected_streams: tuple[int, ...],
    ) -> BoundaryEvent:
        if self.group_start is None or frame <= self.group_start:
            raise RuntimeError("boundary would create an empty or reversed group")
        if self.last_boundary_frame is not None and frame <= self.last_boundary_frame:
            raise RuntimeError("boundaries must be strictly increasing")
        event = BoundaryEvent(
            frame=int(frame),
            peak_frame=int(peak_frame),
            reason=reason,
            group_start=int(self.group_start),
            group_end=int(frame),
            score=float(score),
            selected_streams=tuple(selected_streams),
        )
        self.group_start = int(frame)
        self.last_boundary_frame = int(frame)
        self._candidate = None
        return event

    def update(self, *, frame: int, residuals: torch.Tensor) -> BoundaryEvent | None:
        frame = int(frame)
        if self.last_frame is None:
            if self.group_start is None:
                self.group_start = frame - self.detector_stride
            elif frame <= self.group_start or (frame - self.group_start) % self.detector_stride:
                raise ValueError("first predictive frame is incompatible with episode start")
        elif frame != self.last_frame + self.detector_stride:
            raise ValueError("detector frames must be consecutive fixed-stride observations")
        if self.group_start is None or (frame - self.group_start) % self.detector_stride:
            raise ValueError("detector frame is not unit aligned")
        self.last_frame = frame
        score, selected_streams = fuse_residual_score(
            residuals, stats=self.stats, weights=self.weights
        )

        if self._candidate is not None and frame > self._candidate[0]:
            peak_frame, peak_score, peak_streams = self._candidate
            eligible_nms = self.last_boundary_frame is None or (
                frame - self.last_boundary_frame >= self.nms_units * self.detector_stride
            )
            if eligible_nms:
                event = self._emit(
                    frame=frame,
                    peak_frame=peak_frame,
                    reason="predictive_surprise_confirmed",
                    score=peak_score,
                    selected_streams=peak_streams,
                )
                self.armed = False
                self.surprise_boundary_count += 1
                return event
            self._candidate = None

        group_units = (frame - self.group_start) // self.detector_stride
        if group_units >= self.max_units:
            event = self._emit(
                frame=frame,
                peak_frame=frame,
                reason="forced_maximum",
                score=score,
                selected_streams=selected_streams,
            )
            self.forced_maximum_boundary_count += 1
            return event

        if not self.armed:
            if score <= self.low_threshold:
                self.armed = True
            return None
        if group_units >= self.min_units and score >= self.high_threshold:
            self._candidate = (frame, score, selected_streams)
        return None

    def finalize(self, *, frame: int) -> BoundaryEvent | None:
        frame = int(frame)
        if self.group_start is None:
            return None
        if self.last_frame is not None and frame < self.last_frame:
            raise ValueError("terminal frame precedes the latest observation")
        if (frame - self.group_start) % self.detector_stride:
            raise ValueError("terminal frame is not detector-unit aligned")
        if frame == self.group_start:
            return None
        event = self._emit(
            frame=frame,
            peak_frame=frame,
            reason="terminal_tail",
            score=0.0,
            selected_streams=(),
        )
        self.terminal_tail_count += 1
        return event
