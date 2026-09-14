"""Validated, phase-pinned dynamic-surprise segment manifests."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F


SCHEMA_VERSION = "putback_dynamic_surprise_segments_v1"
MIN_SEGMENT_LENGTH = 2
MAX_SEGMENT_LENGTH = 8


@dataclass(frozen=True)
class OnlineBoundaryDecision:
    """One causal transition decision, safe to preview before model commit."""

    arrival: int
    close_range: tuple[int, int] | None
    score: float
    threshold: float | None
    reason: str | None


@dataclass(frozen=True)
class ManifestSegment:
    """One offline segment and the reason recorded at its closing boundary."""

    start: int
    end: int
    reason: str


class OnlineSurpriseSegmenter:
    """Incremental form of :func:`causal_boundaries` with atomic preview/commit."""

    def __init__(
        self,
        *,
        min_segment: int = MIN_SEGMENT_LENGTH,
        max_segment: int = MAX_SEGMENT_LENGTH,
        gamma: float = 1.5,
        window: int = 5,
    ) -> None:
        if min_segment < 1 or max_segment < min_segment or window < 1:
            raise ValueError("invalid segment/window lengths")
        self.min_segment = int(min_segment)
        self.max_segment = int(max_segment)
        self.gamma = float(gamma)
        self.window = int(window)
        self._scores: list[float] = []
        self._open_start = 0

    @property
    def scores(self) -> tuple[float, ...]:
        return tuple(self._scores)

    @property
    def open_start(self) -> int:
        return self._open_start

    @property
    def next_arrival(self) -> int:
        return len(self._scores) + 1

    def preview(self, score: float) -> OnlineBoundaryDecision:
        value = float(score)
        arrival = self.next_arrival
        prior = self._scores[max(0, len(self._scores) - self.window) :]
        threshold = None
        if len(prior) >= min(3, self.window):
            mean = sum(prior) / len(prior)
            variance = sum((item - mean) ** 2 for item in prior) / len(prior)
            threshold = mean + self.gamma * math.sqrt(variance)
        segment_length = arrival - self._open_start
        reason = None
        if (
            segment_length >= self.min_segment
            and threshold is not None
            and value > threshold
        ):
            reason = "surprise"
        elif segment_length >= self.max_segment:
            reason = "max_length"
        close_range = (
            (self._open_start, arrival) if reason is not None else None
        )
        return OnlineBoundaryDecision(
            arrival=arrival,
            close_range=close_range,
            score=value,
            threshold=threshold,
            reason=reason,
        )

    def commit(self, decision: OnlineBoundaryDecision) -> None:
        expected = self.preview(decision.score)
        if decision != expected:
            raise ValueError("decision does not match current preview")
        self._scores.append(decision.score)
        if decision.close_range is not None:
            self._open_start = decision.arrival


def recover_clean_latent(
    noisy_latent: torch.Tensor,
    predicted_velocity: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    sigma_tensor = torch.as_tensor(
        sigma, device=noisy_latent.device, dtype=noisy_latent.dtype
    )
    return noisy_latent - sigma_tensor * predicted_velocity


def latent_surprise(
    predicted: torch.Tensor,
    actual: torch.Tensor,
    *,
    l1_weight: float = 0.7,
    eps: float = 1.0e-6,
) -> dict[str, float]:
    predicted_flat = predicted.detach().float().reshape(-1)
    actual_flat = actual.detach().float().reshape(-1)
    if predicted_flat.shape != actual_flat.shape:
        raise ValueError(
            f"latent shapes differ: {tuple(predicted.shape)} != {tuple(actual.shape)}"
        )
    normalized_l1 = float(
        (
            (predicted_flat - actual_flat).abs().mean()
            / actual_flat.abs().mean().clamp_min(eps)
        ).item()
    )
    cosine_distance = float(
        (1.0 - F.cosine_similarity(predicted_flat, actual_flat, dim=0, eps=eps)).item()
    )
    return {
        "normalized_l1": normalized_l1,
        "cosine_distance": cosine_distance,
        "score": float(l1_weight) * normalized_l1
        + (1.0 - float(l1_weight)) * cosine_distance,
    }


def causal_boundaries(
    scores: Sequence[float],
    *,
    min_segment: int = MIN_SEGMENT_LENGTH,
    max_segment: int = MAX_SEGMENT_LENGTH,
    gamma: float = 1.5,
    window: int = 5,
) -> dict[str, Any]:
    """Causally segment scores using a rolling threshold that never resets."""
    if min_segment < 1 or max_segment < min_segment or window < 1:
        raise ValueError("invalid segment/window lengths")
    values = [float(value) for value in scores]
    boundaries = [0]
    reasons = {"0": "start"}
    thresholds: list[float | None] = []
    last_boundary = 0
    for offset, score in enumerate(values):
        arrival = offset + 1
        prior = values[max(0, offset - window) : offset]
        threshold = None
        if len(prior) >= min(3, window):
            mean = sum(prior) / len(prior)
            variance = sum((value - mean) ** 2 for value in prior) / len(prior)
            threshold = mean + float(gamma) * math.sqrt(variance)
        thresholds.append(threshold)
        length = arrival - last_boundary
        reason = None
        if length >= min_segment and threshold is not None and score > threshold:
            reason = "surprise"
        elif length >= max_segment:
            reason = "max_length"
        if reason is not None:
            boundaries.append(arrival)
            reasons[str(arrival)] = reason
            last_boundary = arrival
    observation_count = len(values) + 1
    if boundaries[-1] != observation_count:
        boundaries.append(observation_count)
        reasons[str(observation_count)] = "end"
    return {
        "boundaries": boundaries,
        "reasons": reasons,
        "thresholds": thresholds,
        "segment_lengths": [
            stop - start for start, stop in zip(boundaries, boundaries[1:])
        ],
    }


def _require_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return value


def validate_episode_segments(
    payload: Mapping[str, Any], decision_count: int | None = None
) -> tuple[tuple[int, int], ...]:
    """Validate and return a complete half-open partition of one episode.

    A final one-frame segment is permitted because an episode can naturally end
    one decision after a boundary. Interior one-frame segments are forbidden.
    """

    recorded_count = _require_integer(payload.get("decision_count"), "decision_count")
    if decision_count is None:
        decision_count = recorded_count
    else:
        decision_count = _require_integer(decision_count, "decision_count")
        if recorded_count != decision_count:
            raise ValueError(
                f"episode decision count {recorded_count} does not match expected "
                f"decision count {decision_count}"
            )
    if decision_count < 1:
        raise ValueError("decision count must be positive")

    raw_boundaries = payload.get("boundaries")
    if not isinstance(raw_boundaries, Sequence) or isinstance(raw_boundaries, (str, bytes)):
        raise ValueError("boundaries must be a sequence of integers")
    boundaries = tuple(_require_integer(item, "boundary") for item in raw_boundaries)
    if len(boundaries) < 2:
        raise ValueError("boundaries must contain at least start and end")
    if boundaries[0] != 0:
        raise ValueError("boundaries must start at zero")
    if any(right <= left for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError("boundaries must be strictly increasing")
    if boundaries[-1] != decision_count:
        raise ValueError(
            f"final boundary {boundaries[-1]} does not match decision count {decision_count}"
        )

    reasons = payload.get("reasons")
    if not isinstance(reasons, Mapping):
        raise ValueError("reasons must be a mapping")
    if reasons.get("0") != "start":
        raise ValueError("boundary zero must have reason 'start'")
    if reasons.get(str(decision_count)) != "end":
        raise ValueError("final boundary must have reason 'end'")

    segments = tuple(zip(boundaries, boundaries[1:]))
    for index, (start, end) in enumerate(segments):
        length = end - start
        is_terminal_one_frame_tail = index == len(segments) - 1 and length == 1
        if length > MAX_SEGMENT_LENGTH or (
            length < MIN_SEGMENT_LENGTH and not is_terminal_one_frame_tail
        ):
            raise ValueError(
                f"segment [{start}, {end}) has invalid length {length}; expected "
                f"{MIN_SEGMENT_LENGTH}..{MAX_SEGMENT_LENGTH}, except a terminal length-1 tail"
            )
    return segments


def closed_segments_for_history(
    segments: Sequence[tuple[int, int]], history_frames: int
) -> tuple[tuple[int, ...], ...]:
    """Return segments closed before the current observation.

    ``history_frames`` includes the current decision observation. A boundary at
    decision ``t`` closes ``[start, t)`` and leaves observation ``t`` raw.
    """

    history_frames = _require_integer(history_frames, "history_frames")
    if history_frames < 1:
        raise ValueError("history_frames must be positive")
    current_decision = history_frames - 1
    return tuple(
        tuple(range(start, end))
        for start, end in segments
        if end <= current_decision
    )


class DynamicSurpriseManifestStore:
    """Read-only manifest accessor that enforces train/phase compatibility."""

    def __init__(
        self,
        root: str | Path,
        *,
        expected_boundary_step: int,
        expected_episode_count: int | None = None,
        expected_task: str = "putback",
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        manifest_path = self.root / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"missing surprise manifest: {manifest_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in surprise manifest {manifest_path}: {exc}") from exc

        metadata = manifest.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("manifest metadata must be a mapping")
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {metadata.get('schema_version')!r}; "
                f"expected {SCHEMA_VERSION!r}"
            )
        if metadata.get("complete") is not True:
            raise ValueError("surprise manifest is not marked complete")
        expected_task = str(expected_task)
        if expected_task not in {"putback", "battery_try"}:
            raise ValueError(
                "expected_task must be one of {'putback', 'battery_try'}"
            )
        if metadata.get("task") != expected_task:
            raise ValueError(
                f"surprise manifest task must be {expected_task!r}, "
                f"got {metadata.get('task')!r}"
            )
        if metadata.get("replan_stride") != 16:
            raise ValueError("surprise manifest replan_stride must be 16")
        boundary_step = _require_integer(metadata.get("boundary_step"), "boundary_step")
        if boundary_step != expected_boundary_step:
            raise ValueError(
                f"manifest boundary_step {boundary_step} does not match expected "
                f"boundary_step {expected_boundary_step}"
            )
        episode_count = _require_integer(metadata.get("episode_count"), "episode_count")
        if expected_episode_count is not None and episode_count != expected_episode_count:
            raise ValueError(
                f"manifest episode_count {episode_count} does not match expected "
                f"episode_count {expected_episode_count}"
            )
        episodes = manifest.get("episodes")
        if not isinstance(episodes, Mapping) or len(episodes) != episode_count:
            raise ValueError("manifest episodes mapping does not match episode_count")

        self.metadata = dict(metadata)
        self._episodes = dict(episodes)
        self._cache: dict[int, tuple[tuple[int, int], ...]] = {}
        self._reason_cache: dict[int, dict[int, str]] = {}

    def segments_for_episode(self, episode_index: int) -> tuple[tuple[int, int], ...]:
        episode_index = _require_integer(episode_index, "episode_index")
        if episode_index in self._cache:
            return self._cache[episode_index]
        relative_path = self._episodes.get(str(episode_index))
        if not isinstance(relative_path, str):
            raise KeyError(f"episode {episode_index} is absent from surprise manifest")
        episode_path = (self.root / relative_path).resolve()
        if self.root not in episode_path.parents:
            raise ValueError(f"episode path escapes manifest root: {relative_path}")
        try:
            payload = json.loads(episode_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"missing episode manifest: {episode_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in episode manifest {episode_path}: {exc}") from exc
        if payload.get("episode") != episode_index:
            raise ValueError(
                f"episode file {episode_path} records episode {payload.get('episode')!r}, "
                f"expected {episode_index}"
            )
        segments = validate_episode_segments(payload)
        raw_reasons = payload["reasons"]
        self._reason_cache[episode_index] = {
            int(key): str(value) for key, value in raw_reasons.items()
        }
        self._cache[episode_index] = segments
        return segments

    def segment_records_for_episode(
        self, episode_index: int
    ) -> tuple[ManifestSegment, ...]:
        """Return segments with their causal closing reasons intact."""

        segments = self.segments_for_episode(episode_index)
        reasons = self._reason_cache.get(int(episode_index))
        if reasons is None:
            raise RuntimeError("manifest reason cache was not populated")
        records = []
        for start, end in segments:
            reason = reasons.get(end)
            if reason is None:
                raise ValueError(f"segment [{start}, {end}) has no closing reason")
            records.append(ManifestSegment(start=start, end=end, reason=reason))
        return tuple(records)
