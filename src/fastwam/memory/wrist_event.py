"""Causal online state for a frozen wrist-latent event forecaster."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import torch


WRIST_EVENT_SCHEMA_VERSION = "putback_wrist_latent_event_segments_v1"


@dataclass(frozen=True)
class WristEventArrivalDecision:
    arrival: int
    close_range: tuple[int, int] | None
    probability: float | None
    reason: str | None


class OnlineWristEventSegmenter:
    """Apply a one-decision-ahead forecast without reading future scores."""

    def __init__(
        self,
        *,
        threshold: float,
        min_segment: int = 2,
        max_segment: int = 8,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must lie in [0,1]")
        if min_segment < 1 or max_segment < min_segment:
            raise ValueError("invalid segment lengths")
        self.threshold = float(threshold)
        self.min_segment = int(min_segment)
        self.max_segment = int(max_segment)
        self._next_arrival = 0
        self._open_start = 0
        self._pending_probability: float | None = None

    @property
    def next_arrival(self) -> int:
        return self._next_arrival

    @property
    def open_start(self) -> int:
        return self._open_start

    def preview_arrival(self) -> WristEventArrivalDecision:
        arrival = self._next_arrival
        length = arrival - self._open_start
        reason = None
        if (
            self._pending_probability is not None
            and self._pending_probability >= self.threshold
            and length >= self.min_segment
        ):
            reason = "wrist_event"
        elif length >= self.max_segment:
            reason = "max_length"
        return WristEventArrivalDecision(
            arrival=arrival,
            close_range=(self._open_start, arrival) if reason is not None else None,
            probability=self._pending_probability,
            reason=reason,
        )

    def commit_arrival(self, decision: WristEventArrivalDecision) -> None:
        if decision != self.preview_arrival():
            raise ValueError("arrival decision does not match current preview")
        if decision.close_range is not None:
            self._open_start = decision.arrival
        self._pending_probability = None
        self._next_arrival += 1

    def schedule_next(self, probability: float) -> None:
        if self._pending_probability is not None:
            raise RuntimeError("next arrival already has a scheduled probability")
        value = float(probability)
        if not 0.0 <= value <= 1.0:
            raise ValueError("probability must lie in [0,1]")
        self._pending_probability = value


def causal_latent_window(
    latents: Sequence[torch.Tensor],
    *,
    history: int,
) -> torch.Tensor:
    """Return `[1,H,48,24,20]`, left-padding only with the first latent."""

    if history <= 0:
        raise ValueError("history must be positive")
    if not latents:
        raise ValueError("at least one latent is required")
    normalized = []
    for latent in latents:
        if tuple(latent.shape) != (1, 48, 1, 24, 20):
            raise ValueError(
                "each latent must have shape [1,48,1,24,20], "
                f"got {tuple(latent.shape)}"
            )
        normalized.append(latent[:, :, 0])
    recent = normalized[-history:]
    recent = [recent[0]] * (history - len(recent)) + recent
    return torch.stack(recent, dim=1)


class WristEventManifestStore:
    """Read-only validated accessor for a frozen predictor's PutBack segments."""

    def __init__(
        self,
        root: str | Path,
        *,
        expected_episode_count: int | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        manifest_path = self.root / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"missing wrist-event manifest: {manifest_path}") from exc
        metadata = manifest.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("manifest metadata must be a mapping")
        if metadata.get("schema_version") != WRIST_EVENT_SCHEMA_VERSION:
            raise ValueError("unsupported wrist-event manifest schema")
        if metadata.get("complete") is not True:
            raise ValueError("wrist-event manifest is not complete")
        if metadata.get("task") != "put_back_block":
            raise ValueError("wrist-event manifest task must be put_back_block")
        if metadata.get("replan_stride") != 16:
            raise ValueError("wrist-event manifest replan_stride must be 16")
        if metadata.get("uses_vlm") is not False:
            raise ValueError("wrist-event manifest must explicitly declare uses_vlm=false")
        episode_count = metadata.get("episode_count")
        if isinstance(episode_count, bool) or not isinstance(episode_count, int):
            raise ValueError("episode_count must be an integer")
        if expected_episode_count is not None and episode_count != expected_episode_count:
            raise ValueError(
                f"episode_count {episode_count} does not match {expected_episode_count}"
            )
        episodes = manifest.get("episodes")
        if not isinstance(episodes, Mapping) or len(episodes) != episode_count:
            raise ValueError("episodes mapping does not match episode_count")
        self.metadata = dict(metadata)
        self._episodes = dict(episodes)
        self._cache: dict[int, tuple[tuple[int, int], ...]] = {}

    def segments_for_episode(self, episode_index: int) -> tuple[tuple[int, int], ...]:
        if isinstance(episode_index, bool) or not isinstance(episode_index, int):
            raise ValueError("episode_index must be an integer")
        if episode_index in self._cache:
            return self._cache[episode_index]
        relative = self._episodes.get(str(episode_index))
        if not isinstance(relative, str):
            raise KeyError(f"episode {episode_index} is absent from wrist-event manifest")
        path = (self.root / relative).resolve()
        if self.root not in path.parents:
            raise ValueError("episode path escapes manifest root")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("episode") != episode_index:
            raise ValueError("episode payload identity mismatch")
        from .dynamic_surprise import validate_episode_segments

        segments = validate_episode_segments(payload)
        self._cache[episode_index] = segments
        return segments
