"""Immutable train-free multi-resolution latent-kernel segment manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "trainfree_multires_latent_kernel_regime_k8_v1"


@dataclass(frozen=True)
class LatentKernelSegment:
    start: int
    end: int
    confirmed_at: int
    reason: str

    @property
    def length(self) -> int:
        return self.end - self.start


def validate_segments(payload: Mapping[str, Any]) -> tuple[LatentKernelSegment, ...]:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("latent-kernel episode schema mismatch")
    count = payload.get("decision_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("decision_count must be a positive integer")
    source_ends = payload.get("score_source_end")
    if not isinstance(source_ends, Sequence) or len(source_ends) != count:
        raise ValueError("score_source_end must match decision_count")
    rows = payload.get("segments")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("segments must be a sequence")
    records = []
    expected_start = 2
    previous_confirmation = -1
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("segment rows must be mappings")
        values = (row.get("start"), row.get("end"), row.get("confirmed_at"))
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ValueError("segment indices must be integers")
        start, end, confirmed_at = (int(value) for value in values)
        reason = row.get("reason")
        if start != expected_start or not 4 <= end - start <= 8:
            raise ValueError("segments must be contiguous L4..8 after anchor2")
        if not end <= confirmed_at < count or confirmed_at - end > 4:
            raise ValueError("segment confirmation violates recent4 causality")
        if confirmed_at <= previous_confirmation:
            raise ValueError("segment confirmations must increase")
        if reason not in {"multires_latent_kernel_regime", "stable_max_length"}:
            raise ValueError(f"invalid latent-kernel reason: {reason!r}")
        if reason == "multires_latent_kernel_regime":
            source_end = source_ends[end]
            if source_end is None or int(source_end) > confirmed_at:
                raise ValueError("event boundary reads an unavailable latent")
        records.append(LatentKernelSegment(start, end, confirmed_at, str(reason)))
        expected_start = end
        previous_confirmation = confirmed_at
    return tuple(records)


class LatentKernelManifestStore:
    def __init__(
        self,
        root: str | Path,
        *,
        expected_episode_count: int = 50,
        expected_task: str = "put_back_block",
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        manifest = json.loads((self.root / "manifest.json").read_text())
        metadata = manifest.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("latent-kernel metadata must be a mapping")
        expected = {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "task": str(expected_task),
            "episode_count": int(expected_episode_count),
            "selector_input": "frozen_continuous_causal_vae_latents_only",
            "uses_action": False,
            "uses_proprioception": False,
            "uses_reward": False,
            "uses_task_checkpoint": False,
            "fitted_parameters": None,
            "replan_stride": 16,
            "anchor_frames": 2,
            "recent_frames": 4,
            "min_segment": 4,
            "max_segment": 8,
            "memory_tokens": 8,
            "kernel_window_each_side": 2,
            "fusion": "unweighted_arithmetic_mean",
            "rank_threshold": 0.75,
            "history_window": 8,
        }
        for key, want in expected.items():
            if metadata.get(key) != want:
                raise ValueError(
                    f"latent-kernel manifest {key}: expected {want!r}, got {metadata.get(key)!r}"
                )
        if metadata.get("views") != [
            "appearance_flat",
            "scene_channel_mean",
            "local_channel_direction",
        ]:
            raise ValueError("latent-kernel view contract mismatch")
        episodes = manifest.get("episodes")
        if not isinstance(episodes, Mapping) or len(episodes) != expected_episode_count:
            raise ValueError("latent-kernel episode mapping mismatch")
        self.metadata = dict(metadata)
        self._episodes = dict(episodes)
        self._cache: dict[int, tuple[LatentKernelSegment, ...]] = {}

    def segments_for_episode(self, episode: int) -> tuple[LatentKernelSegment, ...]:
        episode = int(episode)
        if episode not in self._cache:
            relative = self._episodes.get(str(episode))
            if not isinstance(relative, str):
                raise KeyError(f"episode {episode} is absent")
            path = (self.root / relative).resolve()
            if self.root not in path.parents:
                raise ValueError("episode path escapes manifest root")
            payload = json.loads(path.read_text())
            if payload.get("episode") != episode:
                raise ValueError("episode identity mismatch")
            self._cache[episode] = validate_segments(payload)
        return self._cache[episode]
