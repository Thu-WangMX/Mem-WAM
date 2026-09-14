"""Immutable causal multimodal rate-controlled L4--8/K8 manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "multimodal_rate_controlled_l4to8_k8_v2"
VALID_REASONS = {
    "strong_latent",
    "latent_physical_settled_peak",
    "latent_physical_gripper",
    "max8_fallback",
}


@dataclass(frozen=True)
class MultimodalRateSegment:
    start: int
    end: int
    confirmed_at: int
    reason: str

    @property
    def length(self) -> int:
        return self.end - self.start


def validate_segments(payload: Mapping[str, Any]) -> tuple[MultimodalRateSegment, ...]:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("multimodal rate-controlled episode schema mismatch")
    count = payload.get("decision_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("decision_count must be a positive integer")
    rows = payload.get("segments")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("segments must be a sequence")
    records = []
    expected_start = 2
    compressed_length = 0
    previous_confirmation = -1
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError("segment rows must be mappings")
        values = row.get("start"), row.get("end"), row.get("confirmed_at")
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
        if reason not in VALID_REASONS:
            raise ValueError(f"invalid multimodal segment reason: {reason!r}")
        rank = row.get("latent_rank")
        if reason == "strong_latent" and not isinstance(rank, (int, float)):
            raise ValueError("strong latent boundary requires a rank")
        if reason == "strong_latent" and float(rank) < 0.75:
            raise ValueError("strong latent boundary rank is below 0.75")
        if str(reason).startswith("latent_physical_"):
            if not isinstance(rank, (int, float)) or float(rank) < 0.5:
                raise ValueError("physical-assisted boundary rank is below 0.5")
        if reason == "max8_fallback" and rank is not None:
            raise ValueError("max8 fallback must not claim latent support")
        compressed_length += end - start
        cumulative = compressed_length / (index + 1)
        recorded = row.get("cumulative_mean_length")
        if not isinstance(recorded, (int, float)) or abs(float(recorded) - cumulative) > 1e-6:
            raise ValueError("cumulative mean length mismatch")
        if cumulative < 6.0:
            raise ValueError("rate controller violated cumulative mean L>=6")
        records.append(MultimodalRateSegment(start, end, confirmed_at, str(reason)))
        expected_start = end
        previous_confirmation = confirmed_at
    return tuple(records)


class MultimodalRateManifestStore:
    def __init__(
        self,
        root: str | Path,
        *,
        expected_episode_count: int = 50,
        expected_task: str,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        manifest = json.loads((self.root / "manifest.json").read_text())
        metadata = manifest.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("multimodal metadata must be a mapping")
        expected = {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "task": str(expected_task),
            "episode_count": int(expected_episode_count),
            "anchor_frames": 2,
            "recent_frames": 4,
            "min_segment": 4,
            "max_segment": 8,
            "memory_tokens": 8,
            "strong_latent_rank_threshold": 0.75,
            "assisted_latent_rank_threshold": 0.5,
            "physical_alignment_radius_decisions": 1,
            "minimum_cumulative_mean_segment": 6.0,
            "physical_event_selection": "strict_online_first_confirmed_refractory",
            "uses_action": False,
            "uses_reward": False,
            "uses_task_checkpoint": False,
        }
        for key, want in expected.items():
            if metadata.get(key) != want:
                raise ValueError(
                    f"multimodal manifest {key}: expected {want!r}, got {metadata.get(key)!r}"
                )
        if metadata.get("selector_inputs") != [
            "frozen_wan22_vae_latent",
            "observation.state",
        ]:
            raise ValueError("multimodal selector input contract mismatch")
        scale = metadata.get("joint_delta_scale")
        if not isinstance(scale, list) or len(scale) != 12 or any(float(x) <= 0 for x in scale):
            raise ValueError("joint_delta_scale must contain 12 positive values")
        episodes = manifest.get("episodes")
        if not isinstance(episodes, Mapping) or len(episodes) != expected_episode_count:
            raise ValueError("multimodal episode mapping mismatch")
        self.metadata = dict(metadata)
        self._episodes = dict(episodes)
        self._cache: dict[int, tuple[MultimodalRateSegment, ...]] = {}

    def segments_for_episode(self, episode: int) -> tuple[MultimodalRateSegment, ...]:
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
