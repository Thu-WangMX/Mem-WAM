"""Immutable manifests for causal physical settle/gripper L4--8/K8 memory."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "physical_settle_rate_debt_l4to8_k8_v1"
EVENT_REASON = "selfcal_settle_or_gripper"
FALLBACK_REASON = "max8_rate_repay"
VALID_REASONS = {EVENT_REASON, FALLBACK_REASON}


@dataclass(frozen=True)
class PhysicalSettleRateSegment:
    start: int
    end: int
    confirmed_at: int
    reason: str
    evidence: float | None
    rate_debt: float
    cumulative_mean_length: float

    @property
    def length(self) -> int:
        return self.end - self.start


def validate_segments(payload: Mapping[str, Any]) -> tuple[PhysicalSettleRateSegment, ...]:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("physical settle/rate-debt episode schema mismatch")
    count = payload.get("decision_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("decision_count must be a positive integer")
    rows = payload.get("segments")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("segments must be a sequence")

    records: list[PhysicalSettleRateSegment] = []
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
        if confirmed_at != start + 8 or not end <= confirmed_at < count:
            raise ValueError("segment confirmation must be exactly start+8")
        if confirmed_at - end > 4 or confirmed_at <= previous_confirmation:
            raise ValueError("segment confirmation violates recent4 causality")
        if reason not in VALID_REASONS:
            raise ValueError(f"invalid physical settle reason: {reason!r}")

        evidence = row.get("evidence")
        if reason == EVENT_REASON:
            if end - start == 8:
                raise ValueError("event-selected segments must be L4..7")
            if not isinstance(evidence, (int, float)) or float(evidence) < 0.8:
                raise ValueError("event-selected segment evidence is below 0.8")
            parsed_evidence: float | None = float(evidence)
        else:
            if end - start != 8 or evidence is not None:
                raise ValueError("rate fallback must be L8 with no claimed evidence")
            parsed_evidence = None

        compressed_length += end - start
        debt = 7.0 * (index + 1) - compressed_length
        recorded_debt = row.get("rate_debt")
        recorded_mean = row.get("cumulative_mean_length")
        if not isinstance(recorded_debt, (int, float)) or abs(float(recorded_debt) - debt) > 1e-6:
            raise ValueError("rate debt mismatch")
        if debt > 5.0 + 1e-6:
            raise ValueError("rate debt exceeds cap5")
        mean = compressed_length / (index + 1)
        if not isinstance(recorded_mean, (int, float)) or abs(float(recorded_mean) - mean) > 1e-6:
            raise ValueError("cumulative mean length mismatch")
        records.append(
            PhysicalSettleRateSegment(
                start=start,
                end=end,
                confirmed_at=confirmed_at,
                reason=str(reason),
                evidence=parsed_evidence,
                rate_debt=float(debt),
                cumulative_mean_length=float(mean),
            )
        )
        expected_start = end
        previous_confirmation = confirmed_at
    return tuple(records)


class PhysicalSettleRateManifestStore:
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
            raise ValueError("physical settle metadata must be a mapping")
        expected = {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "task": str(expected_task),
            "episode_count": int(expected_episode_count),
            "anchor_frames": 2,
            "recent_frames": 4,
            "reader_raw_tail": "complete_open_segment_suffix",
            "fixed_reader_recent_frames": False,
            "maximum_raw_tail_frames": 8,
            "maximum_confirmation_delay_frames": 4,
            "min_segment": 4,
            "max_segment": 8,
            "memory_tokens": 8,
            "history_window": 8,
            "event_threshold": 0.8,
            "target_mean_segment": 7.0,
            "maximum_rate_debt": 5.0,
            "uses_action": False,
            "uses_reward": False,
            "uses_task_checkpoint": False,
            "uses_cross_episode_fit": False,
        }
        for key, want in expected.items():
            if metadata.get(key) != want:
                raise ValueError(
                    f"physical settle manifest {key}: expected {want!r}, got {metadata.get(key)!r}"
                )
        if metadata.get("selector_inputs") != ["observation.state"]:
            raise ValueError("physical settle selector input contract mismatch")
        episodes = manifest.get("episodes")
        if not isinstance(episodes, Mapping) or len(episodes) != expected_episode_count:
            raise ValueError("physical settle episode mapping mismatch")
        self.metadata = dict(metadata)
        self._episodes = dict(episodes)
        self._cache: dict[int, tuple[PhysicalSettleRateSegment, ...]] = {}

    def segments_for_episode(self, episode: int) -> tuple[PhysicalSettleRateSegment, ...]:
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
