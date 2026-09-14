"""Frozen physical dynamic-L manifests shared by training and inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "putback_physical_dynamic_l_k8_v1"


@dataclass(frozen=True)
class PhysicalSegment:
    start: int
    end: int
    confirmed_at: int
    reason: str

    @property
    def length(self) -> int:
        return self.end - self.start


def validate_segments(
    payload: Mapping[str, Any],
    *,
    decision_count: int | None = None,
    anchor_frames: int = 2,
    min_segment: int = 4,
    max_segment: int = 8,
) -> tuple[PhysicalSegment, ...]:
    recorded_count = payload.get("decision_count")
    if isinstance(recorded_count, bool) or not isinstance(recorded_count, int):
        raise ValueError("decision_count must be an integer")
    if decision_count is not None and recorded_count != int(decision_count):
        raise ValueError(
            f"decision count mismatch: manifest={recorded_count}, data={decision_count}"
        )
    raw = payload.get("segments")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("segments must be a sequence")
    records = []
    expected_start = int(anchor_frames)
    previous_confirmation = -1
    for row in raw:
        if not isinstance(row, Mapping):
            raise ValueError("segment rows must be mappings")
        values = (row.get("start"), row.get("end"), row.get("confirmed_at"))
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ValueError("segment start/end/confirmed_at must be integers")
        start, end, confirmed_at = (int(value) for value in values)
        reason = row.get("reason")
        if not isinstance(reason, str) or not reason:
            raise ValueError("segment reason must be a nonempty string")
        if start != expected_start:
            raise ValueError(
                f"segments must be contiguous after anchor{anchor_frames}: "
                f"expected start {expected_start}, got {start}"
            )
        if not min_segment <= end - start <= max_segment:
            raise ValueError(f"segment [{start},{end}) has invalid length {end-start}")
        if end > recorded_count or confirmed_at < end or confirmed_at >= recorded_count:
            raise ValueError(
                f"segment [{start},{end}) has invalid confirmation {confirmed_at} "
                f"for decision_count={recorded_count}"
            )
        if confirmed_at <= previous_confirmation:
            raise ValueError("segment confirmations must be strictly increasing")
        records.append(PhysicalSegment(start, end, confirmed_at, reason))
        expected_start = end
        previous_confirmation = confirmed_at
    return tuple(records)


class PhysicalDynamicLManifestStore:
    """Read-only accessor for one immutable physical-selector snapshot."""

    def __init__(self, root: str | Path, *, expected_episode_count: int = 50) -> None:
        self.root = Path(root).expanduser().resolve()
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        metadata = manifest.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("physical manifest metadata must be a mapping")
        expected = {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "task": "put_back_block",
            "replan_stride": 16,
            "anchor_frames": 2,
            "recent_frames": 4,
            "min_segment": 4,
            "nominal_segment": 6,
            "max_segment": 8,
            "memory_tokens": 8,
            "uses_robotwin_pretrained": False,
        }
        for key, want in expected.items():
            if metadata.get(key) != want:
                raise ValueError(
                    f"physical manifest {key}: expected {want!r}, got {metadata.get(key)!r}"
                )
        if metadata.get("episode_count") != int(expected_episode_count):
            raise ValueError("physical manifest episode_count mismatch")
        episodes = manifest.get("episodes")
        if not isinstance(episodes, Mapping) or len(episodes) != expected_episode_count:
            raise ValueError("physical manifest episode mapping mismatch")
        self.metadata = dict(metadata)
        self._episodes = dict(episodes)
        self._cache: dict[int, tuple[PhysicalSegment, ...]] = {}

    def segments_for_episode(
        self, episode: int, *, decision_count: int | None = None
    ) -> tuple[PhysicalSegment, ...]:
        episode = int(episode)
        if episode not in self._cache:
            relative = self._episodes.get(str(episode))
            if not isinstance(relative, str):
                raise KeyError(f"episode {episode} is absent from physical manifest")
            path = (self.root / relative).resolve()
            if self.root not in path.parents:
                raise ValueError("physical manifest episode path escapes root")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("episode") != episode:
                raise ValueError("physical manifest episode identity mismatch")
            self._cache[episode] = validate_segments(payload)
        records = self._cache[episode]
        if decision_count is not None:
            # Validate the immutable episode count without re-reading the file.
            if records and records[-1].end > int(decision_count):
                raise ValueError("physical segments exceed decision count")
        return records
