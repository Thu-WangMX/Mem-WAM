"""Versioned human-reviewed PutBack manipulation-phase annotations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ANNOTATION_SCHEMA = "putback_semantic_phase_annotations_v1"
TRANSITIONS = (
    "approach_to_contact",
    "contact_to_lift",
    "lift_to_transport",
    "transport_to_place",
    "place_to_release_stable",
)
DEVELOPMENT_EPISODES = tuple(range(30, 40))
HELDOUT_EPISODES = tuple(range(40, 50))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_episode_annotation(
    episode_payload: dict[str, Any], *, require_reviewed: bool = True
) -> None:
    episode = int(episode_payload.get("episode", -1))
    if episode not in (*DEVELOPMENT_EPISODES, *HELDOUT_EPISODES):
        raise ValueError("annotation episode must be between 30 and 49")
    expected_split = "development" if episode in DEVELOPMENT_EPISODES else "heldout"
    if episode_payload.get("split") != expected_split:
        raise ValueError(f"episode {episode} split must be {expected_split}")
    reviewed = episode_payload.get("reviewed") is True
    if require_reviewed and not reviewed:
        raise ValueError("annotation must be human reviewed")
    if reviewed and not str(episode_payload.get("reviewer", "")).strip():
        raise ValueError("reviewed annotation requires a reviewer")
    transitions = episode_payload.get("transitions")
    if not isinstance(transitions, list) or [row.get("name") for row in transitions] != list(
        TRANSITIONS
    ):
        raise ValueError(f"transition names must be exactly {TRANSITIONS}")
    observed_frames: list[int] = []
    for row in transitions:
        if not reviewed and not require_reviewed and row.get("status") == "pending":
            if any(
                row.get(key) is not None
                for key in (
                    "frame",
                    "ambiguity_start",
                    "ambiguity_end",
                    "not_observed",
                    "reason",
                )
            ):
                raise ValueError("pending transition cannot contain reviewed label fields")
            continue
        missing = row.get("not_observed") is True
        if missing:
            if any(row.get(key) is not None for key in ("frame", "ambiguity_start", "ambiguity_end")):
                raise ValueError("not-observed transition cannot contain frame timestamps")
            if not str(row.get("reason", "")).strip():
                raise ValueError("not-observed transition requires a reason")
            continue
        if row.get("not_observed") is not False:
            raise ValueError("transition must explicitly set not_observed")
        frame = row.get("frame")
        start = row.get("ambiguity_start")
        end = row.get("ambiguity_end")
        if not all(isinstance(value, int) for value in (frame, start, end)):
            raise ValueError("observed transition timestamps must be integers")
        if any(value % 4 for value in (frame, start, end)):
            raise ValueError("transition timestamps must be divisible by four")
        if not start <= frame <= end:
            raise ValueError("transition frame must lie inside its ambiguity interval")
        if row.get("reason") is not None:
            raise ValueError("observed transition reason must be null")
        observed_frames.append(frame)
    if any(later <= earlier for earlier, later in zip(observed_frames, observed_frames[1:])):
        raise ValueError("observed transitions must be strictly ordered")


class AnnotationStore:
    """Read annotations while enforcing development/final label isolation."""

    def __init__(
        self,
        path: str | Path,
        *,
        access: str,
        locked_selector_hash: str | None = None,
        heldout_marker: str | Path | None = None,
    ):
        self.path = Path(path).expanduser().resolve()
        payload = json.loads(self.path.read_text())
        if payload.get("schema_version") != ANNOTATION_SCHEMA or payload.get("frame_stride") != 4:
            raise ValueError("annotation manifest contract mismatch")
        rows = payload.get("episodes")
        if not isinstance(rows, list):
            raise ValueError("annotation manifest episodes must be a list")
        self._episodes: dict[int, dict[str, Any]] = {}
        for row in rows:
            validate_episode_annotation(row)
            episode = int(row["episode"])
            if episode in self._episodes:
                raise ValueError(f"duplicate annotation episode {episode}")
            self._episodes[episode] = row
        self.access = str(access)
        if self.access not in {"development", "final"}:
            raise ValueError("access must be development or final")
        self.locked_selector_hash = locked_selector_hash
        if self.access == "final":
            if not isinstance(locked_selector_hash, str) or len(locked_selector_hash) < 8:
                raise ValueError("final access requires a locked selector hash")
            if heldout_marker is None:
                raise ValueError("final access requires a held-out marker path")
            marker = Path(heldout_marker).expanduser().resolve()
            marker_payload = {
                "schema_version": ANNOTATION_SCHEMA,
                "annotation_sha256": _sha256(self.path),
                "locked_selector_hash": locked_selector_hash,
                "heldout_episodes": list(HELDOUT_EPISODES),
            }
            if marker.exists():
                existing = json.loads(marker.read_text())
                if existing != marker_payload:
                    raise RuntimeError("held-out annotations were opened by a different selector")
            else:
                marker.parent.mkdir(parents=True, exist_ok=True)
                temporary = marker.with_suffix(marker.suffix + ".tmp")
                temporary.write_text(json.dumps(marker_payload, indent=2, sort_keys=True) + "\n")
                temporary.replace(marker)

    def load_episode(self, episode: int) -> dict[str, Any]:
        episode = int(episode)
        if episode in HELDOUT_EPISODES and self.access != "final":
            raise PermissionError("held-out annotations are unavailable in development mode")
        if episode not in self._episodes:
            raise KeyError(f"annotation episode {episode} is absent")
        return json.loads(json.dumps(self._episodes[episode]))


def write_reviewed_manifest(path: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite reviewed annotation manifest: {path}")
    if payload.get("schema_version") != ANNOTATION_SCHEMA or payload.get("frame_stride") != 4:
        raise ValueError("reviewed annotation manifest contract mismatch")
    rows = payload.get("episodes")
    if not isinstance(rows, list) or sorted(int(row["episode"]) for row in rows) != list(
        range(30, 50)
    ):
        raise ValueError("reviewed manifest must contain exactly episodes 30-49")
    for row in rows:
        validate_episode_annotation(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return {"sha256": _sha256(path), "episode_count": len(rows), "path": str(path)}
