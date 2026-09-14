from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from fastwam.memory.phase_annotation import AnnotationStore
from fastwam.memory.phase_boundary_metrics import reliability_gate


LOCK_SCHEMA = "putback_locked_predictive_selector_v1"
REQUIRED_HASH_KEYS = {
    "predictor_sha256", "pca_sha256", "stats_sha256", "config_sha256", "code_sha256"
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_passing_candidate(candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    passing = []
    for original in candidates:
        candidate = json.loads(json.dumps(original))
        frequencies = Counter(stream for fold in candidate["fold_streams"] for stream in set(fold))
        candidate["stable_streams"] = sorted(
            stream for stream, count in frequencies.items() if count >= 5
        )
        if candidate["stable_streams"] and reliability_gate(candidate["report"])["pass"]:
            passing.append(candidate)
    if not passing:
        raise RuntimeError("no candidate passes the development reliability gate")
    passing.sort(
        key=lambda row: (
            -row["report"]["macro"]["f1"],
            -row["report"]["macro"]["precision"],
            row["report"]["memory_count"],
            row["name"],
        )
    )
    return passing[0]


def lock_candidate(path: str | Path, *, candidate: dict[str, Any], hashes: dict[str, str]) -> None:
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite locked candidate: {path}")
    if set(hashes) != REQUIRED_HASH_KEYS or any(len(value) != 64 for value in hashes.values()):
        raise ValueError("locked candidate requires exact predictor/PCA/stats/config/code hashes")
    payload = {"schema_version": LOCK_SCHEMA, "candidate": candidate, "hashes": hashes}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


class LockedEvaluationSession:
    def __init__(
        self,
        *,
        annotation_path: str | Path,
        mode: str,
        locked_candidate_path: str | Path | None = None,
        heldout_marker: str | Path | None = None,
    ) -> None:
        annotation_path = Path(annotation_path).resolve()
        if mode == "development":
            self.store = AnnotationStore(annotation_path, access="development")
            return
        if mode != "final" or locked_candidate_path is None or heldout_marker is None:
            raise ValueError("final mode requires locked candidate and held-out marker")
        locked_path = Path(locked_candidate_path).resolve()
        locked = json.loads(locked_path.read_text())
        if locked.get("schema_version") != LOCK_SCHEMA or set(locked.get("hashes", {})) != REQUIRED_HASH_KEYS:
            raise ValueError("locked candidate contract mismatch")
        locked_sha = _sha256(locked_path)
        marker = Path(heldout_marker).resolve()
        if marker.exists():
            existing = json.loads(marker.read_text())
            if existing.get("locked_candidate_sha256") != locked_sha:
                raise RuntimeError("held-out set was opened by a different locked candidate")
        else:
            marker.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "locked_candidate_sha256": locked_sha,
                "heldout_opened_at": datetime.now(timezone.utc).isoformat(),
            }
            temporary = marker.with_suffix(marker.suffix + ".tmp")
            temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            temporary.replace(marker)
        self.store = AnnotationStore(
            annotation_path,
            access="final",
            locked_selector_hash=locked_sha,
            heldout_marker=marker.with_suffix(".annotation_access.json"),
        )

    def load_episode(self, episode: int) -> dict[str, Any]:
        return self.store.load_episode(episode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-json", required=True)
    parser.add_argument("--locked-candidate", required=True)
    args = parser.parse_args()
    comparison = json.loads(Path(args.comparison_json).read_text())
    winner = select_passing_candidate(comparison["candidates"])
    lock_candidate(args.locked_candidate, candidate=winner, hashes=comparison["hashes"])


if __name__ == "__main__":
    main()

