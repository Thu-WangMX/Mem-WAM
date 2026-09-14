"""Analyze initialization-WAM surprise alignment with PutBack gripper events."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from fastwam.memory.semantic_alignment import analyze_episode_alignment
from fastwam.memory.wam_embedding_surprise import validate_causal_trace


SOURCE_SCHEMA = "putback_wam_embedding_surprise_init_v1"
REPORT_SCHEMA = "putback_wam_surprise_semantic_alignment_v1"


def _parse_episodes(text: str) -> list[int]:
    episodes = [int(item) for item in str(text).split(",") if item.strip()]
    if not episodes or len(episodes) != len(set(episodes)):
        raise ValueError("episodes must be nonempty and unique")
    return episodes


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_actions(dataset_root: Path, episode: int) -> np.ndarray:
    path = dataset_root / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
    table = pq.read_table(path, columns=["episode_index", "frame_index", "action"])
    episode_indices = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
    frame_indices = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
    if len(table) < 2 or not np.all(episode_indices == episode):
        raise ValueError(f"parquet episode identity mismatch: {path}")
    if not np.array_equal(frame_indices, np.arange(len(table), dtype=np.int64)):
        raise ValueError(f"parquet frame indices are not contiguous: {path}")
    return np.asarray(table["action"].to_pylist(), dtype=np.float32)


def _micro(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    metrics = [row[key] for row in rows]
    confirmations = sum(int(item["counts"]["confirmations"]) for item in metrics)
    events = sum(int(item["counts"]["events"]) for item in metrics)
    matched = sum(int(item["counts"]["matched"]) for item in metrics)
    precision = matched / confirmations if confirmations else 0.0
    recall = matched / events if events else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    signed = [float(match["signed_error"]) for item in metrics for match in item["matches"]]
    absolute = [float(match["absolute_error"]) for item in metrics for match in item["matches"]]
    return {
        "counts": {
            "confirmations": confirmations,
            "events": events,
            "matched": matched,
        },
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_signed_error": sum(signed) / len(signed) if signed else None,
        "mean_absolute_error": sum(absolute) / len(absolute) if absolute else None,
    }


def _exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite semantic report: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite temporary report: {temporary}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--episodes", default="40,41")
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--tolerance", type=float, default=1.0)
    parser.add_argument("--warmup-end", type=float, default=8.0)
    args = parser.parse_args()

    analysis_root = Path(args.analysis_root).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    rows = []
    for episode in _parse_episodes(args.episodes):
        trace_path = analysis_root / "episodes" / f"episode_{episode:03d}.json"
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        if trace.get("schema_version") != SOURCE_SCHEMA or int(trace["episode"]) != episode:
            raise ValueError(f"incompatible source trace: {trace_path}")
        validate_causal_trace(trace)
        alignment = analyze_episode_alignment(
            trace,
            _load_actions(dataset_root, episode),
            stride=args.stride,
            warmup_end=args.warmup_end,
            tolerance=args.tolerance,
        )
        rows.append(
            {
                "episode": episode,
                "source_trace": str(trace_path),
                "source_trace_sha256": _sha256(trace_path),
                **alignment,
            }
        )

    payload = {
        "schema_version": REPORT_SCHEMA,
        "contract": {
            "episodes": _parse_episodes(args.episodes),
            "stride": int(args.stride),
            "tolerance": float(args.tolerance),
            "warmup_end": float(args.warmup_end),
            "strong_event_source": "action_gripper_dims_6_13_threshold_0.5",
            "matching_time": "peak_confirmed_at",
            "uses_vlm": False,
            "policy_checkpoint": None,
        },
        "episodes": rows,
        "aggregate": {
            "primary": _micro(rows, "primary_metrics"),
            "post_warmup": _micro(rows, "post_warmup_metrics"),
            "retroactive_boundary_count": sum(
                len(row["retroactive_boundaries"]) for row in rows
            ),
            "forced_boundary_count": sum(len(row["forced_boundaries"]) for row in rows),
        },
    }
    output = analysis_root / "semantic_alignment.json"
    _exclusive_json(output, payload)
    print(json.dumps({"output": str(output), **payload["aggregate"]}, sort_keys=True))


if __name__ == "__main__":
    main()
