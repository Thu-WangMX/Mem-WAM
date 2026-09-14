"""Replay persisted initialization-WAM features into causal-v2 traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import torch

from fastwam.memory.wam_embedding_causal_v2 import (
    CAUSAL_V2_SCHEMA,
    build_causal_v2_trace,
)


SOURCE_SCHEMA = "putback_wam_embedding_surprise_init_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _episodes(text: str) -> list[int]:
    result = [int(value) for value in text.split(",") if value.strip()]
    if not result or len(result) != len(set(result)):
        raise ValueError("episodes must be nonempty and unique")
    return result


def _trace_parameters(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "statistics_window": args.statistics_window,
        "threshold_window": args.threshold_window,
        "min_history": args.min_history,
        "min_threshold_history": args.min_threshold_history,
        "gamma": args.gamma,
        "nms_distance": args.nms_distance,
        "min_segment": args.min_segment,
        "max_segment": args.max_segment,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", default="40,41")
    parser.add_argument("--statistics-window", type=int, default=8)
    parser.add_argument("--threshold-window", type=int, default=8)
    parser.add_argument("--min-history", type=int, default=4)
    parser.add_argument("--min-threshold-history", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--nms-distance", type=int, default=2)
    parser.add_argument("--min-segment", type=int, default=2)
    parser.add_argument("--max-segment", type=int, default=8)
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite causal-v2 output: {output}")
    initialization_path = source / "initialization_manifest.json"
    initialization = json.loads(initialization_path.read_text())
    if (
        initialization.get("schema_version") != SOURCE_SCHEMA
        or initialization.get("model_source") != "initialization"
        or initialization.get("policy_checkpoint") is not None
    ):
        raise ValueError("source initialization manifest is incompatible")
    fingerprint = initialization.get("fresh_action_io_proprio_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("source initialization fingerprint is missing")

    (output / "features").mkdir(parents=True)
    (output / "episodes").mkdir()
    parameters = _trace_parameters(args)
    rows = []
    for episode in _episodes(args.episodes):
        source_feature = source / "features" / f"episode_{episode:03d}.pt"
        source_episode = source / "episodes" / f"episode_{episode:03d}.json"
        metadata = json.loads(source_episode.read_text())
        if metadata.get("schema_version") != SOURCE_SCHEMA or int(metadata["episode"]) != episode:
            raise ValueError(f"source episode metadata is incompatible: {source_episode}")
        saved = torch.load(source_feature, map_location="cpu", weights_only=True)
        features = torch.as_tensor(saved["features"])
        frame_indices = torch.as_tensor(saved["decision_frame_indices"], dtype=torch.int64)
        if frame_indices.tolist() != metadata["decision_frame_indices"]:
            raise ValueError(f"source feature indices mismatch for episode {episode}")
        trace = build_causal_v2_trace(features, **parameters)
        payload = {
            "schema_version": CAUSAL_V2_SCHEMA,
            "episode": episode,
            "decision_count": int(features.shape[0]),
            "decision_frame_indices": frame_indices.tolist(),
            "feature_dim": int(features.shape[1]),
            "source_feature": str(source_feature),
            "source_feature_sha256": _sha256(source_feature),
            **trace,
        }
        target_feature = output / "features" / source_feature.name
        shutil.copy2(source_feature, target_feature)
        replay_saved = torch.load(target_feature, map_location="cpu", weights_only=True)
        replay = build_causal_v2_trace(replay_saved["features"], **parameters)
        if trace != replay:
            raise RuntimeError(f"causal-v2 replay mismatch for episode {episode}")
        payload["replay_validation"] = "pass"
        _json(output / "episodes" / f"episode_{episode:03d}.json", payload)
        rows.append(
            {
                "episode": episode,
                "boundaries": trace["boundaries"],
                "segment_lengths": trace["segment_lengths"],
                "peaks": trace["peaks"],
                "peak_confirmed_at": trace["peak_confirmed_at"],
                "retroactive_boundary_count": trace["retroactive_boundary_count"],
            }
        )

    manifest = {
        "schema_version": CAUSAL_V2_SCHEMA,
        "model_source": "initialization",
        "policy_checkpoint": None,
        "source_root": str(source),
        "source_initialization_fingerprint": fingerprint,
        "source_initialization_manifest_sha256": _sha256(initialization_path),
        "episodes": _episodes(args.episodes),
        "trace_parameters": parameters,
    }
    _json(output / "manifest.json", manifest)
    _json(
        output / "aggregate.json",
        {
            "schema_version": CAUSAL_V2_SCHEMA,
            "episodes": rows,
            "retroactive_boundary_count": sum(
                row["retroactive_boundary_count"] for row in rows
            ),
            "replay_validation": "pass",
        },
    )
    print(json.dumps({"output": str(output), "episodes": _episodes(args.episodes), "replay_validation": "pass"}, sort_keys=True))


if __name__ == "__main__":
    main()
