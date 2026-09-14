"""Run the predeclared held-out PutBack WAM warmup comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch

from fastwam.memory.wam_warmup_comparison import (
    STRATEGIES,
    build_strategy_trace,
    calibrate_adjacent_threshold,
    evaluate_decision_rule,
    evaluate_strategy,
)
from scripts.extract_putback_init_wam_embedding_bank import BANK_SCHEMA


REPORT_SCHEMA = "putback_wam_warmup_comparison_v1"
CALIBRATION_EPISODES = list(range(40))
HELDOUT_EPISODES = list(range(40, 50))


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


def _load_bank(bank: Path) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    manifest_path = bank / "bank_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema_version") != BANK_SCHEMA
        or manifest.get("complete") is not True
        or int(manifest.get("episode_count", -1)) != 50
    ):
        raise ValueError("feature bank manifest is incomplete or incompatible")
    rows = manifest.get("episodes")
    if [int(row["episode"]) for row in rows] != list(range(50)):
        raise ValueError("feature bank must contain exactly episodes 0 through 49")
    features: dict[int, torch.Tensor] = {}
    for manifest_row in rows:
        episode = int(manifest_row["episode"])
        feature_path = bank / "features" / f"episode_{episode:03d}.pt"
        metadata_path = bank / "episodes" / f"episode_{episode:03d}.json"
        metadata = json.loads(metadata_path.read_text())
        digest = _sha256(feature_path)
        if (
            metadata.get("schema_version") != BANK_SCHEMA
            or int(metadata.get("episode", -1)) != episode
            or metadata.get("feature_sha256") != digest
            or manifest_row.get("feature_sha256") != digest
            or metadata.get("initialization_fingerprint")
            != manifest.get("initialization_fingerprint")
        ):
            raise ValueError(f"feature bank entry is incompatible: episode {episode}")
        saved = torch.load(feature_path, map_location="cpu", weights_only=True)
        value = torch.as_tensor(saved["features"], dtype=torch.float32)
        frames = torch.as_tensor(saved["decision_frame_indices"], dtype=torch.int64)
        if (
            value.ndim != 2
            or int(value.shape[1]) != int(metadata["feature_dim"])
            or frames.tolist() != metadata["decision_frame_indices"]
            or frames.tolist() != list(range(0, 16 * int(value.shape[0]), 16))
        ):
            raise ValueError(f"feature tensor is incompatible: episode {episode}")
        features[episode] = value
    return features, manifest


def _load_actions(dataset: Path, episode: int) -> np.ndarray:
    path = dataset / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
    table = pq.read_table(path, columns=["episode_index", "frame_index", "action"])
    identities = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
    frames = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
    if not np.all(identities == episode) or not np.array_equal(frames, np.arange(len(table))):
        raise ValueError(f"held-out parquet identity mismatch: episode {episode}")
    return np.asarray(table["action"].to_pylist(), dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    bank = Path(args.bank).expanduser().resolve()
    dataset = Path(args.dataset_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite warmup comparison: {output}")
    features, bank_manifest = _load_bank(bank)
    calibration = calibrate_adjacent_threshold(
        features, CALIBRATION_EPISODES, gamma=1.0
    )
    actions = {episode: _load_actions(dataset, episode) for episode in HELDOUT_EPISODES}

    output.mkdir(parents=True)
    strategy_metrics = {}
    for strategy in STRATEGIES:
        traces = {
            episode: build_strategy_trace(
                features[episode],
                strategy=strategy,
                adjacent_threshold=(
                    calibration["threshold"] if strategy == "adjacent_cosine" else None
                ),
                nms_distance=2,
                min_segment=2,
                max_segment=8,
            )
            for episode in HELDOUT_EPISODES
        }
        trace_dir = output / "traces" / strategy
        trace_dir.mkdir(parents=True)
        for episode, trace in traces.items():
            _json(
                trace_dir / f"episode_{episode:03d}.json",
                {
                    "schema_version": REPORT_SCHEMA,
                    "episode": episode,
                    "decision_frame_indices": list(
                        range(0, 16 * int(features[episode].shape[0]), 16)
                    ),
                    **trace,
                },
            )
        strategy_metrics[strategy] = evaluate_strategy(
            traces, actions, stride=16, early_end=8, tolerance=1.0
        )
    decision_rule = evaluate_decision_rule(
        adjacent=strategy_metrics["adjacent_cosine"],
        current=strategy_metrics["current"],
        short_history=strategy_metrics["short_history"],
    )
    report = {
        "schema_version": REPORT_SCHEMA,
        "contract": {
            "calibration_episodes": CALIBRATION_EPISODES,
            "heldout_episodes": HELDOUT_EPISODES,
            "eligible_calibration_decisions": list(range(2, 9)),
            "startup_suppressed_decision": 1,
            "gamma": 1.0,
            "nms_distance": 2,
            "matching_tolerance": 1.0,
            "policy_checkpoint": None,
            "uses_vlm": False,
        },
        "bank": str(bank),
        "bank_manifest_sha256": _sha256(bank / "bank_manifest.json"),
        "initialization_fingerprint": bank_manifest["initialization_fingerprint"],
        "calibration": calibration,
        "action_loaded_episodes": HELDOUT_EPISODES,
        "strategies": strategy_metrics,
        "decision_rule": decision_rule,
    }
    _json(output / "report.json", report)
    print(
        json.dumps(
            {
                "output": str(output),
                "adjacent_threshold": calibration["threshold"],
                "decision_rule": decision_rule,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
