"""Freeze strictly replayable dynamic frame-selection groups for training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from fastwam.memory.predictive_boundary import PredictiveBoundaryState


SCHEMA = "putback_dynamic_selection_manifest_v1"
SOURCE_HASH_KEYS = {
    "initialization_fingerprint", "code_commit", "predictor_sha256", "pca_sha256",
    "stats_sha256", "annotation_sha256",
}


def _state(config: Mapping[str, Any]) -> PredictiveBoundaryState:
    return PredictiveBoundaryState(
        stats={
            "median": torch.tensor(config["stats"]["median"]),
            "mad_scale": torch.tensor(config["stats"]["mad_scale"]),
        },
        weights=torch.tensor(config["weights"]),
        high_threshold=config["high_threshold"], low_threshold=config["low_threshold"],
        detector_stride=config["detector_stride"], min_units=config["min_units"],
        max_units=config["max_units"], nms_units=config["nms_units"],
        initial_group_start=config.get("initial_group_start"),
    )


def replay_episode(payload: Mapping[str, Any], config: Mapping[str, Any], source_hashes):
    state = _state(config)
    events = []
    for observation in payload["observations"]:
        event = state.update(frame=observation["frame"], residuals=observation["residuals"])
        if event is not None:
            events.append(event)
    tail = state.finalize(frame=payload["terminal_frame"])
    if tail is not None:
        events.append(tail)
    if state.retroactive_boundary_count:
        raise ValueError("retroactive boundary detected")
    groups = [
        {
            "start_frame": event.group_start,
            "end_frame": event.group_end,
            "detector_units": (event.group_end - event.group_start) // config["detector_stride"],
            "confirmation_frame": event.frame,
            "peak_frame": event.peak_frame,
            "reason": event.reason,
            "score": event.score,
            "selected_streams": list(event.selected_streams),
            "source_hashes": dict(source_hashes),
        }
        for event in events
    ]
    expected_start = config.get(
        "initial_group_start",
        payload["observations"][0]["frame"] - config["detector_stride"],
    )
    if not groups or groups[0]["start_frame"] != expected_start:
        raise ValueError("manifest does not cover the first detector unit")
    if groups[-1]["end_frame"] != payload["terminal_frame"] or any(
        left["end_frame"] != right["start_frame"] for left, right in zip(groups, groups[1:])
    ):
        raise ValueError("manifest groups do not provide complete non-overlapping coverage")
    return groups, state


def freeze_selection_manifest(
    *, traces, selector_config, source_hashes, heldout_gate,
    allow_single_dynamic_episode: bool = False,
):
    if heldout_gate.get("pass") is not True:
        raise ValueError("held-out reliability gate must pass before freezing training groups")
    if set(source_hashes) != SOURCE_HASH_KEYS:
        raise ValueError("source hashes are incomplete or incompatible")
    episodes = {}
    lengths = set()
    for episode in sorted(traces):
        groups, _ = replay_episode(traces[episode], selector_config, source_hashes)
        episodes[str(int(episode))] = {
            "terminal_frame": int(traces[episode]["terminal_frame"]),
            "groups": groups,
        }
        lengths.update(group["detector_units"] for group in groups if group["reason"] != "terminal_tail")
    if len(lengths) < 2 and not allow_single_dynamic_episode:
        raise ValueError("fixed-length degeneration: fewer than two non-terminal group lengths")
    return {
        "schema_version": SCHEMA,
        "selector_config": json.loads(json.dumps(selector_config)),
        "source_hashes": dict(source_hashes),
        "heldout_gate": dict(heldout_gate),
        "episodes": episodes,
    }


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
