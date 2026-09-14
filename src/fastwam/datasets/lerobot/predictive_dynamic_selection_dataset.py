"""Manifest-backed dynamic grouping adapter reusing existing history latents."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from fastwam.memory.dynamic_selection_manifest import SCHEMA
from .full_kv_dataset import _as_int


class DynamicSelectionManifestStore:
    def __init__(self, *, manifest, verification, expected_source_hashes):
        if manifest.get("schema_version") != SCHEMA:
            raise ValueError("dynamic manifest schema mismatch")
        if verification.get("exact") is not True:
            raise ValueError("dynamic manifest must be replay verified")
        if manifest.get("source_hashes") != expected_source_hashes:
            raise ValueError("dynamic manifest source hash mismatch")
        self.manifest = manifest

    def groups(self, episode: int):
        try:
            return self.manifest["episodes"][str(int(episode))]["groups"]
        except KeyError as exc:
            raise KeyError(f"episode {episode} absent from dynamic manifest") from exc


def attach_predictive_dynamic_groups(
    sample: dict[str, Any], store: DynamicSelectionManifestStore, *, tokens_per_group: int
) -> dict[str, Any]:
    episode = _as_int(sample["episode_index"])
    decision = _as_int(sample["frame_index"])
    frames = torch.as_tensor(sample["observation_frame_indices"], dtype=torch.int64)
    latents = torch.as_tensor(sample["history_latents"])
    if frames.ndim != 1 or len(frames) != len(latents):
        raise ValueError("observation frames and history latents must align")
    expected = torch.arange(4, decision + 1, 4, dtype=torch.int64)
    if not torch.equal(frames, expected):
        raise ValueError("history has missing, duplicated, or out-of-order 4-frame coverage")
    groups = []
    for frozen in store.groups(episode):
        if frozen["start_frame"] >= decision:
            break
        row = dict(frozen)
        if row["end_frame"] > decision:
            row["end_frame"] = decision
            row["detector_units"] = (decision - row["start_frame"]) // 4
            row["reason"] = "active_prefix"
        groups.append(row)
        if row["end_frame"] == decision:
            break
    if not groups or groups[0]["start_frame"] != 0 or groups[-1]["end_frame"] != decision or any(
        left["end_frame"] != right["start_frame"] for left, right in zip(groups, groups[1:])
    ):
        raise ValueError("manifest grouping does not cover the sample history")
    ranges = []
    chunks = []
    cursor = 0
    for group in groups:
        length = int(group["detector_units"])
        ranges.append((cursor, cursor + length))
        chunks.append(latents[cursor : cursor + length])
        cursor += length
    if cursor != len(latents):
        raise ValueError("group coverage skipped or duplicated observation units")
    sample["history_memory_groups"] = torch.tensor(ranges, dtype=torch.int64)
    sample["history_memory_group_lengths"] = torch.tensor(
        [group["detector_units"] for group in groups], dtype=torch.int64
    )
    sample["history_memory_group_reasons"] = [group["reason"] for group in groups]
    sample["compressor_inputs"] = chunks
    sample["memory_token_count"] = len(groups) * int(tokens_per_group)
    sample["memory_attention_mask"] = torch.ones(sample["memory_token_count"], dtype=torch.int64)
    # Scores and semantic annotations remain debug-only and are never placed in action input.
    sample["boundary_debug_metadata"] = groups
    sample["action_model_metadata"] = {
        "group_ranges": sample["history_memory_groups"],
        "token_count": sample["memory_token_count"],
    }
    return sample

