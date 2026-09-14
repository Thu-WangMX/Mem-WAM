"""PutBack full-KV dataset with phase-frozen dynamic surprise groups."""

from __future__ import annotations

from typing import Any

import torch

from fastwam.memory.dynamic_surprise import (
    DynamicSurpriseManifestStore,
)
from fastwam.memory.event_conditioned_rate import memory_tokens_for_segment

from .full_kv_dataset import FullKVRobotVideoDataset, _as_int


def attach_dynamic_surprise_groups(
    sample: dict[str, Any],
    manifest_store: DynamicSurpriseManifestStore,
    *,
    replan_stride: int,
) -> dict[str, Any]:
    """Attach closed half-open segment ranges to one full-KV sample."""

    history = sample.get("history_latents")
    if not isinstance(history, torch.Tensor) or history.ndim != 5:
        raise RuntimeError("dynamic surprise sample requires rank-5 history_latents")
    history_frames = int(history.shape[0])
    frame_index = _as_int(sample["frame_index"])
    if frame_index % replan_stride:
        raise RuntimeError(
            f"decision frame {frame_index} is not divisible by replan_stride {replan_stride}"
        )
    decision_index = frame_index // replan_stride
    if decision_index != history_frames - 1:
        raise RuntimeError(
            "history/frame mismatch: "
            f"decision_index={decision_index}, history_frames={history_frames}"
        )

    episode_index = _as_int(sample["episode_index"])
    records = manifest_store.segment_records_for_episode(episode_index)
    current_decision = history_frames - 1
    closed = tuple(record for record in records if record.end <= current_decision)
    if closed:
        ranges = torch.tensor(
            [(record.start, record.end) for record in closed],
            dtype=torch.int64,
        )
        token_counts = torch.tensor(
            [
                memory_tokens_for_segment(
                    record.end - record.start,
                    record.reason,
                )
                for record in closed
            ],
            dtype=torch.int64,
        )
    else:
        ranges = torch.empty((0, 2), dtype=torch.int64)
        token_counts = torch.empty((0,), dtype=torch.int64)
    sample["history_memory_groups"] = ranges
    sample["history_memory_token_counts"] = token_counts
    sample["history_memory_group_count"] = torch.tensor(len(closed), dtype=torch.int64)
    return sample


class DynamicSurpriseRobotVideoDataset(FullKVRobotVideoDataset):
    """Full-KV samples pinned to one immutable boundary-snapshot manifest."""

    def __init__(
        self,
        *args,
        surprise_manifest_path: str,
        expected_boundary_step: int,
        expected_manifest_task: str = "putback",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.surprise_manifest_store = DynamicSurpriseManifestStore(
            surprise_manifest_path,
            expected_boundary_step=int(expected_boundary_step),
            expected_task=str(expected_manifest_task),
        )

    def _get(self, idx):
        sample = super()._get(idx)
        return attach_dynamic_surprise_groups(
            sample,
            self.surprise_manifest_store,
            replan_stride=self.replan_stride,
        )
