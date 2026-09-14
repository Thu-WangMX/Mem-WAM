"""PutBack FullKV samples with frozen physical dynamic-L groups and fixed K8."""

from __future__ import annotations

import torch

from fastwam.memory.physical_dynamic_l import PhysicalDynamicLManifestStore

from .full_kv_dataset import FullKVRobotVideoDataset, _as_int


class PhysicalDynamicLRobotVideoDataset(FullKVRobotVideoDataset):
    def __init__(
        self,
        *args,
        physical_manifest_path: str,
        memory_tokens: int = 8,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.physical_manifest_store = PhysicalDynamicLManifestStore(
            physical_manifest_path,
            expected_episode_count=50,
        )
        self.physical_memory_tokens = int(memory_tokens)
        if self.physical_memory_tokens != 8:
            raise ValueError("physical dynamic-L v1 requires fixed K8")

    def _get(self, idx):
        sample = super()._get(idx)
        history = sample.get("history_latents")
        if not isinstance(history, torch.Tensor) or history.ndim != 5:
            raise RuntimeError("physical dynamic-L sample requires rank-5 history_latents")
        current_decision = int(history.shape[0]) - 1
        frame_index = _as_int(sample["frame_index"])
        if frame_index != current_decision * int(self.replan_stride):
            raise RuntimeError("physical dynamic-L history/frame index mismatch")
        episode = _as_int(sample["episode_index"])
        records = self.physical_manifest_store.segments_for_episode(episode)
        closed = tuple(row for row in records if row.confirmed_at <= current_decision)
        sample["history_memory_groups"] = torch.tensor(
            [(row.start, row.end) for row in closed], dtype=torch.int64
        ).reshape(-1, 2)
        sample["history_memory_token_counts"] = torch.full(
            (len(closed),), self.physical_memory_tokens, dtype=torch.int64
        )
        sample["history_memory_group_count"] = torch.tensor(
            len(closed), dtype=torch.int64
        )
        return sample
