"""FullKV samples with frozen multimodal rate-controlled memory groups."""

from __future__ import annotations

import torch

from fastwam.memory.multimodal_rate_controlled import MultimodalRateManifestStore

from .full_kv_dataset import FullKVRobotVideoDataset, _as_int


class MultimodalRateControlledRobotVideoDataset(FullKVRobotVideoDataset):
    def __init__(
        self,
        *args,
        multimodal_manifest_path: str,
        manifest_task: str,
        memory_tokens: int = 8,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.multimodal_store = MultimodalRateManifestStore(
            multimodal_manifest_path,
            expected_episode_count=50,
            expected_task=manifest_task,
        )
        self.multimodal_memory_tokens = int(memory_tokens)
        if self.multimodal_memory_tokens != 8:
            raise ValueError("multimodal rate-controlled v2 requires fixed K8")

    def _get(self, idx):
        sample = super()._get(idx)
        history = sample.get("history_latents")
        if not isinstance(history, torch.Tensor) or history.ndim != 5:
            raise RuntimeError("multimodal sample requires rank-5 history_latents")
        current_decision = int(history.shape[0]) - 1
        frame_index = _as_int(sample["frame_index"])
        if frame_index != current_decision * int(self.replan_stride):
            raise RuntimeError("multimodal history/frame index mismatch")
        episode = _as_int(sample["episode_index"])
        closed = tuple(
            row
            for row in self.multimodal_store.segments_for_episode(episode)
            if row.confirmed_at <= current_decision
        )
        sample["history_memory_groups"] = torch.tensor(
            [(row.start, row.end) for row in closed], dtype=torch.int64
        ).reshape(-1, 2)
        sample["history_memory_token_counts"] = torch.full(
            (len(closed),), self.multimodal_memory_tokens, dtype=torch.int64
        )
        sample["history_memory_group_count"] = torch.tensor(
            len(closed), dtype=torch.int64
        )
        return sample
