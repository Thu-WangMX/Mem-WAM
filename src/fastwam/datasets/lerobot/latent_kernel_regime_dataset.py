"""PutBack FullKV samples with frozen train-free latent-kernel groups."""

from __future__ import annotations

import torch

from fastwam.memory.latent_kernel_regime import LatentKernelManifestStore

from .full_kv_dataset import FullKVRobotVideoDataset, _as_int


class LatentKernelRegimeRobotVideoDataset(FullKVRobotVideoDataset):
    def __init__(
        self,
        *args,
        latent_kernel_manifest_path: str,
        manifest_task: str = "put_back_block",
        memory_tokens: int = 8,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.latent_kernel_store = LatentKernelManifestStore(
            latent_kernel_manifest_path,
            expected_episode_count=50,
            expected_task=manifest_task,
        )
        self.latent_kernel_memory_tokens = int(memory_tokens)
        if self.latent_kernel_memory_tokens != 8:
            raise ValueError("multi-resolution latent-kernel v1 requires fixed K8")

    def _get(self, idx):
        sample = super()._get(idx)
        history = sample.get("history_latents")
        if not isinstance(history, torch.Tensor) or history.ndim != 5:
            raise RuntimeError("latent-kernel sample requires rank-5 history_latents")
        current_decision = int(history.shape[0]) - 1
        frame_index = _as_int(sample["frame_index"])
        if frame_index != current_decision * int(self.replan_stride):
            raise RuntimeError("latent-kernel history/frame index mismatch")
        episode = _as_int(sample["episode_index"])
        closed = tuple(
            row
            for row in self.latent_kernel_store.segments_for_episode(episode)
            if row.confirmed_at <= current_decision
        )
        sample["history_memory_groups"] = torch.tensor(
            [(row.start, row.end) for row in closed], dtype=torch.int64
        ).reshape(-1, 2)
        sample["history_memory_token_counts"] = torch.full(
            (len(closed),), self.latent_kernel_memory_tokens, dtype=torch.int64
        )
        sample["history_memory_group_count"] = torch.tensor(
            len(closed), dtype=torch.int64
        )
        return sample
