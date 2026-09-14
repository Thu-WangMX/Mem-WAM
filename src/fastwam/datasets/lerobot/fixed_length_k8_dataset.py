"""Budget-matched fixed-length history segments for dynamic-boundary ablations."""

from __future__ import annotations

import torch

from .full_kv_dataset import FullKVRobotVideoDataset, _as_int


class FixedLengthK8RobotVideoDataset(FullKVRobotVideoDataset):
    """Expose deterministic contiguous history groups with fixed K8 compression.

    ``fixed_segment_length + confirmation_delay == 8`` keeps the look-ahead and
    write schedule matched to the dynamic L4--8 controller.  The Reader,
    compressor, losses, and token allocation are unchanged; only the temporal
    boundaries differ.
    """

    def __init__(
        self,
        *args,
        fixed_segment_length: int = 7,
        memory_tokens: int = 8,
        anchor_frames: int = 2,
        confirmation_delay: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.fixed_segment_length = int(fixed_segment_length)
        self.fixed_memory_tokens = int(memory_tokens)
        self.fixed_anchor_frames = int(anchor_frames)
        self.fixed_confirmation_delay = int(confirmation_delay)
        if not 4 <= self.fixed_segment_length <= 8:
            raise ValueError("fixed segment length must lie in L4..8")
        if self.fixed_memory_tokens != 8:
            raise ValueError("the budget-matched control requires fixed K8")
        if self.fixed_anchor_frames != 2:
            raise ValueError("the budget-matched control requires anchor2")
        if self.fixed_segment_length + self.fixed_confirmation_delay != 8:
            raise ValueError(
                "fixed length plus confirmation delay must equal the L4..8 horizon"
            )

    def _get(self, idx):
        sample = super()._get(idx)
        history = sample.get("history_latents")
        if not isinstance(history, torch.Tensor) or history.ndim != 5:
            raise RuntimeError("fixed-length sample requires rank-5 history_latents")
        current_decision = int(history.shape[0]) - 1
        frame_index = _as_int(sample["frame_index"])
        if frame_index != current_decision * int(self.replan_stride):
            raise RuntimeError("fixed-length history/frame index mismatch")

        groups = []
        start = self.fixed_anchor_frames
        while (
            start + self.fixed_segment_length + self.fixed_confirmation_delay
            <= current_decision
        ):
            end = start + self.fixed_segment_length
            groups.append((start, end))
            start = end

        sample["history_memory_groups"] = torch.tensor(
            groups, dtype=torch.int64
        ).reshape(-1, 2)
        sample["history_memory_token_counts"] = torch.full(
            (len(groups),), self.fixed_memory_tokens, dtype=torch.int64
        )
        sample["history_memory_group_count"] = torch.tensor(
            len(groups), dtype=torch.int64
        )
        return sample
