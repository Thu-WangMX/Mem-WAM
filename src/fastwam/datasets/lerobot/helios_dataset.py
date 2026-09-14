"""The existing RMBench data pipeline with a fixed-size Helios history window."""
import torch
from .full_kv_dataset import FullKVRobotVideoDataset


from fastwam.memory.helios_history import episode_anchors, fixed_history


class HeliosRobotVideoDataset(FullKVRobotVideoDataset):
    def __init__(self, *args, history_sizes=(16, 2, 1), anchor_frames=2, **kwargs):
        self.history_sizes = tuple(int(x) for x in history_sizes)
        self.anchor_frames = int(anchor_frames)
        if len(self.history_sizes) != 3 or self.history_sizes[-1] != 1:
            raise ValueError("Helios requires long/mid/current sizes with current=1")
        if self.anchor_frames not in (0, 2):
            raise ValueError("Helios supports either zero (ablation) or two episode anchors")
        super().__init__(*args, **kwargs)

    def _get(self, idx):
        sample = super()._get(idx)
        sample['anchor_latents'], sample['anchor_valid'] = episode_anchors(
            sample['history_latents'], self.anchor_frames)
        sample['history_latents'], sample['history_valid'] = fixed_history(
            sample['history_latents'], sum(self.history_sizes))
        return sample
