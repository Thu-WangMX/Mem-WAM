import torch

from fastwam.datasets.lerobot.full_kv_dataset import FullKVRobotVideoDataset
from fastwam.datasets.lerobot.wrist_event_dataset import WristEventRobotVideoDataset


class _Store:
    def segments_for_episode(self, episode_index):
        assert episode_index == 3
        return ((0, 5), (5, 9), (9, 13))


def test_dataset_attaches_only_segments_closed_before_current_observation(monkeypatch):
    sample = {
        "episode_index": torch.tensor(3),
        "frame_index": torch.tensor(9 * 16),
        "history_latents": torch.zeros(10, 48, 1, 24, 20),
    }
    monkeypatch.setattr(FullKVRobotVideoDataset, "_get", lambda self, index: sample)
    dataset = WristEventRobotVideoDataset.__new__(WristEventRobotVideoDataset)
    dataset.wrist_event_manifest_store = _Store()
    dataset.replan_stride = 16

    result = dataset._get(0)

    assert result is sample
    assert torch.equal(result["history_memory_groups"], torch.tensor([[0, 5], [5, 9]]))
    assert int(result["history_memory_group_count"]) == 2
