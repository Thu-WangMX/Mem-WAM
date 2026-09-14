import torch

from fastwam.datasets.lerobot.dynamic_surprise_dataset import (
    attach_dynamic_surprise_groups,
)
from fastwam.memory.dynamic_surprise import ManifestSegment


class _Store:
    def segment_records_for_episode(self, episode_index):
        assert episode_index == 7
        return (
            ManifestSegment(0, 4, "segment_relative_control_information"),
            ManifestSegment(4, 8, "forced_maximum"),
            ManifestSegment(8, 11, "end"),
        )


def test_attach_groups_excludes_boundary_observation():
    sample = {
        "episode_index": torch.tensor(7),
        "frame_index": torch.tensor(64),
        "history_latents": torch.zeros(5, 48, 1, 24, 20),
    }

    result = attach_dynamic_surprise_groups(sample, _Store(), replan_stride=16)

    assert result is sample
    assert torch.equal(result["history_memory_groups"], torch.tensor([[0, 4]]))
    assert torch.equal(result["history_memory_token_counts"], torch.tensor([32]))
    assert int(result["history_memory_group_count"]) == 1


def test_attach_groups_emits_stable_empty_shape_before_first_boundary():
    sample = {
        "episode_index": 7,
        "frame_index": 48,
        "history_latents": torch.zeros(4, 48, 1, 24, 20),
    }

    result = attach_dynamic_surprise_groups(sample, _Store(), replan_stride=16)

    assert result["history_memory_groups"].shape == (0, 2)
    assert result["history_memory_groups"].dtype == torch.int64
    assert result["history_memory_token_counts"].shape == (0,)
    assert int(result["history_memory_group_count"]) == 0


def test_attach_groups_rejects_history_frame_misalignment():
    sample = {
        "episode_index": 7,
        "frame_index": 80,
        "history_latents": torch.zeros(4, 48, 1, 24, 20),
    }

    try:
        attach_dynamic_surprise_groups(sample, _Store(), replan_stride=16)
    except RuntimeError as exc:
        assert "history/frame mismatch" in str(exc)
    else:
        raise AssertionError("misaligned sample was accepted")


def test_attach_groups_allocates_half_rate_to_forced_closes():
    sample = {
        "episode_index": 7,
        "frame_index": 128,
        "history_latents": torch.zeros(9, 48, 1, 24, 20),
    }
    result = attach_dynamic_surprise_groups(sample, _Store(), replan_stride=16)
    assert torch.equal(result["history_memory_token_counts"], torch.tensor([32, 16]))
