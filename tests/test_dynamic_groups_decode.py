import pytest
import torch

from fastwam.models.wan22.fastwam import (
    _decode_dynamic_memory_groups,
    _decode_dynamic_memory_plan,
)


def test_decode_collated_dynamic_groups():
    sample = {
        "history_memory_groups": torch.tensor([[[0, 3], [3, 8]]]),
        "history_memory_group_count": torch.tensor([2]),
    }
    assert _decode_dynamic_memory_groups(sample) == ((0, 1, 2), (3, 4, 5, 6, 7))


def test_decode_explicit_event_conditioned_token_counts():
    sample = {
        "history_memory_groups": torch.tensor([[[0, 3], [3, 8]]]),
        "history_memory_group_count": torch.tensor([2]),
        "history_memory_token_counts": torch.tensor([[24, 24]]),
    }
    assert _decode_dynamic_memory_plan(sample) == (
        ((0, 1, 2), (3, 4, 5, 6, 7)),
        (24, 24),
    )


def test_decode_empty_groups_and_absent_metadata_are_distinct():
    assert _decode_dynamic_memory_groups({}) is None
    sample = {
        "history_memory_groups": torch.empty((1, 0, 2), dtype=torch.int64),
        "history_memory_group_count": torch.tensor([0]),
    }
    assert _decode_dynamic_memory_groups(sample) == ()


def test_decode_rejects_batched_or_inconsistent_metadata():
    with pytest.raises(ValueError, match="batch size 1"):
        _decode_dynamic_memory_groups(
            {
                "history_memory_groups": torch.zeros((2, 1, 2), dtype=torch.int64),
                "history_memory_group_count": torch.ones(2, dtype=torch.int64),
            }
        )
    with pytest.raises(ValueError, match="count"):
        _decode_dynamic_memory_groups(
            {
                "history_memory_groups": torch.tensor([[[0, 3]]]),
                "history_memory_group_count": torch.tensor([0]),
            }
        )
