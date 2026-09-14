import pytest
import torch
import torch.nn as nn

from fastwam.memory.native_cache import (
    LayerwiseBlockMemory,
    build_dynamic_layerwise_training_layout,
)


class _TinyVideoExpert(nn.Module):
    hidden_dim = 12

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.freqs = (
            torch.ones(32, 2),
            torch.ones(1, 2),
            torch.ones(1, 2),
        )


def test_k8l_assigns_eight_tokens_per_source_frame_up_to_six_frames():
    memory = LayerwiseBlockMemory(
        _TinyVideoExpert(),
        memory_tokens=48,
        dynamic_tokens_per_frame=8,
    )

    assert [memory.token_count_for_span(span) for span in range(2, 7)] == [
        16,
        24,
        32,
        40,
        48,
    ]
    assert memory.initial_tokens(
        batch_size=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        token_count=24,
    ).shape == (2, 24, 12)
    with pytest.raises(ValueError, match="exceeds configured memory capacity"):
        memory.token_count_for_span(7)


def test_k8l_training_layout_uses_per_group_token_counts():
    layout = build_dynamic_layerwise_training_layout(
        clean_frames=7,
        noisy_frames=1,
        tokens_per_frame=2,
        action_tokens=3,
        memory_groups=((2, 3), (4, 5)),
        memory_tokens=48,
        memory_token_counts=(16, 32),
        anchor_frames=2,
        recent_frames=4,
        device=torch.device("cpu"),
    )

    assert [
        segment.stop - segment.start
        for segment in layout.segments
        if segment.kind == "memory"
    ] == [16, 32]
    # Anchor2 + K16 + K32 + the one-frame open raw tail.  Frames in already
    # compressed groups are not duplicated merely because they were recent.
    assert layout.retained_clean_tokens == 54
