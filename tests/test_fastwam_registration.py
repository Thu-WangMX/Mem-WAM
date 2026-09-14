import torch
import torch.nn as nn

from fastwam.models.wan22.fastwam import FastWAM
from fastwam.models.wan22.wan_video_dit import DiTBlock


class TinyMoT(nn.Module):
    def __init__(self, video: nn.Module, action: nn.Module):
        super().__init__()
        self.mixtures = nn.ModuleDict({"video": video, "action": action})


class TinyVideo(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = 48
        self.freqs = (
            torch.ones(16, 2, dtype=torch.complex128),
            torch.ones(16, 2, dtype=torch.complex128),
            torch.ones(16, 2, dtype=torch.complex128),
        )
        self.blocks = nn.ModuleList(
            [DiTBlock(48, 12, 4, 96, eps=1.0e-6)]
        )


def test_experts_are_registered_once_through_mot():
    video = nn.Linear(3, 4)
    action = nn.Linear(4, 2)
    mot = TinyMoT(video, action)
    model = FastWAM(
        video_expert=video,
        action_expert=action,
        mot=mot,
        vae=nn.Identity(),
        text_dim=8,
    )

    keys = tuple(model.state_dict())
    assert model.video_expert is video
    assert model.action_expert is action
    assert model.dit is mot
    assert any(key.startswith("mot.mixtures.video.") for key in keys)
    assert any(key.startswith("mot.mixtures.action.") for key in keys)
    assert not any(key.startswith("video_expert.") for key in keys)
    assert not any(key.startswith("action_expert.") for key in keys)
    assert not any(key.startswith("dit.") for key in keys)


def test_native_compressor_is_registered_once_outside_mot():
    video = TinyVideo()
    action = nn.Linear(48, 48)
    mot = TinyMoT(video, action)
    model = FastWAM(
        video_expert=video,
        action_expert=action,
        mot=mot,
        vae=nn.Identity(),
        text_dim=8,
        native_cache={"enabled": True, "group_size": 4},
    )

    keys = tuple(model.state_dict())
    assert model.native_cache_compressor is not None
    assert any(key.startswith("native_cache_compressor.") for key in keys)
    assert not any(
        key.startswith("mot.native_cache_compressor.") for key in keys
    )


def test_layerwise_memory_slots_are_registered_once_outside_mot():
    """Catches accidental shallow-compressor construction for layerwise mode."""
    video = TinyVideo()
    action = nn.Linear(48, 48)
    mot = TinyMoT(video, action)
    model = FastWAM(
        video_expert=video,
        action_expert=action,
        mot=mot,
        vae=nn.Identity(),
        text_dim=8,
        native_cache={
            "enabled": True,
            "mode": "layerwise",
            "memory_tokens": 32,
            "group_size": 4,
            "anchor_frames": 2,
            "recent_frames": 4,
            "recursive": False,
        },
    )

    keys = tuple(model.state_dict())
    assert model.native_cache_compressor is None
    assert model.layerwise_block_memory is not None
    assert model.layerwise_block_memory.memory_tokens == 32
    assert model.layerwise_block_memory.group_size == 4
    assert model.layerwise_block_memory.anchor_frames == 2
    assert model.layerwise_block_memory.recent_frames == 4
    assert model.layerwise_block_memory.recursive is False
    assert "layerwise_block_memory.slots" in keys
    assert not any(key.startswith("mot.layerwise_block_memory.") for key in keys)


def test_layerwise_memory_checkpoint_round_trip(tmp_path):
    """Catches formal checkpoints that silently omit the learned 32 slots."""
    video = TinyVideo()
    video.video_attention_mask_mode = "first_frame_causal"
    action = nn.Linear(48, 48)
    mot = TinyMoT(video, action)
    model = FastWAM(
        video_expert=video,
        action_expert=action,
        mot=mot,
        vae=nn.Identity(),
        text_dim=8,
        native_cache={"enabled": True, "mode": "layerwise", "memory_tokens": 32},
    )
    expected = model.layerwise_block_memory.slots.detach().clone()
    checkpoint = tmp_path / "layerwise.pt"
    model.save_checkpoint(checkpoint, step=1000)
    with torch.no_grad():
        model.layerwise_block_memory.slots.zero_()

    model.load_checkpoint(checkpoint)

    assert torch.equal(model.layerwise_block_memory.slots, expected)
