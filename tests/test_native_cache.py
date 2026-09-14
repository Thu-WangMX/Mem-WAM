import pytest
import torch
import torch.nn as nn

from fastwam.memory.native_cache import (
    CacheUnit,
    LayerwiseBlockMemory,
    LayerwiseMemoryState,
    NativeBlock,
    NativeBlockCompressor,
    NativeCacheState,
    build_layerwise_training_layout,
    partition_layerwise_history,
    summarize_native_cache_state,
)
from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.fastwam import FastWAM
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.wan_video_dit import DiTBlock, WanVideoDiT
from fastwam.trainer import Wan22Trainer


def _tiny_block() -> DiTBlock:
    return DiTBlock(
        hidden_dim=48,
        attn_head_dim=12,
        num_heads=4,
        ffn_dim=96,
        eps=1.0e-6,
    )


def _tiny_video_expert(num_layers=2):
    return WanVideoDiT(
        hidden_dim=48,
        in_dim=4,
        ffn_dim=96,
        out_dim=4,
        text_dim=16,
        freq_dim=8,
        eps=1.0e-6,
        patch_size=(1, 2, 2),
        num_heads=4,
        attn_head_dim=12,
        num_layers=num_layers,
        has_image_input=False,
        seperated_timestep=True,
        require_vae_embedding=False,
        fuse_vae_embedding_in_latents=True,
        action_conditioned=False,
    )


@pytest.mark.parametrize(
    ("frame_count", "anchors", "memory_groups", "raw_tail", "recent"),
    [
        (1, (0,), (), (), ()),
        (2, (0, 1), (), (), ()),
        (3, (0, 1), (), (2,), (2,)),
        (4, (0, 1), (), (2, 3), (2, 3)),
        (5, (0, 1), (), (2, 3, 4), (2, 3, 4)),
        (6, (0, 1), ((2, 3, 4, 5),), (), (2, 3, 4, 5)),
        (7, (0, 1), ((2, 3, 4, 5),), (6,), (3, 4, 5, 6)),
        (8, (0, 1), ((2, 3, 4, 5),), (6, 7), (4, 5, 6, 7)),
        (9, (0, 1), ((2, 3, 4, 5),), (6, 7, 8), (5, 6, 7, 8)),
        (
            10,
            (0, 1),
            ((2, 3, 4, 5), (6, 7, 8, 9)),
            (),
            (6, 7, 8, 9),
        ),
        (
            11,
            (0, 1),
            ((2, 3, 4, 5), (6, 7, 8, 9)),
            (10,),
            (7, 8, 9, 10),
        ),
        (
            12,
            (0, 1),
            ((2, 3, 4, 5), (6, 7, 8, 9)),
            (10, 11),
            (8, 9, 10, 11),
        ),
        (
            13,
            (0, 1),
            ((2, 3, 4, 5), (6, 7, 8, 9)),
            (10, 11, 12),
            (9, 10, 11, 12),
        ),
        (
            14,
            (0, 1),
            ((2, 3, 4, 5), (6, 7, 8, 9), (10, 11, 12, 13)),
            (),
            (10, 11, 12, 13),
        ),
        (
            15,
            (0, 1),
            ((2, 3, 4, 5), (6, 7, 8, 9), (10, 11, 12, 13)),
            (14,),
            (11, 12, 13, 14),
        ),
        (
            16,
            (0, 1),
            ((2, 3, 4, 5), (6, 7, 8, 9), (10, 11, 12, 13)),
            (14, 15),
            (12, 13, 14, 15),
        ),
    ],
)
def test_layerwise_partition_preserves_two_anchors_and_four_recent_frames(
    frame_count, anchors, memory_groups, raw_tail, recent
):
    """Catches early compression of anchors/recent frames or skipped history."""
    partition = partition_layerwise_history(frame_count)

    assert partition.anchors == anchors
    assert partition.memory_groups == memory_groups
    assert partition.raw_tail == raw_tail
    assert partition.recent == recent


def test_layerwise_state_counts_32_token_memory_in_every_cache_layer():
    """Catches states that account for a block memory as a 120-token frame."""
    units = (
        CacheUnit(kind="anchor", endpoint=0, span=1, token_count=120),
        CacheUnit(kind="anchor", endpoint=1, span=1, token_count=120),
        CacheUnit(kind="memory", endpoint=5, span=4, token_count=32),
        CacheUnit(kind="recent", endpoint=6, span=1, token_count=120),
    )
    cache = tuple(
        {
            "k": torch.zeros(1, 392, 4, 12),
            "v": torch.zeros(1, 392, 4, 12),
        }
        for _ in range(2)
    )

    state = LayerwiseMemoryState(units=units, kv_cache=cache)

    assert state.retained_tokens == 392
    assert state.represented_frames == 7


def test_layerwise_state_rejects_cache_length_that_disagrees_with_units():
    """Catches silent metadata/KV drift after online cache replacement."""
    units = (CacheUnit(kind="memory", endpoint=5, span=4, token_count=32),)
    cache = ({"k": torch.zeros(1, 31, 4, 12), "v": torch.zeros(1, 31, 4, 12)},)

    with pytest.raises(ValueError, match="expected 32"):
        LayerwiseMemoryState(units=units, kv_cache=cache)


def test_32_memory_slots_append_layerwise_kv_through_native_video_blocks():
    """Catches a shallow compressor that creates only one pre-DiT summary."""
    torch.manual_seed(101)
    video = _tiny_video_expert(num_layers=2)
    action = ActionDiT(
        hidden_dim=32,
        action_dim=3,
        ffn_dim=64,
        text_dim=16,
        freq_dim=8,
        eps=1.0e-6,
        num_heads=4,
        attn_head_dim=12,
        num_layers=2,
    )
    mot = MoT(
        mixtures={"video": video, "action": action},
        mot_checkpoint_mixed_attn=False,
    )
    memory = LayerwiseBlockMemory(video, memory_tokens=32)
    context = torch.randn(1, 2, 16)
    context_mask = torch.ones(1, 2, dtype=torch.bool)
    raw_pre = video.pre_dit(
        x=torch.randn(1, 4, 4, 4, 4),
        timestep=torch.zeros(1, 4),
        context=context,
        context_mask=context_mask,
        fuse_vae_embedding_in_latents=True,
    )
    raw_len = raw_pre["tokens"].shape[1]
    raw_cache = mot.prefill_video_cache(
        video_tokens=raw_pre["tokens"],
        video_freqs=raw_pre["freqs"],
        video_t_mod=raw_pre["t_mod"],
        video_context_payload={
            "context": raw_pre["context"],
            "mask": raw_pre["context_mask"],
        },
        video_attention_mask=torch.ones(raw_len, raw_len, dtype=torch.bool),
    )
    slots = memory.initial_tokens(
        batch_size=1, device=raw_pre["tokens"].device, dtype=raw_pre["tokens"].dtype
    )
    slot_freqs = memory.build_freqs(endpoint=3, device=slots.device)
    slot_t_mod = memory.build_t_mod(raw_pre["t_mod"][:, -1:])
    slot_context_mask = raw_pre["context_mask"][:, -1:].expand(-1, 32, -1)

    memory_cache, memory_hidden = mot.prefill_video_cache(
        video_tokens=slots,
        video_freqs=slot_freqs,
        video_t_mod=slot_t_mod,
        video_context_payload={"context": raw_pre["context"], "mask": slot_context_mask},
        video_attention_mask=torch.ones(32, raw_len + 32, dtype=torch.bool),
        history_kv_cache=raw_cache,
        return_current_tokens=True,
    )

    assert memory_hidden.shape == (1, 32, 48)
    assert len(memory_cache) == 2
    assert all(layer["k"].shape[1] == raw_len + 32 for layer in memory_cache)
    assert all(layer["v"].shape[1] == raw_len + 32 for layer in memory_cache)


def test_memory_only_loss_backpropagates_to_each_of_four_source_frames():
    """Catches masks or slot paths that ignore any source frame."""
    torch.manual_seed(103)
    video = _tiny_video_expert(num_layers=2)
    action = ActionDiT(
        hidden_dim=32,
        action_dim=3,
        ffn_dim=64,
        text_dim=16,
        freq_dim=8,
        eps=1.0e-6,
        num_heads=4,
        attn_head_dim=12,
        num_layers=2,
    )
    mot = MoT(
        mixtures={"video": video, "action": action},
        mot_checkpoint_mixed_attn=False,
    )
    memory = LayerwiseBlockMemory(video, memory_tokens=32)
    context = torch.randn(1, 2, 16)
    context_mask = torch.ones(1, 2, dtype=torch.bool)
    source = torch.randn(1, 4, 4, 4, 4, requires_grad=True)
    raw_pre = video.pre_dit(
        x=source,
        timestep=torch.zeros(1, 4),
        context=context,
        context_mask=context_mask,
        fuse_vae_embedding_in_latents=True,
    )
    raw_len = raw_pre["tokens"].shape[1]
    raw_cache = mot.prefill_video_cache(
        video_tokens=raw_pre["tokens"],
        video_freqs=raw_pre["freqs"],
        video_t_mod=raw_pre["t_mod"],
        video_context_payload={
            "context": raw_pre["context"],
            "mask": raw_pre["context_mask"],
        },
        video_attention_mask=torch.ones(raw_len, raw_len, dtype=torch.bool),
    )
    slots = memory.initial_tokens(batch_size=1, device=source.device, dtype=source.dtype)
    _, memory_hidden = mot.prefill_video_cache(
        video_tokens=slots,
        video_freqs=memory.build_freqs(endpoint=3, device=source.device),
        video_t_mod=memory.build_t_mod(raw_pre["t_mod"][:, -1:]),
        video_context_payload={
            "context": raw_pre["context"],
            "mask": raw_pre["context_mask"][:, -1:].expand(-1, 32, -1),
        },
        video_attention_mask=torch.ones(32, raw_len + 32, dtype=torch.bool),
        history_kv_cache=raw_cache,
        return_current_tokens=True,
    )
    memory_hidden.square().mean().backward()

    assert source.grad is not None
    assert [source.grad[:, :, frame].abs().sum().item() > 0 for frame in range(4)] == [
        True,
        True,
        True,
        True,
    ]


def test_training_layout_inserts_memory_after_source_and_keeps_recent_visible():
    """Catches uniform-frame packing that treats a 32-token memory as 120 tokens."""
    layout = build_layerwise_training_layout(
        clean_frames=10,
        noisy_frames=2,
        tokens_per_frame=3,
        action_tokens=5,
        memory_tokens=2,
        device=torch.device("cpu"),
    )

    assert [
        (segment.kind, segment.frame_indices, segment.start, segment.stop)
        for segment in layout.segments
    ] == [
        ("anchor", (0,), 0, 3),
        ("anchor", (1,), 3, 6),
        ("source", (2, 3, 4, 5), 6, 18),
        ("memory", (2, 3, 4, 5), 18, 20),
        ("source", (6, 7, 8, 9), 20, 32),
        ("memory", (6, 7, 8, 9), 32, 34),
    ]
    assert layout.noisy_range == (34, 40)
    assert layout.action_range == (40, 45)
    assert layout.retained_clean_tokens == 22


def test_training_mask_blocks_future_and_action_from_evicted_source_tokens():
    """Catches a training shortcut where later tokens bypass block memory."""
    layout = build_layerwise_training_layout(
        clean_frames=10,
        noisy_frames=2,
        tokens_per_frame=3,
        action_tokens=5,
        memory_tokens=2,
        device=torch.device("cpu"),
    )
    mask = layout.attention_mask

    # Memory sees both anchors, all four source frames, and its peer slot.
    assert mask[18:20, 0:20].all()
    # Frame 6 sees raw recent frames 3..5, but frame 2 has already slid out.
    assert not mask[20:23, 6:9].any()
    assert mask[20:23, 9:18].all()
    # Frame 9 sees its own four-frame raw group and the previous memory.
    assert not mask[29:32, 6:18].any()
    assert mask[29:32, 18:32].all()
    # Noisy video sees retained clean cache and noisy peers, never source raw KV.
    assert mask[34:40, 0:6].all()
    assert not mask[34:40, 6:18].any()
    assert mask[34:40, 18:40].all()
    # Action sees retained clean cache plus action peers, but not source/noisy video.
    assert mask[40:45, 0:6].all()
    assert not mask[40:45, 6:18].any()
    assert mask[40:45, 18:34].all()
    assert not mask[40:45, 34:40].any()
    assert mask[40:45, 40:45].all()


def test_fastwam_packer_inserts_slots_without_decoding_source_or_memory_tokens():
    """Catches wrong noisy-output offsets after heterogeneous token packing."""
    tokens_per_frame = 2
    clean_frames = 10
    noisy_frames = 1
    total_frames = clean_frames + noisy_frames
    hidden_dim = 48
    frame_values = torch.arange(total_frames, dtype=torch.float32)
    video_pre = {
        "tokens": frame_values.view(1, total_frames, 1, 1)
        .expand(1, total_frames, tokens_per_frame, hidden_dim)
        .reshape(1, total_frames * tokens_per_frame, hidden_dim)
        .clone(),
        "freqs": torch.ones(
            total_frames * tokens_per_frame, 1, 6, dtype=torch.complex128
        ),
        "t_mod": torch.zeros(1, total_frames * tokens_per_frame, 6, hidden_dim),
        "context_mask": torch.ones(
            1, total_frames * tokens_per_frame, 3, dtype=torch.bool
        ),
        "meta": {
            "grid_size": (total_frames, 1, tokens_per_frame),
            "tokens_per_frame": tokens_per_frame,
            "batch_size": 1,
        },
    }
    model = object.__new__(FastWAM)
    nn.Module.__init__(model)
    video = _tiny_video_expert(num_layers=2)
    model.layerwise_block_memory = LayerwiseBlockMemory(video, memory_tokens=2)
    with torch.no_grad():
        model.layerwise_block_memory.slots.fill_(99.0)

    packed, layout = model._pack_layerwise_memory_training_state(
        video_pre,
        clean_frame_count=clean_frames,
        noisy_frame_count=noisy_frames,
        action_seq_len=3,
    )

    assert packed["tokens"].shape == (1, 26, hidden_dim)
    assert packed["tokens"][0, :, 0].tolist() == [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
        4,
        4,
        5,
        5,
        99,
        99,
        6,
        6,
        7,
        7,
        8,
        8,
        9,
        9,
        99,
        99,
        10,
        10,
    ]
    assert layout.noisy_range == (24, 26)
    assert packed["freqs"].shape[0] == 26
    assert packed["t_mod"].shape[1] == 26
    assert packed["context_mask"].shape[1] == 26


def test_layerwise_training_transformer_routes_future_loss_through_memory_slots():
    """Catches training paths that still invoke the legacy shallow compressor."""
    torch.manual_seed(107)
    video = _tiny_video_expert(num_layers=2)
    action = ActionDiT(
        hidden_dim=32,
        action_dim=3,
        ffn_dim=64,
        text_dim=16,
        freq_dim=8,
        eps=1.0e-6,
        num_heads=4,
        attn_head_dim=12,
        num_layers=2,
    )
    mot = MoT(
        mixtures={"video": video, "action": action},
        mot_checkpoint_mixed_attn=False,
    )
    model = object.__new__(FastWAM)
    nn.Module.__init__(model)
    model.mot = mot
    model.layerwise_block_memory = LayerwiseBlockMemory(video, memory_tokens=2)
    context = torch.randn(1, 3, 16)
    context_mask = torch.ones(1, 3, dtype=torch.bool)
    source = torch.randn(1, 4, 11, 4, 4, requires_grad=True)
    video_pre = video.pre_dit(
        x=source,
        timestep=torch.zeros(1, 11),
        context=context,
        context_mask=context_mask,
        fuse_vae_embedding_in_latents=True,
    )
    action_pre = action.pre_dit(
        action_tokens=torch.randn(1, 5, 3),
        timestep=torch.zeros(1),
        context=context,
        context_mask=context_mask,
    )

    tokens_out, layout = model._run_layerwise_memory_training_transformer(
        video_pre=video_pre,
        action_pre=action_pre,
        clean_frame_count=10,
        noisy_frame_count=1,
    )
    noisy_start, noisy_stop = layout.noisy_range
    loss = (
        tokens_out["video"][:, noisy_start:noisy_stop].square().mean()
        + tokens_out["action"].square().mean()
    )
    loss.backward()

    assert tokens_out["video"].shape[1] == noisy_stop
    assert model.layerwise_block_memory.slots.grad is not None
    assert model.layerwise_block_memory.slots.grad.abs().sum().item() > 0
    assert source.grad is not None
    assert source.grad[:, :, 2:6].abs().sum().item() > 0


def test_full_training_loss_uses_layerwise_memory_and_backpropagates():
    """Catches the formal training entrypoint silently taking the FullKV branch."""
    torch.manual_seed(109)
    video = _tiny_video_expert(num_layers=2)
    action = ActionDiT(
        hidden_dim=32,
        action_dim=3,
        ffn_dim=64,
        text_dim=16,
        freq_dim=8,
        eps=1.0e-6,
        num_heads=4,
        attn_head_dim=12,
        num_layers=2,
    )
    mot = MoT(
        mixtures={"video": video, "action": action},
        mot_checkpoint_mixed_attn=False,
    )
    model = FastWAM(
        video_expert=video,
        action_expert=action,
        mot=mot,
        vae=nn.Identity(),
        text_dim=16,
        native_cache={"enabled": True, "mode": "layerwise", "memory_tokens": 2},
    )
    context = torch.randn(1, 3, 16)
    inputs = {
        "history_latents": torch.randn(1, 10, 4, 1, 4, 4),
        "input_latents": torch.randn(1, 4, 2, 4, 4),
        "context": context,
        "context_mask": torch.ones(1, 3, dtype=torch.bool),
        "video_context": context,
        "video_context_mask": torch.ones(1, 3, dtype=torch.bool),
        "action": torch.randn(1, 5, 3),
        "action_is_pad": None,
        "image_is_pad": None,
    }
    model.build_inputs = lambda sample, tiled=False: inputs

    loss, metrics = model._training_loss_full_kv({"history_latents": True})
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["history_frames"] == 10.0
    assert metrics["native_retained_history_units"] == 8.0
    assert metrics["full_kv_video_tokens"] == 28.0
    assert model.layerwise_block_memory.slots.grad is not None
    assert model.layerwise_block_memory.slots.grad.abs().sum().item() > 0


def test_online_layerwise_cache_keeps_anchors_memories_and_exact_recent_four():
    """Catches delayed block creation or eviction of anchor/recent FullKV."""
    torch.manual_seed(113)
    video = _tiny_video_expert(num_layers=2)
    action = ActionDiT(
        hidden_dim=32,
        action_dim=3,
        ffn_dim=64,
        text_dim=16,
        freq_dim=8,
        eps=1.0e-6,
        num_heads=4,
        attn_head_dim=12,
        num_layers=2,
    )
    mot = MoT(
        mixtures={"video": video, "action": action},
        mot_checkpoint_mixed_attn=False,
    ).eval()
    model = object.__new__(FastWAM)
    nn.Module.__init__(model)
    model.mot = mot
    model.layerwise_block_memory = LayerwiseBlockMemory(video, memory_tokens=2)
    context = torch.randn(1, 2, 16)
    context_mask = torch.ones(1, 2, dtype=torch.bool)
    state = None

    for endpoint in range(11):
        pre = video.pre_dit(
            x=torch.randn(1, 4, 1, 4, 4),
            timestep=torch.zeros(1),
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=True,
            temporal_position_offset=endpoint,
        )
        history_cache = None if state is None else list(state.kv_cache)
        history_len = 0 if state is None else state.retained_tokens
        current_len = pre["tokens"].shape[1]
        current_cache = mot.prefill_video_cache(
            video_tokens=pre["tokens"],
            video_freqs=pre["freqs"],
            video_t_mod=pre["t_mod"],
            video_context_payload={
                "context": pre["context"],
                "mask": pre["context_mask"],
            },
            video_attention_mask=model._build_layerwise_online_video_mask(
                previous_state=state,
                endpoint=endpoint,
                current_token_count=current_len,
                device=pre["tokens"].device,
            ),
            history_kv_cache=history_cache,
        )
        state = model._commit_layerwise_memory_state(
            previous_state=state,
            current_pre=pre,
            current_cache=current_cache,
            endpoint=endpoint,
        )

        raw_endpoints = [
            unit.endpoint for unit in state.units if unit.kind != "memory"
        ]
        memory_endpoints = [
            unit.endpoint for unit in state.units if unit.kind == "memory"
        ]
        expected_recent_start = max(2, endpoint - 3)
        assert raw_endpoints == [
            *range(min(2, endpoint + 1)),
            *range(expected_recent_start, endpoint + 1),
        ]
        assert memory_endpoints == [
            group_endpoint
            for group_endpoint in (5, 9)
            if group_endpoint <= endpoint
        ]
        expected_tokens = len(raw_endpoints) * current_len + len(memory_endpoints) * 2
        assert state.retained_tokens == expected_tokens
        assert all(layer["k"].shape[1] == expected_tokens for layer in state.kv_cache)

    assert state.represented_frames == 11


def test_packed_training_and_online_build_identical_retained_layerwise_kv():
    """Catches any numerical train/inference mismatch in memory formation."""
    torch.manual_seed(127)
    video = _tiny_video_expert(num_layers=2)
    action = ActionDiT(
        hidden_dim=32,
        action_dim=3,
        ffn_dim=64,
        text_dim=16,
        freq_dim=8,
        eps=1.0e-6,
        num_heads=4,
        attn_head_dim=12,
        num_layers=2,
    )
    mot = MoT(
        mixtures={"video": video, "action": action},
        mot_checkpoint_mixed_attn=False,
    ).eval()
    model = object.__new__(FastWAM)
    nn.Module.__init__(model)
    model.mot = mot
    model.layerwise_block_memory = LayerwiseBlockMemory(video, memory_tokens=2)
    context = torch.randn(1, 2, 16)
    context_mask = torch.ones(1, 2, dtype=torch.bool)
    observations = torch.randn(1, 4, 10, 4, 4)
    action_pre = action.pre_dit(
        action_tokens=torch.randn(1, 3, 3),
        timestep=torch.zeros(1),
        context=context,
        context_mask=context_mask,
    )
    noisy_observation = torch.randn(1, 4, 1, 4, 4)
    full_input = torch.cat([observations, noisy_observation], dim=2)
    full_pre = video.pre_dit(
        x=full_input,
        timestep=torch.zeros(1, 11),
        context=context,
        context_mask=context_mask,
        fuse_vae_embedding_in_latents=True,
    )
    packed, layout = model._pack_layerwise_memory_training_state(
        full_pre,
        clean_frame_count=10,
        noisy_frame_count=1,
        action_seq_len=3,
    )
    _, packed_cache = mot(
        embeds_all={"video": packed["tokens"], "action": action_pre["tokens"]},
        attention_mask=layout.attention_mask,
        freqs_all={"video": packed["freqs"], "action": action_pre["freqs"]},
        context_all={
            "video": {"context": packed["context"], "mask": packed["context_mask"]},
            "action": {
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
        },
        t_mod_all={"video": packed["t_mod"], "action": action_pre["t_mod"]},
        return_video_kv_cache=True,
    )
    packed_retained = tuple(
        {
            "k": torch.cat(
                [layer["k"][:, start:stop] for start, stop in layout.retained_ranges],
                dim=1,
            ),
            "v": torch.cat(
                [layer["v"][:, start:stop] for start, stop in layout.retained_ranges],
                dim=1,
            ),
        }
        for layer in packed_cache
    )

    state = None
    for endpoint in range(10):
        pre = video.pre_dit(
            x=observations[:, :, endpoint : endpoint + 1],
            timestep=torch.zeros(1),
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=True,
            temporal_position_offset=endpoint,
        )
        current_len = pre["tokens"].shape[1]
        cache = mot.prefill_video_cache(
            video_tokens=pre["tokens"],
            video_freqs=pre["freqs"],
            video_t_mod=pre["t_mod"],
            video_context_payload={
                "context": pre["context"],
                "mask": pre["context_mask"],
            },
            video_attention_mask=model._build_layerwise_online_video_mask(
                previous_state=state,
                endpoint=endpoint,
                current_token_count=current_len,
                device=pre["tokens"].device,
            ),
            history_kv_cache=None if state is None else list(state.kv_cache),
        )
        state = model._commit_layerwise_memory_state(
            previous_state=state,
            current_pre=pre,
            current_cache=cache,
            endpoint=endpoint,
        )

    for packed_layer, online_layer in zip(packed_retained, state.kv_cache):
        torch.testing.assert_close(packed_layer["k"], online_layer["k"], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(packed_layer["v"], online_layer["v"], rtol=1e-5, atol=1e-6)


def test_zero_gate_is_exact_last_block_identity():
    torch.manual_seed(3)
    compressor = NativeBlockCompressor(_tiny_block(), group_size=4)
    blocks = torch.randn(2, 4, 5, 48)
    freqs = torch.ones(4 * 5, 1, 6, dtype=torch.complex128)

    result = compressor(
        block_tokens=blocks,
        block_freqs=freqs,
        levels=torch.zeros(2, 4, dtype=torch.long),
        spans=torch.ones(2, 4, dtype=torch.long),
    )

    assert result.shape == (2, 5, 48)
    assert torch.equal(result, blocks[:, -1])


def test_gate_parameter_is_fsdp_compatible_one_dimensional_tensor():
    compressor = NativeBlockCompressor(_tiny_block(), group_size=4)
    assert compressor.raw_alpha.shape == (1,)


def test_all_compressor_parameters_inherit_video_block_dtype_for_fsdp():
    block = _tiny_block().to(dtype=torch.bfloat16)
    compressor = NativeBlockCompressor(block, group_size=4)
    assert {parameter.dtype for parameter in compressor.parameters()} == {
        torch.bfloat16
    }


def test_compressor_copies_native_attention_without_aliasing():
    block = _tiny_block()
    compressor = NativeBlockCompressor(block, group_size=4)
    copied = compressor.attention.q.weight.detach().clone()

    with torch.no_grad():
        block.self_attn.q.weight.add_(1.0)

    assert torch.equal(compressor.attention.q.weight, copied)
    assert not torch.equal(compressor.attention.q.weight, block.self_attn.q.weight)


def test_nonzero_gate_reads_early_blocks_and_backpropagates():
    torch.manual_seed(5)
    compressor = NativeBlockCompressor(_tiny_block(), group_size=4)
    with torch.no_grad():
        compressor.raw_alpha.fill_(0.5)
    blocks = torch.randn(1, 4, 5, 48, requires_grad=True)
    freqs = torch.ones(20, 1, 6, dtype=torch.complex128)

    result = compressor(
        block_tokens=blocks,
        block_freqs=freqs,
        levels=torch.zeros(1, 4, dtype=torch.long),
        spans=torch.ones(1, 4, dtype=torch.long),
    )
    result.square().mean().backward()

    assert not torch.equal(result.detach(), blocks[:, -1].detach())
    assert blocks.grad is not None
    assert blocks.grad[:, 0].abs().sum().item() > 0.0
    assert compressor.attention.q.weight.grad is not None


def test_compressor_rejects_invalid_metadata():
    compressor = NativeBlockCompressor(_tiny_block(), group_size=4)
    blocks = torch.randn(1, 4, 5, 48)
    freqs = torch.ones(20, 1, 6, dtype=torch.complex128)
    with pytest.raises(ValueError, match="positive"):
        compressor(
            block_tokens=blocks,
            block_freqs=freqs,
            levels=torch.zeros(1, 4, dtype=torch.long),
            spans=torch.tensor([[1, 1, 0, 1]]),
        )
    with pytest.raises(ValueError, match="levels"):
        compressor(
            block_tokens=blocks,
            block_freqs=freqs,
            levels=torch.tensor([[0, 0, 8, 0]]),
            spans=torch.ones(1, 4, dtype=torch.long),
        )


def test_four_raw_blocks_produce_recursive_metadata():
    compressor = NativeBlockCompressor(_tiny_block(), group_size=4)
    raw = [
        NativeBlock(tokens=torch.randn(1, 5, 48), endpoint=index, span=1, level=0)
        for index in range(4)
    ]
    freqs = torch.ones(20, 1, 6, dtype=torch.complex128)

    summary = compressor.consolidate_blocks(raw, block_freqs=freqs)

    assert summary.tokens.shape == (1, 5, 48)
    assert summary.endpoint == 3
    assert summary.span == 4
    assert summary.level == 1


def test_bfloat16_compressor_consolidates_without_autocast():
    compressor = NativeBlockCompressor(
        _tiny_block().to(dtype=torch.bfloat16), group_size=4
    )
    raw = [
        NativeBlock(
            tokens=torch.randn(1, 5, 48, dtype=torch.bfloat16),
            endpoint=index,
            span=1,
            level=0,
        )
        for index in range(4)
    ]
    freqs = torch.ones(20, 1, 6, dtype=torch.complex128)

    summary = compressor.consolidate_blocks(raw, block_freqs=freqs)

    assert summary.tokens.dtype == torch.bfloat16
    assert summary.span == 4
    assert summary.level == 1


def test_native_cache_state_summary_reports_retained_structure():
    blocks = (
        NativeBlock(tokens=torch.zeros(1, 3, 8), endpoint=3, span=4, level=1),
        NativeBlock(tokens=torch.zeros(1, 3, 8), endpoint=4, span=1, level=0),
    )
    kv_cache = tuple(
        {
            "k": torch.zeros(1, 6, 2, 4),
            "v": torch.zeros(1, 6, 2, 4),
        }
        for _ in range(2)
    )

    summary = summarize_native_cache_state(
        NativeCacheState(blocks=blocks, kv_cache=kv_cache)
    )

    assert summary == {
        "retained_blocks": 2,
        "retained_tokens": 6,
        "represented_frames": 5,
        "latest_endpoint": 4,
        "level_histogram": {"0": 1, "1": 1},
        "kv_layers": 2,
        "kv_tokens_per_layer": 6,
    }


def test_native_cache_state_summary_reports_layerwise_units():
    """Catches telemetry that assumes every native state has legacy `.blocks`."""
    units = (
        CacheUnit(kind="anchor", endpoint=0, span=1, token_count=120),
        CacheUnit(kind="memory", endpoint=4, span=4, token_count=32),
        CacheUnit(kind="recent", endpoint=5, span=1, token_count=120),
    )
    kv_cache = tuple(
        {
            "k": torch.zeros(1, 272, 2, 4),
            "v": torch.zeros(1, 272, 2, 4),
        }
        for _ in range(2)
    )

    summary = summarize_native_cache_state(
        LayerwiseMemoryState(units=units, kv_cache=kv_cache)
    )

    assert summary == {
        "retained_units": 3,
        "retained_tokens": 272,
        "represented_frames": 6,
        "latest_endpoint": 5,
        "unit_kind_histogram": {"anchor": 1, "memory": 1, "recent": 1},
        "level_histogram": {"1": 1},
        "max_level": 1,
        "last_carry_levels": [],
        "kv_layers": 2,
        "kv_tokens_per_layer": 272,
    }


@pytest.mark.parametrize(
    ("clean_frames", "expected_clean_units", "expected_clean_values"),
    [
        (1, 1, [0.0]),
        (4, 4, [0.0, 1.0, 2.0, 3.0]),
        (5, 2, [3.0, 4.0]),
        (9, 3, [3.0, 7.0, 8.0]),
    ],
)
def test_training_prefix_only_consolidates_completed_groups_before_current(
    clean_frames, expected_clean_units, expected_clean_values
):
    tokens_per_frame = 2
    hidden_dim = 48
    noisy_frames = 1
    total_frames = clean_frames + noisy_frames
    frame_values = torch.arange(total_frames, dtype=torch.float32)
    tokens = frame_values.view(1, total_frames, 1, 1).expand(
        1, total_frames, tokens_per_frame, hidden_dim
    ).reshape(1, total_frames * tokens_per_frame, hidden_dim).clone()
    video_pre = {
        "tokens": tokens,
        "freqs": torch.ones(
            total_frames * tokens_per_frame, 1, 6, dtype=torch.complex128
        ),
        "t_mod": torch.zeros(
            1, total_frames * tokens_per_frame, 6, hidden_dim
        ),
        "context_mask": torch.ones(
            1, total_frames * tokens_per_frame, 3, dtype=torch.bool
        ),
        "meta": {
            "grid_size": (total_frames, 1, tokens_per_frame),
            "tokens_per_frame": tokens_per_frame,
            "batch_size": 1,
        },
    }
    model = object.__new__(FastWAM)
    nn.Module.__init__(model)
    model.native_cache_compressor = NativeBlockCompressor(
        _tiny_block(), group_size=4
    )

    retained, clean_units = model._consolidate_native_training_state(
        video_pre,
        clean_frame_count=clean_frames,
        noisy_frame_count=noisy_frames,
    )

    assert clean_units == expected_clean_units
    retained_frames = retained["tokens"].reshape(
        1, expected_clean_units + noisy_frames, tokens_per_frame, hidden_dim
    )
    assert retained_frames[0, :, 0, 0].tolist() == [
        *expected_clean_values,
        float(clean_frames),
    ]


def test_compressor_only_train_mode_freezes_backbone():
    class TrainModeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.dit = nn.Linear(4, 4)
            self.native_cache_compressor = nn.Linear(4, 4)
            self.proprio_encoder = nn.Linear(2, 4)

    model = TrainModeModel()
    Wan22Trainer._apply_dit_only_train_mode(
        model, native_cache_train_mode="compressor_only"
    )

    assert not any(parameter.requires_grad for parameter in model.dit.parameters())
    assert not any(
        parameter.requires_grad for parameter in model.proprio_encoder.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in model.native_cache_compressor.parameters()
    )


def test_full_train_mode_enables_layerwise_memory_slots_with_backbone():
    """Catches formal runs that leave the new slots frozen or out of AdamW."""
    class TrainModeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.dit = nn.Linear(4, 4)
            self.native_cache_compressor = None
            self.layerwise_block_memory = nn.Linear(4, 4)
            self.proprio_encoder = nn.Linear(2, 4)

    model = TrainModeModel()
    Wan22Trainer._apply_dit_only_train_mode(model, native_cache_train_mode="full")

    assert all(parameter.requires_grad for parameter in model.dit.parameters())
    assert all(
        parameter.requires_grad
        for parameter in model.layerwise_block_memory.parameters()
    )


def test_online_four_raw_suffix_rewrites_to_one_native_cache_block():
    torch.manual_seed(17)
    video = _tiny_video_expert(num_layers=2)
    action = ActionDiT(
        hidden_dim=32,
        action_dim=3,
        ffn_dim=64,
        text_dim=16,
        freq_dim=8,
        eps=1.0e-6,
        num_heads=4,
        attn_head_dim=12,
        num_layers=2,
    )
    mot = MoT(
        mixtures={"video": video, "action": action},
        mot_checkpoint_mixed_attn=False,
    ).eval()
    model = object.__new__(FastWAM)
    nn.Module.__init__(model)
    model.mot = mot
    model.native_cache_compressor = NativeBlockCompressor(
        video.blocks[0], group_size=4
    )
    context = torch.randn(1, 2, 16)
    context_mask = torch.ones(1, 2, dtype=torch.bool)
    state = None
    cache_lengths_before_commit = []

    for endpoint in range(4):
        pre = video.pre_dit(
            x=torch.randn(1, 4, 1, 4, 4),
            timestep=torch.zeros(1),
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=True,
            temporal_position_offset=endpoint,
        )
        history_cache = None if state is None else list(state.kv_cache)
        history_len = 0 if history_cache is None else history_cache[0]["k"].shape[1]
        current_cache = mot.prefill_video_cache(
            video_tokens=pre["tokens"],
            video_freqs=pre["freqs"],
            video_t_mod=pre["t_mod"],
            video_context_payload={
                "context": pre["context"],
                "mask": pre["context_mask"],
            },
            video_attention_mask=torch.ones(
                pre["tokens"].shape[1],
                history_len + pre["tokens"].shape[1],
                dtype=torch.bool,
            ),
            history_kv_cache=history_cache,
        )
        cache_lengths_before_commit.append(current_cache[0]["k"].shape[1])
        state = model._commit_native_cache_state(
            previous_state=state,
            current_pre=pre,
            current_cache=current_cache,
            endpoint=endpoint,
        )

    assert cache_lengths_before_commit == [4, 8, 12, 16]
    assert isinstance(state, NativeCacheState)
    assert len(state.blocks) == 1
    assert state.blocks[0].endpoint == 3
    assert state.blocks[0].span == 4
    assert state.blocks[0].level == 1
    assert all(layer["k"].shape[1] == 4 for layer in state.kv_cache)
    assert all(layer["v"].shape[1] == 4 for layer in state.kv_cache)
