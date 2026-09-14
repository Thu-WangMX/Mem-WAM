import pytest
import torch
import torch.nn as nn

from fastwam.memory.native_cache import CacheUnit, DynamicLayerwiseMemoryState
from fastwam.models.wan22.fastwam import FastWAM
from fastwam.memory.native_cache import LayerwiseBlockMemory
from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.wan_video_dit import WanVideoDiT


def _cache(values):
    tensor = torch.tensor(values, dtype=torch.float32).view(1, -1, 1, 1)
    return ({"k": tensor.clone(), "v": tensor.clone() + 1000},)


def _state(units, values, open_start):
    return DynamicLayerwiseMemoryState(
        units=tuple(units), kv_cache=_cache(values), open_start=open_start
    )


class _Memory:
    memory_tokens = 8
    anchor_frames = 2
    recent_frames = 4

    def token_count_for_span(self, span):
        return self.memory_tokens

    def initial_tokens(self, *, batch_size, device, dtype, token_count=None):
        token_count = self.memory_tokens if token_count is None else int(token_count)
        return torch.full((batch_size, token_count, 1), 90, device=device, dtype=dtype)

    def build_freqs(self, *, endpoint, device, token_count=None):
        token_count = self.memory_tokens if token_count is None else int(token_count)
        return torch.zeros(token_count, 1, 1, device=device)

    def build_t_mod(self, endpoint_t_mod, *, token_count=None):
        token_count = self.memory_tokens if token_count is None else int(token_count)
        return endpoint_t_mod.expand(-1, token_count, -1, -1)


class _MoT:
    def prefill_video_cache(self, *, video_tokens, history_kv_cache=None, **kwargs):
        current = video_tokens[..., :1].unsqueeze(-1)
        current = {"k": current, "v": current + 1000}
        if history_kv_cache is None:
            return (current,)
        return tuple(
            {
                "k": torch.cat((old["k"], current["k"]), dim=1),
                "v": torch.cat((old["v"], current["v"]), dim=1),
            }
            for old in history_kv_cache
        )


class _Harness:
    layerwise_block_memory = _Memory()
    mot = _MoT()
    _commit_dynamic_layerwise_memory_state = (
        FastWAM._commit_dynamic_layerwise_memory_state
    )


def _pre(value):
    return {
        "tokens": torch.full((1, 2, 1), value, dtype=torch.float32),
        "freqs": torch.zeros(2, 1, 1),
        "t_mod": torch.zeros(1, 2, 6, 1),
        "context": torch.zeros(1, 1, 1),
        "context_mask": torch.ones(1, 2, 1, dtype=torch.bool),
    }


def test_dynamic_open_tail_starts_after_anchors_arrive():
    state = None
    expected_open_starts = (0, 1, 2)
    for endpoint, expected in enumerate(expected_open_starts):
        state = _Harness()._commit_dynamic_layerwise_memory_state(
            previous_state=state,
            current_pre=_pre(endpoint),
            endpoint=endpoint,
            close_range=None,
        )
        assert state.open_start == expected


@pytest.mark.parametrize("span", [4, 6, 8])
def test_delayed_dynamic_close_keeps_anchors_open_suffix_and_arrival_raw(span):
    arrival = 10
    stop = 2 + span
    previous = _state(
        [
            CacheUnit(
                kind="anchor" if i < 2 else "raw",
                endpoint=i,
                span=1,
                token_count=2,
            )
            for i in range(arrival)
        ],
        [value for i in range(arrival) for value in (i, i)],
        open_start=2,
    )

    result = _Harness()._commit_dynamic_layerwise_memory_state(
        previous_state=previous,
        current_pre=_pre(arrival),
        endpoint=arrival,
        close_range=(2, stop),
        close_token_count=8,
    )

    expected_units = [
        ("anchor", 0, 0, 2),
        ("anchor", 1, 1, 2),
        *[("recent", i, i, 2) for i in range(stop, arrival)],
        ("memory", 2, stop - 1, 8),
        ("recent", arrival, arrival, 2),
    ]
    expected_k = [0, 0, 1, 1]
    expected_k.extend(
        value for i in range(stop, arrival) for value in (i, i)
    )
    expected_k.extend([90] * 8)
    expected_k.extend([arrival, arrival])
    assert [(u.kind, u.start, u.endpoint, u.token_count) for u in result.units] == expected_units
    assert result.open_start == stop
    assert result.kv_cache[0]["k"].flatten().tolist() == expected_k


def test_dynamic_close_preserves_earlier_memory_kv_exactly():
    previous = _state(
        [
            CacheUnit(kind="anchor", endpoint=0, span=1, token_count=2),
            CacheUnit(kind="anchor", endpoint=1, span=1, token_count=2),
            CacheUnit(kind="memory", endpoint=5, span=4, token_count=8),
            CacheUnit(kind="recent", endpoint=6, span=1, token_count=2),
            CacheUnit(kind="recent", endpoint=7, span=1, token_count=2),
        ],
        [0, 0, 1, 1, *range(10, 18), 6, 6, 7, 7],
        open_start=6,
    )

    result = _Harness()._commit_dynamic_layerwise_memory_state(
        previous_state=previous,
        current_pre=_pre(8),
        endpoint=8,
        close_range=(6, 8),
        close_token_count=8,
    )

    assert torch.equal(result.kv_cache[0]["k"][:, 4:12], previous.kv_cache[0]["k"][:, 4:12])
    assert [(u.kind, u.start, u.endpoint) for u in result.units] == [
        ("anchor", 0, 0),
        ("anchor", 1, 1),
        ("memory", 2, 5),
        ("memory", 6, 7),
        ("recent", 8, 8),
    ]


def test_dynamic_action_exposes_the_complete_open_raw_tail():
    state = _state(
        [
            CacheUnit(kind="anchor", endpoint=0, span=1, token_count=2),
            CacheUnit(kind="anchor", endpoint=1, span=1, token_count=2),
            CacheUnit(kind="memory", endpoint=5, span=4, token_count=8),
            *[
                CacheUnit(kind="raw", endpoint=i, span=1, token_count=2)
                for i in range(6, 12)
            ],
        ],
        [0] * 24,
        open_start=6,
    )

    mask = FastWAM._build_dynamic_layerwise_action_mask(
        _Harness(),
        state=state,
        action_tokens=3,
        device=torch.device("cpu"),
    )

    assert mask.shape == (3, 27)
    assert mask[:, :12].all()  # two anchors and one memory
    assert mask[:, 12:].all()  # raw frames 6..11 and all action tokens


def test_dynamic_state_rejects_noncontiguous_coverage():
    with pytest.raises(ValueError, match="contiguously"):
        _state(
            [CacheUnit(kind="raw", endpoint=1, span=1, token_count=2)],
            [1, 1],
            open_start=1,
        )


def test_dynamic_close_rejects_range_that_includes_arrival():
    previous = _state(
        [
            CacheUnit(
                kind="anchor" if i < 2 else "raw",
                endpoint=i,
                span=1,
                token_count=2,
            )
            for i in range(6)
        ],
        [value for i in range(6) for value in (i, i)],
        open_start=2,
    )
    with pytest.raises(ValueError, match="prefix of the open tail"):
        _Harness()._commit_dynamic_layerwise_memory_state(
            previous_state=previous,
            current_pre=_pre(6),
            endpoint=6,
            close_range=(2, 7),
            close_token_count=8,
        )


@pytest.mark.parametrize(
    ("memory_tokens", "dynamic_tokens_per_frame"),
    [(8, None), (48, 8)],
)
@pytest.mark.parametrize(
    "groups",
    [
        ((2, 3), (4, 5)),
        ((2, 3, 4, 5),),
    ],
)
def test_dynamic_packed_training_and_online_cache_are_numerically_identical(
    memory_tokens, dynamic_tokens_per_frame, groups
):
    torch.manual_seed(211)
    video = WanVideoDiT(
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
        num_layers=2,
        has_image_input=False,
        seperated_timestep=True,
        require_vae_embedding=False,
        fuse_vae_embedding_in_latents=True,
        action_conditioned=False,
    )
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
    model.layerwise_block_memory = LayerwiseBlockMemory(
        video,
        memory_tokens=memory_tokens,
        dynamic_tokens_per_frame=dynamic_tokens_per_frame,
    )
    context = torch.randn(1, 2, 16)
    context_mask = torch.ones(1, 2, dtype=torch.bool)
    observations = torch.randn(1, 4, 7, 4, 4)
    noisy = torch.randn(1, 4, 1, 4, 4)
    action_pre = action.pre_dit(
        action_tokens=torch.randn(1, 3, 3),
        timestep=torch.zeros(1),
        context=context,
        context_mask=context_mask,
    )
    full_pre = video.pre_dit(
        x=torch.cat((observations, noisy), dim=2),
        timestep=torch.zeros(1, 8),
        context=context,
        context_mask=context_mask,
        fuse_vae_embedding_in_latents=True,
    )
    packed, layout = model._pack_layerwise_memory_training_state(
        full_pre,
        clean_frame_count=7,
        noisy_frame_count=1,
        action_seq_len=3,
        memory_groups=groups,
    )
    _, packed_cache = mot(
        embeds_all={"video": packed["tokens"], "action": action_pre["tokens"]},
        attention_mask=layout.attention_mask,
        freqs_all={"video": packed["freqs"], "action": action_pre["freqs"]},
        context_all={
            "video": {"context": packed["context"], "mask": packed["context_mask"]},
            "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
        },
        t_mod_all={"video": packed["t_mod"], "action": action_pre["t_mod"]},
        return_video_kv_cache=True,
    )
    packed_retained = tuple(
        {
            key: torch.cat(
                [layer[key][:, start:stop] for start, stop in layout.retained_ranges],
                dim=1,
            )
            for key in ("k", "v")
        }
        for layer in packed_cache
    )

    state = None
    close_ranges = {
        group[-1] + 1: (group[0], group[-1] + 1) for group in groups
    }
    for endpoint in range(7):
        pre = video.pre_dit(
            x=observations[:, :, endpoint : endpoint + 1],
            timestep=torch.zeros(1),
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=True,
            temporal_position_offset=endpoint,
        )
        close_range = close_ranges.get(endpoint)
        close_token_count = (
            None
            if close_range is None
            else model.layerwise_block_memory.token_count_for_span(
                close_range[1] - close_range[0]
            )
        )
        state = model._commit_dynamic_layerwise_memory_state(
            previous_state=state,
            current_pre=pre,
            endpoint=endpoint,
            close_range=close_range,
            close_token_count=close_token_count,
        )

    for packed_layer, online_layer in zip(packed_retained, state.kv_cache):
        torch.testing.assert_close(packed_layer["k"], online_layer["k"], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(packed_layer["v"], online_layer["v"], rtol=1e-5, atol=1e-6)

    expected_memory_counts = [
        memory_tokens if dynamic_tokens_per_frame is None else len(group) * 8
        for group in groups
    ]
    assert [
        unit.token_count for unit in state.units if unit.kind == "memory"
    ] == expected_memory_counts
