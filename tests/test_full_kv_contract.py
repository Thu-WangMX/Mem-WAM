import json
from types import SimpleNamespace

import pytest
import torch

from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.fastwam import FastWAM
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.schedulers.scheduler_continuous import (
    WanContinuousFlowMatchScheduler,
)
from fastwam.models.wan22.wan_video_dit import WanVideoDiT
from fastwam.datasets.lerobot.full_kv_dataset import (
    SCHEMA_VERSION,
    FullKVObservationStore,
    _decision_source_indices,
)


def test_full_history_mask_blocks_future_video_from_action():
    mask = FastWAM._build_full_history_training_mask(
        clean_video_frames=3,
        noisy_video_frames=1,
        video_tokens_per_frame=2,
        action_seq_len=4,
        device=torch.device("cpu"),
    )
    assert mask.shape == (12, 12)
    # First clean frame is an attention sink only to itself.
    assert mask[:2, :2].all()
    assert not mask[:2, 2:].any()
    # Latest clean frame sees all clean history, but not noisy future/action.
    assert mask[4:6, :6].all()
    assert not mask[4:6, 6:].any()
    # Noisy future sees all video context in its own diffusion group.
    assert mask[6:8, :8].all()
    assert not mask[6:8, 8:].any()
    # Action sees every clean video token and action, never noisy future.
    assert mask[8:, :6].all()
    assert not mask[8:, 6:8].any()
    assert mask[8:, 8:].all()


def _cache(values):
    return [
        {
            "k": torch.tensor(values, dtype=torch.float32).view(1, -1, 1),
            "v": torch.tensor(values, dtype=torch.float32).view(1, -1, 1),
        }
    ]


def test_action_cache_accepts_rectangular_mask_shape_guard():
    mot = object.__new__(MoT)
    torch.nn.Module.__init__(mot)
    mot.num_layers = 1
    mot.num_heads = 1
    mot.attn_head_dim = 1
    mot.mixtures = torch.nn.ModuleDict()
    with pytest.raises(ValueError, match="requires `action` expert"):
        mot.forward_action_with_video_cache(
            action_tokens=torch.zeros(1, 2, 1),
            action_freqs=torch.zeros(2, 1, 1),
            action_t_mod=torch.zeros(1, 6, 1),
            action_context_payload=None,
            video_kv_cache=_cache([1, 2, 3]),
            attention_mask=torch.ones(2, 5, dtype=torch.bool),
            video_seq_len=3,
        )


def test_full_history_mask_rejects_zero_dimensions():
    with pytest.raises(ValueError, match="positive"):
        FastWAM._build_full_history_training_mask(
            clean_video_frames=1,
            noisy_video_frames=0,
            video_tokens_per_frame=2,
            action_seq_len=4,
            device=torch.device("cpu"),
        )


def test_terminal_decision_indices_keep_partial_horizon():
    assert _decision_source_indices(100, 135, 16) == [100, 116, 132]
    with pytest.raises(ValueError, match="greater"):
        _decision_source_indices(100, 100, 16)
    with pytest.raises(ValueError, match="positive"):
        _decision_source_indices(100, 135, 0)


def test_decision_indices_can_require_a_completed_native_group():
    assert _decision_source_indices(
        100, 200, 16, minimum_history_frames=5
    ) == [164, 180, 196]


def test_video_loss_excludes_fully_padded_terminal_latent():
    model = object.__new__(FastWAM)
    torch.nn.Module.__init__(model)
    model.vae = SimpleNamespace(temporal_downsample_factor=4)
    pred = torch.zeros(1, 1, 2, 1, 1)
    target = torch.tensor([[[[[1.0]], [[3.0]]]]])
    image_is_pad = torch.tensor(
        [[False, False, False, False, False, True, True, True, True]]
    )
    loss = model._compute_video_loss_per_sample(
        pred_video=pred,
        target_video=target,
        image_is_pad=image_is_pad,
        include_initial_video_step=False,
    )
    torch.testing.assert_close(loss, torch.tensor([1.0]))


def test_build_inputs_keeps_proprio_out_of_video_context():
    model = object.__new__(FastWAM)
    torch.nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.torch_dtype = torch.float32
    model.proprio_dim = 3
    model.proprio_encoder = torch.nn.Linear(3, 8)
    model.mot = SimpleNamespace(
        mixtures={"video": SimpleNamespace(fuse_vae_embedding_in_latents=False)}
    )
    model._encode_video_latents = lambda video, tiled=False: torch.zeros(
        video.shape[0], 4, 2, 2, 2
    )

    result = FastWAM.build_inputs(
        model,
        {
            "video": torch.zeros(1, 3, 5, 16, 16),
            "context": torch.randn(1, 4, 8),
            "context_mask": torch.ones(1, 4, dtype=torch.bool),
            "proprio": torch.randn(1, 1, 3),
            "action": torch.randn(1, 4, 2),
        },
    )
    assert result["video_context"].shape[1] == 4
    assert result["video_context_mask"].shape[1] == 4
    assert result["context"].shape[1] == 5
    assert result["context_mask"].shape[1] == 5


def test_observation_store_returns_strict_causal_history_including_current(tmp_path):
    shard = tmp_path / "episode.pt"
    torch.save(
        {
            "frame_indices": torch.tensor([0, 16, 32]),
            "latents": torch.arange(
                3 * 48 * 24 * 20, dtype=torch.float32
            ).reshape(3, 48, 1, 24, 20).to(torch.bfloat16),
        },
        shard,
    )
    manifest = {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "replan_stride": 16,
            "temporal_subframes": 4,
            "temporal_subframe_stride": 4,
            "latent_shape": [48, 1, 24, 20],
            "latent_dtype": "bfloat16",
            "mosaic": "wrists_top_head_bottom_384x320",
            "color_contract": "simulator_rgb_preserved",
            "encoding": "continuous_episode_stride4_causal_vae",
        },
        "episodes": {"put_back_block/7": "episode.pt"},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    store = FullKVObservationStore(tmp_path, expected_replan_stride=16)
    history = store.strict_history(
        {
            "dataset_name": "/datasets/put_back_block",
            "episode_index": 7,
            "frame_index": 32,
        }
    )
    assert history.shape == (3, 48, 1, 24, 20)
    initial = store.strict_history(
        {
            "dataset_name": "/datasets/put_back_block",
            "episode_index": 7,
            "frame_index": 0,
        }
    )
    assert initial.shape == (1, 48, 1, 24, 20)
    with pytest.raises(KeyError, match="absent"):
        store.strict_history(
            {
                "dataset_name": "/datasets/put_back_block",
                "episode_index": 7,
                "frame_index": 24,
            }
        )


def _tiny_video_expert(num_layers=2):
    return WanVideoDiT(
        hidden_dim=48,
        in_dim=4,
        ffn_dim=96,
        out_dim=4,
        text_dim=16,
        freq_dim=8,
        eps=1e-6,
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


def test_video_pre_dit_supports_per_frame_timestep_and_global_offset():
    expert = _tiny_video_expert(num_layers=1)
    x = torch.randn(1, 4, 3, 4, 4)
    context = torch.randn(1, 2, 16)
    context_mask = torch.ones(1, 2, dtype=torch.bool)
    per_frame_t = torch.tensor([[0.0, 0.0, 0.75]])
    first = expert.pre_dit(
        x=x,
        timestep=per_frame_t,
        context=context,
        context_mask=context_mask,
        fuse_vae_embedding_in_latents=True,
        temporal_position_offset=0,
    )
    shifted = expert.pre_dit(
        x=x,
        timestep=per_frame_t,
        context=context,
        context_mask=context_mask,
        fuse_vae_embedding_in_latents=True,
        temporal_position_offset=5,
    )
    assert first["t_mod"].shape == (1, 12, 6, 48)
    assert first["freqs"].shape == shifted["freqs"].shape
    assert not torch.equal(first["freqs"], shifted["freqs"])


def test_video_prefill_appends_every_layer_without_recomputing_history():
    video = _tiny_video_expert(num_layers=2)
    action = ActionDiT(
        hidden_dim=32,
        action_dim=3,
        ffn_dim=64,
        text_dim=16,
        freq_dim=8,
        eps=1e-6,
        num_heads=4,
        attn_head_dim=12,
        num_layers=2,
    )
    mot = MoT(
        mixtures={"video": video, "action": action},
        mot_checkpoint_mixed_attn=False,
    ).eval()
    context = torch.randn(1, 2, 16)
    context_mask = torch.ones(1, 2, dtype=torch.bool)

    def pre(frame, offset):
        return video.pre_dit(
            x=frame,
            timestep=torch.zeros(1),
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=True,
            temporal_position_offset=offset,
        )

    first = pre(torch.randn(1, 4, 1, 4, 4), 0)
    cache1 = mot.prefill_video_cache(
        video_tokens=first["tokens"],
        video_freqs=first["freqs"],
        video_t_mod=first["t_mod"],
        video_context_payload={
            "context": first["context"],
            "mask": first["context_mask"],
        },
        video_attention_mask=torch.ones(4, 4, dtype=torch.bool),
    )
    cache1_snapshot = [
        {"k": item["k"].clone(), "v": item["v"].clone()} for item in cache1
    ]

    second = pre(torch.randn(1, 4, 1, 4, 4), 1)
    cache2 = mot.prefill_video_cache(
        video_tokens=second["tokens"],
        video_freqs=second["freqs"],
        video_t_mod=second["t_mod"],
        video_context_payload={
            "context": second["context"],
            "mask": second["context_mask"],
        },
        video_attention_mask=torch.ones(4, 8, dtype=torch.bool),
        history_kv_cache=cache1,
    )
    assert len(cache2) == 2
    assert all(item["k"].shape[1] == 8 for item in cache2)
    for old, snapshot, combined in zip(cache1, cache1_snapshot, cache2):
        assert torch.equal(old["k"], snapshot["k"])
        assert torch.equal(old["v"], snapshot["v"])
        assert torch.equal(combined["k"][:, :4], snapshot["k"])
        assert torch.equal(combined["v"][:, :4], snapshot["v"])


def test_training_timestep_sampling_supports_fastwam_and_logit_normal():
    torch.manual_seed(11)
    default_scheduler = WanContinuousFlowMatchScheduler(shift=5.0)
    actual = default_scheduler.sample_training_t(
        batch_size=32,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    torch.manual_seed(11)
    u = torch.rand(32)
    expected = 1000.0 * (5.0 * u / (1.0 + 4.0 * u))
    assert torch.equal(actual, expected)

    torch.manual_seed(13)
    logit_normal = WanContinuousFlowMatchScheduler(
        shift=1.0,
        training_sampling_scheme="logit_normal",
    ).sample_training_t(
        batch_size=100_000,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    normalized = logit_normal / 1000.0
    assert bool(((normalized > 0.0) & (normalized < 1.0)).all())
    assert normalized.mean().item() == pytest.approx(0.5, abs=0.005)
    assert normalized.std().item() == pytest.approx(0.208, abs=0.005)


def test_scheduler_can_disable_fastwam_training_weight():
    scheduler = WanContinuousFlowMatchScheduler(training_weight_scheme="none")
    assert torch.equal(
        scheduler.training_weight(torch.tensor([0.0, 500.0, 999.0])),
        torch.ones(3),
    )
    with pytest.raises(ValueError, match="training_sampling_scheme"):
        WanContinuousFlowMatchScheduler(training_sampling_scheme="unknown")


def test_action_rope_spatial_marker_is_configurable():
    model = object.__new__(FastWAM)
    torch.nn.Module.__init__(model)
    model.mot = SimpleNamespace(
        mixtures={"video": SimpleNamespace(freqs=(
            torch.tensor(
                [[1.0, 1.0], [1.0j, 1.0j]] + [[1.0, 1.0]] * 6,
                dtype=torch.complex64,
            ),
            torch.tensor(
                [[1.0, 1.0], [1.0j, 1.0j], [-1.0, -1.0]],
                dtype=torch.complex64,
            ),
            torch.tensor(
                [[1.0, 1.0], [-1.0j, -1.0j], [-1.0, -1.0]],
                dtype=torch.complex64,
            ),
        ))}
    )

    model.action_rope_spatial_mode = "origin"
    origin = model._build_video_aligned_action_freqs(
        action_seq_len=2,
        temporal_base=0.0,
        grid_h=3,
        grid_w=3,
        device=torch.device("cpu"),
    )
    model.action_rope_spatial_mode = "center"
    center = model._build_video_aligned_action_freqs(
        action_seq_len=2,
        temporal_base=0.0,
        grid_h=3,
        grid_w=3,
        device=torch.device("cpu"),
    )
    model.action_rope_spatial_mode = "memorywam"
    memorywam = model._build_video_aligned_action_freqs(
        action_seq_len=2,
        temporal_base=0.0,
        grid_h=3,
        grid_w=3,
        device=torch.device("cpu"),
    )

    assert torch.equal(origin[..., 2:4], torch.ones_like(origin[..., 2:4]))
    assert torch.equal(origin[..., 4:6], torch.ones_like(origin[..., 4:6]))
    assert torch.equal(center[..., 2:4], torch.full_like(center[..., 2:4], 1.0j))
    assert torch.equal(center[..., 4:6], torch.full_like(center[..., 4:6], -1.0j))
    expected_temporal = torch.polar(
        torch.ones(2, 2),
        torch.tensor([[1.0 / 3.0], [2.0 / 3.0]]) * (torch.pi / 2),
    ).unsqueeze(1)
    torch.testing.assert_close(memorywam[..., :2], expected_temporal)
    assert torch.equal(memorywam[..., 2:4], torch.full_like(memorywam[..., 2:4], -1.0j))
    assert torch.equal(memorywam[..., 4:6], torch.full_like(memorywam[..., 4:6], 1.0j))


def test_joint_full_attention_matches_sequential_cached_action_forward():
    torch.manual_seed(7)
    video = _tiny_video_expert(num_layers=2)
    action = ActionDiT(
        hidden_dim=32,
        action_dim=3,
        ffn_dim=64,
        text_dim=16,
        freq_dim=8,
        eps=1e-6,
        num_heads=4,
        attn_head_dim=12,
        num_layers=2,
    )
    mot = MoT(
        mixtures={"video": video, "action": action},
        mot_checkpoint_mixed_attn=False,
    ).eval()
    context = torch.randn(1, 3, 16)
    context_mask = torch.ones(1, 3, dtype=torch.bool)
    frames = [torch.randn(1, 4, 1, 4, 4) for _ in range(2)]
    action_input = torch.randn(1, 4, 3)

    video_all = video.pre_dit(
        x=torch.cat(frames, dim=2),
        timestep=torch.zeros(1, 2),
        context=context,
        context_mask=context_mask,
        fuse_vae_embedding_in_latents=True,
    )
    action_all = action.pre_dit(
        action_tokens=action_input,
        timestep=torch.tensor([0.6]),
        context=context,
        context_mask=context_mask,
    )
    shim = object.__new__(FastWAM)
    torch.nn.Module.__init__(shim)
    shim.video_expert = video
    action_freqs = FastWAM._build_video_aligned_action_freqs(
        shim,
        action_seq_len=action_input.shape[1],
        temporal_base=1.0,
        grid_h=2,
        grid_w=2,
        device=torch.device("cpu"),
    )
    action_all["freqs"] = action_freqs

    video_seq_len = video_all["tokens"].shape[1]
    action_seq_len = action_all["tokens"].shape[1]
    tokens_per_frame = int(video_all["meta"]["tokens_per_frame"])
    joint_mask = torch.zeros(
        video_seq_len + action_seq_len,
        video_seq_len + action_seq_len,
        dtype=torch.bool,
    )
    joint_mask[:tokens_per_frame, :tokens_per_frame] = True
    joint_mask[tokens_per_frame:video_seq_len, :video_seq_len] = True
    joint_mask[video_seq_len:, :video_seq_len] = True
    joint_mask[video_seq_len:, video_seq_len:] = True
    joint_action = mot(
        embeds_all={
            "video": video_all["tokens"],
            "action": action_all["tokens"],
        },
        attention_mask=joint_mask,
        freqs_all={"video": video_all["freqs"], "action": action_freqs},
        context_all={
            "video": {
                "context": video_all["context"],
                "mask": video_all["context_mask"],
            },
            "action": {
                "context": action_all["context"],
                "mask": action_all["context_mask"],
            },
        },
        t_mod_all={
            "video": video_all["t_mod"],
            "action": action_all["t_mod"],
        },
    )["action"]

    cache = None
    for offset, frame in enumerate(frames):
        frame_pre = video.pre_dit(
            x=frame,
            timestep=torch.zeros(1),
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=True,
            temporal_position_offset=offset,
        )
        cached_video_tokens = (offset + 1) * tokens_per_frame
        cache = mot.prefill_video_cache(
            video_tokens=frame_pre["tokens"],
            video_freqs=frame_pre["freqs"],
            video_t_mod=frame_pre["t_mod"],
            video_context_payload={
                "context": frame_pre["context"],
                "mask": frame_pre["context_mask"],
            },
            video_attention_mask=torch.ones(
                tokens_per_frame, cached_video_tokens, dtype=torch.bool
            ),
            history_kv_cache=cache,
        )
    cached_action = mot.forward_action_with_video_cache(
        action_tokens=action_all["tokens"],
        action_freqs=action_freqs,
        action_t_mod=action_all["t_mod"],
        action_context_payload={
            "context": action_all["context"],
            "mask": action_all["context_mask"],
        },
        video_kv_cache=cache,
        attention_mask=torch.ones(
            action_seq_len,
            video_seq_len + action_seq_len,
            dtype=torch.bool,
        ),
        video_seq_len=video_seq_len,
    )
    assert torch.allclose(joint_action, cached_action, atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        action.post_dit(joint_action, action_all),
        action.post_dit(cached_action, action_all),
        atol=1e-5,
        rtol=1e-5,
    )
