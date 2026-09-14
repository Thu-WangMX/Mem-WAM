"""Shared fixed-noise clean-latent transition scorer."""

from __future__ import annotations

import torch

from fastwam.memory.dynamic_surprise import latent_surprise, recover_clean_latent


def transition_noise_seed(
    episode_index: int,
    transition_index: int,
    *,
    mode: str = "per_transition",
) -> int:
    key = str(mode).strip().lower()
    if key not in {"per_transition", "per_episode"}:
        raise ValueError(
            "dynamic surprise noise mode must be 'per_transition' or "
            f"'per_episode', got {mode!r}"
        )
    base = int(episode_index) * 100000
    return base if key == "per_episode" else base + int(transition_index)


@torch.no_grad()
def score_transition(
    model,
    *,
    history_latents: torch.Tensor,
    actual_latent: torch.Tensor,
    action: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    video_context: torch.Tensor,
    video_context_mask: torch.Tensor,
    memory_groups: tuple[tuple[int, ...], ...] | None,
    sigma: float,
    noise_seed: int,
) -> dict[str, float]:
    """Predict one observed arrival from causal history and score its surprise."""

    if history_latents.ndim != 5 or actual_latent.ndim != 5:
        raise ValueError("history_latents and actual_latent must be [B,C,T,H,W]")
    if history_latents.shape[0] != 1 or actual_latent.shape[0] != 1:
        raise ValueError("dynamic surprise scoring requires batch size one")
    if actual_latent.shape[2] != 1:
        raise ValueError("actual_latent must contain exactly one arrival frame")
    actual = actual_latent.to(device=model.device, dtype=model.torch_dtype)
    history = history_latents.to(device=model.device, dtype=model.torch_dtype)
    generator = torch.Generator(device="cpu").manual_seed(int(noise_seed))
    noise = torch.randn(
        actual.shape, generator=generator, dtype=torch.float32
    ).to(device=model.device, dtype=model.torch_dtype)
    noisy = (1.0 - float(sigma)) * actual + float(sigma) * noise
    clean_frames = int(history.shape[2])
    video_input = torch.cat([history, noisy], dim=2)
    video_t = torch.cat(
        [
            torch.zeros(
                (1, clean_frames), device=model.device, dtype=model.torch_dtype
            ),
            torch.full(
                (1, 1),
                float(sigma) * 1000.0,
                device=model.device,
                dtype=model.torch_dtype,
            ),
        ],
        dim=1,
    )
    video_pre = model.video_expert.pre_dit(
        x=video_input,
        timestep=video_t,
        context=video_context,
        context_mask=video_context_mask,
        action=None,
        fuse_vae_embedding_in_latents=True,
    )
    action_pre = model.action_expert.pre_dit(
        action_tokens=action,
        timestep=torch.zeros(1, device=model.device, dtype=model.torch_dtype),
        context=context,
        context_mask=context_mask,
    )
    _, grid_h, grid_w = video_pre["meta"]["grid_size"]
    action_pre["freqs"] = model._build_video_aligned_action_freqs(
        action_seq_len=int(action_pre["tokens"].shape[1]),
        temporal_base=float(clean_frames - 1),
        grid_h=int(grid_h),
        grid_w=int(grid_w),
        device=model.device,
    )
    tokens_out, layout = model._run_layerwise_memory_training_transformer(
        video_pre=video_pre,
        action_pre=action_pre,
        clean_frame_count=clean_frames,
        noisy_frame_count=1,
        memory_groups=memory_groups,
    )
    start, stop = layout.noisy_range
    tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])
    future_pre = dict(video_pre)
    future_pre["t"] = video_pre["t"][:, clean_frames * tokens_per_frame :]
    future_pre["meta"] = dict(video_pre["meta"])
    future_pre["meta"]["grid_size"] = (1, int(grid_h), int(grid_w))
    velocity = model.video_expert.post_dit(
        tokens_out["video"][:, start:stop], future_pre
    )
    predicted = recover_clean_latent(noisy, velocity, float(sigma))
    return latent_surprise(predicted, actual)
