"""Auditable multi-layer spatial features from a frozen initialized Video Expert."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F


FEATURE_LAYERS = (5, 11, 17, 23, 29)
FEATURE_REGIONS = ("global", "left_wrist", "right_wrist", "head")
NATIVE_LATENT_HEIGHT = 24
NATIVE_LATENT_WIDTH = 20
PATCH_SIZE = 2


def build_region_masks(*, grid_h: int = 12, grid_w: int = 10) -> dict[str, torch.Tensor]:
    """Return fixed token masks for the wrists-top/head-bottom RoboTwin mosaic."""

    grid_h = int(grid_h)
    grid_w = int(grid_w)
    if grid_h <= 0 or grid_w <= 0 or grid_h % 3 or grid_w % 2:
        raise ValueError("token grid must be positive and divisible into fixed camera regions")
    wrist_rows = grid_h // 3
    half_width = grid_w // 2
    global_mask = torch.ones((grid_h, grid_w), dtype=torch.bool)
    left_wrist = torch.zeros_like(global_mask)
    right_wrist = torch.zeros_like(global_mask)
    head = torch.zeros_like(global_mask)
    left_wrist[:wrist_rows, :half_width] = True
    right_wrist[:wrist_rows, half_width:] = True
    head[wrist_rows:, :] = True
    return {
        "global": global_mask,
        "left_wrist": left_wrist,
        "right_wrist": right_wrist,
        "head": head,
    }


@torch.inference_mode()
def capture_spatial_features(
    model: Any,
    *,
    latents: torch.Tensor,
    video_context: torch.Tensor,
    video_context_mask: torch.Tensor,
    layer_indices: Sequence[int] = FEATURE_LAYERS,
) -> dict[int, dict[str, torch.Tensor]]:
    """Capture normalized latest-frame region features from multiple blocks."""

    if latents.ndim != 4 or tuple(latents.shape[-2:]) != (
        NATIVE_LATENT_HEIGHT,
        NATIVE_LATENT_WIDTH,
    ):
        raise ValueError("latents must have shape [48,T,24,20] on the native 24x20 grid")
    if latents.shape[1] < 1:
        raise ValueError("latents must contain at least one temporal frame")
    if video_context.ndim != 3 or video_context.shape[0] != 1:
        raise ValueError("video_context must have shape [1,L,D]")
    if video_context_mask.ndim != 2 or video_context_mask.shape[0] != 1:
        raise ValueError("video_context_mask must have shape [1,L]")
    layers = tuple(int(value) for value in layer_indices)
    if not layers or len(layers) != len(set(layers)):
        raise ValueError("layer_indices must be nonempty and unique")

    video_expert = model.video_expert
    blocks = video_expert.blocks
    if any(layer < 0 or layer >= len(blocks) for layer in layers):
        raise ValueError(f"layer_indices are outside {len(blocks)} Video Expert blocks")
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def hook_for(layer: int):
        def save_hidden(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
                raise RuntimeError(f"block {layer} hook did not receive [B,S,D] tokens")
            if layer in captured:
                raise RuntimeError(f"block {layer} executed more than once")
            captured[layer] = hidden.detach()

        return save_hidden

    for layer in layers:
        handles.append(blocks[layer].register_forward_hook(hook_for(layer)))
    checkpointing = bool(getattr(video_expert, "use_gradient_checkpointing", False))
    if hasattr(video_expert, "use_gradient_checkpointing"):
        video_expert.use_gradient_checkpointing = False
    try:
        video = latents.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
        frame_count = int(video.shape[2])
        timestep = torch.zeros(
            (1, frame_count), device=model.device, dtype=model.torch_dtype
        )
        video_expert(
            x=video,
            timestep=timestep,
            context=video_context.to(device=model.device, dtype=model.torch_dtype),
            context_mask=video_context_mask.to(device=model.device),
            action=None,
            fuse_vae_embedding_in_latents=True,
        )
    finally:
        for handle in handles:
            handle.remove()
        if hasattr(video_expert, "use_gradient_checkpointing"):
            video_expert.use_gradient_checkpointing = checkpointing

    if tuple(captured) != layers:
        raise RuntimeError(f"expected hooks {layers}, received {tuple(captured)}")
    grid_h = NATIVE_LATENT_HEIGHT // PATCH_SIZE
    grid_w = NATIVE_LATENT_WIDTH // PATCH_SIZE
    tokens_per_frame = grid_h * grid_w
    masks = build_region_masks(grid_h=grid_h, grid_w=grid_w)
    result: dict[int, dict[str, torch.Tensor]] = {}
    for layer in layers:
        hidden = captured[layer]
        if hidden.shape[0] != 1 or hidden.shape[1] != frame_count * tokens_per_frame:
            raise RuntimeError(
                f"block {layer} token shape {tuple(hidden.shape)} is incompatible with "
                f"{frame_count} frames on a {grid_h}x{grid_w} grid"
            )
        latest = hidden[0, -tokens_per_frame:].float().reshape(grid_h, grid_w, -1)
        regions: dict[str, torch.Tensor] = {}
        for name in FEATURE_REGIONS:
            pooled = latest[masks[name].to(device=latest.device)].mean(dim=0)
            if not bool(torch.isfinite(pooled).all()) or float(pooled.norm().item()) == 0.0:
                raise RuntimeError(f"block {layer} region {name} is zero or non-finite")
            regions[name] = F.normalize(pooled, dim=0).cpu()
        result[layer] = regions
    return result
