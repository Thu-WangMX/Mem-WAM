"""FastWAM Multi-Term Memory (MTM) variant.

Implements Helios-style multi-term memory on top of FastWAM. The key idea: replace
the single `current_latent` block in fastwam by `history_context = [long(16) | mid(2) | current(1) | pred(1)]`,
where `long / mid / current` are clean history latents (zero-timestep AdaLN, attention sees them as
single-direction guidance), and only `pred` is denoised.

Adapted from Thu-WangMX/Fastwam-helios (7e0ec698). Only MTMVideoDiT is retained here.
Original class descriptions:
    - `MTMVideoDiT`: subclass of `WanVideoDiT` with three patch convs (short / mid / long),
      shared RoPE frame indices, MTM attention mask builder, and overridden `pre_dit / post_dit`.
    - `MTMMoT`: subclass of `MoT` that threads `layer_idx + mtm_history_seq_len` into
      `_build_expert_attention_io` to apply per-head learnable history-key amplification.
    - `FastWAMMemory`: subclass of `FastWAM` that wires history fields from the dataloader
      sample into `pre_dit`, builds the MTM joint attention mask, and routes inference paths.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from fastwam.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .fastwam import FastWAM
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT
from .wan_video_dit import (
    WanVideoDiT,
    create_group_causal_attn_mask,
    precompute_freqs_cis_3d,
    sinusoidal_embedding_1d,
)

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# 3D conv padding / center down-sample helpers (ported from helios).
# -----------------------------------------------------------------------------


def pad_for_3d_conv(x: torch.Tensor, kernel_size: Tuple[int, int, int]) -> torch.Tensor:
    """Replicate-pad a 5D tensor `(B, C, T, H, W)` so that `T/H/W` are multiples of `kernel_size`.

    Mirrors helios' `pad_for_3d_conv` so that mid / long history latents can be cleanly fed into
    `Conv3d(stride=kernel_size)` without losing trailing voxels. Padding is applied to the right
    end of each spatial-temporal axis to keep the leading index aligned with global frame index 0.
    """
    if x.dim() != 5:
        raise ValueError(f"`pad_for_3d_conv` expects 5D `(B,C,T,H,W)`, got shape {tuple(x.shape)}")
    kt, kh, kw = int(kernel_size[0]), int(kernel_size[1]), int(kernel_size[2])
    if kt <= 0 or kh <= 0 or kw <= 0:
        raise ValueError(f"`kernel_size` entries must be positive, got {kernel_size}")
    _, _, t, h, w = x.shape
    pad_t = (-t) % kt
    pad_h = (-h) % kh
    pad_w = (-w) % kw
    if pad_t == 0 and pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (0, pad_w, 0, pad_h, 0, pad_t), mode="replicate")


def center_down_sample_3d(x: torch.Tensor, ratio: Tuple[int, int, int]) -> torch.Tensor:
    """Average-pool a real-valued tensor `(T, H, W, D)` along T/H/W by `ratio`.

    Used to downsample dense RoPE frequencies (treated as real `head_dim` channels) onto the coarse
    mid / long token grid so that all segments share the same spatial/temporal coordinate basis.
    """
    if x.dim() != 4:
        raise ValueError(f"`center_down_sample_3d` expects 4D `(T,H,W,D)`, got shape {tuple(x.shape)}")
    rt, rh, rw = int(ratio[0]), int(ratio[1]), int(ratio[2])
    if rt <= 0 or rh <= 0 or rw <= 0:
        raise ValueError(f"`ratio` entries must be positive, got {ratio}")
    if rt == 1 and rh == 1 and rw == 1:
        return x
    # Treat the trailing dim as channel for avg_pool3d. Insert batch dim for the API.
    pooled = F.avg_pool3d(
        x.permute(3, 0, 1, 2).unsqueeze(0).contiguous(),
        kernel_size=(rt, rh, rw),
        stride=(rt, rh, rw),
    )
    return pooled.squeeze(0).permute(1, 2, 3, 0).contiguous()


def _downsample_freqs_cis(
    freqs_cis: torch.Tensor, ratio: Tuple[int, int, int]
) -> torch.Tensor:
    """Down-sample a complex RoPE frequency grid `(T, H, W, head_dim_part)` along T/H/W.

    Real and imaginary parts are pooled independently. The result is no longer unit-modulus, but
    matches helios behavior and applies as a (slightly) attenuated phase rotation in `rope_apply`.
    """
    if not torch.is_complex(freqs_cis):
        raise ValueError("`_downsample_freqs_cis` expects complex-valued freqs.")
    real = center_down_sample_3d(freqs_cis.real.to(torch.float32), ratio)
    imag = center_down_sample_3d(freqs_cis.imag.to(torch.float32), ratio)
    return torch.complex(real, imag).to(freqs_cis.dtype)


# -----------------------------------------------------------------------------
# MTMVideoDiT: WanVideoDiT + multi-term memory (three patches + shared RoPE).
# -----------------------------------------------------------------------------


class MTMVideoDiT(WanVideoDiT):
    """`WanVideoDiT` extended with helios-style multi-term memory.

    Adds:
      - `patch_mid` (kernel `(2,4,4)`) and `patch_long` (`(4,8,8)`) Conv3d for compressed history.
      - Optional per-layer-per-head learnable `history_key_scale_logit` for `is_amplify_history`.
      - `pre_dit` builds a `[long | mid | short | pred]` token sequence with helios-style shared RoPE
        frame indices `[0..15] | [16,17] | [18] | [19, 19+T_pred-1]`, and per-token zero-history-timestep
        AdaLN modulation when `mtm_zero_history_timestep=True`.
      - `post_dit` slices off history tokens and only un-patchifies the pred segment.
      - `build_mtm_self_attn_mask` returns the joint self-attention visibility matrix.

    The `patch_short` path is *not* a separate module: the user-confirmed contract puts the most
    recent clean latent (`current(1)`) and the noisy `pred(T_pred)` on the same `(1,2,2)` patch grid,
    so we reuse `self.patch_embedding` for both segments. Only mid / long need their own conv.
    """

    def __init__(
        self,
        *args: Any,
        multi_term_memory: bool = False,
        mtm_history_sizes: Tuple[int, int, int] = (16, 2, 1),
        mtm_anchor_size: int = 0,
        mtm_patch_kernel_long: Tuple[int, int, int] = (4, 8, 8),
        mtm_patch_kernel_mid: Tuple[int, int, int] = (2, 4, 4),
        mtm_patch_kernel_current: Tuple[int, int, int] = (1, 2, 2),
        mtm_pred_size: int = 2,
        mtm_amplify_history: bool = False,
        mtm_zero_history_timestep: bool = True,
        **kwargs: Any,
    ) -> None:
        # The base class hard-asserts `fuse_vae_embedding_in_latents=True`; in MTM mode we want
        # to *disable* it at runtime (no fixed first-frame anchor). Force-pass True to satisfy
        # the assertion, then flip the attribute below if the caller actually requested False.
        requested_fuse = kwargs.get("fuse_vae_embedding_in_latents", True)
        if multi_term_memory and not requested_fuse:
            kwargs["fuse_vae_embedding_in_latents"] = True
        super().__init__(*args, **kwargs)

        self.multi_term_memory = bool(multi_term_memory)
        self.mtm_history_sizes = tuple(int(s) for s in mtm_history_sizes)
        self.mtm_anchor_size = int(mtm_anchor_size)
        self.mtm_patch_kernel_long = tuple(int(k) for k in mtm_patch_kernel_long)
        self.mtm_patch_kernel_mid = tuple(int(k) for k in mtm_patch_kernel_mid)
        self.mtm_patch_kernel_current = tuple(int(k) for k in mtm_patch_kernel_current)
        self.mtm_pred_size = int(mtm_pred_size)
        self.mtm_amplify_history = bool(mtm_amplify_history)
        self.mtm_zero_history_timestep = bool(mtm_zero_history_timestep)

        if not self.multi_term_memory:
            return

        if len(self.mtm_history_sizes) != 3:
            raise ValueError(
                f"`mtm_history_sizes` must be a length-3 tuple `(long, mid, current)`, "
                f"got {self.mtm_history_sizes}"
            )
        long_size, mid_size, curr_size = self.mtm_history_sizes
        if self.mtm_anchor_size < 0:
            raise ValueError(f"`mtm_anchor_size` must be non-negative, got {self.mtm_anchor_size}")
        if long_size < 0 or mid_size < 0:
            raise ValueError(
                f"`mtm_history_sizes` long/mid must be >= 0, got {self.mtm_history_sizes}"
            )
        if curr_size < 1:
            raise ValueError(
                f"`mtm_history_sizes` current (last element) must be >= 1, got {curr_size}"
            )
        if self.mtm_pred_size <= 0:
            raise ValueError(f"`mtm_pred_size` must be positive, got {self.mtm_pred_size}")
        if not self.seperated_timestep:
            raise ValueError(
                "MTM requires `seperated_timestep=True` so each token can have its own timestep."
            )

        # Validate patch kernels against base patch_size.
        base_ps = tuple(self.patch_size)
        if self.mtm_patch_kernel_current != base_ps:
            raise ValueError(
                f"`mtm_patch_kernel_current` must equal `patch_size` {base_ps}, "
                f"got {self.mtm_patch_kernel_current}. Current segment shares `patch_embedding` with pred."
            )
        for name, kernel in [("long", self.mtm_patch_kernel_long), ("mid", self.mtm_patch_kernel_mid)]:
            if len(kernel) != 3 or any(k <= 0 for k in kernel):
                raise ValueError(f"`mtm_patch_kernel_{name}` must be 3 positive ints, got {kernel}")
            kt, kh, kw = kernel
            if kh != kw:
                raise ValueError(
                    f"`mtm_patch_kernel_{name}` spatial dims must be equal (square), got ({kh}, {kw})"
                )
            if kh % base_ps[1] != 0 or kw % base_ps[2] != 0 or kt % base_ps[0] != 0:
                raise ValueError(
                    f"`mtm_patch_kernel_{name}` {kernel} must be integer multiples of "
                    f"`patch_size` {base_ps}"
                )

        # Precompute temporal/spatial ratios for RoPE downsampling.
        base_t, base_h, _ = base_ps
        self._long_temporal_ratio = self.mtm_patch_kernel_long[0] // base_t
        self._long_spatial_ratio = self.mtm_patch_kernel_long[1] // base_h
        self._mid_temporal_ratio = self.mtm_patch_kernel_mid[0] // base_t
        self._mid_spatial_ratio = self.mtm_patch_kernel_mid[1] // base_h

        # Honor the original request: in MTM mode we never run the I2V first-frame fusion path.
        if multi_term_memory and not requested_fuse:
            self.fuse_vae_embedding_in_latents = False

        # Three-patch embedding: short branch reuses `patch_embedding` (same kernel as pred).
        self.patch_mid = nn.Conv3d(
            self.in_dim, self.hidden_dim,
            kernel_size=self.mtm_patch_kernel_mid, stride=self.mtm_patch_kernel_mid,
        )
        self.patch_long = nn.Conv3d(
            self.in_dim, self.hidden_dim,
            kernel_size=self.mtm_patch_kernel_long, stride=self.mtm_patch_kernel_long,
        )
        self._init_mtm_patches_from_patch_embedding()

        if self.mtm_amplify_history:
            num_layers = len(self.blocks)
            # Match helios: init logit=1 -> scale = 1 + 9 * sigmoid(1) ≈ 7.58.
            self.history_key_scale_logit = nn.Parameter(
                torch.ones(num_layers, self.num_heads)
            )

    def _init_mtm_patches_from_patch_embedding(self) -> None:
        """Copy `patch_embedding` weights into `patch_mid / patch_long` with spatial replication.

        Mirrors helios' `initialize_weight_from_another_conv3d` so that on a constant input the mid /
        long convs initially produce the same per-voxel mean as `patch_embedding`. The bias is copied
        verbatim; weight is repeated along T/H/W and divided by the replication factor.
        """
        weight = self.patch_embedding.weight.detach().clone()  # (D, C, base_t, base_h, base_w)
        bias = self.patch_embedding.bias.detach().clone()
        base_ps = tuple(self.patch_size)
        if weight.shape[2:] != base_ps:
            raise ValueError(
                f"Expected `patch_embedding` kernel `{base_ps}`, got {tuple(weight.shape[2:])}"
            )
        rt_m, rs_m = self._mid_temporal_ratio, self._mid_spatial_ratio
        rt_l, rs_l = self._long_temporal_ratio, self._long_spatial_ratio
        mid_factor = float(rt_m * rs_m * rs_m)
        long_factor = float(rt_l * rs_l * rs_l)
        with torch.no_grad():
            self.patch_mid.weight.copy_(
                repeat(weight, f"d c t h w -> d c (t {rt_m}) (h {rs_m}) (w {rs_m})") / mid_factor
            )
            self.patch_mid.bias.copy_(bias)
            self.patch_long.weight.copy_(
                repeat(weight, f"d c t h w -> d c (t {rt_l}) (h {rs_l}) (w {rs_l})") / long_factor
            )
            self.patch_long.bias.copy_(bias)

    def build_mtm_self_attn_mask(
        self,
        L_long: int,
        L_mid: int,
        L_curr: int,
        L_pred: int,
        tokens_per_frame_pred: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return the joint `(total, total)` boolean self-attention mask for `[history | pred]`.

        Following the user-confirmed `pred_only_mode`:
          - history (`long + mid + current`) attends bidirectionally within itself.
          - pred attends to the full history (`pred -> history = True`).
          - pred attends within itself per `video_attention_mask_mode` (`bidirectional` /
            `per_frame_causal` / `first_frame_causal`).
          - history *cannot* see pred (single-direction guidance).
        """
        if min(L_long, L_mid, L_curr, L_pred) < 0:
            raise ValueError(
                f"All segment lengths must be non-negative, got long={L_long}, mid={L_mid}, "
                f"curr={L_curr}, pred={L_pred}"
            )
        L_hist = L_long + L_mid + L_curr
        total = L_hist + L_pred
        mask = torch.zeros((total, total), dtype=torch.bool, device=device)
        if L_hist > 0:
            mask[:L_hist, :L_hist] = True
            if L_pred > 0:
                mask[L_hist:, :L_hist] = True
        if L_pred > 0:
            mask[L_hist:, L_hist:] = self.build_video_to_video_mask(
                video_seq_len=L_pred,
                video_tokens_per_frame=tokens_per_frame_pred,
                device=device,
            )
        return mask

    def _build_segment_freqs(
        self,
        frame_indices: torch.Tensor,
        h: int,
        w: int,
        spatial_ratio: int,
        temporal_ratio: int,
    ) -> torch.Tensor:
        """Build RoPE freqs for one history/pred segment using a shared `(h, w)` spatial basis.

        Produces freqs at the dense grid `(len(frame_indices), h, w, head_dim/2)`, then average-pools
        by `(temporal_ratio, spatial_ratio, spatial_ratio)` so the result matches the segment's
        post-patch token count exactly.

        Returns a `(L_seg, 1, head_dim/2)` complex tensor ready for `torch.cat` along the seq axis.
        """
        f_freqs_cis, h_freqs_cis, w_freqs_cis = self.freqs
        device = frame_indices.device
        f_part = f_freqs_cis.to(device=device).index_select(0, frame_indices)  # (T_dense, dim_t)
        h_part = h_freqs_cis.to(device=device)[:h]  # (h, dim_h)
        w_part = w_freqs_cis.to(device=device)[:w]  # (w, dim_w)
        t_dense = int(frame_indices.shape[0])
        f_grid = f_part.view(t_dense, 1, 1, -1).expand(t_dense, h, w, -1)
        h_grid = h_part.view(1, h, 1, -1).expand(t_dense, h, w, -1)
        w_grid = w_part.view(1, 1, w, -1).expand(t_dense, h, w, -1)
        dense = torch.cat([f_grid, h_grid, w_grid], dim=-1)  # (T_dense, h, w, head_dim/2) complex
        if temporal_ratio > 1 or spatial_ratio > 1:
            # Pad before pool so trailing voxels survive (mirrors `pad_for_3d_conv`).
            dense_real = dense.real.to(torch.float32).permute(3, 0, 1, 2).unsqueeze(0)
            dense_imag = dense.imag.to(torch.float32).permute(3, 0, 1, 2).unsqueeze(0)
            pad_t = (-t_dense) % temporal_ratio
            pad_h = (-h) % spatial_ratio
            pad_w = (-w) % spatial_ratio
            if pad_t or pad_h or pad_w:
                dense_real = F.pad(dense_real, (0, pad_w, 0, pad_h, 0, pad_t), mode="replicate")
                dense_imag = F.pad(dense_imag, (0, pad_w, 0, pad_h, 0, pad_t), mode="replicate")
            kernel = (temporal_ratio, spatial_ratio, spatial_ratio)
            real_pool = F.avg_pool3d(dense_real, kernel_size=kernel, stride=kernel)
            imag_pool = F.avg_pool3d(dense_imag, kernel_size=kernel, stride=kernel)
            real_pool = real_pool.squeeze(0).permute(1, 2, 3, 0).contiguous()
            imag_pool = imag_pool.squeeze(0).permute(1, 2, 3, 0).contiguous()
            dense = torch.complex(real_pool, imag_pool).to(f_freqs_cis.dtype)
        seg_t, seg_h, seg_w, _ = dense.shape
        return dense.reshape(seg_t * seg_h * seg_w, 1, -1)

    def _validate_mtm_inputs(
        self,
        x: torch.Tensor,
        history_anchors: torch.Tensor,
        history_long: torch.Tensor,
        history_mid: torch.Tensor,
        history_short: torch.Tensor,
    ) -> None:
        if (x.dim() != 5 or history_anchors.dim() != 5 or history_long.dim() != 5
                or history_mid.dim() != 5 or history_short.dim() != 5):
            raise ValueError(
                "All MTM tensors must be 5D `(B,C,T,H,W)`; got "
                f"x={tuple(x.shape)}, anchors={tuple(history_anchors.shape)}, "
                f"long={tuple(history_long.shape)}, "
                f"mid={tuple(history_mid.shape)}, short={tuple(history_short.shape)}"
            )
        if x.shape[2] != self.mtm_pred_size:
            raise ValueError(
                f"`x` must have T={self.mtm_pred_size} (pred), got T={x.shape[2]}"
            )
        long_size, mid_size, curr_size = self.mtm_history_sizes
        if history_anchors.shape[2] != self.mtm_anchor_size:
            raise ValueError(
                f"`history_anchors` T must be {self.mtm_anchor_size}, "
                f"got {history_anchors.shape[2]}"
            )
        if history_long.shape[2] != long_size:
            raise ValueError(f"`history_long` T must be {long_size}, got {history_long.shape[2]}")
        if history_mid.shape[2] != mid_size:
            raise ValueError(f"`history_mid` T must be {mid_size}, got {history_mid.shape[2]}")
        if history_short.shape[2] != curr_size:
            raise ValueError(
                f"`history_short` T must be {curr_size}, got {history_short.shape[2]}"
            )
        ref = (x.shape[0], x.shape[1], x.shape[3], x.shape[4])
        for name, tensor in (
            ("history_anchors", history_anchors),
            ("history_long", history_long),
            ("history_mid", history_mid),
            ("history_short", history_short),
        ):
            if (tensor.shape[0], tensor.shape[1], tensor.shape[3], tensor.shape[4]) != ref:
                raise ValueError(
                    f"`{name}` must share `(B, C, H, W)` with `x` `{ref}`, "
                    f"got {(tensor.shape[0], tensor.shape[1], tensor.shape[3], tensor.shape[4])}"
                )

    def pre_dit(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        fuse_vae_embedding_in_latents: bool = False,
        control_camera_latents_input: Optional[torch.Tensor] = None,
        history_anchors: Optional[torch.Tensor] = None,
        history_long: Optional[torch.Tensor] = None,
        history_mid: Optional[torch.Tensor] = None,
        history_short: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        # Fall back to base pre_dit when MTM is disabled or no history tensors provided.
        no_history = history_long is None and history_mid is None and history_short is None
        if (not self.multi_term_memory) or no_history:
            return super().pre_dit(
                x=x,
                timestep=timestep,
                context=context,
                context_mask=context_mask,
                action=action,
                fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
                control_camera_latents_input=control_camera_latents_input,
            )

        if history_long is None or history_mid is None or history_short is None:
            raise ValueError(
                "MTM mode requires all three history segments."
            )
        if history_anchors is None:
            if self.mtm_anchor_size:
                raise ValueError("MTM mode requires `history_anchors` when anchors are configured.")
            history_anchors = history_short[:, :, :0]
        if fuse_vae_embedding_in_latents:
            raise ValueError(
                "MTM mode is incompatible with `fuse_vae_embedding_in_latents=True`."
            )
        if control_camera_latents_input is not None:
            raise NotImplementedError("MTM mode does not support `control_camera_latents_input` yet.")

        # Reuse base validation for `(x, timestep, context, context_mask, action)` shape contracts.
        x, timestep, context_mask = self._validate_forward_inputs(
            x=x,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=action,
        )
        self._validate_mtm_inputs(
            x, history_anchors, history_long, history_mid, history_short)

        batch_size = x.shape[0]
        patch_h = int(self.patch_size[1])
        patch_w = int(self.patch_size[2])
        if x.shape[3] % patch_h != 0 or x.shape[4] % patch_w != 0:
            raise ValueError(
                "Latent spatial shape must be divisible by DiT patch size, "
                f"got HxW=({x.shape[3]}, {x.shape[4]}), patch=({patch_h}, {patch_w})"
            )
        tokens_per_frame_pred = (x.shape[3] // patch_h) * (x.shape[4] // patch_w)

        # ---- 1. Patchify each segment via its dedicated conv. -----------------
        x_pred = self.patchify(x)  # (B, D, T_pred, h, w)
        x_anchors = (
            self.patch_embedding(history_anchors)  # same detail as current
            if self.mtm_anchor_size else None
        )
        x_short = self.patch_embedding(history_short)  # (B, D, S, h, w)

        f_pred, h, w = x_pred.shape[2], x_pred.shape[3], x_pred.shape[4]
        f_short, h_short, w_short = x_short.shape[2], x_short.shape[3], x_short.shape[4]
        if (h_short, w_short) != (h, w):
            raise ValueError(
                f"`patch_short` spatial shape ({h_short},{w_short}) must match pred ({h},{w})."
            )

        long_size, mid_size, curr_size = self.mtm_history_sizes
        device = x.device
        segments_tokens: list[torch.Tensor] = []
        segments_freqs: list[torch.Tensor] = []

        # Fixed episode anchors use the same (1,2,2) projection as current/pred.
        L_anchor = 0
        if x_anchors is not None:
            f_anchor, h_anchor, w_anchor = x_anchors.shape[2:]
            if (h_anchor, w_anchor) != (h, w):
                raise ValueError(
                    f"`patch_anchor` spatial shape ({h_anchor},{w_anchor}) must match pred ({h},{w})."
                )
            L_anchor = f_anchor * h_anchor * w_anchor
            segments_tokens.append(
                rearrange(x_anchors, "b d t h w -> b (t h w) d"))
            idx_anchor = torch.arange(0, self.mtm_anchor_size, device=device)
            segments_freqs.append(self._build_segment_freqs(
                frame_indices=idx_anchor, h=h, w=w,
                spatial_ratio=1, temporal_ratio=1))

        # Long segment (skipped when long_size == 0).
        if long_size > 0:
            x_long_in = pad_for_3d_conv(history_long, self.mtm_patch_kernel_long)
            x_long = self.patch_long(x_long_in)
            f_long, h_long, w_long = x_long.shape[2], x_long.shape[3], x_long.shape[4]
            L_long = f_long * h_long * w_long
            segments_tokens.append(rearrange(x_long, "b d t h w -> b (t h w) d"))
            idx_long = torch.arange(
                self.mtm_anchor_size, self.mtm_anchor_size + long_size, device=device)
            freqs_long = self._build_segment_freqs(
                frame_indices=idx_long, h=h, w=w,
                spatial_ratio=self._long_spatial_ratio,
                temporal_ratio=self._long_temporal_ratio,
            )
            if freqs_long.shape[0] != L_long:
                raise RuntimeError(
                    f"long freqs len {freqs_long.shape[0]} != L_long {L_long} (downsample mismatch)"
                )
            segments_freqs.append(freqs_long)
        else:
            L_long = 0

        # Mid segment (skipped when mid_size == 0).
        if mid_size > 0:
            x_mid_in = pad_for_3d_conv(history_mid, self.mtm_patch_kernel_mid)
            x_mid = self.patch_mid(x_mid_in)
            f_mid, h_mid, w_mid = x_mid.shape[2], x_mid.shape[3], x_mid.shape[4]
            L_mid = f_mid * h_mid * w_mid
            segments_tokens.append(rearrange(x_mid, "b d t h w -> b (t h w) d"))
            idx_mid = torch.arange(
                self.mtm_anchor_size + long_size,
                self.mtm_anchor_size + long_size + mid_size, device=device)
            freqs_mid = self._build_segment_freqs(
                frame_indices=idx_mid, h=h, w=w,
                spatial_ratio=self._mid_spatial_ratio,
                temporal_ratio=self._mid_temporal_ratio,
            )
            if freqs_mid.shape[0] != L_mid:
                raise RuntimeError(
                    f"mid freqs len {freqs_mid.shape[0]} != L_mid {L_mid} (downsample mismatch)"
                )
            segments_freqs.append(freqs_mid)
        else:
            L_mid = 0

        # Short (current) segment — always present (curr_size >= 1).
        L_curr = f_short * h_short * w_short
        segments_tokens.append(rearrange(x_short, "b d t h w -> b (t h w) d"))
        idx_curr = torch.arange(
            self.mtm_anchor_size + long_size + mid_size,
            self.mtm_anchor_size + long_size + mid_size + curr_size, device=device
        )
        freqs_short = self._build_segment_freqs(
            frame_indices=idx_curr, h=h, w=w, spatial_ratio=1, temporal_ratio=1
        )
        segments_freqs.append(freqs_short)

        # Pred segment — always present.
        L_pred = f_pred * h * w
        segments_tokens.append(rearrange(x_pred, "b d t h w -> b (t h w) d"))
        pred_start = self.mtm_anchor_size + long_size + mid_size + curr_size
        idx_pred = torch.arange(pred_start, pred_start + f_pred, device=device)
        freqs_pred = self._build_segment_freqs(
            frame_indices=idx_pred, h=h, w=w, spatial_ratio=1, temporal_ratio=1
        )
        segments_freqs.append(freqs_pred)

        L_hist = L_anchor + L_long + L_mid + L_curr
        total_seq = L_hist + L_pred
        tokens = torch.cat(segments_tokens, dim=1).contiguous()
        freqs = torch.cat(segments_freqs, dim=0).to(device)

        # ---- 3. Per-token timestep with zero-history-timestep. ----------------
        token_timesteps = torch.empty(
            (batch_size, total_seq), dtype=timestep.dtype, device=timestep.device
        )
        if self.mtm_zero_history_timestep:
            if L_hist > 0:
                token_timesteps[:, :L_hist] = 0
        else:
            token_timesteps[:, :L_hist] = timestep.view(batch_size, 1)
        token_timesteps[:, L_hist:] = timestep.view(batch_size, 1)
        t_emb = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, token_timesteps.reshape(-1))
        ).reshape(batch_size, total_seq, self.hidden_dim)
        t_mod = self.time_projection(t_emb).unflatten(2, (6, self.hidden_dim))
        t_for_head = t_emb[:, L_hist:, :].contiguous()

        # ---- 4. Text / action context + per-token visibility mask. ------------
        context = self.text_embedding(context)
        context_len = context.shape[1]
        if self.action_conditioned and action is not None:
            num_temporal_groups = self.mtm_pred_size - 1
            if num_temporal_groups <= 0:
                raise ValueError(
                    "Action-conditioned MTM requires `mtm_pred_size >= 2`, "
                    f"got {self.mtm_pred_size}."
                )
            if action.shape[1] % num_temporal_groups != 0:
                raise ValueError(
                    f"`action` length {action.shape[1]} must be divisible by `pred-1`={num_temporal_groups}."
                )
            action_len = action.shape[1]
            action_emb = self.action_embedding(action)
            action_pos = sinusoidal_embedding_1d(
                self.hidden_dim,
                torch.arange(action_len, device=action_emb.device),
            )
            action_emb = action_emb + action_pos.unsqueeze(0)
            context = torch.cat([context, action_emb], dim=1)
            action_group_mask = create_group_causal_attn_mask(
                num_temporal_groups=num_temporal_groups,
                num_query_per_group=tokens_per_frame_pred,
                num_key_per_group=action_len // num_temporal_groups,
                mode=self.action_group_causal_mask_mode,
            ).to(device=context.device)
            final_context_mask = torch.zeros(
                (batch_size, total_seq, context.shape[1]), dtype=torch.bool, device=context.device
            )
            final_context_mask[:, :, :context_len] = context_mask.unsqueeze(1).expand(
                -1, total_seq, -1
            )
            # Action key visible only to pred frames `[1:]` (skip the first pred frame, like base).
            pred_action_start = L_hist + tokens_per_frame_pred
            final_context_mask[:, pred_action_start:total_seq, context_len:] = action_group_mask.unsqueeze(0).expand(
                batch_size, -1, -1
            )
            final_context_mask = final_context_mask
        else:
            if self.action_conditioned and action is None:
                raise ValueError(
                    "Action-conditioned MTM requires non-null `action`. "
                    "Pure-text fallback (`f==1`) is not supported in MTM mode."
                )
            final_context_mask = context_mask.unsqueeze(1).expand(-1, total_seq, -1)

        return {
            "tokens": tokens,
            "freqs": freqs,
            "t": t_for_head,
            "t_mod": t_mod,
            "context": context,
            "context_mask": final_context_mask,
            "meta": {
                "grid_size": (f_pred, h, w),  # kept for back-compat (post_dit fallback)
                "tokens_per_frame": tokens_per_frame_pred,
                "batch_size": batch_size,
                "mtm": {
                    "L_anchor": L_anchor,
                    "L_long": L_long,
                    "L_mid": L_mid,
                    "L_curr": L_curr,
                    "L_pred": L_pred,
                    "L_hist": L_hist,
                    "tokens_per_frame_pred": tokens_per_frame_pred,
                    "grid_size_pred": (f_pred, h, w),
                },
            },
        }

    def post_dit(self, x_tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        mtm_meta = pre_state["meta"].get("mtm")
        if mtm_meta is None:
            return super().post_dit(x_tokens, pre_state)
        L_hist = int(mtm_meta["L_hist"])
        f, h, w = mtm_meta["grid_size_pred"]
        x_tokens = x_tokens[:, L_hist:, :].contiguous()
        x = self.head(x_tokens, pre_state["t"])
        return self.unpatchify(x, (f, h, w))


