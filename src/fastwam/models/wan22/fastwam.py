from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from fastwam.memory.native_cache import (
    CacheUnit,
    DynamicLayerwiseMemoryState,
    LayerwiseBlockMemory,
    LayerwiseMemoryState,
    MemoryCarry,
    NativeBlock,
    NativeBlockCompressor,
    NativeCacheState,
    build_dynamic_layerwise_training_layout,
    build_layerwise_training_layout,
    build_recursive_layerwise_training_layout,
    partition_layerwise_history,
)
from fastwam.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler

logger = get_logger(__name__)


def _decode_dynamic_memory_plan(
    sample: dict[str, Any],
) -> tuple[tuple[tuple[int, ...], ...] | None, tuple[int, ...] | None]:
    """Decode batch-size-one ranges and their explicit memory-token budgets."""
    ranges = sample.get("history_memory_groups")
    count = sample.get("history_memory_group_count")
    token_counts = sample.get("history_memory_token_counts")
    if ranges is None and count is None:
        if token_counts is not None:
            raise ValueError("dynamic memory token counts require ranges/count")
        return None, None
    if not isinstance(ranges, torch.Tensor) or not isinstance(count, torch.Tensor):
        raise ValueError("dynamic memory group ranges/count must both be tensors")
    if ranges.ndim != 3 or ranges.shape[0] != 1 or ranges.shape[2] != 2:
        raise ValueError(
            "dynamic memory groups require batch size 1 and shape [1,S,2], "
            f"got {tuple(ranges.shape)}"
        )
    if count.numel() != 1:
        raise ValueError("dynamic memory group count requires batch size 1")
    expected_count = int(count.reshape(-1)[0].item())
    if expected_count != int(ranges.shape[1]):
        raise ValueError(
            f"dynamic memory group count {expected_count} does not match "
            f"range count {ranges.shape[1]}"
        )
    groups = []
    for start_tensor, stop_tensor in ranges[0]:
        start = int(start_tensor.item())
        stop = int(stop_tensor.item())
        if stop <= start:
            raise ValueError(f"invalid dynamic memory range [{start}, {stop})")
        groups.append(tuple(range(start, stop)))
    if token_counts is None:
        return tuple(groups), None
    if not isinstance(token_counts, torch.Tensor):
        raise ValueError("dynamic memory token counts must be a tensor")
    if token_counts.ndim != 2 or token_counts.shape[0] != 1:
        raise ValueError(
            "dynamic memory token counts require batch size 1 and shape [1,S], "
            f"got {tuple(token_counts.shape)}"
        )
    if int(token_counts.shape[1]) != expected_count:
        raise ValueError("dynamic memory token count length does not match groups")
    decoded_counts = tuple(int(value.item()) for value in token_counts[0])
    if any(value <= 0 for value in decoded_counts):
        raise ValueError("dynamic memory token counts must be positive")
    return tuple(groups), decoded_counts


def _decode_dynamic_memory_groups(
    sample: dict[str, Any],
) -> tuple[tuple[int, ...], ...] | None:
    """Compatibility wrapper returning only ranges."""

    groups, _ = _decode_dynamic_memory_plan(sample)
    return groups


class FastWAM(torch.nn.Module):
    """MoT world model with video/action experts."""

    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        video_training_sampling_scheme: str = "uniform",
        video_logit_mean: float = 0.0,
        video_logit_std: float = 1.0,
        video_training_weight_scheme: str = "fastwam",
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        action_training_sampling_scheme: str = "uniform",
        action_logit_mean: float = 0.0,
        action_logit_std: float = 1.0,
        action_training_weight_scheme: str = "fastwam",
        action_rope_spatial_mode: str = "memorywam",
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        native_cache: Optional[dict[str, Any]] = None,
    ):
        super().__init__()
        if mot.mixtures["video"] is not video_expert:
            raise ValueError("`video_expert` must be the video expert registered by `mot`.")
        if mot.mixtures["action"] is not action_expert:
            raise ValueError("`action_expert` must be the action expert registered by `mot`.")
        self.mot = mot

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError("`text_dim` is required when `text_encoder` is not loaded.")
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
        else:
            self.proprio_encoder = None

        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
            training_sampling_scheme=video_training_sampling_scheme,
            logit_mean=video_logit_mean,
            logit_std=video_logit_std,
            training_weight_scheme=video_training_weight_scheme,
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
            training_sampling_scheme=action_training_sampling_scheme,
            logit_mean=action_logit_mean,
            logit_std=action_logit_std,
            training_weight_scheme=action_training_weight_scheme,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )
        if action_rope_spatial_mode not in {"memorywam", "center", "origin"}:
            raise ValueError(
                "`action_rope_spatial_mode` must be 'memorywam', 'center', or 'origin', "
                f"got {action_rope_spatial_mode!r}"
            )
        self.action_rope_spatial_mode = action_rope_spatial_mode
        # Optional aliases for consistency with Wan22Core naming.
        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)
        self.native_cache_compressor = None
        self.layerwise_block_memory = None
        if native_cache is not None and bool(native_cache.get("enabled", True)):
            cache_mode = str(native_cache.get("mode", "legacy"))
            if cache_mode == "layerwise":
                self.layerwise_block_memory = LayerwiseBlockMemory(
                    self.video_expert,
                    memory_tokens=int(native_cache.get("memory_tokens", 32)),
                    dynamic_tokens_per_frame=(
                        None
                        if native_cache.get("dynamic_tokens_per_frame") is None
                        else int(native_cache.get("dynamic_tokens_per_frame"))
                    ),
                    allocation_mode=str(native_cache.get("allocation_mode", "span_full")),
                    group_size=int(native_cache.get("group_size", 4)),
                    anchor_frames=int(native_cache.get("anchor_frames", 2)),
                    recent_frames=int(native_cache.get("recent_frames", 4)),
                    reader_use_anchor=bool(native_cache.get("reader_use_anchor", True)),
                    reader_use_memory=bool(native_cache.get("reader_use_memory", True)),
                    reader_use_recent=bool(native_cache.get("reader_use_recent", True)),
                    recursive=bool(native_cache.get("recursive", False)),
                    max_levels=int(native_cache.get("max_levels", 8)),
                )
            elif cache_mode == "legacy":
                self.native_cache_compressor = NativeBlockCompressor(
                    self.video_expert.blocks[0],
                    group_size=int(native_cache.get("group_size", 4)),
                    alpha_max=float(native_cache.get("alpha_max", 1.0)),
                    max_levels=int(native_cache.get("max_levels", 8)),
                )
            else:
                raise ValueError(f"Unsupported native cache mode: {cache_mode!r}")

        self.to(self.device)

    @property
    def video_expert(self):
        """Video expert view without registering a duplicate module path."""
        return self.mot.mixtures["video"]

    @property
    def action_expert(self):
        """Action expert view without registering a duplicate module path."""
        return self.mot.mixtures["action"]

    @property
    def dit(self):
        """Backward-compatible trainer alias without duplicate registration."""
        return self.mot

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        video_training_sampling_scheme: str = "uniform",
        video_logit_mean: float = 0.0,
        video_logit_std: float = 1.0,
        video_training_weight_scheme: str = "fastwam",
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        action_training_sampling_scheme: str = "uniform",
        action_logit_mean: float = 0.0,
        action_logit_std: float = 1.0,
        action_training_weight_scheme: str = "fastwam",
        action_rope_spatial_mode: str = "center",
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        native_cache: Optional[dict[str, Any]] = None,
    ):
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for FastWAM.from_wan22_pretrained().")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required for FastWAM.")

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )

        video_expert = components.dit
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")
        if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match video expert.")

        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            video_training_sampling_scheme=video_training_sampling_scheme,
            video_logit_mean=video_logit_mean,
            video_logit_std=video_logit_std,
            video_training_weight_scheme=video_training_weight_scheme,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            action_training_sampling_scheme=action_training_sampling_scheme,
            action_logit_mean=action_logit_mean,
            action_logit_std=action_logit_std,
            action_training_weight_scheme=action_training_weight_scheme,
            action_rope_spatial_mode=action_rope_spatial_mode,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            native_cache=native_cache,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
        }
        return model

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        return self

    @staticmethod
    def _check_resize_height_width(height, width, num_frames):
        if height % 16 != 0:
            height = (height + 15) // 16 * 16
        if width % 16 != 0:
            width = (width + 15) // 16 * 16
        if num_frames % 4 != 1:
            num_frames = (num_frames + 3) // 4 * 4 + 1
        return height, width, num_frames

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        # FIXME: original implementation's zero padding is visible in cross-attn.
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        return prompt_emb.to(device=self.device), mask

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}")
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
            )
        proprio_token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype) # [B, 1, D]
        proprio_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    @torch.no_grad()
    def _encode_video_latents(self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        z = self.vae.encode(
            video_tensor,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return z

    @torch.no_grad()
    def _encode_input_image_latents_tensor(self, input_image: torch.Tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim == 4:
            input_video = input_image[0].unsqueeze(1)
        elif input_image.ndim == 5:
            input_video = input_image[0]
        else:
            raise ValueError(
                "`input_image` must have shape [3,H,W], [1,3,H,W], or "
                f"[1,3,T,H,W], got {tuple(input_image.shape)}"
            )
        if (
            input_video.shape[0] != 3
            or input_video.shape[1] < 1
            or (input_video.shape[1] - 1) % 4 != 0
        ):
            raise ValueError(
                "Continuous temporal memory input must contain 1 + 4k frames, "
                f"got {tuple(input_video.shape)}"
            )
        input_video = input_video.to(device=self.device)
        z = self.vae.encode([input_video], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if isinstance(z, list):
            z = z[0].unsqueeze(0)
        return z

    def _build_video_aligned_action_freqs(
        self,
        *,
        action_seq_len: int,
        temporal_base: float,
        grid_h: int,
        grid_w: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Map action tokens into the VideoDiT 3D RoPE coordinate system.

        Action time positions cover the interval after the current observation
        latent. MemoryWAM uses fractional offsets i / (H + 1) and the spatial
        sentinel (-1, -1); center/origin remain available for controlled
        comparisons. All modes use the VideoDiT frequency basis.
        """
        if action_seq_len <= 0:
            raise ValueError("`action_seq_len` must be positive")
        video_freqs = self.video_expert.freqs
        if len(video_freqs) != 3 or any(freq.shape[0] < 2 for freq in video_freqs):
            raise ValueError("Video expert does not expose a valid 3D RoPE cache")

        dtype = torch.float32
        position_mode = getattr(self, "action_rope_spatial_mode", "memorywam")
        temporal_denominator = (
            action_seq_len + 1 if position_mode == "memorywam" else action_seq_len
        )
        positions = (
            float(temporal_base)
            + torch.arange(
                1,
                action_seq_len + 1,
                device=device,
                dtype=dtype,
            )
            / float(temporal_denominator)
        )
        temporal_unit_phase = torch.angle(video_freqs[0][1]).to(
            device=device,
            dtype=dtype,
        )
        temporal_phase = positions[:, None] * temporal_unit_phase[None, :]
        temporal_freqs = torch.polar(
            torch.ones_like(temporal_phase),
            temporal_phase,
        )

        spatial_mode = position_mode
        if spatial_mode == "center":
            marker_h = int(grid_h) // 2
            marker_w = int(grid_w) // 2
        elif spatial_mode == "origin":
            marker_h = 0
            marker_w = 0
        elif spatial_mode == "memorywam":
            height_freqs = video_freqs[1][1].to(device=device).conj()
            width_freqs = video_freqs[2][1].to(device=device).conj()
            height_freqs = height_freqs.unsqueeze(0).expand(action_seq_len, -1)
            width_freqs = width_freqs.unsqueeze(0).expand(action_seq_len, -1)
            return torch.cat(
                [temporal_freqs, height_freqs, width_freqs],
                dim=-1,
            ).unsqueeze(1)
        else:
            raise ValueError(f"Unsupported action RoPE spatial mode: {spatial_mode!r}")
        marker_h = min(max(marker_h, 0), video_freqs[1].shape[0] - 1)
        marker_w = min(max(marker_w, 0), video_freqs[2].shape[0] - 1)
        height_freqs = video_freqs[1][marker_h].to(device=device)
        width_freqs = video_freqs[2][marker_w].to(device=device)
        height_freqs = height_freqs.unsqueeze(0).expand(action_seq_len, -1)
        width_freqs = width_freqs.unsqueeze(0).expand(action_seq_len, -1)
        return torch.cat(
            [temporal_freqs, height_freqs, width_freqs],
            dim=-1,
        ).unsqueeze(1)

    def _decode_latents(self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        video_tensor = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    def build_inputs(self, sample, tiled: bool = False):
        video = sample["video"]
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError(
                "FastWAM training requires `sample['context']` and `sample['context_mask']`."
            )
        context = sample["context"]
        context_mask = sample["context_mask"]
        proprio = sample.get("proprio", None)
        if video.ndim != 5:
            raise ValueError(f"`sample['video']` must be 5D [B, 3, T, H, W], got shape {tuple(video.shape)}")
        if video.shape[1] != 3:
            raise ValueError(f"`sample['video']` channel dimension must be 3, got shape {tuple(video.shape)}")

        batch_size, _, num_frames, height, width = video.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"Video spatial dims must be multiples of 16, got H={height}, W={width}"
            )
        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
        if num_frames <= 1:
            raise ValueError(f"Video T must be > 1 for action-conditioned training, got T={num_frames}")

        if "action" not in sample:
            raise ValueError("`sample['action']` is required for FastWAM training.")

        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
        action_horizon = int(action.shape[1])
        if action_horizon % (num_frames - 1) != 0:
            raise ValueError(
                f"`sample['action']` temporal dimension must be divisible by video transitions ({num_frames - 1}), got {action_horizon}"
            )

        action_is_pad = sample.get("action_is_pad", None)
        if action_is_pad is not None:
            if action_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['action_is_pad']` must be 2D [B, T], got shape {tuple(action_is_pad.shape)}"
                )
            if action_is_pad.shape[0] != batch_size or action_is_pad.shape[1] != action_horizon:
                raise ValueError(
                    "`sample['action_is_pad']` shape mismatch: "
                    f"got {tuple(action_is_pad.shape)} vs expected ({batch_size}, {action_horizon})"
                )

        image_is_pad = sample.get("image_is_pad", None)
        if image_is_pad is not None:
            if image_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['image_is_pad']` must be 2D [B, T], got shape {tuple(image_is_pad.shape)}"
                )
            if image_is_pad.shape[0] != batch_size or image_is_pad.shape[1] != num_frames:
                raise ValueError(
                    "`sample['image_is_pad']` shape mismatch: "
                    f"got {tuple(image_is_pad.shape)} vs expected ({batch_size}, {num_frames})"
                )
        
        input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        input_latents = self._encode_video_latents(input_video, tiled=tiled)

        first_frame_latents = None
        fuse_flag = False
        if getattr(self.video_expert, "fuse_vae_embedding_in_latents", False):
            first_frame_latents = input_latents[:, :, 0:1]
            fuse_flag = True

        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        # Memory-style conditioning keeps proprioception private to the action
        # expert.  In particular, historical video tokens must not be
        # recomputed as though the current robot state had existed in the past.
        video_context = context
        video_context_mask = context_mask
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("`sample['proprio']` is required when `proprio_dim` is enabled.")
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")
            if proprio.shape[2] != self.proprio_dim:
                raise ValueError(
                    f"`sample['proprio']` last dim must be {self.proprio_dim}, got {proprio.shape[2]}"
                )
            proprio = proprio[:, 0, :] # [B, D]
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
            )
        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

        result = {
            "context": context,
            "context_mask": context_mask,
            "video_context": video_context,
            "video_context_mask": video_context_mask,
            "input_latents": input_latents,
            "first_frame_latents": first_frame_latents,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "action": action,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
        }
        history_latents = sample.get("history_latents")
        if history_latents is not None:
            if history_latents.ndim != 6:
                raise ValueError(
                    "`history_latents` must be [B,H,C,1,h,w], "
                    f"got {tuple(history_latents.shape)}"
                )
            if history_latents.shape[0] != batch_size:
                raise ValueError("History/current batch sizes must match")
            if history_latents.shape[2] != input_latents.shape[1]:
                raise ValueError("History/current latent channel counts must match")
            if history_latents.shape[3] != 1:
                raise ValueError("Every cached history observation must contain one latent frame")
            if tuple(history_latents.shape[-2:]) != tuple(input_latents.shape[-2:]):
                raise ValueError("History/current latent spatial shapes must match")
            result["history_latents"] = history_latents.to(
                device=self.device,
                dtype=self.torch_dtype,
                non_blocking=True,
            )
        return result

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        # video -> video
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        # action -> action
        mask[video_seq_len:, video_seq_len:] = True
        # action -> first-frame video only
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        mask[video_seq_len:, :first_frame_tokens] = True
        return mask

    @staticmethod
    @torch.no_grad()
    def _build_full_history_training_mask(
        clean_video_frames: int,
        noisy_video_frames: int,
        video_tokens_per_frame: int,
        action_seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """AR+diffusion mask for clean history, noisy future, and action.

        Clean observations are causal. Noisy future-video queries read all
        clean observations and their own diffusion group. Action queries read
        every clean observation and all action tokens, but never noisy future
        video, preventing test-time future leakage.
        """
        clean_video_frames = int(clean_video_frames)
        noisy_video_frames = int(noisy_video_frames)
        video_tokens_per_frame = int(video_tokens_per_frame)
        action_seq_len = int(action_seq_len)
        if min(
            clean_video_frames,
            noisy_video_frames,
            video_tokens_per_frame,
            action_seq_len,
        ) <= 0:
            raise ValueError("Full-history mask dimensions must all be positive")

        video_frames = clean_video_frames + noisy_video_frames
        video_seq_len = video_frames * video_tokens_per_frame
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros(
            (total_seq_len, total_seq_len),
            dtype=torch.bool,
            device=device,
        )

        for frame_idx in range(clean_video_frames):
            q = slice(
                frame_idx * video_tokens_per_frame,
                (frame_idx + 1) * video_tokens_per_frame,
            )
            visible_clean_end = (frame_idx + 1) * video_tokens_per_frame
            mask[q, :visible_clean_end] = True

        clean_token_count = clean_video_frames * video_tokens_per_frame
        noisy_token_count = noisy_video_frames * video_tokens_per_frame
        noisy_rows = slice(clean_token_count, clean_token_count + noisy_token_count)
        mask[noisy_rows, :clean_token_count + noisy_token_count] = True

        action_rows = slice(video_seq_len, total_seq_len)
        mask[action_rows, :clean_token_count] = True
        mask[action_rows, video_seq_len:total_seq_len] = True
        return mask

    def _consolidate_native_training_state(
        self,
        video_pre: dict[str, Any],
        *,
        clean_frame_count: int,
        noisy_frame_count: int,
    ) -> tuple[dict[str, Any], int]:
        """Compress completed level-0 history groups while keeping current raw."""
        compressor = self.native_cache_compressor
        if compressor is None:
            return video_pre, int(clean_frame_count)
        clean_frame_count = int(clean_frame_count)
        noisy_frame_count = int(noisy_frame_count)
        tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])
        total_frames = clean_frame_count + noisy_frame_count
        expected_tokens = total_frames * tokens_per_frame
        if int(video_pre["tokens"].shape[1]) != expected_tokens:
            raise ValueError(
                "video_pre token length does not match clean/noisy frame counts: "
                f"tokens={video_pre['tokens'].shape[1]}, expected={expected_tokens}"
            )

        compressible_frames = (
            (clean_frame_count - 1) // compressor.group_size
        ) * compressor.group_size
        group_count = compressible_frames // compressor.group_size
        if group_count == 0:
            return video_pre, clean_frame_count

        summary_tokens = []
        summary_freqs = []
        summary_t_mod = []
        summary_context_mask = []
        batch = int(video_pre["tokens"].shape[0])
        for group_index in range(group_count):
            frame_start = group_index * compressor.group_size
            frame_stop = frame_start + compressor.group_size
            token_start = frame_start * tokens_per_frame
            token_stop = frame_stop * tokens_per_frame
            blocks = video_pre["tokens"][:, token_start:token_stop].reshape(
                batch,
                compressor.group_size,
                tokens_per_frame,
                compressor.hidden_dim,
            )
            summary_tokens.append(
                compressor(
                    block_tokens=blocks,
                    block_freqs=video_pre["freqs"][token_start:token_stop],
                    levels=torch.zeros(
                        batch,
                        compressor.group_size,
                        dtype=torch.long,
                        device=blocks.device,
                    ),
                    spans=torch.ones(
                        batch,
                        compressor.group_size,
                        dtype=torch.long,
                        device=blocks.device,
                    ),
                )
            )
            endpoint_start = (frame_stop - 1) * tokens_per_frame
            endpoint_stop = frame_stop * tokens_per_frame
            summary_freqs.append(video_pre["freqs"][endpoint_start:endpoint_stop])
            summary_t_mod.append(video_pre["t_mod"][:, endpoint_start:endpoint_stop])
            summary_context_mask.append(
                video_pre["context_mask"][:, endpoint_start:endpoint_stop]
            )

        raw_start = compressible_frames * tokens_per_frame
        clean_stop = clean_frame_count * tokens_per_frame
        retained = dict(video_pre)
        retained["tokens"] = torch.cat(
            [*summary_tokens, video_pre["tokens"][:, raw_start:]], dim=1
        )
        retained["freqs"] = torch.cat(
            [*summary_freqs, video_pre["freqs"][raw_start:]], dim=0
        )
        retained["t_mod"] = torch.cat(
            [*summary_t_mod, video_pre["t_mod"][:, raw_start:]], dim=1
        )
        retained["context_mask"] = torch.cat(
            [
                *summary_context_mask,
                video_pre["context_mask"][:, raw_start:],
            ],
            dim=1,
        )
        clean_unit_count = group_count + (clean_frame_count - compressible_frames)
        _, grid_h, grid_w = video_pre["meta"]["grid_size"]
        retained["meta"] = dict(video_pre["meta"])
        retained["meta"]["grid_size"] = (
            clean_unit_count + noisy_frame_count,
            int(grid_h),
            int(grid_w),
        )
        retained["native_original_clean_token_count"] = clean_stop
        return retained, clean_unit_count

    def _pack_layerwise_memory_training_state(
        self,
        video_pre: dict[str, Any],
        *,
        clean_frame_count: int,
        noisy_frame_count: int,
        action_seq_len: int,
        memory_groups: tuple[tuple[int, ...], ...] | None = None,
        memory_token_counts: tuple[int, ...] | None = None,
    ):
        """Insert layerwise memory slots while retaining source tokens as helpers."""
        memory = self.layerwise_block_memory
        if memory is None:
            raise ValueError("Layerwise block memory is disabled")
        tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])
        layout_kwargs = {
            "clean_frames": clean_frame_count,
            "noisy_frames": noisy_frame_count,
            "tokens_per_frame": tokens_per_frame,
            "action_tokens": action_seq_len,
            "memory_tokens": memory.memory_tokens,
            "device": video_pre["tokens"].device,
        }
        if memory_groups is not None:
            if memory_token_counts is None:
                memory_token_counts = tuple(
                    memory.token_count_for_span(len(group)) for group in memory_groups
                )
            if len(memory_token_counts) != len(memory_groups):
                raise ValueError("memory token counts must match memory groups")
            layout = build_dynamic_layerwise_training_layout(
                memory_groups=memory_groups,
                memory_token_counts=memory_token_counts,
                anchor_frames=memory.anchor_frames,
                recent_frames=memory.recent_frames,
                reader_use_anchor=memory.reader_use_anchor,
                reader_use_memory=memory.reader_use_memory,
                reader_use_recent=memory.reader_use_recent,
                **layout_kwargs,
            )
        elif memory.recursive:
            layout_kwargs.update(
                anchor_frames=memory.anchor_frames,
                recent_frames=memory.recent_frames,
                group_size=memory.group_size,
                max_levels=memory.max_levels,
                recursive=True,
            )
            layout = build_recursive_layerwise_training_layout(**layout_kwargs)
        else:
            layout_kwargs.update(
                anchor_frames=memory.anchor_frames,
                recent_frames=memory.recent_frames,
                group_size=memory.group_size,
            )
            layout = build_layerwise_training_layout(**layout_kwargs)
        batch_size = int(video_pre["tokens"].shape[0])
        token_pieces = []
        freq_pieces = []
        t_mod_pieces = []
        context_mask_pieces = []
        for segment in layout.segments:
            if segment.kind == "memory":
                endpoint = int(segment.frame_indices[-1])
                endpoint_token = endpoint * tokens_per_frame
                memory_token_count = int(segment.stop) - int(segment.start)
                token_pieces.append(
                    memory.initial_tokens(
                        batch_size=batch_size,
                        device=video_pre["tokens"].device,
                        dtype=video_pre["tokens"].dtype,
                        level=int(segment.level),
                        token_count=memory_token_count,
                    )
                )
                freq_pieces.append(
                    memory.build_freqs(
                        endpoint=endpoint,
                        device=video_pre["tokens"].device,
                        token_count=memory_token_count,
                    )
                )
                t_mod_pieces.append(
                    memory.build_t_mod(
                        video_pre["t_mod"][:, endpoint_token : endpoint_token + 1],
                        token_count=memory_token_count,
                    )
                )
                context_mask_pieces.append(
                    video_pre["context_mask"][:, endpoint_token : endpoint_token + 1].expand(
                        -1, memory_token_count, -1
                    )
                )
                continue
            frame_start = int(segment.frame_indices[0])
            frame_stop = int(segment.frame_indices[-1]) + 1
            token_start = frame_start * tokens_per_frame
            token_stop = frame_stop * tokens_per_frame
            token_pieces.append(video_pre["tokens"][:, token_start:token_stop])
            freq_pieces.append(video_pre["freqs"][token_start:token_stop])
            t_mod_pieces.append(video_pre["t_mod"][:, token_start:token_stop])
            context_mask_pieces.append(
                video_pre["context_mask"][:, token_start:token_stop]
            )

        noisy_start = clean_frame_count * tokens_per_frame
        noisy_stop = (clean_frame_count + noisy_frame_count) * tokens_per_frame
        token_pieces.append(video_pre["tokens"][:, noisy_start:noisy_stop])
        freq_pieces.append(video_pre["freqs"][noisy_start:noisy_stop])
        t_mod_pieces.append(video_pre["t_mod"][:, noisy_start:noisy_stop])
        context_mask_pieces.append(video_pre["context_mask"][:, noisy_start:noisy_stop])
        packed = dict(video_pre)
        packed["tokens"] = torch.cat(token_pieces, dim=1)
        packed["freqs"] = torch.cat(freq_pieces, dim=0)
        packed["t_mod"] = torch.cat(t_mod_pieces, dim=1)
        packed["context_mask"] = torch.cat(context_mask_pieces, dim=1)
        packed["layerwise_memory_layout"] = layout
        return packed, layout

    def _run_layerwise_memory_training_transformer(
        self,
        *,
        video_pre: dict[str, Any],
        action_pre: dict[str, Any],
        clean_frame_count: int,
        noisy_frame_count: int,
        memory_groups: tuple[tuple[int, ...], ...] | None = None,
        memory_token_counts: tuple[int, ...] | None = None,
    ):
        packed, layout = self._pack_layerwise_memory_training_state(
            video_pre,
            clean_frame_count=clean_frame_count,
            noisy_frame_count=noisy_frame_count,
            action_seq_len=int(action_pre["tokens"].shape[1]),
            memory_groups=memory_groups,
            memory_token_counts=memory_token_counts,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": packed["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=layout.attention_mask,
            freqs_all={
                "video": packed["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": packed["context"],
                    "mask": packed["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": packed["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        return tokens_out, layout

    def _native_block_freqs(
        self,
        *,
        endpoints: Sequence[int],
        grid_h: int,
        grid_w: int,
        device: torch.device,
    ) -> torch.Tensor:
        temporal, vertical, horizontal = self.video_expert.freqs
        pieces = []
        for endpoint in endpoints:
            if endpoint < 0 or endpoint >= temporal.shape[0]:
                raise ValueError(
                    f"Native block endpoint {endpoint} exceeds RoPE capacity {temporal.shape[0]}"
                )
            pieces.append(
                torch.cat(
                    [
                        temporal[endpoint].view(1, 1, -1).expand(
                            grid_h, grid_w, -1
                        ),
                        vertical[:grid_h].view(grid_h, 1, -1).expand(
                            grid_h, grid_w, -1
                        ),
                        horizontal[:grid_w].view(1, grid_w, -1).expand(
                            grid_h, grid_w, -1
                        ),
                    ],
                    dim=-1,
                ).reshape(grid_h * grid_w, 1, -1)
            )
        return torch.cat(pieces, dim=0).to(device=device)

    def _commit_native_cache_state(
        self,
        *,
        previous_state: Optional[NativeCacheState],
        current_pre: dict[str, Any],
        current_cache: Sequence[dict[str, torch.Tensor]],
        endpoint: int,
    ) -> NativeCacheState:
        """Commit current raw block, then rewrite a completed level-0 suffix."""
        compressor = self.native_cache_compressor
        if compressor is None:
            raise ValueError("Native cache compressor is disabled")
        current_block = NativeBlock(
            tokens=current_pre["tokens"], endpoint=int(endpoint), span=1, level=0
        )
        prefix_blocks = () if previous_state is None else previous_state.blocks
        appended_blocks = (*prefix_blocks, current_block)
        appended_state = NativeCacheState(
            blocks=appended_blocks,
            kv_cache=tuple({"k": layer["k"], "v": layer["v"]} for layer in current_cache),
        )
        if len(appended_blocks) < compressor.group_size:
            return appended_state
        suffix = appended_blocks[-compressor.group_size :]
        if any(block.level != 0 or block.span != 1 for block in suffix):
            return appended_state

        _, grid_h, grid_w = current_pre["meta"]["grid_size"]
        block_freqs = self._native_block_freqs(
            endpoints=[block.endpoint for block in suffix],
            grid_h=int(grid_h),
            grid_w=int(grid_w),
            device=current_pre["tokens"].device,
        )
        summary = compressor.consolidate_blocks(
            suffix, block_freqs=block_freqs
        )
        retained_prefix = appended_blocks[: -compressor.group_size]
        tokens_per_block = int(current_pre["tokens"].shape[1])
        prefix_token_count = len(retained_prefix) * tokens_per_block
        history_cache = None
        if prefix_token_count:
            history_cache = [
                {
                    "k": layer["k"][:, :prefix_token_count],
                    "v": layer["v"][:, :prefix_token_count],
                }
                for layer in current_cache
            ]
        compacted_cache = self.mot.prefill_video_cache(
            video_tokens=summary.tokens,
            video_freqs=current_pre["freqs"],
            video_t_mod=current_pre["t_mod"],
            video_context_payload={
                "context": current_pre["context"],
                "mask": current_pre["context_mask"],
            },
            video_attention_mask=torch.ones(
                tokens_per_block,
                prefix_token_count + tokens_per_block,
                dtype=torch.bool,
                device=summary.tokens.device,
            ),
            history_kv_cache=history_cache,
        )
        return NativeCacheState(
            blocks=(*retained_prefix, summary),
            kv_cache=tuple(
                {"k": layer["k"], "v": layer["v"]}
                for layer in compacted_cache
            ),
        )

    def _carry_recursive_layerwise_memories(
        self,
        *,
        units: list[CacheUnit],
        cache: tuple[dict[str, torch.Tensor], ...],
        current_pre: dict[str, Any],
    ) -> tuple[
        list[CacheUnit],
        tuple[dict[str, torch.Tensor], ...],
        tuple[int, ...],
    ]:
        """Collapse completed equal-level suffixes using native layerwise slots."""
        memory = self.layerwise_block_memory
        if memory is None or not memory.recursive:
            return units, cache, ()
        carry_levels: list[int] = []
        while True:
            memory_indices = [
                index for index, unit in enumerate(units) if unit.kind == "memory"
            ]
            if len(memory_indices) < memory.group_size:
                break
            child_indices = tuple(memory_indices[-memory.group_size :])
            children = tuple(units[index] for index in child_indices)
            if len({int(child.level) for child in children}) != 1:
                break
            if len({int(child.span) for child in children}) != 1:
                raise ValueError("Recursive carry suffix has unequal child spans")
            output_level = int(children[0].level) + 1
            if output_level > memory.max_levels:
                raise ValueError(
                    f"Recursive memory level {output_level} exceeds "
                    f"max_levels={memory.max_levels}"
                )
            parent = CacheUnit(
                kind="memory",
                endpoint=int(children[-1].endpoint),
                span=sum(int(child.span) for child in children),
                token_count=memory.memory_tokens,
                level=output_level,
            )
            MemoryCarry(children=children, parent=parent)

            offsets: list[tuple[int, int]] = []
            token_cursor = 0
            for unit in units:
                offsets.append((token_cursor, token_cursor + int(unit.token_count)))
                token_cursor += int(unit.token_count)
            visibility = torch.zeros(
                memory.memory_tokens,
                token_cursor + memory.memory_tokens,
                dtype=torch.bool,
                device=current_pre["tokens"].device,
            )
            child_index_set = set(child_indices)
            for index, (unit, token_range) in enumerate(zip(units, offsets)):
                expose = (
                    unit.kind == "anchor"
                    or index in child_index_set
                    or (
                        unit.kind == "memory"
                        and int(unit.endpoint) < int(children[0].start)
                    )
                )
                if expose:
                    visibility[:, token_range[0] : token_range[1]] = True
            visibility[:, token_cursor:] = True

            slots = memory.initial_tokens(
                batch_size=int(current_pre["tokens"].shape[0]),
                device=current_pre["tokens"].device,
                dtype=current_pre["tokens"].dtype,
                level=output_level,
            )
            parent_cache = self.mot.prefill_video_cache(
                video_tokens=slots,
                video_freqs=memory.build_freqs(
                    endpoint=int(parent.endpoint),
                    device=slots.device,
                ),
                video_t_mod=memory.build_t_mod(current_pre["t_mod"][:, :1]),
                video_context_payload={
                    "context": current_pre["context"],
                    "mask": current_pre["context_mask"][:, :1].expand(
                        -1, memory.memory_tokens, -1
                    ),
                },
                video_attention_mask=visibility,
                history_kv_cache=list(cache),
            )

            extended_units = [*units, parent]
            extended_offsets = [*offsets, (token_cursor, token_cursor + memory.memory_tokens)]
            keep_indices = [
                index
                for index in range(len(extended_units))
                if index not in child_index_set
            ]
            kept_ranges = [extended_offsets[index] for index in keep_indices]
            units = [extended_units[index] for index in keep_indices]
            cache = tuple(
                {
                    "k": torch.cat(
                        [layer["k"][:, start:stop] for start, stop in kept_ranges],
                        dim=1,
                    ),
                    "v": torch.cat(
                        [layer["v"][:, start:stop] for start, stop in kept_ranges],
                        dim=1,
                    ),
                }
                for layer in parent_cache
            )
            carry_levels.append(output_level)
        return units, cache, tuple(carry_levels)

    def _commit_layerwise_memory_state(
        self,
        *,
        previous_state: Optional[LayerwiseMemoryState],
        current_pre: dict[str, Any],
        current_cache: Sequence[dict[str, torch.Tensor]],
        endpoint: int,
    ) -> LayerwiseMemoryState:
        """Append one raw frame, form completed block gist, then evict old raw K/V."""
        memory = self.layerwise_block_memory
        if memory is None:
            raise ValueError("Layerwise block memory is disabled")
        endpoint = int(endpoint)
        current_token_count = int(current_pre["tokens"].shape[1])
        previous_units = () if previous_state is None else previous_state.units
        units: list[CacheUnit] = [
            *previous_units,
            CacheUnit(
                kind="anchor" if endpoint < memory.anchor_frames else "recent",
                endpoint=endpoint,
                span=1,
                token_count=current_token_count,
            ),
        ]
        cache = tuple({"k": layer["k"], "v": layer["v"]} for layer in current_cache)
        partition = partition_layerwise_history(
            endpoint + 1,
            anchor_frames=memory.anchor_frames,
            recent_frames=memory.recent_frames,
            group_size=memory.group_size,
        )
        completed_group = next(
            (group for group in partition.memory_groups if group[-1] == endpoint),
            None,
        )
        has_memory = any(
            unit.kind == "memory" and unit.endpoint == endpoint for unit in units
        )
        if completed_group is not None and not has_memory:
            group_set = set(completed_group)
            raw_group_endpoints = {
                unit.endpoint
                for unit in units
                if unit.kind != "memory" and unit.endpoint in group_set
            }
            if raw_group_endpoints != group_set:
                raise ValueError(
                    "Completed memory group is missing raw source K/V: "
                    f"expected {sorted(group_set)}, got {sorted(raw_group_endpoints)}"
                )
            total_history_tokens = sum(unit.token_count for unit in units)
            visibility = torch.zeros(
                memory.memory_tokens,
                total_history_tokens + memory.memory_tokens,
                dtype=torch.bool,
                device=current_pre["tokens"].device,
            )
            token_cursor = 0
            for unit in units:
                token_stop = token_cursor + unit.token_count
                if (
                    unit.kind in {"anchor", "memory"}
                    or unit.endpoint in group_set
                ):
                    visibility[:, token_cursor:token_stop] = True
                token_cursor = token_stop
            visibility[:, total_history_tokens:] = True
            slots = memory.initial_tokens(
                batch_size=int(current_pre["tokens"].shape[0]),
                device=current_pre["tokens"].device,
                dtype=current_pre["tokens"].dtype,
            )
            slot_cache = self.mot.prefill_video_cache(
                video_tokens=slots,
                video_freqs=memory.build_freqs(
                    endpoint=endpoint, device=slots.device
                ),
                video_t_mod=memory.build_t_mod(current_pre["t_mod"][:, :1]),
                video_context_payload={
                    "context": current_pre["context"],
                    "mask": current_pre["context_mask"][:, :1].expand(
                        -1, memory.memory_tokens, -1
                    ),
                },
                video_attention_mask=visibility,
                history_kv_cache=list(cache),
            )
            cache = tuple(
                {"k": layer["k"], "v": layer["v"]} for layer in slot_cache
            )
            units.append(
                CacheUnit(
                    kind="memory",
                    endpoint=endpoint,
                    span=memory.group_size,
                    token_count=memory.memory_tokens,
                    level=1,
                )
            )

        units, cache, carry_levels = self._carry_recursive_layerwise_memories(
            units=units,
            cache=cache,
            current_pre=current_pre,
        )
        offsets = []
        token_cursor = 0
        for unit in units:
            offsets.append((token_cursor, token_cursor + unit.token_count))
            token_cursor += unit.token_count
        anchor_set = set(partition.anchors)
        recent_set = set(partition.recent)
        kept_units: list[CacheUnit] = []
        kept_ranges: list[tuple[int, int]] = []
        for unit, token_range in zip(units, offsets):
            if unit.kind == "memory":
                kept_units.append(unit)
                kept_ranges.append(token_range)
            elif unit.endpoint in anchor_set:
                kept_units.append(
                    CacheUnit(
                        kind="anchor",
                        endpoint=unit.endpoint,
                        span=1,
                        token_count=unit.token_count,
                    )
                )
                kept_ranges.append(token_range)
            elif unit.endpoint in recent_set:
                kept_units.append(
                    CacheUnit(
                        kind="recent",
                        endpoint=unit.endpoint,
                        span=1,
                        token_count=unit.token_count,
                    )
                )
                kept_ranges.append(token_range)

        compacted = tuple(
            {
                "k": torch.cat(
                    [layer["k"][:, start:stop] for start, stop in kept_ranges],
                    dim=1,
                ),
                "v": torch.cat(
                    [layer["v"][:, start:stop] for start, stop in kept_ranges],
                    dim=1,
                ),
            }
            for layer in cache
        )
        return LayerwiseMemoryState(
            units=tuple(kept_units),
            kv_cache=compacted,
            carry_levels=carry_levels,
        )

    def _build_layerwise_online_video_mask(
        self,
        *,
        previous_state: Optional[LayerwiseMemoryState],
        endpoint: int,
        current_token_count: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Expose anchors, memories, and the preceding three raw recent frames."""
        endpoint = int(endpoint)
        current_token_count = int(current_token_count)
        if current_token_count < 1:
            raise ValueError("current_token_count must be positive")
        history_tokens = 0 if previous_state is None else previous_state.retained_tokens
        mask = torch.zeros(
            current_token_count,
            history_tokens + current_token_count,
            dtype=torch.bool,
            device=device,
        )
        if previous_state is not None:
            memory = self.layerwise_block_memory
            if memory is None:
                raise ValueError("Layerwise block memory is disabled")
            recent_start = max(
                memory.anchor_frames,
                endpoint - memory.recent_frames + 1,
            )
            cursor = 0
            for unit in previous_state.units:
                stop = cursor + unit.token_count
                if (
                    unit.kind in {"anchor", "memory"}
                    or recent_start <= unit.endpoint < endpoint
                ):
                    mask[:, cursor:stop] = True
                cursor = stop
        mask[:, history_tokens:] = True
        return mask

    def _commit_dynamic_layerwise_memory_state(
        self,
        *,
        previous_state: Optional[DynamicLayerwiseMemoryState],
        current_pre: dict[str, Any],
        endpoint: int,
        close_range: Optional[tuple[int, int]],
        close_token_count: Optional[int] = None,
    ) -> DynamicLayerwiseMemoryState:
        """Close [b,t) to K slots while retaining anchors and the open raw tail."""
        memory = self.layerwise_block_memory
        if memory is None:
            raise ValueError("Dynamic layerwise memory is disabled")
        endpoint = int(endpoint)
        if previous_state is None:
            if endpoint != 0:
                raise ValueError("dynamic cache must start at endpoint zero")
            if close_range is not None:
                raise ValueError("cannot close a dynamic segment before frame zero")
            if close_token_count is not None:
                raise ValueError("cannot allocate memory tokens without a close range")
            history_units: list[CacheUnit] = []
            history_cache = None
            # Anchors are retained verbatim.  While they are still arriving,
            # advance the logical open tail one frame at a time; after the
            # final anchor, the first compressible segment starts at
            # ``anchor_frames``.
            open_start = 0
        else:
            if previous_state.represented_frames != endpoint:
                raise ValueError(
                    "dynamic cache must precede the arrival exactly; "
                    f"state has {previous_state.represented_frames} frames, arrival={endpoint}"
                )
            history_units = list(previous_state.units)
            history_cache = list(previous_state.kv_cache)
            open_start = int(previous_state.open_start)
            if endpoint <= int(memory.anchor_frames):
                open_start = min(int(memory.anchor_frames), endpoint)

        if close_range is not None:
            start, stop = (int(close_range[0]), int(close_range[1]))
            if (
                stop > endpoint
                or endpoint - stop > int(memory.recent_frames)
                or start != open_start
                or not 2 <= stop - start <= 8
            ):
                raise ValueError(
                    "dynamic close range must be a 2..8-frame prefix of the open "
                    "tail with at most recent_frames causal confirmation delay; "
                    f"open_start={open_start}, arrival={endpoint}, close={close_range}"
                )
            source_indices = [
                index
                for index, unit in enumerate(history_units)
                if unit.kind != "memory" and start <= unit.endpoint < stop
            ]
            if [history_units[index].endpoint for index in source_indices] != list(range(start, stop)):
                raise ValueError("dynamic close range is missing raw source K/V")
            if history_cache is None:
                raise ValueError("dynamic close range requires history K/V")
            if close_token_count is None:
                raise ValueError("event-conditioned close requires an explicit token count")
            memory_token_count = int(close_token_count)
            if (
                memory_token_count <= 0
                or memory_token_count > int(memory.memory_tokens)
                or memory_token_count % 8 != 0
            ):
                raise ValueError(
                    "dynamic close token count must be a positive multiple of 8 "
                    f"not exceeding {memory.memory_tokens}, got {memory_token_count}"
                )
            slots = memory.initial_tokens(
                batch_size=int(current_pre["tokens"].shape[0]),
                device=current_pre["tokens"].device,
                dtype=current_pre["tokens"].dtype,
                token_count=memory_token_count,
            )
            offsets: list[tuple[int, int]] = []
            cursor = 0
            for unit in history_units:
                offsets.append((cursor, cursor + unit.token_count))
                cursor += unit.token_count
            group_set = set(range(start, stop))
            memory_visibility = torch.zeros(
                memory_token_count,
                cursor + memory_token_count,
                dtype=torch.bool,
                device=slots.device,
            )
            for unit, (left, right) in zip(history_units, offsets):
                if (
                    unit.kind == "memory"
                    or (
                        getattr(memory, "reader_use_anchor", True)
                        and int(unit.endpoint) < int(memory.anchor_frames)
                    )
                    or int(unit.endpoint) in group_set
                ):
                    memory_visibility[:, left:right] = True
            memory_visibility[:, cursor:] = True
            slot_cache = self.mot.prefill_video_cache(
                video_tokens=slots,
                video_freqs=memory.build_freqs(
                    endpoint=stop - 1,
                    device=slots.device,
                    token_count=memory_token_count,
                ),
                video_t_mod=memory.build_t_mod(
                    current_pre["t_mod"][:, :1],
                    token_count=memory_token_count,
                ),
                video_context_payload={
                    "context": current_pre["context"],
                    "mask": current_pre["context_mask"][:, :1].expand(
                        -1, memory_token_count, -1
                    ),
                },
                video_attention_mask=memory_visibility,
                history_kv_cache=history_cache,
            )
            keep_indices = [
                index
                for index, unit in enumerate(history_units)
                if (
                    unit.kind == "memory"
                    or int(unit.endpoint) < int(memory.anchor_frames)
                    or int(unit.endpoint) >= stop
                )
            ]
            kept_ranges = [offsets[index] for index in keep_indices]
            kept_ranges.append((cursor, cursor + memory_token_count))
            history_cache = [
                {
                    "k": torch.cat([layer["k"][:, a:b] for a, b in kept_ranges], dim=1),
                    "v": torch.cat([layer["v"][:, a:b] for a, b in kept_ranges], dim=1),
                }
                for layer in slot_cache
            ]
            retained_units: list[CacheUnit] = []
            for index in keep_indices:
                unit = history_units[index]
                if unit.kind == "memory":
                    retained_units.append(unit)
                else:
                    retained_units.append(
                        CacheUnit(
                            kind=(
                                "anchor"
                                if int(unit.endpoint) < int(memory.anchor_frames)
                                else "recent"
                            ),
                            endpoint=unit.endpoint,
                            span=1,
                            token_count=unit.token_count,
                        )
                    )
            history_units = retained_units
            history_units.append(
                CacheUnit(
                    kind="memory",
                    endpoint=stop - 1,
                    span=stop - start,
                    token_count=memory_token_count,
                    level=1,
                )
            )
            # A delayed physical confirmation may close a prefix [start,stop)
            # while [stop,endpoint) is already buffered. Keep that suffix raw
            # as the beginning of the next dynamic segment.
            open_start = stop
        elif close_token_count is not None:
            raise ValueError("close_token_count requires close_range")

        history_tokens = sum(unit.token_count for unit in history_units)
        current_tokens = int(current_pre["tokens"].shape[1])
        current_visibility = torch.zeros(
            current_tokens,
            history_tokens + current_tokens,
            dtype=torch.bool,
            device=current_pre["tokens"].device,
        )
        cursor = 0
        for unit in history_units:
            stop = cursor + unit.token_count
            if (
                (
                    getattr(memory, "reader_use_anchor", True)
                    and int(unit.endpoint) < int(memory.anchor_frames)
                )
                or (unit.kind != "memory" and int(unit.endpoint) >= open_start)
            ):
                current_visibility[:, cursor:stop] = True
            cursor = stop
        current_visibility[:, history_tokens:] = True
        current_cache = self.mot.prefill_video_cache(
            video_tokens=current_pre["tokens"],
            video_freqs=current_pre["freqs"],
            video_t_mod=current_pre["t_mod"],
            video_context_payload={
                "context": current_pre["context"],
                "mask": current_pre["context_mask"],
            },
            video_attention_mask=current_visibility,
            history_kv_cache=history_cache,
        )
        history_units.append(
            CacheUnit(
                kind="anchor" if endpoint < int(memory.anchor_frames) else "recent",
                endpoint=endpoint,
                span=1,
                token_count=current_tokens,
                level=0,
            )
        )
        return DynamicLayerwiseMemoryState(
            units=tuple(history_units),
            kv_cache=tuple({"k": layer["k"], "v": layer["v"]} for layer in current_cache),
            open_start=open_start,
        )

    def _build_dynamic_layerwise_action_mask(
        self,
        *,
        state: DynamicLayerwiseMemoryState,
        action_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Expose anchors, memories, the complete open raw tail, and actions."""
        memory = self.layerwise_block_memory
        if memory is None:
            raise ValueError("Dynamic layerwise memory is disabled")
        action_tokens = int(action_tokens)
        if action_tokens < 1:
            raise ValueError("action_tokens must be positive")
        video_tokens = int(state.retained_tokens)
        mask = torch.zeros(
            action_tokens,
            video_tokens + action_tokens,
            dtype=torch.bool,
            device=device,
        )
        cursor = 0
        for unit in state.units:
            stop = cursor + int(unit.token_count)
            if (
                (
                    getattr(memory, "reader_use_memory", True)
                    and unit.kind == "memory"
                )
                or (
                    getattr(memory, "reader_use_anchor", True)
                    and int(unit.endpoint) < int(memory.anchor_frames)
                )
                or (
                    getattr(memory, "reader_use_recent", True)
                    and unit.kind != "memory"
                    and int(unit.endpoint) >= int(state.open_start)
                )
            ):
                mask[:, cursor:stop] = True
            cursor = stop
        mask[:, video_tokens:] = True
        return mask

    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        include_initial_video_step: bool,
    ) -> torch.Tensor:
        video_loss_token = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return video_loss_token.mean(dim=1)

        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}.")
        if image_is_pad.shape[1] < 1:
            raise ValueError("`image_is_pad` must contain at least one frame.")
        if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(
                "Cannot align `image_is_pad` with video latent steps: "
                f"num_frames={image_is_pad.shape[1]}, temporal_downsample_factor={temporal_factor}."
            )

        tail_is_pad = image_is_pad[:, 1:]
        latent_tail_is_pad = tail_is_pad.view(image_is_pad.shape[0], -1, temporal_factor).all(dim=2)
        if include_initial_video_step:
            video_is_pad = torch.cat([image_is_pad[:, :1], latent_tail_is_pad], dim=1)
        else:
            video_is_pad = latent_tail_is_pad

        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(
                "Video-loss mask shape mismatch: "
                f"mask steps={video_is_pad.shape[1]}, loss steps={video_loss_token.shape[1]}."
            )

        valid = (~video_is_pad).to(device=video_loss_token.device, dtype=video_loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (video_loss_token * valid).sum(dim=1) / valid_sum

    def _training_loss_full_kv(self, sample, tiled: bool = False):
        """Train FastWAM on full clean history plus current video/action targets."""
        dynamic_memory_groups, dynamic_memory_token_counts = (
            _decode_dynamic_memory_plan(sample)
        )
        inputs = self.build_inputs(sample, tiled=tiled)
        history = inputs["history_latents"]
        if history.shape[0] != 1:
            raise ValueError(
                "Full-KV training currently requires per-GPU batch size 1 "
                "because each episode position has a different history length"
            )

        input_latents = inputs["input_latents"]
        if input_latents.shape[2] < 2:
            raise ValueError("Full-KV training needs current plus future video latents")
        batch_size = input_latents.shape[0]
        action_context = inputs["context"]
        action_context_mask = inputs["context_mask"]
        video_context = inputs["video_context"]
        video_context_mask = inputs["video_context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]

        # [B,H,C,1,h,w] -> [B,C,H,h,w]
        history = history.squeeze(3).permute(0, 2, 1, 3, 4).contiguous()
        # `history` is inclusive of the current decision point. Its latest
        # latent is Wan's causal temporal summary of the preceding action
        # chunk, so the independently encoded current single frame is not
        # duplicated in the clean KV prefix.
        clean_latents = history
        future_latents = input_latents[:, :, 1:]
        clean_frame_count = int(clean_latents.shape[2])
        noisy_frame_count = int(future_latents.shape[2])

        # MemoryWAM conditioning augmentation: every clean conditioning frame
        # is linearly mixed with independent Gaussian noise at U[0,1].
        condition_ratio = torch.rand(
            (batch_size, 1, clean_frame_count, 1, 1),
            device=clean_latents.device,
            dtype=torch.float32,
        ).to(dtype=clean_latents.dtype)
        conditioned_clean = (
            clean_latents * (1.0 - condition_ratio)
            + torch.randn_like(clean_latents) * condition_ratio
        )

        noise_video = torch.randn_like(future_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=future_latents.dtype,
        )
        noisy_future = self.train_video_scheduler.add_noise(
            future_latents, noise_video, timestep_video
        )
        target_video = self.train_video_scheduler.training_target(
            future_latents, noise_video, timestep_video
        )
        video_model_input = torch.cat([conditioned_clean, noisy_future], dim=2)
        video_frame_timesteps = torch.cat(
            [
                torch.zeros(
                    (batch_size, clean_frame_count),
                    device=self.device,
                    dtype=timestep_video.dtype,
                ),
                timestep_video[:, None].expand(-1, noisy_frame_count),
            ],
            dim=1,
        )

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(
            action, noise_action, timestep_action
        )
        target_action = self.train_action_scheduler.training_target(
            action, noise_action, timestep_action
        )

        video_pre = self.video_expert.pre_dit(
            x=video_model_input,
            timestep=video_frame_timesteps,
            context=video_context,
            context_mask=video_context_mask,
            action=None,
            fuse_vae_embedding_in_latents=True,
        )
        tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])
        future_token_start = clean_frame_count * tokens_per_frame
        future_pre = dict(video_pre)
        future_pre["t"] = video_pre["t"][:, future_token_start:]
        _, grid_h, grid_w = video_pre["meta"]["grid_size"]
        future_pre["meta"] = dict(video_pre["meta"])
        future_pre["meta"]["grid_size"] = (
            noisy_frame_count,
            int(grid_h),
            int(grid_w),
        )
        layerwise_layout = None
        retained_clean_count = clean_frame_count
        if self.layerwise_block_memory is None:
            video_pre, retained_clean_count = (
                self._consolidate_native_training_state(
                    video_pre,
                    clean_frame_count=clean_frame_count,
                    noisy_frame_count=noisy_frame_count,
                    memory_groups=dynamic_memory_groups,
                    memory_token_counts=dynamic_memory_token_counts,
                )
            )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=action_context,
            context_mask=action_context_mask,
        )
        action_pre["freqs"] = self._build_video_aligned_action_freqs(
            action_seq_len=int(action_pre["tokens"].shape[1]),
            temporal_base=float(clean_frame_count - 1),
            grid_h=int(grid_h),
            grid_w=int(grid_w),
            device=action_pre["tokens"].device,
        )
        if self.layerwise_block_memory is not None:
            tokens_out, layerwise_layout = (
                self._run_layerwise_memory_training_transformer(
                    video_pre=video_pre,
                    action_pre=action_pre,
                    clean_frame_count=clean_frame_count,
                    noisy_frame_count=noisy_frame_count,
                    memory_groups=dynamic_memory_groups,
                    memory_token_counts=dynamic_memory_token_counts,
                )
            )
            if dynamic_memory_token_counts is not None:
                actual_memory_token_counts = tuple(
                    int(segment.stop) - int(segment.start)
                    for segment in layerwise_layout.segments
                    if segment.kind == "memory"
                )
                if actual_memory_token_counts != dynamic_memory_token_counts:
                    raise RuntimeError(
                        "layerwise training ignored the manifest memory-token plan: "
                        f"actual={actual_memory_token_counts}, "
                        f"expected={dynamic_memory_token_counts}"
                    )
            retained_clean_count = len(layerwise_layout.retained_ranges)
            retained_clean_tokens = layerwise_layout.retained_clean_tokens
            noisy_token_start, noisy_token_stop = layerwise_layout.noisy_range
            predicted_future_tokens = tokens_out["video"][
                :, noisy_token_start:noisy_token_stop
            ]
        else:
            attention_mask = self._build_full_history_training_mask(
                clean_video_frames=retained_clean_count,
                noisy_video_frames=noisy_frame_count,
                video_tokens_per_frame=tokens_per_frame,
                action_seq_len=action_pre["tokens"].shape[1],
                device=video_pre["tokens"].device,
            )
            tokens_out = self.mot(
                embeds_all={
                    "video": video_pre["tokens"],
                    "action": action_pre["tokens"],
                },
                attention_mask=attention_mask,
                freqs_all={
                    "video": video_pre["freqs"],
                    "action": action_pre["freqs"],
                },
                context_all={
                    "video": {
                        "context": video_pre["context"],
                        "mask": video_pre["context_mask"],
                    },
                    "action": {
                        "context": action_pre["context"],
                        "mask": action_pre["context_mask"],
                    },
                },
                t_mod_all={
                    "video": video_pre["t_mod"],
                    "action": action_pre["t_mod"],
                },
            )
            retained_clean_tokens = retained_clean_count * tokens_per_frame
            predicted_future_tokens = tokens_out["video"][:, retained_clean_tokens:]

        pred_video = self.video_expert.post_dit(
            predicted_future_tokens, future_pre
        )
        video_loss_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=False,
        )
        video_weight = self.train_video_scheduler.training_weight(
            timestep_video
        ).to(video_loss_per_sample)
        loss_video = (video_loss_per_sample * video_weight).mean()

        pred_action = self.action_expert.post_dit(
            tokens_out["action"], action_pre
        )
        action_loss_token = F.mse_loss(
            pred_action.float(), target_action.float(), reduction="none"
        ).mean(dim=2)
        if action_is_pad is not None:
            valid = (~action_is_pad).to(
                device=action_loss_token.device,
                dtype=action_loss_token.dtype,
            )
            action_loss_per_sample = (
                action_loss_token * valid
            ).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)
        action_weight = self.train_action_scheduler.training_weight(
            timestep_action
        ).to(action_loss_per_sample)
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_total = (
            self.loss_lambda_video * loss_video
            + self.loss_lambda_action * loss_action
        )
        return loss_total, {
            "loss_video": self.loss_lambda_video
            * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action
            * float(loss_action.detach().item()),
            "history_frames": float(clean_frame_count),
            "native_retained_history_units": float(retained_clean_count),
            "full_kv_video_tokens": float(
                retained_clean_tokens
            ),
        }

    def forward(self, sample, tiled: bool = False):
        """FSDP/DDP entrypoint for the joint video-action training loss."""
        return self.training_loss(sample, tiled=tiled)

    def training_loss(self, sample, tiled: bool = False):
        if "history_latents" in sample:
            return self._training_loss_full_kv(sample, tiled=tiled)
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)

        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0:1] = inputs["first_frame_latents"]

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )

        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        video_tokens = video_pre["tokens"]
        action_tokens = action_pre["tokens"]

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_tokens.shape[1],
            action_seq_len=action_tokens.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_tokens.device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_tokens,
                "action": action_tokens,
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)

        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        include_initial_video_step = inputs["first_frame_latents"] is None
        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2) # [B, T]
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }
        return loss_total, loss_dict

    @torch.no_grad()
    def _predict_joint_noise(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        gt_action: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=gt_action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )

        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_video, pred_action

    @torch.no_grad()
    def _predict_action_noise(
        self,
        first_frame_latents: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
    ) -> torch.Tensor:
        timestep_video = torch.zeros_like(timestep_action, dtype=first_frame_latents.dtype, device=self.device)
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_action

    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        action_freqs: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_pre["freqs"] = action_freqs
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None, # NOTE: this is gt action for conditioning videos, not for action expert
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        test_action_with_infer_action: bool = True,
    ) -> dict[str, Any]:
        self.eval()
        if test_action_with_infer_action:
            if seed is None:
                raise ValueError("`test_action_with_infer_action=True` requires non-null `seed`.")
            action_only_out = self.infer_action(
                prompt=prompt,
                input_image=input_image.clone(),
                action_horizon=action_horizon,
                context=context.clone() if context is not None else None,
                context_mask=context_mask.clone() if context_mask is not None else None,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                proprio=proprio.clone() if proprio is not None else None,
            )["action"]
        
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )
        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3 or action.shape[0] != 1 or action.shape[1] != action_horizon:
                # NOTE: This enforces action condition to have the same shape as action horizon to predict, which may be unnecessary
                raise ValueError(
                    f"`action` must have shape [1, T, a_dim] or [T, a_dim], got {tuple(action.shape)} with action_horizon={action_horizon}"
                )
            action = action.to(device=self.device, dtype=self.torch_dtype)
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(
            input_image=input_image,
            tiled=tiled,
        )
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        video_context = context
        video_context_mask = context_mask
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
            infer_timesteps_video,
            infer_deltas_video,
            infer_timesteps_action,
            infer_deltas_action,
        ):
            timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_video_posi, pred_action_posi = self._predict_joint_noise(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                gt_action=action,
            )
            pred_video = pred_video_posi
            pred_action = pred_action_posi

            latents_video = self.infer_video_scheduler.step(pred_video, step_delta_video, latents_video)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        action_out = latents_action[0].detach().to(device="cpu", dtype=torch.float32)
        if test_action_with_infer_action:
            if not torch.allclose(action_out, action_only_out, atol=1e-2, rtol=1e-2):
                max_abs_diff = (action_out - action_only_out).abs().max().item()
                logger.warning(
                    f"Action from infer_joint and infer_action differ with max abs diff {max_abs_diff:.6f}. "
                )

        return {
            "video": self._decode_latents(latents_video, tiled=tiled),
            "action": action_out,
        }

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        full_kv_cache: Optional[list[dict[str, torch.Tensor]]] = None,
        full_kv_frame_index: int = 0,
        native_cache_state: Optional[
            Union[NativeCacheState, LayerwiseMemoryState, DynamicLayerwiseMemoryState]
        ] = None,
        dynamic_surprise_online: bool = False,
        dynamic_close_range: Optional[tuple[int, int]] = None,
        dynamic_close_token_count: Optional[int] = None,
    ) -> dict[str, Any]:
        self.eval()
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`infer_action` requires `video_attention_mask_mode='first_frame_causal'`."
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if (
            input_image.ndim not in (4, 5)
            or input_image.shape[0] != 1
            or input_image.shape[1] != 3
        ):
            raise ValueError(
                "`input_image` must have shape [1,3,H,W], [3,H,W], or "
                f"[1,3,T,H,W], got {tuple(input_image.shape)}"
            )
        height, width = input_image.shape[-2:]
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        observation_latents = self._encode_input_image_latents_tensor(
            input_image=input_image,
            tiled=tiled,
        )
        input_frame_count = 1 if input_image.ndim == 4 else int(input_image.shape[2])
        expected_latent_frames = 1 + (input_frame_count - 1) // 4
        if observation_latents.shape[2] != expected_latent_frames:
            raise ValueError(
                "Unexpected temporal VAE output for memory observation: "
                f"input={tuple(input_image.shape)}, latent={tuple(observation_latents.shape)}"
            )
        first_frame_latents = observation_latents[:, :, -1:]
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        video_context = context
        video_context_mask = context_mask
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=video_context,
            context_mask=video_context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
            temporal_position_offset=int(full_kv_frame_index),
        )
        use_dynamic_memory = bool(dynamic_surprise_online) or isinstance(
            native_cache_state, DynamicLayerwiseMemoryState
        )
        if dynamic_close_range is not None and not use_dynamic_memory:
            raise ValueError("dynamic_close_range requires dynamic_surprise_online=True")
        if dynamic_close_token_count is not None and not use_dynamic_memory:
            raise ValueError("dynamic_close_token_count requires dynamic memory")
        if native_cache_state is not None:
            if (
                self.native_cache_compressor is None
                and self.layerwise_block_memory is None
            ):
                raise ValueError(
                    "native_cache_state was provided but native cache is disabled"
                )
            if full_kv_cache is not None:
                raise ValueError(
                    "native_cache_state and full_kv_cache are mutually exclusive"
                )
            latest_endpoint = (
                native_cache_state.blocks[-1].endpoint
                if isinstance(native_cache_state, NativeCacheState)
                else native_cache_state.represented_frames - 1
            )
            if latest_endpoint >= int(full_kv_frame_index):
                raise ValueError(
                    "native cache endpoint must precede the current frame index"
                )
            if not use_dynamic_memory:
                full_kv_cache = list(native_cache_state.kv_cache)
        current_video_seq_len = int(video_pre["tokens"].shape[1])
        history_video_seq_len = 0
        if full_kv_cache is not None:
            if len(full_kv_cache) != self.mot.num_layers:
                raise ValueError(
                    f"`full_kv_cache` must contain {self.mot.num_layers} layers, "
                    f"got {len(full_kv_cache)}"
                )
            history_video_seq_len = int(full_kv_cache[0]["k"].shape[1])
        committed_native_state = None
        if use_dynamic_memory:
            if self.layerwise_block_memory is None:
                raise ValueError("dynamic surprise inference requires layerwise memory")
            if native_cache_state is not None and not isinstance(
                native_cache_state, DynamicLayerwiseMemoryState
            ):
                raise ValueError(
                    "dynamic surprise inference requires DynamicLayerwiseMemoryState"
                )
            committed_native_state = self._commit_dynamic_layerwise_memory_state(
                previous_state=native_cache_state,
                current_pre=video_pre,
                endpoint=int(full_kv_frame_index),
                close_range=dynamic_close_range,
                close_token_count=dynamic_close_token_count,
            )
            video_kv_cache = list(committed_native_state.kv_cache)
            video_seq_len = committed_native_state.retained_tokens
        elif self.layerwise_block_memory is not None:
            if native_cache_state is not None and not isinstance(
                native_cache_state, LayerwiseMemoryState
            ):
                raise ValueError("Layerwise memory requires LayerwiseMemoryState")
            video_prefill_mask = self._build_layerwise_online_video_mask(
                previous_state=native_cache_state,
                endpoint=int(full_kv_frame_index),
                current_token_count=current_video_seq_len,
                device=video_pre["tokens"].device,
            )
        else:
            video_prefill_mask = torch.ones(
                (
                    current_video_seq_len,
                    history_video_seq_len + current_video_seq_len,
                ),
                dtype=torch.bool,
                device=video_pre["tokens"].device,
            )
        if not use_dynamic_memory:
            video_kv_cache = self.mot.prefill_video_cache(
                video_tokens=video_pre["tokens"],
                video_freqs=video_pre["freqs"],
                video_t_mod=video_pre["t_mod"],
                video_context_payload={
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                video_attention_mask=video_prefill_mask,
                history_kv_cache=full_kv_cache,
            )
            if self.layerwise_block_memory is not None:
                committed_native_state = self._commit_layerwise_memory_state(
                    previous_state=native_cache_state,
                    current_pre=video_pre,
                    current_cache=video_kv_cache,
                    endpoint=int(full_kv_frame_index),
                )
                video_kv_cache = list(committed_native_state.kv_cache)
                video_seq_len = committed_native_state.retained_tokens
            else:
                video_seq_len = history_video_seq_len + current_video_seq_len
        action_seq_len = int(latents_action.shape[1])
        _, grid_h, grid_w = video_pre["meta"]["grid_size"]
        action_freqs = self._build_video_aligned_action_freqs(
            action_seq_len=action_seq_len,
            temporal_base=float(full_kv_frame_index),
            grid_h=int(grid_h),
            grid_w=int(grid_w),
            device=video_pre["tokens"].device,
        )
        # Rectangular action-query mask avoids allocating an O(history^2)
        # square mask. Action reads all observed video K/V and all action K/V.
        if use_dynamic_memory:
            if not isinstance(committed_native_state, DynamicLayerwiseMemoryState):
                raise ValueError("dynamic memory did not produce a committed state")
            attention_mask = self._build_dynamic_layerwise_action_mask(
                state=committed_native_state,
                action_tokens=action_seq_len,
                device=video_pre["tokens"].device,
            )
        else:
            attention_mask = torch.ones(
                (action_seq_len, video_seq_len + action_seq_len),
                dtype=torch.bool,
                device=video_pre["tokens"].device,
            )

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_action_posi = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                action_freqs=action_freqs,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            pred_action = pred_action_posi

            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        if self.native_cache_compressor is not None:
            if native_cache_state is not None and not isinstance(
                native_cache_state, NativeCacheState
            ):
                raise ValueError("Legacy native cache requires NativeCacheState")
            committed_native_state = self._commit_native_cache_state(
                previous_state=native_cache_state,
                current_pre=video_pre,
                current_cache=video_kv_cache,
                endpoint=int(full_kv_frame_index),
            )
            video_kv_cache = list(committed_native_state.kv_cache)

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
            "full_kv_cache": video_kv_cache,
            "native_cache_state": committed_native_state,
            # Reuse the exact causal-VAE latent already encoded for this
            # decision.  A lightweight online event head must not trigger a
            # second full-history VAE pass.
            "current_observation_latent": first_frame_latents.detach(),
            "full_kv_frame_index": int(full_kv_frame_index),
            "full_kv_video_tokens": int(video_seq_len),
            # Deployment keeps these exact source-decision conditionings for
            # the next transition's causal surprise score.
            "action_context": context.detach(),
            "action_context_mask": context_mask.detach(),
            "video_context": video_context.detach(),
            "video_context_mask": video_context_mask.detach(),
        }

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ):
        return self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
        )

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
            "inference_contract": {
                "video_scheduler_shift": float(self.infer_video_scheduler.shift),
                "action_scheduler_shift": float(self.infer_action_scheduler.shift),
                "video_num_train_timesteps": int(
                    self.infer_video_scheduler.num_train_timesteps
                ),
                "action_num_train_timesteps": int(
                    self.infer_action_scheduler.num_train_timesteps
                ),
                "video_attention_mask_mode": str(
                    self.video_expert.video_attention_mask_mode
                ),
                "full_kv_cache": True,
            },
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if self.native_cache_compressor is not None:
            payload["native_cache_compressor"] = (
                self.native_cache_compressor.state_dict()
            )
        if self.layerwise_block_memory is not None:
            payload["inference_contract"].update(
                {
                    "reader_use_anchor": bool(
                        self.layerwise_block_memory.reader_use_anchor
                    ),
                    "reader_use_memory": bool(
                        self.layerwise_block_memory.reader_use_memory
                    ),
                    "reader_use_recent": bool(
                        self.layerwise_block_memory.reader_use_recent
                    ),
                }
            )
            payload["layerwise_block_memory"] = (
                self.layerwise_block_memory.state_dict()
            )
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")
        if "mot" in payload:
            self.mot.load_state_dict(payload["mot"], strict=True)
        elif "dit" in payload:
            logger.warning("Loading legacy `dit` checkpoint into video expert only.")
            self.video_expert.load_state_dict(payload["dit"], strict=False)
        else:
            raise ValueError(f"Checkpoint missing both `mot` and `dit` keys: {path}")
        if "native_cache_compressor" in payload:
            if self.native_cache_compressor is None:
                raise ValueError(
                    "Checkpoint contains native-cache weights but native_cache is disabled"
                )
            self.native_cache_compressor.load_state_dict(
                payload["native_cache_compressor"], strict=True
            )
        elif self.native_cache_compressor is not None:
            self.native_cache_compressor.initialize_from_video_block(
                self.video_expert.blocks[0]
            )
            logger.info(
                "Initialized native-cache attention from loaded VideoDiT block 0"
            )
        if "layerwise_block_memory" in payload:
            if self.layerwise_block_memory is None:
                raise ValueError(
                    "Checkpoint contains layerwise block-memory weights but "
                    "layerwise memory is disabled"
                )
            self.layerwise_block_memory.load_state_dict(
                payload["layerwise_block_memory"], strict=True
            )
        elif self.layerwise_block_memory is not None:
            raise ValueError(
                "Layerwise memory checkpoint is missing `layerwise_block_memory` weights"
            )
        contract = payload.get("inference_contract")
        if contract is not None:
            expected = {
                "video_scheduler_shift": float(self.infer_video_scheduler.shift),
                "action_scheduler_shift": float(self.infer_action_scheduler.shift),
                "video_num_train_timesteps": int(
                    self.infer_video_scheduler.num_train_timesteps
                ),
                "action_num_train_timesteps": int(
                    self.infer_action_scheduler.num_train_timesteps
                ),
                "video_attention_mask_mode": str(
                    self.video_expert.video_attention_mask_mode
                ),
                "full_kv_cache": True,
            }
            if self.layerwise_block_memory is not None:
                expected.update(
                    {
                        "reader_use_anchor": bool(
                            self.layerwise_block_memory.reader_use_anchor
                        ),
                        "reader_use_memory": bool(
                            self.layerwise_block_memory.reader_use_memory
                        ),
                        "reader_use_recent": bool(
                            self.layerwise_block_memory.reader_use_recent
                        ),
                    }
                )
            mismatches = {
                key: (expected[key], contract.get(key))
                for key in expected
                if contract.get(key) != expected[key]
            }
            if mismatches:
                raise ValueError(
                    "Checkpoint inference contract does not match the runtime "
                    f"model config: {mismatches}"
                )
        if self.proprio_encoder is not None:
            if "proprio_encoder" in payload:
                self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
            else:
                logger.warning("Checkpoint has no `proprio_encoder` weights; keeping current `proprio_encoder` params.")
        elif "proprio_encoder" in payload:
            logger.warning("Checkpoint contains `proprio_encoder` weights but current model has `proprio_dim=None`; ignoring.")

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
