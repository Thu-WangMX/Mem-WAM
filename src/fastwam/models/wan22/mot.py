from __future__ import annotations

import os
from typing import Dict, Optional

import torch
import torch.nn as nn

from .wan_video_dit import flash_attention, modulate, rope_apply
from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)


class MoT(nn.Module):
    def __init__(
        self,
        mixtures: Dict[str, nn.Module],
        mot_checkpoint_mixed_attn: bool = True,
    ):
        super().__init__()
        if not mixtures:
            raise ValueError("`mixtures` cannot be empty.")
        if "video" not in mixtures or "action" not in mixtures:
            raise ValueError("`mixtures` must include both 'video' and 'action' experts.")

        self.mixtures = nn.ModuleDict(mixtures)
        self.expert_order = list(self.mixtures.keys())
        self.mot_checkpoint_mixed_attn = mot_checkpoint_mixed_attn
        if mot_checkpoint_mixed_attn:
            logger.info("Using gradient checkpointing for mixture attention. This will save memory but use more computation.")

        first_expert = self.mixtures[self.expert_order[0]]
        self.num_layers = len(first_expert.blocks)
        self.num_heads = first_expert.num_heads
        self.attn_head_dim = first_expert.attn_head_dim

        for name in self.expert_order[1:]:
            expert = self.mixtures[name]
            if len(expert.blocks) != self.num_layers:
                raise ValueError(
                    f"All experts must have same number of layers; got {self.num_layers} and {len(expert.blocks)}"
                )
            if expert.num_heads != self.num_heads:
                raise ValueError(
                    f"All experts must have same num_heads; got {self.num_heads} and {expert.num_heads}"
                )
            if expert.attn_head_dim != self.attn_head_dim:
                raise ValueError(
                    "All experts must have same attn_head_dim; "
                    f"got {self.attn_head_dim} and {expert.attn_head_dim}"
                )
        
        logger.info(f"Initialized MoT with experts: {self.expert_order}, num_layers={self.num_layers}")
        for name in self.expert_order:
            expert = self.mixtures[name]
            logger.info(f"  Expert '{name}': num_params={sum(p.numel() for p in expert.parameters()) / 1e9:.2f} B")

    @staticmethod
    def _split_modulation(block, t_mod: torch.Tensor):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1

        base_mod = block.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (base_mod + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            # means t_mod has separate modulation for each token, otherwise same modulation for all tokens in the block
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

    def _mixed_attention(
        self,
        q_cat: torch.Tensor,
        k_cat: torch.Tensor,
        v_cat: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        attn_mask = attention_mask.to(device=q_cat.device)

        def _forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=attn_mask)

        if self.mot_checkpoint_mixed_attn and self.training:
            return torch.utils.checkpoint.checkpoint(
                _forward,
                q_cat,
                k_cat,
                v_cat,
                use_reentrant=False,
            )
        return _forward(q_cat, k_cat, v_cat)

    @staticmethod
    def _apply_ffn(block, mlp_input: torch.Tensor) -> torch.Tensor:
        """Apply the token-wise FFN with an optional memory-safe sequence split.

        Transformer FFNs do not mix tokens, so splitting along the sequence
        dimension is mathematically equivalent to one full-sequence call while
        avoiding a large temporary expansion for long Full-KV histories.
        """
        chunk_size = int(os.environ.get("FASTWAM_FFN_CHUNK_SIZE", "0"))
        if chunk_size <= 0 or mlp_input.shape[1] <= chunk_size:
            return block.ffn(mlp_input)
        output = torch.empty_like(mlp_input)
        for start in range(0, mlp_input.shape[1], chunk_size):
            stop = min(start + chunk_size, mlp_input.shape[1])
            output[:, start:stop] = block.ffn(mlp_input[:, start:stop])
        return output

    @staticmethod
    def _project_qkv_with_rope(block, attn_input: torch.Tensor, freqs: torch.Tensor):
        """Project Q/K/V with optional sequence chunks for long histories."""
        chunk_size = int(os.environ.get("FASTWAM_QKV_CHUNK_SIZE", "0"))
        if chunk_size <= 0 or attn_input.shape[1] <= chunk_size:
            q = block.self_attn.norm_q(block.self_attn.q(attn_input))
            k = block.self_attn.norm_k(block.self_attn.k(attn_input))
            v = block.self_attn.v(attn_input)
            return (
                rope_apply(q, freqs, block.num_heads),
                rope_apply(k, freqs, block.num_heads),
                v,
            )

        q_chunks, k_chunks, v_chunks = [], [], []
        for start in range(0, attn_input.shape[1], chunk_size):
            stop = min(start + chunk_size, attn_input.shape[1])
            x_chunk = attn_input[:, start:stop]
            freq_chunk = freqs[start:stop]
            q_chunk = block.self_attn.norm_q(block.self_attn.q(x_chunk))
            k_chunk = block.self_attn.norm_k(block.self_attn.k(x_chunk))
            v_chunk = block.self_attn.v(x_chunk)
            q_chunks.append(rope_apply(q_chunk, freq_chunk, block.num_heads))
            k_chunks.append(rope_apply(k_chunk, freq_chunk, block.num_heads))
            v_chunks.append(v_chunk)
        return (
            torch.cat(q_chunks, dim=1),
            torch.cat(k_chunks, dim=1),
            torch.cat(v_chunks, dim=1),
        )

    @staticmethod
    def _build_qkv_from_tokens(
        block,
        x: torch.Tensor,
        shift_msa: torch.Tensor,
        scale_msa: torch.Tensor,
        freqs: torch.Tensor,
    ):
        """Normalize/modulate/project tokens without a full-sequence temporary."""
        chunk_size = int(os.environ.get("FASTWAM_QKV_CHUNK_SIZE", "0"))
        if chunk_size <= 0 or x.shape[1] <= chunk_size:
            attn_input = modulate(block.norm1(x), shift_msa, scale_msa)
            return MoT._project_qkv_with_rope(block, attn_input, freqs)

        q_chunks, k_chunks, v_chunks = [], [], []
        sequence_len = x.shape[1]
        for start in range(0, sequence_len, chunk_size):
            stop = min(start + chunk_size, sequence_len)
            shift_chunk = (
                shift_msa[:, start:stop]
                if shift_msa.dim() >= 3 and shift_msa.shape[1] == sequence_len
                else shift_msa
            )
            scale_chunk = (
                scale_msa[:, start:stop]
                if scale_msa.dim() >= 3 and scale_msa.shape[1] == sequence_len
                else scale_msa
            )
            attn_chunk = modulate(
                block.norm1(x[:, start:stop]), shift_chunk, scale_chunk
            )
            q_chunk, k_chunk, v_chunk = MoT._project_qkv_with_rope(
                block, attn_chunk, freqs[start:stop]
            )
            q_chunks.append(q_chunk)
            k_chunks.append(k_chunk)
            v_chunks.append(v_chunk)
        return (
            torch.cat(q_chunks, dim=1),
            torch.cat(k_chunks, dim=1),
            torch.cat(v_chunks, dim=1),
        )

    @staticmethod
    def _apply_cross_attention(block, x: torch.Tensor, context: torch.Tensor, context_mask):
        """Apply cross-attention with optional query-token chunks."""
        chunk_size = int(os.environ.get("FASTWAM_CROSS_ATTN_CHUNK_SIZE", "0"))
        normed_x = block.norm3(x)
        if chunk_size <= 0 or normed_x.shape[1] <= chunk_size:
            return block.cross_attn(normed_x, context, ctx_mask=context_mask)

        outputs = []
        for start in range(0, normed_x.shape[1], chunk_size):
            stop = min(start + chunk_size, normed_x.shape[1])
            mask_chunk = context_mask
            if context_mask is not None:
                if context_mask.dim() == 4:
                    mask_chunk = context_mask[:, :, start:stop, :]
                elif context_mask.dim() == 3:
                    mask_chunk = context_mask[:, start:stop, :]
            outputs.append(
                block.cross_attn(
                    normed_x[:, start:stop],
                    context,
                    ctx_mask=mask_chunk,
                )
            )
        return torch.cat(outputs, dim=1)

    @staticmethod
    def _apply_expert_post_block(
        block,
        residual_x: torch.Tensor,
        mixed_attn_out: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        context_payload: Optional[dict],
    ) -> torch.Tensor:
        x = block.gate(residual_x, gate_msa, block.self_attn.o(mixed_attn_out))

        if context_payload is not None:
            context = context_payload.get("context")
            if context is not None:
                context_mask = context_payload.get("mask")
                if context_mask is not None and context_mask.dim() == 3:
                    context_mask = context_mask.unsqueeze(1)
                x = x + MoT._apply_cross_attention(
                    block, x, context, context_mask
                )

        mlp_input = modulate(block.norm2(x), shift_mlp, scale_mlp)
        x = block.gate(x, gate_mlp, MoT._apply_ffn(block, mlp_input))
        return x

    def _build_expert_attention_io(
        self,
        expert,
        block,
        x: torch.Tensor,
        freqs: torch.Tensor,
        t_mod: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        bool,
    ]:
        """Build per-expert attention tensors and post-block states.

        Args:
            expert: Expert module that owns this `block`; only used to read
                `use_gradient_checkpointing`.
            block: Transformer block for current layer (`expert.blocks[layer_idx]`).
            x: Current expert tokens, shape [B, S, D].
            freqs: RoPE frequencies aligned with token sequence, shape [S, 1, rope_dim].
            t_mod: Time modulation tensor for this expert/layer.

        Returns:
            q: Query after q-proj, RMSNorm, and RoPE, shape [B, S, H*Dh].
            k: Key after k-proj, RMSNorm, and RoPE, shape [B, S, H*Dh].
            v: Value after v-proj, shape [B, S, H*Dh].
            residual_x: Original input `x` for residual path in post block.
            gate_msa: Gating tensor for self-attention residual branch.
            shift_mlp: Shift tensor for MLP modulation.
            scale_mlp: Scale tensor for MLP modulation.
            gate_mlp: Gating tensor for MLP residual branch.
            use_gradient_checkpointing: Whether this expert enables checkpointing.
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._split_modulation(block, t_mod)
        q, k, v = self._build_qkv_from_tokens(
            block, x, shift_msa, scale_msa, freqs
        )

        use_gradient_checkpointing = bool(getattr(expert, "use_gradient_checkpointing", False))
        return (
            q,
            k,
            v,
            x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            use_gradient_checkpointing,
        )

    def _apply_post_with_optional_checkpoint(
        self,
        block,
        residual_x: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        use_gradient_checkpointing: bool,
        mixed_slice: torch.Tensor,
        context_payload: Optional[dict],
    ) -> torch.Tensor:
        """Apply post-attention computations, with optional checkpointing.

        Args:
            block: Transformer block for current layer.
            residual_x: Residual input tokens before attention update, shape [B, S, D].
            gate_msa: Gating tensor used after mixed self-attention.
            shift_mlp: Shift tensor for MLP input modulation.
            scale_mlp: Scale tensor for MLP input modulation.
            gate_mlp: Gating tensor used after MLP.
            use_gradient_checkpointing: If True and training, checkpoint this post block.
            mixed_slice: Mixed-attention output for this expert, shape [B, S, H*Dh].
            context_payload: Optional dict for cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, S, L] or [B, 1, S, L]

        Returns:
            Updated expert tokens after self-attn residual, optional cross-attn, and MLP.
        """
        def _post_fn(
            _mixed_slice: torch.Tensor,
            _x: torch.Tensor,
            _gate_msa: torch.Tensor,
            _shift_mlp: torch.Tensor,
            _scale_mlp: torch.Tensor,
            _gate_mlp: torch.Tensor,
            _block=block,
            _context_payload=context_payload,
        ) -> torch.Tensor:
            return self._apply_expert_post_block(
                block=_block,
                residual_x=_x,
                mixed_attn_out=_mixed_slice,
                gate_msa=_gate_msa,
                shift_mlp=_shift_mlp,
                scale_mlp=_scale_mlp,
                gate_mlp=_gate_mlp,
                context_payload=_context_payload,
            )

        if use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                _post_fn,
                mixed_slice,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_reentrant=False,
            )
        return _post_fn(
            mixed_slice,
            residual_x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        )

    def prefill_video_cache(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        video_attention_mask: torch.Tensor,
        history_kv_cache: Optional[list[dict[str, torch.Tensor]]] = None,
        return_current_tokens: bool = False,
    ):
        """Prefill current video tokens and append their K/V to full history.

        Args:
            video_tokens: Video tokens before layer 0, shape [B, Sv, D].
            video_freqs: Video RoPE frequencies, shape [Sv, 1, rope_dim].
            video_t_mod: Video time modulation tensor.
            video_context_payload: Optional dict for video cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, Sv, L] or [B, 1, Sv, L]
            video_attention_mask: Current-video query mask, shape
                [S_current, S_history + S_current]. With no history the
                original square [S_current, S_current] contract is unchanged.
            history_kv_cache: Optional immutable layer-wise cache from earlier
                observations. Historical K/V is never recomputed or evicted.

        Returns:
            Layer-wise cache list with length `num_layers`.
            Each entry contains the concatenated historical and current K/V.
        """
        if "video" not in self.mixtures:
            raise ValueError("MoT requires `video` expert for `prefill_video_cache`.")
        if video_attention_mask.ndim != 2:
            raise ValueError(
                "`video_attention_mask` must be 2D [S_current,S_total], "
                f"got shape {tuple(video_attention_mask.shape)}"
            )
        if video_attention_mask.shape[0] != video_tokens.shape[1]:
            raise ValueError(
                "`video_attention_mask` query length mismatch: "
                f"mask={video_attention_mask.shape[0]} vs tokens={video_tokens.shape[1]}"
            )
        if history_kv_cache is not None and len(history_kv_cache) != self.num_layers:
            raise ValueError(
                f"`history_kv_cache` must contain {self.num_layers} layers, "
                f"got {len(history_kv_cache)}"
            )

        expert = self.mixtures["video"]
        x = video_tokens
        kv_cache: list[dict[str, torch.Tensor]] = []
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            # Build video Q/K/V from current layer input tokens.
            (
                q,
                k,
                v,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=video_freqs,
                t_mod=video_t_mod,
            )
            if history_kv_cache is None:
                k_total = k
                v_total = v
            else:
                historical = history_kv_cache[layer_idx]
                if "k" not in historical or "v" not in historical:
                    raise ValueError(
                        f"`history_kv_cache[{layer_idx}]` must contain `k` and `v`"
                    )
                k_history = historical["k"]
                v_history = historical["v"]
                if k_history.shape[0] != k.shape[0] or v_history.shape[0] != v.shape[0]:
                    raise ValueError(
                        "Historical/current video-cache batch sizes must match"
                    )
                k_total = torch.cat([k_history, k], dim=1)
                v_total = torch.cat([v_history, v], dim=1)
            if video_attention_mask.shape[1] != k_total.shape[1]:
                raise ValueError(
                    "`video_attention_mask` key length mismatch: "
                    f"mask={video_attention_mask.shape[1]} vs cache={k_total.shape[1]}"
                )

            # Current queries read every retained historical key plus the
            # current observation. Only current tokens advance through blocks.
            mixed = self._mixed_attention(
                q_cat=q,
                k_cat=k_total,
                v_cat=v_total,
                attention_mask=video_attention_mask,
            )
            # Update video tokens for the next layer and persist current layer K/V.
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=video_context_payload,
            )
            kv_cache.append({"k": k_total, "v": v_total})
        if return_current_tokens:
            return kv_cache, x
        return kv_cache

    def forward_action_with_video_cache(
        self,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context_payload: Optional[dict],
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        """Run action branch with cached video K/V instead of recomputing video tokens.

        Args:
            action_tokens: Action tokens before layer 0, shape [B, Sa, D].
            action_freqs: Action RoPE frequencies, shape [Sa, 1, rope_dim].
            action_t_mod: Action time modulation tensor.
            action_context_payload: Optional dict for action cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, Sa, L] or [B, 1, Sa, L]
            video_kv_cache: Layer-wise cached video K/V from `prefill_video_cache`.
            attention_mask: Joint [video+action] mask, shape [Sv+Sa, Sv+Sa].
            video_seq_len: Video token count `Sv` in the joint sequence prefix.

        Returns:
            Updated action tokens after all layers, shape [B, Sa, D].
        """
        if "action" not in self.mixtures:
            raise ValueError("MoT requires `action` expert for `forward_action_with_video_cache`.")
        if len(video_kv_cache) != self.num_layers:
            raise ValueError(
                f"`video_kv_cache` must contain {self.num_layers} layers, got {len(video_kv_cache)}."
            )
        if attention_mask.ndim != 2:
            raise ValueError(
                "`attention_mask` must be square [S_total,S_total] or "
                f"rectangular action rows [S_action,S_total], got {tuple(attention_mask.shape)}"
            )

        action_seq_len = int(action_tokens.shape[1])
        total_seq_len = int(video_seq_len) + action_seq_len
        if attention_mask.shape == (action_seq_len, total_seq_len):
            action_attention_mask = attention_mask
        elif attention_mask.shape == (total_seq_len, total_seq_len):
            action_attention_mask = attention_mask[
                video_seq_len:total_seq_len, :total_seq_len
            ]
        else:
            raise ValueError(
                "`attention_mask` shape mismatch: "
                f"got {tuple(attention_mask.shape)}, expected "
                f"{(action_seq_len, total_seq_len)} or "
                f"{(total_seq_len, total_seq_len)}"
            )

        expert = self.mixtures["action"]
        x = action_tokens
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            # Action query/key/value are still step-dependent and must be recomputed each step.
            (
                q_action,
                k_action,
                v_action,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=action_freqs,
                t_mod=action_t_mod,
            )
            layer_cache = video_kv_cache[layer_idx]
            if "k" not in layer_cache or "v" not in layer_cache:
                raise ValueError(
                    f"`video_kv_cache[{layer_idx}]` must contain `k` and `v`."
                )

            k_video = layer_cache["k"]
            v_video = layer_cache["v"]
            if k_video.shape[1] != video_seq_len or v_video.shape[1] != video_seq_len:
                raise ValueError(
                    f"`video_kv_cache[{layer_idx}]` seq len mismatch, expected {video_seq_len}."
                )

            # Mixed attention: action queries attend to cached video K/V plus current action K/V.
            k_cat = torch.cat([k_video, k_action], dim=1)
            v_cat = torch.cat([v_video, v_action], dim=1)
            mixed = self._mixed_attention(
                q_cat=q_action,
                k_cat=k_cat,
                v_cat=v_cat,
                attention_mask=action_attention_mask,
            )
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=action_context_payload,
            )
        return x

    def forward(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
        return_video_kv_cache: bool = False,
    ):
        missing = [k for k in self.expert_order if k not in embeds_all]
        if missing:
            raise ValueError(f"Missing expert tokens for {missing}")
        missing = [k for k in self.expert_order if k not in freqs_all]
        if missing:
            raise ValueError(f"Missing expert freqs for {missing}")
        missing = [k for k in self.expert_order if k not in t_mod_all]
        if missing:
            raise ValueError(f"Missing expert t_mod for {missing}")

        if attention_mask.ndim != 2:
            raise ValueError(f"`attention_mask` must be 2D [S, S], got shape {tuple(attention_mask.shape)}")
        if attention_mask.shape[0] != attention_mask.shape[1]:
            raise ValueError(f"`attention_mask` must be square, got shape {tuple(attention_mask.shape)}")

        tokens_all = {k: v for k, v in embeds_all.items()}

        video_kv_cache: list[dict[str, torch.Tensor]] = []
        for layer_idx in range(self.num_layers):
            q_chunks = []
            k_chunks = []
            v_chunks = []
            cached = {}
            seq_lens = []

            for name in self.expert_order:
                expert = self.mixtures[name]
                block = expert.blocks[layer_idx]
                x = tokens_all[name]
                freqs = freqs_all[name]
                t_mod = t_mod_all[name]

                (
                    q,
                    k,
                    v,
                    residual_x,
                    gate_msa,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                    use_gradient_checkpointing,
                ) = self._build_expert_attention_io(
                    expert=expert,
                    block=block,
                    x=x,
                    freqs=freqs,
                    t_mod=t_mod,
                )

                q_chunks.append(q)
                k_chunks.append(k)
                v_chunks.append(v)
                if return_video_kv_cache and name == "video":
                    video_kv_cache.append({"k": k, "v": v})
                seq_lens.append(x.shape[1])
                cached[name] = {
                    "block": block,
                    "residual_x": residual_x,
                    "gate_msa": gate_msa,
                    "shift_mlp": shift_mlp,
                    "scale_mlp": scale_mlp,
                    "gate_mlp": gate_mlp,
                    "use_gradient_checkpointing": use_gradient_checkpointing,
                }

            # 3. concat all tokens for mixed attention
            q_cat = torch.cat(q_chunks, dim=1)
            k_cat = torch.cat(k_chunks, dim=1)
            v_cat = torch.cat(v_chunks, dim=1)

            total_seq = q_cat.shape[1]
            if attention_mask.shape[0] != total_seq:
                raise ValueError(
                    "Attention mask seq length mismatch: "
                    f"mask={attention_mask.shape[0]} vs tokens={total_seq}"
                )

            mixed = self._mixed_attention(q_cat=q_cat, k_cat=k_cat, v_cat=v_cat, attention_mask=attention_mask)

            start = 0
            for name, seq_len in zip(self.expert_order, seq_lens):
                # 4. split mixed attention output and apply post-attention blocks for each expert
                end = start + seq_len
                mixed_slice = mixed[:, start:end, :]
                cached_expert = cached[name]
                block = cached_expert["block"]
                context_payload = context_all.get(name)

                updated_tokens = self._apply_post_with_optional_checkpoint(
                    block=block,
                    residual_x=cached_expert["residual_x"],
                    gate_msa=cached_expert["gate_msa"],
                    shift_mlp=cached_expert["shift_mlp"],
                    scale_mlp=cached_expert["scale_mlp"],
                    gate_mlp=cached_expert["gate_mlp"],
                    use_gradient_checkpointing=cached_expert["use_gradient_checkpointing"],
                    mixed_slice=mixed_slice,
                    context_payload=context_payload,
                )

                tokens_all[name] = updated_tokens
                start = end

        if return_video_kv_cache:
            return tokens_all, video_kv_cache
        return tokens_all
