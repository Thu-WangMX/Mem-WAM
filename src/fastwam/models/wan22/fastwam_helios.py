"""Helios latent memory on the shared RMBench FastWAM training/deployment contract.

No cross-decision Transformer KV is retained. The policy supplies its existing
continuous RGB prefix; action denoising reuses one compressed-history prefill.
"""
from copy import deepcopy
import logging
import torch
import torch.nn.functional as F
from .fastwam import FastWAM
from .helios_video import MTMVideoDiT
from fastwam.memory.helios_history import episode_anchors, fixed_history


def load_backbone_initialization(model, path):
    """Strictly initialize the base FastWAM before adding Helios layers.

    This is weight initialization, not resume: optimizer and step are ignored.
    mmap avoids eagerly buffering the entire shared-filesystem checkpoint.
    """
    payload = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    model.mot.load_state_dict(payload['mot'], strict=True)
    if model.proprio_encoder is not None:
        model.proprio_encoder.load_state_dict(payload['proprio_encoder'], strict=True)
    elif payload.get('proprio_encoder'):
        raise ValueError('Initialization checkpoint has proprio weights but model does not')
    logging.getLogger(__name__).info(
        'BACKBONE_INIT_OK path=%s source_step=%s strict=True optimizer_and_step_reset=True',
        path, payload.get('step'))


class FastWAMHelios(FastWAM):
    @classmethod
    def from_wan22_pretrained(cls, *, helios_config=None, **kwargs):
        config = dict(helios_config or {})
        if (kwargs.get('native_cache') or {}).get('enabled', False):
            raise ValueError('Helios must not be stacked with native_cache memory')
        if config.get('amplify_history', False):
            raise ValueError('Shared MoT port currently uses the original default amplify_history=false')
        model = super().from_wan22_pretrained(**kwargs)
        backbone_init = config.get('backbone_init_checkpoint')
        if backbone_init:
            load_backbone_initialization(model, backbone_init)
        sizes = tuple(config.get('history_sizes', (16, 2, 1)))
        anchor_frames = int(config.get('anchor_frames', 2))
        if len(sizes) != 3 or sizes[-1] != 1:
            raise ValueError('Expected long/mid/current history sizes, current=1')
        if anchor_frames not in (0, 2):
            raise ValueError('Helios supports either zero (ablation) or two episode anchors')
        dit_cfg = deepcopy(dict(kwargs['video_dit_config']))
        dit_cfg['fuse_vae_embedding_in_latents'] = False
        dit_cfg.update(multi_term_memory=True, mtm_history_sizes=sizes,
                       mtm_anchor_size=anchor_frames,
                       mtm_pred_size=1, mtm_zero_history_timestep=True,
                       mtm_patch_kernel_long=tuple(config.get('patch_long', (4, 8, 8))),
                       mtm_patch_kernel_mid=tuple(config.get('patch_mid', (2, 4, 4))))
        # Share the already-loaded backbone parameters rather than allocating a
        # second 5B model. Only the two new patch convolutions need storage.
        with torch.device('meta'):
            video = MTMVideoDiT(**dit_cfg)
        result = video.load_state_dict(model.video_expert.state_dict(), strict=False, assign=True)
        expected = {'patch_mid.weight', 'patch_mid.bias', 'patch_long.weight', 'patch_long.bias'}
        if set(result.missing_keys) != expected or result.unexpected_keys:
            raise ValueError(f'Unexpected backbone transfer: {result}')
        for patch in [video.patch_mid, video.patch_long]:
            patch.to_empty(device=model.device)
            patch.to(dtype=model.torch_dtype)
        video._init_mtm_patches_from_patch_embedding()
        # RoPE tables are plain tensors, not registered state-dict buffers.
        video.freqs = model.video_expert.freqs
        model.mot.mixtures['video'] = video
        model.helios_config = {'history_sizes': list(sizes),
                               'anchor_frames': anchor_frames,
                               'patch_long': list(video.mtm_patch_kernel_long),
                               'patch_mid': list(video.mtm_patch_kernel_mid),
                               'condition_augmentation': bool(config.get('condition_augmentation', True))}
        return model

    def _segments(self, history):
        # Input [B,H,C,1,h,w], output [B,C,H,h,w].
        sizes = self.helios_config['history_sizes']
        if history.ndim != 6 or history.shape[1] != sum(sizes):
            raise ValueError(f'Expected fixed {sum(sizes)}-frame Helios history')
        return history.squeeze(3).permute(0, 2, 1, 3, 4).split(sizes, dim=2)

    def _video_pre(self, anchors, history, future, timestep, context, context_mask):
        long, mid, current = self._segments(history)
        return self.video_expert.pre_dit(
            x=future, timestep=timestep, context=context, context_mask=context_mask,
            fuse_vae_embedding_in_latents=False,
            history_anchors=anchors.squeeze(3).permute(0, 2, 1, 3, 4),
            history_long=long, history_mid=mid, history_short=current)

    def _joint_mask(self, pre, action_length):
        m = pre['meta']['mtm']
        h, p = m['L_hist'], m['L_pred']
        mask = torch.zeros((h+p+action_length, h+p+action_length),
                           device=pre['tokens'].device, dtype=torch.bool)
        mask[:h, :h] = True
        mask[h:h+p, :h+p] = True
        mask[h+p:, :h] = True
        mask[h+p:, h+p:] = True
        return mask

    def _action_freqs(self, action_length, pre):
        _, h, w = pre['meta']['grid_size']
        return self._build_video_aligned_action_freqs(
            action_seq_len=action_length,
            temporal_base=(self.helios_config['anchor_frames'] +
                           sum(self.helios_config['history_sizes']) - 1),
            grid_h=h, grid_w=w, device=pre['tokens'].device)

    def training_loss(self, sample, tiled=False):
        inputs = self.build_inputs(sample, tiled=tiled)
        history = inputs['history_latents']
        anchors = sample.get('anchor_latents')
        anchor_valid = sample.get('anchor_valid')
        history_valid = sample.get('history_valid')
        expected_anchors = self.helios_config['anchor_frames']
        if (not isinstance(anchors, torch.Tensor) or anchors.ndim != 6
                or anchors.shape[0] != history.shape[0]
                or anchors.shape[1] != expected_anchors
                or tuple(anchors.shape[2:]) != tuple(history.shape[2:])):
            raise ValueError(
                'anchor_latents must be [B,A,C,1,h,w] and match history, '
                f'expected A={expected_anchors}')
        if (not isinstance(anchor_valid, torch.Tensor)
                or tuple(anchor_valid.shape) != tuple(anchors.shape[:2])):
            raise ValueError('anchor_valid must be [B,A] and match anchor_latents')
        if (not isinstance(history_valid, torch.Tensor)
                or tuple(history_valid.shape) != tuple(history.shape[:2])):
            raise ValueError('history_valid must be [B,H] and match history_latents')
        anchors = anchors.to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True)
        future = inputs['input_latents'][:, :, 1:]
        action = inputs['action']
        if action.shape[1] != 16 or future.shape[2] != 1:
            raise ValueError('Shared RMBench comparison requires 16 actions / 1 future latent')
        if self.helios_config['condition_augmentation']:
            ratio = torch.rand((history.shape[0], history.shape[1], 1, 1, 1, 1),
                               device=history.device).to(history.dtype)
            augmented = history*(1-ratio) + torch.randn_like(history)*ratio
            valid = history_valid.to(history.device).view(*history.shape[:2], 1, 1, 1, 1)
            history = torch.where(valid, augmented, history)
            anchor_ratio = torch.rand(
                (anchors.shape[0], anchors.shape[1], 1, 1, 1, 1),
                device=anchors.device, dtype=anchors.dtype)
            augmented_anchors = anchors * (1-anchor_ratio) + torch.randn_like(anchors) * anchor_ratio
            anchor_valid_mask = anchor_valid.to(anchors.device).view(
                *anchors.shape[:2], 1, 1, 1, 1)
            anchors = torch.where(anchor_valid_mask, augmented_anchors, anchors)
        tv = self.train_video_scheduler.sample_training_t(action.shape[0], self.device, future.dtype)
        ta = self.train_action_scheduler.sample_training_t(action.shape[0], self.device, action.dtype)
        nv, na = torch.randn_like(future), torch.randn_like(action)
        vp = self._video_pre(anchors, history,
                             self.train_video_scheduler.add_noise(future, nv, tv), tv,
                             inputs['video_context'], inputs['video_context_mask'])
        ap = self.action_expert.pre_dit(
            self.train_action_scheduler.add_noise(action, na, ta), ta,
            inputs['context'], inputs['context_mask'])
        ap['freqs'] = self._action_freqs(action.shape[1], vp)
        out = self.mot(
            embeds_all={'video': vp['tokens'], 'action': ap['tokens']},
            attention_mask=self._joint_mask(vp, action.shape[1]),
            freqs_all={'video': vp['freqs'], 'action': ap['freqs']},
            context_all={'video': {'context': vp['context'], 'mask': vp['context_mask']},
                         'action': {'context': ap['context'], 'mask': ap['context_mask']}},
            t_mod_all={'video': vp['t_mod'], 'action': ap['t_mod']})
        pv = self.video_expert.post_dit(out['video'], vp)
        pa = self.action_expert.post_dit(out['action'], ap)
        lv = self._compute_video_loss_per_sample(
            pv, self.train_video_scheduler.training_target(future, nv, tv),
            inputs['image_is_pad'], include_initial_video_step=False)
        la = F.mse_loss(pa.float(), self.train_action_scheduler.training_target(action, na, ta).float(),
                        reduction='none').mean(dim=-1)
        if inputs['action_is_pad'] is not None:
            valid = (~inputs['action_is_pad']).to(la.dtype)
            la = (la*valid).sum(1)/valid.sum(1).clamp(min=1)
        else:
            la = la.mean(1)
        lv = (lv*self.train_video_scheduler.training_weight(tv)).mean()*self.loss_lambda_video
        la = (la*self.train_action_scheduler.training_weight(ta)).mean()*self.loss_lambda_action
        return lv+la, {'loss_video': float(lv.detach()), 'loss_action': float(la.detach()),
                      'helios_history_tokens': float(vp['meta']['mtm']['L_hist'])}

    @torch.no_grad()
    def infer_action(self, prompt, input_image, action_horizon, proprio=None,
                     context=None, context_mask=None, negative_prompt=None, text_cfg_scale=1.,
                     num_inference_steps=50, sigma_shift=None, seed=None, rand_device='cpu',
                     tiled=False, full_kv_cache=None, full_kv_frame_index=0,
                     native_cache_state=None, **kwargs):
        if action_horizon != 16 or full_kv_cache is not None or native_cache_state is not None:
            raise ValueError('Helios requires horizon=16 and fresh per-decision Transformer KV')
        if self.proprio_encoder is not None and proprio is None:
            raise ValueError('This checkpoint requires proprioception, as in training')
        if text_cfg_scale != 1. or negative_prompt not in (None, '') or any(kwargs.values()):
            raise ValueError('Helios comparison does not enable guidance or external event-memory modes')
        self.eval()
        observations = self._encode_input_image_latents_tensor(input_image, tiled=tiled)
        if observations.shape[2] != full_kv_frame_index+1:
            raise ValueError('Online VAE prefix and decision index are inconsistent')
        history, _ = fixed_history(observations[0].permute(1, 0, 2, 3).unsqueeze(2),
                                   sum(self.helios_config['history_sizes']))
        anchors, _ = episode_anchors(
            observations[0].permute(1, 0, 2, 3).unsqueeze(2),
            self.helios_config['anchor_frames'])
        if prompt is not None:
            if context is not None or context_mask is not None:
                raise ValueError('Supply either prompt or cached text context')
            context, context_mask = self.encode_prompt(prompt)
        elif context is None or context_mask is None:
            raise ValueError('Text conditioning is required')
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if context_mask.ndim == 1:
            context_mask = context_mask.unsqueeze(0)
        context = context.to(device=self.device, dtype=self.torch_dtype)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool)
        action_context, action_mask = context, context_mask
        if proprio is not None:
            proprio = proprio.reshape(1, -1).to(device=self.device, dtype=self.torch_dtype)
            action_context, action_mask = self._append_proprio_to_context(context, context_mask, proprio)
        future = observations[:, :, -1:].new_zeros(observations[:, :, -1:].shape)
        vp = self._video_pre(anchors.unsqueeze(0), history.unsqueeze(0), future,
                             torch.zeros(1, device=self.device, dtype=self.torch_dtype), context, context_mask)
        # Future placeholders never influence history and are omitted from prefill.
        h = vp['meta']['mtm']['L_hist']
        kv = self.mot.prefill_video_cache(
            video_tokens=vp['tokens'][:, :h], video_freqs=vp['freqs'][:h],
            video_t_mod=vp['t_mod'][:, :h],
            video_context_payload={'context': vp['context'], 'mask': vp['context_mask'][:, :h]},
            video_attention_mask=torch.ones((h, h), dtype=torch.bool, device=self.device))
        gen = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action = torch.randn((1, action_horizon, self.action_expert.action_dim),
                              generator=gen, device=rand_device).to(device=self.device, dtype=self.torch_dtype)
        mask = torch.ones((action_horizon, h+action_horizon), dtype=torch.bool, device=self.device)
        freqs = self._action_freqs(action_horizon, vp)
        steps, deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps, self.device, action.dtype, shift_override=sigma_shift)
        for t, delta in zip(steps, deltas):
            prediction = self._predict_action_noise_with_cache(
                action, t.unsqueeze(0), action_context, action_mask, freqs, kv, mask, h)
            action = self.infer_action_scheduler.step(prediction, delta, action)
        return {'action': action[0].float().cpu(), 'full_kv_cache': None, 'native_cache_state': None,
                'current_observation_latent': observations[:, :, -1:],
                'full_kv_frame_index': full_kv_frame_index, 'full_kv_video_tokens': h,
                'action_context': action_context, 'action_context_mask': action_mask,
                'video_context': context, 'video_context_mask': context_mask}

    def _helios_inference_contract(self):
        return {'action_horizon': 16, 'action_rope_spatial_mode': self.action_rope_spatial_mode,
                'action_scheduler_shift': float(self.infer_action_scheduler.shift),
                'video_scheduler_shift': float(self.infer_video_scheduler.shift),
                'action_num_train_timesteps': self.infer_action_scheduler.num_train_timesteps,
                'video_num_train_timesteps': self.infer_video_scheduler.num_train_timesteps,
                'history_vae': 'continuous_episode_stride4_causal_vae',
                'replan_stride': 16, 'anchor_frames': self.helios_config['anchor_frames'],
                'anchor_source_indices': ([0, 16] if self.helios_config['anchor_frames'] else []),
                'cross_decision_kv': False}

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {'mot': self.mot.state_dict(), 'step': step,
                   'helios_contract': self.helios_config,
                   'inference_contract': self._helios_inference_contract(),
                   'memory_backend': 'helios', 'proprio_encoder': self.proprio_encoder.state_dict()
                   if self.proprio_encoder is not None else None}
        if optimizer is not None:
            payload['optimizer'] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location='cpu', weights_only=False)
        saved_helios = dict(payload.get('helios_contract') or {})
        saved_helios.setdefault('anchor_frames', 0)
        if payload.get('memory_backend') != 'helios' or saved_helios != self.helios_config:
            raise ValueError('Checkpoint memory backend/history contract differs from Helios config')
        saved_inference = dict(payload.get('inference_contract') or {})
        saved_inference.setdefault('anchor_frames', 0)
        saved_inference.setdefault('anchor_source_indices', [])
        if saved_inference != self._helios_inference_contract():
            raise ValueError('Checkpoint inference contract differs from runtime config')
        self.mot.load_state_dict(payload['mot'], strict=True)
        if self.proprio_encoder is not None:
            self.proprio_encoder.load_state_dict(payload['proprio_encoder'], strict=True)
        if optimizer is not None and 'optimizer' in payload:
            optimizer.load_state_dict(payload['optimizer'])
        return payload.get('step')
