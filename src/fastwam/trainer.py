import hashlib
import logging
import json
import inspect
import os
import re
from datetime import timedelta
from math import ceil
from pathlib import Path
import time

import numpy as np
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from omegaconf import DictConfig
from PIL import Image
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .utils.fs import ensure_dir
from .utils.compact_checkpoint import (
    export_portable_model_state,
    export_validate_and_prune,
    prune_old_training_states,
    validate_portable_checkpoint,
)
from .utils.logging_config import get_logger, setup_logging
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import ResumableEpochSampler
from .utils.video_io import save_mp4
from .utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim

logger = get_logger(__name__)


def build_accelerator_kwargs_handlers(
    timeout_seconds: int,
) -> list[InitProcessGroupKwargs]:
    timeout_seconds = int(timeout_seconds)
    if timeout_seconds <= 0:
        raise ValueError("`process_group_timeout_seconds` must be positive.")
    return [
        InitProcessGroupKwargs(timeout=timedelta(seconds=timeout_seconds))
    ]


def _normalize_save_steps(save_steps):
    if save_steps is None:
        return frozenset()
    normalized = frozenset(int(step) for step in save_steps)
    if any(step <= 0 for step in normalized):
        raise ValueError("`save_steps` entries must be positive integers.")
    return normalized


def _is_checkpoint_step(*, step: int, save_every: int, save_steps) -> bool:
    step = int(step)
    return step in save_steps or (
        int(save_every) > 0 and step % int(save_every) == 0
    )


def _is_training_state_step(
    *, step: int, save_training_state_every: int, max_steps: int | None
) -> bool:
    """Select sparse exact-resume checkpoints while always retaining the final one."""

    step = int(step)
    interval = int(save_training_state_every)
    if step <= 0 or interval <= 0:
        raise ValueError("step and save_training_state_every must be positive")
    return (max_steps is not None and step == int(max_steps)) or step % interval == 0


def _validate_weight_only_resume_step(resume, configured_step: int) -> int:
    """Validate an opt-in cumulative step offset for portable-weight resumes."""

    step = int(configured_step)
    if step < 0:
        raise ValueError("`weight_only_resume_step` must be non-negative.")
    if step == 0:
        return 0
    if not resume:
        raise ValueError(
            "`weight_only_resume_step` requires `resume` to point to a portable .pt checkpoint."
        )
    resume_path = Path(str(resume))
    if resume_path.is_dir():
        raise ValueError(
            "`weight_only_resume_step` is only valid for portable weight checkpoints, "
            "not full training-state directories."
        )
    match = re.search(r"step[_-](\d+)\.pt$", resume_path.name)
    if match is None:
        raise ValueError(
            "Portable resume filename must end in `step_<N>.pt` when "
            "`weight_only_resume_step` is set."
        )
    filename_step = int(match.group(1))
    if filename_step != step:
        raise ValueError(
            "Portable resume step mismatch: "
            f"filename encodes {filename_step}, configured {step}."
        )
    return step


class Wan22Trainer:
    def __init__(self, model, train_dataset, val_dataset=None, *, cfg: DictConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.save_steps = _normalize_save_steps(cfg.get("save_steps", []))
        self.save_training_state = bool(cfg.get("save_training_state", True))
        self.save_training_state_every = int(
            cfg.get("save_training_state_every", self.save_every)
        )
        if self.save_training_state and self.save_training_state_every <= 0:
            raise ValueError("`save_training_state_every` must be positive.")
        self.save_final_checkpoint = bool(cfg.get("save_final_checkpoint", True))
        self.save_portable_fsdp_checkpoint = bool(
            cfg.get("save_portable_fsdp_checkpoint", True)
        )
        self.keep_training_state_checkpoints = int(
            cfg.get("keep_training_state_checkpoints", 1)
        )
        if self.keep_training_state_checkpoints < 1:
            raise ValueError("`keep_training_state_checkpoints` must be at least 1.")
        self.checkpoint_tmp_root = str(cfg.get("checkpoint_tmp_root", "/dev/shm"))
        self.eval_every = int(cfg.eval_every)
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)
        self.process_group_timeout_seconds = int(
            cfg.get("process_group_timeout_seconds", 600)
        )
        
        self.resume = cfg.resume
        self.weight_only_resume_step = _validate_weight_only_resume_step(
            self.resume,
            cfg.get("weight_only_resume_step", 0),
        )
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. "
                "Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.wandb_enabled = bool(cfg.wandb.enabled)
        self._last_saved_step = -1
        self._last_checkpoint_info = None

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
            kwargs_handlers=build_accelerator_kwargs_handlers(
                self.process_group_timeout_seconds
            ),
        )
        
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage = (
            deepspeed_plugin.deepspeed_config
            .get("zero_optimization", {})
            .get("stage", "n/a")
            if deepspeed_plugin is not None
            else "n/a"
        )
        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d grad_clip=%.4f",
            self.accelerator.distributed_type,
            zero_stage,
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
        )
        logger.info("using accelerator.device=%s", self.accelerator.device)
        # A weight-only checkpoint must be loaded before AdamW/DeepSpeed
        # creates its FP32 master parameters. Loading it after
        # accelerator.prepare() updates only the model copy; the first
        # optimizer step then restores stale pre-checkpoint master weights.
        self._weight_only_resume_loaded = False
        self._load_weight_only_resume_before_optimizer()
        self._verify_fresh_initialization_consistent()
        self.native_cache_train_mode = str(
            cfg.get("native_cache_train_mode", "full")
        ).strip().lower()
        if self.native_cache_train_mode not in {"full", "compressor_only"}:
            raise ValueError(
                "native_cache_train_mode must be 'full' or 'compressor_only', "
                f"got {self.native_cache_train_mode!r}"
            )
        if (
            self.native_cache_train_mode == "compressor_only"
            and getattr(self.model, "native_cache_compressor", None) is None
        ):
            raise ValueError(
                "compressor_only mode requires model.native_cache_compressor"
            )
        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")

        # Freeze non-trainable modules before optimizer/deepspeed initialization.
        # This keeps DiT (+ optional proprio encoder) as trainable when ZeRO builds optimizer state.
        self._apply_dit_only_train_mode(
            self.model,
            native_cache_train_mode=self.native_cache_train_mode,
        )
        self._inference_contract = {
            "video_scheduler_shift": float(self.model.infer_video_scheduler.shift),
            "action_scheduler_shift": float(self.model.infer_action_scheduler.shift),
            "video_num_train_timesteps": int(
                self.model.infer_video_scheduler.num_train_timesteps
            ),
            "action_num_train_timesteps": int(
                self.model.infer_action_scheduler.num_train_timesteps
            ),
            "video_attention_mask_mode": str(
                self.model.video_expert.video_attention_mask_mode
            ),
            "full_kv_cache": True,
        }
        native_compressor = getattr(
            self.model, "native_cache_compressor", None
        )
        layerwise_memory = getattr(
            self.model, "layerwise_block_memory", None
        )
        if layerwise_memory is not None:
            self._inference_contract.update(
                {
                    "native_cache_mode": "layerwise",
                    "memory_tokens": int(layerwise_memory.memory_tokens),
                    "dynamic_tokens_per_frame": (
                        None
                        if layerwise_memory.dynamic_tokens_per_frame is None
                        else int(layerwise_memory.dynamic_tokens_per_frame)
                    ),
                    "anchor_frames": int(layerwise_memory.anchor_frames),
                    "recent_frames": int(layerwise_memory.recent_frames),
                    "reader_use_anchor": bool(layerwise_memory.reader_use_anchor),
                    "reader_use_memory": bool(layerwise_memory.reader_use_memory),
                    "reader_use_recent": bool(layerwise_memory.reader_use_recent),
                    "variable_rate_rule": (
                        "information:K=8L,forced:K=8ceil(L/2)"
                        if cfg.get("memory_allocation_rule")
                        == "event_full_forced_half"
                        else (
                            None
                            if layerwise_memory.dynamic_tokens_per_frame is None
                            else "K=dynamic_tokens_per_frame*segment_span"
                        )
                    ),
                    "memory_allocation_rule": cfg.get("memory_allocation_rule"),
                }
            )
            data_train_cfg = cfg.get("data", {}).get("train", {})
            dataset_target = str(data_train_cfg.get("_target_", ""))
            if dataset_target.endswith(".FixedLengthK8RobotVideoDataset"):
                self._inference_contract.update(
                    {
                        "memory_segmentation_mode": "fixed_length",
                        "fixed_segment_length": int(
                            data_train_cfg.get("fixed_segment_length")
                        ),
                        "confirmation_delay": int(
                            data_train_cfg.get("confirmation_delay")
                        ),
                    }
                )
            elif dataset_target.endswith(
                ".PhysicalSettleRateDebtRobotVideoDataset"
            ):
                self._inference_contract["memory_segmentation_mode"] = (
                    "physical_settle_rate_debt"
                )
        if self.native_cache_train_mode == "compressor_only":
            trainable_params = list(native_compressor.parameters())
        else:
            trainable_params = list(self.model.dit.parameters())
            if native_compressor is not None:
                trainable_params.extend(list(native_compressor.parameters()))
            if layerwise_memory is not None:
                trainable_params.extend(list(layerwise_memory.parameters()))
        proprio_encoder = getattr(self.model, "proprio_encoder", None)
        if (
            proprio_encoder is not None
            and self.native_cache_train_mode != "compressor_only"
        ):
            trainable_params.extend(list(proprio_encoder.parameters()))
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
        
        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        warmup_ratio = float(cfg.get("warmup_ratio", 0.05))
        if not 0.0 <= warmup_ratio < 1.0:
            raise ValueError("`warmup_ratio` must be in [0, 1).")
        warmup_steps = int(total_train_steps * warmup_ratio)
        self.scheduler = self._build_scheduler(
            scheduler_type=cfg.lr_scheduler_type,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
        )
        self.global_step = 0
        if self._weight_only_resume_loaded and self.weight_only_resume_step:
            self.global_step = self.weight_only_resume_step
            logger.warning(
                "Portable weights loaded with cumulative global_step=%d; "
                "optimizer/scheduler state starts fresh.",
                self.global_step,
            )
        self.epoch = 0
        self.batch_in_epoch = 0

        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")

        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.eval_dir)

        # FSDP must not shard frozen inference-only components that are called
        # from the model forward (notably the causal VAE).  They have no
        # gradients and remain replicated, while the trainable MoT/ActionDiT
        # parameters are fully sharded.
        fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
        if fsdp_plugin is not None:
            ignored_modules = [self.model.vae]
            if getattr(self.model, "text_encoder", None) is not None:
                ignored_modules.append(self.model.text_encoder)
            if self.native_cache_train_mode == "compressor_only":
                ignored_modules.append(self.model.mot)
                if getattr(self.model, "proprio_encoder", None) is not None:
                    ignored_modules.append(self.model.proprio_encoder)
            fsdp_plugin.ignored_modules = ignored_modules
            logger.info(
                "FSDP ignored frozen modules: %s",
                [type(module).__name__ for module in ignored_modules],
            )

        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.wandb_run = None
        self._init_wandb()
        self._resume_or_load_checkpoint()

        val_size = len(self.val_dataset) if self.val_dataset is not None else len(self.train_dataset)
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)

    def _verify_fresh_initialization_consistent(self):
        action_expert = getattr(self.model, "action_expert", None)
        if action_expert is None:
            return
        modules = [
            ("action_encoder", action_expert.action_encoder),
            ("action_head", action_expert.head),
        ]
        proprio_encoder = getattr(self.model, "proprio_encoder", None)
        if proprio_encoder is not None:
            modules.append(("proprio_encoder", proprio_encoder))

        digest = hashlib.sha256()
        for module_name, module in modules:
            for parameter_name, tensor in module.state_dict().items():
                value = tensor.detach().contiguous()
                digest.update(f"{module_name}.{parameter_name}".encode("utf-8"))
                digest.update(str(tuple(value.shape)).encode("ascii"))
                digest.update(value.view(torch.uint8).cpu().numpy().tobytes())
        fingerprint = digest.hexdigest()

        fingerprints = [fingerprint]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            fingerprints = [
                None for _ in range(torch.distributed.get_world_size())
            ]
            torch.distributed.all_gather_object(fingerprints, fingerprint)
        if len(set(fingerprints)) != 1:
            raise RuntimeError(
                "Fresh ActionDiT/proprio initialization differs across ranks: "
                f"{fingerprints}"
            )
        if self.accelerator.is_main_process:
            logger.info(
                "Fresh initialization fingerprint (action I/O + proprio, "
                "all ranks verified): %s",
                fingerprint,
            )

    def _init_wandb(self):
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb logging is enabled in config (`wandb.enabled=true`) but wandb is not installed."
            ) from e

        self.wandb_run = wandb.init(
            entity=self.cfg.wandb.workspace,
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name,
            group=None if self.cfg.wandb.group in (None, "null", "") else str(self.cfg.wandb.group),
            mode=self.cfg.wandb.mode,
            dir=self.output_dir,
        )
        logger.info(
            "Initialized wandb run: workspace=%s project=%s name=%s",
            self.cfg.wandb.workspace,
            self.cfg.wandb.project,
            self.cfg.wandb.name,
        )

    def _wandb_log(self, payload: dict):
        if self.wandb_run is None:
            return
        self.wandb_run.log(payload, step=self.global_step)

    def _finish_wandb(self):
        if self.wandb_run is None:
            return
        self.wandb_run.finish()
        self.wandb_run = None

    def _build_loader(self, dataset, worker_init_fn=None):
        self.train_sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
        )

    def _assert_dataset_length_consistent(self, dataset, dataset_name: str):
        if not hasattr(dataset, "__len__"):
            raise TypeError(f"`{dataset_name}` must implement __len__ for rank consistency checks.")

        local_length = len(dataset)
        gathered_lengths = self.accelerator.gather(
            torch.tensor([local_length], device=self.accelerator.device, dtype=torch.int64)
        ).reshape(-1)
        if torch.all(gathered_lengths == gathered_lengths[0]):
            return

        if self.accelerator.is_main_process:
            print(f"[dataset-check] {dataset_name} length mismatch across ranks after initialization:")
            for rank, rank_length in enumerate(gathered_lengths.cpu().tolist()):
                print(f"rank {rank}: {rank_length}")
        self.accelerator.wait_for_everyone()
        raise RuntimeError(
            f"{dataset_name} length mismatch across ranks: {gathered_lengths.cpu().tolist()}"
        )

    def _estimate_total_train_steps(self) -> int:
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)

        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError("`train_dataset` must implement __len__ when `max_steps` is None.")

        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        micro_steps_per_epoch = max(ceil(len(self.train_dataset) / global_batch_size), 1)
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps),
            1,
        )
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)

        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_steps,
                eta_min=self.learning_rate * 0.01,
            )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=remaining_steps)
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. "
                "Expected one of: ['cosine', 'constant']."
            )

        if warmup_steps <= 0:
            return main_scheduler

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
    
    def _estimate_eta(self):
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-6)
        done_steps = max(self.global_step - self.run_start_step, 1)
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-9))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec

    def _resume_or_load_checkpoint(self):
        resume = self.resume
        if not resume:
            return
        if self._weight_only_resume_loaded:
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            logger.info("Resuming full training state from directory: %s", resume)
            self.load_training_state(str(resume_path))
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        logger.info("Loading weight checkpoint only: %s", resume)
        self.accelerator.unwrap_model(self.model).load_checkpoint(str(resume_path), optimizer=None)
        logger.warning("Loaded .pt weights only; optimizer/scheduler/step were not restored under ZeRO2.")

    def _load_weight_only_resume_before_optimizer(self):
        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        logger.info(
            "Loading weight checkpoint before optimizer/DeepSpeed initialization: %s",
            resume,
        )
        self.model.load_checkpoint(str(resume_path), optimizer=None)
        self._weight_only_resume_loaded = True
        logger.warning(
            "Loaded .pt weights before optimizer initialization; "
            "optimizer/scheduler/global_step start fresh."
        )

    def _set_dit_only_train_mode(self):
        # Match DiffSynth's freeze_except("dit"): only DiT stays trainable/in-train-mode.
        logger.info("Setting DiT to train mode and freezing other model components.")
        model = self.accelerator.unwrap_model(self.model)
        self._apply_dit_only_train_mode(
            model,
            native_cache_train_mode=self.native_cache_train_mode,
        )

    @staticmethod
    def _apply_dit_only_train_mode(
        model, native_cache_train_mode: str = "full"
    ):
        model.eval()
        model.requires_grad_(False)
        native_compressor = getattr(model, "native_cache_compressor", None)
        if native_cache_train_mode == "compressor_only":
            if native_compressor is None:
                raise ValueError(
                    "compressor_only mode requires model.native_cache_compressor"
                )
            native_compressor.train()
            native_compressor.requires_grad_(True)
            return
        if native_cache_train_mode != "full":
            raise ValueError(
                "native_cache_train_mode must be 'full' or 'compressor_only'"
            )
        model.dit.train()
        model.dit.requires_grad_(True)
        if native_compressor is not None:
            native_compressor.train()
            native_compressor.requires_grad_(True)
        layerwise_memory = getattr(model, "layerwise_block_memory", None)
        if layerwise_memory is not None:
            layerwise_memory.train()
            layerwise_memory.requires_grad_(True)
        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.train()
            proprio_encoder.requires_grad_(True)

    @staticmethod
    def _to_batched_eval_sample(sample):
        video = sample["video"]
        prompt = sample["prompt"]
        action = sample.get("action", None)
        proprio = sample.get("proprio", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"Expected tensor video for evaluation, got {type(video)}. "
                "Evaluation now expects `video` with shape [3,T,H,W] or [B,3,T,H,W]."
            )
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(f"Expected video shape [3,T,H,W] or [B,3,T,H,W], got {tuple(video.shape)}")
        num_video_frames = video.shape[2]
        if num_video_frames <= 1:
            raise ValueError(f"`sample['video']` must have at least 2 frames for action evaluation, got {num_video_frames}")

        if isinstance(prompt, str):
            prompt = [prompt]
        elif isinstance(prompt, tuple):
            prompt = list(prompt)
        elif not isinstance(prompt, list):
            raise TypeError(f"Expected prompt type str/list[str], got {type(prompt)}")
        if len(prompt) != video.shape[0]:
            raise ValueError(f"Prompt batch mismatch: len(prompt)={len(prompt)} vs video batch={video.shape[0]}")
        
        action_horizon = None
        action = None
        if "action" in sample:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                raise TypeError(
                    f"`sample['action']` must be a torch.Tensor, got {type(action)}"
                )
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3:
                raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
            if action.shape[1] % (num_video_frames - 1) != 0:
                raise ValueError(f"`sample['action']` temporal dimension must be divisible by video frames-1={num_video_frames - 1}, got {action.shape[1]}")
            action_horizon = int(action.shape[1])

        proprio = None
        if "proprio" in sample:
            proprio = sample["proprio"]
            if not isinstance(proprio, torch.Tensor):
                raise TypeError(f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}")
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both exist in eval sample.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )

        return {
            "video": video,
            "prompt": prompt,
            "action": action,
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
            "action_horizon": action_horizon,
        }

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None

        model = self.accelerator.unwrap_model(self.model)
        was_dit_training = model.dit.training
        model.eval()

        # eval_index = (self.global_step + self.accelerator.process_index) % len(self.val_dataset)
        rng = torch.Generator(device="cpu").manual_seed(self.global_step + self.accelerator.process_index)
        eval_index = torch.randint(0, len(self.val_dataset), (1,), generator=rng).item()
        sample = self._to_batched_eval_sample(self.val_dataset[eval_index])

        # 1. training loss
        with self.accelerator.autocast():
            val_loss, _ = model.training_loss(sample)
            val_loss = val_loss.float().item()
        
        prompt = sample["prompt"][0]
        video0 = sample["video"][0] # Tensor [3, T, H, W] in (-1, 1)
        action = sample["action"][0] if "action" in sample and sample["action"] is not None else None
        proprio = sample["proprio"][0, 0] if "proprio" in sample and sample["proprio"] is not None else None # from [1, T, d] to [d]
        input_image = video0[:, 0].unsqueeze(0)
        _, num_frames, _, _ = video0.shape

        # 2. inference and video saving
        infer_kwargs = {
            "input_image": input_image,
            "num_frames": num_frames,
            "action": action,
            "action_horizon": sample['action_horizon'],
            "proprio": proprio,
            "text_cfg_scale": 1.0,
            "action_cfg_scale": 1.0,
            "num_inference_steps": self.eval_num_inference_steps,
            "seed": 42,
            "tiled": False,
        }
        if sample["context"] is not None:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = sample["context"][0]
            infer_kwargs["context_mask"] = sample["context_mask"][0]
        else:
            infer_kwargs["prompt"] = prompt

        pred = model.infer(
            **infer_kwargs,
        )
        
        pred_video = pred["video"]
        pred_action = pred.get("action", None)

        # 3. inference metrics against GT video
        pred_video_tensor = pil_frames_to_video_tensor(pred_video)
        gt_video_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()

        assert pred_video_tensor.shape == gt_video_tensor.shape, (
            "Eval infer prediction/GT shape mismatch: "
            f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_rollout_vs_gt = video_psnr(pred=pred_video_tensor, target=gt_video_tensor)
        ssim_rollout_vs_gt = video_ssim(pred=pred_video_tensor, target=gt_video_tensor)

        action_l1 = None
        action_l2 = None
        if action is not None and pred_action is not None:
            if sample["proprio"] is None:
                raise ValueError("Eval sample must contain `proprio` for action denormalization.")
            proprio = sample["proprio"].detach().to(device="cpu", dtype=torch.float32)
            
            processor = self.val_dataset.lerobot_dataset.processor

            denorm_actions = {}
            action_meta = processor.shape_meta["action"]
            state_meta = processor.shape_meta["state"]
            for action_name, raw_action in (("pred", pred_action), ("gt", action)):
                if not isinstance(raw_action, torch.Tensor):
                    raise TypeError(f"{action_name} action must be a torch.Tensor, got {type(raw_action)}")
                if raw_action.ndim == 2:
                    action_btd = raw_action.unsqueeze(0)
                elif raw_action.ndim == 3 and raw_action.shape[0] == 1:
                    action_btd = raw_action
                else:
                    raise ValueError(
                        f"{action_name} action must have shape [T, D] or [1, T, D], got {tuple(raw_action.shape)}"
                    )
                action_btd = action_btd.detach().to(device="cpu", dtype=torch.float32)

                batch = {
                    "action": action_btd,
                    "state": proprio,
                }
                batch = processor.action_state_merger.backward(batch)
                batch = processor.normalizer.backward(batch)
                merged_batch = {
                    "action": {meta["key"]: batch["action"][meta["key"]].squeeze(0) for meta in action_meta},
                    "state": {meta["key"]: batch["state"][meta["key"]].squeeze(0) for meta in state_meta},
                }
                merged_batch = processor.action_state_merger.forward(merged_batch)
                denorm_action = merged_batch["action"].unsqueeze(0)
                if denorm_action.ndim != 3 or denorm_action.shape[0] != 1:
                    raise ValueError(
                        f"Denormalized {action_name} action must have shape [1, T, D], got {tuple(denorm_action.shape)}"
                    )
                denorm_actions[action_name] = denorm_action

            pred_action_denorm = denorm_actions["pred"]
            gt_action_denorm = denorm_actions["gt"]

            if pred_action_denorm.shape != gt_action_denorm.shape:
                raise ValueError(
                    "Predicted action/GT action shape mismatch after denormalization: "
                    f"pred={tuple(pred_action_denorm.shape)} vs gt={tuple(gt_action_denorm.shape)}"
                )
            action_diff = pred_action_denorm - gt_action_denorm
            action_l1 = action_diff.abs().mean().item()
            action_l2 = action_diff.pow(2).mean().item()

        # 4. VAE reconstruction metrics against GT video
        gt_video_batch = video0.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
        vae_latents = model._encode_video_latents(gt_video_batch, tiled=False)
        vae_recon_video = model._decode_latents(vae_latents, tiled=False)
        vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)

        assert vae_video_tensor.shape == gt_video_tensor.shape, (
            "Eval VAE reconstruction/GT shape mismatch: "
            f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_decode_vs_gt = video_psnr(pred=vae_video_tensor, target=gt_video_tensor)
        ssim_decode_vs_gt = video_ssim(pred=vae_video_tensor, target=gt_video_tensor)

        psnr_rollout_vs_decode = video_psnr(pred=pred_video_tensor, target=vae_video_tensor)
        ssim_rollout_vs_decode = video_ssim(pred=pred_video_tensor, target=vae_video_tensor)

        stitched_video_tensor = torch.cat(
            [pred_video_tensor, vae_video_tensor, gt_video_tensor],
            dim=2,
        ).contiguous()
        stitched_frames = []
        for t in range(stitched_video_tensor.shape[1]):
            frame = (stitched_video_tensor[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
            stitched_frames.append(Image.fromarray(frame))

        video_path = os.path.join(
            self.eval_dir,
            f"step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}.mp4",
        )
        save_mp4(stitched_frames, video_path, fps=8)

        local_metrics = torch.tensor(
            [
                float(val_loss),
                float(psnr_rollout_vs_gt),
                float(ssim_rollout_vs_gt),
                float(psnr_rollout_vs_decode),
                float(ssim_rollout_vs_decode),
                float(psnr_decode_vs_gt),
                float(ssim_decode_vs_gt),
                float(action_l2) if action_l2 is not None else -1.0,
                float(action_l1) if action_l1 is not None else -1.0,
            ],
            device=self.accelerator.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        gathered_metrics = self.accelerator.gather_for_metrics(local_metrics)
        mean_metrics = gathered_metrics[:, :7].mean(dim=0)
        action_l2_mean = gathered_metrics[:, 7].mean().item() if action_l2 is not None else None
        action_l1_mean = gathered_metrics[:, 8].mean().item() if action_l1 is not None else None

        if was_dit_training:
            self._set_dit_only_train_mode()

        result = {
            "val_loss": float(mean_metrics[0].item()),
            "psnr_rg": float(mean_metrics[1].item()),
            "ssim_rg": float(mean_metrics[2].item()),
            "psnr_rd": float(mean_metrics[3].item()),
            "ssim_rd": float(mean_metrics[4].item()),
            "psnr_dg": float(mean_metrics[5].item()),
            "ssim_dg": float(mean_metrics[6].item()),
            "video_path": video_path,
        }
        if action_l2_mean is not None:
            result["action_l2"] = float(action_l2_mean)
        if action_l1_mean is not None:
            result["action_l1"] = float(action_l1_mean)
        return result

    def _save_weights_checkpoint(self, step_tag: str):
        model = self.accelerator.unwrap_model(self.model)
        ckpt_path = os.path.join(self.weights_dir, f"{step_tag}.pt")
        model.save_checkpoint(ckpt_path, optimizer=None, step=self.global_step)
        return ckpt_path

    def _save_trainer_state(self, state_path: str):
        state_file = os.path.join(state_path, "trainer_state.json")
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def save_checkpoint(self):
        step_tag = f"step_{self.global_step:06d}"
        if self._last_saved_step == self.global_step:
            return self._last_checkpoint_info

        self.accelerator.wait_for_everyone()
        ckpt_path = None
        is_fsdp = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        save_full_training_state = self.save_training_state and _is_training_state_step(
            step=self.global_step,
            save_training_state_every=self.save_training_state_every,
            max_steps=self.max_steps,
        )
        if is_fsdp and not self.save_portable_fsdp_checkpoint and not save_full_training_state:
            raise RuntimeError(
                "FSDP checkpoint step has neither portable export nor full training state."
            )
        if not is_fsdp and self.accelerator.is_main_process:
            ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
        elif is_fsdp and not save_full_training_state:
            # Accelerate gathers a full BF16 state on rank zero and returns an
            # empty mapping on the other ranks. This avoids writing optimizer
            # shards merely to produce an inference checkpoint.
            model_state = self.accelerator.get_state_dict(self.model)
            if self.accelerator.is_main_process:
                portable_path = os.path.join(self.weights_dir, f"{step_tag}.pt")
                ckpt_path = str(
                    export_portable_model_state(
                        model_state,
                        portable_path,
                        step=self.global_step,
                        inference_contract=self._inference_contract,
                    )
                )
            del model_state
        self.accelerator.wait_for_everyone()

        state_path = None
        if save_full_training_state:
            state_path = os.path.join(self.state_dir, step_tag)
            ensure_dir(state_path)
            self.accelerator.save_state(output_dir=state_path)
            if self.accelerator.is_main_process:
                self._save_trainer_state(state_path)
            self.accelerator.wait_for_everyone()
            if is_fsdp and self.accelerator.is_main_process:
                if self.save_portable_fsdp_checkpoint:
                    dcp_dir = os.path.join(state_path, "pytorch_model_fsdp_0")
                    portable_path = os.path.join(self.weights_dir, f"{step_tag}.pt")
                    portable, removed = export_validate_and_prune(
                        dcp_dir=dcp_dir,
                        output_path=portable_path,
                        state_root=self.state_dir,
                        step=self.global_step,
                        inference_contract=self._inference_contract,
                        keep_training_states=self.keep_training_state_checkpoints,
                        tmp_root=self.checkpoint_tmp_root,
                    )
                    ckpt_path = str(portable)
                    if removed:
                        logger.info(
                            "Pruned older complete FSDP training states: %s",
                            [str(path) for path in removed],
                        )
                else:
                    ckpt_path = f"{state_path} (FSDP sharded model state)"
            elif self.accelerator.is_main_process and ckpt_path is not None:
                validate_portable_checkpoint(
                    ckpt_path,
                    expected_step=self.global_step,
                )
                removed = prune_old_training_states(
                    self.state_dir,
                    keep=self.keep_training_state_checkpoints,
                )
                if removed:
                    logger.info(
                        "Pruned older complete DeepSpeed training states: %s",
                        [str(path) for path in removed],
                    )
            self.accelerator.wait_for_everyone()

        self._last_saved_step = self.global_step
        self._last_checkpoint_info = {
            "weights_path": ckpt_path,
            "state_path": state_path,
        }
        return self._last_checkpoint_info

    def load_training_state(self, state_dir: str):
        self.accelerator.load_state(input_dir=state_dir)
        state_file = Path(state_dir) / "trainer_state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.global_step = int(payload["global_step"])

            if "epoch" in payload and "batch_in_epoch" in payload:
                self.epoch = int(payload["epoch"])
                self.batch_in_epoch = int(payload["batch_in_epoch"])
                self.train_sampler.set_epoch_offset(self.epoch)
                self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
                logger.info(
                    "Restored dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d",
                    self.epoch,
                    self.batch_in_epoch,
                    self.batch_in_epoch * self.batch_size * self.accelerator.num_processes,
                )
            else:
                self.epoch = 0
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                logger.warning(
                    "State file does not contain `epoch`/`batch_in_epoch`; "
                    "optimizer/scheduler were restored, but dataloader progress resume is skipped."
                )
            self.accelerator.wait_for_everyone()
            return

        match = re.search(r"step[_-](\d+)$", str(state_dir).rstrip("/"))
        if match:
            self.global_step = int(match.group(1))
        else:
            self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.train_sampler.clear_resume_batch_offset()
        self.accelerator.wait_for_everyone()
        logger.info("Loaded accelerate training state from %s at step=%d", state_dir, self.global_step)
        logger.warning(
            "State file `%s` is missing; dataloader progress resume is skipped.",
            state_file,
        )

    def train(self):
        self._set_dit_only_train_mode()

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        if self.max_steps is None:
            raise ValueError("`max_steps` must be set before entering the while-step training loop.")

        logger.info("Starting training with max_steps=%d.", self.max_steps)
        data_iter = iter(self.train_loader)
        self.run_start_step = self.global_step
        self.run_start_time = time.perf_counter()

        while self.global_step < self.max_steps:
            try:
                sample = next(data_iter)
                self.batch_in_epoch += 1
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                data_iter = iter(self.train_loader)
                continue

            with self.accelerator.accumulate(self.model):
                with self.accelerator.autocast():
                    if getattr(self.accelerator.state, "fsdp_plugin", None) is not None:
                        # Enter through FSDP.forward so parameter all-gathers
                        # and post-forward resharding are honored.
                        loss, loss_dict = self.model(sample)
                    else:
                        train_model = (
                            self.model
                            if hasattr(self.model, "training_loss")
                            else self.accelerator.unwrap_model(self.model)
                        )
                        loss, loss_dict = train_model.training_loss(sample)
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    if not self.accelerator.optimizer_step_was_skipped:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    global_loss = float(
                        self.accelerator.gather(loss.detach().float().reshape(1)).mean().item()
                    )
                    global_loss_metrics = {}
                    for key, value in loss_dict.items():
                        metric_tensor = torch.tensor(float(value), device=loss.device, dtype=torch.float32).reshape(1)
                        global_loss_metrics[key] = float(
                            self.accelerator.gather(metric_tensor).mean().item()
                        )
                    grad_norm_tensor = torch.tensor(grad_norm, device=loss.device, dtype=torch.float32)
                    global_grad_norm = float(self.accelerator.gather(grad_norm_tensor).mean().item())

                    current_lr = float(self.optimizer.param_groups[0]["lr"])

                    if self.log_every > 0 and self.global_step % self.log_every == 0 and self.accelerator.is_main_process:
                        eta_str, steps_per_sec = self._estimate_eta()
                        description = "[train] epoch=%d step=%d/%d loss=%.4f " % (
                            self.epoch,
                            self.global_step,
                            self.max_steps,
                            global_loss,
                        )
                        if global_loss_metrics:
                            detail_str = " ".join([f"{k}={v:.4f}" for k, v in sorted(global_loss_metrics.items())])
                            description += detail_str + " "
                        description += "lr=%.2e speed=%.2f step/s, %.2f samples/s eta=%s" % (
                            current_lr,
                            steps_per_sec,
                            steps_per_sec * self.batch_size * self.accelerator.num_processes,
                            eta_str,
                        )
                        logger.info(description)

                        wandb_payload = {
                            "train/loss": global_loss,
                            "train/grad_norm": global_grad_norm,
                            "train/lr": current_lr,
                            "performance/steps_per_sec": steps_per_sec,
                            "performance/samples_per_sec": steps_per_sec * self.batch_size * self.accelerator.num_processes,
                        }
                        for key, value in global_loss_metrics.items():
                            wandb_payload[f"train/{key}"] = value
                        self._wandb_log(wandb_payload)

                    if (
                        self.eval_every > 0
                        and self.val_dataset is not None
                        and self.global_step % self.eval_every == 0
                    ):
                        metrics = self.evaluate()
                        self.accelerator.wait_for_everyone()
                        if metrics is not None and self.accelerator.is_main_process:
                            description = "[eval] step=%d val_loss=%.4f infer_psnr=%.4f infer_ssim=%.4f" % (
                                self.global_step,
                                metrics["val_loss"],
                                metrics["psnr_rd"],
                                metrics["ssim_rd"],
                            )
                            if "action_l2" in metrics:
                                description += " action_l2=%.4f" % metrics["action_l2"]
                            if "action_l1" in metrics:
                                description += " action_l1=%.4f" % metrics["action_l1"]
                            logger.info(description)
                            eval_payload = {
                                "eval/val_loss": float(metrics["val_loss"]),
                                "eval/psnr_rg": float(metrics["psnr_rg"]),
                                "eval/ssim_rg": float(metrics["ssim_rg"]),
                                "eval/psnr_rd": float(metrics["psnr_rd"]),
                                "eval/ssim_rd": float(metrics["ssim_rd"]),
                                "eval/psnr_dg": float(metrics["psnr_dg"]),
                                "eval/ssim_dg": float(metrics["ssim_dg"]),
                            }
                            if "action_l2" in metrics:
                                eval_payload["eval/action_l2"] = float(metrics["action_l2"])
                            if "action_l1" in metrics:
                                eval_payload["eval/action_l1"] = float(metrics["action_l1"])
                            self._wandb_log(eval_payload)

                    if _is_checkpoint_step(
                        step=self.global_step,
                        save_every=self.save_every,
                        save_steps=self.save_steps,
                    ):
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[ckpt] step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )

                    if self.global_step >= self.max_steps:
                        if self.save_final_checkpoint:
                            ckpt_info = self.save_checkpoint()
                            if self.accelerator.is_main_process:
                                logger.info(
                                    "[done] max_steps reached step=%d weights=%s state=%s",
                                    self.global_step,
                                    ckpt_info["weights_path"],
                                    ckpt_info["state_path"],
                                )
                        elif self.accelerator.is_main_process:
                            logger.info(
                                "[done] max_steps reached step=%d; final checkpoint disabled",
                                self.global_step,
                            )
                        return

        if self.save_final_checkpoint:
            ckpt_info = self.save_checkpoint()
            if self.accelerator.is_main_process:
                logger.info(
                    "[done] training finished step=%d weights=%s state=%s",
                    self.global_step,
                    ckpt_info["weights_path"],
                    ckpt_info["state_path"],
                )
        elif self.accelerator.is_main_process:
            logger.info(
                "[done] training finished step=%d; final checkpoint disabled",
                self.global_step,
            )
        
