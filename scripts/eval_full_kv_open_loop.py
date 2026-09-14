from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from fastwam.utils import misc
from fastwam.utils.pytorch_utils import set_global_seed


def _prefill_history(model, history, context, context_mask):
    cache = None
    total_tokens = 0
    last_pre = None
    for offset, latent in enumerate(history):
        video_pre = model.video_expert.pre_dit(
            x=latent.unsqueeze(0).to(
                device=model.device, dtype=model.torch_dtype
            ),
            timestep=torch.zeros(1, device=model.device, dtype=model.torch_dtype),
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=True,
            temporal_position_offset=offset,
        )
        current_tokens = int(video_pre["tokens"].shape[1])
        total_tokens += current_tokens
        cache = model.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=torch.ones(
                current_tokens,
                total_tokens,
                dtype=torch.bool,
                device=model.device,
            ),
            history_kv_cache=cache,
        )
        last_pre = video_pre
    return cache, total_tokens, last_pre


@torch.no_grad()
def _infer_sample(model, sample, num_inference_steps, sigma_shift, seed):
    video_context = sample["context"].unsqueeze(0).to(
        device=model.device, dtype=model.torch_dtype
    )
    video_context_mask = sample["context_mask"].unsqueeze(0).to(
        device=model.device, dtype=torch.bool
    )
    cache, video_seq_len, last_video_pre = _prefill_history(
        model,
        sample["history_latents"],
        video_context,
        video_context_mask,
    )
    action_context, action_context_mask = model._append_proprio_to_context(
        context=video_context,
        context_mask=video_context_mask,
        proprio=sample["proprio"][0:1].to(
            device=model.device, dtype=model.torch_dtype
        ),
    )
    action_horizon = int(sample["action"].shape[0])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    latents_action = torch.randn(
        (1, action_horizon, model.action_expert.action_dim),
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    ).to(device=model.device, dtype=model.torch_dtype)
    _, grid_h, grid_w = last_video_pre["meta"]["grid_size"]
    action_freqs = model._build_video_aligned_action_freqs(
        action_seq_len=action_horizon,
        temporal_base=float(len(sample["history_latents"]) - 1),
        grid_h=int(grid_h),
        grid_w=int(grid_w),
        device=model.device,
    )
    attention_mask = torch.ones(
        action_horizon,
        video_seq_len + action_horizon,
        dtype=torch.bool,
        device=model.device,
    )
    timesteps, deltas = model.infer_action_scheduler.build_inference_schedule(
        num_inference_steps=num_inference_steps,
        device=model.device,
        dtype=latents_action.dtype,
        shift_override=sigma_shift,
    )
    for timestep, delta in zip(timesteps, deltas):
        predicted_velocity = model._predict_action_noise_with_cache(
            latents_action=latents_action,
            timestep_action=timestep.unsqueeze(0).to(
                device=model.device, dtype=latents_action.dtype
            ),
            context=action_context,
            context_mask=action_context_mask,
            action_freqs=action_freqs,
            video_kv_cache=cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        latents_action = model.infer_action_scheduler.step(
            predicted_velocity, delta, latents_action
        )
    return latents_action[0].float().cpu()


def _global_stats(path: Path, key: str):
    payload = json.loads(path.read_text())[key]["default"]
    return (
        torch.tensor(payload["global_mean"], dtype=torch.float32),
        torch.tensor(payload["global_std"], dtype=torch.float32),
    )


def _sample_indices(dataset, count: int) -> list[int]:
    if count <= 0:
        raise ValueError("`--max-samples` must be positive")
    return sorted(
        set(
            np.linspace(
                0,
                len(dataset) - 1,
                min(count, len(dataset)),
                dtype=np.int64,
            ).tolist()
        )
    )


def _threshold_switch_events(values: torch.Tensor, threshold: float) -> list[dict]:
    values = values.detach().float().cpu()
    states = values >= threshold
    return [
        {
            "index": index,
            "from_qpos": float(values[index - 1].item()),
            "to_qpos": float(values[index].item()),
            "from_above_threshold": bool(states[index - 1].item()),
            "to_above_threshold": bool(states[index].item()),
        }
        for index in range(1, int(values.numel()))
        if bool(states[index].item()) != bool(states[index - 1].item())
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--config-name", default="train")
    parser.add_argument("--task", default="rmbench_putback_memorywam_fullkv_5k")
    parser.add_argument("--stats", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--sigma-shift", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gripper-dims", nargs="+", type=int, default=[6, 13])
    parser.add_argument("--gripper-threshold", type=float, default=0.5)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional Hydra override; may be supplied more than once.",
    )
    args = parser.parse_args()

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    work_dir = output.parent / f".{output.stem}_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    misc.register_work_dir(str(work_dir))
    set_global_seed(max(args.seed, 1), rank_offset=False)
    configs_root = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(
        version_base="1.3", config_dir=str(configs_root)
    ):
        cfg = compose(
            config_name=args.config_name,
            overrides=[
                f"task={args.task}",
                f"output_dir={work_dir}",
                *args.override,
            ],
        )
    dataset = instantiate(cfg.data.train)
    indices = _sample_indices(dataset, args.max_samples)
    samples = [dataset[index] for index in indices]

    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    model_cfg.load_text_encoder = False
    model_cfg.skip_dit_load_from_pretrain = True
    model_cfg.action_dit_pretrained_path = None
    model = instantiate(
        model_cfg,
        model_dtype=torch.bfloat16,
        device=args.device,
    ).eval()

    action_mean, action_std = _global_stats(Path(args.stats), "action")
    state_mean, state_std = _global_stats(Path(args.stats), "state")
    reports = []
    for checkpoint in args.checkpoints:
        checkpoint_path = Path(checkpoint).resolve()
        model.load_checkpoint(str(checkpoint_path))
        sample_reports = []
        started = time.perf_counter()
        for dataset_index, sample in zip(indices, samples):
            predicted = _infer_sample(
                model,
                sample,
                num_inference_steps=args.num_inference_steps,
                sigma_shift=args.sigma_shift,
                seed=args.seed,
            )
            target = sample["action"].float()
            valid = ~sample["action_is_pad"].bool()
            predicted = predicted[valid]
            target = target[valid]
            norm_abs = (predicted - target).abs()
            predicted_qpos = predicted * action_std + action_mean
            target_qpos = target * action_std + action_mean
            current_qpos = sample["proprio"][0].float() * state_std + state_mean
            physical_abs = (predicted_qpos - target_qpos).abs()
            hold_abs = (current_qpos.unsqueeze(0) - target_qpos).abs()
            gripper_trajectories = {}
            for dim in args.gripper_dims:
                if dim < 0 or dim >= int(predicted_qpos.shape[-1]):
                    raise ValueError(
                        f"Gripper dimension {dim} is outside action dimension "
                        f"{predicted_qpos.shape[-1]}"
                    )
                predicted_values = predicted_qpos[:, dim]
                target_values = target_qpos[:, dim]
                gripper_trajectories[str(dim)] = {
                    "current_qpos": float(current_qpos[dim].item()),
                    "predicted_qpos": predicted_values.tolist(),
                    "target_qpos": target_values.tolist(),
                    "predicted_above_threshold": (
                        predicted_values >= args.gripper_threshold
                    ).tolist(),
                    "target_above_threshold": (
                        target_values >= args.gripper_threshold
                    ).tolist(),
                    "predicted_switch_events": _threshold_switch_events(
                        predicted_values, args.gripper_threshold
                    ),
                    "target_switch_events": _threshold_switch_events(
                        target_values, args.gripper_threshold
                    ),
                }
            sample_report = {
                "dataset_index": dataset_index,
                "episode": int(sample["episode_index"]),
                "frame": int(sample["frame_index"]),
                "history_frames": int(sample["history_latents"].shape[0]),
                "normalized_mae": float(norm_abs.mean().item()),
                "normalized_first_action_mae": float(norm_abs[0].mean().item()),
                "qpos_mae": float(physical_abs.mean().item()),
                "qpos_first_action_mae": float(physical_abs[0].mean().item()),
                "hold_current_qpos_mae": float(hold_abs.mean().item()),
                "normalized_mae_per_dim": norm_abs.mean(dim=0).tolist(),
                "normalized_first_action_abs_per_dim": norm_abs[0].tolist(),
                "qpos_mae_per_dim": physical_abs.mean(dim=0).tolist(),
                "qpos_first_action_abs_per_dim": physical_abs[0].tolist(),
                "hold_current_qpos_mae_per_dim": hold_abs.mean(dim=0).tolist(),
                "gripper_trajectories": gripper_trajectories,
            }
            sample_reports.append(sample_report)
            print(json.dumps(sample_report), flush=True)
        reports.append(
            {
                "checkpoint": str(checkpoint_path),
                "elapsed_seconds": time.perf_counter() - started,
                "normalized_mae": float(
                    np.mean([row["normalized_mae"] for row in sample_reports])
                ),
                "normalized_first_action_mae": float(
                    np.mean(
                        [row["normalized_first_action_mae"] for row in sample_reports]
                    )
                ),
                "qpos_mae": float(
                    np.mean([row["qpos_mae"] for row in sample_reports])
                ),
                "qpos_first_action_mae": float(
                    np.mean([row["qpos_first_action_mae"] for row in sample_reports])
                ),
                "hold_current_qpos_mae": float(
                    np.mean([row["hold_current_qpos_mae"] for row in sample_reports])
                ),
                "normalized_mae_per_dim": np.mean(
                    [row["normalized_mae_per_dim"] for row in sample_reports],
                    axis=0,
                ).tolist(),
                "normalized_first_action_abs_per_dim": np.mean(
                    [
                        row["normalized_first_action_abs_per_dim"]
                        for row in sample_reports
                    ],
                    axis=0,
                ).tolist(),
                "qpos_mae_per_dim": np.mean(
                    [row["qpos_mae_per_dim"] for row in sample_reports],
                    axis=0,
                ).tolist(),
                "qpos_first_action_abs_per_dim": np.mean(
                    [row["qpos_first_action_abs_per_dim"] for row in sample_reports],
                    axis=0,
                ).tolist(),
                "hold_current_qpos_mae_per_dim": np.mean(
                    [row["hold_current_qpos_mae_per_dim"] for row in sample_reports],
                    axis=0,
                ).tolist(),
                "samples": sample_reports,
            }
        )
    result = {
        "task": args.task,
        "num_inference_steps": args.num_inference_steps,
        "sigma_shift": args.sigma_shift,
        "seed": args.seed,
        "gripper_dims": args.gripper_dims,
        "gripper_threshold": args.gripper_threshold,
        "overrides": args.override,
        "sample_indices": indices,
        "reports": reports,
    }
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
