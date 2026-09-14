#!/usr/bin/env python3
"""Diagnose action jitter on an expert RMBench trajectory.

This deliberately bypasses the simulator.  It feeds expert observations to the
deployed policy, records every denormalized action, and separates motion error
inside a chunk from discontinuity at replanning boundaries.  Optional repeated
seeds at selected frames measure diffusion sensitivity on an identical input.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch


RMBENCH = Path("/mnt/vepfs02/datasets/kevin_wang/code/projects/RMBench")
MODEL_ROOT = Path("/mnt/vepfs02/datasets/kevin_wang/models")


def decode_rgb(value):
    if isinstance(value, np.ndarray) and value.ndim == 3:
        return value
    bgr = cv2.imdecode(np.frombuffer(value, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("failed to decode training frame")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def make_observation(handle, frame: int):
    return {
        "observation": {
            name: {"rgb": decode_rgb(handle[f"/observation/{name}/rgb"][frame])}
            for name in ("head_camera", "left_camera", "right_camera")
        },
        "joint_action": {
            "vector": np.asarray(handle["/joint_action/vector"][frame], np.float32)
        },
    }


def rms(value) -> float:
    value = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.square(value).mean()))


def direction_reversals(action: np.ndarray, threshold: float = 1e-3) -> int:
    velocity = np.diff(action, axis=0)
    if len(velocity) < 2:
        return 0
    product = velocity[:-1] * velocity[1:]
    active = (np.abs(velocity[:-1]) > threshold) & (np.abs(velocity[1:]) > threshold)
    return int(((product < 0) & active).sum())


def arm_metrics(pred, gt, current, arm_slice):
    pred_arm = pred[:, arm_slice]
    gt_arm = gt[:, arm_slice]
    current_arm = current[arm_slice]
    gt_motion = gt_arm - current_arm[None]
    pred_motion = pred_arm - current_arm[None]
    return {
        "mae": float(np.abs(pred_arm - gt_arm).mean()),
        "gt_motion_rms": rms(gt_motion),
        "pred_motion_rms": rms(pred_motion),
        "gt_velocity_rms": rms(np.diff(gt_arm, axis=0)),
        "pred_velocity_rms": rms(np.diff(pred_arm, axis=0)),
        "pred_direction_reversals": direction_reversals(pred_arm),
        "gt_direction_reversals": direction_reversals(gt_arm),
        "first_jump_from_current_rms": rms(pred_arm[0] - current_arm),
    }


def aggregate(rows, arm: str, static_threshold: float):
    values = [row[arm] for row in rows]
    static = [value for value in values if value["gt_motion_rms"] <= static_threshold]
    source = static if static else values
    return {
        "decision_count": len(values),
        "static_decision_count": len(static),
        "mean_mae": float(np.mean([value["mae"] for value in values])),
        "mean_pred_velocity_rms": float(
            np.mean([value["pred_velocity_rms"] for value in values])
        ),
        "mean_gt_velocity_rms": float(
            np.mean([value["gt_velocity_rms"] for value in values])
        ),
        "static_mean_pred_motion_rms": float(
            np.mean([value["pred_motion_rms"] for value in source])
        ),
        "static_mean_gt_motion_rms": float(
            np.mean([value["gt_motion_rms"] for value in source])
        ),
        "static_pred_direction_reversals": int(
            sum(value["pred_direction_reversals"] for value in source)
        ),
        "static_gt_direction_reversals": int(
            sum(value["gt_direction_reversals"] for value in source)
        ),
    }


def infer_without_commit(policy, observation, instruction: str, seed: int):
    # Mirrors WorldActionRobotWinPolicy._infer_action_chunk, but leaves its
    # cache/frame counters untouched so seeds see exactly the same condition.
    import fastwam_policy.deploy_policy as deploy_policy

    image = policy._build_robotwin_image_tensor(observation)
    temporal_video = torch.stack([*policy._temporal_frames, image], dim=2)
    state = np.asarray(observation["joint_action"]["vector"], np.float32)
    proprio = policy._normalize_state(state)
    pred = policy.model.infer_action(
        prompt=deploy_policy.DEFAULT_PROMPT.format(task=instruction),
        input_image=temporal_video,
        action_horizon=policy.action_horizon,
        proprio=proprio,
        negative_prompt=policy.negative_prompt,
        text_cfg_scale=policy.text_cfg_scale,
        num_inference_steps=policy.num_inference_steps,
        sigma_shift=policy.sigma_shift,
        seed=seed,
        rand_device=policy.rand_device,
        tiled=policy.tiled,
        full_kv_cache=policy._full_kv_cache,
        full_kv_frame_index=policy._full_kv_frame_index,
    )
    return policy._denormalize_action(pred["action"])[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", default="swap_blocks")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat-frames", default="0,64,160,320")
    parser.add_argument("--repeat-seeds", default="0,1,2,3")
    parser.add_argument("--static-threshold", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()

    sys.path[:0] = [
        str(RMBENCH),
        str(RMBENCH / "policy"),
        str(cli.repo),
        str(cli.repo / "src"),
        str(cli.repo / "experiments/robotwin"),
    ]
    from fastwam_policy.deploy_policy import get_model

    instruction_payload = json.loads(
        (RMBENCH / "description/task_instruction" / f"{cli.task}.json").read_text()
    )
    instruction = instruction_payload["seen"][0]
    episode_path = (
        RMBENCH / "data" / cli.task / "demo_clean/data" / f"episode{cli.episode}.hdf5"
    )
    repeat_frames = {int(value) for value in cli.repeat_frames.split(",") if value}
    repeat_seeds = [int(value) for value in cli.repeat_seeds.split(",") if value]

    policy = get_model(
        {
            "sim_cfg_name": "sim_robotwin_full_kv.yaml",
            "sim_task": "robotwin_full_kv_eval",
            "ckpt_setting": str(cli.checkpoint),
            "device": cli.device,
            "mixed_precision": "bf16",
            "dataset_stats_path": str(cli.run / "dataset_stats.json"),
            "wan_model_id": str(MODEL_ROOT / "diffsynth/Wan-AI/Wan2.2-TI2V-5B"),
            "tokenizer_model_id": str(MODEL_ROOT / "diffsynth/Wan-AI/Wan2.1-T2V-1.3B"),
            "redirect_common_files": True,
            "skip_dit_load_from_pretrain": True,
            "action_horizon": 16,
            "replan_steps": 16,
            "num_inference_steps": 50,
            "sigma_shift": 1.0,
            "seed": cli.seed,
            "text_cfg_scale": 1.0,
            "negative_prompt": "",
            "rand_device": "cpu",
            "tiled": False,
            "timing_enabled": True,
        }
    )

    rows = []
    repeats = []
    previous_pred = None
    with h5py.File(episode_path, "r") as handle:
        qpos = np.asarray(handle["/joint_action/vector"], np.float32)
        for frame in range(0, len(qpos) - 16, 16):
            if frame > 0:
                for subframe in (frame - 12, frame - 8, frame - 4):
                    policy._temporal_frames.append(
                        policy._build_robotwin_image_tensor(make_observation(handle, subframe))
                    )
            observation = make_observation(handle, frame)
            if frame in repeat_frames:
                samples = [
                    infer_without_commit(policy, observation, instruction, seed)
                    for seed in repeat_seeds
                ]
                stack = np.stack(samples)
                repeats.append(
                    {
                        "frame": frame,
                        "seeds": repeat_seeds,
                        "across_seed_std": float(stack.std(axis=0).mean()),
                        "across_seed_pairwise_rms": float(
                            np.mean(
                                [rms(samples[i] - samples[j]) for i in range(len(samples)) for j in range(i)]
                            )
                        ),
                        "samples": [sample.tolist() for sample in samples],
                    }
                )
            pred = policy._infer_action_chunk(observation, instruction)
            gt = qpos[frame + 1 : frame + 17]
            current = qpos[frame]
            row = {
                "frame": frame,
                "all_mae": float(np.abs(pred - gt).mean()),
                "all_pred_velocity_rms": rms(np.diff(pred, axis=0)),
                "all_gt_velocity_rms": rms(np.diff(gt, axis=0)),
                "boundary_jump_from_current_rms": rms(pred[0] - current),
                "cross_chunk_plan_jump_rms": (
                    None if previous_pred is None else rms(pred[0] - previous_pred[-1])
                ),
                "left": arm_metrics(pred, gt, current, slice(0, 7)),
                "right": arm_metrics(pred, gt, current, slice(7, 14)),
                "current_qpos": current.tolist(),
                "pred_action_qpos": pred.tolist(),
                "gt_action_qpos": gt.tolist(),
            }
            rows.append(row)
            previous_pred = pred

    result = {
        "task": cli.task,
        "episode": cli.episode,
        "checkpoint": str(cli.checkpoint),
        "static_threshold": cli.static_threshold,
        "decision_count": len(rows),
        "mean_mae": float(np.mean([row["all_mae"] for row in rows])),
        "mean_boundary_jump_from_current_rms": float(
            np.mean([row["boundary_jump_from_current_rms"] for row in rows])
        ),
        "mean_cross_chunk_plan_jump_rms": float(
            np.mean([row["cross_chunk_plan_jump_rms"] for row in rows[1:]])
        ),
        "left": aggregate(rows, "left", cli.static_threshold),
        "right": aggregate(rows, "right", cli.static_threshold),
        "repeated_seed_tests": repeats,
        "rows": rows,
    }
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows" and key != "repeated_seed_tests"}, indent=2))
    print(json.dumps({"repeated_seed_tests": [{k: v for k, v in row.items() if k != "samples"} for row in repeats]}, indent=2))


if __name__ == "__main__":
    main()
