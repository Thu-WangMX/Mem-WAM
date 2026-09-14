#!/usr/bin/env python3
"""Measure FullKV action predictions on an expert RMBench observation trajectory."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np


RMBENCH = Path("/mnt/vepfs02/datasets/kevin_wang/code/projects/RMBench")
FASTWAM = Path("/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_memory_fullkv")
MODEL_ROOT = Path("/mnt/vepfs02/datasets/kevin_wang/models")

sys.path[:0] = [
    str(RMBENCH),
    str(RMBENCH / "policy"),
    str(FASTWAM),
    str(FASTWAM / "src"),
    str(FASTWAM / "experiments/robotwin"),
]

from fastwam_policy.deploy_policy import get_model  # noqa: E402


def decode_rgb(value):
    if isinstance(value, np.ndarray) and value.ndim == 3:
        return value
    bgr = cv2.imdecode(np.frombuffer(value, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("failed to decode training frame")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def make_observation(handle, frame):
    return {
        "observation": {
            name: {"rgb": decode_rgb(handle[f"/observation/{name}/rgb"][frame])}
            for name in ("head_camera", "left_camera", "right_camera")
        },
        "joint_action": {
            "vector": np.asarray(handle["/joint_action/vector"][frame], np.float32)
        },
    }


def rms(value):
    return float(np.sqrt(np.square(value).mean()))


parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--episode", type=int, default=0)
parser.add_argument("--episode-path", type=Path, default=None)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output", type=Path, required=True)
cli = parser.parse_args()

instruction_payload = json.loads(
    (RMBENCH / "description/task_instruction" / f"{cli.task}.json").read_text()
)
instruction = instruction_payload["seen"][0]
episode_path = (
    cli.episode_path
    if cli.episode_path is not None
    else RMBENCH
    / "data"
    / cli.task
    / "demo_clean/data"
    / f"episode{cli.episode}.hdf5"
)

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
with h5py.File(episode_path, "r") as handle:
    qpos = np.asarray(handle["/joint_action/vector"], np.float32)
    for frame in range(0, len(qpos) - 16, 16):
        if frame > 0:
            for subframe in (frame - 12, frame - 8, frame - 4):
                image = policy._build_robotwin_image_tensor(
                    make_observation(handle, subframe)
                )
                policy._temporal_frames.append(image)
        pred = policy._infer_action_chunk(
            make_observation(handle, frame), instruction
        )
        gt = qpos[frame + 1 : frame + 17]
        current = qpos[frame]
        diff = pred - gt
        pred_velocity = np.diff(pred, axis=0)
        gt_velocity = np.diff(gt, axis=0)
        rows.append(
            {
                "frame": frame,
                "mae": float(np.abs(diff).mean()),
                "rmse": rms(diff),
                "first_action_mae": float(np.abs(pred[0] - gt[0]).mean()),
                "first_from_current_rms": rms(pred[0] - current),
                "pred_velocity_rms": rms(pred_velocity),
                "gt_velocity_rms": rms(gt_velocity),
                "per_joint_mae": np.abs(diff).mean(axis=0).tolist(),
                "current_qpos": current.tolist(),
                "pred_action_qpos": pred.tolist(),
                "gt_action_qpos": gt.tolist(),
                "pred_left_gripper": pred[:, 6].tolist(),
                "gt_left_gripper": gt[:, 6].tolist(),
                "pred_right_gripper": pred[:, 13].tolist(),
                "gt_right_gripper": gt[:, 13].tolist(),
            }
        )

result = {
    "task": cli.task,
    "episode": cli.episode,
    "checkpoint": str(cli.checkpoint),
    "instruction": instruction,
    "decision_count": len(rows),
    "mean_mae": float(np.mean([row["mae"] for row in rows])),
    "mean_first_action_mae": float(
        np.mean([row["first_action_mae"] for row in rows])
    ),
    "mean_first_from_current_rms": float(
        np.mean([row["first_from_current_rms"] for row in rows])
    ),
    "mean_pred_velocity_rms": float(
        np.mean([row["pred_velocity_rms"] for row in rows])
    ),
    "mean_gt_velocity_rms": float(
        np.mean([row["gt_velocity_rms"] for row in rows])
    ),
    "rows": rows,
}
cli.output.parent.mkdir(parents=True, exist_ok=True)
cli.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
