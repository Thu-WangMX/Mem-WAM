#!/usr/bin/env python3
"""Build one validated 50-episode RMBench task as a LeRobot RGB dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np

from fastwam.datasets.lerobot.lerobot.lerobot_dataset import LeRobotDataset
from scripts.build_rmbench_putback_rgb_v3 import (
    CAMERAS,
    decode_simulator_rgb,
    save_frame_images,
    validate_rgb,
)


def load_seen_instruction(rmbench_root: Path, task: str) -> str:
    path = rmbench_root / "description" / "task_instruction" / f"{task}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = payload.get("seen")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"{path}: missing seen instructions")
    return str(candidates[0]).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rmbench-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rmbench_root = args.rmbench_root.resolve()
    output_root = args.output_root.resolve()
    task = args.task
    task_root = output_root / "lerobot" / task
    raw_root = rmbench_root / "data" / task / "demo_clean" / "data"
    raw_paths = sorted(raw_root.glob("episode*.hdf5"))
    if len(raw_paths) != 50:
        raise ValueError(f"Expected 50 {task} episodes, got {len(raw_paths)}")
    if task_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {task_root}")
        shutil.rmtree(task_root)

    features = {
        **{
            key: {
                "dtype": "video",
                "shape": (3, 480, 640),
                "names": ["channel", "height", "width"],
            }
            for key in CAMERAS
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (14,),
            "names": [f"joint_{index}" for index in range(14)],
        },
        "action": {
            "dtype": "float32",
            "shape": (14,),
            "names": [f"joint_{index}" for index in range(14)],
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=f"kevin_wang/rmbench_{task}_demo_clean_rgb_v3",
        root=task_root,
        fps=50,
        features=features,
        robot_type="aloha-agilex",
        use_videos=True,
        video_codec="h264",
        is_compute_episode_stats_image=False,
    )
    instruction = load_seen_instruction(rmbench_root, task)
    task_tokens = [task.replace("_", " "), instruction, "clean", "success"]

    for raw_path in raw_paths:
        with h5py.File(raw_path, "r") as handle:
            vector = np.asarray(handle["/joint_action/vector"], dtype=np.float32)
            if vector.ndim != 2 or vector.shape[1] != 14:
                raise ValueError(f"{raw_path}: expected [T,14], got {vector.shape}")
            episode_index = int(dataset.episode_buffer["episode_index"])
            for frame_index in range(vector.shape[0] - 1):
                images = {
                    key: decode_simulator_rgb(
                        handle[f"/observation/{camera}/rgb"][frame_index],
                        size_hw=(480, 640),
                    )
                    for key, camera in CAMERAS.items()
                }
                dataset.add_frame(
                    {
                        **images,
                        "observation.state": vector[frame_index],
                        "action": vector[frame_index + 1],
                    },
                    task=task_tokens,
                )
                save_frame_images(dataset, episode_index, frame_index, images)
            dataset.save_episode(raw_file_name=str(raw_path))
        print(f"converted {task}/{raw_path.name}", flush=True)

    rgb_report = output_root / "_build_logs" / "rgb_contract.json"
    validate_rgb(task_root, raw_paths, rgb_report)
    manifest = {
        "format": "rmbench_lerobot_v21_rgb_v3",
        "task": task,
        "episodes_per_task": 50,
        "fps": 50,
        "action_contract": "joint_vector[t+1]",
        "instruction_contract": "seen[0]",
        "color_contract": "simulator_rgb_preserved",
        "rgb_report": str(rgb_report),
        "dataset": str(task_root),
    }
    (output_root / "_build_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
