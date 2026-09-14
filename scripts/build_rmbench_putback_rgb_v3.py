#!/usr/bin/env python3
"""Build and validate the 50-episode RMBench Put Back LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import av
import cv2
import h5py
import numpy as np
from PIL import Image

from fastwam.datasets.lerobot.lerobot.lerobot_dataset import LeRobotDataset


CAMERAS = {
    "observation.images.cam_high": "head_camera",
    "observation.images.cam_left_wrist": "left_camera",
    "observation.images.cam_right_wrist": "right_camera",
}


def decode_simulator_rgb(
    encoded,
    size_hw: tuple[int, int] | None = None,
) -> np.ndarray:
    """Decode RMBench JPEG bytes without an additional channel swap."""
    raw = bytes(encoded).rstrip(b"\0")
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode RMBench JPEG")
    if size_hw is not None and image.shape[:2] != size_hw:
        target_h, target_w = size_hw
        image = cv2.resize(
            image,
            (target_w, target_h),
            interpolation=cv2.INTER_AREA,
        )
    return image


def load_seen_instruction(rmbench_root: Path) -> str:
    path = (
        rmbench_root
        / "description"
        / "task_instruction"
        / "put_back_block.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = payload.get("seen")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"{path}: missing seen instructions")
    return str(candidates[0]).strip()


def save_frame_images(
    dataset: LeRobotDataset,
    episode_index: int,
    frame_index: int,
    images: dict[str, np.ndarray],
) -> None:
    for key, image in images.items():
        path = dataset._get_image_file_path(episode_index, key, frame_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(path, format="JPEG", quality=95)


def decode_video_first_frame(path: Path) -> np.ndarray:
    with av.open(str(path)) as container:
        frame = next(container.decode(video=0), None)
    if frame is None:
        raise ValueError(f"No decodable frame: {path}")
    return frame.to_ndarray(format="rgb24")


def validate_rgb(
    task_root: Path,
    raw_paths: list[Path],
    output_path: Path,
) -> dict:
    rows = []
    for episode_index in (0, 1, 2):
        raw_path = raw_paths[episode_index]
        with h5py.File(raw_path, "r") as handle:
            for feature_key, raw_camera in CAMERAS.items():
                raw_rgb = decode_simulator_rgb(
                    handle[f"/observation/{raw_camera}/rgb"][0]
                )
                short_key = feature_key.removeprefix("observation.images.")
                video_path = (
                    task_root
                    / "videos"
                    / "chunk-000"
                    / feature_key
                    / f"episode_{episode_index:06d}.mp4"
                )
                video_rgb = decode_video_first_frame(video_path)
                if video_rgb.shape[:2] != raw_rgb.shape[:2]:
                    video_rgb = cv2.resize(
                        video_rgb,
                        (raw_rgb.shape[1], raw_rgb.shape[0]),
                        interpolation=cv2.INTER_AREA,
                    )
                correct = float(
                    np.abs(
                        video_rgb.astype(np.float32)
                        - raw_rgb.astype(np.float32)
                    ).mean()
                )
                swapped = float(
                    np.abs(
                        video_rgb.astype(np.float32)
                        - raw_rgb[..., ::-1].astype(np.float32)
                    ).mean()
                )
                rows.append(
                    {
                        "episode": episode_index,
                        "camera": short_key,
                        "correct_rgb_mae": correct,
                        "red_blue_swapped_mae": swapped,
                    }
                )
    correct = float(np.mean([row["correct_rgb_mae"] for row in rows]))
    swapped = float(np.mean([row["red_blue_swapped_mae"] for row in rows]))
    passed = correct / max(swapped, 1.0e-12) <= 0.75 and swapped - correct >= 1.0
    report = {
        "samples": len(rows),
        "mean_correct_rgb_mae": correct,
        "mean_red_blue_swapped_mae": swapped,
        "passed": passed,
        "rows": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    if not passed:
        raise RuntimeError(f"RGB contract failed: {report}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rmbench-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rmbench_root = args.rmbench_root.resolve()
    output_root = args.output_root.resolve()
    task_root = output_root / "put_back_block"
    raw_root = (
        rmbench_root
        / "data"
        / "put_back_block"
        / "demo_clean"
        / "data"
    )
    raw_paths = sorted(raw_root.glob("episode*.hdf5"))
    if len(raw_paths) != 50:
        raise ValueError(f"Expected 50 Put Back episodes, got {len(raw_paths)}")
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
        repo_id="kevin_wang/rmbench_put_back_block_demo_clean_rgb_v3",
        root=task_root,
        fps=50,
        features=features,
        robot_type="aloha-agilex",
        use_videos=True,
        video_codec="h264",
        is_compute_episode_stats_image=False,
    )
    instruction = load_seen_instruction(rmbench_root)
    task_tokens = ["put back block", instruction, "clean", "success"]

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
                save_frame_images(
                    dataset,
                    episode_index,
                    frame_index,
                    images,
                )
            dataset.save_episode(raw_file_name=str(raw_path))
        print(f"converted {raw_path.name}", flush=True)

    rgb_report = output_root / "_build_logs" / "rgb_contract.json"
    validate_rgb(task_root, raw_paths, rgb_report)
    manifest = {
        "format": "rmbench_lerobot_v21_rgb_v3",
        "task": "put_back_block",
        "episodes_per_task": 50,
        "fps": 50,
        "action_contract": "joint_vector[t+1]",
        "instruction_contract": "seen[0]",
        "color_contract": "simulator_rgb_preserved",
        "rgb_report": str(rgb_report),
    }
    (output_root / "_build_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
