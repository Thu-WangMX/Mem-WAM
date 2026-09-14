from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import cv2
import h5py
import numpy as np
import pyarrow.parquet as pq

from fastwam.memory.phase_annotation import ANNOTATION_SCHEMA, TRANSITIONS
from scripts.precompute_full_kv_observation_latents import CAMERAS, _decode_rgb


def propose_gripper_events(
    actions: Sequence[Sequence[float]], frame_indices: Sequence[int], threshold: float = 0.5
) -> list[dict[str, Any]]:
    values = np.asarray(actions, dtype=np.float32)
    frames = [int(value) for value in frame_indices]
    if values.ndim != 2 or values.shape[1] != 14 or len(values) != len(frames):
        raise ValueError("actions must be [T,14] and align with frame_indices")
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for dimension in (6, 13):
        closed = values[:, dimension] >= float(threshold)
        for index in range(1, len(frames)):
            if bool(closed[index]) == bool(closed[index - 1]):
                continue
            kind = "gripper_close" if bool(closed[index]) else "gripper_open"
            grouped[(kind, frames[index])].append(dimension)
    return [
        {"kind": kind, "frame": frame, "dimensions": dimensions}
        for (kind, frame), dimensions in sorted(grouped.items(), key=lambda row: row[0][1])
    ]


def build_annotation_draft(
    *, episode: int, proposals: list[dict[str, Any]]
) -> dict[str, Any]:
    episode = int(episode)
    if episode < 30 or episode > 49:
        raise ValueError("draft episode must be between 30 and 49")
    return {
        "episode": episode,
        "split": "development" if episode < 40 else "heldout",
        "reviewed": False,
        "reviewer": None,
        "proposals": json.loads(json.dumps(proposals)),
        "transitions": [
            {
                "name": name,
                "status": "pending",
                "frame": None,
                "ambiguity_start": None,
                "ambiguity_end": None,
                "not_observed": None,
                "reason": None,
            }
            for name in TRANSITIONS
        ],
    }


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary output exists: {temporary}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_episode_control(
    dataset_root: str | Path, episode: int, frame_indices: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    parquet = (
        Path(dataset_root).expanduser().resolve()
        / "data"
        / "chunk-000"
        / f"episode_{int(episode):06d}.parquet"
    )
    table = pq.read_table(
        parquet, columns=["frame_index", "action", "observation.state"]
    )
    stored_frames = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
    requested = np.asarray([int(value) for value in frame_indices], dtype=np.int64)
    if stored_frames.tolist() != list(range(len(stored_frames))):
        raise ValueError(f"episode {episode} parquet frame_index is not contiguous")
    if requested.size and (
        int(requested.min()) < 0 or int(requested.max()) >= len(stored_frames)
    ):
        raise IndexError(f"episode {episode} requested control frame is out of range")
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    if actions.shape != (len(stored_frames), 14) or states.shape != (
        len(stored_frames),
        14,
    ):
        raise ValueError(f"episode {episode} control arrays must be [T,14]")
    return actions[requested], states[requested]


def _camera_strip(handle: h5py.File, frame: int) -> np.ndarray:
    images = []
    for camera in CAMERAS:
        image = _decode_rgb(handle[f"observation/{camera}/rgb"][frame])
        image = cv2.resize(image, (160, 120), interpolation=cv2.INTER_AREA)
        images.append(image)
    return np.concatenate(images, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lerobot-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", default="30-49")
    args = parser.parse_args()
    left, right = (int(value) for value in args.episodes.split("-", 1))
    episodes = list(range(left, right + 1))
    if episodes != list(range(30, 50)):
        raise ValueError("annotation preparation requires exactly episodes 30-49")
    root = Path(args.lerobot_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite annotation preparation root: {output}")
    (output / "drafts").mkdir(parents=True)
    (output / "episodes").mkdir()
    rows = [
        json.loads(line)
        for line in (root / "meta" / "episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    metadata = {int(row["episode_index"]): row for row in rows}
    drafts = []
    for episode in episodes:
        row = metadata[episode]
        raw_path = Path(row["raw_file_name"]).expanduser().resolve()
        length = int(row["length"])
        frame_indices = list(range(0, length, 4))
        action, proprio = load_episode_control(root, episode, frame_indices)
        with h5py.File(raw_path, "r") as handle:
            frames = np.stack([_camera_strip(handle, frame) for frame in frame_indices])
        proposals = propose_gripper_events(action, frame_indices)
        draft = build_annotation_draft(episode=episode, proposals=proposals)
        _atomic_json(output / "drafts" / f"episode_{episode:03d}.json", draft)
        np.savez_compressed(
            output / "episodes" / f"episode_{episode:03d}.npz",
            frames=frames,
            frame_indices=np.asarray(frame_indices, dtype=np.int64),
            actions=action,
            proprio=proprio,
        )
        drafts.append(draft)
    _atomic_json(
        output / "draft_manifest.json",
        {
            "schema_version": ANNOTATION_SCHEMA,
            "frame_stride": 4,
            "reviewed": False,
            "episodes": drafts,
        },
    )


if __name__ == "__main__":
    main()
