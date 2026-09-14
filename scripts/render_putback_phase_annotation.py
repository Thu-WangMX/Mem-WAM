from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from fastwam.memory.phase_annotation import TRANSITIONS, validate_episode_annotation


COLORS = (
    (255, 180, 0),
    (0, 220, 255),
    (80, 255, 80),
    (255, 80, 220),
    (0, 120, 255),
)


def _trace_points(values: np.ndarray, width: int, top: int, height: int) -> np.ndarray:
    low = float(values.min())
    high = float(values.max())
    scale = max(high - low, 1e-6)
    xs = np.linspace(0, width - 1, len(values))
    ys = top + height - 1 - (values - low) / scale * (height - 1)
    return np.stack([xs, ys], axis=1).round().astype(np.int32)


def render_annotation_video(
    *,
    frames: list[np.ndarray],
    frame_indices: list[int],
    actions: np.ndarray,
    proprio: np.ndarray,
    annotation: dict,
    output: str | Path,
    fps: int = 10,
) -> None:
    validate_episode_annotation(annotation)
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite annotation video: {output}")
    if not frames or len(frames) != len(frame_indices):
        raise ValueError("frames and frame_indices must be nonempty and aligned")
    if actions.shape != (len(frames), 14) or proprio.shape != (len(frames), 14):
        raise ValueError("actions and proprio must have shape [T,14]")
    height, width = frames[0].shape[:2]
    if any(frame.shape[:2] != (height, width) for frame in frames):
        raise ValueError("all rendered frames must share one size")
    panel_height = 190
    temporary = output.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height + panel_height)
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV failed to open annotation video writer")
    try:
        for index, (image, simulator_frame) in enumerate(zip(frames, frame_indices)):
            canvas = np.zeros((height + panel_height, width, 3), dtype=np.uint8)
            canvas[:height] = image
            cv2.putText(
                canvas,
                f"episode={annotation['episode']} frame={simulator_frame} detector_step={simulator_frame // 4}",
                (8, height + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            for transition_index, transition in enumerate(annotation["transitions"]):
                if transition["not_observed"]:
                    continue
                frame = int(transition["frame"])
                start = int(transition["ambiguity_start"])
                end = int(transition["ambiguity_end"])
                color = COLORS[transition_index]
                x = int(round(frame / max(frame_indices[-1], 1) * (width - 1)))
                cv2.line(canvas, (x, height + 38), (x, height + 84), color, 2)
                if start <= simulator_frame <= end:
                    cv2.putText(
                        canvas,
                        TRANSITIONS[transition_index],
                        (8, height + 46 + transition_index * 17),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        color,
                        1,
                        cv2.LINE_AA,
                    )
            for values, color, trace_top in (
                (actions[:, 6], (0, 255, 255), height + 100),
                (actions[:, 13], (255, 255, 0), height + 138),
            ):
                points = _trace_points(values, width, trace_top, 30)
                cv2.polylines(canvas, [points], False, color, 1, cv2.LINE_AA)
                cursor_x = int(round(index / max(len(frames) - 1, 1) * (width - 1)))
                cv2.line(canvas, (cursor_x, trace_top), (cursor_x, trace_top + 29), (255, 255, 255), 1)
            cv2.putText(
                canvas,
                f"proprio_norm={np.linalg.norm(proprio[index]):.3f}",
                (8, height + 184),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )
            writer.write(canvas)
    finally:
        writer.release()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("ffmpeg is required for H.264 annotation videos")
    result = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-i",
            str(temporary),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ],
        text=True,
        capture_output=True,
    )
    temporary.unlink(missing_ok=True)
    if result.returncode != 0 or not output.is_file():
        output.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed to encode annotation video: {result.stderr}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--episode-npz", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    annotation = json.loads(Path(args.annotation).read_text())
    episode = np.load(args.episode_npz)
    render_annotation_video(
        frames=[frame for frame in episode["frames"]],
        frame_indices=episode["frame_indices"].tolist(),
        actions=episode["actions"],
        proprio=episode["proprio"],
        annotation=annotation,
        output=args.output,
    )


if __name__ == "__main__":
    main()
