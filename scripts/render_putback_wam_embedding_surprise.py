"""Render a three-camera PutBack video with WAM-surprise diagnostics."""

from __future__ import annotations

import argparse
import bisect
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


CAMERAS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
CANVAS_WIDTH = 640
CANVAS_HEIGHT = 840
TIMELINE_TOP = 720


def _video_path(root: Path, camera: str, episode: int) -> Path:
    return (
        root
        / "videos"
        / "chunk-000"
        / camera
        / f"episode_{episode:06d}.mp4"
    )


def _format_value(value) -> str:
    return "warmup" if value is None else f"{float(value):.3f}"


def _draw_timeline(canvas: np.ndarray, payload: dict, decision: int) -> None:
    left, right = 20, CANVAS_WIDTH - 20
    top, bottom = TIMELINE_TOP + 35, CANVAS_HEIGHT - 18
    scores = payload["scores"]
    thresholds = payload["thresholds"]
    valid = [float(value) for value in scores if value is not None]
    upper = max(valid) if valid else 1.0
    lower = min(valid) if valid else 0.0
    if upper <= lower:
        upper = lower + 1.0

    def xy(index: int, value: float) -> tuple[int, int]:
        denominator = max(len(scores) - 1, 1)
        x = left + round((right - left) * index / denominator)
        y = bottom - round((bottom - top) * (float(value) - lower) / (upper - lower))
        return int(x), int(np.clip(y, top, bottom))

    cv2.rectangle(canvas, (left, top), (right, bottom), (70, 70, 70), 1)
    for boundary in payload["boundaries"]:
        if boundary >= len(scores):
            continue
        x, _ = xy(int(boundary), lower)
        cv2.line(canvas, (x, top), (x, bottom), (255, 220, 0), 1)
    previous = None
    for index, score in enumerate(scores):
        if score is None:
            continue
        point = xy(index, float(score))
        if previous is not None:
            cv2.line(canvas, previous, point, (60, 220, 255), 2)
        previous = point
        if index in payload["peaks"]:
            cv2.circle(canvas, point, 5, (0, 0, 255), -1)
    previous = None
    for index, threshold in enumerate(thresholds):
        if threshold is None:
            continue
        point = xy(index, float(threshold))
        if previous is not None:
            cv2.line(canvas, previous, point, (0, 255, 255), 1)
        previous = point
    current_x, _ = xy(decision, lower)
    cv2.line(canvas, (current_x, top), (current_x, bottom), (255, 255, 255), 2)


def _active_segment(payload: dict, decision: int) -> tuple[int, int]:
    boundaries = payload["boundaries"]
    index = max(0, bisect.bisect_right(boundaries, decision) - 1)
    index = min(index, len(boundaries) - 2)
    return int(boundaries[index]), int(boundaries[index + 1])


def render(*, analysis_root: Path, dataset_root: Path, episode: int, output: Path) -> None:
    payload_path = analysis_root / "episodes" / f"episode_{episode:03d}.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if int(payload["episode"]) != episode:
        raise ValueError("episode JSON does not match requested episode")

    paths = [_video_path(dataset_root, camera, episode) for camera in CAMERAS]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing camera videos: {missing}")
    captures = [cv2.VideoCapture(str(path)) for path in paths]
    if not all(capture.isOpened() for capture in captures):
        raise RuntimeError("failed to open one or more camera videos")
    fps = float(captures[0].get(cv2.CAP_PROP_FPS)) or 50.0

    output.parent.mkdir(parents=True, exist_ok=True)
    intermediate = output.with_name(output.stem + ".intermediate.mp4")
    writer = cv2.VideoWriter(
        str(intermediate),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (CANVAS_WIDTH, CANVAS_HEIGHT),
    )
    if not writer.isOpened():
        raise RuntimeError("failed to open intermediate video writer")

    frame_indices = [int(value) for value in payload["decision_frame_indices"]]
    frame_number = 0
    written = 0
    try:
        while True:
            decoded = [capture.read() for capture in captures]
            if not all(ok for ok, _frame in decoded):
                break
            high, left_wrist, right_wrist = [frame for _ok, frame in decoded]
            high = cv2.resize(high, (640, 480), interpolation=cv2.INTER_AREA)
            left_wrist = cv2.resize(left_wrist, (320, 240), interpolation=cv2.INTER_AREA)
            right_wrist = cv2.resize(right_wrist, (320, 240), interpolation=cv2.INTER_AREA)
            canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
            canvas[:480] = high
            canvas[480:720, :320] = left_wrist
            canvas[480:720, 320:] = right_wrist

            decision = bisect.bisect_right(frame_indices, frame_number) - 1
            decision = min(max(decision, 0), int(payload["decision_count"]) - 1)
            segment = _active_segment(payload, decision)
            score = payload["scores"][decision]
            threshold = payload["thresholds"][decision]
            is_peak = decision in payload["peaks"]
            confirmed = decision in payload["peak_confirmed_at"]

            overlay = canvas.copy()
            cv2.rectangle(overlay, (0, 0), (640, 96), (0, 0, 0), -1)
            canvas = cv2.addWeighted(overlay, 0.62, canvas, 0.38, 0)
            cv2.putText(
                canvas,
                f"INIT-WAM embedding surprise | expert episode {episode}",
                (12, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"decision={decision:02d} score={_format_value(score)} threshold={_format_value(threshold)}",
                (12, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (60, 220, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"active=[{segment[0]},{segment[1]}) len={segment[1]-segment[0]} peak={is_peak} confirmed_now={confirmed}",
                (12, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.49,
                (0, 255, 120) if is_peak else (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(canvas, "cam_high", (8, 472), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(canvas, "left_wrist", (8, 712), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(canvas, "right_wrist", (328, 712), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(
                canvas,
                "segments=" + "/".join(str(value) for value in payload["segment_lengths"]),
                (20, TIMELINE_TOP + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            if decision in payload["boundaries"]:
                cv2.rectangle(canvas, (2, 2), (637, 717), (255, 220, 0), 4)
            _draw_timeline(canvas, payload, decision)
            writer.write(canvas)
            frame_number += 1
            written += 1
    finally:
        writer.release()
        for capture in captures:
            capture.release()
    if written == 0:
        raise RuntimeError("camera videos contained no shared frames")

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(intermediate),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )
    intermediate.unlink()
    print(
        json.dumps(
            {
                "episode": episode,
                "frames": written,
                "fps": fps,
                "output": str(output.resolve()),
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--episode", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    render(
        analysis_root=Path(args.analysis_root).resolve(),
        dataset_root=Path(args.dataset_root).resolve(),
        episode=int(args.episode),
        output=Path(args.output).resolve(),
    )


if __name__ == "__main__":
    main()
