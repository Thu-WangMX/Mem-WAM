"""Render held-out PutBack comparison of three WAM warmup strategies."""

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
STRATEGY_COLORS = {
    "current": (255, 180, 0),
    "short_history": (255, 0, 255),
    "adjacent_cosine": (0, 220, 255),
}
WIDTH, HEIGHT = 640, 900


def _camera(root: Path, name: str, episode: int) -> Path:
    return root / "videos" / "chunk-000" / name / f"episode_{episode:06d}.mp4"


def _x(value: float, count: int) -> int:
    return 115 + round(500 * float(value) / max(count - 1, 1))


def render(comparison: Path, dataset: Path, episode: int, output: Path) -> None:
    report = json.loads((comparison / "report.json").read_text())
    if report.get("schema_version") != "putback_wam_warmup_comparison_v1":
        raise ValueError("incompatible comparison report")
    traces = {
        name: json.loads((comparison / "traces" / name / f"episode_{episode:03d}.json").read_text())
        for name in STRATEGY_COLORS
    }
    if any(int(trace["episode"]) != episode for trace in traces.values()):
        raise ValueError("trace episode mismatch")
    frame_indices = [int(value) for value in traces["current"]["decision_frame_indices"]]
    count = len(frame_indices)
    current_row = next(
        row for row in report["strategies"]["current"]["per_episode"]
        if int(row["episode"]) == episode
    )
    events = current_row["events"]
    paths = [_camera(dataset, name, episode) for name in CAMERAS]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError(f"camera input missing among {paths}")
    captures = [cv2.VideoCapture(str(path)) for path in paths]
    if not all(capture.isOpened() for capture in captures):
        raise RuntimeError("failed to open camera videos")
    fps = float(captures[0].get(cv2.CAP_PROP_FPS)) or 50.0
    output.parent.mkdir(parents=True, exist_ok=True)
    intermediate = output.with_name(output.stem + ".intermediate.mp4")
    writer = cv2.VideoWriter(str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"), fps, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError("failed to open video writer")
    written = 0
    try:
        while True:
            decoded = [capture.read() for capture in captures]
            if not all(ok for ok, _ in decoded):
                break
            high, left, right = [frame for _, frame in decoded]
            canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            canvas[:480] = cv2.resize(high, (640, 480))
            canvas[480:720, :320] = cv2.resize(left, (320, 240))
            canvas[480:720, 320:] = cv2.resize(right, (320, 240))
            decision = max(0, bisect.bisect_right(frame_indices, written) - 1)
            decision = min(decision, count - 1)
            cv2.rectangle(canvas, (0, 0), (640, 96), (0, 0, 0), -1)
            cv2.putText(canvas, f"INIT-WAM warmup held-out | episode {episode}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2)
            cv2.putText(canvas, f"decision={decision:02d} adjacent_threshold={report['calibration']['threshold']:.5f}", (10, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (60, 220, 255), 1)
            cv2.putText(canvas, f"advance={report['decision_rule']['advance_to_training_boundary_experiment']} white=gripper event circles=confirmation", (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (220, 220, 220), 1)
            for row_index, (name, color) in enumerate(STRATEGY_COLORS.items()):
                y = 755 + 47 * row_index
                cv2.putText(canvas, name, (5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
                cv2.line(canvas, (115, y), (615, y), (80, 80, 80), 1)
                for event in events:
                    x = _x(float(event["decision"]), count)
                    cv2.line(canvas, (x, y - 13), (x, y + 13), (255, 255, 255), 1)
                for boundary in traces[name]["boundaries"][1:-1]:
                    x = _x(boundary, count)
                    cv2.line(canvas, (x, y - 9), (x, y + 9), color, 1)
                for confirmation in traces[name]["peak_confirmed_at"]:
                    cv2.circle(canvas, (_x(confirmation, count), y), 5, color, -1)
                cv2.line(canvas, (_x(decision, count), y - 15), (_x(decision, count), y + 15), (180, 180, 180), 1)
            writer.write(canvas)
            written += 1
    finally:
        writer.release()
        for capture in captures:
            capture.release()
    if written == 0:
        raise RuntimeError("no shared video frames")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(intermediate), "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)], check=True)
    intermediate.unlink()
    print(json.dumps({"episode": episode, "frames": written, "fps": fps, "output": str(output.resolve())}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--episode", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    render(Path(args.comparison_root).resolve(), Path(args.dataset_root).resolve(), args.episode, Path(args.output).resolve())


if __name__ == "__main__":
    main()
