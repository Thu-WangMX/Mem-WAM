"""Render WAM surprise confirmations against PutBack strong events."""

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
WIDTH, HEIGHT, TIMELINE_TOP = 640, 900, 720


def _camera_path(root: Path, camera: str, episode: int) -> Path:
    return root / "videos" / "chunk-000" / camera / f"episode_{episode:06d}.mp4"


def _episode_row(report: dict, episode: int) -> dict:
    rows = [row for row in report["episodes"] if int(row["episode"]) == episode]
    if len(rows) != 1:
        raise ValueError(f"semantic report has {len(rows)} rows for episode {episode}")
    return rows[0]


def _x(index: float, count: int) -> int:
    return 20 + round(600 * float(index) / max(count - 1, 1))


def _timeline(canvas: np.ndarray, trace: dict, row: dict, decision: int) -> None:
    count = int(trace["decision_count"])
    cv2.rectangle(canvas, (20, TIMELINE_TOP + 45), (620, 880), (80, 80, 80), 1)
    scores = trace["scores"]
    valid = [float(value) for value in scores if value is not None]
    lo, hi = (min(valid), max(valid)) if valid else (0.0, 1.0)
    hi = max(hi, lo + 1.0e-6)
    previous = None
    for index, value in enumerate(scores):
        if value is None:
            continue
        point = (_x(index, count), 835 - round(55 * (float(value) - lo) / (hi - lo)))
        if previous is not None:
            cv2.line(canvas, previous, point, (50, 220, 255), 2)
        previous = point
    for event_index, event in enumerate(row["strong_events"]):
        color = (0, 220, 0) if any(int(match["event_index"]) == event_index for match in row["primary_metrics"]["matches"]) else (0, 0, 255)
        x = _x(float(event["decision"]), count)
        cv2.line(canvas, (x, 765), (x, 880), color, 2)
    for confirmation in row["confirmations"]:
        x = _x(float(confirmation), count)
        cv2.circle(canvas, (x, 750), 5, (255, 0, 255), -1)
    for boundary in row["forced_boundaries"]:
        x = _x(boundary, count)
        cv2.line(canvas, (x, 765), (x, 880), (255, 255, 0), 1)
    for item in row["retroactive_boundaries"]:
        x = _x(item["peak_boundary"], count)
        cv2.line(canvas, (x, 765), (x, 880), (0, 140, 255), 2)
    for boundary in row["online_counterfactual"]["boundaries"][1:-1]:
        x = _x(boundary, count)
        cv2.line(canvas, (x, 765), (x, 880), (255, 180, 0), 2)
    x = _x(decision, count)
    cv2.line(canvas, (x, 740), (x, 890), (255, 255, 255), 2)


def render(analysis_root: Path, dataset_root: Path, episode: int, output: Path) -> None:
    trace = json.loads((analysis_root / "episodes" / f"episode_{episode:03d}.json").read_text())
    report = json.loads((analysis_root / "semantic_alignment.json").read_text())
    if report.get("schema_version") != "putback_wam_surprise_semantic_alignment_v1":
        raise ValueError("incompatible semantic report")
    row = _episode_row(report, episode)
    paths = [_camera_path(dataset_root, camera, episode) for camera in CAMERAS]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError(f"missing camera video among {paths}")
    captures = [cv2.VideoCapture(str(path)) for path in paths]
    if not all(capture.isOpened() for capture in captures):
        raise RuntimeError("failed to open camera videos")
    fps = float(captures[0].get(cv2.CAP_PROP_FPS)) or 50.0
    output.parent.mkdir(parents=True, exist_ok=True)
    intermediate = output.with_name(output.stem + ".intermediate.mp4")
    writer = cv2.VideoWriter(str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"), fps, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError("failed to open video writer")
    frame_indices = [int(value) for value in trace["decision_frame_indices"]]
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
            decision = min(decision, int(trace["decision_count"]) - 1)
            cv2.rectangle(canvas, (0, 0), (640, 105), (0, 0, 0), -1)
            cv2.putText(canvas, f"INIT-WAM semantic alignment | episode {episode}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
            cv2.putText(canvas, f"decision={decision:02d} magenta=confirmation green=matched red=miss", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
            cv2.putText(canvas, "orange=retroactive boundary blue=online boundary cyan=forced", (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
            counts = row["primary_metrics"]["counts"]
            cv2.putText(canvas, f"strong-event matched={counts['matched']}/{counts['events']} confirmations={counts['confirmations']}", (10, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 220, 255), 1)
            _timeline(canvas, trace, row, decision)
            writer.write(canvas)
            written += 1
    finally:
        writer.release()
        for capture in captures:
            capture.release()
    if not written:
        raise RuntimeError("no shared video frames")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(intermediate), "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)], check=True)
    intermediate.unlink()
    print(json.dumps({"episode": episode, "frames": written, "fps": fps, "output": str(output.resolve())}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--episode", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    render(Path(args.analysis_root).resolve(), Path(args.dataset_root).resolve(), args.episode, Path(args.output).resolve())


if __name__ == "__main__":
    main()
