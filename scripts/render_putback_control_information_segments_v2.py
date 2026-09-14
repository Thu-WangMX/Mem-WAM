from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import imageio_ffmpeg
import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _episodes(value: str) -> list[int]:
    result = [int(item) for item in value.split(",") if item.strip()]
    if not result or len(result) != len(set(result)):
        raise ValueError("episodes must be a non-empty unique comma-separated list")
    return result


def _segment_index(boundaries: list[int], decision: int) -> int:
    for index, (left, right) in enumerate(zip(boundaries, boundaries[1:])):
        if left <= decision < right:
            return index
    return max(0, len(boundaries) - 2)


def _draw_timeline(
    canvas: np.ndarray, *, boundaries: list[int], decision: int
) -> None:
    left, right, top, bottom = 24, canvas.shape[1] - 24, 348, 382
    total = max(boundaries[-1], 1)
    colors = ((52, 152, 219), (46, 204, 113), (155, 89, 182), (241, 196, 15))
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        x0 = left + round((right - left) * start / total)
        x1 = left + round((right - left) * end / total)
        cv2.rectangle(canvas, (x0, top), (max(x0 + 1, x1), bottom), colors[index % 4], -1)
        cv2.line(canvas, (x0, top), (x0, bottom), (255, 255, 255), 1)
    cursor = left + round((right - left) * min(decision, total) / total)
    cv2.line(canvas, (cursor, top - 5), (cursor, bottom + 5), (0, 0, 255), 4)


def _render(
    *, episode: int, review_path: Path, segment_path: Path, output: Path
) -> dict[str, Any]:
    review = np.load(review_path, allow_pickle=False)
    images = review["frames"]
    frames = review["frame_indices"].astype(np.int64)
    if len(images) != len(frames):
        raise ValueError(f"episode {episode} review images and indices differ")
    segment = json.loads(segment_path.read_text())
    adaptation_factor = float(
        segment.get(
            "adaptation_factor", segment.get("episode_adaptation_factor")
        )
    )
    boundaries = [int(value) for value in segment["boundaries"]]
    detector_reason = {
        int(event["confirmation_frame"]): str(event["reason"])
        for event in segment["detector_events"]
    }
    aligned_reason = {
        int(row["aligned_frame"]): str(row["reason"])
        for row in segment["alignment"]
        if row["status"] == "kept"
    }
    width, height = 960, 410
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(output),
        (width, height),
        fps=8,
        codec="libx264",
        pix_fmt_in="rgb24",
        output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart", "-crf", "18"],
    )
    writer.send(None)
    try:
        for image, raw_frame in zip(images, frames.tolist()):
            frame = int(raw_frame)
            decision = frame // 16
            group = _segment_index(boundaries, decision)
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            canvas[:240] = cv2.resize(image, (width, 240), interpolation=cv2.INTER_LINEAR)
            detector = detector_reason.get(frame)
            aligned = aligned_reason.get(frame)
            if detector is not None:
                cv2.rectangle(canvas, (3, 3), (width - 4, 236), (255, 64, 64), 5)
            text = (
                f"episode={episode:03d} frame={frame:03d} decision={decision:02d} "
                f"segment={group:02d}/{len(boundaries) - 1:02d}"
            )
            cv2.putText(canvas, text, (18, 278), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(
                canvas,
                f"detector={detector or '-'}  forward_aligned={aligned or '-'}  episode_factor={adaptation_factor:.3f}",
                (18, 320),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            _draw_timeline(canvas, boundaries=boundaries, decision=decision)
            writer.send(canvas.tobytes())
    finally:
        writer.close()
    return {
        "episode": episode,
        "file": output.name,
        "sha256": _sha256(output),
        "frames": len(images),
        "boundaries": boundaries,
        "adaptation_factor": adaptation_factor,
        "detector_reason_counts": {
            reason: list(detector_reason.values()).count(reason)
            for reason in sorted(set(detector_reason.values()))
        },
        "codec": "h264/yuv420p",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--review-root", required=True)
    parser.add_argument("--planning-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite v2 segment videos: {output}")
    review_root = Path(args.review_root).expanduser().resolve()
    manifest_root = Path(args.planning_manifest).expanduser().resolve()
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale v2 video temporary: {temporary}")
    temporary.mkdir(parents=True)
    videos = []
    for episode in _episodes(args.episodes):
        videos.append(
            _render(
                episode=episode,
                review_path=review_root / "episodes" / f"episode_{episode:03d}.npz",
                segment_path=manifest_root / "episodes" / f"episode_{episode:03d}.json",
                output=temporary / f"episode_{episode:03d}_control_information_v2.mp4",
            )
        )
    report = {
        "schema_version": "putback_control_information_segment_videos_v2",
        "videos": videos,
    }
    (temporary / "render_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    temporary.replace(output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
