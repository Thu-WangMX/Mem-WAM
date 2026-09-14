"""Render reviewable H.264 videos for the locked control-information selector."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import cv2
import imageio_ffmpeg
import numpy as np
import torch

from fastwam.memory.control_information_boundary import (
    ControlInformationBoundaryState,
    contextual_information_z,
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_episodes(value: str) -> list[int]:
    episodes = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not episodes or len(episodes) != len(set(episodes)):
        raise ValueError("render episodes must be a non-empty unique list")
    return episodes


def _trace_rows(
    *,
    trace: dict[str, Any],
    episode_length: int,
    statistics: dict[str, Any],
    selector_config: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    frames = torch.as_tensor(trace["frame_indices"], dtype=torch.int64)
    information = torch.as_tensor(trace["information"], dtype=torch.float32)
    information_by_frame = {
        int(frame): float(value)
        for frame, value in zip(frames.tolist(), information.tolist())
    }
    state = ControlInformationBoundaryState(
        statistics=statistics, **selector_config
    )
    rows = {}
    for frame in range(0, int(episode_length), 4):
        value = None if frame < 16 else information_by_frame[frame]
        event = state.update(frame=frame, information=value)
        rows[frame] = {
            "information": value,
            "z": None
            if frame < 16
            else contextual_information_z(
                float(value), frame=frame, statistics=statistics
            ),
            "cusum": event.cusum if event is not None else state.cusum,
            "event": event,
        }
    return rows


def _group_index(boundaries: list[int], decision: int) -> int:
    for index, (left, right) in enumerate(zip(boundaries, boundaries[1:])):
        if left <= decision < right:
            return index
    return max(0, len(boundaries) - 2)


def _render_episode(
    *,
    episode: int,
    npz_path: Path,
    trace: dict[str, Any],
    episode_length: int,
    statistics: dict[str, Any],
    selector_config: dict[str, Any],
    segment_payload: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    payload = np.load(npz_path, allow_pickle=False)
    images = payload["frames"]
    frame_indices = payload["frame_indices"].astype(np.int64)
    if len(images) != len(frame_indices):
        raise ValueError(f"episode {episode} review frames are inconsistent")
    rows = _trace_rows(
        trace=trace,
        episode_length=episode_length,
        statistics=statistics,
        selector_config=selector_config,
    )
    width, height = 960, 400
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
    boundaries = [int(value) for value in segment_payload["boundaries"]]
    event_frames = []
    try:
        for image, raw_frame in zip(images, frame_indices.tolist()):
            frame = int(raw_frame)
            row = rows[frame]
            event = row["event"]
            if event is not None:
                event_frames.append(frame)
            view = cv2.resize(image, (width, 240), interpolation=cv2.INTER_LINEAR)
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            canvas[:240] = view
            if event is not None:
                cv2.rectangle(canvas, (2, 2), (width - 3, 237), (255, 64, 64), 5)
            decision = frame // 16
            group = _group_index(boundaries, decision)
            information = row["information"]
            z_value = row["z"]
            reason = "-" if event is None else event.reason
            lines = [
                f"episode={episode:03d}  simulator_frame={frame:03d}  planning_decision={decision:02d}  group={group:02d}",
                "counterfactual_information="
                + ("warmup" if information is None else f"{information:.5f}")
                + "  robust_z="
                + ("warmup" if z_value is None else f"{float(z_value):.3f}")
                + f"  CUSUM={float(row['cusum']):.3f}",
                f"boundary={reason}  forward_aligned_decision={math.ceil(frame / 16) if event is not None else '-'}",
            ]
            for index, line in enumerate(lines):
                cv2.putText(
                    canvas,
                    line,
                    (18, 278 + index * 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            writer.send(canvas.tobytes())
    finally:
        writer.close()
    return {
        "episode": episode,
        "file": output.name,
        "sha256": _sha256(output),
        "frames": len(images),
        "event_frames": event_frames,
        "codec": "h264/yuv420p",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--review-root", required=True)
    parser.add_argument("--traces-0-39", required=True)
    parser.add_argument("--traces-40-49", required=True)
    parser.add_argument("--planning-manifest", required=True)
    parser.add_argument("--contextual-statistics", required=True)
    parser.add_argument("--runtime-lock", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite selector videos: {output}")
    traces = {}
    lengths = {}
    trace_paths = [Path(args.traces_0_39), Path(args.traces_40_49)]
    for path in trace_paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        traces.update({int(key): value for key, value in payload["episodes"].items()})
        lengths.update(
            {int(key): int(value) for key, value in payload["episode_lengths"].items()}
        )
    runtime_lock_path = Path(args.runtime_lock).expanduser().resolve()
    runtime_lock = json.loads(runtime_lock_path.read_text())
    selector_config = runtime_lock["candidate"]["candidate"]["selector_config"]
    statistics = torch.load(
        args.contextual_statistics, map_location="cpu", weights_only=True
    )
    manifest_root = Path(args.planning_manifest).expanduser().resolve()
    review_root = Path(args.review_root).expanduser().resolve()
    temporary = output.with_name(output.name + ".tmp")
    temporary.mkdir(parents=True)
    rendered = []
    for episode in _parse_episodes(args.episodes):
        rendered.append(
            _render_episode(
                episode=episode,
                npz_path=review_root / "episodes" / f"episode_{episode:03d}.npz",
                trace=traces[episode],
                episode_length=lengths[episode],
                statistics=statistics,
                selector_config=selector_config,
                segment_payload=json.loads(
                    (manifest_root / "episodes" / f"episode_{episode:03d}.json").read_text()
                ),
                output=temporary / f"episode_{episode:03d}_control_information.mp4",
            )
        )
    report = {
        "schema_version": "putback_control_information_review_videos_v1",
        "runtime_lock_sha256": _sha256(runtime_lock_path),
        "trace_sha256": {str(path): _sha256(path) for path in trace_paths},
        "videos": rendered,
    }
    (temporary / "render_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    temporary.replace(output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
