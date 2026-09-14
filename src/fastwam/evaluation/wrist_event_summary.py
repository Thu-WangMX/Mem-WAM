"""Structured summary for four-scene PutBack wrist-event evaluations."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Iterable, Sequence


_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_SUCCESS_RATE = re.compile(r"Success rate:\s*(\d+)\s*/\s*(\d+)", re.IGNORECASE)
_VIDEO_SUCCESS = re.compile(r"_success-(true|false)\.mp4$", re.IGNORECASE)


def _json_records(text: str, marker: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    prefix = marker + " "
    for line in text.splitlines():
        if not line.startswith(prefix):
            continue
        try:
            payload = json.loads(line[len(prefix) :])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _scene_videos(run_root: Path, scene: int, policy: int) -> list[Path]:
    needle = f"scene{scene}_policy{policy}"
    return sorted(
        path.resolve()
        for path in run_root.rglob("*.mp4")
        if needle in str(path)
    )


def _official_success(text: str) -> tuple[bool | None, int | None]:
    matches = list(_SUCCESS_RATE.finditer(_ANSI.sub("", text)))
    if not matches:
        return None, None
    succeeded, total = (int(value) for value in matches[-1].groups())
    if total <= 0:
        return None, total
    return succeeded == total, total


def summarize_wrist_event_run(
    run_root: str | Path,
    scene_policy_pairs: Iterable[Sequence[int]],
) -> dict[str, object]:
    """Parse official success, online-event records, cache metrics, and videos."""

    root = Path(run_root).expanduser().resolve()
    scenes: list[dict[str, object]] = []
    all_videos: list[str] = []
    total_event_closures = 0
    total_forecasts = 0

    for pair in scene_policy_pairs:
        scene, policy = (int(value) for value in pair)
        log_path = root / f"scene{scene}_policy{policy}.log"
        text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
        official_success, official_episodes = _official_success(text)
        arrivals = _json_records(text, "FASTWAM_WRIST_EVENT_ARRIVAL")
        forecasts = _json_records(text, "FASTWAM_WRIST_EVENT_FORECAST")
        cache_metrics = _json_records(text, "FASTWAM_NATIVE_CACHE_METRICS")
        event_closures = sum(
            record.get("reason") == "wrist_event" for record in arrivals
        )
        max_length_closures = sum(
            record.get("reason") == "max_length" for record in arrivals
        )
        videos = _scene_videos(root, scene, policy)
        video_success_values = [
            match.group(1).lower() == "true"
            for video in videos
            if (match := _VIDEO_SUCCESS.search(video.name)) is not None
        ]
        video_success = (
            video_success_values[-1] if video_success_values else None
        )
        total_event_closures += event_closures
        total_forecasts += len(forecasts)
        all_videos.extend(str(video) for video in videos)
        scenes.append(
            {
                "scene_seed": scene,
                "policy_seed": policy,
                "log_path": str(log_path),
                "log_exists": log_path.is_file(),
                "traceback": "Traceback (most recent call last)" in text,
                "official_success": official_success,
                "official_episodes": official_episodes,
                "video_success": video_success,
                "success_sources_agree": (
                    video_success is None
                    or official_success is None
                    or video_success == official_success
                ),
                "arrival_count": len(arrivals),
                "forecast_count": len(forecasts),
                "event_closures": event_closures,
                "max_length_closures": max_length_closures,
                "final_cache_metrics": cache_metrics[-1] if cache_metrics else None,
                "videos": [str(video) for video in videos],
            }
        )

    complete = bool(scenes) and all(
        scene["log_exists"]
        and not scene["traceback"]
        and scene["official_success"] is not None
        and bool(scene["videos"])
        and scene["success_sources_agree"]
        for scene in scenes
    )
    successes = sum(scene["official_success"] is True for scene in scenes)
    return {
        "run_root": str(root),
        "complete": complete,
        "episodes": len(scenes),
        "successes": successes,
        "success_rate": successes / len(scenes) if scenes else 0.0,
        "total_event_closures": total_event_closures,
        "total_forecasts": total_forecasts,
        "videos": all_videos,
        "scenes": scenes,
    }

