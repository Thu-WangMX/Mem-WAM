from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.render_putback_control_information_segments_v2 import (
    _episodes,
    _render,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--review-root", required=True)
    parser.add_argument("--planning-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite v3 segment videos: {output}")
    review_root = Path(args.review_root).expanduser().resolve()
    manifest_root = Path(args.planning_manifest).expanduser().resolve()
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale v3 video temporary: {temporary}")
    temporary.mkdir(parents=True)
    videos = []
    for episode in _episodes(args.episodes):
        videos.append(
            _render(
                episode=episode,
                review_path=review_root / "episodes" / f"episode_{episode:03d}.npz",
                segment_path=manifest_root / "episodes" / f"episode_{episode:03d}.json",
                output=temporary / f"episode_{episode:03d}_control_information_v3.mp4",
            )
        )
    report = {
        "schema_version": "putback_control_information_segment_videos_v3",
        "videos": videos,
    }
    (temporary / "render_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    temporary.replace(output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
