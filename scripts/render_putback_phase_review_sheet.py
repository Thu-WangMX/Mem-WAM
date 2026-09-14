from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def render_sheet(npz_path: str | Path, output: str | Path, *, columns: int = 3, samples: int = 18):
    payload = np.load(npz_path)
    frames = payload["frames"]
    indices = payload["frame_indices"]
    selected = np.linspace(0, len(frames) - 1, samples).round().astype(int)
    tile_h, tile_w = frames[0].shape[:2]
    rows = (samples + columns - 1) // columns
    canvas = np.zeros((rows * tile_h, columns * tile_w, 3), dtype=np.uint8)
    for slot, source in enumerate(selected):
        row, column = divmod(slot, columns)
        image = frames[source].copy()
        cv2.rectangle(image, (0, 0), (110, 22), (0, 0, 0), -1)
        cv2.putText(image, f"frame={int(indices[source])}", (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1, cv2.LINE_AA)
        canvas[row * tile_h:(row + 1) * tile_h, column * tile_w:(column + 1) * tile_w] = image
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite review sheet: {output}")
    if not cv2.imwrite(str(output), canvas):
        raise RuntimeError("OpenCV failed to write review sheet")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-npz", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=18)
    args = parser.parse_args()
    render_sheet(args.episode_npz, args.output, samples=args.samples)


if __name__ == "__main__":
    main()

