from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


def render_predictive_selector_video(
    *, frames, frame_indices, scores, high_threshold, low_threshold,
    transitions, events, output, fps=10,
) -> None:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite selector video: {output}")
    if not frames or len(frames) != len(frame_indices) or len(scores) != len(frames):
        raise ValueError("frames, indices, and scores must align")
    height, width = frames[0].shape[:2]
    panel = 150
    raw = output.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(
        str(raw), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height + panel)
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV failed to create selector video")
    maximum_frame = max(frame_indices[-1], 1)
    maximum_score = max(float(np.max(scores)), float(high_threshold), 1e-6)
    try:
        for index, (image, frame) in enumerate(zip(frames, frame_indices)):
            canvas = np.zeros((height + panel, width, 3), dtype=np.uint8)
            canvas[:height] = image
            cv2.putText(canvas, f"frame={frame} score={scores[index]:.2f}", (8, height + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, .48, (255,255,255), 1, cv2.LINE_AA)
            for value, color, label in ((high_threshold,(0,0,255),"high"),(low_threshold,(0,200,255),"low")):
                y = height + 130 - int(value / maximum_score * 90)
                cv2.line(canvas, (0,y), (width-1,y), color, 1)
                cv2.putText(canvas, label, (width-45,y-2), cv2.FONT_HERSHEY_SIMPLEX,.35,color,1)
            points = []
            for score_index, score in enumerate(scores):
                x = int(score_index / max(len(scores)-1,1) * (width-1))
                y = height + 130 - int(float(score) / maximum_score * 90)
                points.append((x,y))
            cv2.polylines(canvas, [np.asarray(points,np.int32)], False, (80,255,80), 2)
            cursor = int(index / max(len(frames)-1,1) * (width-1))
            cv2.line(canvas, (cursor,height+35), (cursor,height+140), (255,255,255), 1)
            for transition in transitions:
                if transition["ambiguity_start"] <= frame <= transition["ambiguity_end"]:
                    cv2.putText(canvas, f"GT {transition['name']}", (8,height+42),
                                cv2.FONT_HERSHEY_SIMPLEX,.42,(255,180,0),1,cv2.LINE_AA)
            for event_index, event in enumerate(events):
                x = int(event["frame"] / maximum_frame * (width-1))
                cv2.line(canvas, (x,height+30), (x,height+145), (255,80,220), 2)
                if event["group_start"] <= frame < event["group_end"]:
                    text = f"group={event['group_end']-event['group_start']} {event['reason']}"
                    cv2.putText(canvas,text,(8,height+62+17*(event_index%3)),
                                cv2.FONT_HERSHEY_SIMPLEX,.38,(255,80,220),1,cv2.LINE_AA)
            writer.write(canvas)
    finally:
        writer.release()
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raw.unlink(missing_ok=True)
        raise RuntimeError("ffmpeg and ffprobe are required")
    encoded = subprocess.run([ffmpeg,"-v","error","-y","-i",str(raw),"-c:v","libx264","-pix_fmt","yuv420p",str(output)],capture_output=True,text=True)
    raw.unlink(missing_ok=True)
    if encoded.returncode:
        output.unlink(missing_ok=True)
        raise RuntimeError(encoded.stderr)
    probe = subprocess.run([ffprobe,"-v","error","-select_streams","v:0","-show_entries","stream=codec_name,pix_fmt","-of","csv=p=0",str(output)],capture_output=True,text=True)
    if probe.returncode or not probe.stdout.strip().startswith("h264,") or "yuv420p" not in probe.stdout:
        output.unlink(missing_ok=True)
        raise RuntimeError("selector output failed H.264/yuv420p validation")

