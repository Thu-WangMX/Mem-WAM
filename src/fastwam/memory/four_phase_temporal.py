"""Temporal indexing contract for four interleaved native-rate WAM streams."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


PHASE_OFFSETS = (0, 4, 8, 12)
VIDEO_EXPERT_FRAME_STRIDE = 16
DETECTOR_FRAME_STRIDE = 4


def phase_frame_indices(
    length: int,
    offset: int,
    *,
    step: int = VIDEO_EXPERT_FRAME_STRIDE,
) -> list[int]:
    """Return one phase's simulator-frame indices at the native WAM interval."""

    length = int(length)
    offset = int(offset)
    step = int(step)
    if length <= 0:
        raise ValueError("length must be positive")
    if offset not in PHASE_OFFSETS:
        raise ValueError(f"offset must be one of {PHASE_OFFSETS}, got {offset}")
    if step != VIDEO_EXPERT_FRAME_STRIDE:
        raise ValueError(
            f"step must preserve the {VIDEO_EXPERT_FRAME_STRIDE}-frame "
            f"Video Expert interval, got {step}"
        )
    return list(range(offset, length, step))


def interleave_phase_indices(
    by_phase: Mapping[int, Sequence[int]],
) -> list[tuple[int, int]]:
    """Interleave complete phase streams into one four-frame detector timeline."""

    if set(int(value) for value in by_phase) != set(PHASE_OFFSETS):
        raise ValueError(f"phases must be exactly {PHASE_OFFSETS}")
    rows: list[tuple[int, int]] = []
    for raw_phase, raw_frames in by_phase.items():
        phase = int(raw_phase)
        frames = [int(value) for value in raw_frames]
        if any(frame % VIDEO_EXPERT_FRAME_STRIDE != phase for frame in frames):
            raise ValueError(f"phase {phase} contains an incompatible frame")
        if any(
            later - earlier != VIDEO_EXPERT_FRAME_STRIDE
            for earlier, later in zip(frames, frames[1:])
        ):
            raise ValueError(f"phase {phase} does not preserve native interval")
        rows.extend((phase, frame) for frame in frames)
    rows.sort(key=lambda row: row[1])
    if any(
        later[1] - earlier[1] != DETECTOR_FRAME_STRIDE
        for earlier, later in zip(rows, rows[1:])
    ):
        raise ValueError("phase streams do not form a contiguous four-frame timeline")
    return rows
