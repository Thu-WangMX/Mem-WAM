from __future__ import annotations

import pytest

from fastwam.memory.four_phase_temporal import (
    PHASE_OFFSETS,
    interleave_phase_indices,
    phase_frame_indices,
)


def test_four_phases_preserve_native_interval_and_interleave_at_four_frames():
    phases = {phase: phase_frame_indices(49, phase) for phase in PHASE_OFFSETS}

    assert phases[0] == [0, 16, 32, 48]
    assert phases[4] == [4, 20, 36]
    assert phases[8] == [8, 24, 40]
    assert phases[12] == [12, 28, 44]
    assert all(
        later - earlier == 16
        for values in phases.values()
        for earlier, later in zip(values, values[1:])
    )
    assert interleave_phase_indices(phases) == [
        (0, 0),
        (4, 4),
        (8, 8),
        (12, 12),
        (0, 16),
        (4, 20),
        (8, 24),
        (12, 28),
        (0, 32),
        (4, 36),
        (8, 40),
        (12, 44),
        (0, 48),
    ]


@pytest.mark.parametrize(
    ("length", "offset", "step", "match"),
    [
        (0, 0, 16, "length"),
        (32, 3, 16, "offset"),
        (32, 0, 8, "step"),
    ],
)
def test_invalid_phase_contract_is_rejected(length, offset, step, match):
    with pytest.raises(ValueError, match=match):
        phase_frame_indices(length, offset, step=step)


def test_interleave_rejects_missing_or_nonuniform_phase_streams():
    with pytest.raises(ValueError, match="phases"):
        interleave_phase_indices({0: [0, 16], 4: [4, 20]})
    with pytest.raises(ValueError, match="four-frame"):
        interleave_phase_indices(
            {0: [0, 16], 4: [4, 20], 8: [8], 12: [12, 28]}
        )
