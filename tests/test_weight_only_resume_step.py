from pathlib import Path

import pytest

from fastwam.trainer import (
    _is_checkpoint_step,
    _validate_weight_only_resume_step,
)


def test_weight_only_resume_step_matches_portable_filename(tmp_path: Path):
    checkpoint = tmp_path / "step_030000.pt"
    checkpoint.touch()

    assert _validate_weight_only_resume_step(checkpoint, 30000) == 30000


def test_weight_only_resume_step_rejects_mismatch(tmp_path: Path):
    checkpoint = tmp_path / "step_030000.pt"
    checkpoint.touch()

    with pytest.raises(ValueError, match="step mismatch"):
        _validate_weight_only_resume_step(checkpoint, 20000)


def test_resume_checkpoint_schedule_only_saves_requested_steps():
    requested = {40000, 45000, 50000}
    saved = {
        step
        for step in range(30001, 50001)
        if _is_checkpoint_step(step=step, save_every=0, save_steps=requested)
    }

    assert saved == requested
