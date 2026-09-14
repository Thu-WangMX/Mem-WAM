import pytest

from fastwam.trainer import (
    _is_checkpoint_step,
    _is_training_state_step,
    _normalize_save_steps,
)


def test_explicit_1k_plus_every_5k_selects_requested_25k_milestones():
    save_steps = _normalize_save_steps([1000])

    selected = {
        step
        for step in range(1, 25001)
        if _is_checkpoint_step(
            step=step,
            save_every=5000,
            save_steps=save_steps,
        )
    }

    assert selected == {1000, 5000, 10000, 15000, 20000, 25000}


@pytest.mark.parametrize("invalid", [[0], [-1], [1000, 0]])
def test_explicit_save_steps_must_be_positive(invalid):
    with pytest.raises(ValueError, match="save_steps.*positive"):
        _normalize_save_steps(invalid)


def test_portable_every_5k_but_full_training_state_every_10k_and_final():
    portable = {
        step
        for step in range(5000, 40001, 5000)
        if _is_checkpoint_step(step=step, save_every=5000, save_steps=frozenset())
    }
    full_state = {
        step
        for step in portable
        if _is_training_state_step(
            step=step,
            save_training_state_every=10000,
            max_steps=40000,
        )
    }

    assert portable == {5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000}
    assert full_state == {10000, 20000, 30000, 40000}


def test_final_step_always_gets_full_training_state():
    assert _is_training_state_step(
        step=25000,
        save_training_state_every=10000,
        max_steps=25000,
    )
