from __future__ import annotations

import pytest
import torch

from fastwam.memory.control_information_dataset import (
    build_control_probe_examples,
    normalize_control_tensor,
)


def _four_phase_features(*, episode_length: int, feature_dim: int = 6):
    streams = {}
    for phase in (0, 4, 8, 12):
        frames = torch.arange(phase, episode_length, 16, dtype=torch.int64)
        streams[str(phase)] = {
            "frame_indices": frames,
            "features": torch.stack(
                [torch.full((feature_dim,), float(frame)) for frame in frames]
            ),
            "warmup": torch.tensor(
                [index == 0 for index in range(len(frames))], dtype=torch.bool
            ),
        }
    return streams


def _build(*, episode_length: int = 40):
    actions = torch.arange(episode_length * 14, dtype=torch.float32).reshape(
        episode_length, 14
    )
    proprio = torch.arange(episode_length * 14, dtype=torch.float32).reshape(
        episode_length, 14
    )
    return build_control_probe_examples(
        episode=3,
        projected_phases=_four_phase_features(episode_length=episode_length),
        actions=actions,
        proprio=proprio,
        action_mean=torch.zeros(14),
        action_std=torch.ones(14),
        proprio_mean=torch.zeros(14),
        proprio_std=torch.ones(14),
        horizon=16,
    )


def test_control_probe_example_uses_current_state_and_future_action_labels():
    rows = _build()
    row = next(value for value in rows if value["frame"] == 16)

    assert set(row) == {
        "episode",
        "phase",
        "frame",
        "feature",
        "proprio",
        "target_actions",
        "target_mask",
    }
    assert row["episode"] == 3
    assert row["phase"] == 0
    assert row["feature"].tolist() == [16.0] * 6
    assert row["target_actions"].shape == (16, 14)
    assert row["target_mask"].dtype == torch.bool
    assert bool(row["target_mask"].all())
    torch.testing.assert_close(
        row["target_actions"],
        torch.arange(40 * 14, dtype=torch.float32).reshape(40, 14)[16:32],
    )


def test_terminal_action_targets_are_zero_padded_and_masked():
    row = next(value for value in _build(episode_length=35) if value["frame"] == 28)

    assert int(row["target_mask"].sum()) == 7
    assert row["target_mask"].tolist() == [True] * 7 + [False] * 9
    assert torch.count_nonzero(row["target_actions"][7:]).item() == 0


def test_normalization_is_applied_to_proprio_and_action_targets():
    episode_length = 24
    actions = torch.full((episode_length, 14), 6.0)
    proprio = torch.full((episode_length, 14), 9.0)
    rows = build_control_probe_examples(
        episode=0,
        projected_phases=_four_phase_features(episode_length=episode_length),
        actions=actions,
        proprio=proprio,
        action_mean=torch.full((14,), 2.0),
        action_std=torch.full((14,), 2.0),
        proprio_mean=torch.full((14,), 3.0),
        proprio_std=torch.full((14,), 3.0),
        horizon=16,
    )

    assert torch.unique(rows[0]["proprio"]).tolist() == [2.0]
    assert torch.unique(rows[0]["target_actions"][rows[0]["target_mask"]]).tolist() == [2.0]


def test_builder_rejects_nonpositive_normalization_scale():
    with pytest.raises(ValueError, match="action_std must be finite and positive"):
        normalize_control_tensor(
            torch.zeros(2, 14),
            mean=torch.zeros(14),
            std=torch.zeros(14),
            name="actions",
        )


def test_builder_requires_all_four_strict_phase_streams():
    streams = _four_phase_features(episode_length=32)
    del streams["12"]
    with pytest.raises(ValueError, match="four phase streams"):
        build_control_probe_examples(
            episode=0,
            projected_phases=streams,
            actions=torch.zeros(32, 14),
            proprio=torch.zeros(32, 14),
            action_mean=torch.zeros(14),
            action_std=torch.ones(14),
            proprio_mean=torch.zeros(14),
            proprio_std=torch.ones(14),
        )
