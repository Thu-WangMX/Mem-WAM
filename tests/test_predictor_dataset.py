from __future__ import annotations

import torch

from fastwam.memory.predictor_dataset import build_predictor_examples


def _projected_phases(feature_dim: int = 6):
    return {
        str(phase): {
            "frame_indices": torch.tensor([phase + 16 * step for step in range(4)]),
            "features": torch.stack(
                [torch.full((feature_dim,), phase + step, dtype=torch.float32) for step in range(4)]
            ),
            "warmup": torch.tensor([True, False, False, False]),
        }
        for phase in (0, 4, 8, 12)
    }


def test_predictor_examples_are_same_phase_and_strictly_causal():
    actions = torch.arange(80 * 14, dtype=torch.float32).reshape(80, 14)
    proprio = actions + 10000

    examples = build_predictor_examples(
        episode=3,
        projected_phases=_projected_phases(),
        actions=actions,
        proprio=proprio,
        mode="visual_action",
        max_history=2,
    )

    assert len(examples) == 12
    example = next(row for row in examples if row["target_frame"] == 36)
    assert example["episode"] == 3
    assert example["phase"] == 4
    assert example["history_frames"] == [4, 20]
    assert max(example["history_frames"]) < example["target_frame"]
    assert example["condition_action_frames"] == list(range(20, 36))
    assert example["proprio_frame"] == 36
    assert example["history"].shape == (2, 6)
    assert example["target"].shape == (6,)
    expected_condition = torch.cat([actions[20:36].reshape(-1), proprio[36]])
    torch.testing.assert_close(example["condition"], expected_condition)


def test_visual_only_and_action_examples_have_identical_targets_and_condition_size():
    actions = torch.randn((80, 14), generator=torch.Generator().manual_seed(4))
    proprio = torch.randn((80, 14), generator=torch.Generator().manual_seed(5))
    kwargs = {
        "episode": 2,
        "projected_phases": _projected_phases(),
        "actions": actions,
        "proprio": proprio,
        "max_history": 8,
    }

    visual = build_predictor_examples(mode="visual_only", **kwargs)
    action = build_predictor_examples(mode="visual_action", **kwargs)

    assert len(visual) == len(action)
    for visual_row, action_row in zip(visual, action):
        assert visual_row["target_frame"] == action_row["target_frame"]
        torch.testing.assert_close(visual_row["history"], action_row["history"])
        torch.testing.assert_close(visual_row["target"], action_row["target"])
        assert visual_row["condition"].shape == action_row["condition"].shape == (238,)
        assert torch.count_nonzero(visual_row["condition"]) == 0


def test_predictor_examples_never_cross_episode_or_use_future_actions():
    phases = _projected_phases()
    actions = torch.zeros((80, 14))
    proprio = torch.zeros((80, 14))
    examples = build_predictor_examples(
        episode=9,
        projected_phases=phases,
        actions=actions,
        proprio=proprio,
        mode="visual_action",
    )

    assert {row["episode"] for row in examples} == {9}
    assert all(
        max(row["condition_action_frames"]) < row["target_frame"]
        and row["phase"] == row["target_frame"] % 16
        for row in examples
    )
