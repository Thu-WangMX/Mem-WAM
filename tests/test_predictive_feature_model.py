from __future__ import annotations

import torch

from fastwam.memory.predictive_feature_model import (
    FeaturePredictor,
    huber_prediction_loss,
    predict_phase_streams,
)


def test_visual_and_action_candidates_have_exact_architecture_parity():
    torch.manual_seed(42)
    visual = FeaturePredictor(feature_dim=20, condition_dim=7, hidden_dim=16)
    torch.manual_seed(42)
    action = FeaturePredictor(feature_dim=20, condition_dim=7, hidden_dim=16)

    assert sum(p.numel() for p in visual.parameters()) == sum(
        p.numel() for p in action.parameters()
    )
    for left, right in zip(visual.state_dict().values(), action.state_dict().values()):
        torch.testing.assert_close(left, right)

    history = torch.randn(3, 4, 20)
    condition = torch.randn(3, 7)
    visual_prediction = visual(history, torch.zeros_like(condition))
    action_prediction = action(history, condition)
    assert visual_prediction.shape == action_prediction.shape == (3, 20)
    assert torch.isfinite(visual_prediction).all()
    assert torch.isfinite(action_prediction).all()


def test_phase_stream_helper_resets_recurrent_state_between_phases():
    torch.manual_seed(7)
    model = FeaturePredictor(feature_dim=6, condition_dim=3, hidden_dim=8)
    histories = {
        0: torch.randn(1, 3, 6),
        4: torch.randn(1, 2, 6),
    }
    conditions = {0: torch.randn(1, 3), 4: torch.randn(1, 3)}

    together = predict_phase_streams(model, histories, conditions)
    separately = {
        phase: model(histories[phase], conditions[phase]) for phase in histories
    }

    assert set(together) == {0, 4}
    for phase in together:
        torch.testing.assert_close(together[phase], separately[phase])


def test_prediction_loss_is_huber_delta_one():
    prediction = torch.tensor([[0.0, 3.0]])
    target = torch.tensor([[0.5, 0.0]])
    # 0.5 * 0.5^2 + (3.0 - 0.5), averaged over two coordinates.
    expected = torch.tensor((0.125 + 2.5) / 2)
    torch.testing.assert_close(huber_prediction_loss(prediction, target), expected)

