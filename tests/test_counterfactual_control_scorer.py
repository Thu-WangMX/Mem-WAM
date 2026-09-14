from __future__ import annotations

import pytest
import torch

from fastwam.memory.control_information_probe import (
    score_counterfactual_control_information,
)


class _CopyLastFeature(torch.nn.Module):
    def forward(self, history, condition, *, lengths=None):
        assert lengths.tolist() == [history.shape[1]]
        return history[:, -1]


class _FirstFeatureControlProbe(torch.nn.Module):
    def forward(self, feature, proprio):
        assert proprio.shape == (len(feature), 1)
        value = feature[:, :1]
        return value[:, None, :].expand(-1, 2, -1)


def _models():
    return _CopyLastFeature().eval(), _FirstFeatureControlProbe().eval()


def _score(observed):
    dynamics, control = _models()
    return score_counterfactual_control_information(
        feature_predictor=dynamics,
        control_probe=control,
        history=torch.tensor([[0.0, 0.0], [1.0, 2.0]]),
        dynamics_condition=torch.zeros(3),
        observed_feature=torch.as_tensor(observed, dtype=torch.float32),
        normalized_proprio=torch.zeros(1),
    )


def test_information_is_zero_when_observed_feature_matches_prediction():
    result = _score([1.0, 2.0])

    assert result.information == 0.0
    torch.testing.assert_close(result.predicted_feature, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(result.prior_action, result.posterior_action)


def test_information_is_positive_only_for_control_sensitive_residual():
    sensitive = _score([3.0, 2.0])
    nullspace = _score([1.0, 20.0])

    torch.testing.assert_close(torch.tensor(sensitive.information), torch.tensor(2.0))
    assert nullspace.information == 0.0


def test_scorer_rejects_trainable_or_training_mode_models():
    dynamics, control = _models()
    control.register_parameter("unexpected", torch.nn.Parameter(torch.ones(1)))
    with pytest.raises(ValueError, match="frozen"):
        score_counterfactual_control_information(
            feature_predictor=dynamics,
            control_probe=control,
            history=torch.tensor([[1.0, 2.0]]),
            dynamics_condition=torch.zeros(3),
            observed_feature=torch.tensor([1.0, 2.0]),
            normalized_proprio=torch.zeros(1),
        )


def test_scorer_rejects_nonfinite_observation():
    with pytest.raises(ValueError, match="finite"):
        _score([float("nan"), 2.0])
