from __future__ import annotations

import pytest
import torch

from scripts.trace_putback_control_information import (
    trace_episode,
    validate_trace_access,
)


class _CopyLastFeature(torch.nn.Module):
    def forward(self, history, condition, *, lengths=None):
        return history[:, -1]


class _FirstFeatureControlProbe(torch.nn.Module):
    def forward(self, feature, proprio):
        value = feature[:, :1]
        return value[:, None, :].expand(-1, 2, -1)


def _models():
    return _CopyLastFeature().eval(), _FirstFeatureControlProbe().eval()


def _dynamics_rows():
    return [
        {
            "episode": 7,
            "target_frame": 16,
            "history": torch.tensor([[1.0, 2.0]]),
            "condition": torch.zeros(3),
            "target": torch.tensor([3.0, 2.0]),
        },
        {
            "episode": 7,
            "target_frame": 20,
            "history": torch.tensor([[2.0, 2.0]]),
            "condition": torch.zeros(3),
            "target": torch.tensor([2.0, 10.0]),
        },
    ]


def _probe_rows():
    return [
        {
            "episode": 7,
            "frame": 16,
            "feature": torch.tensor([3.0, 2.0]),
            "proprio": torch.zeros(1),
        },
        {
            "episode": 7,
            "frame": 20,
            "feature": torch.tensor([2.0, 10.0]),
            "proprio": torch.zeros(1),
        },
    ]


def test_trace_uses_same_frame_actual_and_counterfactual_wam_states():
    dynamics, control = _models()

    trace = trace_episode(
        episode=7,
        dynamics_examples=_dynamics_rows(),
        control_probe_examples=_probe_rows(),
        feature_predictor=dynamics,
        control_probe=control,
    )

    assert trace["episode"] == 7
    assert trace["frame_indices"].tolist() == [16, 20]
    torch.testing.assert_close(trace["information"], torch.tensor([2.0, 0.0]))
    assert trace["action_shift_by_dim"].shape == (2, 1)


def test_trace_rejects_feature_mismatch_between_artifacts():
    dynamics, control = _models()
    probe_rows = _probe_rows()
    probe_rows[0]["feature"] = torch.tensor([30.0, 2.0])

    with pytest.raises(ValueError, match="feature mismatch"):
        trace_episode(
            episode=7,
            dynamics_examples=_dynamics_rows(),
            control_probe_examples=probe_rows,
            feature_predictor=dynamics,
            control_probe=control,
        )


def test_heldout_trace_requires_final_runtime_lock(tmp_path):
    with pytest.raises(ValueError, match="runtime lock"):
        validate_trace_access(list(range(40, 50)), runtime_lock_path=None)

    lock = tmp_path / "runtime_lock.json"
    lock.write_text(
        '{"schema_version":"putback_locked_control_information_runtime_v1"}\n'
    )
    validated = validate_trace_access(
        list(range(40, 50)), runtime_lock_path=lock
    )
    assert validated == lock.resolve()


def test_candidate_trace_rejects_accidental_runtime_lock(tmp_path):
    lock = tmp_path / "runtime_lock.json"
    lock.write_text("{}")
    with pytest.raises(ValueError, match="must not use"):
        validate_trace_access(list(range(40)), runtime_lock_path=lock)
