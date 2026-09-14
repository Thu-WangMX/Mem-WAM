from __future__ import annotations

import json

import pytest
import torch

from fastwam.evaluation.control_information_online_v2 import (
    RUNTIME_SOURCE_KEYS_V2,
    ControlInformationOnlineRuntimeV2,
    finalize_runtime_lock_v2,
    verify_locked_runtime_dependencies_v2,
)
from fastwam.memory.control_information_boundary_v2 import (
    ControlInformationBoundaryStateV2,
)
from scripts.calibrate_putback_control_information_selector_v2 import (
    REQUIRED_DEPENDENCY_KEYS_V2,
    lock_candidate_v2,
)


class _CopyLastFeature(torch.nn.Module):
    def forward(self, history, condition, *, lengths=None):
        return history[:, -1]


class _FirstFeatureControlProbe(torch.nn.Module):
    def forward(self, feature, proprio):
        return feature[:, :1, None].expand(-1, 1, 1)


def _statistics() -> dict:
    return {
        "schema_version": "putback_control_information_statistics_v1",
        "episodes": list(range(30)),
        "phases": [0, 4, 8, 12],
        "depth_cap": 4,
        "median": torch.ones(4, 4),
        "mad_scale": torch.ones(4, 4),
    }


def _selector_config() -> dict:
    return {
        "threshold": 2.0,
        "drift": 0.0,
        "decay": 1.0,
        "detector_stride": 4,
        "min_units": 8,
        "max_units": 24,
        "initial_group_start": 0,
        "calibration_samples": 4,
        "adaptation_epsilon": 1e-6,
        "max_abs_log_bias": 4.0,
    }


def test_v2_runtime_keeps_v1_score_contract_but_uses_causal_episode_state():
    runtime = ControlInformationOnlineRuntimeV2(
        feature_predictor=_CopyLastFeature().eval(),
        control_probe=_FirstFeatureControlProbe().eval(),
        normalization={
            "action_mean": torch.zeros(1),
            "action_std": torch.ones(1),
            "proprio_mean": torch.zeros(1),
            "proprio_std": torch.ones(1),
        },
        contextual_statistics=_statistics(),
        selector_config=_selector_config(),
    )
    actions = [torch.zeros(1) for _ in range(64)]
    for frame in range(0, 32, 4):
        value = 0.0 if frame < 16 else 1.0
        event, _ = runtime.observe_projected(
            frame=frame,
            projected_feature=torch.tensor([value, 0.0]),
            raw_proprio=torch.zeros(1),
            executed_actions=actions[: max(frame, 16)],
        )
        assert event is None

    assert isinstance(runtime.boundary_state, ControlInformationBoundaryStateV2)
    assert runtime.boundary_state.adaptation_factor == pytest.approx(1.0)
    assert runtime.boundary_state.adapter.calibration_frames == (16, 20, 24, 28)


def test_v2_runtime_lock_verifies_candidate_artifacts_and_sources(tmp_path):
    assert "base_online_source" in RUNTIME_SOURCE_KEYS_V2
    artifacts = {}
    for index, key in enumerate(sorted(REQUIRED_DEPENDENCY_KEYS_V2)):
        path = tmp_path / f"artifact_{key}.bin"
        path.write_bytes(f"artifact-{index}".encode())
        artifacts[key] = path
    candidate_path = tmp_path / "candidate.json"
    lock_candidate_v2(
        candidate_path,
        candidate={"selector_config": _selector_config(), "metrics": {}},
        dependency_paths=artifacts,
    )
    sources = {}
    for index, key in enumerate(sorted(RUNTIME_SOURCE_KEYS_V2)):
        path = tmp_path / f"source_{key}.py"
        path.write_text(f"source-{index}")
        sources[key] = path
    runtime_path = tmp_path / "runtime.json"

    payload = finalize_runtime_lock_v2(
        runtime_path,
        candidate_lock_path=candidate_path,
        source_paths=sources,
    )

    assert payload["schema_version"] == "putback_locked_control_information_runtime_v2"
    assert json.loads(runtime_path.read_text()) == payload
    assert verify_locked_runtime_dependencies_v2(
        runtime_lock_path=runtime_path,
        candidate_lock_path=candidate_path,
        artifact_paths=artifacts,
        source_paths=sources,
    ) == payload
    sources["online_source"].write_text("tampered")
    with pytest.raises(ValueError, match="locked online_source sha256 mismatch"):
        verify_locked_runtime_dependencies_v2(
            runtime_lock_path=runtime_path,
            candidate_lock_path=candidate_path,
            artifact_paths=artifacts,
            source_paths=sources,
        )
