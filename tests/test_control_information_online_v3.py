from __future__ import annotations

import json

import pytest
import torch

from fastwam.evaluation.control_information_online_v3 import (
    LOCKED_ARTIFACT_KEYS_V3,
    RUNTIME_SOURCE_KEYS_V3,
    ControlInformationOnlineRuntimeV3,
    finalize_runtime_lock_v3,
    verify_locked_runtime_dependencies_v3,
)
from fastwam.memory.control_information_boundary_v3 import (
    ControlInformationBoundaryStateV3,
)
from scripts.calibrate_putback_control_information_selector_v3 import (
    REQUIRED_DEPENDENCY_KEYS_V3,
    lock_candidate_v3,
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
        "drift": 1.0,
        "decay": 0.8,
        "detector_stride": 4,
        "min_units": 8,
        "max_units": 24,
        "initial_group_start": 0,
        "episode_calibration_samples": 4,
        "segment_calibration_samples": 4,
        "adaptation_epsilon": 1e-6,
        "max_abs_log_bias": 4.0,
        "segment_scale_floor": 0.5,
        "minimum_consecutive_evidence": 2,
    }


def test_v3_online_runtime_keeps_score_path_and_uses_segment_state():
    runtime = ControlInformationOnlineRuntimeV3(
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
        event, _ = runtime.observe_projected(
            frame=frame,
            projected_feature=torch.tensor(
                [0.0 if frame < 16 else 1.0, 0.0]
            ),
            raw_proprio=torch.zeros(1),
            executed_actions=actions[: max(frame, 16)],
        )
        assert event is None

    assert isinstance(runtime.boundary_state, ControlInformationBoundaryStateV3)
    assert runtime.boundary_state.episode_adaptation_factor is not None
    assert runtime.boundary_state.segment_baseline.ready


def test_v3_runtime_lock_verifies_every_reused_source(tmp_path):
    assert LOCKED_ARTIFACT_KEYS_V3 == REQUIRED_DEPENDENCY_KEYS_V3
    artifacts = {}
    for index, key in enumerate(sorted(LOCKED_ARTIFACT_KEYS_V3)):
        path = tmp_path / f"artifact_{key}.bin"
        path.write_bytes(f"artifact-{index}".encode())
        artifacts[key] = path
    candidate_path = tmp_path / "candidate.json"
    lock_candidate_v3(
        candidate_path,
        candidate={"selector_config": _selector_config(), "metrics": {}},
        dependency_paths=artifacts,
    )
    sources = {}
    for index, key in enumerate(sorted(RUNTIME_SOURCE_KEYS_V3)):
        path = tmp_path / f"source_{key}.py"
        path.write_text(f"source-{index}")
        sources[key] = path
    runtime_path = tmp_path / "runtime.json"

    payload = finalize_runtime_lock_v3(
        runtime_path,
        candidate_lock_path=candidate_path,
        source_paths=sources,
    )

    assert payload["schema_version"] == "putback_locked_control_information_runtime_v3"
    assert json.loads(runtime_path.read_text()) == payload
    assert verify_locked_runtime_dependencies_v3(
        runtime_lock_path=runtime_path,
        candidate_lock_path=candidate_path,
        artifact_paths=artifacts,
        source_paths=sources,
    ) == payload
    sources["base_online_source"].write_text("tampered")
    with pytest.raises(ValueError, match="locked base_online_source sha256 mismatch"):
        verify_locked_runtime_dependencies_v3(
            runtime_lock_path=runtime_path,
            candidate_lock_path=candidate_path,
            artifact_paths=artifacts,
            source_paths=sources,
        )
