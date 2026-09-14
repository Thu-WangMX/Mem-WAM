from __future__ import annotations

import hashlib
import json

import pytest
import torch

from fastwam.evaluation.control_information_online import (
    ControlInformationOnlineRuntime,
    finalize_runtime_lock,
    verify_locked_runtime_dependencies,
)


class _CopyLastFeature(torch.nn.Module):
    def forward(self, history, condition, *, lengths=None):
        return history[:, -1]


class _FirstFeatureControlProbe(torch.nn.Module):
    def forward(self, feature, proprio):
        value = feature[:, :1]
        return value[:, None, :].expand(-1, 2, -1)


def _statistics():
    return {
        "schema_version": "putback_control_information_statistics_v1",
        "episodes": list(range(30)),
        "phases": [0, 4, 8, 12],
        "depth_cap": 4,
        "median": torch.zeros(4, 4),
        "mad_scale": torch.ones(4, 4),
    }


def _selector_config():
    return {
        "threshold": 2.0,
        "drift": 0.5,
        "decay": 1.0,
        "detector_stride": 4,
        "min_units": 4,
        "max_units": 24,
        "initial_group_start": 0,
    }


def _runtime():
    return ControlInformationOnlineRuntime(
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
        feature_window=8,
    )


def test_online_confirmation_is_applied_only_at_forward_planning_arrival():
    runtime = _runtime()
    executed_actions = []
    for frame in range(0, 24, 4):
        # Phase 4 sees an unpredictable, control-sensitive state at frame 20.
        feature = torch.tensor([3.0, 0.0]) if frame == 20 else torch.zeros(2)
        event, score = runtime.observe_projected(
            frame=frame,
            projected_feature=feature,
            raw_proprio=torch.zeros(1),
            executed_actions=executed_actions,
        )
        if frame == 20:
            assert event is not None
            assert event.confirmation_frame == 20
            assert score.information == 3.0
        executed_actions.extend([torch.zeros(1) for _ in range(4)])

    assert runtime.arrive_planning(frame=0) is None
    assert runtime.arrive_planning(frame=16) is None
    assert runtime.arrive_planning(frame=32) == (0, 2)
    assert runtime.aligner.retroactive_boundary_count == 0


def test_runtime_lock_extends_candidate_without_changing_selector(tmp_path):
    candidate = {
        "schema_version": "putback_locked_control_information_candidate_v1",
        "candidate": {"selector_config": _selector_config()},
        "hashes": {"control_probe": "a" * 64},
    }
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps(candidate, sort_keys=True) + "\n")
    sources = {}
    for name in (
        "online_source",
        "planning_aligner_source",
        "offline_freezer_source",
    ):
        path = tmp_path / f"{name}.py"
        path.write_text(name)
        sources[name] = path
    target = tmp_path / "locked_selector.json"

    payload = finalize_runtime_lock(
        target,
        candidate_lock_path=candidate_path,
        source_paths=sources,
    )

    assert payload["schema_version"] == "putback_locked_control_information_runtime_v1"
    assert payload["candidate"] == candidate
    assert payload["candidate_sha256"] == hashlib.sha256(
        candidate_path.read_bytes()
    ).hexdigest()
    assert set(payload["runtime_hashes"]) == set(sources)


def test_runtime_dependency_verifier_rejects_any_loaded_artifact_drift(tmp_path):
    artifact_names = {
        "initialization_manifest",
        "feature_bank_manifest",
        "pca",
        "feature_predictor",
        "control_probe",
        "contextual_statistics",
        "normalization",
    }
    artifacts = {}
    hashes = {"boundary_source": "1" * 64}
    for name in artifact_names:
        path = tmp_path / f"{name}.artifact"
        path.write_text(name)
        artifacts[name] = path
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    candidate = {
        "schema_version": "putback_locked_control_information_candidate_v1",
        "candidate": {"selector_config": _selector_config()},
        "hashes": hashes,
    }
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps(candidate, sort_keys=True) + "\n")
    sources = {}
    for name in (
        "online_source",
        "planning_aligner_source",
        "offline_freezer_source",
    ):
        path = tmp_path / f"{name}.py"
        path.write_text(name)
        sources[name] = path
    runtime_path = tmp_path / "runtime.json"
    finalize_runtime_lock(
        runtime_path,
        candidate_lock_path=candidate_path,
        source_paths=sources,
    )

    verified = verify_locked_runtime_dependencies(
        runtime_lock_path=runtime_path,
        candidate_lock_path=candidate_path,
        artifact_paths=artifacts,
        source_paths=sources,
    )
    assert verified["candidate"] == candidate

    artifacts["control_probe"].write_text("tampered")
    with pytest.raises(ValueError, match="control_probe sha256 mismatch"):
        verify_locked_runtime_dependencies(
            runtime_lock_path=runtime_path,
            candidate_lock_path=candidate_path,
            artifact_paths=artifacts,
            source_paths=sources,
        )
