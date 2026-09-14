from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.robotwin import eval_robotwin_single
from experiments.robotwin.fastwam_policy import deploy_policy
from experiments.robotwin.fastwam_policy.deploy_policy import (
    _control_information_diagnostic_metrics,
    _fresh_control_information_runtime,
    _resolve_control_information_paths,
    _validate_online_selector_modes,
)
from fastwam.evaluation.control_information_online import (
    ControlInformationOnlineRuntime,
)
from fastwam.evaluation.control_information_online_v2 import (
    ControlInformationOnlineRuntimeV2,
)
from fastwam.evaluation.control_information_online_v3 import (
    ControlInformationOnlineRuntimeV3,
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
        selector_config={
            "threshold": 2.0,
            "drift": 0.5,
            "decay": 1.0,
            "detector_stride": 4,
            "min_units": 4,
            "max_units": 24,
            "initial_group_start": 0,
        },
        feature_window=8,
    )


def _runtime_v2():
    return ControlInformationOnlineRuntimeV2(
        feature_predictor=_CopyLastFeature().eval(),
        control_probe=_FirstFeatureControlProbe().eval(),
        normalization={
            "action_mean": torch.zeros(1),
            "action_std": torch.ones(1),
            "proprio_mean": torch.zeros(1),
            "proprio_std": torch.ones(1),
        },
        contextual_statistics=_statistics(),
        selector_config={
            "threshold": 2.0,
            "drift": 0.5,
            "decay": 1.0,
            "detector_stride": 4,
            "min_units": 8,
            "max_units": 24,
            "initial_group_start": 0,
            "calibration_samples": 4,
            "adaptation_epsilon": 1e-6,
            "max_abs_log_bias": 4.0,
        },
        feature_window=8,
    )


def _runtime_v3():
    config = {
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
    return ControlInformationOnlineRuntimeV3(
        feature_predictor=_CopyLastFeature().eval(),
        control_probe=_FirstFeatureControlProbe().eval(),
        normalization={
            "action_mean": torch.zeros(1),
            "action_std": torch.ones(1),
            "proprio_mean": torch.zeros(1),
            "proprio_std": torch.ones(1),
        },
        contextual_statistics={**_statistics(), "median": torch.ones(4, 4)},
        selector_config=config,
        feature_window=8,
    )


def test_new_selector_is_mutually_exclusive_with_old_selectors():
    with pytest.raises(ValueError, match="exactly one online selector"):
        _validate_online_selector_modes(
            dynamic_surprise_online=False,
            wrist_event_online=False,
            embodied_information_online=True,
            control_information_online=True,
        )


def test_missing_locked_artifact_aborts_without_fixed_group_fallback(tmp_path):
    present = tmp_path / "present.pt"
    present.write_text("present")
    with pytest.raises(FileNotFoundError, match="control-information lock"):
        _resolve_control_information_paths(
            runtime_lock="/missing/lock.json",
            pca=present,
            feature_predictor=present,
            control_probe=present,
            statistics=present,
            initialization_action_dit=present,
            normalization=present,
            project_root=tmp_path,
        )


def test_episode_reset_clears_selector_feature_and_cusum_state():
    previous = _runtime()
    previous.observe_projected(
        frame=0,
        projected_feature=torch.zeros(2),
        raw_proprio=torch.zeros(1),
        executed_actions=[],
    )
    assert previous.boundary_state.last_frame == 0

    fresh = _fresh_control_information_runtime(previous)

    assert fresh.boundary_state.last_frame is None
    assert all(not values for values in fresh._features.values())
    assert fresh.aligner.last_decision == -1


def test_episode_reset_preserves_v2_runtime_and_clears_episode_adapter():
    previous = _runtime_v2()
    for frame in range(0, 32, 4):
        previous.observe_projected(
            frame=frame,
            projected_feature=torch.zeros(2),
            raw_proprio=torch.zeros(1),
            executed_actions=[torch.zeros(1) for _ in range(max(frame, 16))],
        )
    assert previous.boundary_state.adaptation_factor is not None

    fresh = _fresh_control_information_runtime(previous)

    assert isinstance(fresh, ControlInformationOnlineRuntimeV2)
    assert fresh.boundary_state.adaptation_factor is None
    assert fresh.boundary_state.last_frame is None
    assert all(not values for values in fresh._features.values())
    assert fresh.aligner.last_decision == -1


def test_episode_reset_preserves_v3_runtime_and_clears_segment_baseline():
    previous = _runtime_v3()
    fresh = _fresh_control_information_runtime(previous)

    assert isinstance(fresh, ControlInformationOnlineRuntimeV3)
    assert fresh.boundary_state.episode_adaptation_factor is None
    assert not fresh.boundary_state.segment_baseline.ready
    assert fresh.boundary_state.last_frame is None
    assert all(not values for values in fresh._features.values())


def test_v2_path_resolution_selects_only_v2_locked_sources(tmp_path):
    analysis = tmp_path / "analysis"
    selector = analysis / "selector_v2"
    runtime = selector / "selector_runtime" / "locked_selector.json"
    candidate = selector / "selector_candidate" / "locked_candidate.json"
    runtime.parent.mkdir(parents=True)
    candidate.parent.mkdir(parents=True)
    runtime.write_text(
        '{"schema_version":"putback_locked_control_information_runtime_v2"}\n'
    )
    candidate.write_text(
        '{"schema_version":"putback_locked_control_information_candidate_v2"}\n'
    )
    dependencies = {}
    for name in (
        "pca",
        "feature_predictor",
        "control_probe",
        "statistics",
        "initialization_action_dit",
        "normalization",
    ):
        path = analysis / f"{name}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
        dependencies[name] = path
    bank = analysis / "putback_four_phase_multilayer_wam_features_v1"
    bank.mkdir()
    (bank / "initialization_manifest.json").write_text("{}\n")
    (bank / "bank_manifest.json").write_text("{}\n")

    resolved = _resolve_control_information_paths(
        runtime_lock=runtime,
        pca=dependencies["pca"],
        feature_predictor=dependencies["feature_predictor"],
        control_probe=dependencies["control_probe"],
        statistics=dependencies["statistics"],
        initialization_action_dit=dependencies["initialization_action_dit"],
        normalization=dependencies["normalization"],
        project_root=tmp_path,
    )

    assert resolved["runtime_version"] == "v2"
    assert set(resolved["artifact_paths"]) == {
        "initialization_manifest",
        "feature_bank_manifest",
        "pca",
        "feature_predictor",
        "control_probe",
        "contextual_statistics",
        "normalization",
        "base_score_source",
        "base_boundary_source",
        "episode_adapter_source",
    }
    assert set(resolved["source_paths"]) == {
        "base_online_source",
        "online_source",
        "planning_aligner_source",
        "offline_freezer_source",
    }


def test_v3_path_resolution_includes_segment_boundary_source(tmp_path):
    analysis = tmp_path / "analysis"
    selector = analysis / "selector_v3"
    runtime = selector / "selector_runtime" / "locked_selector.json"
    candidate = selector / "selector_candidate" / "locked_candidate.json"
    runtime.parent.mkdir(parents=True)
    candidate.parent.mkdir(parents=True)
    runtime.write_text(
        '{"schema_version":"putback_locked_control_information_runtime_v3"}\n'
    )
    candidate.write_text(
        '{"schema_version":"putback_locked_control_information_candidate_v3"}\n'
    )
    dependencies = {}
    for name in (
        "pca",
        "feature_predictor",
        "control_probe",
        "statistics",
        "initialization_action_dit",
        "normalization",
    ):
        path = analysis / f"{name}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
        dependencies[name] = path
    bank = analysis / "putback_four_phase_multilayer_wam_features_v1"
    bank.mkdir()
    (bank / "initialization_manifest.json").write_text("{}\n")
    (bank / "bank_manifest.json").write_text("{}\n")

    resolved = _resolve_control_information_paths(
        runtime_lock=runtime,
        pca=dependencies["pca"],
        feature_predictor=dependencies["feature_predictor"],
        control_probe=dependencies["control_probe"],
        statistics=dependencies["statistics"],
        initialization_action_dit=dependencies["initialization_action_dit"],
        normalization=dependencies["normalization"],
        project_root=tmp_path,
    )

    assert resolved["runtime_version"] == "v3"
    assert "segment_boundary_source" in resolved["artifact_paths"]
    assert "episode_adapter_source" in resolved["artifact_paths"]
    assert set(resolved["source_paths"]) == {
        "base_online_source",
        "online_source",
        "planning_aligner_source",
        "offline_freezer_source",
    }


def test_robotwin_forwards_only_explicit_control_information_arguments():
    required = {
        "control_information_online",
        "control_information_lock",
        "control_information_pca",
        "control_information_feature_predictor",
        "control_information_control_probe",
        "control_information_statistics",
        "control_information_init_action_dit_path",
    }
    assert required <= set(eval_robotwin_single.MEMORY_POLICY_FORWARD_KEYS)


def test_control_information_diagnostic_metrics_expose_rollout_ood_sources():
    class _Score:
        information = 1.0
        predicted_feature = torch.tensor([1.0, 1.0])
        prior_action = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        posterior_action = torch.tensor([[2.0, 2.0], [2.0, 2.0]])

    metrics = _control_information_diagnostic_metrics(
        score=_Score(),
        observed_feature=torch.tensor([2.0, 2.0]),
        raw_proprio=torch.tensor([3.0, 4.0]),
        executed_actions=[torch.tensor([0.0, 0.0]), torch.tensor([3.0, 4.0])],
    )

    assert metrics == pytest.approx(
        {
            "observed_feature_rms": 2.0,
            "observed_feature_abs_max": 2.0,
            "predicted_feature_rms": 1.0,
            "feature_prediction_residual_rms": 1.0,
            "prior_action_rms": 1.0,
            "posterior_action_rms": 2.0,
            "relative_control_information": 0.5,
            "raw_proprio_rms": 5.0 / (2.0**0.5),
            "executed_action_rms": 2.5,
            "executed_action_delta_rms": 5.0 / (2.0**0.5),
        }
    )


def test_control_information_event_payload_accepts_v3_contextual_score():
    assert hasattr(deploy_policy, "_control_information_event_payload")
    event = SimpleNamespace(
        confirmation_frame=36,
        group_start=0,
        group_end=8,
        reason="learned",
        information=0.9,
        adapted_information=0.3,
        episode_adaptation_factor=3.0,
        contextual_standardized_information=1.25,
        segment_relative_innovation=4.5,
        segment_baseline_location=0.2,
        segment_baseline_scale=0.5,
        consecutive_evidence=2,
        cusum=3.5,
    )

    payload = deploy_policy._control_information_event_payload(event)

    assert payload["standardized_information"] == pytest.approx(1.25)
    assert payload["contextual_standardized_information"] == pytest.approx(1.25)
    assert payload["adaptation_factor"] == pytest.approx(3.0)
