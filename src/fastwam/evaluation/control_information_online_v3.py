"""Strict-online segment-relative frozen-WAM control information."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch

from fastwam.evaluation.control_information_online import (
    ControlInformationOnlineRuntime,
    sha256_file,
)
from fastwam.evaluation.embodied_information_online import load_predictor
from fastwam.memory.control_information_dataset import normalization_from_policy_stats
from fastwam.memory.control_information_probe import load_compact_control_probe
from fastwam.memory.control_information_boundary_v3 import (
    ControlInformationBoundaryStateV3,
)
from fastwam.memory.multilayer_spatial_feature import capture_spatial_features


CANDIDATE_LOCK_SCHEMA_V3 = "putback_locked_control_information_candidate_v3"
RUNTIME_LOCK_SCHEMA_V3 = "putback_locked_control_information_runtime_v3"
RUNTIME_SOURCE_KEYS_V3 = {
    "base_online_source",
    "online_source",
    "planning_aligner_source",
    "offline_freezer_source",
}
LOCKED_ARTIFACT_KEYS_V3 = {
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
    "segment_boundary_source",
}
BASE_SELECTOR_KEYS = {
    "threshold",
    "drift",
    "decay",
    "detector_stride",
    "min_units",
    "max_units",
    "initial_group_start",
}


def finalize_runtime_lock_v3(
    path: str | Path,
    *,
    candidate_lock_path: str | Path,
    source_paths: Mapping[str, str | Path],
) -> dict[str, Any]:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite v3 runtime lock: {path}")
    if set(source_paths) != RUNTIME_SOURCE_KEYS_V3:
        raise ValueError(
            f"v3 runtime sources must be exactly {sorted(RUNTIME_SOURCE_KEYS_V3)}"
        )
    candidate_path = Path(candidate_lock_path)
    candidate = json.loads(candidate_path.read_text())
    if candidate.get("schema_version") != CANDIDATE_LOCK_SCHEMA_V3:
        raise ValueError("v3 control-information candidate lock schema is incompatible")
    payload = {
        "schema_version": RUNTIME_LOCK_SCHEMA_V3,
        "method": "segment_relative_counterfactual_control_information",
        "candidate_sha256": sha256_file(candidate_path),
        "candidate": candidate,
        "runtime_hashes": {
            key: sha256_file(source_paths[key])
            for key in sorted(RUNTIME_SOURCE_KEYS_V3)
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return payload


def verify_locked_runtime_dependencies_v3(
    *,
    runtime_lock_path: str | Path,
    candidate_lock_path: str | Path,
    artifact_paths: Mapping[str, str | Path],
    source_paths: Mapping[str, str | Path],
) -> dict[str, Any]:
    runtime_path = Path(runtime_lock_path).resolve()
    candidate_path = Path(candidate_lock_path).resolve()
    runtime = json.loads(runtime_path.read_text())
    candidate = json.loads(candidate_path.read_text())
    if runtime.get("schema_version") != RUNTIME_LOCK_SCHEMA_V3:
        raise ValueError("v3 control-information runtime lock schema is incompatible")
    if candidate.get("schema_version") != CANDIDATE_LOCK_SCHEMA_V3:
        raise ValueError("v3 control-information candidate lock schema is incompatible")
    if runtime.get("candidate_sha256") != sha256_file(candidate_path):
        raise ValueError("v3 candidate lock sha256 mismatch")
    if runtime.get("candidate") != candidate:
        raise ValueError("v3 runtime lock does not embed the candidate verbatim")
    if set(artifact_paths) != LOCKED_ARTIFACT_KEYS_V3:
        raise ValueError(
            f"v3 locked artifacts must be exactly {sorted(LOCKED_ARTIFACT_KEYS_V3)}"
        )
    if set(source_paths) != RUNTIME_SOURCE_KEYS_V3:
        raise ValueError(
            f"v3 runtime sources must be exactly {sorted(RUNTIME_SOURCE_KEYS_V3)}"
        )
    candidate_hashes = candidate.get("hashes", {})
    for name in sorted(LOCKED_ARTIFACT_KEYS_V3):
        actual = sha256_file(artifact_paths[name])
        expected = candidate_hashes.get(name)
        if actual != expected:
            raise ValueError(f"locked {name} sha256 mismatch: {actual} != {expected}")
    runtime_hashes = runtime.get("runtime_hashes", {})
    for name in sorted(RUNTIME_SOURCE_KEYS_V3):
        actual = sha256_file(source_paths[name])
        expected = runtime_hashes.get(name)
        if actual != expected:
            raise ValueError(f"locked {name} sha256 mismatch: {actual} != {expected}")
    return runtime


def load_locked_control_information_artifacts_v3(
    *,
    runtime_lock_path: str | Path,
    candidate_lock_path: str | Path,
    artifact_paths: Mapping[str, str | Path],
    source_paths: Mapping[str, str | Path],
    device: torch.device,
) -> dict[str, Any]:
    runtime = verify_locked_runtime_dependencies_v3(
        runtime_lock_path=runtime_lock_path,
        candidate_lock_path=candidate_lock_path,
        artifact_paths=artifact_paths,
        source_paths=source_paths,
    )
    paths = {name: Path(path).resolve() for name, path in artifact_paths.items()}
    pca = torch.load(paths["pca"], map_location="cpu", weights_only=True)
    statistics = torch.load(
        paths["contextual_statistics"], map_location="cpu", weights_only=True
    )
    feature_predictor = load_predictor(paths["feature_predictor"], device=device)
    control_probe, control_probe_metadata = load_compact_control_probe(
        paths["control_probe"], device=device
    )
    normalization = normalization_from_policy_stats(
        json.loads(paths["normalization"].read_text())
    )
    candidate = runtime["candidate"]
    return {
        "runtime_lock": runtime,
        "pca": pca,
        "feature_predictor": feature_predictor,
        "control_probe": control_probe,
        "control_probe_metadata": control_probe_metadata,
        "contextual_statistics": statistics,
        "normalization": normalization,
        "selector_config": dict(candidate["candidate"]["selector_config"]),
    }


class ControlInformationOnlineRuntimeV3(ControlInformationOnlineRuntime):
    """Reuse the verified base score/image path with v3 segment-relative state."""

    def __init__(
        self,
        *,
        feature_predictor: torch.nn.Module,
        control_probe: torch.nn.Module,
        normalization: Mapping[str, torch.Tensor],
        contextual_statistics: Mapping[str, Any],
        selector_config: Mapping[str, Any],
        feature_window: int = 8,
        policy_model: Any | None = None,
        initialization_model: Any | None = None,
        pca_artifact: Mapping[str, Any] | None = None,
        capture_fn: Callable[..., Mapping[int, Mapping[str, torch.Tensor]]] = capture_spatial_features,
    ) -> None:
        complete_config = dict(selector_config)
        base_config = {
            key: complete_config[key] for key in sorted(BASE_SELECTOR_KEYS)
        }
        super().__init__(
            feature_predictor=feature_predictor,
            control_probe=control_probe,
            normalization=normalization,
            contextual_statistics=contextual_statistics,
            selector_config=base_config,
            feature_window=feature_window,
            policy_model=policy_model,
            initialization_model=initialization_model,
            pca_artifact=pca_artifact,
            capture_fn=capture_fn,
        )
        self.selector_config = complete_config
        self.boundary_state = ControlInformationBoundaryStateV3(
            statistics=self.contextual_statistics,
            **self.selector_config,
        )
