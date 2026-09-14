from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from fastwam.evaluation.control_information_online_v2 import (
    CANDIDATE_LOCK_SCHEMA_V2,
    RUNTIME_LOCK_SCHEMA_V2,
)
from fastwam.evaluation.embodied_information_online import load_predictor
from fastwam.memory.control_information_probe import load_compact_control_probe
from scripts.build_putback_control_information_dataset import DATASET_SCHEMA
from scripts.build_putback_feature_predictor_dataset import build_projected_episode
from scripts.prepare_putback_phase_annotation import load_episode_control
from scripts.trace_putback_control_information import (
    TRACE_SCHEMA,
    _sha256,
    trace_episode,
)


def _validate_heldout_trace_access(
    episodes: Sequence[int],
    *,
    runtime_lock_path: str | Path,
    candidate_lock_path: str | Path,
    runtime_lock_schema: str,
    candidate_lock_schema: str,
    version_label: str,
) -> tuple[Path, Path]:
    if [int(value) for value in episodes] != list(range(40, 50)):
        raise ValueError(
            f"{version_label} heldout trace is locked to exactly episodes 40-49"
        )
    runtime_path = Path(runtime_lock_path).expanduser().resolve()
    candidate_path = Path(candidate_lock_path).expanduser().resolve()
    runtime = json.loads(runtime_path.read_text())
    candidate = json.loads(candidate_path.read_text())
    if runtime.get("schema_version") != runtime_lock_schema:
        raise ValueError(
            f"{version_label} heldout trace runtime lock schema is incompatible"
        )
    if candidate.get("schema_version") != candidate_lock_schema:
        raise ValueError(
            f"{version_label} heldout trace candidate lock schema is incompatible"
        )
    if runtime.get("candidate_sha256") != _sha256(candidate_path):
        raise ValueError(
            f"{version_label} heldout trace candidate lock sha256 mismatch"
        )
    if runtime.get("candidate") != candidate:
        raise ValueError(
            f"{version_label} runtime does not embed the candidate verbatim"
        )
    return runtime_path, candidate_path


def validate_heldout_trace_access_v2(
    episodes: Sequence[int],
    *,
    runtime_lock_path: str | Path,
    candidate_lock_path: str | Path,
) -> tuple[Path, Path]:
    return _validate_heldout_trace_access(
        episodes,
        runtime_lock_path=runtime_lock_path,
        candidate_lock_path=candidate_lock_path,
        runtime_lock_schema=RUNTIME_LOCK_SCHEMA_V2,
        candidate_lock_schema=CANDIDATE_LOCK_SCHEMA_V2,
        version_label="v2",
    )


def _verify_scoring_dependencies(
    *,
    candidate: dict[str, Any],
    feature_root: Path,
    pca_path: Path,
    feature_predictor_path: Path,
    control_probe_path: Path,
    base_score_source: Path,
    version_label: str = "v2",
) -> None:
    expected = candidate["hashes"]
    actual = {
        "feature_bank_manifest": _sha256(feature_root / "bank_manifest.json"),
        "pca": _sha256(pca_path),
        "feature_predictor": _sha256(feature_predictor_path),
        "control_probe": _sha256(control_probe_path),
        "base_score_source": _sha256(base_score_source),
    }
    for name, digest in actual.items():
        if expected.get(name) != digest:
            raise ValueError(
                f"{version_label} heldout scoring dependency {name} sha256 mismatch: "
                f"{digest} != {expected.get(name)}"
            )


def run_heldout_trace_cli(
    *,
    runtime_lock_schema: str,
    candidate_lock_schema: str,
    trace_authorization: str,
    version_label: str,
) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-bank", required=True)
    parser.add_argument("--pca", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--probe-dataset", required=True)
    parser.add_argument("--feature-predictor", required=True)
    parser.add_argument("--control-probe", required=True)
    parser.add_argument("--base-score-source", required=True)
    parser.add_argument("--runtime-lock", required=True)
    parser.add_argument("--candidate-lock", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-history", type=int, default=8)
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite {version_label} heldout traces: {output}"
        )
    episodes = list(range(40, 50))
    runtime_path, candidate_path = _validate_heldout_trace_access(
        episodes,
        runtime_lock_path=args.runtime_lock,
        candidate_lock_path=args.candidate_lock,
        runtime_lock_schema=runtime_lock_schema,
        candidate_lock_schema=candidate_lock_schema,
        version_label=version_label,
    )
    if int(args.max_history) != 8:
        raise ValueError("feature-predictor history is locked to eight")

    candidate = json.loads(candidate_path.read_text())
    feature_root = Path(args.feature_bank).resolve()
    pca_path = Path(args.pca).resolve()
    feature_predictor_path = Path(args.feature_predictor).resolve()
    control_probe_path = Path(args.control_probe).resolve()
    base_score_source = Path(args.base_score_source).resolve()
    _verify_scoring_dependencies(
        candidate=candidate,
        feature_root=feature_root,
        pca_path=pca_path,
        feature_predictor_path=feature_predictor_path,
        control_probe_path=control_probe_path,
        base_score_source=base_score_source,
        version_label=version_label,
    )

    device = torch.device(args.device)
    feature_predictor = load_predictor(feature_predictor_path, device=device)
    control_probe, control_metadata = load_compact_control_probe(
        control_probe_path, device=device
    )
    pca = torch.load(pca_path, map_location="cpu", weights_only=True)
    probe_dataset_path = Path(args.probe_dataset).resolve()
    probe_payload = torch.load(
        probe_dataset_path, map_location="cpu", weights_only=False
    )
    if probe_payload.get("schema_version") != DATASET_SCHEMA:
        raise ValueError("control-probe dataset schema is incompatible")
    probe_rows_by_episode = {
        episode: [
            row
            for row in probe_payload["rows"]
            if int(row["episode"]) == episode
        ]
        for episode in episodes
    }
    dataset_root = Path(args.dataset_root).resolve()
    traces = {}
    episode_lengths = {}
    for episode in episodes:
        feature = torch.load(
            feature_root / "features" / f"episode_{episode:03d}.pt",
            map_location="cpu",
            weights_only=True,
        )
        maximum_frame = max(
            int(stream["frame_indices"].max())
            for stream in feature["phases"].values()
        )
        actions, proprio = load_episode_control(
            dataset_root, episode, list(range(maximum_frame + 1))
        )
        dynamics = build_projected_episode(
            episode=episode,
            feature_payload=feature,
            pca_artifact=pca,
            actions=torch.from_numpy(actions),
            proprio=torch.from_numpy(proprio),
            max_history=args.max_history,
        )["visual_action"]
        traces[episode] = trace_episode(
            episode=episode,
            dynamics_examples=dynamics,
            control_probe_examples=probe_rows_by_episode[episode],
            feature_predictor=feature_predictor,
            control_probe=control_probe,
        )
        episode_lengths[episode] = len(actions)
        print(
            json.dumps(
                {
                    "episode": episode,
                    "scores": len(traces[episode]["frame_indices"]),
                    "mean_information": float(
                        traces[episode]["information"].mean()
                    ),
                }
            ),
            flush=True,
        )

    payload = {
        "schema_version": TRACE_SCHEMA,
        "trace_authorization": trace_authorization,
        "complete": True,
        "episodes": traces,
        "episode_lengths": episode_lengths,
        "traced_episodes": episodes,
        "feature_predictor_schema": "putback_feature_predictor_model_v1",
        "control_probe_schema": control_metadata["schema_version"],
        "hashes": {
            "feature_bank_manifest": _sha256(feature_root / "bank_manifest.json"),
            "pca": _sha256(pca_path),
            "probe_dataset": _sha256(probe_dataset_path),
            "feature_predictor": _sha256(feature_predictor_path),
            "control_probe": _sha256(control_probe_path),
            "base_score_source": _sha256(base_score_source),
            "candidate_lock": _sha256(candidate_path),
            "runtime_lock": _sha256(runtime_path),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)


def main() -> None:
    run_heldout_trace_cli(
        runtime_lock_schema=RUNTIME_LOCK_SCHEMA_V2,
        candidate_lock_schema=CANDIDATE_LOCK_SCHEMA_V2,
        trace_authorization="locked_control_information_runtime_v2",
        version_label="v2",
    )


if __name__ == "__main__":
    main()
