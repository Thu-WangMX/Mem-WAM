from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from fastwam.evaluation.embodied_information_online import load_predictor
from fastwam.memory.control_information_probe import (
    load_compact_control_probe,
    score_counterfactual_control_information,
)
from scripts.build_putback_feature_predictor_dataset import build_projected_episode
from scripts.build_putback_control_information_dataset import DATASET_SCHEMA
from scripts.prepare_putback_phase_annotation import load_episode_control


TRACE_SCHEMA = "putback_counterfactual_control_information_traces_v1"
RUNTIME_LOCK_SCHEMA = "putback_locked_control_information_runtime_v1"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def trace_episode(
    *,
    episode: int,
    dynamics_examples: Sequence[dict[str, Any]],
    control_probe_examples: Sequence[dict[str, Any]],
    feature_predictor: torch.nn.Module,
    control_probe: torch.nn.Module,
) -> dict[str, Any]:
    """Score one episode in increasing detector-frame order."""

    episode = int(episode)
    probe_by_frame = {
        int(row["frame"]): row
        for row in control_probe_examples
        if int(row["episode"]) == episode
    }
    dynamics_rows = sorted(
        (
            row
            for row in dynamics_examples
            if int(row["episode"]) == episode
        ),
        key=lambda row: int(row["target_frame"]),
    )
    if not dynamics_rows:
        raise ValueError(f"episode {episode} has no dynamics examples")
    frames = []
    information = []
    action_shift = []
    for row in dynamics_rows:
        frame = int(row["target_frame"])
        if frame < 16 or frame % 4:
            raise ValueError("dynamics target is not a post-warmup detector frame")
        if frame not in probe_by_frame:
            raise KeyError(f"control-probe example is missing frame {frame}")
        probe_row = probe_by_frame[frame]
        observed = torch.as_tensor(row["target"], dtype=torch.float32)
        probe_feature = torch.as_tensor(probe_row["feature"], dtype=torch.float32)
        if observed.shape != probe_feature.shape or not torch.allclose(
            observed, probe_feature, rtol=0, atol=1e-6
        ):
            raise ValueError(f"feature mismatch between dynamics and probe artifacts at {frame}")
        result = score_counterfactual_control_information(
            feature_predictor=feature_predictor,
            control_probe=control_probe,
            history=torch.as_tensor(row["history"], dtype=torch.float32),
            dynamics_condition=torch.as_tensor(row["condition"], dtype=torch.float32),
            observed_feature=observed,
            normalized_proprio=torch.as_tensor(
                probe_row["proprio"], dtype=torch.float32
            ),
        )
        frames.append(frame)
        information.append(result.information)
        action_shift.append(
            torch.sqrt(
                torch.mean(
                    (result.posterior_action - result.prior_action).square(), dim=0
                )
            )
        )
    frame_tensor = torch.tensor(frames, dtype=torch.int64)
    if len(frame_tensor) > 1 and not bool((frame_tensor[1:] > frame_tensor[:-1]).all()):
        raise RuntimeError("counterfactual information frames are not strictly increasing")
    return {
        "schema_version": "putback_counterfactual_control_information_episode_v1",
        "episode": episode,
        "frame_indices": frame_tensor,
        "information": torch.tensor(information, dtype=torch.float32),
        "action_shift_by_dim": torch.stack(action_shift),
    }


def _episode_range(value: str) -> list[int]:
    left, right = (int(part) for part in value.split("-", 1))
    episodes = list(range(left, right + 1))
    if not episodes or left < 0 or right > 49:
        raise ValueError("trace episodes must be a nonempty subset of 0-49")
    return episodes


def validate_trace_access(
    episodes: Sequence[int],
    *,
    runtime_lock_path: str | Path | None,
) -> Path | None:
    selected = [int(value) for value in episodes]
    if selected == list(range(40)):
        if runtime_lock_path is not None:
            raise ValueError("candidate trace must not use a runtime lock")
        return None
    if selected != list(range(40, 50)):
        raise ValueError("trace access is locked to episodes 0-39 or 40-49")
    if runtime_lock_path is None:
        raise ValueError("heldout trace requires the final runtime lock")
    path = Path(runtime_lock_path).expanduser().resolve()
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != RUNTIME_LOCK_SCHEMA:
        raise ValueError("heldout trace runtime lock schema is incompatible")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-bank", required=True)
    parser.add_argument("--pca", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--probe-dataset", required=True)
    parser.add_argument("--feature-predictor", required=True)
    parser.add_argument("--control-probe", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", default="0-39")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-history", type=int, default=8)
    parser.add_argument("--runtime-lock")
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite information traces: {output}")
    episodes = _episode_range(args.episodes)
    runtime_lock_path = validate_trace_access(
        episodes, runtime_lock_path=args.runtime_lock
    )
    if int(args.max_history) != 8:
        raise ValueError("feature-predictor history is locked to eight")

    device = torch.device(args.device)
    feature_predictor_path = Path(args.feature_predictor).resolve()
    control_probe_path = Path(args.control_probe).resolve()
    feature_predictor = load_predictor(feature_predictor_path, device=device)
    control_probe, control_metadata = load_compact_control_probe(
        control_probe_path, device=device
    )
    pca_path = Path(args.pca).resolve()
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
    feature_root = Path(args.feature_bank).resolve()
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
                    "mean_information": float(traces[episode]["information"].mean()),
                }
            ),
            flush=True,
        )

    payload = {
        "schema_version": TRACE_SCHEMA,
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
            **(
                {}
                if runtime_lock_path is None
                else {"runtime_lock": _sha256(runtime_lock_path)}
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)


if __name__ == "__main__":
    main()
