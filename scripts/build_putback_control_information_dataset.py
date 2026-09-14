from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from fastwam.memory.control_information_dataset import (
    build_control_probe_examples,
    normalization_from_policy_stats,
)
from scripts.fit_putback_wam_pca import PCA_SCHEMA, transform_feature_tensor
from scripts.prepare_putback_phase_annotation import load_episode_control


DATASET_SCHEMA = "putback_control_information_probe_dataset_v1"
SPLIT_ROLES = {
    "train": list(range(26)),
    "validation": list(range(26, 30)),
    "calibration": list(range(30, 40)),
    "heldout": list(range(40, 50)),
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalization_from_stats(payload: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Load the exact global z-score tensors used by the PutBack policy."""

    return normalization_from_policy_stats(payload)


def build_projected_episode(
    *,
    episode: int,
    feature_payload: Mapping[str, Any],
    pca_artifact: Mapping[str, Any],
    actions: torch.Tensor,
    proprio: torch.Tensor,
    normalization: Mapping[str, torch.Tensor],
    horizon: int = 16,
) -> list[dict[str, Any]]:
    projected = {}
    for phase, stream in feature_payload["phases"].items():
        values = transform_feature_tensor(stream["features"], dict(pca_artifact))
        projected[str(phase)] = {
            "frame_indices": stream["frame_indices"],
            "features": values.flatten(start_dim=1),
            "warmup": stream["warmup"],
        }
    return build_control_probe_examples(
        episode=int(episode),
        projected_phases=projected,
        actions=torch.as_tensor(actions),
        proprio=torch.as_tensor(proprio),
        action_mean=normalization["action_mean"],
        action_std=normalization["action_std"],
        proprio_mean=normalization["proprio_mean"],
        proprio_std=normalization["proprio_std"],
        horizon=int(horizon),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-bank", required=True)
    parser.add_argument("--pca", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--normalization-stats", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", default="0-49")
    parser.add_argument("--horizon", type=int, default=16)
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite probe dataset: {output}")
    left, right = (int(value) for value in args.episodes.split("-", 1))
    episodes = list(range(left, right + 1))
    if episodes != list(range(50)):
        raise ValueError("control-information dataset requires exactly episodes 0-49")
    if int(args.horizon) != 16:
        raise ValueError("control-information action horizon is locked to 16")

    feature_root = Path(args.feature_bank).expanduser().resolve()
    feature_manifest_path = feature_root / "bank_manifest.json"
    feature_manifest = json.loads(feature_manifest_path.read_text())
    if feature_manifest.get("complete") is not True or int(
        feature_manifest.get("episode_count", -1)
    ) != 50:
        raise ValueError("feature bank must be complete for 50 episodes")
    pca_path = Path(args.pca).expanduser().resolve()
    pca = torch.load(pca_path, map_location="cpu", weights_only=True)
    if pca.get("schema_version") != PCA_SCHEMA or pca.get("episodes") != list(range(30)):
        raise ValueError("PCA must be the locked episodes 0-29 artifact")
    statistics_path = Path(args.normalization_stats).expanduser().resolve()
    normalization = normalization_from_stats(json.loads(statistics_path.read_text()))

    rows: list[dict[str, Any]] = []
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    counts: dict[str, int] = {}
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
        requested_frames = list(range(maximum_frame + 1))
        actions, proprio = load_episode_control(
            dataset_root, episode, requested_frames
        )
        episode_rows = build_projected_episode(
            episode=episode,
            feature_payload=feature,
            pca_artifact=pca,
            actions=torch.from_numpy(actions),
            proprio=torch.from_numpy(proprio),
            normalization=normalization,
            horizon=args.horizon,
        )
        if not episode_rows:
            raise RuntimeError(f"episode {episode} produced no control-probe examples")
        rows.extend(episode_rows)
        counts[str(episode)] = len(episode_rows)
        print(json.dumps({"episode": episode, "examples": len(episode_rows)}), flush=True)

    payload = {
        "schema_version": DATASET_SCHEMA,
        "complete": True,
        "episodes": episodes,
        "split_roles": SPLIT_ROLES,
        "horizon": int(args.horizon),
        "feature_dim": int(rows[0]["feature"].numel()),
        "action_dim": 14,
        "proprio_dim": 14,
        "episode_example_counts": counts,
        "hashes": {
            "feature_bank_manifest": _sha256(feature_manifest_path),
            "pca": _sha256(pca_path),
            "normalization_stats": _sha256(statistics_path),
        },
        "normalization": {key: value.clone() for key, value in normalization.items()},
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)


if __name__ == "__main__":
    main()
