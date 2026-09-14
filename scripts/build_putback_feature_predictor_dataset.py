from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from fastwam.memory.predictor_dataset import build_predictor_examples
from scripts.fit_putback_wam_pca import PCA_SCHEMA, transform_feature_tensor
from scripts.prepare_putback_phase_annotation import load_episode_control
from scripts.train_putback_feature_predictor import make_split_manifest


DATASET_SCHEMA = "putback_feature_predictor_dataset_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_projected_episode(
    *, episode: int, feature_payload: dict[str, Any], pca_artifact: dict[str, Any],
    actions: torch.Tensor, proprio: torch.Tensor, max_history: int,
) -> dict[str, list[dict[str, Any]]]:
    projected = {}
    for phase, row in feature_payload["phases"].items():
        values = transform_feature_tensor(row["features"], pca_artifact)
        projected[phase] = {
            "frame_indices": row["frame_indices"],
            "features": values.flatten(start_dim=1),
            "warmup": row["warmup"],
        }
    common = dict(
        episode=int(episode), projected_phases=projected, actions=actions,
        proprio=proprio, max_history=int(max_history),
    )
    return {
        "visual_only": build_predictor_examples(mode="visual_only", **common),
        "visual_action": build_predictor_examples(mode="visual_action", **common),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-bank", required=True)
    parser.add_argument("--pca", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", default="0-29")
    parser.add_argument("--max-history", type=int, default=8)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite predictor dataset: {output}")
    left, right = (int(value) for value in args.episodes.split("-", 1))
    episodes = list(range(left, right + 1))
    if episodes != list(range(30)):
        raise ValueError("predictor artifact requires exactly episodes 0-29")
    root = Path(args.feature_bank).resolve()
    pca_path = Path(args.pca).resolve()
    pca = torch.load(pca_path, map_location="cpu", weights_only=True)
    if pca.get("schema_version") != PCA_SCHEMA or pca.get("episodes") != episodes:
        raise ValueError("PCA artifact is incompatible with the locked split")
    rows = {"visual_only": [], "visual_action": []}
    for episode in episodes:
        feature = torch.load(
            root / "features" / f"episode_{episode:03d}.pt",
            map_location="cpu", weights_only=True,
        )
        maximum_frame = max(
            int(stream["frame_indices"].max()) for stream in feature["phases"].values()
        )
        requested = list(range(maximum_frame + 1))
        actions, proprio = load_episode_control(args.dataset_root, episode, requested)
        built = build_projected_episode(
            episode=episode, feature_payload=feature, pca_artifact=pca,
            actions=torch.from_numpy(actions), proprio=torch.from_numpy(proprio),
            max_history=args.max_history,
        )
        for mode in rows:
            rows[mode].extend(built[mode])
        print(json.dumps({"episode": episode, "examples": len(built["visual_only"])}), flush=True)
    payload = {
        "schema_version": DATASET_SCHEMA,
        "split_manifest": make_split_manifest(
            train_episodes=range(26), val_episodes=range(26, 30)
        ),
        "pca_sha256": _sha256(pca_path),
        "feature_bank_manifest_sha256": _sha256(root / "bank_manifest.json"),
        "max_history": args.max_history,
        **rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)


if __name__ == "__main__":
    main()

