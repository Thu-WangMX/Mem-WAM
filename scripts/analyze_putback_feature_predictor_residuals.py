from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import torch

from fastwam.memory.predictive_boundary import fit_residual_statistics
from fastwam.memory.predictive_feature_model import FeaturePredictor
from scripts.build_putback_feature_predictor_dataset import build_projected_episode
from scripts.prepare_putback_phase_annotation import load_episode_control
from scripts.train_putback_feature_predictor import PredictorTrainingConfig, _collate


@torch.no_grad()
def predict_stream_residuals(
    model, examples, *, layers: int, regions: int, components: int,
    batch_size: int = 256,
):
    device = next(model.parameters()).device
    model.eval()
    ordered = sorted(examples, key=lambda row: int(row["target_frame"]))
    frames, residual_rows = [], []
    for start in range(0, len(ordered), batch_size):
        rows = ordered[start : start + batch_size]
        history, condition, target, lengths = _collate(rows, device)
        prediction = model(history, condition, lengths=lengths)
        error = (prediction - target).reshape(-1, layers, regions, components)
        residual = error.square().mean(dim=-1).sqrt().flatten(start_dim=1)
        frames.extend(int(row["target_frame"]) for row in rows)
        residual_rows.append(residual.cpu())
    return {
        "frame_indices": torch.tensor(frames, dtype=torch.int64),
        "residuals": torch.cat(residual_rows),
    }


def _load_model(path: Path, device: torch.device):
    resume = torch.load(path, map_location="cpu", weights_only=False)
    allowed = {field.name for field in fields(PredictorTrainingConfig)}
    config = PredictorTrainingConfig(**{
        key: value for key, value in resume["config"].items() if key in allowed
    })
    model = FeaturePredictor(
        feature_dim=config.feature_dim, condition_dim=config.condition_dim,
        hidden_dim=config.hidden_dim,
        condition_embedding_dim=config.condition_embedding_dim,
    ).to(device)
    model.load_state_dict(resume["best_model"])
    return model, config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-bank", required=True)
    parser.add_argument("--pca", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--predictor-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", default="0-39")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite residual analysis: {output}")
    left, right = (int(value) for value in args.episodes.split("-", 1))
    episodes = list(range(left, right + 1))
    if episodes != list(range(40)):
        raise ValueError("residual analysis is locked to episodes 0-39")
    feature_root = Path(args.feature_bank).resolve()
    pca = torch.load(args.pca, map_location="cpu", weights_only=True)
    component_dim = int(pca["component_dim"])
    output.mkdir(parents=True)
    summary = {}
    for mode in ("visual_only", "visual_action"):
        model, config = _load_model(
            Path(args.predictor_root) / mode / "resume.pt", torch.device(args.device)
        )
        traces = {}
        for episode in episodes:
            feature = torch.load(
                feature_root / "features" / f"episode_{episode:03d}.pt",
                map_location="cpu", weights_only=True,
            )
            maximum = max(
                int(row["frame_indices"].max()) for row in feature["phases"].values()
            )
            actions, proprio = load_episode_control(
                args.dataset_root, episode, list(range(maximum + 1))
            )
            examples = build_projected_episode(
                episode=episode, feature_payload=feature, pca_artifact=pca,
                actions=torch.from_numpy(actions), proprio=torch.from_numpy(proprio),
                max_history=8,
            )[mode]
            traces[episode] = predict_stream_residuals(
                model, examples, layers=5, regions=4, components=component_dim,
                batch_size=config.batch_size,
            )
        stats = fit_residual_statistics(
            {episode: traces[episode]["residuals"] for episode in range(30)},
            episodes=range(30),
        )
        artifact = {
            "schema_version": "putback_predictor_residual_traces_v1",
            "mode": mode, "episodes": traces, "statistics": stats,
            "stream_layout": {"layers": [5,11,17,23,29],
                              "regions": ["global","left_wrist","right_wrist","head"]},
        }
        torch.save(artifact, output / f"{mode}.pt")
        summary[mode] = {
            "training_median_min": float(stats["median"].min()),
            "training_median_max": float(stats["median"].max()),
            "mad_scale_min": float(stats["mad_scale"].min()),
            "mad_scale_max": float(stats["mad_scale"].max()),
        }
        del model
        torch.cuda.empty_cache()
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()

