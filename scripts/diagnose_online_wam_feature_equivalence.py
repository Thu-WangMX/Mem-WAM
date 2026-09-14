from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate

from experiments.robotwin.fastwam_policy.deploy_policy import (
    _compose_sim_cfg,
    _frozen_init_scorer_model_cfg,
)
from fastwam.evaluation.embodied_information_online import (
    _stack_captured,
    initialization_fingerprint,
    project_captured_features,
)
from fastwam.memory.multilayer_spatial_feature import capture_spatial_features
from fastwam.memory.wam_embedding_feature import load_episode_text_context


EXPECTED_FINGERPRINT = (
    "51652b699bc50c5e1b0fa582a996788ea041d341f013043cf7174f52de679bc8"
)


def _comparison(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | bool]:
    actual = actual.float().reshape(-1)
    expected = expected.float().reshape(-1)
    delta = actual - expected
    return {
        "exact": bool(torch.equal(actual, expected)),
        "allclose": bool(torch.allclose(actual, expected, atol=2e-5, rtol=2e-4)),
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
        "rmse": float(delta.square().mean().sqrt()),
        "cosine": float(torch.nn.functional.cosine_similarity(actual, expected, dim=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model-base", type=Path, required=True)
    parser.add_argument("--action-init", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--pca", type=Path, required=True)
    parser.add_argument("--text-cache", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=30)
    parser.add_argument("--output-indices", default="0,1,2,4,8,12,20")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing existing output: {args.output}")

    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(args.model_base.resolve())
    os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
    os.environ["FASTWAM_ACTION_DIT_INIT"] = str(args.action_init.resolve())
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    cfg = _compose_sim_cfg(None, "sim_robotwin_control_information", "robotwin_control_information_eval")
    init_cfg = _frozen_init_scorer_model_cfg(
        cfg.model, official_action_dit_path=str(args.action_init.resolve())
    )
    started = time.perf_counter()
    model = instantiate(init_cfg, model_dtype=torch.bfloat16, device="cuda").to("cuda").eval()
    load_seconds = time.perf_counter() - started
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    fingerprint = initialization_fingerprint(model)
    if fingerprint != EXPECTED_FINGERPRINT:
        raise RuntimeError(f"initialization fingerprint mismatch: {fingerprint}")

    latent_manifest = json.loads((args.latent_root / "manifest.json").read_text())
    latent_entry = next(
        row for row in latent_manifest["episodes"] if int(row["episode"]) == args.episode
    )
    latent_payload = torch.load(
        args.latent_root / latent_entry["relative_path"],
        map_location="cpu",
        weights_only=True,
    )
    feature_payload = torch.load(
        args.feature_root / "features" / f"episode_{args.episode:03d}.pt",
        map_location="cpu",
        weights_only=True,
    )
    pca = torch.load(args.pca, map_location="cpu", weights_only=True)
    text = load_episode_text_context(args.text_cache, args.dataset_root, args.episode)
    output_indices = [int(value) for value in args.output_indices.split(",")]

    results = []
    for phase in (0, 4, 8, 12):
        source = latent_payload["phases"][str(phase)]
        latents = torch.as_tensor(source["latents"], dtype=torch.bfloat16)
        expected_features = torch.as_tensor(
            feature_payload["phases"][str(phase)]["features"], dtype=torch.float32
        )
        frames = torch.as_tensor(source["frame_indices"], dtype=torch.int64)
        for index in output_indices:
            if index >= len(frames):
                continue
            start = max(0, index + 1 - 8)
            causal = latents[start : index + 1, :, 0].permute(1, 0, 2, 3)
            torch.cuda.synchronize()
            capture_started = time.perf_counter()
            captured = capture_spatial_features(
                model,
                latents=causal,
                video_context=text["context"],
                video_context_mask=text["mask"],
            )
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - capture_started) * 1000.0
            actual_raw = _stack_captured(captured)
            expected_raw = expected_features[index]
            actual_projected = project_captured_features(captured, pca).reshape(-1)
            expected_projected = torch.einsum(
                "lrd,lrcd->lrc",
                expected_raw - torch.as_tensor(pca["mean"], dtype=torch.float32),
                torch.as_tensor(pca["components"], dtype=torch.float32),
            ) / torch.as_tensor(pca["projected_scale"], dtype=torch.float32)
            result = {
                "phase": phase,
                "frame": int(frames[index]),
                "output_index": index,
                "history_length": index + 1 - start,
                "elapsed_ms": elapsed_ms,
                "raw": _comparison(actual_raw, expected_raw),
                "projected": _comparison(actual_projected, expected_projected),
            }
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)

    report = {
        "schema_version": "online_wam_feature_equivalence_diagnostic_v1",
        "episode": int(args.episode),
        "initialization_fingerprint": fingerprint,
        "model_load_seconds": load_seconds,
        "comparisons": results,
        "raw_allclose": all(item["raw"]["allclose"] for item in results),
        "projected_allclose": all(item["projected"]["allclose"] for item in results),
        "raw_global_max_abs": max(item["raw"]["max_abs"] for item in results),
        "projected_global_max_abs": max(
            item["projected"]["max_abs"] for item in results
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"report": str(args.output), **report}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
