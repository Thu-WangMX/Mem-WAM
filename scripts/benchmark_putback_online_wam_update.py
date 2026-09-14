from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from fastwam.memory.multilayer_spatial_feature import capture_spatial_features
from scripts.extract_putback_four_phase_wam_features import _instantiate_initialization_model
from scripts.extract_putback_init_wam_embedding_bank import load_episode_text_context


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--latent-cache", required=True)
    parser.add_argument("--text-cache", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite latency benchmark: {output}")
    latent_root = Path(args.latent_cache).resolve()
    manifest = json.loads((latent_root / "manifest.json").read_text())
    entry = next(row for row in manifest["episodes"] if row["episode"] == args.episode)
    payload = torch.load(latent_root / entry["relative_path"], map_location="cpu", weights_only=True)
    phase = payload["phases"]["0"]["latents"]
    causal = phase[max(0, len(phase)-8):, :, 0].permute(1,0,2,3)
    text = load_episode_text_context(args.text_cache, args.dataset_root, args.episode)
    load_start = time.perf_counter()
    model, _ = _instantiate_initialization_model(
        repo=Path(args.repo).resolve(), output=output.parent, device="cuda:0"
    )
    model_load_seconds = time.perf_counter() - load_start
    for _ in range(3):
        capture_spatial_features(
            model, latents=causal, video_context=text["context"],
            video_context_mask=text["mask"],
        )
    torch.cuda.synchronize()
    wall=[]; cuda=[]
    for _ in range(args.iterations):
        start_event=torch.cuda.Event(enable_timing=True); end_event=torch.cuda.Event(enable_timing=True)
        begin=time.perf_counter(); start_event.record()
        capture_spatial_features(
            model, latents=causal, video_context=text["context"],
            video_context_mask=text["mask"],
        )
        end_event.record(); torch.cuda.synchronize()
        wall.append(time.perf_counter()-begin); cuda.append(start_event.elapsed_time(end_event)/1000)
    report={
        "schema_version":"putback_online_wam_update_latency_v1",
        "episode":args.episode,"iterations":args.iterations,"history_frames":int(causal.shape[1]),
        "model_load_seconds":model_load_seconds,
        "wall_mean_seconds":statistics.mean(wall),"wall_p95_seconds":sorted(wall)[int(.95*(len(wall)-1))],
        "wall_max_seconds":max(wall),"cuda_mean_seconds":statistics.mean(cuda),
        "deadline_seconds":0.4,"all_updates_before_deadline":max(wall)<0.4,
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,indent=2,sort_keys=True))


if __name__ == "__main__":
    main()

