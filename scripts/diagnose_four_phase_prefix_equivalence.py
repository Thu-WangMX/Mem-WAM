from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import torch

from scripts.build_putback_four_phase_latents import load_registered_vae
from scripts.precompute_full_kv_observation_latents import _encode_batch, _mosaic


def _episode_row(dataset_root: Path, episode: int) -> dict:
    rows = [
        json.loads(line)
        for line in (dataset_root / "meta" / "episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    matches = [row for row in rows if int(row["episode_index"]) == int(episode)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one metadata row for episode {episode}, got {len(matches)}")
    return matches[0]


def _latent_entry(latent_root: Path, episode: int) -> tuple[Path, dict]:
    manifest = json.loads((latent_root / "manifest.json").read_text())
    matches = [row for row in manifest["episodes"] if int(row["episode"]) == int(episode)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one latent entry for episode {episode}, got {len(matches)}")
    return latent_root / matches[0]["relative_path"], matches[0]


def _comparison(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | bool]:
    actual = actual.float().reshape(-1)
    expected = expected.float().reshape(-1)
    delta = actual - expected
    return {
        "exact": bool(torch.equal(actual, expected)),
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
        "rmse": float(delta.square().mean().sqrt()),
        "cosine": float(torch.nn.functional.cosine_similarity(actual, expected, dim=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--vae-path", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=30)
    parser.add_argument("--output-indices", default="0,1,2,4,8,12,20")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing existing output: {args.output}")
    output_indices = [int(value) for value in args.output_indices.split(",")]
    if not output_indices or min(output_indices) < 0:
        raise ValueError("output indices must be nonempty and nonnegative")

    row = _episode_row(args.dataset_root, args.episode)
    latent_path, latent_entry = _latent_entry(args.latent_root, args.episode)
    cached = torch.load(latent_path, map_location="cpu", weights_only=True)
    device = torch.device("cuda")
    vae = load_registered_vae(args.vae_path, device=device)
    results = []
    raw_path = Path(row["raw_file_name"])
    with h5py.File(raw_path, "r") as handle:
        mosaics = {
            frame: _mosaic(handle, frame)
            for frame in range(0, int(row["length"]), 4)
        }
    for phase in (0, 4, 8, 12):
        phase_row = cached["phases"][str(phase)]
        expected_latents = torch.as_tensor(phase_row["latents"])
        for output_index in output_indices:
            if output_index >= len(expected_latents):
                continue
            prefix_frames = list(range(phase, phase + 16 * output_index + 1, 4))
            video = torch.stack([mosaics[frame] for frame in prefix_frames], dim=1)
            torch.cuda.synchronize()
            started = time.perf_counter()
            encoded = _encode_batch(vae, [video], device)[0]
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if int(encoded.shape[1]) != output_index + 1:
                raise RuntimeError(
                    f"phase {phase} index {output_index}: prefix produced "
                    f"{encoded.shape[1]} latents, expected {output_index + 1}"
                )
            actual = encoded[:, -1:]
            expected = expected_latents[output_index].squeeze(1)
            comparison = _comparison(actual, expected)
            result = {
                "phase": phase,
                "frame": phase + 16 * output_index,
                "output_index": output_index,
                "input_frames": len(prefix_frames),
                "elapsed_ms": elapsed_ms,
                **comparison,
            }
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
            del video, encoded, actual, expected
            torch.cuda.empty_cache()

    report = {
        "schema_version": "four_phase_prefix_equivalence_diagnostic_v1",
        "episode": int(args.episode),
        "raw_path": str(raw_path),
        "latent_path": str(latent_path),
        "latent_file_sha256": latent_entry["file_sha256"],
        "vae_sha256": cached["vae_sha256"],
        "comparisons": results,
        "all_exact": all(item["exact"] for item in results),
        "global_max_abs": max(item["max_abs"] for item in results),
        "global_mean_abs": sum(item["mean_abs"] for item in results) / len(results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"report": str(args.output), **report}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
