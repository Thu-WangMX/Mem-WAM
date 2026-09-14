from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from fastwam.memory.four_phase_temporal import PHASE_OFFSETS
from fastwam.memory.multilayer_spatial_feature import FEATURE_LAYERS, FEATURE_REGIONS
from scripts.extract_putback_four_phase_wam_features import FEATURE_BANK_SCHEMA


PCA_SCHEMA = "putback_wam_stream_pca_v1"


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit_pca_artifact(
    feature_bank: Mapping[int, dict[str, Any]],
    *,
    episodes: Sequence[int],
    component_dim: int = 64,
    source_bank_sha256: str,
) -> dict[str, Any]:
    selected = [int(value) for value in episodes]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("PCA episodes must be nonempty and unique")
    if any(episode < 0 or episode > 29 for episode in selected):
        raise ValueError("PCA may use only episodes 0-29")
    if any(episode not in feature_bank for episode in selected):
        raise KeyError("PCA feature bank is missing a selected episode")
    component_dim = int(component_dim)
    if component_dim <= 0:
        raise ValueError("component_dim must be positive")
    samples = []
    feature_dim: int | None = None
    for episode in selected:
        payload = feature_bank[episode]
        if payload.get("schema_version") != FEATURE_BANK_SCHEMA:
            raise ValueError("feature bank schema mismatch")
        for phase in PHASE_OFFSETS:
            row = payload["phases"][str(phase)]
            features = torch.as_tensor(row["features"], dtype=torch.float32)
            warmup = torch.as_tensor(row["warmup"], dtype=torch.bool)
            if tuple(features.shape[1:3]) != (
                len(FEATURE_LAYERS),
                len(FEATURE_REGIONS),
            ) or warmup.shape != (len(features),):
                raise ValueError("feature stream shape mismatch")
            feature_dim = int(features.shape[-1])
            samples.append(features[~warmup])
    matrix = torch.cat(samples, dim=0)
    if matrix.shape[0] <= component_dim:
        raise ValueError("PCA has too few training samples")
    means = torch.empty(
        (len(FEATURE_LAYERS), len(FEATURE_REGIONS), int(feature_dim)),
        dtype=torch.float32,
    )
    components = torch.empty(
        (
            len(FEATURE_LAYERS),
            len(FEATURE_REGIONS),
            component_dim,
            int(feature_dim),
        ),
        dtype=torch.float32,
    )
    projected_scale = torch.empty(
        (len(FEATURE_LAYERS), len(FEATURE_REGIONS), component_dim),
        dtype=torch.float32,
    )
    explained_variance = torch.empty_like(projected_scale)
    for layer_index in range(len(FEATURE_LAYERS)):
        for region_index in range(len(FEATURE_REGIONS)):
            stream = matrix[:, layer_index, region_index]
            mean = stream.mean(dim=0)
            centered = stream - mean
            _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
            tolerance = torch.finfo(singular.dtype).eps * max(centered.shape) * float(
                singular.max().item()
            )
            rank = int((singular > tolerance).sum().item())
            if rank < component_dim:
                raise ValueError(
                    f"stream ({layer_index},{region_index}) rank {rank} is below {component_dim}"
                )
            basis = vh[:component_dim]
            projected = centered @ basis.T
            scale = projected.std(dim=0, unbiased=False)
            if not bool(torch.isfinite(scale).all()) or bool((scale <= 1e-8).any()):
                raise ValueError("PCA projected scale collapsed")
            means[layer_index, region_index] = mean
            components[layer_index, region_index] = basis
            projected_scale[layer_index, region_index] = scale
            explained_variance[layer_index, region_index] = (
                singular[:component_dim].square() / max(int(centered.shape[0]) - 1, 1)
            )
    return {
        "schema_version": PCA_SCHEMA,
        "episodes": selected,
        "source_bank_sha256": str(source_bank_sha256),
        "feature_layers": list(FEATURE_LAYERS),
        "feature_regions": list(FEATURE_REGIONS),
        "feature_dim": int(feature_dim),
        "component_dim": component_dim,
        "mean": means,
        "components": components,
        "projected_scale": projected_scale,
        "explained_variance": explained_variance,
    }


def transform_feature_tensor(features: torch.Tensor, artifact: dict[str, Any]) -> torch.Tensor:
    if artifact.get("schema_version") != PCA_SCHEMA:
        raise ValueError("PCA artifact schema mismatch")
    values = torch.as_tensor(features, dtype=torch.float32)
    expected_tail = (
        len(FEATURE_LAYERS),
        len(FEATURE_REGIONS),
        int(artifact["feature_dim"]),
    )
    if tuple(values.shape[-3:]) != expected_tail:
        raise ValueError(f"feature tail must be {expected_tail}")
    centered = values - torch.as_tensor(artifact["mean"])
    projected = torch.einsum(
        "...lrd,lrcd->...lrc",
        centered,
        torch.as_tensor(artifact["components"]),
    )
    return projected / torch.as_tensor(artifact["projected_scale"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-bank", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", default="0-29")
    parser.add_argument("--component-dim", type=int, default=64)
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite PCA artifact: {output}")
    root = Path(args.feature_bank).expanduser().resolve()
    manifest_path = root / "bank_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("complete") is not True or manifest.get("schema_version") != FEATURE_BANK_SCHEMA:
        raise ValueError("feature bank is incomplete or incompatible")
    left, right = (int(value) for value in args.episodes.split("-", 1))
    episodes = list(range(left, right + 1))
    bank = {
        episode: torch.load(
            root / "features" / f"episode_{episode:03d}.pt",
            map_location="cpu",
            weights_only=True,
        )
        for episode in episodes
    }
    artifact = fit_pca_artifact(
        bank,
        episodes=episodes,
        component_dim=args.component_dim,
        source_bank_sha256=_sha256(manifest_path),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(artifact, temporary)
    temporary.replace(output)


if __name__ == "__main__":
    main()

