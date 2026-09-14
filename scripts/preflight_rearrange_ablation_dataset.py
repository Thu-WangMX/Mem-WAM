#!/usr/bin/env python3
"""Warm and validate the exact dataset before distributed workers start."""

from __future__ import annotations

import argparse
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    config_dir = Path(__file__).resolve().parents[1] / "configs"
    overrides = [f"task={args.task}", *args.override]
    with initialize_config_dir(
        config_dir=str(config_dir), version_base="1.3"
    ):
        cfg = compose(config_name="train", overrides=overrides)

    dataset = instantiate(cfg.data.train)
    if len(dataset) < 1:
        raise RuntimeError("resolved training dataset is empty")

    probe_indices = sorted({0, len(dataset) // 2, len(dataset) - 1})
    for index in probe_indices:
        sample = dataset[index]
        history = sample.get("history_latents")
        groups = sample.get("history_memory_groups")
        token_counts = sample.get("history_memory_token_counts")
        if history is None or getattr(history, "ndim", None) != 5:
            raise RuntimeError(
                f"sample {index} has invalid history_latents"
            )
        if groups is None or tuple(groups.shape[-1:]) != (2,):
            raise RuntimeError(
                f"sample {index} has invalid history_memory_groups"
            )
        if token_counts is None or int(token_counts.numel()) != int(
            groups.shape[0]
        ):
            raise RuntimeError(
                f"sample {index} has mismatched memory token counts"
            )

    print(
        "dataset_preflight_status=ok "
        f"target={cfg.data.train._target_} "
        f"samples={len(dataset)} probes={probe_indices}",
        flush=True,
    )


if __name__ == "__main__":
    main()
