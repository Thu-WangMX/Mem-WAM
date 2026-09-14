#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


REQUIRED_STATE_DICTS = (
    "mot",
    "proprio_encoder",
    "layerwise_block_memory",
)


def _validate_state_dict(payload: Mapping[str, Any], key: str) -> int:
    state_dict = payload.get(key)
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError(f"Checkpoint requires a non-empty `{key}` state dictionary")
    non_tensors = [name for name, value in state_dict.items() if not torch.is_tensor(value)]
    if non_tensors:
        raise ValueError(
            f"Checkpoint `{key}` contains non-tensor entries: {non_tensors[:3]}"
        )
    return len(state_dict)


def validate_checkpoint(path: Path) -> dict[str, object]:
    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(payload, Mapping):
        raise ValueError("Checkpoint payload must be a mapping")

    tensor_counts = {
        key: _validate_state_dict(payload, key) for key in REQUIRED_STATE_DICTS
    }
    slots = payload["layerwise_block_memory"].get("slots")
    if not torch.is_tensor(slots) or slots.ndim != 3 or tuple(slots.shape[:2]) != (1, 32):
        raise ValueError(
            "Checkpoint `layerwise_block_memory.slots` must have shape [1,32,D]"
        )
    mot = payload["mot"]
    for prefix in ("mixtures.video.", "mixtures.action."):
        if not any(str(name).startswith(prefix) for name in mot):
            raise ValueError(f"Checkpoint `mot` contains no `{prefix}*` weights")
    step_value = payload.get("step")
    step = None if step_value is None else int(step_value)
    return {
        "checkpoint": str(checkpoint),
        "step": step,
        "tensor_counts": tensor_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a portable FastWAM native-cache checkpoint."
    )
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_checkpoint(args.checkpoint), sort_keys=True))


if __name__ == "__main__":
    main()
