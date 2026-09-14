"""Portable checkpoint export and conservative FSDP state retention."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Mapping

import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save


_STEP_DIR = re.compile(r"^step_(\d{6})$")


def _portable_tensor(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach().cpu()
    if tensor.is_floating_point():
        tensor = tensor.to(dtype=torch.bfloat16)
    return tensor.contiguous()


def build_portable_payload(
    model_state: Mapping[str, torch.Tensor],
    *,
    step: int,
    inference_contract: Mapping[str, object] | None = None,
) -> dict:
    mot = {
        key.removeprefix("mot."): _portable_tensor(value)
        for key, value in model_state.items()
        if key.startswith("mot.")
    }
    if not mot:
        raise ValueError("FSDP model state contains no `mot.*` weights.")

    payload = {
        "mot": mot,
        "step": int(step),
        "torch_dtype": "torch.bfloat16",
        "inference_contract": dict(inference_contract or {}),
    }
    proprio = {
        key.removeprefix("proprio_encoder."): _portable_tensor(value)
        for key, value in model_state.items()
        if key.startswith("proprio_encoder.")
    }
    if proprio:
        payload["proprio_encoder"] = proprio
    native_cache = {
        key.removeprefix("native_cache_compressor."): _portable_tensor(value)
        for key, value in model_state.items()
        if key.startswith("native_cache_compressor.")
    }
    if native_cache:
        payload["native_cache_compressor"] = native_cache
    layerwise_memory = {
        key.removeprefix("layerwise_block_memory."): _portable_tensor(value)
        for key, value in model_state.items()
        if key.startswith("layerwise_block_memory.")
    }
    if layerwise_memory:
        payload["layerwise_block_memory"] = layerwise_memory
    return payload


def validate_portable_checkpoint(path: str | os.PathLike, *, expected_step: int) -> None:
    checkpoint = Path(path)
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise ValueError(f"Portable checkpoint is missing or empty: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    if int(payload.get("step", -1)) != int(expected_step):
        raise ValueError(
            f"Portable checkpoint step mismatch: expected {expected_step}, "
            f"found {payload.get('step')}"
        )
    mot = payload.get("mot")
    if not isinstance(mot, dict) or not mot:
        raise ValueError("Portable checkpoint has no non-empty `mot` state.")
    wrong_dtype = [
        key
        for key, value in mot.items()
        if value.is_floating_point() and value.dtype != torch.bfloat16
    ]
    if wrong_dtype:
        raise ValueError(f"Portable checkpoint contains non-BF16 floating weights: {wrong_dtype[:5]}")


def export_portable_model_state(
    model_state: Mapping[str, torch.Tensor],
    output_path: str | os.PathLike,
    *,
    step: int,
    inference_contract: Mapping[str, object] | None = None,
) -> Path:
    """Atomically save a gathered FSDP full state without optimizer state."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        payload = build_portable_payload(
            model_state,
            step=step,
            inference_contract=inference_contract,
        )
        torch.save(payload, temporary)
        validate_portable_checkpoint(temporary, expected_step=step)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def export_fsdp_checkpoint(
    dcp_dir: str | os.PathLike,
    output_path: str | os.PathLike,
    *,
    step: int,
    inference_contract: Mapping[str, object] | None = None,
    tmp_root: str | os.PathLike = "/dev/shm",
) -> Path:
    source = Path(dcp_dir)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_destination = destination.with_name(destination.name + ".tmp")

    with tempfile.TemporaryDirectory(prefix="memorywam-dcp-", dir=tmp_root) as tmp_dir:
        consolidated = Path(tmp_dir) / "model_fp32.pt"
        dcp_to_torch_save(source, consolidated)
        state = torch.load(consolidated, map_location="cpu", mmap=True, weights_only=False)
        model_state = state.get("model", state)
        payload = build_portable_payload(
            model_state,
            step=step,
            inference_contract=inference_contract,
        )
        torch.save(payload, temporary_destination)
        validate_portable_checkpoint(temporary_destination, expected_step=step)
        os.replace(temporary_destination, destination)
    return destination


def _is_complete_training_state(path: Path) -> bool:
    if not (path / "trainer_state.json").is_file():
        return False
    fsdp_complete = (
        (path / "pytorch_model_fsdp_0").is_dir()
        and (path / "optimizer_0").is_dir()
    )
    deepspeed_dir = path / "pytorch_model"
    deepspeed_complete = (
        (path / "latest").is_file()
        and deepspeed_dir.is_dir()
        and any(deepspeed_dir.glob("*_optim_states.pt"))
    )
    return fsdp_complete or deepspeed_complete


def prune_old_training_states(
    state_root: str | os.PathLike,
    *,
    keep: int = 1,
) -> list[Path]:
    if keep < 1:
        raise ValueError("`keep` must be at least 1.")
    root = Path(state_root)
    candidates = []
    for path in root.iterdir():
        match = _STEP_DIR.fullmatch(path.name)
        if path.is_dir() and match and _is_complete_training_state(path):
            candidates.append((int(match.group(1)), path))
    candidates.sort()
    removed = []
    for _, path in candidates[:-keep]:
        shutil.rmtree(path)
        removed.append(path)
    return removed


def export_validate_and_prune(
    *,
    dcp_dir: str | os.PathLike,
    output_path: str | os.PathLike,
    state_root: str | os.PathLike,
    step: int,
    inference_contract: Mapping[str, object] | None,
    keep_training_states: int = 1,
    tmp_root: str | os.PathLike = "/dev/shm",
) -> tuple[Path, list[Path]]:
    portable = export_fsdp_checkpoint(
        dcp_dir,
        output_path,
        step=step,
        inference_contract=inference_contract,
        tmp_root=tmp_root,
    )
    validate_portable_checkpoint(portable, expected_step=step)
    removed = prune_old_training_states(state_root, keep=keep_training_states)
    return portable, removed
