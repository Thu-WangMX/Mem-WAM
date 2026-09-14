"""Feature extraction from a frozen initialized FastWAM video expert."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT


def load_episode_text_context(
    cache_root: str | Path,
    dataset_root: str | Path,
    episode: int,
) -> dict[str, Any]:
    """Resolve the exact cached prompt selected by an episode's task index."""

    cache_root = Path(cache_root).expanduser().resolve()
    dataset_root = Path(dataset_root).expanduser().resolve()
    episode = int(episode)
    parquet_path = (
        dataset_root
        / "data"
        / "chunk-000"
        / f"episode_{episode:06d}.parquet"
    )
    task_indices = set(
        int(value)
        for value in pq.read_table(parquet_path, columns=["task_index"])
        .column("task_index")
        .to_pylist()
    )
    if len(task_indices) != 1:
        raise ValueError(
            f"episode {episode} must use exactly one task index, got {task_indices}"
        )
    task_index = next(iter(task_indices))
    tasks = {}
    tasks_path = dataset_root / "meta" / "tasks.jsonl"
    for line in tasks_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        tasks[int(record["task_index"])] = str(record["task"])
    if task_index not in tasks:
        raise KeyError(f"task index {task_index} is absent from {tasks_path}")
    prompt = DEFAULT_PROMPT.format(task=tasks[task_index])
    hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    source = cache_root / f"{hashed}.t5_len128.wan22ti2v5b.pt"
    if not source.is_file():
        raise FileNotFoundError(
            f"episode {episode} text cache does not match task index {task_index}: {source}"
        )
    payload = torch.load(source, map_location="cpu", weights_only=True)
    context = torch.as_tensor(payload["context"]).clone()
    mask = torch.as_tensor(payload["mask"], dtype=torch.bool).clone()
    if context.ndim != 2 or mask.shape != (context.shape[0],):
        raise ValueError(f"malformed cached text context: {source}")
    context[~mask] = 0.0
    mask = torch.ones_like(mask)
    return {
        "context": context.unsqueeze(0),
        "mask": mask.unsqueeze(0),
        "prompt": prompt,
        "task_index": task_index,
        "source": source,
    }


@torch.inference_mode()
def capture_last_frame_feature(
    model: Any,
    *,
    latents: torch.Tensor,
    video_context: torch.Tensor,
    video_context_mask: torch.Tensor,
    layer_index: int = -1,
) -> torch.Tensor:
    """Return the normalized pooled hidden state for the newest causal frame."""

    if latents.ndim != 4 or latents.shape[1] < 1:
        raise ValueError("latents must have shape [C,T,H,W] with T >= 1")
    if video_context.ndim != 3 or video_context.shape[0] != 1:
        raise ValueError("video_context must have shape [1,L,D]")
    if video_context_mask.ndim != 2 or video_context_mask.shape[0] != 1:
        raise ValueError("video_context_mask must have shape [1,L]")

    video_expert = model.video_expert
    blocks = video_expert.blocks
    resolved_layer = int(layer_index)
    if resolved_layer < 0:
        resolved_layer += len(blocks)
    if resolved_layer < 0 or resolved_layer >= len(blocks):
        raise ValueError(f"layer_index {layer_index} is out of range")

    captured: list[torch.Tensor] = []

    def save_hidden(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise RuntimeError("video block hook did not receive [B,S,D] tokens")
        captured.append(hidden.detach())

    handle = blocks[resolved_layer].register_forward_hook(save_hidden)
    checkpointing = bool(getattr(video_expert, "use_gradient_checkpointing", False))
    if hasattr(video_expert, "use_gradient_checkpointing"):
        video_expert.use_gradient_checkpointing = False
    try:
        video = latents.unsqueeze(0).to(
            device=model.device, dtype=model.torch_dtype
        )
        frame_count = int(video.shape[2])
        timestep = torch.zeros(
            (1, frame_count), device=model.device, dtype=model.torch_dtype
        )
        video_expert(
            x=video,
            timestep=timestep,
            context=video_context.to(
                device=model.device, dtype=model.torch_dtype
            ),
            context_mask=video_context_mask.to(device=model.device),
            action=None,
            fuse_vae_embedding_in_latents=True,
        )
    finally:
        handle.remove()
        if hasattr(video_expert, "use_gradient_checkpointing"):
            video_expert.use_gradient_checkpointing = checkpointing

    if len(captured) != 1:
        raise RuntimeError(
            f"expected exactly one layer hook result, received {len(captured)}"
        )
    hidden = captured[0]
    if hidden.shape[0] != 1 or hidden.shape[1] % frame_count:
        raise RuntimeError("captured tokens cannot be divided into causal frames")
    tokens_per_frame = int(hidden.shape[1]) // frame_count
    pooled = hidden[0, -tokens_per_frame:].float().mean(dim=0)
    if not bool(torch.isfinite(pooled).all()) or float(pooled.norm().item()) == 0.0:
        raise RuntimeError("captured WAM feature is zero or non-finite")
    return F.normalize(pooled, dim=0).cpu()
