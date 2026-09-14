from __future__ import annotations

import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn

from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.memory.wam_embedding_feature import (
    capture_last_frame_feature,
    load_episode_text_context,
)


class IdentityBlock(nn.Module):
    def forward(self, tokens):
        return tokens


class FakeVideoExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([IdentityBlock()])
        self.use_gradient_checkpointing = True
        self.seen_frames = None

    def forward(
        self,
        *,
        x,
        timestep,
        context,
        context_mask,
        action,
        fuse_vae_embedding_in_latents,
    ):
        self.seen_frames = int(x.shape[2])
        tokens = []
        for frame in range(self.seen_frames):
            value = torch.zeros(2, 3, device=x.device, dtype=x.dtype)
            value[:, frame % 3] = float(frame + 1)
            tokens.append(value)
        hidden = torch.cat(tokens, dim=0).unsqueeze(0)
        hidden = self.blocks[-1](hidden)
        return hidden


class FakeModel:
    def __init__(self):
        self.video_expert = FakeVideoExpert()
        self.device = torch.device("cpu")
        self.torch_dtype = torch.float32


def test_capture_pools_only_latest_frame_tokens_and_restores_checkpointing():
    model = FakeModel()
    latents = torch.zeros(48, 3, 2, 2)
    feature = capture_last_frame_feature(
        model,
        latents=latents,
        video_context=torch.zeros(1, 2, 4),
        video_context_mask=torch.ones(1, 2, dtype=torch.bool),
    )

    assert torch.equal(feature, torch.tensor([0.0, 0.0, 1.0]))
    assert model.video_expert.seen_frames == 3
    assert model.video_expert.use_gradient_checkpointing is True


def test_capture_rejects_noncausal_empty_or_batched_input():
    model = FakeModel()

    for bad in (torch.zeros(48, 0, 2, 2), torch.zeros(2, 48, 3, 2, 2)):
        try:
            capture_last_frame_feature(
                model,
                latents=bad,
                video_context=torch.zeros(1, 2, 4),
                video_context_mask=torch.ones(1, 2, dtype=torch.bool),
            )
        except ValueError as error:
            assert "latents" in str(error)
        else:
            raise AssertionError("invalid latents were accepted")


def test_text_context_matches_episode_task_index_not_cache_listing_order(tmp_path):
    dataset = tmp_path / "dataset"
    cache = tmp_path / "cache"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "data" / "chunk-000").mkdir(parents=True)
    cache.mkdir()
    tasks = ["the complete put back instruction", "clean", "put back block", "success"]
    with (dataset / "meta" / "tasks.jsonl").open("w", encoding="utf-8") as stream:
        for index, task in enumerate(tasks):
            stream.write(json.dumps({"task_index": index, "task": task}) + "\n")
    pq.write_table(
        pa.table({"task_index": [0, 0], "frame_index": [0, 1]}),
        dataset / "data" / "chunk-000" / "episode_000040.parquet",
    )
    expected_hash = None
    for index, task in enumerate(tasks):
        prompt = DEFAULT_PROMPT.format(task=task)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if index == 0:
            expected_hash = hashed
        mask = torch.tensor([True, False])
        torch.save(
            {
                "context": torch.tensor([[float(index)], [99.0]]),
                "mask": mask,
            },
            cache / f"{hashed}.t5_len128.wan22ti2v5b.pt",
        )

    loaded = load_episode_text_context(cache, dataset, 40)

    assert loaded["prompt"] == DEFAULT_PROMPT.format(task=tasks[0])
    assert loaded["source"].name.startswith(expected_hash)
    assert torch.equal(loaded["context"], torch.tensor([[[0.0], [0.0]]]))
    assert torch.equal(loaded["mask"], torch.tensor([[True, True]]))
