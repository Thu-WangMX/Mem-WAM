from __future__ import annotations

import pytest
import torch
from torch import nn

from fastwam.memory.multilayer_spatial_feature import (
    FEATURE_LAYERS,
    FEATURE_REGIONS,
    build_region_masks,
    capture_spatial_features,
)
from fastwam.memory.wam_embedding_feature import capture_last_frame_feature


class _FakeBlock(nn.Module):
    def __init__(self, layer: int):
        super().__init__()
        self.layer = layer

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + float(self.layer + 1) / 100.0


class _FakeVideoExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_FakeBlock(layer) for layer in range(30)])
        self.use_gradient_checkpointing = True
        self.forward_count = 0

    def forward(self, *, x, timestep, context, context_mask, action, **kwargs):
        self.forward_count += 1
        batch, _, frames, height, width = x.shape
        assert (height, width) == (24, 20)
        grid_h, grid_w, hidden_dim = 12, 10, 3072
        spatial = torch.arange(grid_h * grid_w, device=x.device, dtype=x.dtype)
        spatial = spatial[:, None].expand(-1, hidden_dim)
        hidden = spatial.repeat(frames, 1).unsqueeze(0).expand(batch, -1, -1)
        for block in self.blocks:
            hidden = block(hidden)
        return hidden


class _FakeModel:
    def __init__(self):
        self.video_expert = _FakeVideoExpert()
        self.device = torch.device("cpu")
        self.torch_dtype = torch.float32


def _inputs():
    return {
        "latents": torch.ones((48, 2, 24, 20), dtype=torch.float32),
        "video_context": torch.ones((1, 3, 4), dtype=torch.float32),
        "video_context_mask": torch.ones((1, 3), dtype=torch.bool),
    }


def test_region_masks_partition_fixed_robotwin_mosaic():
    masks = build_region_masks(grid_h=12, grid_w=10)

    assert tuple(masks) == FEATURE_REGIONS
    assert int(masks["global"].sum()) == 120
    assert int(masks["left_wrist"].sum()) == 20
    assert int(masks["right_wrist"].sum()) == 20
    assert int(masks["head"].sum()) == 80
    assert not torch.logical_and(masks["left_wrist"], masks["right_wrist"]).any()
    assert torch.equal(
        masks["left_wrist"] | masks["right_wrist"] | masks["head"],
        masks["global"],
    )


def test_capture_returns_all_layer_region_vectors_from_one_forward():
    model = _FakeModel()

    output = capture_spatial_features(model, **_inputs())

    assert tuple(output) == FEATURE_LAYERS
    assert all(tuple(regions) == FEATURE_REGIONS for regions in output.values())
    assert all(
        vector.shape == (3072,)
        and torch.isfinite(vector).all()
        and torch.allclose(vector.norm(), torch.tensor(1.0), atol=1e-5)
        for regions in output.values()
        for vector in regions.values()
    )
    assert model.video_expert.forward_count == 1
    assert model.video_expert.use_gradient_checkpointing is True
    assert all(not block._forward_hooks for block in model.video_expert.blocks)


def test_block29_global_matches_existing_last_frame_capture():
    model = _FakeModel()

    multilayer = capture_spatial_features(model, **_inputs())[29]["global"]
    existing = capture_last_frame_feature(model, layer_index=29, **_inputs())

    torch.testing.assert_close(multilayer, existing, atol=2e-5, rtol=2e-4)


def test_capture_rejects_non_native_latent_grid():
    model = _FakeModel()
    inputs = _inputs()
    inputs["latents"] = torch.ones((48, 2, 12, 10), dtype=torch.float32)

    with pytest.raises(ValueError, match="24x20"):
        capture_spatial_features(model, **inputs)
