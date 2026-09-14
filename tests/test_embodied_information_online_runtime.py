from __future__ import annotations

import torch

from fastwam.evaluation.embodied_information_online import (
    EmbodiedInformationOnlineRuntime,
    project_captured_features,
)


class _FakePolicyModel:
    device = torch.device("cpu")
    torch_dtype = torch.float32

    def _encode_input_image_latents_tensor(self, *, input_image, tiled=False):
        assert input_image.shape[2] % 4 == 1
        value = float(input_image.shape[2])
        return torch.full((1, 48, 1, 24, 20), value)


class _FakeFeatureModel:
    device = torch.device("cpu")
    torch_dtype = torch.float32


class _FakePredictor(torch.nn.Module):
    feature_dim = 1280
    condition_dim = 238

    def forward(self, history, condition, *, lengths=None):
        assert history.shape[0] == condition.shape[0] == 1
        assert lengths.tolist() == [history.shape[1]]
        return history[:, -1]


def _capture(_model, *, latents, video_context, video_context_mask):
    assert latents.shape[0] == 48
    value = float(latents.shape[1])
    return {
        layer: {
            region: torch.full((3,), value + layer_index + region_index)
            for region_index, region in enumerate(
                ("global", "left_wrist", "right_wrist", "head")
            )
        }
        for layer_index, layer in enumerate((5, 11, 17, 23, 29))
    }


def _artifacts():
    mean = torch.zeros(5, 4, 3)
    components = torch.eye(3).expand(5, 4, 3, 3).clone()
    pca = {
        "schema_version": "putback_wam_stream_pca_v1",
        "feature_dim": 3,
        "component_dim": 3,
        "mean": mean,
        "components": components,
        "projected_scale": torch.ones(5, 4, 3),
    }
    stats = {
        "depth_cap": 4,
        "median": torch.zeros(4, 4, 20),
        "mad_scale": torch.ones(4, 4, 20),
    }
    return pca, stats


def test_project_captured_features_preserves_locked_stream_order():
    pca, _ = _artifacts()
    projected = project_captured_features(_capture(
        None,
        latents=torch.zeros(48, 1, 24, 20),
        video_context=torch.zeros(1, 1, 1),
        video_context_mask=torch.ones(1, 1, dtype=torch.bool),
    ), pca)
    assert projected.shape == (5, 4, 3)
    assert projected[0, 0].tolist() == [1.0, 1.0, 1.0]
    assert projected[4, 3].tolist() == [8.0, 8.0, 8.0]


def test_runtime_is_strictly_forward_and_closes_only_at_planning_arrival():
    pca, stats = _artifacts()
    runtime = EmbodiedInformationOnlineRuntime(
        policy_model=_FakePolicyModel(),
        initialization_model=_FakeFeatureModel(),
        pca_artifact=pca,
        predictor=_FakePredictor(),
        contextual_statistics=stats,
        selector_config={
            "information_budget": 1e9,
            "top_k": 5,
            "clip_z": 8.0,
            "detector_stride": 4,
            "min_information_units": 4,
            "max_units": 24,
            "initial_group_start": 0,
            "gripper_dimensions": [6, 13],
            "gripper_threshold": 0.5,
        },
        capture_fn=_capture,
        feature_window=8,
    )
    runtime.set_prompt(
        torch.zeros(1, 2, 3), torch.ones(1, 2, dtype=torch.bool)
    )
    image = torch.zeros(1, 3, 16, 16)
    actions = []
    for frame in range(0, 20, 4):
        proprio = torch.zeros(14)
        if frame >= 16:
            proprio[6] = 1.0
        event = runtime.observe(
            frame=frame,
            image=image,
            proprio=proprio,
            executed_actions=actions,
        )
        if frame < 16:
            assert event is None
        actions.extend([torch.zeros(14) for _ in range(4)])

    assert runtime.arrive_planning(frame=0) is None
    assert runtime.arrive_planning(frame=16) is None
    # The event confirmed at frame 16 is coalesced because a one-decision
    # memory group is forbidden; it is never backdated to decision zero.
    assert runtime.aligner.retroactive_boundary_count == 0
    assert runtime.boundary_state.embodied_boundary_count == 1

