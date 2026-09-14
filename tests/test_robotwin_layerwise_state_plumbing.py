from __future__ import annotations

import numpy as np
import torch

from experiments.robotwin.fastwam_policy.deploy_policy import WorldActionRobotWinPolicy
from fastwam.memory.native_cache import CacheUnit, LayerwiseMemoryState


def _layerwise_state(endpoint: int) -> LayerwiseMemoryState:
    units = tuple(
        CacheUnit(kind="raw", endpoint=index, span=1, token_count=2)
        for index in range(endpoint + 1)
    )
    tokens = 2 * len(units)
    cache = (
        {
            "k": torch.zeros(1, tokens, 1, 2),
            "v": torch.zeros(1, tokens, 1, 2),
        },
    )
    return LayerwiseMemoryState(units=units, kv_cache=cache)


class _LayerwiseFakeModel:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.torch_dtype = torch.float32
        self.native_cache_compressor = None
        self.layerwise_block_memory = object()
        self.calls: list[dict[str, object]] = []

    def infer_action(self, **kwargs):
        self.calls.append(kwargs)
        endpoint = int(kwargs["full_kv_frame_index"])
        state = _layerwise_state(endpoint)
        return {
            "action": torch.zeros(2, 14),
            "full_kv_cache": list(state.kv_cache),
            "native_cache_state": state,
        }


def test_robotwin_second_replan_passes_complete_layerwise_state(capsys):
    """Catches deployment that degrades layerwise state into bare K/V after replan one."""
    policy = WorldActionRobotWinPolicy.__new__(WorldActionRobotWinPolicy)
    policy.model = _LayerwiseFakeModel()
    frame = torch.zeros(1, 3, 16, 16)
    policy._build_robotwin_image_tensor = lambda observation: frame
    policy._normalize_state = lambda state: torch.zeros(14)
    policy._denormalize_action = lambda action: action.unsqueeze(0).numpy()
    policy._temporal_frames = []
    policy._temporal_subframes = 4
    policy._full_kv_cache = None
    policy._native_cache_state = None
    policy._full_kv_frame_index = 0
    policy.action_horizon = 2
    policy.negative_prompt = ""
    policy.text_cfg_scale = 1.0
    policy.num_inference_steps = 1
    policy.sigma_shift = None
    policy.seed = None
    policy.rand_device = "cpu"
    policy.tiled = False
    policy.timing_enabled = False
    policy._num_video_frames = 1
    policy._advance_policy_seed = False
    policy._replan_count = 0
    policy.episode_count = 0
    observation = {"joint_action": {"vector": np.zeros(14, dtype=np.float32)}}

    policy._infer_action_chunk(observation, "put the block back")
    first_state = policy._native_cache_state
    policy._temporal_frames.extend([frame, frame, frame])
    policy._replan_count = 1
    policy._infer_action_chunk(observation, "put the block back")

    assert policy.model.calls[0]["native_cache_state"] is None
    assert policy.model.calls[1]["native_cache_state"] is first_state
    assert "full_kv_cache" not in policy.model.calls[1]
    metrics_lines = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("FASTWAM_NATIVE_CACHE_METRICS ")
    ]
    assert len(metrics_lines) == 2
