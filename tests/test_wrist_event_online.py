from __future__ import annotations

import json
import numpy as np
import torch

from experiments.robotwin.fastwam_policy.deploy_policy import WorldActionRobotWinPolicy
from fastwam.memory.native_cache import CacheUnit, LayerwiseMemoryState
from fastwam.memory.wrist_event import (
    OnlineWristEventSegmenter,
    causal_latent_window,
)


def test_online_event_segmenter_uses_prior_forecast_at_arrival_and_is_atomic():
    segmenter = OnlineWristEventSegmenter(threshold=0.5, min_segment=2, max_segment=8)

    arrival0 = segmenter.preview_arrival()
    assert arrival0.arrival == 0
    assert arrival0.close_range is None
    segmenter.commit_arrival(arrival0)
    segmenter.schedule_next(0.2)

    arrival1 = segmenter.preview_arrival()
    assert arrival1.arrival == 1
    assert arrival1.close_range is None
    segmenter.commit_arrival(arrival1)
    segmenter.schedule_next(0.8)

    arrival2 = segmenter.preview_arrival()
    assert arrival2.probability == 0.8
    assert arrival2.close_range == (0, 2)
    assert arrival2.reason == "wrist_event"
    assert segmenter.open_start == 0
    segmenter.commit_arrival(arrival2)
    assert segmenter.open_start == 2


def test_online_event_segmenter_forces_a_boundary_after_eight_missed_events():
    segmenter = OnlineWristEventSegmenter(threshold=0.5, min_segment=2, max_segment=8)
    decisions = []
    for _ in range(9):
        decision = segmenter.preview_arrival()
        segmenter.commit_arrival(decision)
        decisions.append(decision)
        segmenter.schedule_next(0.0)

    assert decisions[-1].arrival == 8
    assert decisions[-1].close_range == (0, 8)
    assert decisions[-1].reason == "max_length"


def test_causal_window_pads_only_with_first_available_latent():
    first = torch.full((1, 48, 1, 24, 20), 1.0)
    second = torch.full((1, 48, 1, 24, 20), 2.0)

    window = causal_latent_window([first, second], history=3)

    assert window.shape == (1, 3, 48, 24, 20)
    assert window[:, :, 0, 0, 0].tolist() == [[1.0, 1.0, 2.0]]


def _layerwise_state(endpoint: int) -> LayerwiseMemoryState:
    units = tuple(
        CacheUnit(kind="raw", endpoint=index, span=1, token_count=2)
        for index in range(endpoint + 1)
    )
    cache = (
        {
            "k": torch.zeros(1, 2 * len(units), 1, 2),
            "v": torch.zeros(1, 2 * len(units), 1, 2),
        },
    )
    return LayerwiseMemoryState(units=units, kv_cache=cache)


class _FakeWam:
    def __init__(self):
        self.device = torch.device("cpu")
        self.torch_dtype = torch.float32
        self.layerwise_block_memory = object()
        self.native_cache_compressor = None
        self.calls = []

    def infer_action(self, **kwargs):
        self.calls.append(kwargs)
        endpoint = int(kwargs["full_kv_frame_index"])
        state = _layerwise_state(endpoint)
        return {
            "action": torch.zeros(2, 14),
            "full_kv_cache": list(state.kv_cache),
            "native_cache_state": state,
            "current_observation_latent": torch.full((1, 48, 1, 24, 20), float(endpoint)),
        }


class _AlwaysEvent(torch.nn.Module):
    def forward(self, latents):
        return torch.full((latents.shape[0],), 2.0)


def test_policy_forecast_closes_memory_at_the_future_arrival(capsys):
    policy = WorldActionRobotWinPolicy.__new__(WorldActionRobotWinPolicy)
    policy.model = _FakeWam()
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
    policy.dynamic_surprise_online = False
    policy.wrist_event_online = True
    policy._wrist_event_segmenter = OnlineWristEventSegmenter(
        threshold=0.5, min_segment=2, max_segment=8
    )
    policy._wrist_event_predictor = _AlwaysEvent()
    policy._wrist_event_latents = []
    policy._wrist_event_history = 3
    policy._wrist_innovation_fn = lambda previous, current: {
        "left": (current - previous).abs().mean(),
        "right": (current - previous).abs().mean(),
        "combined": (current - previous).abs().mean(),
    }
    observation = {"joint_action": {"vector": np.zeros(14, dtype=np.float32)}}

    for replan in range(3):
        if replan:
            policy._temporal_frames.extend([frame, frame, frame])
        policy._replan_count = replan
        policy._infer_action_chunk(observation, "put the block back")

    assert policy.model.calls[0]["dynamic_surprise_online"] is True
    assert policy.model.calls[0]["dynamic_close_range"] is None
    assert policy.model.calls[1]["dynamic_close_range"] is None
    assert policy.model.calls[2]["dynamic_close_range"] == (0, 2)
    forecasts = [
        json.loads(line.split("FASTWAM_WRIST_EVENT_FORECAST ", 1)[1])
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("FASTWAM_WRIST_EVENT_FORECAST ")
    ]
    assert len(policy.model.calls) == 3
    assert len(forecasts) == 3
    assert forecasts[0]["source_wrist_innovation"] is None
    assert forecasts[0]["source_left_innovation"] is None
    assert forecasts[0]["source_right_innovation"] is None
    assert forecasts[1]["source_wrist_innovation"] == 1.0
    assert forecasts[2]["source_wrist_innovation"] == 1.0
