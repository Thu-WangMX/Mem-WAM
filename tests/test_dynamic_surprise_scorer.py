from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from fastwam.memory.dynamic_surprise_scorer import score_transition
from scripts.generate_putback_surprise_manifest import _score_transition


class _VideoExpert:
    def __init__(self):
        self.last_x = None

    def pre_dit(self, *, x, timestep, context, context_mask, **kwargs):
        self.last_x = x
        frames = x.shape[2]
        return {
            "tokens": torch.zeros(1, frames, 1),
            "freqs": torch.zeros(frames, 1),
            "t_mod": torch.zeros(1, frames, 1),
            "context": context,
            "context_mask": context_mask,
            "t": torch.zeros(1, frames, 1),
            "meta": {"grid_size": (frames, 1, 1), "tokens_per_frame": 1},
        }

    def post_dit(self, tokens, pre):
        return torch.zeros_like(self.last_x[:, :, -1:])


class _ActionExpert:
    def pre_dit(self, *, action_tokens, timestep, context, context_mask):
        return {
            "tokens": action_tokens,
            "freqs": torch.zeros(action_tokens.shape[1], 1),
            "context": context,
            "context_mask": context_mask,
            "t_mod": torch.zeros(1, action_tokens.shape[1], 1),
        }


class _FakeModel:
    def __init__(self):
        self.device = torch.device("cpu")
        self.torch_dtype = torch.float32
        self.video_expert = _VideoExpert()
        self.action_expert = _ActionExpert()
        self.seen_groups = None
        self.inputs = {
            "history_latents": torch.zeros(1, 2, 1, 1, 1, 1),
            "action": torch.zeros(1, 2, 1),
            "context": torch.zeros(1, 1, 1),
            "context_mask": torch.ones(1, 1, dtype=torch.bool),
            "video_context": torch.zeros(1, 1, 1),
            "video_context_mask": torch.ones(1, 1, dtype=torch.bool),
        }

    def build_inputs(self, batch):
        return self.inputs

    def _build_video_aligned_action_freqs(self, *, action_seq_len, **kwargs):
        return torch.zeros(action_seq_len, 1)

    def _run_layerwise_memory_training_transformer(
        self, *, video_pre, memory_groups, **kwargs
    ):
        self.seen_groups = memory_groups
        return {"video": video_pre["tokens"]}, SimpleNamespace(noisy_range=(2, 3))


def _shared_score(model: _FakeModel):
    return score_transition(
        model,
        history_latents=torch.zeros(1, 1, 2, 1, 1),
        actual_latent=torch.full((1, 1, 1, 1, 1), 2.0),
        action=model.inputs["action"],
        context=model.inputs["context"],
        context_mask=model.inputs["context_mask"],
        video_context=model.inputs["video_context"],
        video_context_mask=model.inputs["video_context_mask"],
        memory_groups=((0, 1),),
        sigma=1.0,
        noise_seed=0,
    )


def test_shared_scorer_is_deterministic_and_preserves_memory_groups():
    model = _FakeModel()

    first = _shared_score(model)
    second = _shared_score(model)

    assert first == second
    assert first["normalized_l1"] == pytest.approx(0.22950196, abs=1e-6)
    assert first["cosine_distance"] == pytest.approx(0.0, abs=1e-6)
    assert first["score"] == pytest.approx(0.16065137, abs=1e-6)
    assert model.seen_groups == ((0, 1),)


def test_manifest_adapter_matches_shared_tensor_interface():
    model = _FakeModel()
    sample = {
        "history_latents": torch.zeros(2, 1, 1, 1, 1),
        "history_memory_groups": torch.tensor([[0, 1]], dtype=torch.int64),
        "history_memory_group_count": torch.tensor(1, dtype=torch.int64),
    }
    next_sample = {"history_latents": torch.full((3, 1, 1, 1, 1), 2.0)}

    adapter = _score_transition(model, sample, next_sample, sigma=1.0, seed=0)
    shared = _shared_score(model)

    assert adapter == shared
