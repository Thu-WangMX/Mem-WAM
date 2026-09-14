from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from scripts.build_putback_four_phase_latents import (
    LATENT_SCHEMA,
    build_four_phase_episode,
    episodes_for_rank,
    validate_existing_episode,
    load_registered_vae,
)


def _fake_encode(video: torch.Tensor) -> torch.Tensor:
    assert video.ndim == 4 and video.shape[0] == 3
    latent_count = (int(video.shape[1]) - 1) // 4 + 1
    output = torch.zeros((48, latent_count, 24, 20), dtype=torch.bfloat16)
    output[0, :, 0, 0] = video[0, ::4, 0, 0][:latent_count]
    return output


def test_build_four_phase_episode_preserves_each_native_stream():
    requested: list[int] = []

    def mosaic_at(frame: int) -> torch.Tensor:
        requested.append(frame)
        return torch.full((3, 8, 8), float(frame))

    payload = build_four_phase_episode(
        episode=7,
        length=37,
        mosaic_at=mosaic_at,
        encode_video=_fake_encode,
        source_hdf5_sha256="source-sha",
        vae_sha256="vae-sha",
    )

    assert payload["schema_version"] == LATENT_SCHEMA
    assert payload["episode"] == 7
    assert payload["phase_offsets"] == [0, 4, 8, 12]
    assert payload["source_rgb_stride"] == 4
    assert payload["video_expert_frame_stride"] == 16
    assert payload["mosaic"] == "wrists_top_head_bottom_384x320"
    assert payload["phase0_equivalence_pending"] is True
    assert payload["phases"]["0"]["frame_indices"].tolist() == [0, 16, 32]
    assert payload["phases"]["4"]["frame_indices"].tolist() == [4, 20, 36]
    assert payload["phases"]["8"]["frame_indices"].tolist() == [8, 24]
    assert payload["phases"]["12"]["frame_indices"].tolist() == [12, 28]
    assert payload["phases"]["0"]["latents"].shape == (3, 48, 1, 24, 20)
    assert payload["phases"]["4"]["latents"][2, 0, 0, 0, 0].item() == 36
    assert requested == list(range(0, 37, 4))


def test_rank_assignment_is_complete_and_invalid_ranks_are_rejected():
    episodes = list(range(50))
    assigned = [episodes_for_rank(episodes, rank, 8) for rank in range(8)]

    assert assigned[0] == [0, 8, 16, 24, 32, 40, 48]
    assert sorted(value for shard in assigned for value in shard) == episodes
    with pytest.raises(ValueError, match="rank"):
        episodes_for_rank(episodes, 8, 8)
    with pytest.raises(ValueError, match="world_size"):
        episodes_for_rank(episodes, 0, 0)


def test_existing_episode_validation_checks_identity_hashes_and_content(tmp_path):
    path = tmp_path / "episode_000007.pt"
    payload = build_four_phase_episode(
        episode=7,
        length=37,
        mosaic_at=lambda frame: torch.full((3, 8, 8), float(frame)),
        encode_video=_fake_encode,
        source_hdf5_sha256="source-sha",
        vae_sha256="vae-sha",
    )
    torch.save(payload, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    assert validate_existing_episode(
        path,
        episode=7,
        length=37,
        source_hdf5_sha256="source-sha",
        vae_sha256="vae-sha",
        expected_sha256=digest,
    )
    assert not validate_existing_episode(
        path,
        episode=7,
        length=37,
        source_hdf5_sha256="wrong",
        vae_sha256="vae-sha",
        expected_sha256=digest,
    )
    assert not validate_existing_episode(
        path,
        episode=7,
        length=37,
        source_hdf5_sha256="source-sha",
        vae_sha256="vae-sha",
        expected_sha256="wrong",
    )


def test_registered_vae_loader_receives_string_path_not_pathlib(tmp_path):
    vae_path = tmp_path / "Wan2.2_VAE.safetensors"
    vae_path.write_bytes(b"fake")
    observed = {}

    class FakeVAE:
        def eval(self):
            return self

    def strict_legacy_loader(file_path, model_name, **kwargs):
        # Reproduce FastWAM's legacy loader contract from helpers/io.py.
        assert file_path.endswith(".safetensors")
        observed.update(path=file_path, model_name=model_name, kwargs=kwargs)
        return FakeVAE()

    model = load_registered_vae(
        vae_path, device=torch.device("cuda:3"), loader=strict_legacy_loader
    )

    assert isinstance(observed["path"], str)
    assert observed["model_name"] == "wan_video_vae"
    assert observed["kwargs"]["device"] == "cuda:3"
    assert model.eval() is model
