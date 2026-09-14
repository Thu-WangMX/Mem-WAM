from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest
import torch

from fastwam.memory.multilayer_spatial_feature import FEATURE_LAYERS, FEATURE_REGIONS
from scripts.extract_putback_four_phase_wam_features import (
    FEATURE_BANK_SCHEMA,
    compare_phase0_features,
    extract_episode_features,
    validate_existing_feature_episode,
)


def _latent_payload() -> dict:
    phases = {}
    for phase in (0, 4, 8, 12):
        frames = torch.tensor([phase, phase + 16, phase + 32], dtype=torch.int64)
        latents = torch.zeros((3, 48, 1, 24, 20), dtype=torch.bfloat16)
        latents[:, 0, 0, 0, 0] = frames.to(torch.bfloat16)
        phases[str(phase)] = {"frame_indices": frames, "latents": latents}
    return {
        "schema_version": "putback_four_phase_latents_v1",
        "episode": 3,
        "phase_offsets": [0, 4, 8, 12],
        "phases": phases,
    }


def _capture(*, latents: torch.Tensor):
    value = float(latents[0, -1, 0, 0].item())
    return {
        layer: {
            region: torch.full((6,), value + layer + region_index)
            for region_index, region in enumerate(FEATURE_REGIONS)
        }
        for layer in FEATURE_LAYERS
    }


def test_episode_extraction_preserves_phase_order_warmup_and_raw_streams():
    payload = extract_episode_features(
        _latent_payload(),
        capture=_capture,
        initialization_fingerprint="init-fingerprint",
        feature_window=2,
    )

    assert payload["schema_version"] == FEATURE_BANK_SCHEMA
    assert payload["episode"] == 3
    assert payload["feature_layers"] == list(FEATURE_LAYERS)
    assert payload["feature_regions"] == list(FEATURE_REGIONS)
    assert payload["feature_dim"] == 6
    for phase in (0, 4, 8, 12):
        row = payload["phases"][str(phase)]
        assert row["features"].shape == (3, 5, 4, 6)
        assert row["warmup"].tolist() == [True, False, False]
        assert row["history_lengths"].tolist() == [1, 2, 2]
        assert row["frame_indices"].tolist() == [phase, phase + 16, phase + 32]


def test_phase0_equivalence_requires_matching_frames_and_block29_global():
    new = extract_episode_features(
        _latent_payload(),
        capture=_capture,
        initialization_fingerprint="init-fingerprint",
        feature_window=2,
    )
    phase0 = new["phases"]["0"]
    existing = {
        "features": phase0["features"][:, 4, 0].clone(),
        "decision_frame_indices": phase0["frame_indices"].clone(),
    }

    report = compare_phase0_features(new, existing, atol=2e-5, rtol=2e-4)
    assert report["pass"] is True
    assert report["compared"] == 3
    existing["features"][1, 0] += 1
    assert compare_phase0_features(new, existing, atol=2e-5, rtol=2e-4)["pass"] is False
    existing["decision_frame_indices"][1] = 99
    with pytest.raises(ValueError, match="frame"):
        compare_phase0_features(new, existing)


def test_existing_feature_episode_checks_fingerprint_hash_and_shape(tmp_path):
    path = tmp_path / "episode_003.pt"
    payload = extract_episode_features(
        _latent_payload(),
        capture=_capture,
        initialization_fingerprint="init-fingerprint",
        feature_window=2,
    )
    torch.save(payload, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    assert validate_existing_feature_episode(
        path,
        episode=3,
        initialization_fingerprint="init-fingerprint",
        expected_sha256=digest,
        expected_feature_dim=6,
    )
    assert not validate_existing_feature_episode(
        path,
        episode=3,
        initialization_fingerprint="wrong",
        expected_sha256=digest,
        expected_feature_dim=6,
    )


def test_feature_launcher_declares_exact_contract(tmp_path):
    root = Path(__file__).resolve().parents[1]
    launcher = root / "ops" / "extract_putback_four_phase_wam_features_8gpu.sh"
    latent = tmp_path / "latent"
    text = tmp_path / "text"
    dataset = tmp_path / "dataset"
    model_base = tmp_path / "models"
    for path in (latent, text, dataset, model_base):
        path.mkdir()
    (latent / "manifest.json").write_text(
        json.dumps({"complete": True, "phase0_equivalence_pending": True})
    )
    action_init = tmp_path / "action.pt"
    action_init.write_bytes(b"action")
    init_reference = tmp_path / "initialization.json"
    init_reference.write_text(
        json.dumps(
            {
                "initialization_fingerprint": "init-fingerprint",
                "model_source": "initialization",
                "policy_checkpoint": None,
            }
        )
    )
    old_bank = tmp_path / "old_bank"
    old_bank.mkdir()
    (old_bank / "bank_manifest.json").write_text(json.dumps({"complete": True}))
    runtime = Path(os.sys.executable).parent
    env = {
        **os.environ,
        "REPO_ROOT": str(root),
        "OUTPUT_DIR": str(tmp_path / "features"),
        "FOUR_PHASE_LATENTS": str(latent),
        "TEXT_CACHE": str(text),
        "DATASET_ROOT": str(dataset),
        "ACTION_INIT": str(action_init),
        "MODEL_BASE": str(model_base),
        "INIT_REFERENCE": str(init_reference),
        "PHASE0_BANK": str(old_bank),
        "RUNTIME_BIN": str(runtime),
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "PREFLIGHT_ONLY": "1",
    }

    result = subprocess.run(
        ["bash", str(launcher)], cwd=root, env=env, text=True, capture_output=True
    )

    assert result.returncode == 0, result.stderr
    assert "feature_layers=5,11,17,23,29" in result.stdout
    assert "feature_regions=global,left_wrist,right_wrist,head" in result.stdout
    assert "policy_checkpoint=null" in result.stdout
