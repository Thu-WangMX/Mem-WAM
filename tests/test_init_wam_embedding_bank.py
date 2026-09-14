from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from scripts.extract_putback_init_wam_embedding_bank import (
    BANK_SCHEMA,
    episodes_for_rank,
    validate_existing_entry,
)


def test_round_robin_assignment_is_complete_disjoint_and_deterministic():
    episodes = list(range(50))
    assignments = [episodes_for_rank(episodes, rank, 8) for rank in range(8)]

    assert assignments[0] == [0, 8, 16, 24, 32, 40, 48]
    assert assignments[7] == [7, 15, 23, 31, 39, 47]
    assert sorted(value for rows in assignments for value in rows) == episodes
    assert sum(len(rows) for rows in assignments) == len(
        set(value for rows in assignments for value in rows)
    )
    with pytest.raises(ValueError):
        episodes_for_rank(episodes, 8, 8)
    with pytest.raises(ValueError):
        episodes_for_rank(episodes, 0, 0)


def test_existing_entry_requires_every_identity_and_content_check(tmp_path):
    feature_path = tmp_path / "episode_040.pt"
    metadata_path = tmp_path / "episode_040.json"
    features = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    frames = torch.tensor([0, 16, 32, 48], dtype=torch.int64)
    torch.save({"features": features, "decision_frame_indices": frames}, feature_path)
    digest = hashlib.sha256(feature_path.read_bytes()).hexdigest()
    metadata = {
        "schema_version": BANK_SCHEMA,
        "episode": 40,
        "initialization_fingerprint": "init-42",
        "decision_frame_indices": frames.tolist(),
        "feature_dim": 3,
        "feature_sha256": digest,
    }
    metadata_path.write_text(json.dumps(metadata))

    assert validate_existing_entry(
        feature_path,
        metadata_path,
        episode=40,
        initialization_fingerprint="init-42",
        expected_frame_indices=frames.tolist(),
        expected_feature_dim=3,
    )

    metadata["initialization_fingerprint"] = "wrong"
    metadata_path.write_text(json.dumps(metadata))
    assert not validate_existing_entry(
        feature_path,
        metadata_path,
        episode=40,
        initialization_fingerprint="init-42",
        expected_frame_indices=frames.tolist(),
        expected_feature_dim=3,
    )
    metadata["initialization_fingerprint"] = "init-42"
    metadata_path.write_text(json.dumps(metadata))
    feature_path.write_bytes(feature_path.read_bytes() + b"corrupt")
    assert not validate_existing_entry(
        feature_path,
        metadata_path,
        episode=40,
        initialization_fingerprint="init-42",
        expected_frame_indices=frames.tolist(),
        expected_feature_dim=3,
    )
