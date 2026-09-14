from pathlib import Path

import torch

from fastwam.utils.compact_checkpoint import (
    build_portable_payload,
    export_portable_model_state,
    export_validate_and_prune,
    prune_old_training_states,
    validate_portable_checkpoint,
)


def test_build_portable_payload_keeps_trainable_weights_in_bfloat16():
    state = {
        "mot.mixtures.video.weight": torch.randn(2, 3, dtype=torch.float32),
        "mot.mixtures.action.weight": torch.randn(3, 2, dtype=torch.float32),
        "layerwise_block_memory.slots": torch.randn(1, 32, 48),
        "proprio_encoder.weight": torch.randn(4, 14, dtype=torch.float32),
        "proprio_encoder.bias": torch.randn(4, dtype=torch.float32),
    }

    payload = build_portable_payload(state, step=2000, inference_contract={"full_kv_cache": True})

    assert payload["step"] == 2000
    assert set(payload["mot"]) == {
        "mixtures.video.weight",
        "mixtures.action.weight",
    }
    assert all(t.dtype == torch.bfloat16 for t in payload["mot"].values())
    assert set(payload["layerwise_block_memory"]) == {"slots"}
    assert payload["layerwise_block_memory"]["slots"].dtype == torch.bfloat16
    assert all(t.dtype == torch.bfloat16 for t in payload["proprio_encoder"].values())


def test_validate_and_prune_only_completed_step_directories(tmp_path: Path):
    weights = tmp_path / "weights"
    state_root = tmp_path / "state"
    weights.mkdir()
    state_root.mkdir()

    portable = weights / "step_004000.pt"
    torch.save(
        {
            "step": 4000,
            "mot": {"mixtures.video.weight": torch.ones(1, dtype=torch.bfloat16)},
        },
        portable,
    )
    validate_portable_checkpoint(portable, expected_step=4000)

    for step in (2000, 4000):
        directory = state_root / f"step_{step:06d}"
        directory.mkdir()
        (directory / "trainer_state.json").write_text("{}")
        (directory / "pytorch_model_fsdp_0").mkdir()
        (directory / "optimizer_0").mkdir()
    unrelated = state_root / "manual_backup"
    unrelated.mkdir()

    removed = prune_old_training_states(state_root, keep=1)

    assert removed == [state_root / "step_002000"]
    assert not (state_root / "step_002000").exists()
    assert (state_root / "step_004000").exists()
    assert unrelated.exists()


def test_export_portable_model_state_is_atomic_and_validated(tmp_path: Path):
    destination = tmp_path / "weights" / "step_015000.pt"
    state = {
        "mot.mixtures.video.weight": torch.ones(2, dtype=torch.bfloat16),
        "layerwise_block_memory.slots": torch.ones(1, 8, 4),
    }

    result = export_portable_model_state(
        state,
        destination,
        step=15000,
        inference_contract={"full_kv_cache": True},
    )

    assert result == destination
    assert destination.is_file()
    assert not destination.with_name(destination.name + ".tmp").exists()
    validate_portable_checkpoint(destination, expected_step=15000)


def test_export_validates_before_pruning(monkeypatch, tmp_path: Path):
    calls = []
    output = tmp_path / "weights" / "step_002000.pt"

    monkeypatch.setattr(
        "fastwam.utils.compact_checkpoint.export_fsdp_checkpoint",
        lambda *args, **kwargs: calls.append("export") or output,
    )
    monkeypatch.setattr(
        "fastwam.utils.compact_checkpoint.validate_portable_checkpoint",
        lambda *args, **kwargs: calls.append("validate"),
    )
    monkeypatch.setattr(
        "fastwam.utils.compact_checkpoint.prune_old_training_states",
        lambda *args, **kwargs: calls.append("prune") or [],
    )

    export_validate_and_prune(
        dcp_dir=tmp_path / "dcp",
        output_path=output,
        state_root=tmp_path / "state",
        step=2000,
        inference_contract={"full_kv_cache": True},
        keep_training_states=1,
    )

    assert calls == ["export", "validate", "prune"]


def test_prune_recognizes_complete_deepspeed_zero2_states(tmp_path: Path):
    state_root = tmp_path / "state"
    state_root.mkdir()
    for step in (2000, 4000):
        directory = state_root / f"step_{step:06d}"
        model_dir = directory / "pytorch_model"
        model_dir.mkdir(parents=True)
        (directory / "trainer_state.json").write_text("{}")
        (directory / "latest").write_text("pytorch_model")
        (model_dir / "bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt").write_bytes(b"ok")

    removed = prune_old_training_states(state_root, keep=1)

    assert removed == [state_root / "step_002000"]
    assert (state_root / "step_004000").exists()
