from pathlib import Path

import pytest
import torch

from scripts.validate_native_cache_checkpoint import validate_checkpoint


def _state_dict(value: float) -> dict[str, torch.Tensor]:
    return {"weight": torch.tensor([value])}


def test_validate_checkpoint_reports_required_parameter_groups(tmp_path: Path):
    checkpoint = tmp_path / "step_001000.pt"
    torch.save(
        {
            "step": 1000,
            "mot": {
                "mixtures.video.patch_embedding.weight": torch.tensor([1.0]),
                "mixtures.action.patch_embedding.weight": torch.tensor([2.0]),
            },
            "proprio_encoder": _state_dict(3.0),
            "layerwise_block_memory": {
                "slots": torch.ones(1, 32, 48),
            },
        },
        checkpoint,
    )

    summary = validate_checkpoint(checkpoint)

    assert summary["checkpoint"] == str(checkpoint.resolve())
    assert summary["step"] == 1000
    assert summary["tensor_counts"] == {
        "mot": 2,
        "proprio_encoder": 1,
        "layerwise_block_memory": 1,
    }


def test_validate_checkpoint_rejects_missing_native_cache_weights(tmp_path: Path):
    checkpoint = tmp_path / "fullkv.pt"
    torch.save(
        {
            "mot": {
                "mixtures.video.patch_embedding.weight": torch.tensor([1.0]),
                "mixtures.action.patch_embedding.weight": torch.tensor([2.0]),
            },
            "proprio_encoder": _state_dict(3.0),
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="layerwise_block_memory"):
        validate_checkpoint(checkpoint)
