from __future__ import annotations

import torch

from fastwam.memory.control_information_probe import (
    FutureActionProbe,
    load_compact_control_probe,
    masked_action_huber_loss,
    save_compact_control_probe,
)


def _config():
    return {
        "feature_dim": 8,
        "proprio_dim": 3,
        "hidden_dim": 16,
        "proprio_hidden_dim": 4,
        "horizon": 4,
        "action_dim": 2,
    }


def _probe():
    torch.manual_seed(17)
    return FutureActionProbe(**_config())


def test_probe_output_shape_and_finite_values():
    prediction = _probe()(torch.randn(2, 8), torch.randn(2, 3))

    assert prediction.shape == (2, 4, 2)
    assert bool(torch.isfinite(prediction).all())


def test_masked_loss_ignores_terminal_padding():
    prediction = torch.zeros(2, 4, 2)
    target = prediction.clone()
    target[0, 3] = 1000
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)

    loss = masked_action_huber_loss(prediction, target, mask)

    assert loss.item() == 0.0


def test_masked_loss_counts_each_valid_action_dimension():
    prediction = torch.zeros(1, 2, 2)
    target = torch.tensor([[[1.0, 3.0], [100.0, 100.0]]])
    mask = torch.tensor([[True, False]])

    loss = masked_action_huber_loss(prediction, target, mask)

    # Huber(delta=1): 0.5 for error 1 and 2.5 for error 3.
    torch.testing.assert_close(loss, torch.tensor(1.5))


def test_compact_probe_round_trip_is_bit_exact_and_frozen(tmp_path):
    source = _probe()
    path = tmp_path / "probe.cipbin"
    save_compact_control_probe(
        path,
        source,
        config=_config(),
        report={"validation_mae": 0.25},
    )

    loaded, metadata = load_compact_control_probe(
        path, device=torch.device("cpu")
    )

    assert metadata["schema_version"] == "putback_control_information_probe_v1"
    assert metadata["report"] == {"validation_mae": 0.25}
    assert loaded.training is False
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
    for key, value in source.state_dict().items():
        torch.testing.assert_close(
            value, loaded.state_dict()[key], rtol=0, atol=0
        )
