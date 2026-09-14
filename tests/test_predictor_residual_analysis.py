from __future__ import annotations

import torch

from fastwam.memory.predictive_feature_model import FeaturePredictor
from scripts.analyze_putback_feature_predictor_residuals import predict_stream_residuals


def test_residual_analysis_returns_chronological_twenty_stream_rms():
    torch.manual_seed(3)
    model = FeaturePredictor(
        feature_dim=40, condition_dim=3, hidden_dim=12, condition_embedding_dim=5
    )
    rows = []
    for index, frame in enumerate((16, 20, 24)):
        rows.append({
            "episode": 0, "phase": frame % 16, "target_frame": frame,
            "history": torch.randn(index + 1, 40),
            "condition": torch.randn(3), "target": torch.randn(40),
        })
    result = predict_stream_residuals(
        model, rows, layers=5, regions=4, components=2, batch_size=2,
    )
    assert result["frame_indices"].tolist() == [16, 20, 24]
    assert result["residuals"].shape == (3, 20)
    assert torch.isfinite(result["residuals"]).all()
    assert torch.all(result["residuals"] >= 0)

