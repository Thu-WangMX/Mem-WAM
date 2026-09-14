"""Matched causal predictors for PCA-projected WAM feature streams."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class FeaturePredictor(nn.Module):
    """Predict the next multi-stream WAM feature from same-phase history."""

    def __init__(
        self,
        *,
        feature_dim: int,
        condition_dim: int,
        hidden_dim: int = 512,
        condition_embedding_dim: int = 128,
    ) -> None:
        super().__init__()
        if min(feature_dim, condition_dim, hidden_dim, condition_embedding_dim) <= 0:
            raise ValueError("all predictor dimensions must be positive")
        self.feature_dim = int(feature_dim)
        self.condition_dim = int(condition_dim)
        self.hidden_dim = int(hidden_dim)
        self.condition_embedding_dim = int(condition_embedding_dim)
        self.condition_mlp = nn.Sequential(
            nn.Linear(self.condition_dim, self.condition_embedding_dim),
            nn.SiLU(),
            nn.Linear(self.condition_embedding_dim, self.condition_embedding_dim),
        )
        self.recurrent = nn.GRU(
            self.feature_dim + self.condition_embedding_dim,
            self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.output = nn.Linear(self.hidden_dim, self.feature_dim)

    def forward(
        self,
        history: torch.Tensor,
        condition: torch.Tensor,
        *,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if history.ndim != 3 or history.shape[-1] != self.feature_dim:
            raise ValueError("history must have shape [B,T,feature_dim]")
        if condition.shape != (history.shape[0], self.condition_dim):
            raise ValueError("condition must have shape [B,condition_dim]")
        condition_feature = self.condition_mlp(condition)
        repeated = condition_feature[:, None].expand(-1, history.shape[1], -1)
        recurrent_input = torch.cat([history, repeated], dim=-1)
        if lengths is None:
            _, final_state = self.recurrent(recurrent_input)
        else:
            lengths = torch.as_tensor(lengths, dtype=torch.int64, device="cpu")
            if lengths.shape != (history.shape[0],) or bool(
                ((lengths <= 0) | (lengths > history.shape[1])).any()
            ):
                raise ValueError("lengths must contain one valid length per sequence")
            packed = pack_padded_sequence(
                recurrent_input,
                lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            _, final_state = self.recurrent(packed)
        return self.output(final_state[-1])


def huber_prediction_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.huber_loss(prediction, target, delta=1.0, reduction="mean")


@torch.no_grad()
def predict_phase_streams(
    model: FeaturePredictor,
    histories: Mapping[int, torch.Tensor],
    conditions: Mapping[int, torch.Tensor],
) -> dict[int, torch.Tensor]:
    """Run each phase independently, making recurrent reset explicit."""

    if set(histories) != set(conditions):
        raise ValueError("history and condition phase keys differ")
    return {
        int(phase): model(histories[phase], conditions[phase])
        for phase in histories
    }

