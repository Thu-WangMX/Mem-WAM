"""Strict-online train-free multi-resolution latent-kernel segmentation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _normalize_rows(values: torch.Tensor) -> torch.Tensor:
    return F.normalize(values.float(), dim=1, eps=1e-12)


def _kernel_scores(features: torch.Tensor) -> list[float | None]:
    values = _normalize_rows(features)
    scores: list[float | None] = [None] * len(values)
    for boundary in range(2, len(values) - 1):
        left = values[boundary - 2 : boundary]
        right = values[boundary : boundary + 2]
        left_coherence = float(left[0] @ left[1])
        right_coherence = float(right[0] @ right[1])
        cross_similarity = float((left @ right.T).mean())
        gap = max(0.0, 0.5 * (left_coherence + right_coherence) - cross_similarity)
        persistence = max(0.0, min(1.0, 0.5 * (right_coherence + 1.0)))
        scores[boundary] = gap * persistence**0.5
    return scores


def _causal_ranks(
    scores: list[float | None], *, history_window: int
) -> list[float | None]:
    ranks: list[float | None] = []
    for boundary, score in enumerate(scores):
        if score is None:
            ranks.append(None)
            continue
        prior = [
            float(value)
            for value in scores[max(0, boundary - history_window) : boundary]
            if value is not None
        ]
        if len(prior) < 3:
            ranks.append(None)
        else:
            ranks.append(
                (1.0 + sum(value <= float(score) for value in prior))
                / (len(prior) + 1.0)
            )
    return ranks


class OnlineLatentKernelRegimeSegmenter:
    """Online replay of the frozen VAE-only diagnostic selector."""

    def __init__(
        self,
        *,
        anchor_frames: int = 2,
        min_segment: int = 4,
        max_segment: int = 8,
        rank_threshold: float = 0.75,
        history_window: int = 8,
    ) -> None:
        self.anchor_frames = int(anchor_frames)
        self.min_segment = int(min_segment)
        self.max_segment = int(max_segment)
        self.rank_threshold = float(rank_threshold)
        self.history_window = int(history_window)
        self.reset()

    def reset(self) -> None:
        self._latents: list[torch.Tensor] = []
        self._open_start = self.anchor_frames
        self._last_arrival = -1

    def observe(self, *, decision: int, latent: torch.Tensor) -> None:
        decision = int(decision)
        if decision != len(self._latents):
            raise ValueError(
                f"latent decisions must be sequential: {decision} != {len(self._latents)}"
            )
        value = torch.as_tensor(latent).detach().float().cpu()
        if value.ndim == 5 and value.shape[0] == 1:
            value = value[0]
        if tuple(value.shape) != (48, 1, 24, 20):
            raise ValueError(f"expected [48,1,24,20], got {tuple(value.shape)}")
        self._latents.append(value)

    def _scores_and_ranks(self) -> tuple[list[float | None], list[float | None]]:
        values = torch.stack(self._latents, dim=0)
        spatial = values[:, :, 0]
        views = (
            spatial.flatten(1),
            spatial.mean(dim=(-1, -2)),
            F.normalize(spatial, dim=1).flatten(1),
        )
        per_view = [_kernel_scores(view) for view in views]
        fused: list[float | None] = []
        for candidates in zip(*per_view):
            finite = [float(value) for value in candidates if value is not None]
            fused.append(None if not finite else sum(finite) / len(finite))
        return fused, _causal_ranks(fused, history_window=self.history_window)

    def arrive_planning(self, *, decision: int) -> tuple[int, int] | None:
        decision = int(decision)
        if decision != len(self._latents) - 1:
            raise ValueError("planning arrival must follow the current latent observation")
        if decision <= self._last_arrival:
            raise ValueError("planning arrivals must be strictly increasing")
        self._last_arrival = decision
        start = self._open_start
        if decision < start + self.max_segment:
            return None
        scores, ranks = self._scores_and_ranks()
        candidates = range(start + self.min_segment, start + self.max_segment)
        eligible = [
            boundary
            for boundary in candidates
            if boundary + 1 <= decision
            and scores[boundary] is not None
            and float(scores[boundary]) > 1e-8
            and ranks[boundary] is not None
            and float(ranks[boundary]) >= self.rank_threshold
        ]
        if eligible:
            end = max(
                eligible,
                key=lambda boundary: (
                    float(scores[boundary]),
                    float(ranks[boundary]),
                    -abs((boundary - start) - 6),
                ),
            )
        else:
            end = start + self.max_segment
        closed = (start, end)
        if not self.min_segment <= closed[1] - closed[0] <= self.max_segment:
            raise RuntimeError("latent-kernel selector produced an invalid segment")
        if decision - closed[1] > 4:
            raise RuntimeError("latent-kernel selector exceeded recent4 delay")
        self._open_start = closed[1]
        return closed
