"""Strict-online self-calibrated settle/gripper segmentation with rate debt."""

from __future__ import annotations

import math

import numpy as np

from .physical_settle_rate_debt import (
    EVENT_REASON,
    FALLBACK_REASON,
    PhysicalSettleRateSegment,
)


ARM_DIMS = tuple(range(6)) + tuple(range(7, 13))
GRIPPER_DIMS = (6, 13)


def causal_percentile(values: list[float], window: int = 8) -> list[float]:
    output: list[float] = []
    for index, value in enumerate(values):
        history = np.asarray(values[max(0, index - window) : index], dtype=np.float64)
        history = history[np.isfinite(history)]
        if history.size < 2:
            output.append(0.5)
            continue
        less = float(np.sum(history < value))
        equal = float(np.sum(history == value))
        output.append((less + 0.5 * equal + 0.5) / (history.size + 1.0))
    return output


def _causal_scale(deltas: np.ndarray, index: int, window: int) -> np.ndarray:
    history = np.abs(deltas[max(0, index - window) : index])
    if history.shape[0] < 2:
        return np.ones(deltas.shape[1], dtype=np.float64)
    scale = np.median(history, axis=0)
    positive = scale[scale > 1e-8]
    fallback = float(np.median(positive)) if positive.size else 1.0
    return np.maximum(scale, max(fallback * 0.05, 1e-6))


def physical_evidence(states: np.ndarray, history_window: int = 8) -> list[float]:
    """Prefix-invariant move-to-settle/gripper evidence on the decision clock."""
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 14:
        raise ValueError(f"expected [T,14] qpos, got {states.shape}")
    delta = np.diff(states, axis=0, prepend=states[:1])
    speed = np.zeros(len(states), dtype=np.float64)
    gripper_change = np.zeros(len(states), dtype=np.float64)
    for index in range(len(states)):
        arm_scale = _causal_scale(delta[:, ARM_DIMS], index, history_window)
        arm_velocity = delta[index, ARM_DIMS] / arm_scale
        speed[index] = float(np.linalg.norm(arm_velocity) / math.sqrt(len(ARM_DIMS)))
        grip_scale = _causal_scale(delta[:, GRIPPER_DIMS], index, history_window)
        gripper_change[index] = float(
            np.linalg.norm(delta[index, GRIPPER_DIMS] / grip_scale)
        )

    speed_rank = np.asarray(causal_percentile(speed.tolist(), history_window))
    slowdown = np.zeros(len(states), dtype=np.float64)
    prior_motion_rank = np.zeros(len(states), dtype=np.float64)
    for index in range(2, len(states)):
        prior = float(np.max(speed[index - 2 : index]))
        slowdown[index] = max(
            0.0, math.log((prior + 1e-6) / (float(speed[index]) + 1e-6))
        )
        prior_motion_rank[index] = float(np.max(speed_rank[index - 2 : index]))
    slowdown_rank = np.asarray(causal_percentile(slowdown.tolist(), history_window))
    settle = np.sqrt(slowdown_rank * prior_motion_rank)
    settle[slowdown <= 1e-12] = 0.0

    gripper_rank = np.asarray(
        causal_percentile(gripper_change.tolist(), history_window)
    )
    gripper_rank[gripper_change <= 1e-12] = 0.0
    return np.maximum(settle, gripper_rank).astype(float).tolist()


class OnlinePhysicalSettleRateSegmenter:
    def __init__(
        self,
        *,
        history_window: int = 8,
        event_threshold: float = 0.8,
        target_mean_segment: float = 7.0,
        maximum_rate_debt: float = 5.0,
    ) -> None:
        self.history_window = int(history_window)
        self.event_threshold = float(event_threshold)
        self.target_mean_segment = float(target_mean_segment)
        self.maximum_rate_debt = float(maximum_rate_debt)
        if self.history_window != 8 or self.event_threshold != 0.8:
            raise ValueError("v1 fixes history_window=8 and event_threshold=0.8")
        if self.target_mean_segment != 7.0 or self.maximum_rate_debt != 5.0:
            raise ValueError("v1 fixes target L7 and rate-debt cap5")
        self.reset()

    def reset(self) -> None:
        self._states: list[np.ndarray] = []
        self._open_start = 2
        self._compressed_length = 0
        self._segment_count = 0

    def arrive_planning(
        self, *, decision: int, state: np.ndarray
    ) -> PhysicalSettleRateSegment | None:
        decision = int(decision)
        if decision != len(self._states):
            raise ValueError("planning decisions must arrive once and sequentially")
        value = np.asarray(state, dtype=np.float32).reshape(-1)
        if value.shape != (14,):
            raise ValueError(f"expected 14-D qpos, got {value.shape}")
        self._states.append(value.copy())
        start = self._open_start
        if decision < start + 8:
            return None
        if decision != start + 8:
            raise RuntimeError("segment confirmation clock skipped start+8")

        evidence = physical_evidence(
            np.asarray(self._states), history_window=self.history_window
        )
        eligible: list[tuple[int, float]] = []
        for boundary in range(start + 4, start + 8):
            length = boundary - start
            new_debt = self.target_mean_segment * (
                self._segment_count + 1
            ) - (self._compressed_length + length)
            if new_debt > self.maximum_rate_debt + 1e-12:
                continue
            score = float(evidence[boundary])
            if score >= self.event_threshold:
                eligible.append((boundary, score))

        if eligible:
            end, selected = max(
                eligible,
                key=lambda row: (
                    row[1],
                    -abs((row[0] - start) - 6),
                    row[0],
                ),
            )
            reason = EVENT_REASON
        else:
            end = start + 8
            selected = None
            reason = FALLBACK_REASON

        length = end - start
        self._compressed_length += length
        self._segment_count += 1
        debt = (
            self.target_mean_segment * self._segment_count
            - self._compressed_length
        )
        if debt > self.maximum_rate_debt + 1e-9:
            raise RuntimeError("rate-debt controller exceeded cap5")
        mean = self._compressed_length / self._segment_count
        self._open_start = end
        return PhysicalSettleRateSegment(
            start=start,
            end=end,
            confirmed_at=decision,
            reason=reason,
            evidence=selected,
            rate_debt=float(debt),
            cumulative_mean_length=float(mean),
        )
