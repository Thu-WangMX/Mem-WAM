"""Strict-online replay for multimodal rate-controlled L4--8 segmentation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .latent_kernel_regime_online import _causal_ranks, _kernel_scores
from .multimodal_rate_controlled import MultimodalRateSegment


ARM_DIMS = tuple(range(6)) + tuple(range(7, 13))
GRIPPER_DIMS = (6, 13)


@dataclass(frozen=True)
class PhysicalEvent:
    raw_frame: int
    planning_index: int
    known_at: int
    family: str
    score: float


class OnlineMultimodalRateSegmenter:
    def __init__(
        self,
        *,
        joint_delta_scale: list[float] | tuple[float, ...],
        motion_gate: float,
        planning_stride: int = 16,
    ) -> None:
        self.scale = np.asarray(joint_delta_scale, dtype=np.float32)
        if self.scale.shape != (12,) or np.any(self.scale <= 0):
            raise ValueError("joint_delta_scale must contain 12 positive values")
        self.motion_gate = float(motion_gate)
        self.planning_stride = int(planning_stride)
        self.reset()

    def reset(self) -> None:
        self._states: list[np.ndarray] = []
        self._latents: list[torch.Tensor] = []
        self._events: list[PhysicalEvent] = []
        self._motion: list[float] = []
        self._settled: list[float] = []
        self._stable_grippers: list[int | None] = [None, None]
        self._open_start = 2
        self._compressed_length = 0
        self._segment_count = 0
        self._last_planning = -1

    def _ceil_planning(self, raw_frame: int) -> int:
        return (int(raw_frame) + self.planning_stride - 1) // self.planning_stride

    @staticmethod
    def _gripper_state(value: float) -> int | None:
        return 0 if value <= 0.2 else (1 if value >= 0.8 else None)

    def observe_frame(self, *, frame: int, state: np.ndarray) -> None:
        frame = int(frame)
        if frame != len(self._states):
            raise ValueError(f"physical frame {frame} is not sequential")
        value = np.asarray(state, dtype=np.float32).reshape(-1)
        if value.shape != (14,):
            raise ValueError(f"expected 14-D qpos, got {value.shape}")
        self._states.append(value.copy())
        if frame == 0:
            motion = 0.0
        else:
            delta = (
                value[list(ARM_DIMS)] - self._states[-2][list(ARM_DIMS)]
            ) / np.maximum(self.scale, 1e-5)
            motion = float(np.sqrt(np.mean(delta * delta)))
        self._motion.append(motion)
        smooth = float(np.mean(self._motion[max(0, frame - 7) : frame + 1]))
        self._settled.append(1.0 / (1.0 + smooth))

        for slot, dimension in enumerate(GRIPPER_DIMS):
            reached = self._gripper_state(float(value[dimension]))
            if reached is None:
                continue
            stable = self._stable_grippers[slot]
            if stable is not None and reached != stable:
                planning = self._ceil_planning(frame)
                self._events.append(
                    PhysicalEvent(frame, planning, planning, "gripper", 1.0)
                )
            self._stable_grippers[slot] = reached

        candidate = frame - 4
        if candidate < 4:
            return
        neighborhood = self._settled[candidate - 4 : candidate + 5]
        prior = self._motion[max(1, candidate - 16) : candidate + 1]
        if len(neighborhood) != 9 or not prior:
            return
        if self._settled[candidate] < float(max(neighborhood)):
            return
        if float(max(prior)) <= self.motion_gate:
            return
        if any(
            event.family == "settled_peak"
            and abs(candidate - event.raw_frame) < 16
            for event in self._events
        ):
            return
        self._events.append(
            PhysicalEvent(
                candidate,
                self._ceil_planning(candidate),
                self._ceil_planning(frame),
                "settled_peak",
                float(
                    self._settled[candidate]
                    - min(neighborhood[0], neighborhood[-1])
                ),
            )
        )

    def _scores_and_ranks(self) -> tuple[list[float | None], list[float | None]]:
        values = torch.stack(self._latents).float()
        spatial = values[:, :, 0]
        views = (
            spatial.flatten(1),
            spatial.mean(dim=(-1, -2)),
            F.normalize(spatial, dim=1).flatten(1),
        )
        per_view = [_kernel_scores(view) for view in views]
        fused = []
        for candidates in zip(*per_view):
            finite = [float(value) for value in candidates if value is not None]
            fused.append(None if not finite else sum(finite) / len(finite))
        return fused, _causal_ranks(fused, history_window=8)

    def _physical_family(self, boundary: int, confirmed_at: int) -> str | None:
        matches = [
            event
            for event in self._events
            if abs((event.planning_index + 1) - boundary) <= 1
            and event.known_at <= confirmed_at
        ]
        if not matches:
            return None
        selected = min(
            matches,
            key=lambda event: (
                0 if event.family == "gripper" else 1,
                abs((event.planning_index + 1) - boundary),
                -event.score,
            ),
        )
        return selected.family

    def arrive_planning(
        self, *, decision: int, latent: torch.Tensor
    ) -> MultimodalRateSegment | None:
        decision = int(decision)
        if decision != len(self._latents) or decision <= self._last_planning:
            raise ValueError("planning decisions must arrive once and sequentially")
        value = torch.as_tensor(latent).detach().float().cpu()
        if value.ndim == 5 and value.shape[0] == 1:
            value = value[0]
        if tuple(value.shape) != (48, 1, 24, 20):
            raise ValueError(f"expected [48,1,24,20], got {tuple(value.shape)}")
        self._latents.append(value)
        self._last_planning = decision
        start = self._open_start
        if decision < start + 8:
            return None

        scores, ranks = self._scores_and_ranks()
        valid = []
        for boundary in range(start + 4, start + 8):
            length = boundary - start
            if (self._compressed_length + length) / (self._segment_count + 1) < 6.0:
                continue
            rank = ranks[boundary]
            if rank is None or boundary + 1 > decision:
                continue
            valid.append(
                (boundary, float(rank), self._physical_family(boundary, decision))
            )
        strong = [row for row in valid if row[1] >= 0.75]
        if strong:
            end, _, _ = max(
                strong,
                key=lambda row: (
                    float(scores[row[0]]),
                    row[1],
                    -abs((row[0] - start) - 6),
                ),
            )
            reason = "strong_latent"
        else:
            assisted = [row for row in valid if row[1] >= 0.5 and row[2] is not None]
            if assisted:
                end, _, family = max(assisted, key=lambda row: (row[0], row[1]))
                reason = f"latent_physical_{family}"
            else:
                end, reason = start + 8, "max8_fallback"
        self._compressed_length += end - start
        self._segment_count += 1
        self._open_start = end
        return MultimodalRateSegment(start, end, decision, reason)
