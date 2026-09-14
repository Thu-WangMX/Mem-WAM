"""Causally align four-frame detector events to native planning observations."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


class OnlinePlanningBoundaryAligner:
    """Queue detector confirmations and close memory only at planning arrivals."""

    def __init__(self, *, replan_stride: int = 16,
                 minimum_segment_decisions: int = 2) -> None:
        self.replan_stride = int(replan_stride)
        self.minimum_segment_decisions = int(minimum_segment_decisions)
        if self.replan_stride <= 0 or self.minimum_segment_decisions < 1:
            raise ValueError("stride and minimum segment must be positive")
        self.last_boundary = 0
        self.last_decision = -1
        self._pending: dict[int, str] = {}
        self.retroactive_boundary_count = 0
        self.coalesced_confirmation_count = 0

    def observe_confirmation(self, *, confirmation_frame: int, reason: str, **_) -> None:
        confirmation_frame = int(confirmation_frame)
        decision_index = math.ceil(confirmation_frame / self.replan_stride)
        if decision_index <= self.last_decision:
            self.retroactive_boundary_count += 1
            raise ValueError("confirmation arrived after its aligned planning decision")
        prior = self._pending.get(decision_index)
        # Embodied anchors win when two detector events land in one action chunk.
        if prior is None or reason == "embodied_gripper_transition":
            self._pending[decision_index] = str(reason)
        else:
            self.coalesced_confirmation_count += 1

    def arrive_decision(self, frame: int) -> tuple[int, int] | None:
        frame = int(frame)
        if frame % self.replan_stride:
            raise ValueError("planning arrival is not stride aligned")
        decision_index = frame // self.replan_stride
        if decision_index != self.last_decision + 1:
            raise ValueError("planning decisions must arrive contiguously")
        self.last_decision = decision_index
        reason = self._pending.pop(decision_index, None)
        if reason is None:
            return None
        if decision_index - self.last_boundary < self.minimum_segment_decisions:
            self.coalesced_confirmation_count += 1
            return None
        close = (self.last_boundary, decision_index)
        self.last_boundary = decision_index
        return close


def align_detector_events(
    events: Sequence[Mapping[str, Any]],
    *,
    episode_frames: int,
    replan_stride: int = 16,
    minimum_segment_decisions: int = 2,
    maximum_segment_decisions: int = 8,
) -> dict[str, Any]:
    """Return a complete planning-index partition without backdating events.

    A detector confirmation at frame ``f`` becomes actionable only at the first
    policy decision at or after ``f``.  Confirmations that would create an
    interior one-observation memory group are coalesced into the open group.
    """

    episode_frames = int(episode_frames)
    replan_stride = int(replan_stride)
    minimum_segment_decisions = int(minimum_segment_decisions)
    maximum_segment_decisions = int(maximum_segment_decisions)
    if (episode_frames <= 0 or replan_stride <= 0 or minimum_segment_decisions < 1
            or maximum_segment_decisions < minimum_segment_decisions):
        raise ValueError("episode, stride, and minimum segment must be positive")
    decision_count = (episode_frames + replan_stride - 1) // replan_stride
    last_decision_index = decision_count - 1
    boundaries = [0]
    reasons = {"0": "start"}
    mapping = []
    previous_confirmation = -1
    for raw in events:
        confirmation = int(raw["confirmation_frame"])
        if confirmation < previous_confirmation:
            raise ValueError("detector confirmations must be ordered")
        previous_confirmation = confirmation
        reason = str(raw["reason"])
        decision_index = math.ceil(confirmation / replan_stride)
        row = {
            "confirmation_frame": confirmation,
            "aligned_frame": decision_index * replan_stride,
            "decision_index": decision_index,
            "reason": reason,
        }
        if decision_index > last_decision_index:
            row["status"] = "after_last_decision"
        elif decision_index <= boundaries[-1]:
            row["status"] = "coalesced_same_decision"
        elif decision_index - boundaries[-1] < minimum_segment_decisions:
            row["status"] = "coalesced_short_group"
        else:
            boundaries.append(decision_index)
            reasons[str(decision_index)] = reason
            row["status"] = "kept"
        mapping.append(row)
    if boundaries[-1] != decision_count:
        boundaries.append(decision_count)
    reasons[str(decision_count)] = "end"
    covered = [index for left, right in zip(boundaries, boundaries[1:])
               for index in range(left, right)]
    if covered != list(range(decision_count)):
        raise RuntimeError("planning partition skipped or duplicated observations")
    lengths = [right - left for left, right in zip(boundaries, boundaries[1:])]
    if any(length > maximum_segment_decisions for length in lengths):
        raise ValueError(
            f"aligned segment exceeds maximum {maximum_segment_decisions}: {lengths}"
        )
    return {
        "decision_count": decision_count,
        "boundaries": boundaries,
        "reasons": reasons,
        "confirmation_to_decision": mapping,
        "retroactive_boundary_count": 0,
    }
