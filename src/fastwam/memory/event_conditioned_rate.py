"""One shared train/inference contract for event-conditioned memory rates."""

from __future__ import annotations

import math


INFORMATION_EVENT = "segment_relative_control_information"
FORCED_MAXIMUM = "forced_maximum"


def online_memory_tokens_for_segment(
    span: int,
    *,
    allocation_mode: str,
    reason: str | None,
    tokens_per_frame: int = 8,
    maximum_tokens: int = 48,
) -> int:
    """Resolve the online allocation mode without silently mixing experiments."""

    if allocation_mode == "span_full":
        span = int(span)
        count = span * int(tokens_per_frame)
        if count < 1 or count > int(maximum_tokens):
            raise ValueError(
                f"full-rate span {span} needs {count} tokens, maximum is {maximum_tokens}"
            )
        return count
    if allocation_mode == "event_full_forced_half":
        if reason is None:
            raise ValueError("event-conditioned allocation requires a boundary reason")
        return memory_tokens_for_segment(
            span,
            reason,
            tokens_per_frame=tokens_per_frame,
            maximum_tokens=maximum_tokens,
        )
    raise ValueError(f"unsupported online memory allocation mode: {allocation_mode!r}")


def memory_tokens_for_segment(
    span: int,
    reason: str,
    *,
    tokens_per_frame: int = 8,
    maximum_tokens: int = 48,
) -> int:
    """Allocate full rate to learned events and half frame-rate to forced closes."""

    span = int(span)
    tokens_per_frame = int(tokens_per_frame)
    maximum_tokens = int(maximum_tokens)
    if span < 1 or tokens_per_frame < 1 or maximum_tokens < 1:
        raise ValueError("span, tokens_per_frame, and maximum_tokens must be positive")
    if reason == INFORMATION_EVENT:
        represented_frames = span
    elif reason == FORCED_MAXIMUM:
        represented_frames = math.ceil(span / 2)
    else:
        raise ValueError(f"unsupported event-conditioned boundary reason: {reason!r}")
    count = represented_frames * tokens_per_frame
    if count > maximum_tokens:
        raise ValueError(
            f"segment span {span} with reason {reason!r} needs {count} tokens, "
            f"exceeding maximum {maximum_tokens}"
        )
    return count
