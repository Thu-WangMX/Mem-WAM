"""Shared train/deploy Helios history slicing, independent of data loading."""
import torch


def episode_anchors(history, size=2):
    """Return the earliest causal decision latents as fixed episode anchors.

    Missing startup anchors are zero-filled on the right. This preserves a
    fixed token layout without ever copying a future observation backwards.
    """
    if history.ndim != 5 or history.shape[0] < 1 or size < 0:
        raise ValueError("Expected nonempty history [H,C,1,h,w] and non-negative size")
    if size == 0:
        return history[:0], torch.zeros(0, device=history.device, dtype=torch.bool)
    selected = history[:size]
    valid = torch.arange(size, device=history.device) < len(selected)
    if len(selected) < size:
        selected = torch.cat(
            [selected, history.new_zeros((size - len(selected), *history.shape[1:]))]
        )
    return selected, valid


def fixed_history(history, size=19):
    """Keep the current decision and its predecessors; left-pad startup with zero latents.

    Padding is the original Helios startup convention, not a future observation.
    Both training and inference use this function before patchification.
    """
    if history.ndim != 5 or history.shape[0] < 1 or size < 1:
        raise ValueError("Expected nonempty history [H,C,1,h,w] and positive size")
    selected = history[-size:]
    valid = torch.arange(size, device=history.device) >= size - len(selected)
    if len(selected) < size:
        selected = torch.cat([history.new_zeros((size-len(selected), *history.shape[1:])), selected])
    return selected, valid

