"""Shared native geometry-delta resampling for commit and external controllers."""

from __future__ import annotations

import torch
from torch import Tensor


def resize_geometry_delta(delta: Tensor, size_hw: tuple[int, int]) -> Tensor:
    """The linear resampling operator used for native-to-low/video deltas.

    Keep this operation differentiable: controllers can use its transpose to
    reject native corrections that would influence unknown video pixels.
    Callers own shape, numeric and allocation-budget validation.
    """
    return torch.nn.functional.interpolate(delta, size=size_hw, mode="bilinear", align_corners=False)


def add_geometry_delta(logits: Tensor, native_delta: Tensor) -> Tensor:
    """Add a resized correction to the existing baseline without clipping it."""
    if logits.ndim != 4 or native_delta.ndim != 4 or logits.shape[:2] != native_delta.shape[:2]:
        raise ValueError("geometry logits and delta must have matching [objects,channels,H,W] axes")
    if not logits.is_floating_point() or not native_delta.is_floating_point():
        raise TypeError("geometry logits and delta must be floating point")
    delta = resize_geometry_delta(native_delta, (logits.shape[-2], logits.shape[-1]))
    return logits + delta.to(device=logits.device, dtype=logits.dtype)
