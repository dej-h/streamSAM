"""Decoded frame passed from a live source to a tracking session."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CapturedFrame:
    """Source pixels and timing for one frame.

    Pixels are an H × W × 3 uint8 BGR array. The decoded sequence increases
    within a source connection and may have gaps. Receipt time uses the
    local monotonic clock. Presentation time uses the source's media clock
    when the source exposes one.
    """

    bgr: NDArray[np.uint8]
    decoded_sequence: int
    received_at_monotonic: float
    presentation_time_seconds: float | None
