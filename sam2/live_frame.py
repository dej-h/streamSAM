"""Decoded frame passed from a live source to a tracking session."""

from __future__ import annotations

from dataclasses import dataclass

from av import VideoFrame
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

    @property
    def pixels(self) -> NDArray[np.uint8]:
        return self.bgr

    @property
    def dimensions(self) -> tuple[int, int]:
        return self.bgr.shape[0], self.bgr.shape[1]


@dataclass(frozen=True)
class CapturedVideoFrame:
    """A native decoded video frame with its original color metadata.

    A CUDA-backed frame owns its decoder surface while queued or in use by
    the tracking session. The shared timing fields use the same clocks as
    ``CapturedFrame``.
    """

    pixels: VideoFrame
    decoded_sequence: int
    received_at_monotonic: float
    presentation_time_seconds: float | None

    @property
    def bgr(self) -> NDArray[np.uint8]:
        """Materialize a CPU BGR image for callers that need to render it."""
        return self.pixels.to_ndarray(format="bgr24")

    @property
    def dimensions(self) -> tuple[int, int]:
        return self.pixels.height, self.pixels.width


LiveFrame = CapturedFrame | CapturedVideoFrame
