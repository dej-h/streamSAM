"""Bounded, latest-frame RTSP input for live SAM 2 sessions."""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from threading import Condition, Thread
from types import TracebackType
from typing import Callable, Protocol
from urllib.parse import urlsplit

import cv2
import numpy as np
from numpy.typing import NDArray

from sam2.live_frame import CapturedFrame


class RtspDisconnected(ConnectionError):
    """The current camera epoch ended or could not be opened."""


@dataclass(frozen=True)
class RtspReaderStats:
    capacity: int
    current_depth: int
    maximum_depth: int
    decoded_frames: int
    dropped_frames: int
    disconnected: bool
    closed: bool


class VideoCaptureLike(Protocol):
    def isOpened(self) -> bool: ...

    def read(self) -> tuple[bool, NDArray[np.uint8]]: ...

    def get(self, property_id: int) -> float: ...

    def release(self) -> None: ...


CaptureFactory = Callable[[str, int, list[int]], VideoCaptureLike]
_MAX_OPENCV_TIMEOUT_SECONDS = (2**31 - 1) / 1000


class RtspFrameReader:
    """Decode RTSP in the background and return the newest available frame.

    This reader ends an epoch on a failed open or read. Callers start a new
    reader and predictor session after a disconnect, so temporal state is never
    silently carried across a camera gap.

    Decoder timeouts use OpenCV's defaults unless explicitly overridden.
    ``read()`` waits for a frame or decoder termination; an optional caller
    timeout limits only that wait. ``close()`` joins the decoder worker and
    can wait for an in-progress backend operation to finish.
    """

    def __init__(
        self,
        url: str,
        *,
        capacity: int = 4,
        open_timeout_seconds: float | None = None,
        read_timeout_seconds: float | None = None,
        capture_factory: CaptureFactory = cv2.VideoCapture,
    ) -> None:
        if urlsplit(url).scheme not in ("rtsp", "rtsps"):
            raise ValueError("RTSP URL must use rtsp:// or rtsps://")
        if capacity <= 0:
            raise ValueError("RTSP queue capacity must be positive")
        for timeout in (open_timeout_seconds, read_timeout_seconds):
            if timeout is not None and not 0.001 <= timeout <= _MAX_OPENCV_TIMEOUT_SECONDS:
                raise ValueError("RTSP timeouts must fit in positive OpenCV milliseconds")
        self._url = url
        self._capacity = capacity
        self._open_timeout_seconds = open_timeout_seconds
        self._read_timeout_seconds = read_timeout_seconds
        self._capture_factory = capture_factory
        self._condition = Condition()
        self._frames: deque[CapturedFrame] = deque()
        self._decoded_frames = 0
        self._dropped_frames = 0
        self._maximum_depth = 0
        self._disconnected = False
        self._closed = False
        self._failure: str | None = None
        self._worker = Thread(
            target=self._decode_loop, name="sam2-rtsp-reader", daemon=True
        )
        self._worker.start()

    def read(self, timeout_seconds: float | None = None) -> CapturedFrame:
        if timeout_seconds is not None and (
            not math.isfinite(timeout_seconds) or timeout_seconds <= 0
        ):
            raise ValueError("read timeout must be finite and positive")
        deadline = (
            None if timeout_seconds is None else time.monotonic() + timeout_seconds
        )
        with self._condition:
            while not self._frames:
                if self._closed:
                    raise RuntimeError("RTSP reader is closed")
                if self._failure is not None:
                    raise RtspDisconnected(self._failure) from None
                if self._disconnected:
                    raise RtspDisconnected("RTSP stream disconnected")
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("no RTSP frame arrived before the read timeout")
                self._condition.wait(remaining)
            newest = self._frames.pop()
            self._dropped_frames += len(self._frames)
            self._frames.clear()
            return newest

    def stats(self) -> RtspReaderStats:
        with self._condition:
            return RtspReaderStats(
                capacity=self._capacity,
                current_depth=len(self._frames),
                maximum_depth=self._maximum_depth,
                decoded_frames=self._decoded_frames,
                dropped_frames=self._dropped_frames,
                disconnected=self._disconnected,
                closed=self._closed,
            )

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._dropped_frames += len(self._frames)
            self._frames.clear()
            self._condition.notify_all()
        self._worker.join()

    def __enter__(self) -> RtspFrameReader:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _decode_loop(self) -> None:
        capture: VideoCaptureLike | None = None
        failure = "RTSP stream could not be opened"
        try:
            parameters: list[int] = []
            if self._open_timeout_seconds is not None:
                parameters.extend(
                    [
                        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                        int(self._open_timeout_seconds * 1000),
                    ]
                )
            if self._read_timeout_seconds is not None:
                parameters.extend(
                    [
                        cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                        int(self._read_timeout_seconds * 1000),
                    ]
                )
            capture = self._capture_factory(
                self._url,
                cv2.CAP_FFMPEG,
                parameters,
            )
            if not capture.isOpened():
                raise RtspDisconnected(failure)
            failure = "RTSP decoder failed"
            while True:
                with self._condition:
                    if self._closed:
                        return
                decoded, bgr = capture.read()
                received_at = time.monotonic()
                if not decoded:
                    with self._condition:
                        self._disconnected = True
                        self._condition.notify_all()
                    return
                if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
                    raise ValueError("RTSP decoder returned a non-BGR frame")
                pts = self._presentation_time_seconds(capture)
                with self._condition:
                    if self._closed:
                        return
                    frame = CapturedFrame(
                        bgr=bgr,
                        decoded_sequence=self._decoded_frames,
                        received_at_monotonic=received_at,
                        presentation_time_seconds=pts,
                    )
                    self._decoded_frames += 1
                    if len(self._frames) == self._capacity:
                        self._frames.popleft()
                        self._dropped_frames += 1
                    self._frames.append(frame)
                    self._maximum_depth = max(self._maximum_depth, len(self._frames))
                    self._condition.notify_all()
        except BaseException:
            with self._condition:
                self._failure = failure
                self._disconnected = True
                self._condition.notify_all()
        finally:
            if capture is not None:
                capture.release()

    @staticmethod
    def _presentation_time_seconds(capture: VideoCaptureLike) -> float | None:
        property_id = getattr(cv2, "CAP_PROP_PTS", None)
        if property_id is None:
            return None
        fps = capture.get(cv2.CAP_PROP_FPS)
        pts = capture.get(property_id)
        if not math.isfinite(fps) or fps <= 0 or not math.isfinite(pts):
            return None
        return pts / fps
