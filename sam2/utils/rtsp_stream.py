"""Bounded, latest-frame RTSP input for live SAM 2 sessions."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import closing
from itertools import count
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
import torch

from sam2.live_frame import CapturedFrame, CapturedVideoFrame, LiveFrame
from sam2.utils.rtsp_decoder import (
    DecodeMode,
    DecoderBackend,
    RtspDecoderUnavailable,
    decode_rtsp,
)


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
    decoder_backend: DecoderBackend | None = None
    fallback_reason: str | None = None


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

    PyAV uses CUDA by default when the inference device and stream support it.
    Open and read deadlines match OpenCV's FFmpeg defaults unless overridden.
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
        decoder: DecodeMode = "auto",
        inference_device: torch.device | str | None = None,
        capture_factory: CaptureFactory | None = None,
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
        if decoder not in ("auto", "cpu", "cuda"):
            raise ValueError("decoder must be auto, cpu, or cuda")
        if capture_factory is not None and decoder == "cuda":
            raise ValueError("an OpenCV capture factory supplies CPU frames")
        self._decoder = decoder
        self._inference_device = (
            torch.device(inference_device) if inference_device is not None else None
        )
        self._capture_factory = capture_factory
        self._condition = Condition()
        self._frames: deque[LiveFrame] = deque()
        self._decoded_frames = 0
        self._dropped_frames = 0
        self._maximum_depth = 0
        self._disconnected = False
        self._closed = False
        self._failure: tuple[type[RtspDisconnected] | type[RtspDecoderUnavailable], str] | None = None
        self._decoder_backend: DecoderBackend | None = None
        self._fallback_reason: str | None = None
        self._worker = Thread(
            target=self._decode_loop, name="sam2-rtsp-reader", daemon=True
        )
        self._worker.start()

    def read(self, timeout_seconds: float | None = None) -> LiveFrame:
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
                    error_type, reason = self._failure
                    raise error_type(reason) from None
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
                decoder_backend=self._decoder_backend,
                fallback_reason=self._fallback_reason,
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
        failure = "RTSP stream could not be opened"
        try:
            with closing(self._iter_frames()) as frames:
                while True:
                    with self._condition:
                        if self._closed:
                            return
                    frame, backend, reason = next(frames)
                    failure = "RTSP decoder failed"
                    with self._condition:
                        if self._closed:
                            return
                        self._decoder_backend = backend
                        self._fallback_reason = reason
                        self._decoded_frames += 1
                        if len(self._frames) == self._capacity:
                            self._frames.popleft()
                            self._dropped_frames += 1
                        self._frames.append(frame)
                        self._maximum_depth = max(self._maximum_depth, len(self._frames))
                        self._condition.notify_all()
        except StopIteration:
            self._finish()
        except RtspDecoderUnavailable as error:
            self._finish((RtspDecoderUnavailable, str(error)))
        except Exception as error:
            self._finish((RtspDisconnected, f"{failure} ({type(error).__name__})"))

    def _finish(
        self,
        failure: tuple[type[RtspDisconnected] | type[RtspDecoderUnavailable], str] | None = None,
    ) -> None:
        with self._condition:
            self._failure = failure
            self._disconnected = True
            self._condition.notify_all()

    def _iter_frames(self) -> Generator[tuple[LiveFrame, DecoderBackend, str | None], None, None]:
        factory = self._capture_factory
        if factory is None:
            with closing(decode_rtsp(
                self._url,
                mode=self._decoder,
                inference_device=self._inference_device,
                open_timeout_seconds=self._open_timeout_seconds,
                read_timeout_seconds=self._read_timeout_seconds,
            )) as frames:
                for index, (pixels, backend, reason) in enumerate(frames):
                    yield CapturedVideoFrame(pixels, index, time.monotonic(), pixels.time), backend, reason
            return

        parameters: list[int] = []
        if self._open_timeout_seconds is not None:
            parameters.extend([
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(self._open_timeout_seconds * 1000)
            ])
        if self._read_timeout_seconds is not None:
            parameters.extend([
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(self._read_timeout_seconds * 1000)
            ])
        capture = factory(self._url, cv2.CAP_FFMPEG, parameters)
        try:
            if not capture.isOpened():
                raise RtspDisconnected("RTSP stream could not be opened")
            for index in count():
                decoded, bgr = capture.read()
                received_at = time.monotonic()
                if not decoded:
                    return
                if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
                    raise ValueError("RTSP decoder returned a non-BGR frame")
                pts = self._presentation_time_seconds(capture)
                yield CapturedFrame(bgr, index, received_at, pts), "cpu", None
        finally:
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
