from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray

from sam2.utils.h264_video_encoder import H264VideoEncoder


_QUEUE_POLL_SECONDS: Final = 0.1


@dataclass(frozen=True)
class VideoMaskFrame:
    frame_idx: int
    object_ids: tuple[int, ...]
    masks: NDArray[np.bool_]
    frame_bgr: NDArray[np.uint8] | None = None


@dataclass(frozen=True)
class VideoOutputWriterStats:
    capacity: int
    submitted_frames: int
    written_frames: int
    current_depth: int
    maximum_depth: int
    producer_wait_count: int
    producer_wait_seconds: float
    worker_seconds: float
    closed: bool


class _MaskOverlay:
    def __init__(
        self,
        frame_size: tuple[int, int],
        alpha: float,
        color_rgb: tuple[int, int, int],
    ) -> None:
        width, height = frame_size
        levels = np.arange(256, dtype=np.float32).reshape(256, 1, 1)
        color_bgr = np.asarray(color_rgb[::-1], dtype=np.float32)
        # Match the float32 blend and uint8 truncation for every input level.
        self._color_lut = (levels * (1.0 - alpha) + color_bgr * alpha).astype(np.uint8)
        self._frame_bgr = np.empty((height, width, 3), dtype=np.uint8)
        self._mask = np.empty((height, width), dtype=np.bool_)

    def compose(
        self, frame_bgr: NDArray[np.uint8], masks: NDArray[np.bool_]
    ) -> NDArray[np.uint8]:
        """Return the source for an empty mask, or a BGR overlay in a reusable buffer."""
        np.any(masks, axis=0, out=self._mask)
        mask_bytes = self._mask.view(np.uint8)
        if not cv2.countNonZero(mask_bytes):
            return frame_bgr
        np.logical_not(self._mask, out=self._mask)
        cv2.LUT(frame_bgr, self._color_lut, dst=self._frame_bgr)
        cv2.copyTo(frame_bgr, mask_bytes, self._frame_bgr)
        return self._frame_bgr


class BoundedVideoMaskWriter:
    """Overlay ordered masks and encode video on a bounded worker queue.

    Supply an input path to decode frames in the worker, or submit each original
    frame with its mask when the inference decoder already provides the frames.
    Set ``encode_h264`` for browser playback. This requires FFmpeg with libx264.
    """

    def __init__(
        self,
        *,
        input_path: str | Path | None,
        output_path: str | Path,
        fps: float,
        frame_size: tuple[int, int],
        expected_frame_count: int,
        expected_object_ids: tuple[int, ...],
        capacity: int = 4,
        alpha: float = 0.6,
        color_rgb: tuple[int, int, int] = (31, 119, 180),
        encode_h264: bool = False,
    ) -> None:
        if capacity <= 0:
            raise ValueError("output queue capacity must be positive")
        if fps <= 0:
            raise ValueError("output FPS must be positive")
        if expected_frame_count <= 0:
            raise ValueError("expected frame count must be positive")
        width, height = frame_size
        if width <= 0 or height <= 0:
            raise ValueError("frame dimensions must be positive")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("mask alpha must be between zero and one")

        self._input_path = str(input_path) if input_path is not None else None
        self._output_path = str(output_path)
        self._encode_h264 = encode_h264
        self._fps = fps
        self._frame_size = frame_size
        self._expected_frame_count = expected_frame_count
        self._expected_object_ids = expected_object_ids
        self._alpha = alpha
        self._color_rgb = color_rgb
        self._capacity = capacity
        self._queue: queue.Queue[VideoMaskFrame | None] = queue.Queue(capacity)
        self._lock = threading.Lock()
        self._worker_error: BaseException | None = None
        self._submitted_frames = 0
        self._written_frames = 0
        self._maximum_depth = 0
        self._producer_wait_count = 0
        self._producer_wait_seconds = 0.0
        self._worker_seconds = 0.0
        self._closed = False
        self._worker = threading.Thread(
            target=self._run,
            name="sam2-video-output-writer",
            daemon=True,
        )
        self._worker.start()

    @property
    def output_path(self) -> str:
        return self._output_path

    def submit(
        self,
        frame_idx: int,
        object_ids: tuple[int, ...],
        masks: NDArray[np.bool_],
        *,
        frame_bgr: NDArray[np.uint8] | None = None,
    ) -> None:
        """Queue masks and, when no input path was given, take the original frame.

        The caller must leave the submitted frame unchanged while the worker owns
        it. The worker retains the array without copying it.
        """
        if self._input_path is None:
            if frame_bgr is None:
                raise ValueError("submit a BGR frame when no input path is configured")
            width, height = self._frame_size
            if frame_bgr.dtype != np.uint8 or frame_bgr.shape != (height, width, 3):
                raise ValueError(
                    "submitted frame must be a uint8 BGR image at output size"
                )
        elif frame_bgr is not None:
            raise ValueError("cannot submit a BGR frame when an input path is configured")
        with self._lock:
            self._raise_worker_error_locked()
            if self._closed:
                raise RuntimeError("video output writer is closed")
            if frame_idx != self._submitted_frames:
                raise ValueError(
                    f"received frame {frame_idx}; expected {self._submitted_frames}"
                )
            if object_ids != self._expected_object_ids:
                raise ValueError(
                    f"received object ids {object_ids}; expected "
                    f"{self._expected_object_ids}"
                )
        owned_masks = np.asarray(masks, dtype=np.bool_).copy()
        if owned_masks.ndim != 3:
            raise ValueError("masks must have shape (objects, height, width)")
        if owned_masks.shape[0] != len(object_ids):
            raise ValueError("mask count does not match object id count")

        item = VideoMaskFrame(frame_idx, object_ids, owned_masks, frame_bgr)
        wait_started_at = time.perf_counter()
        waited = False
        while True:
            with self._lock:
                self._raise_worker_error_locked()
            try:
                self._queue.put(item, timeout=_QUEUE_POLL_SECONDS)
                break
            except queue.Full:
                waited = True
        with self._lock:
            self._submitted_frames += 1
            if waited:
                self._producer_wait_count += 1
                self._producer_wait_seconds += time.perf_counter() - wait_started_at
            self._maximum_depth = max(self._maximum_depth, self._queue.qsize())

    def close(self) -> VideoOutputWriterStats:
        with self._lock:
            if self._closed:
                self._raise_worker_error_locked()
                return self._stats_locked()
            self._closed = True
        while self._worker.is_alive():
            with self._lock:
                self._raise_worker_error_locked()
            try:
                self._queue.put(None, timeout=_QUEUE_POLL_SECONDS)
                break
            except queue.Full:
                continue
        self._worker.join()
        with self._lock:
            self._raise_worker_error_locked()
            if self._submitted_frames != self._expected_frame_count:
                raise RuntimeError(
                    f"submitted {self._submitted_frames} frames; expected "
                    f"{self._expected_frame_count}"
                )
            if self._written_frames != self._expected_frame_count:
                raise RuntimeError(
                    f"wrote {self._written_frames} frames; expected "
                    f"{self._expected_frame_count}"
                )
        return self.stats()

    def stats(self) -> VideoOutputWriterStats:
        with self._lock:
            return self._stats_locked()

    def _stats_locked(self) -> VideoOutputWriterStats:
        return VideoOutputWriterStats(
            capacity=self._capacity,
            submitted_frames=self._submitted_frames,
            written_frames=self._written_frames,
            current_depth=self._queue.qsize(),
            maximum_depth=self._maximum_depth,
            producer_wait_count=self._producer_wait_count,
            producer_wait_seconds=self._producer_wait_seconds,
            worker_seconds=self._worker_seconds,
            closed=self._closed,
        )

    def __enter__(self) -> BoundedVideoMaskWriter:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exception_type is None:
            self.close()
        else:
            self.abort()

    def abort(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        while self._worker.is_alive():
            try:
                self._queue.put_nowait(None)
                break
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
        self._worker.join()

    def _run(self) -> None:
        capture: cv2.VideoCapture | None = None
        writer: cv2.VideoWriter | H264VideoEncoder | None = None
        started_at = time.perf_counter()
        try:
            if self._input_path is not None:
                capture = cv2.VideoCapture(self._input_path)
                if not capture.isOpened():
                    raise RuntimeError(f"could not open input video: {self._input_path}")
            width, height = self._frame_size
            if self._encode_h264:
                writer = H264VideoEncoder(self._output_path, self._fps, self._frame_size)
            else:
                writer = cv2.VideoWriter(
                    self._output_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    self._fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"could not open output video: {self._output_path}")

            overlay = _MaskOverlay(self._frame_size, self._alpha, self._color_rgb)
            next_frame_idx = 0
            while True:
                item = self._queue.get()
                if item is None:
                    break
                if item.frame_idx != next_frame_idx:
                    raise RuntimeError(
                        f"writer received frame {item.frame_idx}; expected "
                        f"{next_frame_idx}"
                    )
                frame_bgr = item.frame_bgr
                if capture is not None:
                    ok, frame_bgr = capture.read()
                    if not ok:
                        raise RuntimeError(
                            f"input video ended before frame {item.frame_idx}"
                        )
                if frame_bgr is None:
                    raise RuntimeError(f"missing original frame {item.frame_idx}")
                if (frame_bgr.shape[1], frame_bgr.shape[0]) != self._frame_size:
                    raise RuntimeError(
                        f"source frame {item.frame_idx} has dimensions "
                        f"{frame_bgr.shape[1]}x{frame_bgr.shape[0]}; expected "
                        f"{width}x{height}"
                    )
                if item.masks.shape[1:] != (height, width):
                    raise RuntimeError(
                        f"mask frame {item.frame_idx} has shape "
                        f"{item.masks.shape[1:]}; expected {(height, width)}"
                    )
                writer.write(overlay.compose(frame_bgr, item.masks))
                next_frame_idx += 1
                with self._lock:
                    self._written_frames = next_frame_idx
        except BaseException as error:
            with self._lock:
                self._worker_error = error
        finally:
            if capture is not None:
                capture.release()
            if writer is not None:
                try:
                    writer.release()
                except BaseException as error:
                    with self._lock:
                        if self._worker_error is None:
                            self._worker_error = error
            with self._lock:
                self._worker_seconds = time.perf_counter() - started_at

    def _raise_worker_error_locked(self) -> None:
        if self._worker_error is not None:
            raise RuntimeError("video output worker failed") from self._worker_error
