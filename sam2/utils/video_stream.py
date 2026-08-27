from __future__ import annotations

import os
import tempfile
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import Condition, Thread
from typing import Literal, Protocol, Sequence, runtime_checkable

import cv2
import numpy as np
import torch

from sam2.utils.misc import _load_img_as_tensor, load_video_frames


VideoInput = str | bytes
FrameLoadingMode = Literal["eager", "lazy"]
RgbMean = tuple[float, float, float]
RgbStd = tuple[float, float, float]


@dataclass(frozen=True)
class FrameSourceMetadata:
    frame_count: int
    video_height: int
    video_width: int
    fps: float | None
    image_size: int
    preprocessing_identity: str


@dataclass(frozen=True)
class FrameSourceStats:
    capacity: int
    current_depth: int
    maximum_depth: int
    decoded_frames: int
    seek_count: int
    wait_count: int
    wait_seconds: float
    closed: bool


@runtime_checkable
class FrameSource(Protocol):
    @property
    def metadata(self) -> FrameSourceMetadata: ...

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> torch.Tensor: ...

    def stats(self) -> FrameSourceStats: ...

    def close(self) -> None: ...


class EagerFrameSource:
    def __init__(
        self,
        frames: Sequence[torch.Tensor],
        metadata: FrameSourceMetadata,
    ) -> None:
        if len(frames) != metadata.frame_count:
            raise ValueError("frame count does not match eager frame storage")
        self._frames = frames
        self._metadata = metadata
        self._closed = False

    @property
    def metadata(self) -> FrameSourceMetadata:
        return self._metadata

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, index: int) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("frame source is closed")
        return self._frames[index]

    def stats(self) -> FrameSourceStats:
        frame_count = len(self._frames)
        return FrameSourceStats(
            capacity=frame_count,
            current_depth=frame_count if not self._closed else 0,
            maximum_depth=frame_count,
            decoded_frames=frame_count,
            seek_count=0,
            wait_count=0,
            wait_seconds=0.0,
            closed=self._closed,
        )

    def close(self) -> None:
        self._closed = True


class SequentialMp4FrameSource:
    """Pull normalized MP4 frames into a bounded CPU queue."""

    def __init__(
        self,
        video_path: VideoInput,
        image_size: int,
        capacity: int,
        img_mean: RgbMean,
        img_std: RgbStd,
        pin_memory: bool = False,
    ) -> None:
        if image_size <= 0:
            raise ValueError("image size must be positive")
        if capacity <= 0:
            raise ValueError("frame source capacity must be positive")

        self._video_path, self._temporary_video_path = self._materialize_video_path(
            video_path
        )
        metadata_capture = self._open_capture()
        try:
            frame_count = int(metadata_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            video_height = int(metadata_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_width = int(metadata_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            fps_value = float(metadata_capture.get(cv2.CAP_PROP_FPS))
        finally:
            metadata_capture.release()
        if frame_count <= 0:
            self._cleanup_temporary_video()
            raise RuntimeError("video contains no frames")
        if video_height <= 0 or video_width <= 0:
            self._cleanup_temporary_video()
            raise RuntimeError("video dimensions are unavailable")

        self._image_size = image_size
        self._metadata = FrameSourceMetadata(
            frame_count=frame_count,
            video_height=video_height,
            video_width=video_width,
            fps=fps_value if fps_value > 0 else None,
            image_size=image_size,
            preprocessing_identity=_preprocessing_identity(
                image_size, img_mean, img_std
            ),
        )
        self._capacity = capacity
        self._img_mean = torch.tensor(
            img_mean, dtype=torch.float32
        )[:, None, None]
        self._img_std = torch.tensor(
            img_std, dtype=torch.float32
        )[:, None, None]
        self._pin_memory = pin_memory and torch.cuda.is_available()
        self._condition = Condition()
        self._frames: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._generation = 0
        self._decode_index = 0
        self._seek_request: tuple[int, int] | None = None
        self._closed = False
        self._exception: BaseException | None = None
        self._decoded_frames = 0
        self._seek_count = 0
        self._wait_count = 0
        self._wait_seconds = 0.0
        self._maximum_depth = 0
        self._thread = Thread(
            target=self._decode_loop,
            name="sam2-sequential-frame-source",
            daemon=True,
        )
        self._thread.start()

    @property
    def metadata(self) -> FrameSourceMetadata:
        return self._metadata

    def __len__(self) -> int:
        return self._metadata.frame_count

    def __getitem__(self, index: int) -> torch.Tensor:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        with self._condition:
            while True:
                self._raise_if_unavailable_locked()
                frame = self._frames.pop(index, None)
                if frame is not None:
                    self._drop_frames_before_locked(index)
                    self._condition.notify_all()
                    return frame

                self._drop_frames_before_locked(index)
                if self._requires_seek_locked(index):
                    self._request_seek_locked(index)

                wait_started_at = time.perf_counter()
                self._wait_count += 1
                self._condition.wait()
                self._wait_seconds += time.perf_counter() - wait_started_at

    def stats(self) -> FrameSourceStats:
        with self._condition:
            return FrameSourceStats(
                capacity=self._capacity,
                current_depth=len(self._frames),
                maximum_depth=self._maximum_depth,
                decoded_frames=self._decoded_frames,
                seek_count=self._seek_count,
                wait_count=self._wait_count,
                wait_seconds=self._wait_seconds,
                closed=self._closed,
            )

    def close(self) -> None:
        with self._condition:
            should_join = not self._closed
            if should_join:
                self._closed = True
                self._frames.clear()
                self._condition.notify_all()
        if should_join:
            self._thread.join()
        self._cleanup_temporary_video()

    def _decode_loop(self) -> None:
        decode_index = 0
        generation = 0
        capture = self._open_capture()
        try:
            while True:
                seek_index: int | None = None
                with self._condition:
                    while True:
                        if self._closed:
                            return
                        if self._seek_request is not None:
                            seek_index, generation = self._seek_request
                            self._seek_request = None
                            decode_index = seek_index
                            self._decode_index = decode_index
                            break
                        if (
                            decode_index < len(self)
                            and len(self._frames) < self._capacity
                        ):
                            self._decode_index = decode_index
                            break
                        self._condition.wait()

                if seek_index is not None:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, seek_index)
                decoded, bgr_frame = capture.read()
                if not decoded:
                    raise RuntimeError(f"failed to decode frame {decode_index}")
                frame = self._normalize_frame(bgr_frame)

                with self._condition:
                    if self._closed:
                        return
                    if generation != self._generation:
                        continue
                    self._frames[decode_index] = frame
                    self._decoded_frames += 1
                    self._maximum_depth = max(
                        self._maximum_depth, len(self._frames)
                    )
                    decode_index += 1
                    self._decode_index = decode_index
                    self._condition.notify_all()
        except BaseException as error:
            with self._condition:
                self._exception = error
                self._condition.notify_all()
        finally:
            capture.release()

    def _normalize_frame(self, bgr_frame: np.ndarray) -> torch.Tensor:
        resized_bgr = cv2.resize(
            bgr_frame,
            (self._image_size, self._image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        rgb_frame = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
        normalized = (
            torch.from_numpy(rgb_frame)
            .permute(2, 0, 1)
            .to(dtype=torch.float32)
            .div_(255.0)
        )
        normalized.sub_(self._img_mean).div_(self._img_std)
        return normalized.pin_memory() if self._pin_memory else normalized

    def _open_capture(self) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(self._video_path)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"failed to open MP4 video {self._video_path}")
        return capture

    @staticmethod
    def _materialize_video_path(video_path: VideoInput) -> tuple[str, Path | None]:
        if isinstance(video_path, str):
            return video_path, None
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temporary:
            temporary.write(video_path)
            return temporary.name, Path(temporary.name)

    def _cleanup_temporary_video(self) -> None:
        if self._temporary_video_path is None:
            return
        self._temporary_video_path.unlink(missing_ok=True)
        self._temporary_video_path = None

    def _drop_frames_before_locked(self, index: int) -> None:
        for frame_index in tuple(self._frames):
            if frame_index < index:
                del self._frames[frame_index]

    def _requires_seek_locked(self, index: int) -> bool:
        if self._seek_request is not None:
            return self._seek_request[0] != index
        if index < self._decode_index:
            return True
        return index >= self._decode_index + self._capacity

    def _request_seek_locked(self, index: int) -> None:
        self._generation += 1
        self._frames.clear()
        self._seek_request = (index, self._generation)
        self._seek_count += 1
        self._condition.notify_all()

    def _raise_if_unavailable_locked(self) -> None:
        if self._exception is not None:
            raise RuntimeError("frame decode worker failed") from self._exception
        if self._closed:
            raise RuntimeError("frame source is closed")


class SequentialImageDirectoryFrameSource:
    """Prefetch normalized numbered image frames through a bounded CPU queue."""

    _IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".JPG", ".JPEG"})

    def __init__(
        self,
        video_path: str,
        image_size: int,
        capacity: int,
        img_mean: RgbMean,
        img_std: RgbStd,
    ) -> None:
        if image_size <= 0:
            raise ValueError("image size must be positive")
        if capacity <= 0:
            raise ValueError("frame source capacity must be positive")

        image_directory = Path(video_path).expanduser().resolve()
        if not image_directory.is_dir():
            raise NotADirectoryError(image_directory)
        frame_paths = sorted(
            (
                path
                for path in image_directory.iterdir()
                if path.suffix in self._IMAGE_SUFFIXES
            ),
            key=lambda path: int(path.stem),
        )
        if not frame_paths:
            raise RuntimeError(f"no numbered JPEG images found in {image_directory}")

        _, video_height, video_width = _load_img_as_tensor(
            str(frame_paths[0]), image_size
        )
        self._metadata = FrameSourceMetadata(
            frame_count=len(frame_paths),
            video_height=video_height,
            video_width=video_width,
            fps=None,
            image_size=image_size,
            preprocessing_identity=_preprocessing_identity(
                image_size, img_mean, img_std
            ),
        )
        self._frame_paths = tuple(frame_paths)
        self._capacity = capacity
        self._mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
        self._std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
        self._condition = Condition()
        self._frames: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._next_decode_index = 0
        self._inflight_index: int | None = None
        self._generation = 0
        self._worker_error: BaseException | None = None
        self._closed = False
        self._decoded_frames = 0
        self._seek_count = 0
        self._wait_count = 0
        self._wait_seconds = 0.0
        self._maximum_depth = 0
        self._worker = Thread(
            target=self._decode_loop,
            name="sam2-image-directory-decoder",
            daemon=True,
        )
        self._worker.start()

    @property
    def metadata(self) -> FrameSourceMetadata:
        return self._metadata

    def __len__(self) -> int:
        return len(self._frame_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        if index < 0 or index >= len(self):
            raise IndexError(f"frame index {index} is outside the image sequence")

        wait_started_at = time.perf_counter()
        waited = False
        with self._condition:
            self._raise_worker_error_locked()
            if self._closed:
                raise RuntimeError("frame source is closed")
            if (
                index not in self._frames
                and index != self._inflight_index
                and index != self._next_decode_index
            ):
                self._generation += 1
                self._frames.clear()
                self._next_decode_index = index
                self._seek_count += 1
                self._condition.notify_all()

            while index not in self._frames:
                self._raise_worker_error_locked()
                if self._closed:
                    raise RuntimeError("frame source closed before frame was decoded")
                waited = True
                self._condition.wait()

            frame = self._frames.pop(index)
            if waited:
                self._wait_count += 1
                self._wait_seconds += time.perf_counter() - wait_started_at
            self._condition.notify_all()
            return frame

    def stats(self) -> FrameSourceStats:
        with self._condition:
            return FrameSourceStats(
                capacity=self._capacity,
                current_depth=len(self._frames),
                maximum_depth=self._maximum_depth,
                decoded_frames=self._decoded_frames,
                seek_count=self._seek_count,
                wait_count=self._wait_count,
                wait_seconds=self._wait_seconds,
                closed=self._closed,
            )

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._frames.clear()
            self._condition.notify_all()
        self._worker.join()

    def _decode_loop(self) -> None:
        try:
            while True:
                with self._condition:
                    while (
                        not self._closed
                        and (
                            len(self._frames) >= self._capacity
                            or self._next_decode_index >= len(self)
                        )
                    ):
                        self._condition.wait()
                    if self._closed:
                        return
                    frame_idx = self._next_decode_index
                    self._inflight_index = frame_idx
                    generation = self._generation
                    self._next_decode_index += 1

                frame, height, width = _load_img_as_tensor(
                    str(self._frame_paths[frame_idx]), self._metadata.image_size
                )
                if (
                    height != self._metadata.video_height
                    or width != self._metadata.video_width
                ):
                    raise ValueError(
                        f"frame {frame_idx} has dimensions {width}x{height}; expected "
                        f"{self._metadata.video_width}x{self._metadata.video_height}"
                    )
                # The eager JPEG loader copies PIL's float64 tensor into a float32
                # batch before normalization. Match that ordering exactly so lazy
                # directory input is mask-equivalent to the existing core path.
                frame = frame.to(dtype=torch.float32)
                frame.sub_(self._mean).div_(self._std)

                with self._condition:
                    if self._closed:
                        return
                    if generation != self._generation:
                        if self._inflight_index == frame_idx:
                            self._inflight_index = None
                        self._condition.notify_all()
                        continue
                    self._frames[frame_idx] = frame
                    self._inflight_index = None
                    self._decoded_frames += 1
                    self._maximum_depth = max(
                        self._maximum_depth, len(self._frames)
                    )
                    self._condition.notify_all()
        except BaseException as error:
            with self._condition:
                self._worker_error = error
                self._condition.notify_all()

    def _raise_worker_error_locked(self) -> None:
        if self._worker_error is not None:
            raise RuntimeError("image-directory decoder failed") from self._worker_error


def create_video_frame_source(
    *,
    video_path: VideoInput,
    image_size: int,
    offload_video_to_cpu: bool,
    loading_mode: FrameLoadingMode,
    buffer_capacity: int,
    img_mean: RgbMean = (0.485, 0.456, 0.406),
    img_std: RgbStd = (0.229, 0.224, 0.225),
    async_loading_frames: bool = False,
    compute_device: torch.device = torch.device("cuda"),
) -> FrameSource:
    is_mp4_input = isinstance(video_path, bytes) or os.path.splitext(video_path)[
        -1
    ] in (".mp4", ".MP4")
    if loading_mode == "lazy":
        if is_mp4_input:
            return SequentialMp4FrameSource(
                video_path=video_path,
                image_size=image_size,
                capacity=buffer_capacity,
                img_mean=img_mean,
                img_std=img_std,
                # Pinning every decoded tensor makes PyTorch's pinned allocator retain
                # host memory over long videos. A bounded GPU stager must own and reuse
                # pinned buffers instead of making them part of the decoder queue.
                pin_memory=False,
            )
        if isinstance(video_path, str) and os.path.isdir(video_path):
            return SequentialImageDirectoryFrameSource(
                video_path=video_path,
                image_size=image_size,
                capacity=buffer_capacity,
                img_mean=img_mean,
                img_std=img_std,
            )
        raise ValueError(
            "lazy frame loading supports MP4 input or a numbered JPEG directory"
        )

    if is_mp4_input:
        source = SequentialMp4FrameSource(
            video_path=video_path,
            image_size=image_size,
            capacity=buffer_capacity,
            img_mean=img_mean,
            img_std=img_std,
            pin_memory=False,
        )
        metadata = source.metadata
        frames = torch.empty(
            (len(source), 3, image_size, image_size),
            dtype=torch.float32,
            device="cpu",
        )
        try:
            for frame_idx in range(len(source)):
                frames[frame_idx].copy_(source[frame_idx])
        finally:
            source.close()
        if not offload_video_to_cpu:
            frames = frames.to(compute_device)
        return EagerFrameSource(frames=frames, metadata=metadata)

    frames, video_height, video_width = load_video_frames(
        video_path=video_path,
        image_size=image_size,
        offload_video_to_cpu=offload_video_to_cpu,
        img_mean=img_mean,
        img_std=img_std,
        async_loading_frames=async_loading_frames,
        compute_device=compute_device,
    )
    return EagerFrameSource(
        frames=frames,
        metadata=FrameSourceMetadata(
            frame_count=len(frames),
            video_height=video_height,
            video_width=video_width,
            fps=None,
            image_size=image_size,
            preprocessing_identity=_preprocessing_identity(
                image_size, img_mean, img_std
            ),
        ),
    )


def _preprocessing_identity(
    image_size: int, img_mean: RgbMean, img_std: RgbStd
) -> str:
    return f"opencv-rgb-square-{image_size}:mean={img_mean}:std={img_std}"
