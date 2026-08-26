from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from types import TracebackType
from typing import Final, Literal

import torch

from sam2.utils.video_stream import FrameSource


# Two slots are the minimum for overlap: inference owns one frame while the
# staging thread prepares the next frame in the other slot.
DEFAULT_GPU_STAGING_SLOT_COUNT: Final = 2

_SlotState = Literal["free", "staging", "ready", "in_use", "retiring"]


@dataclass(frozen=True)
class GpuFrameStagerStats:
    capacity: int
    requested_frames: int
    staged_frames: int
    resident_hits: int
    consumer_wait_count: int
    consumer_wait_seconds: float
    host_copy_seconds: float
    transfer_count: int
    transfer_milliseconds: float
    mean_transfer_milliseconds: float | None
    maximum_pending_depth: int
    maximum_ready_depth: int
    pinned_host_bytes: int
    device_bytes: int
    closed: bool


@dataclass
class _StagingSlot:
    slot_index: int
    host_tensor: torch.Tensor
    device_tensor: torch.Tensor
    copy_started_event: torch.cuda.Event
    copy_finished_event: torch.cuda.Event
    consumed_event: torch.cuda.Event
    state: _SlotState = "free"
    frame_idx: int | None = None
    published_sequence: int = 0
    transfer_timing_pending: bool = False


class GpuFrameLease:
    """Exclusive access to one staged GPU frame until inference releases it."""

    def __init__(self, stager: GpuFrameStager, slot: _StagingSlot) -> None:
        self._stager = stager
        self._slot = slot
        self._released = False

    @property
    def frame_idx(self) -> int:
        frame_idx = self._slot.frame_idx
        if frame_idx is None:
            raise RuntimeError("staged frame lease has no frame index")
        return frame_idx

    @property
    def tensor(self) -> torch.Tensor:
        if self._released:
            raise RuntimeError("staged frame lease has already been released")
        return self._slot.device_tensor

    def release(self) -> None:
        if self._released:
            return
        self._stager._release(self._slot)
        self._released = True

    def __enter__(self) -> torch.Tensor:
        return self.tensor

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()


class GpuFrameStager:
    """Stage normalized CPU frames through a fixed pinned and GPU buffer pool."""

    def __init__(
        self,
        *,
        frame_source: FrameSource,
        device: torch.device,
        slot_count: int = DEFAULT_GPU_STAGING_SLOT_COUNT,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("GPU frame staging requires a CUDA device")
        if slot_count < 2:
            raise ValueError("GPU frame staging requires at least two slots")

        image_size = frame_source.metadata.image_size
        frame_shape = (3, image_size, image_size)
        self._frame_source = frame_source
        self._device = device
        self._condition = threading.Condition()
        self._pending_indices: deque[int] = deque()
        self._pending_index_set: set[int] = set()
        self._worker_error: BaseException | None = None
        self._closed = False
        self._last_acquired_index: int | None = None
        self._prefetch_direction = 1
        self._published_sequence = 0

        self._requested_frames = 0
        self._staged_frames = 0
        self._resident_hits = 0
        self._consumer_wait_count = 0
        self._consumer_wait_seconds = 0.0
        self._host_copy_seconds = 0.0
        self._transfer_count = 0
        self._transfer_milliseconds = 0.0
        self._maximum_pending_depth = 0
        self._maximum_ready_depth = 0

        with torch.cuda.device(device):
            self._transfer_stream = torch.cuda.Stream(device=device)
            self._slots = tuple(
                _StagingSlot(
                    slot_index=slot_index,
                    host_tensor=torch.empty(
                        frame_shape,
                        dtype=torch.float32,
                        device="cpu",
                        pin_memory=True,
                    ),
                    device_tensor=torch.empty(
                        frame_shape,
                        dtype=torch.float32,
                        device=device,
                    ),
                    copy_started_event=torch.cuda.Event(enable_timing=True),
                    copy_finished_event=torch.cuda.Event(enable_timing=True),
                    consumed_event=torch.cuda.Event(enable_timing=False),
                )
                for slot_index in range(slot_count)
            )

        self._pinned_host_bytes = sum(
            slot.host_tensor.numel() * slot.host_tensor.element_size()
            for slot in self._slots
        )
        self._device_bytes = sum(
            slot.device_tensor.numel() * slot.device_tensor.element_size()
            for slot in self._slots
        )
        self._worker = threading.Thread(
            target=self._run,
            name="sam2-gpu-frame-stager",
            daemon=True,
        )
        self._worker.start()

    def prefetch(self, frame_idx: int) -> None:
        if frame_idx < 0 or frame_idx >= len(self._frame_source):
            return
        with self._condition:
            self._raise_worker_error_locked()
            if self._closed:
                raise RuntimeError("GPU frame stager is closed")
            if self._is_resident_locked(frame_idx) or frame_idx in self._pending_index_set:
                self._resident_hits += 1
                return
            self._pending_indices.append(frame_idx)
            self._pending_index_set.add(frame_idx)
            self._requested_frames += 1
            self._maximum_pending_depth = max(
                self._maximum_pending_depth, len(self._pending_indices)
            )
            self._condition.notify_all()

    def prefetch_adjacent(self, frame_idx: int) -> None:
        with self._condition:
            previous_index = self._last_acquired_index
            if previous_index is not None and frame_idx != previous_index:
                self._prefetch_direction = 1 if frame_idx > previous_index else -1
            self._last_acquired_index = frame_idx
            adjacent_index = frame_idx + self._prefetch_direction
        self.prefetch(adjacent_index)

    def acquire(self, frame_idx: int) -> GpuFrameLease:
        if frame_idx < 0 or frame_idx >= len(self._frame_source):
            raise IndexError(f"frame index {frame_idx} is outside the video")
        self.prefetch(frame_idx)
        wait_started_at = time.perf_counter()
        waited = False
        with self._condition:
            while True:
                self._raise_worker_error_locked()
                if self._closed:
                    raise RuntimeError("GPU frame stager is closed")
                slot = self._ready_slot_locked(frame_idx)
                if slot is not None:
                    slot.state = "in_use"
                    if waited:
                        self._consumer_wait_count += 1
                        self._consumer_wait_seconds += (
                            time.perf_counter() - wait_started_at
                        )
                    else:
                        self._resident_hits += 1
                    break
                waited = True
                self._condition.wait()

        # This inserts a dependency into the inference stream without blocking
        # the CPU. The following model kernels cannot read the slot before H2D
        # has completed on the transfer stream.
        torch.cuda.current_stream(self._device).wait_event(slot.copy_finished_event)
        return GpuFrameLease(self, slot)

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._pending_indices.clear()
            self._pending_index_set.clear()
            self._condition.notify_all()
        self._worker.join()
        self._transfer_stream.synchronize()
        for slot in self._slots:
            self._record_transfer_timing(slot)

    def stats(self) -> GpuFrameStagerStats:
        with self._condition:
            mean_transfer_milliseconds = (
                self._transfer_milliseconds / self._transfer_count
                if self._transfer_count > 0
                else None
            )
            return GpuFrameStagerStats(
                capacity=len(self._slots),
                requested_frames=self._requested_frames,
                staged_frames=self._staged_frames,
                resident_hits=self._resident_hits,
                consumer_wait_count=self._consumer_wait_count,
                consumer_wait_seconds=self._consumer_wait_seconds,
                host_copy_seconds=self._host_copy_seconds,
                transfer_count=self._transfer_count,
                transfer_milliseconds=self._transfer_milliseconds,
                mean_transfer_milliseconds=mean_transfer_milliseconds,
                maximum_pending_depth=self._maximum_pending_depth,
                maximum_ready_depth=self._maximum_ready_depth,
                pinned_host_bytes=self._pinned_host_bytes,
                device_bytes=self._device_bytes,
                closed=self._closed,
            )

    def _release(self, slot: _StagingSlot) -> None:
        with self._condition:
            if slot.state != "in_use":
                raise RuntimeError(
                    f"GPU staging slot {slot.slot_index} is not in use"
                )

        # The event is recorded after every operation already enqueued by the
        # predictor. The worker cannot recycle this slot until the event fires.
        slot.consumed_event.record(torch.cuda.current_stream(self._device))
        with self._condition:
            slot.state = "retiring"
            self._condition.notify_all()

    @torch.inference_mode()
    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    reservation = self._wait_for_reservation_locked()
                    if reservation is None:
                        return
                    frame_idx, slot, previous_state = reservation

                self._prepare_slot_for_write(slot, previous_state)
                source_frame = self._frame_source[frame_idx]
                self._validate_source_frame(source_frame, slot)

                host_copy_started_at = time.perf_counter()
                slot.host_tensor.copy_(source_frame)
                host_copy_seconds = time.perf_counter() - host_copy_started_at

                with torch.cuda.stream(self._transfer_stream):
                    slot.copy_started_event.record(self._transfer_stream)
                    slot.device_tensor.copy_(slot.host_tensor, non_blocking=True)
                    slot.copy_finished_event.record(self._transfer_stream)
                slot.transfer_timing_pending = True

                with self._condition:
                    self._host_copy_seconds += host_copy_seconds
                    if self._closed:
                        slot.state = "free"
                        slot.frame_idx = None
                        self._condition.notify_all()
                        return
                    self._published_sequence += 1
                    slot.published_sequence = self._published_sequence
                    slot.state = "ready"
                    self._staged_frames += 1
                    ready_depth = sum(
                        candidate.state == "ready" for candidate in self._slots
                    )
                    self._maximum_ready_depth = max(
                        self._maximum_ready_depth, ready_depth
                    )
                    self._condition.notify_all()
        except BaseException as error:
            with self._condition:
                self._worker_error = error
                self._condition.notify_all()

    def _wait_for_reservation_locked(
        self,
    ) -> tuple[int, _StagingSlot, _SlotState] | None:
        while True:
            if self._closed:
                return None
            if self._pending_indices:
                slot = self._reusable_slot_locked()
                if slot is not None:
                    frame_idx = self._pending_indices.popleft()
                    self._pending_index_set.remove(frame_idx)
                    previous_state = slot.state
                    slot.state = "staging"
                    slot.frame_idx = frame_idx
                    return frame_idx, slot, previous_state
            self._condition.wait()

    def _reusable_slot_locked(self) -> _StagingSlot | None:
        for state in ("free", "retiring"):
            for slot in self._slots:
                if slot.state == state:
                    return slot
        ready_slots = [slot for slot in self._slots if slot.state == "ready"]
        if ready_slots:
            return min(ready_slots, key=lambda slot: slot.published_sequence)
        return None

    def _prepare_slot_for_write(
        self, slot: _StagingSlot, previous_state: _SlotState
    ) -> None:
        if previous_state == "retiring":
            slot.consumed_event.synchronize()
        elif previous_state == "ready":
            slot.copy_finished_event.synchronize()
        self._record_transfer_timing(slot)

    def _record_transfer_timing(self, slot: _StagingSlot) -> None:
        if not slot.transfer_timing_pending:
            return
        transfer_milliseconds = slot.copy_started_event.elapsed_time(
            slot.copy_finished_event
        )
        with self._condition:
            if not slot.transfer_timing_pending:
                return
            slot.transfer_timing_pending = False
            self._transfer_count += 1
            self._transfer_milliseconds += transfer_milliseconds

    def _validate_source_frame(
        self, source_frame: torch.Tensor, slot: _StagingSlot
    ) -> None:
        if source_frame.device.type != "cpu":
            raise ValueError("GPU frame stager requires CPU source frames")
        if source_frame.dtype != slot.host_tensor.dtype:
            raise ValueError(
                "GPU frame stager source dtype does not match its pinned buffers"
            )
        if source_frame.shape != slot.host_tensor.shape:
            raise ValueError(
                "GPU frame stager source shape does not match its fixed buffers"
            )

    def _ready_slot_locked(self, frame_idx: int) -> _StagingSlot | None:
        return next(
            (
                slot
                for slot in self._slots
                if slot.frame_idx == frame_idx and slot.state == "ready"
            ),
            None,
        )

    def _is_resident_locked(self, frame_idx: int) -> bool:
        return any(
            slot.frame_idx == frame_idx
            and slot.state in ("staging", "ready", "in_use")
            for slot in self._slots
        )

    def _raise_worker_error_locked(self) -> None:
        if self._worker_error is not None:
            raise RuntimeError("GPU frame staging worker failed") from self._worker_error
