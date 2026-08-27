from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable

import torch

from sam2.utils.gpu_frame_stager import GpuFrameLease
from sam2.utils.gpu_stage_profiler import GpuStageProfiler, GpuStageRole


BackboneOutput = dict[str, list[torch.Tensor]]


@dataclass(frozen=True)
class EncodedImageFeature:
    frame_idx: int
    image: torch.Tensor
    backbone_out: BackboneOutput
    frame_lease: GpuFrameLease | None


@dataclass(frozen=True)
class GpuImageFeaturePipelineStats:
    capacity: int
    requested_frames: int
    encoded_frames: int
    resident_hits: int
    skipped_prefetches: int
    consumer_wait_count: int
    consumer_wait_seconds: float
    maximum_pending_depth: int
    maximum_ready_depth: int
    maximum_feature_bytes: int
    mean_prepare_milliseconds: float | None
    closed: bool


@dataclass
class _PendingFeature:
    feature: EncodedImageFeature
    ready_event: torch.cuda.Event
    prepare_start_event: torch.cuda.Event


FeatureProducer = Callable[[int], EncodedImageFeature]


class GpuImageFeaturePipeline:
    """Keep one image-backbone result ahead on a dedicated CUDA stream."""

    def __init__(
        self,
        *,
        producer: FeatureProducer,
        frame_count: int,
        device: torch.device,
        autocast_enabled: bool,
        autocast_dtype: torch.dtype,
        profiler: GpuStageProfiler | None = None,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("concurrent image encoding requires a CUDA device")
        if frame_count <= 0:
            raise ValueError("concurrent image encoding requires at least one frame")
        self._producer = producer
        self._frame_count = frame_count
        self._device = device
        self._autocast_enabled = autocast_enabled
        self._autocast_dtype = autocast_dtype
        self._profiler = profiler
        self._condition = threading.Condition()
        self._requested: tuple[int, GpuStageRole] | None = None
        self._encoding_frame_idx: int | None = None
        self._ready: _PendingFeature | None = None
        self._worker_error: BaseException | None = None
        self._closed = False
        self._prepare_event_pairs: list[
            tuple[torch.cuda.Event, torch.cuda.Event]
        ] = []

        self._requested_frames = 0
        self._encoded_frames = 0
        self._resident_hits = 0
        self._skipped_prefetches = 0
        self._consumer_wait_count = 0
        self._consumer_wait_seconds = 0.0
        self._maximum_pending_depth = 0
        self._maximum_ready_depth = 0
        self._maximum_feature_bytes = 0

        with torch.cuda.device(device):
            self._encoder_stream = torch.cuda.Stream(device=device)
        self._worker = threading.Thread(
            target=self._run,
            name="sam2-image-encoder-prefetch",
            daemon=True,
        )
        self._worker.start()

    def prefetch(self, frame_idx: int) -> bool:
        if frame_idx < 0 or frame_idx >= self._frame_count:
            return False
        with self._condition:
            self._raise_worker_error_locked()
            if self._closed:
                raise RuntimeError("image feature pipeline is closed")
            if self._contains_frame_locked(frame_idx):
                self._resident_hits += 1
                return True
            if self._has_work_locked():
                self._skipped_prefetches += 1
                return False
            self._request_locked(frame_idx, "prefetch")
            return True

    def acquire(self, frame_idx: int) -> EncodedImageFeature:
        if frame_idx < 0 or frame_idx >= self._frame_count:
            raise IndexError(f"frame index {frame_idx} is outside the video")
        wait_started_at = time.perf_counter()
        waited = False
        stale: _PendingFeature | None = None
        while True:
            if stale is not None:
                self._dispose(stale)
                stale = None
            with self._condition:
                self._raise_worker_error_locked()
                if self._closed:
                    raise RuntimeError("image feature pipeline is closed")
                if (
                    self._ready is not None
                    and self._ready.feature.frame_idx == frame_idx
                ):
                    pending = self._ready
                    self._ready = None
                    if waited:
                        self._consumer_wait_count += 1
                        self._consumer_wait_seconds += (
                            time.perf_counter() - wait_started_at
                        )
                    else:
                        self._resident_hits += 1
                    self._condition.notify_all()
                    break
                if self._ready is not None:
                    stale = self._ready
                    self._ready = None
                    self._condition.notify_all()
                    continue
                if not self._has_work_locked():
                    self._request_locked(frame_idx, "demand")
                waited = True
                self._condition.wait()

        ready_before_acquire = pending.ready_event.query()
        consumer_stream = torch.cuda.current_stream(self._device)
        consumer_stream.wait_event(pending.ready_event)
        for tensor in _feature_tensors(pending.feature):
            tensor.record_stream(consumer_stream)
        if not ready_before_acquire:
            with self._condition:
                self._consumer_wait_count += 1
        return pending.feature

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._requested = None
            self._condition.notify_all()
        self._worker.join()
        self._encoder_stream.synchronize()
        with self._condition:
            ready = self._ready
            self._ready = None
        if ready is not None:
            self._dispose(ready)

    def stats(self) -> GpuImageFeaturePipelineStats:
        completed_durations = [
            start.elapsed_time(end)
            for start, end in self._prepare_event_pairs
            if end.query()
        ]
        with self._condition:
            return GpuImageFeaturePipelineStats(
                capacity=1,
                requested_frames=self._requested_frames,
                encoded_frames=self._encoded_frames,
                resident_hits=self._resident_hits,
                skipped_prefetches=self._skipped_prefetches,
                consumer_wait_count=self._consumer_wait_count,
                consumer_wait_seconds=self._consumer_wait_seconds,
                maximum_pending_depth=self._maximum_pending_depth,
                maximum_ready_depth=self._maximum_ready_depth,
                maximum_feature_bytes=self._maximum_feature_bytes,
                mean_prepare_milliseconds=(
                    sum(completed_durations) / len(completed_durations)
                    if completed_durations
                    else None
                ),
                closed=self._closed,
            )

    @torch.inference_mode()
    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    while self._requested is None and not self._closed:
                        self._condition.wait()
                    if self._closed:
                        return
                    frame_idx, role = self._requested
                    self._requested = None
                    self._encoding_frame_idx = frame_idx

                with (
                    torch.cuda.device(self._device),
                    torch.cuda.stream(self._encoder_stream),
                    torch.autocast(
                        device_type="cuda",
                        dtype=self._autocast_dtype,
                        enabled=self._autocast_enabled,
                    ),
                    _profile_frame(self._profiler, frame_idx, role),
                ):
                    prepare_start = torch.cuda.Event(enable_timing=True)
                    ready_event = torch.cuda.Event(enable_timing=True)
                    prepare_start.record(self._encoder_stream)
                    feature = self._producer(frame_idx)
                    ready_event.record(self._encoder_stream)

                pending = _PendingFeature(
                    feature=feature,
                    ready_event=ready_event,
                    prepare_start_event=prepare_start,
                )
                with self._condition:
                    self._encoding_frame_idx = None
                    if self._closed:
                        self._condition.notify_all()
                    else:
                        self._ready = pending
                        self._prepare_event_pairs.append((prepare_start, ready_event))
                        self._encoded_frames += 1
                        self._maximum_ready_depth = 1
                        self._maximum_feature_bytes = max(
                            self._maximum_feature_bytes,
                            _feature_bytes(feature),
                        )
                        self._condition.notify_all()
                if self._closed:
                    self._dispose(pending)
                    return
        except BaseException as error:
            with self._condition:
                self._worker_error = error
                self._encoding_frame_idx = None
                self._condition.notify_all()

    def _request_locked(self, frame_idx: int, role: GpuStageRole) -> None:
        self._requested = (frame_idx, role)
        self._requested_frames += 1
        self._maximum_pending_depth = 1
        self._condition.notify_all()

    def _contains_frame_locked(self, frame_idx: int) -> bool:
        return (
            self._requested is not None and self._requested[0] == frame_idx
        ) or self._encoding_frame_idx == frame_idx or (
            self._ready is not None and self._ready.feature.frame_idx == frame_idx
        )

    def _has_work_locked(self) -> bool:
        return (
            self._requested is not None
            or self._encoding_frame_idx is not None
            or self._ready is not None
        )

    def _raise_worker_error_locked(self) -> None:
        if self._worker_error is not None:
            raise RuntimeError("concurrent image encoder failed") from self._worker_error

    @staticmethod
    def _dispose(pending: _PendingFeature) -> None:
        pending.ready_event.synchronize()
        if pending.feature.frame_lease is not None:
            pending.feature.frame_lease.release()


@contextmanager
def _profile_frame(
    profiler: GpuStageProfiler | None,
    frame_idx: int,
    role: GpuStageRole,
):
    if profiler is None:
        yield
        return
    with profiler.frame(frame_idx, role):
        yield


def _feature_tensors(feature: EncodedImageFeature) -> tuple[torch.Tensor, ...]:
    return (feature.image,) + tuple(
        tensor
        for values in feature.backbone_out.values()
        for tensor in values
    )


def _feature_bytes(feature: EncodedImageFeature) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in _feature_tensors(feature))
