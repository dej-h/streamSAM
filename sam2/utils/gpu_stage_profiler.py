from __future__ import annotations

import contextvars
import math
import statistics
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Literal, Mapping

import torch


GpuStageRole = Literal["demand", "prefetch", "tracking", "prompt", "memory"]


@dataclass(frozen=True)
class GpuStageEvent:
    frame_idx: int | None
    role: GpuStageRole
    stage: str
    stream_id: int
    start_milliseconds: float
    duration_milliseconds: float
    end_milliseconds: float


@dataclass(frozen=True)
class GpuStageSummary:
    role: GpuStageRole
    stage: str
    count: int
    total_milliseconds: float
    mean_milliseconds: float
    median_milliseconds: float
    p95_milliseconds: float
    maximum_milliseconds: float


@dataclass(frozen=True)
class GpuStageProfile:
    device: str
    span_milliseconds: float
    events: tuple[GpuStageEvent, ...]
    summaries: tuple[GpuStageSummary, ...]


@dataclass(frozen=True)
class _FrameContext:
    frame_idx: int | None
    role: GpuStageRole


@dataclass
class _PendingEvent:
    frame_idx: int | None
    role: GpuStageRole
    stage: str
    stream_id: int
    start_event: torch.cuda.Event
    end_event: torch.cuda.Event | None = None


class GpuStageProfiler:
    """Record module-level CUDA work without synchronizing the measured loop."""

    def __init__(self, device: torch.device, *, emit_nvtx: bool = True) -> None:
        if device.type != "cuda":
            raise ValueError("GPU stage profiling requires a CUDA device")
        self._device = device
        self._emit_nvtx = emit_nvtx
        self._frame_context = contextvars.ContextVar(
            "sam2_gpu_stage_context",
            default=_FrameContext(frame_idx=None, role="tracking"),
        )
        self._lock = threading.Lock()
        self._thread_local = threading.local()
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._completed_events: list[_PendingEvent] = []
        self._origin_event: torch.cuda.Event | None = None
        self._started = False

    def attach(self, modules: Mapping[str, torch.nn.Module]) -> None:
        if self._handles:
            raise RuntimeError("GPU stage profiler is already attached")
        seen_modules: set[int] = set()
        for stage, module in modules.items():
            module_identity = id(module)
            if module_identity in seen_modules:
                continue
            seen_modules.add(module_identity)
            self._handles.append(
                module.register_forward_pre_hook(
                    self._pre_hook(stage),
                    with_kwargs=True,
                )
            )
            self._handles.append(
                module.register_forward_hook(
                    self._post_hook(stage),
                    with_kwargs=True,
                    always_call=True,
                )
            )

    def start(self) -> None:
        if self._started:
            raise RuntimeError("GPU stage profiler has already started")
        with torch.cuda.device(self._device):
            self._origin_event = torch.cuda.Event(enable_timing=True)
            self._origin_event.record(torch.cuda.current_stream(self._device))
            self._origin_event.synchronize()
        self._started = True

    @contextmanager
    def frame(self, frame_idx: int, role: GpuStageRole) -> Iterator[None]:
        token = self._frame_context.set(_FrameContext(frame_idx, role))
        try:
            yield
        finally:
            self._frame_context.reset(token)

    def finish(self) -> GpuStageProfile:
        if not self._started or self._origin_event is None:
            raise RuntimeError("GPU stage profiler has not been started")
        torch.cuda.synchronize(self._device)
        with self._lock:
            pending_events = tuple(self._completed_events)
        events: list[GpuStageEvent] = []
        for pending in pending_events:
            if pending.end_event is None:
                raise RuntimeError(
                    f"GPU stage {pending.stage!r} did not record an end event"
                )
            start_ms = self._origin_event.elapsed_time(pending.start_event)
            duration_ms = pending.start_event.elapsed_time(pending.end_event)
            events.append(
                GpuStageEvent(
                    frame_idx=pending.frame_idx,
                    role=pending.role,
                    stage=pending.stage,
                    stream_id=pending.stream_id,
                    start_milliseconds=start_ms,
                    duration_milliseconds=duration_ms,
                    end_milliseconds=start_ms + duration_ms,
                )
            )
        events.sort(key=lambda event: event.start_milliseconds)
        span_ms = max((event.end_milliseconds for event in events), default=0.0)
        return GpuStageProfile(
            device=str(self._device),
            span_milliseconds=span_ms,
            events=tuple(events),
            summaries=_summarize_events(events),
        )

    def detach(self) -> None:
        while self._handles:
            self._handles.pop().remove()

    def _pre_hook(self, stage: str):
        def hook(
            module: torch.nn.Module,
            args: tuple[object, ...],
            kwargs: dict[str, object],
        ) -> None:
            del module, args, kwargs
            if not self._started:
                return
            context = self._frame_context.get()
            stream = torch.cuda.current_stream(self._device)
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record(stream)
            pending = _PendingEvent(
                frame_idx=context.frame_idx,
                role=context.role,
                stage=stage,
                stream_id=int(stream.cuda_stream),
                start_event=start_event,
            )
            self._stack(stage).append(pending)
            if self._emit_nvtx:
                torch.cuda.nvtx.range_push(
                    f"sam2:{stage}:frame={context.frame_idx}:role={context.role}"
                )

        return hook

    def _post_hook(self, stage: str):
        def hook(
            module: torch.nn.Module,
            args: tuple[object, ...],
            kwargs: dict[str, object],
            output: object,
        ) -> None:
            del module, args, kwargs, output
            stack = self._stack(stage)
            if not stack:
                return
            pending = stack.pop()
            if self._emit_nvtx:
                torch.cuda.nvtx.range_pop()
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record(torch.cuda.current_stream(self._device))
            pending.end_event = end_event
            with self._lock:
                self._completed_events.append(pending)

        return hook

    def _stack(self, stage: str) -> list[_PendingEvent]:
        stacks = getattr(self._thread_local, "stacks", None)
        if stacks is None:
            stacks = {}
            self._thread_local.stacks = stacks
        stage_stack = stacks.get(stage)
        if stage_stack is None:
            stage_stack = []
            stacks[stage] = stage_stack
        return stage_stack


def gpu_stage_profile_to_chrome_trace(
    profile: GpuStageProfile,
) -> dict[str, object]:
    trace_events: list[dict[str, object]] = []
    for event in profile.events:
        trace_events.append(
            {
                "name": event.stage,
                "cat": f"sam2.gpu.{event.role}",
                "ph": "X",
                "ts": event.start_milliseconds * 1000.0,
                "dur": event.duration_milliseconds * 1000.0,
                "pid": 0,
                "tid": event.stream_id,
                "args": {
                    "frame_idx": event.frame_idx,
                    "role": event.role,
                    "device": profile.device,
                },
            }
        )
    return {
        "displayTimeUnit": "ms",
        "traceEvents": trace_events,
    }


def _summarize_events(
    events: list[GpuStageEvent],
) -> tuple[GpuStageSummary, ...]:
    grouped: dict[tuple[GpuStageRole, str], list[float]] = {}
    for event in events:
        grouped.setdefault((event.role, event.stage), []).append(
            event.duration_milliseconds
        )

    summaries: list[GpuStageSummary] = []
    for (role, stage), durations in sorted(grouped.items()):
        ordered = sorted(durations)
        p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
        summaries.append(
            GpuStageSummary(
                role=role,
                stage=stage,
                count=len(ordered),
                total_milliseconds=sum(ordered),
                mean_milliseconds=statistics.fmean(ordered),
                median_milliseconds=statistics.median(ordered),
                p95_milliseconds=ordered[p95_index],
                maximum_milliseconds=ordered[-1],
            )
        )
    return tuple(summaries)
