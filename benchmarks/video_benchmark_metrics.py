from __future__ import annotations

import hashlib
import json
import os
import platform
import statistics
import threading
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence, TextIO, TypeVar

import torch


@dataclass(frozen=True)
class PromptSpec:
    frame_idx: int
    obj_id: int
    points: tuple[tuple[float, float], ...]
    labels: tuple[int, ...]

    @classmethod
    def from_json_file(cls, path: Path) -> "PromptSpec":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("prompt JSON must contain an object")

        frame_idx = raw.get("frame_idx")
        obj_id = raw.get("obj_id")
        raw_points = raw.get("points")
        raw_labels = raw.get("labels")
        if not isinstance(frame_idx, int) or frame_idx < 0:
            raise ValueError("prompt frame_idx must be a non-negative integer")
        if not isinstance(obj_id, int):
            raise ValueError("prompt obj_id must be an integer")
        if not isinstance(raw_points, list) or not raw_points:
            raise ValueError("prompt points must be a non-empty list")
        if not isinstance(raw_labels, list):
            raise ValueError("prompt labels must be a list")

        points: list[tuple[float, float]] = []
        for raw_point in raw_points:
            if (
                not isinstance(raw_point, list)
                or len(raw_point) != 2
                or not all(isinstance(value, (int, float)) for value in raw_point)
            ):
                raise ValueError("each prompt point must be an [x, y] number pair")
            points.append((float(raw_point[0]), float(raw_point[1])))

        if len(raw_labels) != len(points):
            raise ValueError("prompt labels must have the same length as points")
        labels: list[int] = []
        for raw_label in raw_labels:
            if not isinstance(raw_label, int) or raw_label not in (0, 1):
                raise ValueError("prompt labels must contain only 0 or 1")
            labels.append(raw_label)

        return cls(
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=tuple(points),
            labels=tuple(labels),
        )


@dataclass(frozen=True)
class BenchmarkConfig:
    mode: str
    video_path: str
    model_config: str
    checkpoint_path: str
    prompt_path: str
    device: str
    dtype: str
    compile_image_encoder: bool
    max_frames: int | None
    warmup_frames: int
    repeat_index: int
    chunk_size: int
    max_dimension: int
    memory_sample_interval_seconds: float
    offload_video_to_cpu: bool
    offload_state_to_cpu: bool


@dataclass(frozen=True)
class FrameTrace:
    repeat_index: int
    global_frame_idx: int
    source_frame_idx: int
    chunk_idx: int | None
    object_ids: tuple[int, ...]
    mask_checksums: tuple[str, ...]
    gpu_inference_ms: float
    mask_materialization_ms: float
    output_processing_ms: float
    frame_wall_ms: float
    elapsed_seconds: float


@dataclass(frozen=True)
class MemorySample:
    elapsed_seconds: float
    stage: str
    frame_idx: int | None
    torch_allocated_bytes: int | None
    torch_reserved_bytes: int | None
    torch_peak_allocated_bytes: int | None
    nvml_device_used_bytes: int | None
    nvml_process_bytes: int | None
    process_rss_bytes: int


@dataclass(frozen=True)
class StageDurations:
    model_load_seconds: float
    video_prepare_seconds: float
    init_state_seconds: float
    prompt_seconds: float
    propagation_seconds: float
    output_encode_seconds: float
    measured_pipeline_seconds: float


@dataclass(frozen=True)
class RunSummary:
    status: str
    error: str | None
    mode: str
    repeat_index: int
    frames_processed: int
    warmup_frames: int
    core_gpu_fps: float | None
    propagation_fps: float | None
    end_to_end_fps: float | None
    steady_state_fps: float | None
    gpu_inference_latency_ms: MetricDistribution
    frame_wall_latency_ms: MetricDistribution
    mask_materialization_latency_ms: MetricDistribution
    peak_torch_allocated_bytes: int | None
    peak_torch_reserved_bytes: int | None
    peak_nvml_device_used_bytes: int | None
    peak_nvml_process_bytes: int | None
    peak_process_rss_bytes: int | None
    stages: StageDurations


@dataclass(frozen=True)
class MetricDistribution:
    count: int
    mean: float | None
    median: float | None
    p95: float | None
    minimum: float | None
    maximum: float | None
    standard_deviation: float | None
    coefficient_of_variation: float | None


@dataclass(frozen=True)
class AggregateSummary:
    completed_runs: int
    failed_runs: int
    core_gpu_fps: MetricDistribution
    propagation_fps: MetricDistribution
    end_to_end_fps: MetricDistribution
    steady_state_fps: MetricDistribution


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def repository_revision(repository_root: Path) -> str | None:
    git_entry = repository_root / ".git"
    if git_entry.is_file():
        content = git_entry.read_text(encoding="utf-8").strip()
        if not content.startswith("gitdir: "):
            return None
        git_dir = (repository_root / content.removeprefix("gitdir: ")).resolve()
    elif git_entry.is_dir():
        git_dir = git_entry
    else:
        return None

    head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head
    ref_name = head.removeprefix("ref: ")
    loose_ref = git_dir / ref_name
    if loose_ref.exists():
        return loose_ref.read_text(encoding="utf-8").strip()
    packed_refs = git_dir / "packed-refs"
    if packed_refs.exists():
        for line in packed_refs.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith(("#", "^")):
                revision, packed_ref_name = line.split(" ", maxsplit=1)
                if packed_ref_name == ref_name:
                    return revision
    return None


def runtime_metadata(device: torch.device) -> dict[str, object]:
    gpu_name: str | None = None
    cudnn_version: int | None = None
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        cudnn_version = torch.backends.cudnn.version()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": cudnn_version,
        "gpu": gpu_name,
    }


def materialize_mask_checksums(
    mask_logits: torch.Tensor, object_ids: Sequence[int]
) -> tuple[tuple[str, ...], torch.Tensor]:
    binary_masks = (mask_logits > 0).to(device="cpu", dtype=torch.uint8).contiguous()
    if binary_masks.shape[0] != len(object_ids):
        raise ValueError("mask batch size does not match object ID count")

    checksums: list[str] = []
    for object_id, mask in zip(object_ids, binary_masks):
        digest = hashlib.sha256()
        digest.update(str(object_id).encode("ascii"))
        digest.update(str(tuple(mask.shape)).encode("ascii"))
        digest.update(mask.numpy().tobytes())
        checksums.append(digest.hexdigest())
    return tuple(checksums), binary_masks


def _read_process_rss_bytes() -> int:
    statm = Path("/proc/self/statm")
    if statm.exists():
        resident_pages = int(statm.read_text(encoding="ascii").split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")

    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss * 1024)


class NvmlMemory:
    def __init__(self, device_index: int) -> None:
        try:
            import pynvml
        except ImportError as error:
            raise RuntimeError(
                "nvidia-ml-py is required for a valid CUDA benchmark"
            ) from error

        self._pynvml = pynvml
        self._pynvml.nvmlInit()
        self._handle = self._pynvml.nvmlDeviceGetHandleByIndex(device_index)
        driver_version = self._pynvml.nvmlSystemGetDriverVersion()
        self._driver_version = (
            driver_version.decode("utf-8")
            if isinstance(driver_version, bytes)
            else str(driver_version)
        )
        self._pid = os.getpid()
        self._closed = False

    @property
    def driver_version(self) -> str:
        return self._driver_version

    def used_bytes(self) -> int:
        processes = self._pynvml.nvmlDeviceGetComputeRunningProcesses(self._handle)
        return sum(
            int(process.usedGpuMemory)
            for process in processes
            if process.pid == self._pid and process.usedGpuMemory is not None
        )

    def device_used_bytes(self) -> int:
        return int(self._pynvml.nvmlDeviceGetMemoryInfo(self._handle).used)

    def close(self) -> None:
        if not self._closed:
            self._pynvml.nvmlShutdown()
            self._closed = True


class MemorySampler:
    def __init__(
        self,
        device: torch.device,
        interval_seconds: float,
        sample_sink: Callable[[MemorySample], None] | None = None,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("memory sample interval must be positive")
        self._device = device
        self._interval_seconds = interval_seconds
        self._start_time = time.perf_counter()
        self._samples: list[MemorySample] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._stage = "created"
        self._frame_idx: int | None = None
        self._failure: BaseException | None = None
        self._sample_sink = sample_sink
        self._nvml = (
            NvmlMemory(device.index or torch.cuda.current_device())
            if device.type == "cuda"
            else None
        )

    @property
    def driver_version(self) -> str | None:
        return None if self._nvml is None else self._nvml.driver_version

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("memory sampler has already started")
        self._thread = threading.Thread(
            target=self._sample_loop,
            name="video-benchmark-memory-sampler",
            daemon=True,
        )
        self._thread.start()

    def set_position(self, stage: str, frame_idx: int | None = None) -> None:
        with self._lock:
            self._stage = stage
            self._frame_idx = frame_idx

    def capture(self, stage: str, frame_idx: int | None = None) -> None:
        with self._lock:
            self._stage = stage
            self._frame_idx = frame_idx
            sample = self._take_sample(stage, frame_idx)
            self._samples.append(sample)
        self._publish_sample(sample)

    def stop(self) -> tuple[MemorySample, ...]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        self.capture("sampler_stopped", self._frame_idx)
        if self._nvml is not None:
            self._nvml.close()
        if self._failure is not None:
            raise RuntimeError("memory sampler failed") from self._failure
        return tuple(self._samples)

    def _sample_loop(self) -> None:
        try:
            while not self._stop_event.wait(self._interval_seconds):
                with self._lock:
                    sample = self._take_sample(self._stage, self._frame_idx)
                    self._samples.append(sample)
                self._publish_sample(sample)
        except BaseException as error:
            self._failure = error
            self._stop_event.set()

    def _publish_sample(self, sample: MemorySample) -> None:
        if self._sample_sink is not None:
            self._sample_sink(sample)

    def _take_sample(self, stage: str, frame_idx: int | None) -> MemorySample:
        if self._device.type == "cuda":
            allocated = torch.cuda.memory_allocated(self._device)
            reserved = torch.cuda.memory_reserved(self._device)
            peak_allocated = torch.cuda.max_memory_allocated(self._device)
            nvml_device_bytes = (
                self._nvml.device_used_bytes() if self._nvml is not None else None
            )
            nvml_bytes = self._nvml.used_bytes() if self._nvml is not None else None
        else:
            allocated = None
            reserved = None
            peak_allocated = None
            nvml_device_bytes = None
            nvml_bytes = None
        return MemorySample(
            elapsed_seconds=time.perf_counter() - self._start_time,
            stage=stage,
            frame_idx=frame_idx,
            torch_allocated_bytes=allocated,
            torch_reserved_bytes=reserved,
            torch_peak_allocated_bytes=peak_allocated,
            nvml_device_used_bytes=nvml_device_bytes,
            nvml_process_bytes=nvml_bytes,
            process_rss_bytes=_read_process_rss_bytes(),
        )


def build_run_summary(
    *,
    status: str,
    error: str | None,
    config: BenchmarkConfig,
    traces: Sequence[FrameTrace],
    samples: Sequence[MemorySample],
    stages: StageDurations,
) -> RunSummary:
    frames_processed = len(traces)
    total_gpu_seconds = sum(trace.gpu_inference_ms for trace in traces) / 1000.0
    steady_traces = traces[min(config.warmup_frames, frames_processed) :]
    steady_seconds = sum(trace.frame_wall_ms for trace in steady_traces) / 1000.0

    def fps(frame_count: int, seconds: float) -> float | None:
        return frame_count / seconds if frame_count > 0 and seconds > 0 else None

    return RunSummary(
        status=status,
        error=error,
        mode=config.mode,
        repeat_index=config.repeat_index,
        frames_processed=frames_processed,
        warmup_frames=min(config.warmup_frames, frames_processed),
        core_gpu_fps=fps(frames_processed, total_gpu_seconds),
        propagation_fps=fps(frames_processed, stages.propagation_seconds),
        end_to_end_fps=fps(frames_processed, stages.measured_pipeline_seconds),
        steady_state_fps=fps(len(steady_traces), steady_seconds),
        gpu_inference_latency_ms=_distribution(
            trace.gpu_inference_ms for trace in traces
        ),
        frame_wall_latency_ms=_distribution(trace.frame_wall_ms for trace in traces),
        mask_materialization_latency_ms=_distribution(
            trace.mask_materialization_ms for trace in traces
        ),
        peak_torch_allocated_bytes=_optional_max(
            sample.torch_peak_allocated_bytes for sample in samples
        ),
        peak_torch_reserved_bytes=_optional_max(
            sample.torch_reserved_bytes for sample in samples
        ),
        peak_nvml_device_used_bytes=_optional_max(
            sample.nvml_device_used_bytes for sample in samples
        ),
        peak_nvml_process_bytes=_optional_max(
            sample.nvml_process_bytes for sample in samples
        ),
        peak_process_rss_bytes=max(
            (sample.process_rss_bytes for sample in samples), default=None
        ),
        stages=stages,
    )


def validate_run_artifacts(
    summary: RunSummary,
    traces: Sequence[FrameTrace],
    samples: Sequence[MemorySample],
) -> None:
    errors: list[str] = []
    if summary.frames_processed != len(traces):
        errors.append("summary frame count does not match frame trace")
    if any(trace.frame_wall_ms < 0 for trace in traces):
        errors.append("frame trace contains a negative duration")
    if any(
        len(trace.object_ids) != len(trace.mask_checksums) for trace in traces
    ):
        errors.append("frame trace object and checksum counts differ")
    if any(
        later.elapsed_seconds < earlier.elapsed_seconds
        for earlier, later in zip(samples, samples[1:])
    ):
        errors.append("memory sample timestamps are not monotonic")
    if summary.status == "completed" and not traces:
        errors.append("completed run contains no frame traces")
    if summary.status == "failed" and summary.error is None:
        errors.append("failed run does not record an error")
    if summary.stages.measured_pipeline_seconds < 0:
        errors.append("pipeline duration is negative")
    if errors:
        raise ValueError("invalid benchmark artifacts: " + "; ".join(errors))


def aggregate_summaries(summaries: Sequence[RunSummary]) -> AggregateSummary:
    completed = [summary for summary in summaries if summary.status == "completed"]
    return AggregateSummary(
        completed_runs=len(completed),
        failed_runs=len(summaries) - len(completed),
        core_gpu_fps=_distribution(summary.core_gpu_fps for summary in completed),
        propagation_fps=_distribution(
            summary.propagation_fps for summary in completed
        ),
        end_to_end_fps=_distribution(
            summary.end_to_end_fps for summary in completed
        ),
        steady_state_fps=_distribution(
            summary.steady_state_fps for summary in completed
        ),
    )


RecordT = TypeVar("RecordT")


def write_json(path: Path, record: object) -> None:
    path.write_text(
        json.dumps(_json_value(record), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: Iterable[RecordT]) -> None:
    with path.open("w", encoding="utf-8") as output:
        _write_jsonl_records(output, records)


def _write_jsonl_records(output: TextIO, records: Iterable[RecordT]) -> None:
    for record in records:
        output.write(json.dumps(_json_value(record), sort_keys=True) + "\n")


def _json_value(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _optional_max(values: Iterable[int | None]) -> int | None:
    present = [value for value in values if value is not None]
    return max(present) if present else None


def _distribution(values: Iterable[float | None]) -> MetricDistribution:
    present = [value for value in values if value is not None]
    if not present:
        return MetricDistribution(
            count=0,
            mean=None,
            median=None,
            p95=None,
            minimum=None,
            maximum=None,
            standard_deviation=None,
            coefficient_of_variation=None,
        )
    mean = statistics.fmean(present)
    standard_deviation = statistics.stdev(present) if len(present) > 1 else None
    coefficient = (
        standard_deviation / mean
        if standard_deviation is not None and mean != 0
        else None
    )
    return MetricDistribution(
        count=len(present),
        mean=mean,
        median=statistics.median(present),
        p95=(
            statistics.quantiles(present, n=100, method="inclusive")[94]
            if len(present) > 1
            else present[0]
        ),
        minimum=min(present),
        maximum=max(present),
        standard_deviation=standard_deviation,
        coefficient_of_variation=coefficient,
    )
