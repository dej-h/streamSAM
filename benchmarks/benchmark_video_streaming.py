from __future__ import annotations

import argparse
import contextlib
import json
import queue
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Literal, Sequence, cast

import cv2
import numpy as np
import torch
from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

from benchmarks.video_benchmark_metrics import (
    BenchmarkConfig,
    FrameTrace,
    MemorySampler,
    MemorySample,
    PromptSpec,
    RunSummary,
    StageDurations,
    aggregate_summaries,
    build_run_summary,
    materialize_mask_checksums,
    repository_revision,
    runtime_metadata,
    sha256_file,
    validate_run_artifacts,
    write_json,
    write_jsonl,
)
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor


BenchmarkMode = Literal["core-eager", "demo-chunked"]
DTypeName = Literal["float32", "float16", "bfloat16"]
LiveMetricPayload = MemorySample | FrameTrace | RunSummary


class PositiveInteger(int):
    def __new__(cls, value: str) -> "PositiveInteger":
        try:
            parsed = int(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError("must be an integer") from error
        if parsed <= 0:
            raise argparse.ArgumentTypeError("must be greater than zero")
        return int.__new__(cls, parsed)


class NonNegativeInteger(int):
    def __new__(cls, value: str) -> "NonNegativeInteger":
        try:
            parsed = int(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError("must be an integer") from error
        if parsed < 0:
            raise argparse.ArgumentTypeError("must be zero or greater")
        return int.__new__(cls, parsed)


class PositiveFloat(float):
    def __new__(cls, value: str) -> "PositiveFloat":
        try:
            parsed = float(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError("must be a number") from error
        if parsed <= 0:
            raise argparse.ArgumentTypeError("must be greater than zero")
        return float.__new__(cls, parsed)


@dataclass(frozen=True)
class LiveMetric:
    event: Literal["memory", "frame", "summary"]
    payload: LiveMetricPayload


class LiveMetricsReporter:
    """Write telemetry without making inference wait for console or file I/O."""

    def __init__(self, output_path: Path | None) -> None:
        self._output_path = output_path
        self._queue: queue.Queue[LiveMetric | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._failure: BaseException | None = None

    def start(self) -> None:
        if self._output_path is None:
            return
        if self._thread is not None:
            raise RuntimeError("live metrics reporter has already started")
        self._thread = threading.Thread(
            target=self._run,
            name="video-benchmark-live-metrics",
            daemon=True,
        )
        self._thread.start()

    def publish_memory(self, sample: MemorySample) -> None:
        self._publish(LiveMetric("memory", sample))

    def publish_frame(self, trace: FrameTrace) -> None:
        self._publish(LiveMetric("frame", trace))

    def publish_summary(self, summary: RunSummary) -> None:
        self._publish(LiveMetric("summary", summary))

    def close(self) -> None:
        if self._thread is None:
            return
        self._queue.put_nowait(None)
        self._thread.join()
        if self._failure is not None:
            raise RuntimeError("live metrics reporter failed") from self._failure

    def _publish(self, metric: LiveMetric) -> None:
        if self._thread is not None:
            self._queue.put_nowait(metric)

    def _run(self) -> None:
        if self._output_path is None:
            return
        try:
            with self._output_path.open("w", encoding="utf-8") as output:
                while True:
                    metric = self._queue.get()
                    if metric is None:
                        return
                    payload = asdict(metric.payload)
                    payload["event"] = metric.event
                    line = json.dumps(payload, sort_keys=True)
                    output.write(line + "\n")
                    output.flush()
                    print(line, file=sys.stderr, flush=True)
        except BaseException as error:
            self._failure = error


@dataclass(frozen=True)
class CliArguments:
    mode: BenchmarkMode
    video_path: Path
    model_config: str
    checkpoint_path: Path
    prompt_path: Path
    output_directory: Path
    device: torch.device
    dtype_name: DTypeName
    compile_image_encoder: bool
    max_frames: int | None
    warmup_frames: int
    repeats: int
    chunk_size: int
    max_dimension: int
    memory_sample_interval_seconds: float
    offload_video_to_cpu: bool
    offload_state_to_cpu: bool
    live_metrics: bool


@dataclass(frozen=True)
class LegacyChunk:
    path: Path
    frame_count: int


@dataclass(frozen=True)
class PreparedChunks:
    chunks: tuple[LegacyChunk, ...]
    fps: float
    frame_count: int
    width: int
    height: int


@dataclass(frozen=True)
class Prediction:
    frame_idx: int
    object_ids: tuple[int, ...]
    mask_logits: torch.Tensor
    mask_checksums: tuple[str, ...]
    binary_masks: torch.Tensor
    gpu_inference_ms: float
    mask_materialization_ms: float
    started_at: float


@dataclass(frozen=True)
class ModeResult:
    traces: tuple[FrameTrace, ...]
    video_prepare_seconds: float
    init_state_seconds: float
    prompt_seconds: float
    propagation_seconds: float
    output_encode_seconds: float
    measured_pipeline_seconds: float


def parse_arguments(argv: Sequence[str] | None = None) -> CliArguments:
    parser = argparse.ArgumentParser(
        description="Measure the existing eager EdgeTAM video paths before streaming changes."
    )
    parser.add_argument(
        "--mode", choices=("core-eager", "demo-chunked"), required=True
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument(
        "--output-directory", type=Path, default=Path("benchmark_artifacts")
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument(
        "--compile-image-encoder",
        action="store_true",
        help="compile the image encoder with the model's torch.compile configuration",
    )
    parser.add_argument("--max-frames", type=PositiveInteger)
    parser.add_argument(
        "--warmup-frames",
        type=NonNegativeInteger,
        default=0,
        help=(
            "exclude the first N processed frames from steady_state_fps only; "
            "the frames remain in all other metrics and artifacts"
        ),
    )
    parser.add_argument("--repeats", type=PositiveInteger, default=1)
    parser.add_argument("--chunk-size", type=PositiveInteger, default=96)
    parser.add_argument("--max-dimension", type=PositiveInteger, default=960)
    parser.add_argument(
        "--memory-sample-interval-ms", type=PositiveFloat, default=100.0
    )
    parser.add_argument("--offload-video-to-cpu", action="store_true")
    parser.add_argument("--offload-state-to-cpu", action="store_true")
    parser.add_argument(
        "--live-metrics",
        action="store_true",
        help=(
            "stream memory samples and per-frame latency as JSONL to stderr and "
            "live_metrics.jsonl"
        ),
    )
    namespace = parser.parse_args(argv)

    return CliArguments(
        mode=cast(BenchmarkMode, namespace.mode),
        video_path=namespace.video,
        model_config=namespace.model_config,
        checkpoint_path=namespace.checkpoint,
        prompt_path=namespace.prompt,
        output_directory=namespace.output_directory,
        device=torch.device(namespace.device),
        dtype_name=cast(DTypeName, namespace.dtype),
        compile_image_encoder=namespace.compile_image_encoder,
        max_frames=namespace.max_frames,
        warmup_frames=namespace.warmup_frames,
        repeats=namespace.repeats,
        chunk_size=namespace.chunk_size,
        max_dimension=namespace.max_dimension,
        memory_sample_interval_seconds=namespace.memory_sample_interval_ms
        / 1000.0,
        offload_video_to_cpu=namespace.offload_video_to_cpu,
        offload_state_to_cpu=namespace.offload_state_to_cpu,
        live_metrics=namespace.live_metrics,
    )


def validate_inputs(arguments: CliArguments, prompt: PromptSpec) -> None:
    for path, label in (
        (arguments.video_path, "video"),
        (arguments.checkpoint_path, "checkpoint"),
        (arguments.prompt_path, "prompt"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} file does not exist: {path}")
    if arguments.device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but CUDA is unavailable")
    if arguments.mode == "demo-chunked" and prompt.frame_idx != 0:
        raise ValueError("demo-chunked mode requires a prompt on frame 0")
    if arguments.max_frames is not None and prompt.frame_idx >= arguments.max_frames:
        raise ValueError("prompt frame is outside --max-frames")


def torch_dtype(name: DTypeName) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def inference_context(
    device: torch.device, dtype: torch.dtype
) -> contextlib.AbstractContextManager[object]:
    if device.type == "cuda" and dtype != torch.float32:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


def synchronize_cuda(device: torch.device) -> None:
    """Finish queued CUDA work at a CPU wall-clock measurement boundary."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_predictor(arguments: CliArguments) -> tuple[SAM2VideoPredictor, float]:
    started_at = time.perf_counter()
    predictor = build_sam2_video_predictor(
        arguments.model_config,
        str(arguments.checkpoint_path),
        device=str(arguments.device),
        hydra_overrides_extra=(
            ["++model.compile_image_encoder=true"]
            if arguments.compile_image_encoder
            else []
        ),
    )
    if not isinstance(predictor, SAM2VideoPredictor):
        raise TypeError("video predictor builder returned an unexpected model type")
    synchronize_cuda(arguments.device)
    return predictor, time.perf_counter() - started_at


def measure_prediction(
    iterator: Iterator[tuple[int, Sequence[int], torch.Tensor]],
    device: torch.device,
) -> Prediction:
    started_at = time.perf_counter()
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        start_event = None
        end_event = None

    frame_idx, raw_object_ids, mask_logits = next(iterator)
    if end_event is not None:
        end_event.record()

    object_ids = tuple(int(object_id) for object_id in raw_object_ids)
    materialize_started_at = time.perf_counter()
    checksums, binary_masks = materialize_mask_checksums(mask_logits, object_ids)
    materialize_seconds = time.perf_counter() - materialize_started_at

    if start_event is not None and end_event is not None:
        # CUDA events complete asynchronously; elapsed_time is valid only afterward.
        end_event.synchronize()
        gpu_inference_ms = start_event.elapsed_time(end_event)
    else:
        gpu_inference_ms = (materialize_started_at - started_at) * 1000.0

    return Prediction(
        frame_idx=frame_idx,
        object_ids=object_ids,
        mask_logits=mask_logits,
        mask_checksums=checksums,
        binary_masks=binary_masks,
        gpu_inference_ms=gpu_inference_ms,
        mask_materialization_ms=materialize_seconds * 1000.0,
        started_at=started_at,
    )


def add_prompt(
    predictor: SAM2VideoPredictor, state: dict[str, object], prompt: PromptSpec
) -> None:
    points = np.asarray(prompt.points, dtype=np.float32)
    labels = np.asarray(prompt.labels, dtype=np.int32)
    predictor.add_new_points_or_box(
        inference_state=state,
        frame_idx=prompt.frame_idx,
        obj_id=prompt.obj_id,
        points=points,
        labels=labels,
    )


def run_core_eager(
    *,
    predictor: SAM2VideoPredictor,
    arguments: CliArguments,
    config: BenchmarkConfig,
    prompt: PromptSpec,
    sampler: MemorySampler,
    live_reporter: LiveMetricsReporter,
) -> ModeResult:
    pipeline_started_at = time.perf_counter()
    init_started_at = time.perf_counter()
    state = predictor.init_state(
        video_path=str(arguments.video_path),
        offload_video_to_cpu=arguments.offload_video_to_cpu,
        offload_state_to_cpu=arguments.offload_state_to_cpu,
    )
    init_seconds = time.perf_counter() - init_started_at
    sampler.capture("post_init")

    prompt_started_at = time.perf_counter()
    add_prompt(predictor, state, prompt)
    prompt_seconds = time.perf_counter() - prompt_started_at
    sampler.capture("post_prompt", prompt.frame_idx)

    traces: list[FrameTrace] = []
    propagation_started_at = time.perf_counter()
    iterator = predictor.propagate_in_video(state)
    try:
        while arguments.max_frames is None or len(traces) < arguments.max_frames:
            sampler.set_position("propagation", len(traces))
            try:
                prediction = measure_prediction(iterator, arguments.device)
            except StopIteration:
                break
            elapsed = time.perf_counter() - pipeline_started_at
            trace = FrameTrace(
                    repeat_index=config.repeat_index,
                    global_frame_idx=prediction.frame_idx,
                    source_frame_idx=prediction.frame_idx,
                    chunk_idx=None,
                    object_ids=prediction.object_ids,
                    mask_checksums=prediction.mask_checksums,
                    gpu_inference_ms=prediction.gpu_inference_ms,
                    mask_materialization_ms=prediction.mask_materialization_ms,
                    output_processing_ms=0.0,
                    frame_wall_ms=(time.perf_counter() - prediction.started_at)
                    * 1000.0,
                    elapsed_seconds=elapsed,
            )
            traces.append(trace)
            live_reporter.publish_frame(trace)
    finally:
        iterator.close()
    propagation_seconds = time.perf_counter() - propagation_started_at
    predictor.reset_state(state)
    sampler.capture("run_complete", traces[-1].global_frame_idx if traces else None)
    return ModeResult(
        traces=tuple(traces),
        video_prepare_seconds=0.0,
        init_state_seconds=init_seconds,
        prompt_seconds=prompt_seconds,
        propagation_seconds=propagation_seconds,
        output_encode_seconds=0.0,
        measured_pipeline_seconds=time.perf_counter() - pipeline_started_at,
    )


def prepare_legacy_chunks(
    source_path: Path,
    output_directory: Path,
    chunk_size: int,
    max_dimension: int,
    max_frames: int | None,
) -> PreparedChunks:
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {source_path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    chunks: list[LegacyChunk] = []
    writer: cv2.VideoWriter | None = None
    current_path: Path | None = None
    current_frame_count = 0
    total_frames = 0
    output_width = 0
    output_height = 0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    try:
        while max_frames is None or total_frames < max_frames:
            success, frame = capture.read()
            if not success:
                break
            height, width = frame.shape[:2]
            if max(height, width) > max_dimension:
                scale = max_dimension / max(height, width)
                frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
            output_height, output_width = frame.shape[:2]

            if writer is None:
                current_path = output_directory / f"chunk_{len(chunks):04d}.mp4"
                writer = cv2.VideoWriter(
                    str(current_path),
                    fourcc,
                    fps,
                    (output_width, output_height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"could not create chunk: {current_path}")
                current_frame_count = 0

            writer.write(frame)
            current_frame_count += 1
            total_frames += 1
            if current_frame_count == chunk_size:
                writer.release()
                writer = None
                if current_path is None:
                    raise RuntimeError("chunk path was not initialized")
                chunks.append(LegacyChunk(current_path, current_frame_count))
    finally:
        capture.release()
        if writer is not None:
            writer.release()
            if current_path is not None and current_frame_count > 0:
                chunks.append(LegacyChunk(current_path, current_frame_count))

    if not chunks:
        raise RuntimeError("video contains no readable frames")
    return PreparedChunks(
        chunks=tuple(chunks),
        fps=fps,
        frame_count=total_frames,
        width=output_width,
        height=output_height,
    )


def overlay_mask(frame_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.shape != frame_rgb.shape[:2]:
        raise ValueError("output mask and video frame dimensions differ")
    rgba_mask = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba_mask[mask.astype(bool)] = (31, 119, 180, 153)
    background = Image.fromarray(frame_rgb).convert("RGBA")
    foreground = Image.fromarray(rgba_mask, mode="RGBA")
    return np.asarray(Image.alpha_composite(background, foreground).convert("RGB"))


def run_demo_chunked(
    *,
    predictor: SAM2VideoPredictor,
    arguments: CliArguments,
    config: BenchmarkConfig,
    prompt: PromptSpec,
    sampler: MemorySampler,
    live_reporter: LiveMetricsReporter,
    run_directory: Path,
) -> ModeResult:
    pipeline_started_at = time.perf_counter()
    prepare_started_at = time.perf_counter()
    chunks_directory = run_directory / "legacy_chunks"
    chunks_directory.mkdir()
    prepared = prepare_legacy_chunks(
        arguments.video_path,
        chunks_directory,
        arguments.chunk_size,
        arguments.max_dimension,
        arguments.max_frames,
    )
    prepare_seconds = time.perf_counter() - prepare_started_at
    sampler.capture("chunks_prepared")

    output_path = run_directory / "output.mp4"
    output_writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        prepared.fps,
        (prepared.width, prepared.height),
    )
    if not output_writer.isOpened():
        raise RuntimeError(f"could not create benchmark output: {output_path}")

    traces: list[FrameTrace] = []
    carried_mask: np.ndarray | None = None
    init_seconds = 0.0
    prompt_seconds = 0.0
    propagation_seconds = 0.0
    output_seconds = 0.0
    global_offset = 0
    try:
        for chunk_idx, chunk in enumerate(prepared.chunks):
            init_started_at = time.perf_counter()
            state = predictor.init_state(
                video_path=str(chunk.path),
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
            )
            init_seconds += time.perf_counter() - init_started_at
            sampler.capture("chunk_initialized", global_offset)

            prompt_started_at = time.perf_counter()
            if chunk_idx == 0:
                add_prompt(predictor, state, prompt)
            elif carried_mask is not None:
                predictor.add_new_mask(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=prompt.obj_id,
                    mask=carried_mask,
                )
            else:
                raise RuntimeError("missing mask to carry into the next chunk")
            prompt_seconds += time.perf_counter() - prompt_started_at

            source_capture = cv2.VideoCapture(str(chunk.path))
            if not source_capture.isOpened():
                raise RuntimeError(f"could not reopen chunk: {chunk.path}")
            iterator = predictor.propagate_in_video(state)
            try:
                while True:
                    sampler.set_position("propagation", global_offset + len(traces))
                    propagation_frame_started_at = time.perf_counter()
                    try:
                        prediction = measure_prediction(iterator, arguments.device)
                    except StopIteration:
                        break
                    propagation_seconds += (
                        time.perf_counter() - propagation_frame_started_at
                    )

                    object_position = prediction.object_ids.index(prompt.obj_id)
                    carried_mask = prediction.binary_masks[object_position].numpy()
                    if carried_mask.ndim == 3 and carried_mask.shape[0] == 1:
                        carried_mask = carried_mask[0]

                    output_started_at = time.perf_counter()
                    source_capture.set(cv2.CAP_PROP_POS_FRAMES, prediction.frame_idx)
                    success, frame_bgr = source_capture.read()
                    if not success:
                        raise RuntimeError(
                            f"could not read output frame {prediction.frame_idx}"
                        )
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    output_rgb = overlay_mask(frame_rgb, carried_mask)
                    output_writer.write(cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR))
                    output_frame_seconds = time.perf_counter() - output_started_at
                    output_seconds += output_frame_seconds

                    global_frame_idx = global_offset + prediction.frame_idx
                    trace = FrameTrace(
                            repeat_index=config.repeat_index,
                            global_frame_idx=global_frame_idx,
                            source_frame_idx=global_frame_idx,
                            chunk_idx=chunk_idx,
                            object_ids=prediction.object_ids,
                            mask_checksums=prediction.mask_checksums,
                            gpu_inference_ms=prediction.gpu_inference_ms,
                            mask_materialization_ms=prediction.mask_materialization_ms,
                            output_processing_ms=output_frame_seconds * 1000.0,
                            frame_wall_ms=(
                                time.perf_counter() - prediction.started_at
                            )
                            * 1000.0,
                            elapsed_seconds=time.perf_counter()
                            - pipeline_started_at,
                    )
                    traces.append(trace)
                    live_reporter.publish_frame(trace)
            finally:
                iterator.close()
                source_capture.release()
            predictor.reset_state(state)
            global_offset += chunk.frame_count
    finally:
        output_writer.release()
    sampler.capture("run_complete", traces[-1].global_frame_idx if traces else None)
    return ModeResult(
        traces=tuple(traces),
        video_prepare_seconds=prepare_seconds,
        init_state_seconds=init_seconds,
        prompt_seconds=prompt_seconds,
        propagation_seconds=propagation_seconds,
        output_encode_seconds=output_seconds,
        measured_pipeline_seconds=time.perf_counter() - pipeline_started_at,
    )


def write_run_artifacts(
    *,
    run_directory: Path,
    config: BenchmarkConfig,
    metadata: dict[str, object],
    traces: Sequence[FrameTrace],
    samples: Sequence[MemorySample],
    summary: RunSummary,
) -> None:
    validate_run_artifacts(summary, traces, samples)
    write_json(run_directory / "config.json", config)
    write_json(run_directory / "metadata.json", metadata)
    write_jsonl(run_directory / "frames.jsonl", traces)
    write_jsonl(run_directory / "memory.jsonl", samples)
    write_json(run_directory / "summary.json", summary)


def execute_repeat(
    *,
    predictor: SAM2VideoPredictor,
    arguments: CliArguments,
    prompt: PromptSpec,
    repeat_index: int,
    model_load_seconds: float,
    session_directory: Path,
    shared_metadata: dict[str, object],
) -> RunSummary:
    run_directory = session_directory / f"run_{repeat_index:03d}"
    run_directory.mkdir()
    config = BenchmarkConfig(
        mode=arguments.mode,
        video_path=str(arguments.video_path.resolve()),
        model_config=arguments.model_config,
        checkpoint_path=str(arguments.checkpoint_path.resolve()),
        prompt_path=str(arguments.prompt_path.resolve()),
        device=str(arguments.device),
        dtype=arguments.dtype_name,
        compile_image_encoder=arguments.compile_image_encoder,
        max_frames=arguments.max_frames,
        warmup_frames=arguments.warmup_frames,
        repeat_index=repeat_index,
        chunk_size=arguments.chunk_size,
        max_dimension=arguments.max_dimension,
        memory_sample_interval_seconds=arguments.memory_sample_interval_seconds,
        offload_video_to_cpu=arguments.offload_video_to_cpu,
        offload_state_to_cpu=arguments.offload_state_to_cpu,
    )
    live_reporter = LiveMetricsReporter(
        run_directory / "live_metrics.jsonl" if arguments.live_metrics else None
    )
    sampler = MemorySampler(
        arguments.device,
        arguments.memory_sample_interval_seconds,
        sample_sink=live_reporter.publish_memory,
    )
    live_reporter.start()
    sampler.start()
    sampler.capture("model_loaded")
    traces: tuple[FrameTrace, ...] = ()
    error_message: str | None = None
    status = "completed"
    mode_result: ModeResult | None = None
    try:
        with inference_context(arguments.device, torch_dtype(arguments.dtype_name)):
            if arguments.mode == "core-eager":
                mode_result = run_core_eager(
                    predictor=predictor,
                    arguments=arguments,
                    config=config,
                    prompt=prompt,
                    sampler=sampler,
                    live_reporter=live_reporter,
                )
            else:
                mode_result = run_demo_chunked(
                    predictor=predictor,
                    arguments=arguments,
                    config=config,
                    prompt=prompt,
                    sampler=sampler,
                    live_reporter=live_reporter,
                    run_directory=run_directory,
                )
        traces = mode_result.traces
    except BaseException as error:
        status = "failed"
        error_message = f"{type(error).__name__}: {error}"
    try:
        samples = sampler.stop()
    except BaseException as sampler_error:
        samples = ()
        status = "failed"
        sampler_message = f"{type(sampler_error).__name__}: {sampler_error}"
        error_message = (
            sampler_message
            if error_message is None
            else f"{error_message}; sampler: {sampler_message}"
        )

    if mode_result is None:
        stages = StageDurations(
            model_load_seconds=model_load_seconds,
            video_prepare_seconds=0.0,
            init_state_seconds=0.0,
            prompt_seconds=0.0,
            propagation_seconds=0.0,
            output_encode_seconds=0.0,
            measured_pipeline_seconds=0.0,
        )
    else:
        stages = StageDurations(
            model_load_seconds=model_load_seconds,
            video_prepare_seconds=mode_result.video_prepare_seconds,
            init_state_seconds=mode_result.init_state_seconds,
            prompt_seconds=mode_result.prompt_seconds,
            propagation_seconds=mode_result.propagation_seconds,
            output_encode_seconds=mode_result.output_encode_seconds,
            measured_pipeline_seconds=mode_result.measured_pipeline_seconds,
        )
    summary = build_run_summary(
        status=status,
        error=error_message,
        config=config,
        traces=traces,
        samples=samples,
        stages=stages,
    )
    live_reporter.publish_summary(summary)
    live_reporter.close()
    metadata = dict(shared_metadata)
    metadata["nvml_driver"] = sampler.driver_version
    write_run_artifacts(
        run_directory=run_directory,
        config=config,
        metadata=metadata,
        traces=traces,
        samples=samples,
        summary=summary,
    )
    if status == "failed":
        raise RuntimeError(
            f"benchmark run {repeat_index} failed; see {run_directory / 'summary.json'}"
        )
    return summary


def main() -> None:
    arguments = parse_arguments()
    prompt = PromptSpec.from_json_file(arguments.prompt_path)
    validate_inputs(arguments, prompt)
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    session_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    session_directory = arguments.output_directory / (
        f"{session_name}_{arguments.mode.replace('-', '_')}"
    )
    session_directory.mkdir()

    predictor, model_load_seconds = build_predictor(arguments)
    shared_metadata = runtime_metadata(arguments.device)
    shared_metadata.update(
        {
            "repository_revision": repository_revision(REPOSITORY_ROOT),
            "checkpoint_sha256": sha256_file(arguments.checkpoint_path),
            "prompt_sha256": sha256_file(arguments.prompt_path),
        }
    )

    summaries: list[RunSummary] = []
    for repeat_index in range(arguments.repeats):
        summaries.append(
            execute_repeat(
                predictor=predictor,
                arguments=arguments,
                prompt=prompt,
                repeat_index=repeat_index,
                model_load_seconds=model_load_seconds,
                session_directory=session_directory,
                shared_metadata=shared_metadata,
            )
        )
    write_json(session_directory / "aggregate.json", aggregate_summaries(summaries))
    print(session_directory)


if __name__ == "__main__":
    main()
