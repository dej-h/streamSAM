from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence, cast

import cv2

from benchmarks.benchmark_video_streaming import PositiveInteger


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BenchmarkMode = Literal["core-eager", "demo-chunked"]


@dataclass(frozen=True)
class VideoCase:
    name: str
    source_path: Path


VIDEO_CASES = (
    VideoCase("dog", Path("examples/01_dog.mp4")),
    VideoCase("snowboarder", Path("examples/07_snowboarder.mp4")),
    VideoCase("robotarm", Path("examples/16_robotarm.mp4")),
    VideoCase("doughkneading", Path("examples/20_doughkneading.mp4")),
    VideoCase("clownfish", Path("examples/24_clownfish.mp4")),
)


@dataclass(frozen=True)
class MatrixArguments:
    checkpoint_path: Path
    model_config: str
    output_directory: Path
    frame_count: int
    max_dimension: int
    memory_sample_interval_ms: int


@dataclass(frozen=True)
class PreparedVideo:
    name: str
    source_path: Path
    video_path: Path
    prompt_path: Path
    frame_count: int
    width: int
    height: int
    fps: float


@dataclass(frozen=True)
class Distribution:
    count: int
    mean: float
    median: float
    p95: float
    minimum: float
    maximum: float
    standard_deviation: float | None


@dataclass(frozen=True)
class VideoMeasurement:
    video_index: int
    video_name: str
    mode: BenchmarkMode
    compile_image_encoder: bool
    cold_compilation_cache: bool
    frames_processed: int
    core_gpu_fps: float
    propagation_fps: float
    end_to_end_fps: float
    steady_state_fps: float
    gpu_inference_mean_ms: float
    gpu_inference_p95_ms: float
    frame_wall_mean_ms: float
    frame_wall_p95_ms: float
    peak_torch_allocated_bytes: float
    peak_torch_reserved_bytes: float
    peak_nvml_device_used_bytes: float
    peak_process_rss_bytes: float
    model_load_seconds: float
    video_prepare_seconds: float
    init_state_seconds: float
    prompt_seconds: float
    propagation_seconds: float
    measured_pipeline_seconds: float
    process_wall_seconds: float
    summary_path: str
    frames_path: str
    live_metrics_path: str


@dataclass(frozen=True)
class MatrixAggregate:
    mode: BenchmarkMode
    compile_image_encoder: bool
    videos: int
    frames: int
    core_gpu_fps: Distribution
    propagation_fps: Distribution
    end_to_end_fps: Distribution
    steady_state_fps: Distribution
    gpu_inference_latency_ms: Distribution
    frame_wall_latency_ms: Distribution
    mask_materialization_latency_ms: Distribution
    peak_torch_allocated_bytes: Distribution
    peak_torch_reserved_bytes: Distribution
    peak_nvml_device_used_bytes: Distribution
    peak_process_rss_bytes: Distribution
    model_load_seconds: Distribution
    video_prepare_seconds: Distribution
    init_state_seconds: Distribution
    prompt_seconds: Distribution
    propagation_seconds: Distribution
    measured_pipeline_seconds: Distribution
    process_wall_seconds: Distribution


@dataclass(frozen=True)
class CompilationDelta:
    mode: BenchmarkMode
    scope: Literal["all-five", "warm-cache-four"]
    core_gpu_fps_percent: float
    propagation_fps_percent: float
    end_to_end_fps_percent: float
    gpu_inference_mean_ms_percent: float
    frame_wall_mean_ms_percent: float
    frame_wall_p95_ms_percent: float
    peak_torch_allocated_bytes_percent: float
    peak_nvml_device_used_bytes_percent: float
    init_state_seconds_percent: float
    process_wall_seconds_percent: float


def parse_arguments(argv: Sequence[str] | None = None) -> MatrixArguments:
    parser = argparse.ArgumentParser(
        description="Run the eager/chunked and compiled/uncompiled video matrix."
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/edgetam.pt")
    )
    parser.add_argument("--model-config", default="edgetam.yaml")
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("benchmark_artifacts/video_matrix"),
    )
    parser.add_argument("--frame-count", type=PositiveInteger, default=200)
    parser.add_argument("--max-dimension", type=PositiveInteger, default=960)
    parser.add_argument(
        "--memory-sample-interval-ms", type=PositiveInteger, default=100
    )
    namespace = parser.parse_args(argv)
    return MatrixArguments(
        checkpoint_path=namespace.checkpoint,
        model_config=namespace.model_config,
        output_directory=namespace.output_directory,
        frame_count=namespace.frame_count,
        max_dimension=namespace.max_dimension,
        memory_sample_interval_ms=namespace.memory_sample_interval_ms,
    )


def prepare_video(
    case: VideoCase,
    input_directory: Path,
    frame_count: int,
    max_dimension: int,
) -> PreparedVideo:
    source_path = (REPOSITORY_ROOT / case.source_path).resolve()
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open source video: {source_path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise RuntimeError(f"source video has invalid FPS: {source_path}")
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, max_dimension / max(source_width, source_height))
    width = max(2, round(source_width * scale / 2) * 2)
    height = max(2, round(source_height * scale / 2) * 2)
    video_path = input_directory / f"{case.name}.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"could not create normalized video: {video_path}")

    written = 0
    try:
        while written < frame_count:
            success, frame = capture.read()
            if not success:
                break
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
            written += 1
    finally:
        capture.release()
        writer.release()
    if written != frame_count:
        raise RuntimeError(
            f"{source_path} provided {written} frames; {frame_count} required"
        )

    prompt_path = input_directory / f"{case.name}_prompt.json"
    prompt_path.write_text(
        json.dumps(
            {
                "frame_idx": 0,
                "obj_id": 1,
                "points": [[width / 2, height / 2]],
                "labels": [1],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return PreparedVideo(
        name=case.name,
        source_path=source_path,
        video_path=video_path.resolve(),
        prompt_path=prompt_path.resolve(),
        frame_count=written,
        width=width,
        height=height,
        fps=fps,
    )


def run_cell(
    *,
    arguments: MatrixArguments,
    session_directory: Path,
    prepared: PreparedVideo,
    video_index: int,
    mode: BenchmarkMode,
    compile_image_encoder: bool,
) -> VideoMeasurement:
    condition_name = f"{mode}_{'compiled' if compile_image_encoder else 'eager'}"
    cell_directory = session_directory / "runs" / condition_name / prepared.name
    cell_directory.mkdir(parents=True)
    stdout_path = cell_directory / "process.stdout.log"
    stderr_path = cell_directory / "process.stderr.log"
    command = [
        sys.executable,
        "-m",
        "benchmarks.benchmark_video_streaming",
        "--mode",
        mode,
        "--video",
        str(prepared.video_path),
        "--model-config",
        arguments.model_config,
        "--checkpoint",
        str(arguments.checkpoint_path.resolve()),
        "--prompt",
        str(prepared.prompt_path),
        "--output-directory",
        str(cell_directory),
        "--device",
        "cuda",
        "--dtype",
        "bfloat16",
        "--max-frames",
        str(arguments.frame_count),
        "--warmup-frames",
        "0",
        "--repeats",
        "1",
        "--chunk-size",
        "96",
        "--max-dimension",
        str(arguments.max_dimension),
        "--memory-sample-interval-ms",
        str(arguments.memory_sample_interval_ms),
        "--live-metrics",
    ]
    if compile_image_encoder:
        command.append("--compile-image-encoder")

    environment = os.environ.copy()
    cold_compilation_cache = compile_image_encoder and video_index == 0
    if compile_image_encoder:
        cache_directory = session_directory / "torchinductor_cache" / mode
        cache_directory.mkdir(parents=True, exist_ok=True)
        environment["TORCHINDUCTOR_CACHE_DIR"] = str(cache_directory.resolve())

    started_at = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=False,
        )
    process_wall_seconds = time.perf_counter() - started_at
    if completed.returncode != 0:
        error_tail = "\n".join(
            stderr_path.read_text(encoding="utf-8").splitlines()[-30:]
        )
        raise RuntimeError(
            f"benchmark failed for {condition_name}/{prepared.name}:\n{error_tail}"
        )

    stdout_lines = stdout_path.read_text(encoding="utf-8").splitlines()
    if not stdout_lines:
        raise RuntimeError(f"benchmark returned no session path: {cell_directory}")
    run_session = Path(stdout_lines[-1])
    if not run_session.is_absolute():
        run_session = REPOSITORY_ROOT / run_session
    run_directory = run_session / "run_000"
    summary_path = run_directory / "summary.json"
    summary = _read_json_object(summary_path)
    stages = _required_mapping(summary, "stages")
    gpu_latency = _required_mapping(summary, "gpu_inference_latency_ms")
    frame_latency = _required_mapping(summary, "frame_wall_latency_ms")
    return VideoMeasurement(
        video_index=video_index,
        video_name=prepared.name,
        mode=mode,
        compile_image_encoder=compile_image_encoder,
        cold_compilation_cache=cold_compilation_cache,
        frames_processed=_required_int(summary, "frames_processed"),
        core_gpu_fps=_required_float(summary, "core_gpu_fps"),
        propagation_fps=_required_float(summary, "propagation_fps"),
        end_to_end_fps=_required_float(summary, "end_to_end_fps"),
        steady_state_fps=_required_float(summary, "steady_state_fps"),
        gpu_inference_mean_ms=_required_float(gpu_latency, "mean"),
        gpu_inference_p95_ms=_required_float(gpu_latency, "p95"),
        frame_wall_mean_ms=_required_float(frame_latency, "mean"),
        frame_wall_p95_ms=_required_float(frame_latency, "p95"),
        peak_torch_allocated_bytes=_required_float(
            summary, "peak_torch_allocated_bytes"
        ),
        peak_torch_reserved_bytes=_required_float(
            summary, "peak_torch_reserved_bytes"
        ),
        peak_nvml_device_used_bytes=_required_float(
            summary, "peak_nvml_device_used_bytes"
        ),
        peak_process_rss_bytes=_required_float(summary, "peak_process_rss_bytes"),
        model_load_seconds=_required_float(stages, "model_load_seconds"),
        video_prepare_seconds=_required_float(stages, "video_prepare_seconds"),
        init_state_seconds=_required_float(stages, "init_state_seconds"),
        prompt_seconds=_required_float(stages, "prompt_seconds"),
        propagation_seconds=_required_float(stages, "propagation_seconds"),
        measured_pipeline_seconds=_required_float(
            stages, "measured_pipeline_seconds"
        ),
        process_wall_seconds=process_wall_seconds,
        summary_path=str(summary_path),
        frames_path=str(run_directory / "frames.jsonl"),
        live_metrics_path=str(run_directory / "live_metrics.jsonl"),
    )


def distribution(values: Iterable[float]) -> Distribution:
    present = list(values)
    if not present:
        raise ValueError("cannot aggregate an empty metric")
    return Distribution(
        count=len(present),
        mean=statistics.fmean(present),
        median=statistics.median(present),
        p95=(
            statistics.quantiles(present, n=100, method="inclusive")[94]
            if len(present) > 1
            else present[0]
        ),
        minimum=min(present),
        maximum=max(present),
        standard_deviation=statistics.stdev(present) if len(present) > 1 else None,
    )


def aggregate_measurements(
    measurements: Sequence[VideoMeasurement],
) -> MatrixAggregate:
    if not measurements:
        raise ValueError("cannot aggregate an empty measurement group")
    first = measurements[0]
    frame_records = [
        frame
        for measurement in measurements
        for frame in _read_jsonl_objects(Path(measurement.frames_path))
    ]
    return MatrixAggregate(
        mode=first.mode,
        compile_image_encoder=first.compile_image_encoder,
        videos=len(measurements),
        frames=sum(measurement.frames_processed for measurement in measurements),
        core_gpu_fps=distribution(item.core_gpu_fps for item in measurements),
        propagation_fps=distribution(item.propagation_fps for item in measurements),
        end_to_end_fps=distribution(item.end_to_end_fps for item in measurements),
        steady_state_fps=distribution(item.steady_state_fps for item in measurements),
        gpu_inference_latency_ms=distribution(
            _required_float(frame, "gpu_inference_ms") for frame in frame_records
        ),
        frame_wall_latency_ms=distribution(
            _required_float(frame, "frame_wall_ms") for frame in frame_records
        ),
        mask_materialization_latency_ms=distribution(
            _required_float(frame, "mask_materialization_ms")
            for frame in frame_records
        ),
        peak_torch_allocated_bytes=distribution(
            item.peak_torch_allocated_bytes for item in measurements
        ),
        peak_torch_reserved_bytes=distribution(
            item.peak_torch_reserved_bytes for item in measurements
        ),
        peak_nvml_device_used_bytes=distribution(
            item.peak_nvml_device_used_bytes for item in measurements
        ),
        peak_process_rss_bytes=distribution(
            item.peak_process_rss_bytes for item in measurements
        ),
        model_load_seconds=distribution(
            item.model_load_seconds for item in measurements
        ),
        video_prepare_seconds=distribution(
            item.video_prepare_seconds for item in measurements
        ),
        init_state_seconds=distribution(
            item.init_state_seconds for item in measurements
        ),
        prompt_seconds=distribution(item.prompt_seconds for item in measurements),
        propagation_seconds=distribution(
            item.propagation_seconds for item in measurements
        ),
        measured_pipeline_seconds=distribution(
            item.measured_pipeline_seconds for item in measurements
        ),
        process_wall_seconds=distribution(
            item.process_wall_seconds for item in measurements
        ),
    )


def compilation_delta(
    baseline: MatrixAggregate,
    compiled: MatrixAggregate,
    scope: Literal["all-five", "warm-cache-four"],
) -> CompilationDelta:
    return CompilationDelta(
        mode=baseline.mode,
        scope=scope,
        core_gpu_fps_percent=_percent_change(
            baseline.core_gpu_fps.mean, compiled.core_gpu_fps.mean
        ),
        propagation_fps_percent=_percent_change(
            baseline.propagation_fps.mean, compiled.propagation_fps.mean
        ),
        end_to_end_fps_percent=_percent_change(
            baseline.end_to_end_fps.mean, compiled.end_to_end_fps.mean
        ),
        gpu_inference_mean_ms_percent=_percent_change(
            baseline.gpu_inference_latency_ms.mean,
            compiled.gpu_inference_latency_ms.mean,
        ),
        frame_wall_mean_ms_percent=_percent_change(
            baseline.frame_wall_latency_ms.mean, compiled.frame_wall_latency_ms.mean
        ),
        frame_wall_p95_ms_percent=_percent_change(
            baseline.frame_wall_latency_ms.p95, compiled.frame_wall_latency_ms.p95
        ),
        peak_torch_allocated_bytes_percent=_percent_change(
            baseline.peak_torch_allocated_bytes.mean,
            compiled.peak_torch_allocated_bytes.mean,
        ),
        peak_nvml_device_used_bytes_percent=_percent_change(
            baseline.peak_nvml_device_used_bytes.mean,
            compiled.peak_nvml_device_used_bytes.mean,
        ),
        init_state_seconds_percent=_percent_change(
            baseline.init_state_seconds.mean, compiled.init_state_seconds.mean
        ),
        process_wall_seconds_percent=_percent_change(
            baseline.process_wall_seconds.mean, compiled.process_wall_seconds.mean
        ),
    )


def write_markdown_report(
    path: Path,
    aggregates: Sequence[MatrixAggregate],
    deltas: Sequence[CompilationDelta],
) -> None:
    lines = [
        "# EdgeTAM video benchmark matrix",
        "",
        "Five normalized 200-frame, 960x540 videos per condition. Compilation is ",
        "the repository's image-encoder `torch.compile` path, not full-predictor compilation.",
        "",
        "## Aggregate results",
        "",
        "| Method | Compiled | GPU FPS | Propagation FPS | E2E FPS | GPU mean ms | Frame p95 ms | Torch peak GiB | NVML peak GiB | RSS peak GiB | Init s | Process s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in aggregates:
        lines.append(
            "| "
            f"{item.mode} | {item.compile_image_encoder} | "
            f"{item.core_gpu_fps.mean:.2f} | {item.propagation_fps.mean:.2f} | "
            f"{item.end_to_end_fps.mean:.2f} | "
            f"{item.gpu_inference_latency_ms.mean:.2f} | "
            f"{item.frame_wall_latency_ms.p95:.2f} | "
            f"{_gib(item.peak_torch_allocated_bytes.mean):.2f} | "
            f"{_gib(item.peak_nvml_device_used_bytes.mean):.2f} | "
            f"{_gib(item.peak_process_rss_bytes.mean):.2f} | "
            f"{item.init_state_seconds.mean:.2f} | "
            f"{item.process_wall_seconds.mean:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Compilation deltas",
            "",
            "Positive percentages mean the compiled value is higher. Lower is better for latency, memory, initialization, and process duration.",
            "",
            "| Method | Scope | GPU FPS | Propagation FPS | E2E FPS | GPU mean ms | Frame mean ms | Frame p95 ms | Torch peak | NVML peak | Init | Process |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in deltas:
        lines.append(
            "| "
            f"{item.mode} | {item.scope} | {item.core_gpu_fps_percent:+.1f}% | "
            f"{item.propagation_fps_percent:+.1f}% | "
            f"{item.end_to_end_fps_percent:+.1f}% | "
            f"{item.gpu_inference_mean_ms_percent:+.1f}% | "
            f"{item.frame_wall_mean_ms_percent:+.1f}% | "
            f"{item.frame_wall_p95_ms_percent:+.1f}% | "
            f"{item.peak_torch_allocated_bytes_percent:+.1f}% | "
            f"{item.peak_nvml_device_used_bytes_percent:+.1f}% | "
            f"{item.init_state_seconds_percent:+.1f}% | "
            f"{item.process_wall_seconds_percent:+.1f}% |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    arguments = parse_arguments()
    if not arguments.checkpoint_path.is_file():
        raise FileNotFoundError(arguments.checkpoint_path)
    session_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    session_directory = arguments.output_directory / session_name
    input_directory = session_directory / "inputs"
    input_directory.mkdir(parents=True)

    prepared_videos = tuple(
        prepare_video(
            case,
            input_directory,
            arguments.frame_count,
            arguments.max_dimension,
        )
        for case in VIDEO_CASES
    )
    _write_json(
        session_directory / "manifest.json",
        {
            "arguments": asdict(arguments),
            "videos": [asdict(video) for video in prepared_videos],
        },
    )

    measurements: list[VideoMeasurement] = []
    total_runs = len(prepared_videos) * 4
    run_index = 0
    for mode in cast(tuple[BenchmarkMode, ...], ("core-eager", "demo-chunked")):
        for compile_image_encoder in (False, True):
            for video_index, prepared in enumerate(prepared_videos):
                run_index += 1
                print(
                    f"[{run_index}/{total_runs}] {mode} "
                    f"compiled={compile_image_encoder} video={prepared.name}",
                    flush=True,
                )
                measurement = run_cell(
                    arguments=arguments,
                    session_directory=session_directory,
                    prepared=prepared,
                    video_index=video_index,
                    mode=mode,
                    compile_image_encoder=compile_image_encoder,
                )
                measurements.append(measurement)
                _write_json(
                    session_directory / "measurements.json",
                    [asdict(item) for item in measurements],
                )
                print(
                    f"  {measurement.core_gpu_fps:.2f} GPU FPS, "
                    f"{measurement.frame_wall_mean_ms:.2f} ms frame mean, "
                    f"{measurement.process_wall_seconds:.2f} s process",
                    flush=True,
                )

    aggregates: list[MatrixAggregate] = []
    warm_aggregates: list[MatrixAggregate] = []
    for mode in cast(tuple[BenchmarkMode, ...], ("core-eager", "demo-chunked")):
        for compiled in (False, True):
            group = [
                item
                for item in measurements
                if item.mode == mode and item.compile_image_encoder == compiled
            ]
            aggregates.append(aggregate_measurements(group))
            warm_aggregates.append(
                aggregate_measurements([item for item in group if item.video_index > 0])
            )

    deltas: list[CompilationDelta] = []
    for mode in cast(tuple[BenchmarkMode, ...], ("core-eager", "demo-chunked")):
        all_baseline = _find_aggregate(aggregates, mode, False)
        all_compiled = _find_aggregate(aggregates, mode, True)
        warm_baseline = _find_aggregate(warm_aggregates, mode, False)
        warm_compiled = _find_aggregate(warm_aggregates, mode, True)
        deltas.extend(
            [
                compilation_delta(all_baseline, all_compiled, "all-five"),
                compilation_delta(
                    warm_baseline, warm_compiled, "warm-cache-four"
                ),
            ]
        )

    _write_json(
        session_directory / "matrix_summary.json",
        {
            "aggregates": [asdict(item) for item in aggregates],
            "warm_cache_aggregates": [asdict(item) for item in warm_aggregates],
            "compilation_deltas": [asdict(item) for item in deltas],
        },
    )
    write_markdown_report(
        session_directory / "matrix_report.md", aggregates, deltas
    )
    print(session_directory)


def _find_aggregate(
    aggregates: Sequence[MatrixAggregate],
    mode: BenchmarkMode,
    compiled: bool,
) -> MatrixAggregate:
    return next(
        item
        for item in aggregates
        if item.mode == mode and item.compile_image_encoder == compiled
    )


def _percent_change(baseline: float, candidate: float) -> float:
    if baseline == 0:
        raise ValueError("cannot calculate percentage change from zero")
    return (candidate / baseline - 1.0) * 100.0


def _gib(value: float) -> float:
    return value / (1024**3)


def _read_json_object(path: Path) -> dict[str, object]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in value
    ):
        raise ValueError(f"expected JSON object in {path}")
    return cast(dict[str, object], value)


def _read_jsonl_objects(path: Path) -> list[dict[str, object]]:
    return [
        _json_object_from_line(line, path)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _json_object_from_line(line: str, path: Path) -> dict[str, object]:
    value: object = json.loads(line)
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in value
    ):
        raise ValueError(f"expected JSON object records in {path}")
    return cast(dict[str, object], value)


def _required_mapping(
    source: Mapping[str, object], key: str
) -> dict[str, object]:
    value = source.get(key)
    if not isinstance(value, dict) or not all(
        isinstance(item_key, str) for item_key in value
    ):
        raise ValueError(f"{key} must be an object")
    return cast(dict[str, object], value)


def _required_float(source: Mapping[str, object], key: str) -> float:
    value = source.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key} must be a number")
    return float(value)


def _required_int(source: Mapping[str, object], key: str) -> int:
    value = source.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
