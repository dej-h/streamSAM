from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Mapping, Sequence, cast

from benchmarks.video.render_benchmark_comparison import RenderOptions, render_comparison


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BenchmarkMode = Literal["core-eager", "demo-chunked", "demo-streaming"]
DTypeName = Literal["float32", "float16", "bfloat16"]


@dataclass(frozen=True)
class DemoArguments:
    video_path: Path
    prompt_path: Path
    model_config: str
    checkpoint_path: Path
    output_directory: Path
    device: str
    dtype: DTypeName
    max_frames: int
    warmup_frames: int
    repeats: int
    max_dimension: int
    memory_sample_interval_ms: float
    compile_image_encoder: bool
    compile_video_pipeline: bool
    playback_speed: float
    output_fps: float
    width: int
    height: int
    intro_seconds: float
    hold_seconds: float
    rolling_window_frames: int
    original_rss_safety_limit_gib: float
    reuse_chunked_run: Path | None
    reuse_streaming_run: Path | None


@dataclass(frozen=True)
class BenchmarkProcessResult:
    mode: BenchmarkMode
    command: tuple[str, ...]
    stdout_path: Path | None
    stderr_path: Path | None
    session_directory: Path
    selected_run_directory: Path
    exit_code: int
    peak_observed_rss_bytes: int
    safety_limit_triggered: bool
    reused: bool


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_arguments(argv: Sequence[str] | None = None) -> DemoArguments:
    parser = argparse.ArgumentParser(
        description=(
            "Run original eager, chunked and streaming strategies sequentially, "
            "then render their recorded telemetry as a synchronized video."
        )
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--model-config", default="edgetam.yaml")
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/edgetam.pt")
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("benchmark_artifacts/demo_comparison"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument("--max-frames", type=_positive_int, default=1000)
    parser.add_argument("--warmup-frames", type=_non_negative_int, default=20)
    parser.add_argument("--repeats", type=_positive_int, default=1)
    parser.add_argument("--max-dimension", type=_positive_int, default=960)
    parser.add_argument(
        "--memory-sample-interval-ms", type=_positive_float, default=100.0
    )
    compilation = parser.add_mutually_exclusive_group()
    compilation.add_argument("--compile-image-encoder", action="store_true")
    compilation.add_argument("--compile-video-pipeline", action="store_true")
    parser.add_argument("--playback-speed", type=_positive_float, default=2.0)
    parser.add_argument("--output-fps", type=_positive_float, default=30.0)
    parser.add_argument("--width", type=_positive_int, default=1920)
    parser.add_argument("--height", type=_positive_int, default=1080)
    parser.add_argument("--intro-seconds", type=float, default=2.0)
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    parser.add_argument("--rolling-window-frames", type=_positive_int, default=30)
    parser.add_argument(
        "--original-rss-safety-limit-gib",
        type=_positive_float,
        default=5.0,
        help=(
            "interrupt the original eager process at this RSS before it can "
            "destabilize the host; the stop is recorded in the artifacts"
        ),
    )
    parser.add_argument(
        "--reuse-chunked-run",
        type=Path,
        help="reuse a completed demo-chunked run directory",
    )
    parser.add_argument(
        "--reuse-streaming-run",
        type=Path,
        help="reuse a completed demo-streaming run directory",
    )
    namespace = parser.parse_args(argv)
    if namespace.intro_seconds < 0 or namespace.hold_seconds < 0:
        parser.error("intro and hold durations cannot be negative")

    reuse_chunked_run = (
        None
        if namespace.reuse_chunked_run is None
        else namespace.reuse_chunked_run.resolve()
    )
    reuse_streaming_run = (
        None
        if namespace.reuse_streaming_run is None
        else namespace.reuse_streaming_run.resolve()
    )
    if (reuse_chunked_run is None) != (reuse_streaming_run is None):
        parser.error("provide both reuse run directories or neither")

    return DemoArguments(
        video_path=namespace.video.resolve(),
        prompt_path=namespace.prompt.resolve(),
        model_config=str(namespace.model_config),
        checkpoint_path=namespace.checkpoint.resolve(),
        output_directory=namespace.output_directory.resolve(),
        device=str(namespace.device),
        dtype=cast(DTypeName, namespace.dtype),
        max_frames=namespace.max_frames,
        warmup_frames=namespace.warmup_frames,
        repeats=namespace.repeats,
        max_dimension=namespace.max_dimension,
        memory_sample_interval_ms=namespace.memory_sample_interval_ms,
        compile_image_encoder=namespace.compile_image_encoder,
        compile_video_pipeline=namespace.compile_video_pipeline,
        playback_speed=namespace.playback_speed,
        output_fps=namespace.output_fps,
        width=namespace.width,
        height=namespace.height,
        intro_seconds=namespace.intro_seconds,
        hold_seconds=namespace.hold_seconds,
        rolling_window_frames=namespace.rolling_window_frames,
        original_rss_safety_limit_gib=namespace.original_rss_safety_limit_gib,
        reuse_chunked_run=reuse_chunked_run,
        reuse_streaming_run=reuse_streaming_run,
    )


def _validate_inputs(arguments: DemoArguments) -> None:
    for path, label in (
        (arguments.video_path, "video"),
        (arguments.prompt_path, "prompt"),
        (arguments.checkpoint_path, "checkpoint"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    for path, label in (
        (arguments.reuse_chunked_run, "reused chunked run"),
        (arguments.reuse_streaming_run, "reused streaming run"),
    ):
        if path is not None and not (path / "summary.json").is_file():
            raise FileNotFoundError(f"{label} is incomplete: {path}")


def _benchmark_command(
    mode: BenchmarkMode,
    arguments: DemoArguments,
    output_directory: Path,
) -> tuple[str, ...]:
    command = [
        sys.executable,
        "-m",
        "benchmarks.benchmark_video_streaming",
        "--mode",
        mode,
        "--video",
        str(arguments.video_path),
        "--model-config",
        arguments.model_config,
        "--checkpoint",
        str(arguments.checkpoint_path),
        "--prompt",
        str(arguments.prompt_path),
        "--output-directory",
        str(output_directory),
        "--device",
        arguments.device,
        "--dtype",
        arguments.dtype,
        "--max-frames",
        str(arguments.max_frames),
        "--warmup-frames",
        str(arguments.warmup_frames),
        "--repeats",
        str(arguments.repeats),
        "--max-dimension",
        str(arguments.max_dimension),
        "--memory-sample-interval-ms",
        str(arguments.memory_sample_interval_ms),
        "--live-metrics",
    ]
    if arguments.compile_image_encoder:
        command.append("--compile-image-encoder")
    if arguments.compile_video_pipeline:
        command.append("--compile-video-pipeline")
    return tuple(command)


def _process_rss_bytes(process_id: int) -> int | None:
    statm_path = Path("/proc") / str(process_id) / "statm"
    try:
        resident_pages = int(statm_path.read_text(encoding="ascii").split()[1])
    except (FileNotFoundError, IndexError, ValueError):
        return None
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


def _run_benchmark(
    mode: BenchmarkMode,
    arguments: DemoArguments,
    session_directory: Path,
    *,
    allow_failure: bool = False,
    rss_safety_limit_bytes: int | None = None,
) -> BenchmarkProcessResult:
    label = {
        "core-eager": "original",
        "demo-chunked": "chunked",
        "demo-streaming": "streaming",
    }[mode]
    output_directory = session_directory / label
    output_directory.mkdir()
    stdout_path = session_directory / f"{label}.stdout.log"
    stderr_path = session_directory / f"{label}.stderr.log"
    command = _benchmark_command(mode, arguments, output_directory)

    environment = os.environ.copy()
    if arguments.compile_image_encoder or arguments.compile_video_pipeline:
        compile_cache = session_directory / "torchinductor_cache" / label
        compile_cache.mkdir(parents=True)
        environment["TORCHINDUCTOR_CACHE_DIR"] = str(compile_cache)

    print(f"Running {label} benchmark ({mode})...", flush=True)
    existing_sessions = set(output_directory.iterdir())
    peak_observed_rss_bytes = 0
    safety_limit_triggered = False
    with (
        stdout_path.open("w", encoding="utf-8") as stdout,
        stderr_path.open("w", encoding="utf-8") as stderr,
    ):
        process = subprocess.Popen(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
        while process.poll() is None:
            rss_bytes = _process_rss_bytes(process.pid)
            if rss_bytes is not None:
                peak_observed_rss_bytes = max(peak_observed_rss_bytes, rss_bytes)
            if (
                rss_safety_limit_bytes is not None
                and rss_bytes is not None
                and rss_bytes >= rss_safety_limit_bytes
            ):
                safety_limit_triggered = True
                process.send_signal(signal.SIGINT)
                break
            time.sleep(0.05)
        try:
            exit_code = process.wait(timeout=30.0)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                exit_code = process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                process.kill()
                exit_code = process.wait(timeout=10.0)

    if exit_code != 0 and not allow_failure:
        error_tail = "\n".join(
            stderr_path.read_text(encoding="utf-8").splitlines()[-30:]
        )
        raise RuntimeError(
            f"{label} benchmark failed with exit code {exit_code}:\n" f"{error_tail}"
        )

    stdout_lines = stdout_path.read_text(encoding="utf-8").splitlines()
    benchmark_session: Path | None = None
    if stdout_lines:
        candidate = Path(stdout_lines[-1])
        if not candidate.is_absolute():
            candidate = REPOSITORY_ROOT / candidate
        if candidate.is_dir():
            benchmark_session = candidate
    if benchmark_session is None:
        new_sessions = [
            path
            for path in output_directory.iterdir()
            if path.is_dir() and path not in existing_sessions
        ]
        if len(new_sessions) != 1:
            raise RuntimeError(
                f"could not identify {label} benchmark session: {new_sessions}"
            )
        benchmark_session = new_sessions[0]
    completed_run_directories = sorted(
        path
        for path in benchmark_session.glob("run_*")
        if (path / "summary.json").is_file()
    )
    if not completed_run_directories:
        raise RuntimeError(f"benchmark output is incomplete: {benchmark_session}")
    selected_run = completed_run_directories[-1]

    if safety_limit_triggered:
        if rss_safety_limit_bytes is None:
            raise RuntimeError("RSS safety limit was triggered without a limit")
        (selected_run / "termination.json").write_text(
            json.dumps(
                {
                    "reason": "rss_safety_limit",
                    "limit_bytes": rss_safety_limit_bytes,
                    "peak_observed_rss_bytes": peak_observed_rss_bytes,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    return BenchmarkProcessResult(
        mode=mode,
        command=command,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        session_directory=benchmark_session.resolve(),
        selected_run_directory=selected_run.resolve(),
        exit_code=exit_code,
        peak_observed_rss_bytes=peak_observed_rss_bytes,
        safety_limit_triggered=safety_limit_triggered,
        reused=False,
    )


def _read_summary(path: Path) -> Mapping[str, object]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected summary object: {path}")
    return {str(key): item for key, item in value.items()}


def _reuse_benchmark(
    mode: BenchmarkMode, run_directory: Path
) -> BenchmarkProcessResult:
    config = _read_summary(run_directory / "config.json")
    summary = _read_summary(run_directory / "summary.json")
    if config.get("mode") != mode:
        raise ValueError(
            f"expected reused {mode} run, received {config.get('mode')}: "
            f"{run_directory}"
        )
    if summary.get("status") != "completed":
        raise ValueError(f"reused run is not completed: {run_directory}")
    if not (run_directory / "output.mp4").is_file():
        raise FileNotFoundError(f"reused run has no output video: {run_directory}")
    peak_rss = summary.get("peak_process_rss_bytes")
    if not isinstance(peak_rss, int) or isinstance(peak_rss, bool):
        raise ValueError(f"reused run has no peak RSS value: {run_directory}")
    return BenchmarkProcessResult(
        mode=mode,
        command=(),
        stdout_path=None,
        stderr_path=None,
        session_directory=run_directory.parent,
        selected_run_directory=run_directory,
        exit_code=0,
        peak_observed_rss_bytes=peak_rss,
        safety_limit_triggered=False,
        reused=True,
    )


def _write_manifest(
    path: Path,
    arguments: DemoArguments,
    original: BenchmarkProcessResult,
    chunked: BenchmarkProcessResult,
    streaming: BenchmarkProcessResult,
    video_path: Path,
) -> None:
    original_summary = _read_summary(original.selected_run_directory / "summary.json")
    chunked_summary = _read_summary(chunked.selected_run_directory / "summary.json")
    streaming_summary = _read_summary(streaming.selected_run_directory / "summary.json")
    chunked_fps = chunked_summary.get("end_to_end_fps")
    streaming_fps = streaming_summary.get("end_to_end_fps")
    if not isinstance(chunked_fps, (int, float)) or isinstance(chunked_fps, bool):
        raise ValueError("chunked summary has no numeric end_to_end_fps")
    if not isinstance(streaming_fps, (int, float)) or isinstance(streaming_fps, bool):
        raise ValueError("streaming summary has no numeric end_to_end_fps")
    speedup_percent = (float(streaming_fps) / float(chunked_fps) - 1.0) * 100.0

    def process_payload(result: BenchmarkProcessResult) -> dict[str, object]:
        return {
            "command": list(result.command),
            "exit_code": result.exit_code,
            "peak_observed_rss_bytes": result.peak_observed_rss_bytes,
            "reused": result.reused,
            "run_directory": str(result.selected_run_directory),
            "safety_limit_triggered": result.safety_limit_triggered,
            "session_directory": str(result.session_directory),
            "stderr": (None if result.stderr_path is None else str(result.stderr_path)),
            "stdout": (None if result.stdout_path is None else str(result.stdout_path)),
        }

    payload = {
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(arguments).items()
        },
        "original": {**process_payload(original), "summary": original_summary},
        "chunked": {**process_payload(chunked), "summary": chunked_summary},
        "streaming": {**process_payload(streaming), "summary": streaming_summary},
        "comparison": {
            "chunked_end_to_end_fps": float(chunked_fps),
            "streaming_end_to_end_fps": float(streaming_fps),
            "streaming_change_percent": speedup_percent,
            "rendered_video": str(video_path),
        },
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    arguments = parse_arguments()
    _validate_inputs(arguments)
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    session_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    session_directory = arguments.output_directory / session_name
    session_directory.mkdir()

    original = _run_benchmark(
        "core-eager",
        arguments,
        session_directory,
        allow_failure=True,
        rss_safety_limit_bytes=round(arguments.original_rss_safety_limit_gib * 1024**3),
    )
    if (
        arguments.reuse_chunked_run is not None
        and arguments.reuse_streaming_run is not None
    ):
        chunked = _reuse_benchmark("demo-chunked", arguments.reuse_chunked_run)
        streaming = _reuse_benchmark("demo-streaming", arguments.reuse_streaming_run)
    else:
        chunked = _run_benchmark("demo-chunked", arguments, session_directory)
        streaming = _run_benchmark("demo-streaming", arguments, session_directory)

    video_path = session_directory / "streamsam_benchmark_replay.mp4"
    print("Rendering synchronized benchmark replay...", flush=True)
    render_comparison(
        original.selected_run_directory,
        chunked.selected_run_directory,
        streaming.selected_run_directory,
        RenderOptions(
            output_path=video_path,
            width=arguments.width,
            height=arguments.height,
            output_fps=arguments.output_fps,
            playback_speed=arguments.playback_speed,
            intro_seconds=arguments.intro_seconds,
            hold_seconds=arguments.hold_seconds,
            rolling_window_frames=arguments.rolling_window_frames,
        ),
    )
    _write_manifest(
        session_directory / "comparison.json",
        arguments,
        original,
        chunked,
        streaming,
        video_path.resolve(),
    )
    print(video_path.resolve())


if __name__ == "__main__":
    main()
