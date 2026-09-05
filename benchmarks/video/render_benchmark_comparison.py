from __future__ import annotations

import argparse
import bisect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Final, Iterable, Mapping, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray


Color = tuple[int, int, int]
Point = tuple[float, float]
Image = NDArray[np.uint8]

BACKGROUND: Final[Color] = (20, 22, 25)
PANEL: Final[Color] = (30, 33, 38)
PANEL_BORDER: Final[Color] = (55, 59, 66)
GRID: Final[Color] = (62, 66, 73)
TEXT: Final[Color] = (238, 240, 243)
MUTED_TEXT: Final[Color] = (157, 164, 174)
ORIGINAL: Final[Color] = (104, 104, 232)
CHUNKED: Final[Color] = (92, 171, 255)
STREAMING: Final[Color] = (121, 220, 162)
ACCENT: Final[Color] = (223, 189, 96)
POSITIVE_PROMPT: Final[Color] = (104, 226, 157)
NEGATIVE_PROMPT: Final[Color] = (110, 110, 238)
FONT: Final[int] = cv2.FONT_HERSHEY_DUPLEX
GIB: Final[float] = float(1024**3)


@dataclass(frozen=True)
class FrameTelemetry:
    frame_idx: int
    elapsed_seconds: float
    gpu_inference_ms: float
    frame_wall_ms: float


@dataclass(frozen=True)
class MemoryTelemetry:
    elapsed_seconds: float
    torch_allocated_bytes: int | None
    torch_reserved_bytes: int | None
    nvml_device_used_bytes: int | None
    process_rss_bytes: int


@dataclass(frozen=True)
class SummaryTelemetry:
    status: str
    error: str | None
    frames_processed: int
    end_to_end_fps: float | None
    steady_state_fps: float | None


@dataclass(frozen=True)
class Prompt:
    frame_idx: int
    points: tuple[tuple[float, float], ...]
    labels: tuple[int, ...]


@dataclass(frozen=True)
class BenchmarkRun:
    directory: Path
    mode: str
    source_video: Path
    output_video: Path | None
    prompt_path: Path
    config: Mapping[str, object]
    metadata: Mapping[str, object]
    summary: SummaryTelemetry
    frames: tuple[FrameTelemetry, ...]
    memory: tuple[MemoryTelemetry, ...]
    termination_reason: str | None
    termination_limit_bytes: int | None

    @classmethod
    def load(cls, directory: Path) -> BenchmarkRun:
        directory = directory.resolve()
        config = _read_json_object(directory / "config.json")
        metadata = _read_json_object(directory / "metadata.json")
        raw_summary = _read_json_object(directory / "summary.json")
        frames = tuple(
            _parse_frame(row, directory / "frames.jsonl")
            for row in _read_json_lines(directory / "frames.jsonl")
        )
        memory = tuple(
            _parse_memory(row, directory / "memory.jsonl")
            for row in _read_json_lines(directory / "memory.jsonl")
        )
        summary = SummaryTelemetry(
            status=_required_str(raw_summary, "status"),
            error=_optional_str(raw_summary, "error"),
            frames_processed=_required_int(raw_summary, "frames_processed"),
            end_to_end_fps=_optional_float(raw_summary, "end_to_end_fps"),
            steady_state_fps=_optional_float(raw_summary, "steady_state_fps"),
        )
        if summary.status == "completed" and not frames:
            raise ValueError(f"benchmark run has no frame telemetry: {directory}")
        if summary.frames_processed != len(frames):
            raise ValueError(
                f"summary reports {summary.frames_processed} frames but telemetry has "
                f"{len(frames)}: {directory}"
            )
        _validate_frame_sequence(frames, directory)
        _validate_monotonic_memory(memory, directory)

        source_video = Path(_required_str(config, "video_path")).resolve()
        prompt_path = Path(_required_str(config, "prompt_path")).resolve()
        candidate_output_video = directory / "output.mp4"
        output_video = (
            candidate_output_video if candidate_output_video.is_file() else None
        )
        for path, label in (
            (source_video, "source video"),
            (prompt_path, "prompt"),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{label} does not exist: {path}")

        termination_path = directory / "termination.json"
        termination_reason: str | None = None
        termination_limit_bytes: int | None = None
        if termination_path.is_file():
            termination_data = _read_json_object(termination_path)
            termination_reason = _required_str(termination_data, "reason")
            termination_limit_bytes = _optional_int(termination_data, "limit_bytes")

        return cls(
            directory=directory,
            mode=_required_str(config, "mode"),
            source_video=source_video,
            output_video=output_video,
            prompt_path=prompt_path,
            config=config,
            metadata=metadata,
            summary=summary,
            frames=frames,
            memory=memory,
            termination_reason=termination_reason,
            termination_limit_bytes=termination_limit_bytes,
        )


@dataclass(frozen=True)
class RenderOptions:
    output_path: Path
    width: int = 1920
    height: int = 1080
    output_fps: float = 30.0
    playback_speed: float = 2.0
    intro_seconds: float = 2.0
    hold_seconds: float = 2.0
    rolling_window_frames: int = 30

    def validate(self) -> None:
        if self.width < 1280 or self.height < 720:
            raise ValueError("output dimensions must be at least 1280x720")
        if self.output_fps <= 0:
            raise ValueError("output FPS must be positive")
        if self.playback_speed <= 0:
            raise ValueError("playback speed must be positive")
        if self.intro_seconds < 0 or self.hold_seconds < 0:
            raise ValueError("intro and hold durations cannot be negative")
        if self.rolling_window_frames < 2:
            raise ValueError("rolling FPS window must contain at least two frames")


@dataclass(frozen=True)
class ChartSeries:
    label: str
    color: Color
    points: tuple[Point, ...]


@dataclass(frozen=True)
class RenderResult:
    output_path: Path
    output_frames: int
    output_fps: float
    benchmark_seconds: float
    media_seconds: float
    speedup_percent: float


class SequentialVideoReader:
    """Read monotonically increasing video indices without random seeks."""

    def __init__(self, path: Path) -> None:
        self._capture = cv2.VideoCapture(str(path))
        if not self._capture.isOpened():
            raise RuntimeError(f"could not open video: {path}")
        self._path = path
        self._frame_idx = -1
        self._frame: Image | None = None

    @property
    def frame_count(self) -> int:
        return int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def read_through(self, frame_idx: int) -> Image:
        if frame_idx < self._frame_idx:
            raise ValueError("video reader cannot move backward")
        while self._frame_idx < frame_idx:
            success, frame = self._capture.read()
            if not success:
                raise RuntimeError(
                    f"video ended before frame {frame_idx}: {self._path}"
                )
            self._frame_idx += 1
            self._frame = frame
        if self._frame is None:
            raise RuntimeError(f"video contains no frames: {self._path}")
        return self._frame

    def close(self) -> None:
        self._capture.release()

    def __enter__(self) -> SequentialVideoReader:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _read_json_object(path: Path) -> dict[str, object]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"JSON object contains a non-string key: {path}")
        result[key] = item
    return result


def _read_json_lines(path: Path) -> Iterable[dict[str, object]]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value: object = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            row: dict[str, object] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ValueError(f"non-string key at {path}:{line_number}")
                row[key] = item
            yield row


def _required_str(data: Mapping[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_str(data: Mapping[str, object], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string or null")
    return value


def _required_int(data: Mapping[str, object], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def _optional_int(data: Mapping[str, object], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer or null")
    return value


def _required_float(data: Mapping[str, object], key: str) -> float:
    value = data.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{key} must be finite")
    return result


def _optional_float(data: Mapping[str, object], key: str) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    return _required_float(data, key)


def _parse_frame(data: Mapping[str, object], source: Path) -> FrameTelemetry:
    try:
        return FrameTelemetry(
            frame_idx=_required_int(data, "source_frame_idx"),
            elapsed_seconds=_required_float(data, "elapsed_seconds"),
            gpu_inference_ms=_required_float(data, "gpu_inference_ms"),
            frame_wall_ms=_required_float(data, "frame_wall_ms"),
        )
    except ValueError as error:
        raise ValueError(f"invalid frame telemetry in {source}: {error}") from error


def _parse_memory(data: Mapping[str, object], source: Path) -> MemoryTelemetry:
    try:
        return MemoryTelemetry(
            elapsed_seconds=_required_float(data, "elapsed_seconds"),
            torch_allocated_bytes=_optional_int(data, "torch_allocated_bytes"),
            torch_reserved_bytes=_optional_int(data, "torch_reserved_bytes"),
            nvml_device_used_bytes=_optional_int(data, "nvml_device_used_bytes"),
            process_rss_bytes=_required_int(data, "process_rss_bytes"),
        )
    except ValueError as error:
        raise ValueError(f"invalid memory telemetry in {source}: {error}") from error


def _validate_frame_sequence(frames: Sequence[FrameTelemetry], directory: Path) -> None:
    previous_elapsed = -math.inf
    for expected_idx, frame in enumerate(frames):
        if frame.frame_idx != expected_idx:
            raise ValueError(
                f"frame telemetry is not contiguous at {expected_idx}: {directory}"
            )
        if frame.elapsed_seconds < previous_elapsed:
            raise ValueError(f"frame timestamps are not monotonic: {directory}")
        previous_elapsed = frame.elapsed_seconds


def _validate_monotonic_memory(
    samples: Sequence[MemoryTelemetry], directory: Path
) -> None:
    previous_elapsed = -math.inf
    for sample in samples:
        if sample.elapsed_seconds < previous_elapsed:
            raise ValueError(f"memory timestamps are not monotonic: {directory}")
        previous_elapsed = sample.elapsed_seconds


def _parse_prompt(path: Path) -> Prompt:
    data = _read_json_object(path)
    raw_points = data.get("points")
    raw_labels = data.get("labels")
    if not isinstance(raw_points, list) or not isinstance(raw_labels, list):
        raise ValueError("prompt points and labels must be arrays")
    if len(raw_points) != len(raw_labels) or not raw_points:
        raise ValueError("prompt must contain matching non-empty points and labels")

    points: list[tuple[float, float]] = []
    labels: list[int] = []
    for raw_point, raw_label in zip(raw_points, raw_labels, strict=True):
        if not isinstance(raw_point, list) or len(raw_point) != 2:
            raise ValueError("each prompt point must contain x and y")
        x, y = raw_point
        if not isinstance(x, (int, float)) or isinstance(x, bool):
            raise ValueError("prompt x coordinates must be numeric")
        if not isinstance(y, (int, float)) or isinstance(y, bool):
            raise ValueError("prompt y coordinates must be numeric")
        if not isinstance(raw_label, int) or raw_label not in (0, 1):
            raise ValueError("prompt labels must be 0 or 1")
        points.append((float(x), float(y)))
        labels.append(raw_label)
    return Prompt(
        frame_idx=_required_int(data, "frame_idx"),
        points=tuple(points),
        labels=tuple(labels),
    )


def _validate_matched_runs(
    original: BenchmarkRun,
    chunked: BenchmarkRun,
    streaming: BenchmarkRun,
) -> None:
    expected_modes = (
        (original, "core-eager"),
        (chunked, "demo-chunked"),
        (streaming, "demo-streaming"),
    )
    for run, expected_mode in expected_modes:
        if run.mode != expected_mode:
            raise ValueError(
                f"expected {expected_mode} run, received {run.mode}: {run.directory}"
            )
    for run, label in ((chunked, "chunked"), (streaming, "streaming")):
        if run.summary.status != "completed":
            raise ValueError(f"{label} run did not complete: {run.directory}")
        if run.summary.end_to_end_fps is None:
            raise ValueError(f"{label} run has no end-to-end FPS: {run.directory}")
    if streaming.output_video is None:
        raise FileNotFoundError(
            f"streaming segmented output video is missing: {streaming.directory}"
        )

    matched_keys = (
        "video_path",
        "model_config",
        "checkpoint_path",
        "prompt_path",
        "device",
        "dtype",
        "max_frames",
        "warmup_frames",
        "memory_sample_interval_seconds",
        "compile_image_encoder",
        "compile_video_pipeline",
        "compile_mask_decoder",
    )
    for candidate, label in ((original, "original"), (streaming, "streaming")):
        mismatches = [
            key
            for key in matched_keys
            if chunked.config.get(key) != candidate.config.get(key)
        ]
        if mismatches:
            raise ValueError(
                f"chunked and {label} configurations differ for: "
                + ", ".join(mismatches)
            )
    if chunked.summary.frames_processed != streaming.summary.frames_processed:
        raise ValueError("chunked and streaming runs processed different frame counts")

    for hash_key in ("video_sha256", "checkpoint_sha256", "prompt_sha256"):
        hashes = {run.metadata.get(hash_key) for run in (original, chunked, streaming)}
        if None not in hashes and len(hashes) != 1:
            raise ValueError(f"strategy {hash_key} values differ")


def _read_prompt_frame(video_path: Path, prompt: Prompt) -> Image:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open source video: {video_path}")
    try:
        if prompt.frame_idx > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, prompt.frame_idx)
        success, frame = capture.read()
    finally:
        capture.release()
    if not success:
        raise RuntimeError(f"could not read prompt frame {prompt.frame_idx}")

    marker_radius = max(10, round(min(frame.shape[:2]) * 0.018))
    for (x, y), label in zip(prompt.points, prompt.labels, strict=True):
        center = (round(x), round(y))
        color = POSITIVE_PROMPT if label == 1 else NEGATIVE_PROMPT
        cv2.circle(frame, center, marker_radius + 4, (15, 17, 20), -1, cv2.LINE_AA)
        cv2.circle(frame, center, marker_radius, color, 4, cv2.LINE_AA)
        cv2.line(
            frame,
            (center[0] - marker_radius // 2, center[1]),
            (center[0] + marker_radius // 2, center[1]),
            color,
            3,
            cv2.LINE_AA,
        )
        cv2.line(
            frame,
            (center[0], center[1] - marker_radius // 2),
            (center[0], center[1] + marker_radius // 2),
            color,
            3,
            cv2.LINE_AA,
        )
    return frame


def _rolling_fps(
    frames: Sequence[FrameTelemetry], window_frames: int
) -> tuple[Point, ...]:
    points: list[Point] = []
    for index, frame in enumerate(frames):
        start_index = max(0, index - window_frames + 1)
        start = frames[start_index]
        elapsed = frame.elapsed_seconds - start.elapsed_seconds
        intervals = index - start_index
        if intervals > 0 and elapsed > 0:
            points.append((frame.elapsed_seconds, intervals / elapsed))
    return tuple(points)


def _memory_points(
    samples: Sequence[MemoryTelemetry],
    selector: Callable[[MemoryTelemetry], int | None],
) -> tuple[Point, ...]:
    points: list[Point] = []
    for sample in samples:
        value = selector(sample)
        if value is not None:
            points.append((sample.elapsed_seconds, value / GIB))
    return tuple(points)


def _put_text(
    image: Image,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float,
    color: Color = TEXT,
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        FONT,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _draw_panel(image: Image, rect: tuple[int, int, int, int]) -> None:
    x, y, width, height = rect
    cv2.rectangle(image, (x, y), (x + width, y + height), PANEL, -1)
    cv2.rectangle(image, (x, y), (x + width, y + height), PANEL_BORDER, 1, cv2.LINE_AA)


def _fit_image(source: Image, width: int, height: int) -> Image:
    source_height, source_width = source.shape[:2]
    scale = min(width / source_width, height / source_height)
    target_width = max(1, round(source_width * scale))
    target_height = max(1, round(source_height * scale))
    resized = cv2.resize(
        source,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    fitted = np.full((height, width, 3), BACKGROUND, dtype=np.uint8)
    x = (width - target_width) // 2
    y = (height - target_height) // 2
    fitted[y : y + target_height, x : x + target_width] = resized
    return fitted


def _latest_index(timestamps: Sequence[float], current_time: float) -> int:
    return bisect.bisect_right(timestamps, current_time) - 1


def _format_gib(value: int | None) -> str:
    return "n/a" if value is None else f"{value / GIB:.2f} GiB"


def _format_fps(value: float) -> str:
    return f"{value:.2f} FPS"


def _original_outcome(run: BenchmarkRun) -> str:
    if run.termination_reason == "rss_safety_limit":
        return "SAFETY STOP"
    error = (run.summary.error or "").lower()
    if "out of memory" in error or "cuda oom" in error:
        return "OOM"
    if run.summary.status == "completed":
        return "COMPLETED"
    return "FAILED"


def _original_outcome_detail(run: BenchmarkRun) -> str:
    if (
        run.termination_reason == "rss_safety_limit"
        and run.termination_limit_bytes is not None
    ):
        return f"at {run.termination_limit_bytes / GIB:.1f} GiB RSS"
    if run.summary.status == "failed":
        return "before inference"
    return f"{run.summary.frames_processed} frames"


def _eager_storage_text(run: BenchmarkRun) -> str:
    frame_count = run.metadata.get("source_frame_count")
    if frame_count is None:
        frame_count = run.config.get("max_frames")
    image_size = run.metadata.get("model_image_size")
    gpu_total_bytes = run.metadata.get("gpu_total_memory_bytes")
    if (
        not isinstance(frame_count, int)
        or isinstance(frame_count, bool)
        or not isinstance(image_size, int)
        or isinstance(image_size, bool)
        or not isinstance(gpu_total_bytes, int)
        or isinstance(gpu_total_bytes, bool)
    ):
        return "eager storage requirement unavailable"
    required_bytes = frame_count * 3 * image_size * image_size * 4
    return (
        f"eager preloads {frame_count} frames = {required_bytes / GIB:.2f} GiB  >  "
        f"GPU capacity {gpu_total_bytes / GIB:.2f} GiB"
    )


def _draw_chart(
    image: Image,
    rect: tuple[int, int, int, int],
    *,
    title: str,
    series: Sequence[ChartSeries],
    current_time: float,
    x_max: float,
    y_max: float,
    unit: str,
) -> None:
    _draw_panel(image, rect)
    x, y, width, height = rect
    _put_text(image, title, (x + 18, y + 30), scale=0.55, thickness=1)

    plot_left = x + 58
    plot_right = x + width - 18
    plot_top = y + 48
    plot_bottom = y + height - 27
    for step in range(5):
        grid_y = round(plot_bottom - (plot_bottom - plot_top) * step / 4)
        cv2.line(
            image,
            (plot_left, grid_y),
            (plot_right, grid_y),
            GRID,
            1,
            cv2.LINE_AA,
        )
        label_value = y_max * step / 4
        label = f"{label_value:.0f}" if y_max >= 10 else f"{label_value:.1f}"
        _put_text(
            image,
            label,
            (x + 8, grid_y + 5),
            scale=0.34,
            color=MUTED_TEXT,
        )

    x_denominator = max(x_max, 1e-9)
    y_denominator = max(y_max, 1e-9)
    for item in series:
        visible = [point for point in item.points if point[0] <= current_time]
        if len(visible) >= 2:
            pixels = np.asarray(
                [
                    (
                        round(
                            plot_left
                            + min(max(point_x / x_denominator, 0.0), 1.0)
                            * (plot_right - plot_left)
                        ),
                        round(
                            plot_bottom
                            - min(max(point_y / y_denominator, 0.0), 1.0)
                            * (plot_bottom - plot_top)
                        ),
                    )
                    for point_x, point_y in visible
                ],
                dtype=np.int32,
            )
            cv2.polylines(image, [pixels], False, item.color, 2, cv2.LINE_AA)

    legend_x = max(plot_left, x + width - 252)
    for item in series:
        cv2.line(
            image,
            (legend_x, y + 25),
            (legend_x + 14, y + 25),
            item.color,
            3,
            cv2.LINE_AA,
        )
        _put_text(
            image,
            item.label,
            (legend_x + 19, y + 29),
            scale=0.29,
            color=MUTED_TEXT,
        )
        legend_x += 78
    _put_text(
        image,
        "0s",
        (plot_left, plot_bottom + 17),
        scale=0.29,
        color=MUTED_TEXT,
    )
    _put_text(
        image,
        f"{x_max:.1f}s",
        (plot_right - 38, plot_bottom + 17),
        scale=0.29,
        color=MUTED_TEXT,
    )
    _put_text(
        image,
        unit,
        (plot_right - 32, plot_top + 12),
        scale=0.27,
        color=MUTED_TEXT,
    )


def _max_y(series: Sequence[ChartSeries], x_max: float, minimum: float) -> float:
    values = [
        y for item in series for x, y in item.points if x <= x_max and math.isfinite(y)
    ]
    return max(minimum, max(values, default=minimum) * 1.08)


def _latest_memory(
    samples: Sequence[MemoryTelemetry], current_time: float
) -> MemoryTelemetry | None:
    if not samples:
        return None
    timestamps = [sample.elapsed_seconds for sample in samples]
    index = _latest_index(timestamps, current_time)
    return samples[index] if index >= 0 else None


def _render_frame(
    *,
    original: BenchmarkRun,
    chunked: BenchmarkRun,
    streaming: BenchmarkRun,
    prompt_frame: Image,
    segmented_frame: Image | None,
    benchmark_time: float,
    benchmark_duration: float,
    playback_speed: float,
    width: int,
    height: int,
    charts: Sequence[tuple[str, Sequence[ChartSeries], float, str]],
) -> Image:
    canvas = np.full((height, width, 3), BACKGROUND, dtype=np.uint8)
    margin = max(18, round(width * 0.016))
    header_height = max(92, round(height * 0.1))
    content_top = header_height + margin
    content_bottom = height - margin
    gap = max(14, round(width * 0.009))
    right_width = round(width * 0.35)
    left_width = width - margin * 2 - gap - right_width
    right_x = margin + left_width + gap

    _put_text(canvas, "streamSAM", (margin, 48), scale=1.15, thickness=2)
    _put_text(
        canvas,
        "THREE EXECUTION STRATEGIES / RECORDED TELEMETRY",
        (margin, 78),
        scale=0.42,
        color=ACCENT,
        thickness=1,
    )

    chunked_fps = chunked.summary.end_to_end_fps
    streaming_fps = streaming.summary.end_to_end_fps
    if chunked_fps is None or streaming_fps is None:
        raise ValueError("completed strategies require end-to-end FPS")
    improvement = (streaming_fps / chunked_fps - 1.0) * 100.0
    stat_x = width - margin - 790
    _put_text(canvas, "ORIGINAL EAGER", (stat_x, 28), scale=0.34, color=MUTED_TEXT)
    _put_text(
        canvas,
        _original_outcome(original),
        (stat_x, 59),
        scale=0.60,
        color=ORIGINAL,
    )
    _put_text(
        canvas,
        _original_outcome_detail(original),
        (stat_x, 82),
        scale=0.31,
        color=MUTED_TEXT,
    )
    _put_text(
        canvas, "CHUNKED BATCHES", (stat_x + 265, 28), scale=0.34, color=MUTED_TEXT
    )
    _put_text(
        canvas,
        _format_fps(chunked_fps),
        (stat_x + 265, 59),
        scale=0.72,
        color=CHUNKED,
    )
    _put_text(canvas, "STREAMING", (stat_x + 535, 28), scale=0.34, color=MUTED_TEXT)
    _put_text(
        canvas,
        _format_fps(streaming_fps),
        (stat_x + 535, 59),
        scale=0.72,
        color=STREAMING,
    )
    _put_text(
        canvas,
        f"+{improvement:.1f}% vs chunked",
        (stat_x + 535, 82),
        scale=0.31,
        color=ACCENT,
    )

    left_bottom_height = max(150, round((content_bottom - content_top) * 0.18))
    video_height = content_bottom - content_top - gap - left_bottom_height
    video_rect = (margin, content_top, left_width, video_height)
    _draw_panel(canvas, video_rect)
    vx, vy, vw, vh = video_rect
    active_frame = prompt_frame if segmented_frame is None else segmented_frame
    fitted = _fit_image(active_frame, vw - 4, vh - 4)
    canvas[vy + 2 : vy + vh - 2, vx + 2 : vx + vw - 2] = fitted

    state_label = "INITIAL POINT PROMPT" if segmented_frame is None else "SEGMENTATION"
    cv2.rectangle(canvas, (vx + 18, vy + 18), (vx + 305, vy + 58), BACKGROUND, -1)
    _put_text(canvas, state_label, (vx + 32, vy + 46), scale=0.56, color=TEXT)

    streaming_timestamps = [frame.elapsed_seconds for frame in streaming.frames]
    chunked_timestamps = [frame.elapsed_seconds for frame in chunked.frames]
    streaming_index = _latest_index(streaming_timestamps, benchmark_time)
    chunked_index = _latest_index(chunked_timestamps, benchmark_time)

    info_rect = (
        margin,
        content_top + video_height + gap,
        left_width,
        left_bottom_height,
    )
    _draw_panel(canvas, info_rect)
    ix, iy, iw, ih = info_rect
    prompt_thumb_width = round(iw * 0.25)
    prompt_thumb = _fit_image(prompt_frame, prompt_thumb_width - 24, ih - 34)
    canvas[
        iy + 17 : iy + 17 + prompt_thumb.shape[0],
        ix + 12 : ix + 12 + prompt_thumb.shape[1],
    ] = prompt_thumb
    _put_text(
        canvas,
        "POINT PROMPT",
        (ix + 22, iy + ih - 14),
        scale=0.34,
        color=MUTED_TEXT,
    )

    metric_x = ix + prompt_thumb_width + 24
    processed = max(streaming_index + 1, 0)
    total = len(streaming.frames)
    _put_text(
        canvas,
        f"FRAME  {processed:04d} / {total:04d}",
        (metric_x, iy + 40),
        scale=0.72,
        thickness=1,
    )
    current_streaming = (
        streaming.frames[streaming_index] if streaming_index >= 0 else None
    )
    current_chunked = chunked.frames[chunked_index] if chunked_index >= 0 else None
    streaming_latency = (
        "n/a"
        if current_streaming is None
        else f"{current_streaming.frame_wall_ms:.1f} ms"
    )
    chunked_latency = (
        "n/a" if current_chunked is None else f"{current_chunked.frame_wall_ms:.1f} ms"
    )
    _put_text(
        canvas,
        f"frame latency   chunked {chunked_latency}   streaming {streaming_latency}",
        (metric_x, iy + 77),
        scale=0.46,
        color=MUTED_TEXT,
    )

    streaming_memory = _latest_memory(streaming.memory, benchmark_time)
    current_vram = (
        None if streaming_memory is None else streaming_memory.torch_allocated_bytes
    )
    current_rss = (
        None if streaming_memory is None else streaming_memory.process_rss_bytes
    )
    _put_text(
        canvas,
        f"streaming VRAM  {_format_gib(current_vram)}    RSS  {_format_gib(current_rss)}",
        (metric_x, iy + 111),
        scale=0.46,
        color=MUTED_TEXT,
    )
    _put_text(
        canvas,
        (
            f"{_eager_storage_text(original)}    "
            f"t {benchmark_time:4.1f}/{benchmark_duration:4.1f}s  {playback_speed:.1f}x"
        ),
        (metric_x, iy + 143),
        scale=0.38,
        color=ORIGINAL,
    )

    chart_gap = max(10, round(height * 0.01))
    available_chart_height = content_bottom - content_top
    chart_height = (available_chart_height - chart_gap * (len(charts) - 1)) // len(
        charts
    )
    chart_y = content_top
    for title, series, y_max, unit in charts:
        _draw_chart(
            canvas,
            (right_x, chart_y, right_width, chart_height),
            title=title,
            series=series,
            current_time=benchmark_time,
            x_max=benchmark_duration,
            y_max=y_max,
            unit=unit,
        )
        chart_y += chart_height + chart_gap

    return canvas


def render_comparison(
    original_directory: Path,
    chunked_directory: Path,
    streaming_directory: Path,
    options: RenderOptions,
) -> RenderResult:
    """Render synchronized telemetry after all execution strategies finish."""
    options.validate()
    original = BenchmarkRun.load(original_directory)
    chunked = BenchmarkRun.load(chunked_directory)
    streaming = BenchmarkRun.load(streaming_directory)
    _validate_matched_runs(original, chunked, streaming)
    prompt = _parse_prompt(streaming.prompt_path)
    prompt_frame = _read_prompt_frame(streaming.source_video, prompt)

    benchmark_duration = max(
        original.memory[-1].elapsed_seconds if original.memory else 0.0,
        chunked.frames[-1].elapsed_seconds,
        streaming.frames[-1].elapsed_seconds,
    )
    if benchmark_duration <= 0:
        raise ValueError("benchmark timeline duration must be positive")
    replay_seconds = benchmark_duration / options.playback_speed
    media_seconds = options.intro_seconds + replay_seconds + options.hold_seconds
    output_frame_count = max(1, math.ceil(media_seconds * options.output_fps))

    progress_series = (
        ChartSeries(
            "eager",
            ORIGINAL,
            tuple(
                (frame.elapsed_seconds, frame.frame_idx + 1)
                for frame in original.frames
            ),
        ),
        ChartSeries(
            "chunked",
            CHUNKED,
            tuple(
                (frame.elapsed_seconds, frame.frame_idx + 1) for frame in chunked.frames
            ),
        ),
        ChartSeries(
            "stream",
            STREAMING,
            tuple(
                (frame.elapsed_seconds, frame.frame_idx + 1)
                for frame in streaming.frames
            ),
        ),
    )
    fps_series = (
        ChartSeries(
            "eager",
            ORIGINAL,
            _rolling_fps(original.frames, options.rolling_window_frames),
        ),
        ChartSeries(
            "chunked",
            CHUNKED,
            _rolling_fps(chunked.frames, options.rolling_window_frames),
        ),
        ChartSeries(
            "stream",
            STREAMING,
            _rolling_fps(streaming.frames, options.rolling_window_frames),
        ),
    )
    vram_series = (
        ChartSeries(
            "eager",
            ORIGINAL,
            _memory_points(
                original.memory, lambda sample: sample.torch_allocated_bytes
            ),
        ),
        ChartSeries(
            "chunked",
            CHUNKED,
            _memory_points(chunked.memory, lambda sample: sample.torch_allocated_bytes),
        ),
        ChartSeries(
            "stream",
            STREAMING,
            _memory_points(
                streaming.memory, lambda sample: sample.torch_allocated_bytes
            ),
        ),
    )
    ram_series = (
        ChartSeries(
            "eager",
            ORIGINAL,
            _memory_points(original.memory, lambda sample: sample.process_rss_bytes),
        ),
        ChartSeries(
            "chunked",
            CHUNKED,
            _memory_points(chunked.memory, lambda sample: sample.process_rss_bytes),
        ),
        ChartSeries(
            "stream",
            STREAMING,
            _memory_points(streaming.memory, lambda sample: sample.process_rss_bytes),
        ),
    )
    charts = (
        (
            "PROCESSED FRAMES",
            progress_series,
            float(len(streaming.frames)),
            "frames",
        ),
        (
            f"ROLLING FPS ({options.rolling_window_frames} frames)",
            fps_series,
            _max_y(fps_series, benchmark_duration, 10.0),
            "FPS",
        ),
        (
            "TORCH VRAM ALLOCATED",
            vram_series,
            _max_y(vram_series, benchmark_duration, 0.5),
            "GiB",
        ),
        (
            "PROCESS RSS",
            ram_series,
            _max_y(ram_series, benchmark_duration, 1.0),
            "GiB",
        ),
    )

    options.output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(options.output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        options.output_fps,
        (options.width, options.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not create demo video: {options.output_path}")

    streaming_timestamps = [frame.elapsed_seconds for frame in streaming.frames]
    segmented_frame: Image | None = None
    try:
        if streaming.output_video is None:
            raise RuntimeError("validated streaming output video is unavailable")
        with SequentialVideoReader(streaming.output_video) as segmented_reader:
            if segmented_reader.frame_count not in (0, len(streaming.frames)):
                raise ValueError(
                    "segmented output frame count does not match frame telemetry: "
                    f"{segmented_reader.frame_count} != {len(streaming.frames)}"
                )
            for output_frame_idx in range(output_frame_count):
                media_time = output_frame_idx / options.output_fps
                if media_time < options.intro_seconds:
                    benchmark_time = 0.0
                else:
                    replay_time = media_time - options.intro_seconds
                    benchmark_time = min(
                        benchmark_duration, replay_time * options.playback_speed
                    )
                telemetry_idx = _latest_index(streaming_timestamps, benchmark_time)
                if telemetry_idx >= 0:
                    segmented_frame = segmented_reader.read_through(telemetry_idx)
                rendered = _render_frame(
                    original=original,
                    chunked=chunked,
                    streaming=streaming,
                    prompt_frame=prompt_frame,
                    segmented_frame=segmented_frame,
                    benchmark_time=benchmark_time,
                    benchmark_duration=benchmark_duration,
                    playback_speed=options.playback_speed,
                    width=options.width,
                    height=options.height,
                    charts=charts,
                )
                writer.write(rendered)
    finally:
        writer.release()

    chunked_fps = chunked.summary.end_to_end_fps
    streaming_fps = streaming.summary.end_to_end_fps
    if chunked_fps is None or streaming_fps is None:
        raise ValueError("completed strategies require end-to-end FPS")
    improvement = (streaming_fps / chunked_fps - 1.0) * 100.0
    return RenderResult(
        output_path=options.output_path.resolve(),
        output_frames=output_frame_count,
        output_fps=options.output_fps,
        benchmark_seconds=benchmark_duration,
        media_seconds=output_frame_count / options.output_fps,
        speedup_percent=improvement,
    )


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render an offline, synchronized video from matched benchmark artifacts."
        )
    )
    parser.add_argument("--original-run", type=Path, required=True)
    parser.add_argument("--chunked-run", type=Path, required=True)
    parser.add_argument("--streaming-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--output-fps", type=float, default=30.0)
    parser.add_argument("--playback-speed", type=float, default=2.0)
    parser.add_argument("--intro-seconds", type=float, default=2.0)
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    parser.add_argument("--rolling-window-frames", type=int, default=30)
    return parser.parse_args(argv)


def main() -> None:
    arguments = parse_arguments()
    result = render_comparison(
        arguments.original_run,
        arguments.chunked_run,
        arguments.streaming_run,
        RenderOptions(
            output_path=arguments.output,
            width=arguments.width,
            height=arguments.height,
            output_fps=arguments.output_fps,
            playback_speed=arguments.playback_speed,
            intro_seconds=arguments.intro_seconds,
            hold_seconds=arguments.hold_seconds,
            rolling_window_frames=arguments.rolling_window_frames,
        ),
    )
    print(
        json.dumps(
            {
                "benchmark_seconds": result.benchmark_seconds,
                "media_seconds": result.media_seconds,
                "output_fps": result.output_fps,
                "output_frames": result.output_frames,
                "output_path": str(result.output_path),
                "speedup_percent": result.speedup_percent,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
