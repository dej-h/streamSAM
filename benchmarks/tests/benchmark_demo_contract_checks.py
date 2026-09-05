from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from benchmarks.video.render_benchmark_comparison import RenderOptions, render_comparison


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def _write_video(path: Path, frame_count: int, width: int, height: int) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not create contract video: {path}")
    try:
        for frame_idx in range(frame_count):
            frame = np.full(
                (height, width, 3),
                (30 + frame_idx * 3, 70, 120),
                dtype=np.uint8,
            )
            cv2.rectangle(
                frame,
                (20 + frame_idx * 2, 45),
                (95 + frame_idx * 2, 130),
                (80, 210, 120),
                -1,
            )
            writer.write(frame)
    finally:
        writer.release()


def _create_run(
    directory: Path,
    *,
    mode: str,
    video_path: Path,
    prompt_path: Path,
    frame_count: int,
    frame_seconds: float,
    end_to_end_fps: float | None,
    status: str = "completed",
    error: str | None = None,
    memory_sample_count: int | None = None,
    configured_max_frames: int | None = None,
) -> None:
    directory.mkdir()
    if status == "completed":
        _write_video(directory / "output.mp4", frame_count, 320, 180)
    config = {
        "mode": mode,
        "video_path": str(video_path),
        "model_config": "edgetam.yaml",
        "checkpoint_path": "/tmp/test-edgetam.pt",
        "prompt_path": str(prompt_path),
        "device": "cuda",
        "dtype": "bfloat16",
        "compile_image_encoder": False,
        "compile_video_pipeline": False,
        "compile_mask_decoder": False,
        "frame_buffer_size": 4,
        "max_frames": (
            frame_count if configured_max_frames is None else configured_max_frames
        ),
        "warmup_frames": 2,
        "repeat_index": 0,
        "chunk_size": 96,
        "max_dimension": 960,
        "memory_sample_interval_seconds": 0.1,
        "offload_video_to_cpu": False,
        "offload_state_to_cpu": False,
        "concurrent_image_encoder": False,
        "profile_gpu_stages": False,
    }
    metadata = {
        "video_sha256": "video",
        "checkpoint_sha256": "checkpoint",
        "prompt_sha256": "prompt",
        "repository_revision": "contract",
        "model_image_size": 1024,
        "source_frame_count": 20,
        "gpu_total_memory_bytes": 8 * 1024**3,
    }
    summary = {
        "status": status,
        "error": error,
        "frames_processed": frame_count,
        "end_to_end_fps": end_to_end_fps,
        "steady_state_fps": end_to_end_fps,
    }
    frames = [
        {
            "source_frame_idx": frame_idx,
            "elapsed_seconds": (frame_idx + 1) * frame_seconds,
            "gpu_inference_ms": frame_seconds * 700.0,
            "frame_wall_ms": frame_seconds * 1000.0,
        }
        for frame_idx in range(frame_count)
    ]
    sample_count = memory_sample_count or frame_count + 1
    memory = [
        {
            "elapsed_seconds": sample_idx * frame_seconds,
            "torch_allocated_bytes": 400_000_000 + sample_idx * 1_000_000,
            "torch_reserved_bytes": 500_000_000,
            "nvml_device_used_bytes": 900_000_000,
            "process_rss_bytes": 1_200_000_000 + sample_idx * 2_000_000,
        }
        for sample_idx in range(sample_count)
    ]
    _write_json(directory / "config.json", config)
    _write_json(directory / "metadata.json", metadata)
    _write_json(directory / "summary.json", summary)
    _write_jsonl(directory / "frames.jsonl", frames)
    _write_jsonl(directory / "memory.jsonl", memory)


def main() -> None:
    frame_count = 20
    with tempfile.TemporaryDirectory(prefix="streamsam-demo-contract-") as temp:
        root = Path(temp)
        source_video = root / "source.mp4"
        prompt_path = root / "prompt.json"
        original_directory = root / "original"
        chunked_directory = root / "chunked"
        streaming_directory = root / "streaming"
        output_path = root / "comparison.mp4"

        _write_video(source_video, frame_count, 320, 180)
        _write_json(
            prompt_path,
            {
                "frame_idx": 0,
                "obj_id": 1,
                "points": [[80.0, 90.0]],
                "labels": [1],
            },
        )
        _create_run(
            original_directory,
            mode="core-eager",
            video_path=source_video,
            prompt_path=prompt_path,
            frame_count=0,
            frame_seconds=0.1,
            end_to_end_fps=None,
            status="failed",
            error="KeyboardInterrupt",
            memory_sample_count=12,
            configured_max_frames=frame_count,
        )
        _write_json(
            original_directory / "termination.json",
            {
                "reason": "rss_safety_limit",
                "limit_bytes": 5 * 1024**3,
                "peak_observed_rss_bytes": 5 * 1024**3,
            },
        )
        _create_run(
            chunked_directory,
            mode="demo-chunked",
            video_path=source_video,
            prompt_path=prompt_path,
            frame_count=frame_count,
            frame_seconds=0.1,
            end_to_end_fps=10.0,
        )
        _create_run(
            streaming_directory,
            mode="demo-streaming",
            video_path=source_video,
            prompt_path=prompt_path,
            frame_count=frame_count,
            frame_seconds=0.05,
            end_to_end_fps=20.0,
        )

        result = render_comparison(
            original_directory,
            chunked_directory,
            streaming_directory,
            RenderOptions(
                output_path=output_path,
                width=1280,
                height=720,
                output_fps=10.0,
                playback_speed=4.0,
                intro_seconds=0.2,
                hold_seconds=0.1,
                rolling_window_frames=5,
            ),
        )
        capture = cv2.VideoCapture(str(output_path))
        if not capture.isOpened():
            raise RuntimeError("rendered contract video cannot be opened")
        rendered_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        rendered_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        rendered_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        success, first_frame = capture.read()
        capture.release()

        if not success or first_frame is None:
            raise RuntimeError("rendered contract video has no decodable frames")
        if rendered_frames != result.output_frames:
            raise RuntimeError(
                f"rendered {rendered_frames} frames, expected {result.output_frames}"
            )
        if (rendered_width, rendered_height) != (1280, 720):
            raise RuntimeError(
                "rendered dimensions differ: " f"{rendered_width}x{rendered_height}"
            )
        if abs(result.speedup_percent - 100.0) > 1e-9:
            raise RuntimeError(
                f"unexpected comparison speedup: {result.speedup_percent}"
            )

        print(
            json.dumps(
                {
                    "render": {
                        **asdict(result),
                        "output_path": str(result.output_path),
                    },
                    "status": "passed",
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
