"""Check decoder selection and native-frame preprocessing against one clip."""

from __future__ import annotations

import argparse
from contextlib import closing
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import av
from av import VideoFrame
import torch

from sam2.utils.rtsp_decoder import RtspDecoderUnavailable, decode_rtsp
from sam2.utils.video_preprocessing import normalize_video_frame
from sam2.utils.video_stream import LiveVideoFrameSource


def check_color_range() -> None:
    frame = VideoFrame(4, 4, "yuv420p")
    frame.colorspace = 1  # BT.709
    frame.color_range = 1  # Limited
    mean = torch.zeros(3, 1, 1)
    std = torch.ones(3, 1, 1)
    for luma, expected in ((16, 0.0), (235, 1.0)):
        for plane, sample in zip(frame.planes, (luma, 128, 128), strict=True):
            plane.update(bytes([sample]) * plane.buffer_size)
        image = normalize_video_frame(frame, 4, mean, std)
        torch.testing.assert_close(image, torch.full((3, 4, 4), expected))
    frame.color_range = 2  # Full
    frame.planes[0].update(bytes([16]) * frame.planes[0].buffer_size)
    torch.testing.assert_close(
        normalize_video_frame(frame, 4, mean, std),
        torch.full((3, 4, 4), 16 / 255),
    )


def check_cpu_selection(video: Path) -> None:
    options = dict(
        inference_device=torch.device("cpu"),
        open_timeout_seconds=None,
        read_timeout_seconds=None,
    )
    with closing(decode_rtsp(str(video), mode="auto", **options)) as frames:
        first, backend, reason = next(frames)
        if first.format.name == "cuda" or backend != "cpu" or not reason:
            raise AssertionError("auto did not report the CPU inference-device fallback")
    with closing(decode_rtsp(str(video), mode="cuda", **options)) as frames:
        try:
            next(frames)
        except RtspDecoderUnavailable:
            pass
        else:
            raise AssertionError("required CUDA silently fell back when inference used CPU")


def check_unsupported_codec() -> None:
    options = dict(
        inference_device=torch.device("cuda:0"),
        open_timeout_seconds=None,
        read_timeout_seconds=None,
    )
    with TemporaryDirectory() as directory:
        video = Path(directory) / "software-only.mkv"
        with av.open(str(video), mode="w") as container:
            stream = container.add_stream("ffv1", rate=1)
            stream.width, stream.height, stream.pix_fmt = 64, 64, "yuv420p"
            frame = VideoFrame(stream.width, stream.height, stream.pix_fmt)
            for plane, sample in zip(frame.planes, (32, 128, 128), strict=True):
                plane.update(bytes([sample]) * plane.buffer_size)
            for packet in stream.encode(frame) + stream.encode():
                container.mux(packet)
        with closing(decode_rtsp(str(video), mode="auto", **options)) as frames:
            _, backend, reason = next(frames)
            if backend != "cpu" or not reason:
                raise AssertionError("unsupported hardware codec did not fall back to software")
        with closing(decode_rtsp(str(video), mode="cuda", **options)) as frames:
            try:
                next(frames)
            except RtspDecoderUnavailable:
                pass
            else:
                raise AssertionError("required CUDA silently decoded an unsupported codec")


def check_cuda_parity(video: Path, count: int) -> float:
    options = dict(
        inference_device=torch.device("cuda:0"),
        open_timeout_seconds=None,
        read_timeout_seconds=None,
    )
    mean = torch.tensor((0.485, 0.456, 0.406))[:, None, None]
    std = torch.tensor((0.229, 0.224, 0.225))[:, None, None]
    # The camera example initializes inference before opening the reader.
    torch.empty(1, device="cuda:0").zero_()
    maximum_difference = 0.0
    with (
        closing(decode_rtsp(str(video), mode="cpu", **options)) as software,
        closing(decode_rtsp(str(video), mode="auto", **options)) as automatic,
    ):
        for _ in range(count):
            cpu_frame, cpu_backend, _ = next(software)
            gpu_frame, gpu_backend, reason = next(automatic)
            if cpu_backend != "cpu" or gpu_backend != "cuda" or reason is not None:
                raise AssertionError(f"automatic decoding failed to select CUDA: {reason}")
            if cpu_frame.time != gpu_frame.time:
                raise AssertionError("decoder presentation timestamps diverged")
            cpu = LiveVideoFrameSource(cpu_frame, 256)
            gpu = LiveVideoFrameSource(gpu_frame, 256)
            try:
                if cpu.device.type != "cpu" or gpu.device.type != "cuda":
                    raise AssertionError("the GPU preprocessed tensor crossed to the CPU")
                cpu_rgb = ((cpu[0] * std + mean) * 255).round()
                gpu_rgb = ((gpu[0].cpu() * std + mean) * 255).round()
                difference = (cpu_rgb - gpu_rgb).abs().max().item()
                maximum_difference = max(maximum_difference, difference)
            finally:
                cpu.close()
                gpu.close()
    if maximum_difference > 1:
        raise AssertionError(f"hardware and CPU preprocessing differ by {maximum_difference} RGB levels")
    return maximum_difference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--frame-count", type=int, default=48)
    args = parser.parse_args()
    if args.frame_count < 1:
        parser.error("--frame-count must be positive")
    torch.set_num_threads(2)
    with torch.inference_mode():
        check_color_range()
        check_cpu_selection(args.video)
        result: dict[str, str | bool | int | float] = {
            "status": "passed", "cpu_selection": True, "color_range": True
        }
        if args.require_cuda:
            result["cuda_frames"] = args.frame_count
            result["maximum_rgb_difference"] = check_cuda_parity(args.video, args.frame_count)
            check_unsupported_codec()
            result["unsupported_codec_fallback"] = True
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
