"""Select a PyAV decoder and preserve its native frame storage and metadata."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import nullcontext
from typing import Literal

import av
from av import VideoFrame
from av.codec.hwaccel import HWAccel, hwdevices_available
from av.video.plane import VideoPlane
import torch

from sam2.utils.video_preprocessing import video_frame_device

DecodeMode = Literal["auto", "cpu", "cuda"]
DecoderBackend = Literal["cpu", "cuda"]

# Match the open and read deadlines of the OpenCV FFmpeg backend.
DEFAULT_RTSP_TIMEOUT_SECONDS = 30.0


class RtspDecoderUnavailable(RuntimeError):
    """A requested CUDA decoder cannot provide frames on the inference device."""


def _cuda_device(
    mode: DecodeMode, inference_device: torch.device | None
) -> tuple[torch.device | None, str | None]:
    if mode == "cpu":
        return None, None

    unavailable: str | None = None
    if inference_device is not None and inference_device.type != "cuda":
        unavailable = "inference device is not CUDA"
    elif not torch.cuda.is_available():
        unavailable = "CUDA is unavailable"
    elif not hasattr(VideoPlane, "__dlpack__"):
        unavailable = "the PyAV build cannot share CUDA video frames"
    elif "cuda" not in hwdevices_available():
        unavailable = "the FFmpeg build lacks CUDA decoding support"

    if unavailable is not None:
        if mode == "cuda":
            raise RtspDecoderUnavailable(unavailable)
        return None, unavailable

    device_index = inference_device.index if inference_device is not None else None
    if device_index is None:
        device_index = torch.cuda.current_device()
    return torch.device("cuda", device_index), None


def decode_rtsp(
    url: str,
    *,
    mode: DecodeMode,
    inference_device: torch.device | None,
    open_timeout_seconds: float | None,
    read_timeout_seconds: float | None,
) -> Generator[tuple[VideoFrame, DecoderBackend, str | None], None, None]:
    """Yield frames in display order and report the decoder actually in use.

    FFmpeg negotiates the hardware format during codec initialization.
    Transport errors end the current reader epoch and leave selection to
    the next connection.
    """
    device, fallback_reason = _cuda_device(mode, inference_device)
    timeout = (
        open_timeout_seconds or DEFAULT_RTSP_TIMEOUT_SECONDS,
        read_timeout_seconds or DEFAULT_RTSP_TIMEOUT_SECONDS,
    )
    options = {"rtsp_flags": "prefer_tcp"}
    with torch.cuda.device(device) if device is not None else nullcontext():
        hwaccel = None
        if device is not None:
            # Activate PyTorch's CUDA context on this worker. Asking FFmpeg to
            # retain the primary context can fail after inference starts because
            # FFmpeg requests its own scheduling flags.
            torch.empty(1, device=device).zero_()
            hwaccel = HWAccel(
                "cuda",
                device=str(device.index),
                flags=0,
                options={"primary_ctx": "0", "current_ctx": "1"},
                is_hw_owned=True,
                allow_software_fallback=mode == "auto",
            )
        try:
            container = av.open(url, options=options, timeout=timeout, hwaccel=hwaccel)
        except RuntimeError as error:
            if mode == "cuda" and str(error) == (
                "Hardware accelerated decode requested but no stream is compatible"
            ):
                raise RtspDecoderUnavailable("FFmpeg could not decode this stream on CUDA") from None
            raise
        with container:
            if not container.streams.video:
                raise ValueError("RTSP stream carries no video track")
            for packet in container.demux(container.streams.video[0]):
                try:
                    decoded = packet.decode()
                except av.error.InvalidDataError:
                    # A live client can join between keyframes. The next complete
                    # reference picture can recover the existing decoder.
                    continue
                for frame in decoded:
                    backend: DecoderBackend = "cuda" if frame.format.name == "cuda" else "cpu"
                    reason = fallback_reason
                    if backend == "cpu" and hwaccel is not None:
                        if mode == "cuda":
                            raise RtspDecoderUnavailable("FFmpeg could not decode this stream on CUDA")
                        reason = "FFmpeg selected software decode for this stream"
                    if backend == "cuda" and video_frame_device(frame).type != "cuda":
                        reason = "CUDA decoded pixel layout requires CPU preprocessing"
                        if mode == "cuda":
                            raise RtspDecoderUnavailable(reason)
                    yield frame, backend, reason
