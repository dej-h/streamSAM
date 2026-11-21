"""Convert native decoded video planes into normalized SAM model input."""

from __future__ import annotations

from functools import cache
import logging
from subprocess import CalledProcessError
from typing import Protocol

from av import VideoFrame
from av.video.plane import VideoPlane
import numpy as np
import torch
from torch.nn.functional import interpolate


_LOGGER = logging.getLogger(__name__)

# FFmpeg frame colorspace values and their luma coefficients.
_LUMA_COEFFICIENTS = {
    1: (0.2126, 0.0722),  # BT.709
    4: (0.30, 0.11),  # FCC
    5: (0.299, 0.114),  # BT.470BG
    6: (0.299, 0.114),  # SMPTE 170M
    7: (0.212, 0.087),  # SMPTE 240M
    9: (0.2627, 0.0593),  # BT.2020 non-constant luminance
}
_BT601 = (0.299, 0.114)
_SEMIPLANAR_BIT_DEPTH = {"nv12": 8, "p010le": 10, "p012le": 12, "p016le": 16}
_PLANAR_BIT_DEPTH = {
    "yuv420p": 8,
    "yuvj420p": 8,
    "yuv420p10le": 10,
    "yuv420p12le": 12,
    "yuv420p16le": 16,
}


def _plane_tensor(
    plane: VideoPlane, *, bits: int, channels: int = 1, on_cuda: bool
) -> torch.Tensor:
    if on_cuda:
        # PyAV exports the owning AVFrame reference in the DLPack capsule.
        # It accepts stream=None after the caller synchronizes the device.
        return torch.from_dlpack(plane.__dlpack__())
    dtype = np.dtype(np.uint8 if bits == 8 else "<u2")
    shape = (plane.height, plane.width)
    strides = (plane.line_size, dtype.itemsize * channels)
    if channels == 2:
        shape += (channels,)
        strides += (dtype.itemsize,)
    return torch.from_numpy(np.ndarray(shape, dtype=dtype, buffer=plane, strides=strides))


def _pixel_format(frame: VideoFrame) -> str:
    if frame.format.name != "cuda":
        return frame.format.name
    software_format = getattr(frame, "sw_format", None)
    if software_format is not None:
        return software_format.name
    # PyAV 17 exposes the CUDA planes without exposing AVHWFramesContext.sw_format.
    # Identify its unambiguous 8-bit 4:2:0 layout from the plane shape and type.
    planes = frame.planes
    if len(planes) == 2 and (
        planes[1].width == (frame.width + 1) // 2
        and planes[1].height == (frame.height + 1) // 2
        and torch.from_dlpack(planes[0].__dlpack__()).dtype == torch.uint8
    ):
        return "nv12"
    return "cuda"


def video_frame_device(frame: VideoFrame) -> torch.device:
    """Return the device on which the supported pixel layout is preprocessed."""
    if frame.format.name == "cuda":
        pixel_format = _pixel_format(frame)
        if pixel_format in _SEMIPLANAR_BIT_DEPTH or pixel_format in _PLANAR_BIT_DEPTH:
            device_type, device_id = frame.planes[0].__dlpack_device__()
            if device_type != 2:
                raise RuntimeError("PyAV CUDA frame did not expose a CUDA plane")
            return torch.device("cuda", device_id)
    return torch.device("cpu")


class _NormalizeSemiplanarFrame(Protocol):
    def __call__(
        self,
        y: torch.Tensor,
        uv: torch.Tensor,
        image_size: int,
        mean: torch.Tensor,
        std: torch.Tensor,
        *,
        y_offset: int,
        uv_offset: int,
        y_range: int,
        uv_range: int,
        kr: float,
        kb: float,
    ) -> torch.Tensor | None: ...


@cache
def _cuda_semiplanar_normalizer(device: torch.device) -> _NormalizeSemiplanarFrame | None:
    # Triton is included in supported PyTorch CUDA installations. Other builds
    # can keep the existing eager PyTorch path with the same device and types.
    if torch.cuda.get_device_capability(device)[0] < 8:
        return None
    try:
        from triton.runtime import driver
        from sam2.utils._cuda_video_preprocessing import normalize_semiplanar_frame
    except ImportError:
        return None
    try:
        with torch.cuda.device(device):
            target = driver.active.get_current_target()
    except (AssertionError, CalledProcessError, ImportError, OSError, RuntimeError) as error:
        _LOGGER.warning("CUDA video kernel unavailable on %s: %s", device, error)
        return None
    if target.backend != "cuda":
        return None
    return normalize_semiplanar_frame


def normalize_video_frame(
    frame: VideoFrame,
    image_size: int,
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    input_reads_complete: torch.cuda.Event | None = None,
) -> torch.Tensor:
    """Sample YUV planes at half-pixel centers, then convert and normalize RGB.

    CPU and CUDA use the same coefficients and round to 8-bit RGB after
    conversion. Unspecified matrices use BT.601 and unspecified YUV range is
    limited unless the pixel format is full-range. Native 4:2:0 supports 8 to
    16 bits. PyAV converts other layouts to planar 4:2:0.

    With ``input_reads_complete``, the caller must retain the CUDA frame until
    that event completes. Without it, this function completes all reads before
    returning so the producer can recycle its surface.
    """
    on_cuda = frame.format.name == "cuda"
    pixel_format = _pixel_format(frame)
    full_range = frame.color_range == 2 or (
        frame.color_range == 0 and pixel_format.startswith("yuvj")
    )
    kr, kb = _LUMA_COEFFICIENTS.get(frame.colorspace, _BT601)
    if pixel_format not in _SEMIPLANAR_BIT_DEPTH and pixel_format not in _PLANAR_BIT_DEPTH:
        # PyAV transfers unsupported CUDA layouts to host during reformatting.
        frame = frame.reformat(
            format="yuv420p",
            dst_color_range=2 if full_range else 1,
        )
        pixel_format = frame.format.name
        on_cuda = False

    packed = pixel_format in _SEMIPLANAR_BIT_DEPTH
    bits = _SEMIPLANAR_BIT_DEPTH[pixel_format] if packed else _PLANAR_BIT_DEPTH[pixel_format]
    packing = 2 ** (16 - bits) if packed and bits > 8 else 1
    scale = 2 ** (bits - 8) * packing
    y_offset = 0 if full_range else 16 * scale
    uv_offset = 128 * scale
    y_range = (2**bits - 1) * packing if full_range else 219 * scale
    uv_range = y_range if full_range else 224 * scale

    stream = None
    if on_cuda:
        # PyAV does not publish a producer stream or event through DLPack.
        torch.cuda.synchronize(mean.device)
        stream = torch.cuda.current_stream(mean.device)
    try:
        planes = frame.planes
        y = _plane_tensor(planes[0], bits=bits, on_cuda=on_cuda)
        image = None
        if packed:
            uv = _plane_tensor(planes[1], bits=bits, channels=2, on_cuda=on_cuda)
            if (
                on_cuda and torch.version.cuda
                and mean.shape == std.shape == (3, 1, 1)
                and mean.dtype == std.dtype == torch.float32
                and mean.is_contiguous() and std.is_contiguous()
                and not (mean.requires_grad or std.requires_grad)
                and y.stride(1) == 1 and uv.stride()[1:] == (2, 1)
            ):
                fused = _cuda_semiplanar_normalizer(y.device)
                if fused is not None:
                    image = fused(
                        y, uv, image_size, mean, std,
                        y_offset=y_offset, uv_offset=uv_offset,
                        y_range=y_range, uv_range=uv_range, kr=kr, kb=kb,
                    )
            if image is None:
                uv = uv.permute(2, 0, 1).float()
        else:
            u = _plane_tensor(planes[1], bits=bits, on_cuda=on_cuda).float()
            v = _plane_tensor(planes[2], bits=bits, on_cuda=on_cuda).float()
            uv = torch.stack((u, v))
        if image is None:
            image = _normalize_planes_eager(
                y, uv, image_size, mean, std,
                y_offset=y_offset, uv_offset=uv_offset,
                y_range=y_range, uv_range=uv_range, kr=kr, kb=kb,
            )
        if input_reads_complete is not None and stream is not None:
            input_reads_complete.record(stream)
    except BaseException:
        if stream is not None:
            stream.synchronize()
        raise
    if input_reads_complete is None and stream is not None:
        stream.synchronize()
    return image


def _normalize_planes_eager(
    y: torch.Tensor,
    uv: torch.Tensor,
    image_size: int,
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    y_offset: int,
    uv_offset: int,
    y_range: int,
    uv_range: int,
    kr: float,
    kb: float,
) -> torch.Tensor:
    size = (image_size, image_size)
    y = interpolate(y[None, None].float(), size=size, mode="bilinear", align_corners=False)[0, 0]
    uv = interpolate(uv[None], size=size, mode="bilinear", align_corners=False)[0]
    if y_offset:
        y.sub_(y_offset)
    y.div_(y_range)
    uv.sub_(uv_offset).div_(uv_range)

    u, v = uv.unbind()
    r = y + 2 * (1 - kr) * v
    b = y + 2 * (1 - kb) * u
    g = (y - kr * r - kb * b) / (1 - kr - kb)
    rgb = torch.stack((r, g, b)).clamp_(0, 1).mul_(255).round_().div_(255)
    return rgb.sub_(mean).div_(std)
