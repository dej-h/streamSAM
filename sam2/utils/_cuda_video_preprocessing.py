"""Fuse native CUDA 4:2:0 sampling and model normalization into one kernel."""

from __future__ import annotations

import logging
from subprocess import CalledProcessError

import torch
import triton
import triton.language as tl
from triton.errors import TritonError
from triton.language.extra.cuda import libdevice


_LOGGER = logging.getLogger(__name__)
_FAILED_DEVICES: set[torch.device] = set()


@triton.jit
def _sample_plane(
    plane,
    x,
    y,
    mask,
    width: tl.constexpr,
    height: tl.constexpr,
    row_stride: tl.constexpr,
    pixel_stride: tl.constexpr,
    image_size: tl.constexpr,
):
    # Match torch's bilinear half-pixel sampling with edge replication.
    source_x = tl.maximum(tl.fma(x.to(tl.float32) + 0.5, width / image_size, -0.5), 0.0)
    source_y = tl.maximum(tl.fma(y.to(tl.float32) + 0.5, height / image_size, -0.5), 0.0)
    x0, y0 = source_x.to(tl.int32), source_y.to(tl.int32)
    x1, y1 = tl.minimum(x0 + 1, width - 1), tl.minimum(y0 + 1, height - 1)
    wx, wy = source_x - x0, source_y - y0
    top_left = tl.load(plane + y0 * row_stride + x0 * pixel_stride, mask, 0).to(tl.float32)
    top_right = tl.load(plane + y0 * row_stride + x1 * pixel_stride, mask, 0).to(tl.float32)
    bottom_left = tl.load(plane + y1 * row_stride + x0 * pixel_stride, mask, 0).to(tl.float32)
    bottom_right = tl.load(plane + y1 * row_stride + x1 * pixel_stride, mask, 0).to(tl.float32)
    top = tl.fma(1.0 - wx, top_left, wx * top_right)
    bottom = tl.fma(1.0 - wx, bottom_left, wx * bottom_right)
    return tl.fma(1.0 - wy, top, wy * bottom)


@triton.jit
def _normalize_channel(values, mean, std):
    quantized = libdevice.rint(tl.minimum(tl.maximum(values, 0.0), 1.0) * 255.0)
    return tl.div_rn(quantized * (1.0 / 255.0) - mean, std)


@triton.jit
def _normalize_planes(
    y_plane,
    uv_plane,
    mean,
    std,
    output,
    image_size: tl.constexpr,
    y_width: tl.constexpr,
    y_height: tl.constexpr,
    y_stride: tl.constexpr,
    chroma_width: tl.constexpr,
    chroma_height: tl.constexpr,
    chroma_stride: tl.constexpr,
    y_offset: tl.constexpr,
    uv_offset: tl.constexpr,
    y_range: tl.constexpr,
    uv_range: tl.constexpr,
    kr: tl.constexpr,
    kb: tl.constexpr,
    block_size: tl.constexpr,
):
    plane_size: tl.constexpr = image_size * image_size
    index = tl.program_id(0) * block_size + tl.arange(0, block_size)
    valid = index < plane_size
    x, y = index % image_size, index // image_size
    luma = _sample_plane(y_plane, x, y, valid, y_width, y_height, y_stride, 1, image_size)
    u = _sample_plane(uv_plane, x, y, valid, chroma_width, chroma_height, chroma_stride, 2, image_size)
    v = _sample_plane(uv_plane + 1, x, y, valid, chroma_width, chroma_height, chroma_stride, 2, image_size)
    luma = (luma - y_offset) * (1.0 / y_range)
    u = (u - uv_offset) * (1.0 / uv_range)
    v = (v - uv_offset) * (1.0 / uv_range)
    r = luma + 2.0 * (1.0 - kr) * v
    b = luma + 2.0 * (1.0 - kb) * u
    g = (luma - kr * r - kb * b) * (1.0 / (1.0 - kr - kb))
    tl.store(output + index, _normalize_channel(r, tl.load(mean), tl.load(std)), valid)
    tl.store(output + plane_size + index, _normalize_channel(g, tl.load(mean + 1), tl.load(std + 1)), valid)
    tl.store(output + 2 * plane_size + index, _normalize_channel(b, tl.load(mean + 2), tl.load(std + 2)), valid)


def normalize_semiplanar_frame(
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
) -> torch.Tensor | None:
    """Read pitched NV12/P0xx planes and write the final float32 CHW tensor."""
    if y.device in _FAILED_DEVICES:
        return None

    output = torch.empty((3, image_size, image_size), device=y.device, dtype=torch.float32)
    block_size = 256
    try:
        with torch.cuda.device(y.device):
            _normalize_planes[(triton.cdiv(image_size * image_size, block_size),)](
                y, uv, mean, std, output, image_size,
                y.shape[1], y.shape[0], y.stride(0),
                uv.shape[1], uv.shape[0], uv.stride(0),
                y_offset, uv_offset, y_range, uv_range, kr, kb, block_size,
                enable_fp_fusion=False,
            )
    except (TritonError, CalledProcessError, OSError) as error:
        # A missing compiler or unsupported target can still run eager CUDA.
        _LOGGER.warning("CUDA video kernel unavailable on %s: %s", y.device, error)
        _FAILED_DEVICES.add(y.device)
        return None
    return output
