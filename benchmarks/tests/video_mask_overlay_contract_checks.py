"""Check mask overlay pixels, source ownership, and output buffer reuse."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sam2.utils.video_output_writer import _MaskOverlay


def _reference_overlay(
    frame_bgr: NDArray[np.uint8],
    masks: NDArray[np.bool_],
    alpha: float,
    color_rgb: tuple[int, int, int],
) -> NDArray[np.uint8]:
    foreground = np.any(masks, axis=0)
    reference = frame_bgr.astype(np.float32)
    color_bgr = np.asarray(color_rgb[::-1], dtype=np.float32)
    reference[foreground] = reference[foreground] * (1.0 - alpha) + color_bgr * alpha
    return reference.astype(np.uint8)


def check_mask_overlay() -> None:
    levels = np.arange(256, dtype=np.uint8).reshape(16, 16)
    source = np.stack((levels, levels[::-1], levels[:, ::-1]), axis=-1)
    source.setflags(write=False)
    masks = np.stack((levels < 96, levels > 64))
    color_rgb = (31, 119, 180)

    for alpha in (0.0, 0.037, 0.6, 1.0):
        overlay = _MaskOverlay((16, 16), alpha, color_rgb)
        output = overlay.compose(source, np.ones_like(masks))
        np.testing.assert_array_equal(
            output, _reference_overlay(source, np.ones_like(masks), alpha, color_rgb)
        )
        for selected in (masks, masks[:1]):
            assert overlay.compose(source, selected) is output
            np.testing.assert_array_equal(
                output, _reference_overlay(source, selected, alpha, color_rgb)
            )
        for empty in (masks[:0], np.zeros_like(masks)):
            assert overlay.compose(source, empty) is source
        np.testing.assert_array_equal(
            source, np.stack((levels, levels[::-1], levels[:, ::-1]), axis=-1)
        )


if __name__ == "__main__":
    check_mask_overlay()
