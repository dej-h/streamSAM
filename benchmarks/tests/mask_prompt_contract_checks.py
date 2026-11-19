"""Check interior prompt coverage for holes, disconnected regions, and thin masks."""

from __future__ import annotations

import json

import cv2
import numpy as np
from numpy.typing import NDArray

from sam2.utils.mask_prompts import sample_mask_interior_points


def verify_points(mask: NDArray[np.bool_], count: int) -> NDArray[np.float32]:
    points = sample_mask_interior_points(mask, max_points=count)
    if points.shape != (count, 2) or points.dtype != np.float32:
        raise AssertionError("prompt point count, shape, or dtype differs")
    pixels = points.astype(np.intp)
    if not mask[pixels[:, 1], pixels[:, 0]].all():
        raise AssertionError("a positive prompt fell outside the segmentation")
    if len(np.unique(pixels, axis=0)) != count:
        raise AssertionError("interior prompts contain duplicates")
    if not np.array_equal(points, sample_mask_interior_points(mask, max_points=count)):
        raise AssertionError("interior prompts are not deterministic")
    _, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    distances = cv2.distanceTransform(
        np.pad(mask.astype(np.uint8), 1), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )[1:-1, 1:-1]
    for component in np.unique(labels[pixels[:, 1], pixels[:, 0]]):
        selected = pixels[labels[pixels[:, 1], pixels[:, 0]] == component]
        deepest = distances[labels == component].max()
        if (distances[selected[:, 1], selected[:, 0]] < deepest * 0.5).any():
            raise AssertionError("a prompt is outside its component's interior")
    return points


def main() -> None:
    ring = np.zeros((80, 80), dtype=bool)
    ring[8:72, 8:72] = True
    ring[24:56, 24:56] = False
    verify_points(ring, 64)

    fragments = np.zeros((110, 120), dtype=bool)
    fragments[10:75, 5:70] = True
    fragments[5:100, 95] = True
    points = verify_points(fragments, 64)
    if not (points[:, 0] == 95).any() or not (points[:, 0] < 70).any():
        raise AssertionError("thick and one-pixel-wide fragments need coverage")

    concave = np.zeros((80, 80), dtype=bool)
    concave[5:75, 5:20] = True
    concave[5:20, 5:70] = True
    concave[60:75, 5:70] = True
    verify_points(concave, 64)
    verify_points(np.ones((20, 40), dtype=bool), 64)

    crowded = np.zeros((30, 80), dtype=bool)
    for start, side in ((2, 3), (15, 5), (30, 7), (50, 9)):
        crowded[2 : 2 + side, start : start + side] = True
    points = verify_points(crowded, 3)
    if (points[:, 0] < 15).any():
        raise AssertionError("a tiny component displaced a larger component")
    if not all(
        ((points[:, 0] >= lo) & (points[:, 0] < hi)).any()
        for lo, hi in ((15, 20), (30, 37), (50, 59))
    ):
        raise AssertionError("the largest components were not all represented")

    empty = sample_mask_interior_points(np.zeros((8, 8), dtype=bool))
    if empty.shape != (0, 2):
        raise AssertionError("an empty mask produced prompts")
    single = np.zeros((8, 8), dtype=bool)
    single[3, 4] = True
    if not np.array_equal(sample_mask_interior_points(single), [[4, 3]]):
        raise AssertionError("a one-pixel mask was lost or duplicated")
    for mask, count in ((ring, 0), (ring[..., None], 2), (ring.astype(np.uint8), 2)):
        try:
            sample_mask_interior_points(mask, max_points=count)
        except ValueError:
            pass
        else:
            raise AssertionError("an invalid mask or point budget was accepted")
    print(
        json.dumps(
            {
                "status": "passed",
                "cases": [
                    "ring",
                    "thin_fragments",
                    "concave",
                    "image_edges",
                    "component_budget",
                    "empty",
                    "single_pixel",
                    "input_validation",
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
