"""Compare mask-derived reconnect prompts with a known translated object."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
from numpy.typing import NDArray
import torch

from sam2.build_sam import build_sam2_video_predictor
from sam2.live_frame import CapturedFrame
from sam2.live_video_session import LiveVideoSession
from sam2.utils.mask_prompts import sample_mask_interior_points


def mask_iou(first: NDArray[np.bool_], second: NDArray[np.bool_]) -> float:
    union = int(np.count_nonzero(first | second))
    return int(np.count_nonzero(first & second)) / union if union else 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=Path("examples/01_dog.mp4"))
    parser.add_argument("--point", type=float, nargs=2, default=(600.0, 440.0))
    arguments = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("live reconnect checks require CUDA")
    capture = cv2.VideoCapture(str(arguments.video))
    try:
        decoded, original = capture.read()
    finally:
        capture.release()
    if not decoded:
        raise RuntimeError("could not decode the reconnect fixture")
    height, width = original.shape[:2]
    predictor = build_sam2_video_predictor(
        "edgetam.yaml", "checkpoints/edgetam.pt", device="cuda"
    )

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        first = CapturedFrame(original, 0, time.monotonic(), None)
        with LiveVideoSession(predictor, first) as session:
            initial_mask = session.prompt_point(*arguments.point).masks[0]
            for index in range(1, 36):
                transform = np.array([[1, 0, index * 10], [0, 1, 0]], dtype=np.float32)
                moved = cv2.warpAffine(original, transform, (width, height))
                result = session.track(
                    CapturedFrame(moved, index, time.monotonic(), None)
                )
            last_mask = result.masks[0]
        expected = cv2.warpAffine(
            initial_mask.astype(np.uint8),
            transform,
            (width, height),
            flags=cv2.INTER_NEAREST,
        ).astype(bool)
        points = sample_mask_interior_points(last_mask)
        if len(points) != 64:
            raise AssertionError("the reconnect prompt did not contain 64 points")
        fresh = CapturedFrame(moved, 0, time.monotonic(), None)
        with LiveVideoSession(predictor, fresh) as session:
            reconnected = session.prompt_points(points, mask=last_mask).masks[0]
        with LiveVideoSession(predictor, fresh) as session:
            stale = session.prompt_point(*arguments.point).masks[0]
    scores = {
        "tracked_iou": mask_iou(last_mask, expected),
        "reconnected_iou": mask_iou(reconnected, expected),
        "original_click_iou": mask_iou(stale, expected),
        "prompt_points": len(points),
    }
    print(json.dumps(scores, sort_keys=True))
    if scores["tracked_iou"] < 0.85 or scores["reconnected_iou"] < 0.85:
        raise AssertionError("tracking or reconnect lost the translated object")
    if scores["reconnected_iou"] <= scores["original_click_iou"]:
        raise AssertionError("mask-derived points did not improve reconnect selection")


if __name__ == "__main__":
    main()
