"""Compare live frame delivery with the existing finite-video predictor path."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2_video_predictor
from sam2.live_video_session import LiveVideoSession
from sam2.utils.rtsp_stream import CapturedFrame
from sam2.utils.video_stream import (
    LiveResolutionChanged,
    LiveVideoFrameSource,
    SequentialMp4FrameSource,
)


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--config", default="edgetam.yaml")
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/edgetam.pt")
    )
    parser.add_argument("--frame-count", type=int, default=48)
    parser.add_argument("--memory-stride", type=int, default=1)
    parser.add_argument("--point", type=float, nargs=2, default=(480.0, 270.0))
    parsed = parser.parse_args(argv)
    if parsed.frame_count < 20:
        parser.error("use at least 20 frames to exercise temporal-state pruning")
    if parsed.memory_stride <= 0:
        parser.error("--memory-stride must be positive")
    return parsed


def read_frames(video_path: Path, count: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open test video: {video_path}")
    frames: list[np.ndarray] = []
    try:
        for _ in range(count):
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f"video contains fewer than {count} frames")
            frames.append(bgr)
    finally:
        capture.release()
    return frames


def verify_normalization(frames: list[np.ndarray], video_path: Path) -> None:
    file_source = SequentialMp4FrameSource(
        str(video_path),
        image_size=256,
        capacity=2,
        img_mean=(0.485, 0.456, 0.406),
        img_std=(0.229, 0.224, 0.225),
    )
    live_source = LiveVideoFrameSource(frames[0], image_size=256)
    try:
        for index, frame in enumerate(frames[:5]):
            if index:
                live_source.append_bgr_frame(frame)
            if not torch.equal(file_source[index], live_source[index]):
                raise AssertionError(f"frame {index} normalization differs")
        height, width = frames[0].shape[:2]
        try:
            live_source.append_bgr_frame(np.zeros((height + 1, width, 3), np.uint8))
        except LiveResolutionChanged:
            pass
        else:
            raise AssertionError("live source accepted a changed camera resolution")
    finally:
        file_source.close()
        live_source.close()


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("live predictor parity requires CUDA")
    frames = read_frames(arguments.video, arguments.frame_count)
    verify_normalization(frames, arguments.video)
    predictor = build_sam2_video_predictor(
        arguments.config,
        str(arguments.checkpoint),
        device="cuda",
        apply_postprocessing=False,
    )
    predictor.memory_temporal_stride_for_eval = arguments.memory_stride
    finite_masks: list[np.ndarray] = []
    with torch.inference_mode(), torch.autocast("cuda", enabled=False):
        state = predictor.init_state(
            video_path=str(arguments.video),
            frame_loading="lazy",
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
            enable_gpu_frame_staging=False,
        )
        try:
            predictor.add_new_points_or_box(
                state,
                frame_idx=0,
                obj_id=1,
                points=np.array([arguments.point], dtype=np.float32),
                labels=np.array([1], dtype=np.int32),
            )
            predictions = predictor.propagate_in_video(
                state, max_frame_num_to_track=arguments.frame_count - 1
            )
            try:
                for index, object_ids, logits in predictions:
                    if index != len(finite_masks) or object_ids != [1]:
                        raise AssertionError("finite predictor output order changed")
                    finite_masks.append(
                        (logits > 0).squeeze(1).to("cpu").numpy().copy()
                    )
            finally:
                predictions.close()
        finally:
            predictor.close_video_source(state)
            predictor.reset_state(state)

        first = CapturedFrame(frames[0], 0, time.monotonic(), None)
        with LiveVideoSession(predictor, first) as session:
            live_results = [session.prompt_point(*arguments.point)]
            maximum_retained_outputs = session.retained_output_count
            for index, bgr in enumerate(frames[1:], start=1):
                live_results.append(
                    session.track(CapturedFrame(bgr, index, time.monotonic(), None))
                )
                maximum_retained_outputs = max(
                    maximum_retained_outputs, session.retained_output_count
                )
            retained_output_count = session.retained_output_count

    if len(finite_masks) != arguments.frame_count:
        raise AssertionError("finite predictor returned the wrong frame count")
    for index, (reference, result) in enumerate(zip(finite_masks, live_results)):
        if result.processed_index != index or result.decoded_sequence != index:
            raise AssertionError(f"live frame {index} was out of order")
        if not np.array_equal(reference, result.masks):
            raise AssertionError(
                f"live frame {index} differs from finite predictor: "
                f"{np.count_nonzero(reference != result.masks)} pixels"
            )
    if retained_output_count >= arguments.frame_count:
        raise AssertionError("live predictor retained every processed frame")
    expected_bound = 1 + max(
        1 + max(0, predictor.num_maskmem - 2) * arguments.memory_stride,
        predictor.max_obj_ptrs_in_encoder - 1,
    )
    if maximum_retained_outputs > expected_bound:
        raise AssertionError("live output retention exceeded the expected bound")
    print(
        json.dumps(
            {
                "status": "passed",
                "frames": arguments.frame_count,
                "exact_masks": len(live_results),
                "retained_outputs": retained_output_count,
                "maximum_retained_outputs": maximum_retained_outputs,
                "memory_stride": arguments.memory_stride,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
