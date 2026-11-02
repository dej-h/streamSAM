"""Check latest-frame delivery and disconnects with the real OpenCV decoder."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from sam2.utils.rtsp_stream import RtspDisconnected, RtspFrameReader


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    reference_capture = cv2.VideoCapture(str(arguments.video))
    if not reference_capture.isOpened():
        raise RuntimeError(f"could not open test video: {arguments.video}")
    frame_count = 0
    last_frame: np.ndarray | None = None
    try:
        while True:
            decoded, bgr = reference_capture.read()
            if not decoded:
                break
            frame_count += 1
            last_frame = bgr
    finally:
        reference_capture.release()
    if frame_count < 5 or last_frame is None:
        raise RuntimeError("test video needs at least five decoded frames")

    def open_fixture(_url: str, _backend: int, _params: list[int]) -> cv2.VideoCapture:
        return cv2.VideoCapture(str(arguments.video))

    with RtspFrameReader(
        "rtsp://fixture/live", capacity=2, capture_factory=open_fixture
    ) as reader:
        deadline = time.monotonic() + 20.0
        while not reader.stats().disconnected:
            if time.monotonic() >= deadline:
                raise TimeoutError("fixture decoder did not finish")
            time.sleep(0.01)
        result = reader.read(timeout_seconds=1.0)
        stats = reader.stats()
        if result.decoded_sequence != frame_count - 1:
            raise AssertionError("reader did not return the latest decoded frame")
        if not np.array_equal(result.bgr, last_frame):
            raise AssertionError("reader changed the decoded frame")
        if stats.maximum_depth > 2 or stats.dropped_frames != frame_count - 1:
            raise AssertionError("reader did not bound and account for its queue")
        try:
            reader.read(timeout_seconds=1.0)
        except RtspDisconnected:
            pass
        else:
            raise AssertionError("reader hid the end of the camera epoch")

    def fail_open(url: str, _backend: int, _params: list[int]) -> cv2.VideoCapture:
        raise RuntimeError(f"decoder failed to open {url}")

    with RtspFrameReader(
        "rtsp://camera:secret@fixture/live", capture_factory=fail_open
    ) as reader:
        try:
            reader.read(timeout_seconds=1.0)
        except RtspDisconnected as error:
            if "secret" in str(error) or error.__cause__ is not None:
                raise AssertionError("RTSP error exposed its camera URL") from error
        else:
            raise AssertionError("reader hid its decoder error")
    print(
        json.dumps(
            {
                "status": "passed",
                "decoded_frames": frame_count,
                "dropped_frames": stats.dropped_frames,
                "maximum_depth": stats.maximum_depth,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
