from __future__ import annotations

from collections import defaultdict
from unittest.mock import patch

import cv2
import numpy as np
from numpy.typing import NDArray
import torch

from sam2.utils.video_output_writer import BoundedVideoMaskWriter
from sam2.utils.video_stream import SequentialMp4FrameSource


def main() -> None:
    frame_count, width, height = 12, 8, 6
    originals = [
        np.full((height, width, 3), (index * 10, 60, 120), dtype=np.uint8)
        for index in range(frame_count)
    ]
    reads: list[list[int]] = []
    annotated: dict[str, list[NDArray[np.uint8]]] = defaultdict(list)

    class Capture:
        def __init__(self, path: str) -> None:
            self.index = 0
            self.reads: list[int] = []
            reads.append(self.reads)

        def isOpened(self) -> bool:
            return True

        def get(self, property_id: int) -> float:
            return {
                cv2.CAP_PROP_FRAME_COUNT: frame_count,
                cv2.CAP_PROP_FRAME_WIDTH: width,
                cv2.CAP_PROP_FRAME_HEIGHT: height,
                cv2.CAP_PROP_FPS: 30,
            }[property_id]

        def set(self, property_id: int, value: int) -> bool:
            assert property_id == cv2.CAP_PROP_POS_FRAMES
            self.index = value
            return True

        def read(self) -> tuple[bool, NDArray[np.uint8]]:
            index = self.index
            self.index += 1
            self.reads.append(index)
            return True, originals[index]

        def release(self) -> None:
            pass

    class Writer:
        def __init__(
            self, path: str, codec: int, fps: float, frame_size: tuple[int, int]
        ) -> None:
            self.frames = annotated[path]

        def isOpened(self) -> bool:
            return True

        def write(self, frame: NDArray[np.uint8]) -> None:
            self.frames.append(frame.copy())

        def release(self) -> None:
            pass

    def mask(index: int) -> NDArray[np.bool_]:
        result = np.zeros((1, height, width), dtype=np.bool_)
        result[0, index % height, index % width] = True
        return result

    with patch.object(cv2, "VideoCapture", Capture), patch.object(
        cv2, "VideoWriter", Writer
    ):
        source = SequentialMp4FrameSource(
            video_path="input.mp4",
            image_size=4,
            capacity=1,
            img_mean=(0.0, 0.0, 0.0),
            img_std=(1.0, 1.0, 1.0),
            retain_original_frames=True,
        )
        try:
            torch.testing.assert_close(source[0][2, 0, 0], torch.tensor(0.0))
            staged_frame = source[1]
            assert source.original_frame(0, consume=False) is originals[0]

            source.rewind_original_frames()
            assert source.original_frame(0) is originals[0]
            torch.testing.assert_close(staged_frame[2, 0, 0], torch.tensor(10 / 255))
            assert source.original_frame(1) is originals[1]
            source[2]  # The stager can advance before the writer aborts.

            for output_path, input_path in (("shared", None), ("standalone", "input.mp4")):
                source.rewind_original_frames()
                with BoundedVideoMaskWriter(
                    input_path=input_path,
                    output_path=output_path,
                    fps=30,
                    frame_size=(width, height),
                    expected_frame_count=frame_count,
                    expected_object_ids=(0,),
                    capacity=1,
                ) as writer:
                    for index in range(frame_count):
                        if input_path is not None:
                            writer.submit(index, (0,), mask(index))
                            continue
                        # Render a cached prediction for frame zero. Read remaining
                        # model inputs ahead of the original-frame handoff.
                        if index:
                            tensor = source[index]
                            torch.testing.assert_close(
                                tensor[2, 0, 0], torch.tensor(index * 10 / 255)
                            )
                        frame_bgr = source.original_frame(index)
                        assert frame_bgr is originals[index]
                        writer.submit(index, (0,), mask(index), frame_bgr=frame_bgr)
                assert writer.stats().written_frames == frame_count
        finally:
            source.close()

    assert len(reads) == 3, reads
    assert reads[0] == [], reads
    assert reads[1][-frame_count:] == list(range(frame_count)), reads
    assert reads[2] == list(range(frame_count)), reads
    np.testing.assert_array_equal(annotated["shared"], annotated["standalone"])
    assert source.stats().maximum_depth <= 1
    print("Shared original-frame order, retry, writer parity, and decode bounds passed.")


if __name__ == "__main__":
    main()
