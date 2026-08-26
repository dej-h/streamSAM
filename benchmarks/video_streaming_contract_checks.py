from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

from benchmarks.benchmark_video_streaming import PositiveInteger
from sam2.utils.video_stream import create_video_frame_source


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Stage 2 frame-source ordering, seeking, and bounds."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--image-size", type=PositiveInteger, default=256)
    parser.add_argument("--capacity", type=PositiveInteger, default=4)
    return parser.parse_args(argv)


def main() -> None:
    arguments = parse_arguments()
    eager = create_video_frame_source(
        video_path=str(arguments.video),
        image_size=arguments.image_size,
        offload_video_to_cpu=True,
        loading_mode="eager",
        buffer_capacity=arguments.capacity,
        compute_device=torch.device("cpu"),
    )
    lazy = create_video_frame_source(
        video_path=str(arguments.video),
        image_size=arguments.image_size,
        offload_video_to_cpu=True,
        loading_mode="lazy",
        buffer_capacity=arguments.capacity,
        compute_device=torch.device("cpu"),
    )
    try:
        if eager.metadata.frame_count != lazy.metadata.frame_count:
            raise AssertionError("frame counts differ")
        if eager.metadata.video_height != lazy.metadata.video_height:
            raise AssertionError("video heights differ")
        if eager.metadata.video_width != lazy.metadata.video_width:
            raise AssertionError("video widths differ")

        requested_indices = (0, 1, 2, 7, 3, 4)
        for index in requested_indices:
            if not torch.equal(eager[index], lazy[index]):
                maximum_error = torch.max(torch.abs(eager[index] - lazy[index])).item()
                raise AssertionError(
                    f"frame {index} differs; maximum error={maximum_error}"
                )
    finally:
        eager.close()
        lazy.close()

    stats = lazy.stats()
    if stats.maximum_depth > arguments.capacity:
        raise AssertionError("lazy frame queue exceeded its capacity")
    print(
        json.dumps(
            {
                "status": "passed",
                "requested_indices": requested_indices,
                "metadata": {
                    "frame_count": lazy.metadata.frame_count,
                    "video_height": lazy.metadata.video_height,
                    "video_width": lazy.metadata.video_width,
                    "image_size": lazy.metadata.image_size,
                },
                "stats": {
                    "capacity": stats.capacity,
                    "maximum_depth": stats.maximum_depth,
                    "decoded_frames": stats.decoded_frames,
                    "seek_count": stats.seek_count,
                    "wait_count": stats.wait_count,
                    "wait_seconds": stats.wait_seconds,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
