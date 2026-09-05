from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from sam2.utils.gpu_frame_stager import GpuFrameLease, GpuFrameStager
from sam2.utils.video_stream import create_video_frame_source


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check bounded CUDA staging against eager decoded tensors."
    )
    parser.add_argument("--video", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GPU frame staging contract checks")

    device = torch.device("cuda")
    eager_source = create_video_frame_source(
        video_path=str(arguments.video),
        image_size=256,
        offload_video_to_cpu=True,
        loading_mode="eager",
        buffer_capacity=4,
        compute_device=device,
    )
    lazy_source = create_video_frame_source(
        video_path=str(arguments.video),
        image_size=256,
        offload_video_to_cpu=True,
        loading_mode="lazy",
        buffer_capacity=4,
        compute_device=device,
    )
    stager = GpuFrameStager(
        frame_source=lazy_source,
        device=device,
        slot_count=2,
    )

    frame_count = min(len(lazy_source), 32)
    active_lease: GpuFrameLease | None = None
    try:
        for frame_idx in range(frame_count):
            if active_lease is not None:
                active_lease.release()
            active_lease = stager.acquire(frame_idx)
            stager.prefetch_adjacent(frame_idx)
            staged_frame = active_lease.tensor.cpu()
            if not torch.equal(staged_frame, eager_source[frame_idx]):
                raise AssertionError(
                    f"staged frame {frame_idx} differs from eager decoding"
                )
    finally:
        if active_lease is not None:
            active_lease.release()
        torch.cuda.synchronize(device)
        stager.close()
        lazy_source.close()
        eager_source.close()

    stats = stager.stats()
    if stats.maximum_ready_depth > stats.capacity:
        raise AssertionError("GPU staging exceeded its slot capacity")
    if stats.staged_frames < frame_count:
        raise AssertionError("GPU staging did not process every requested frame")
    if stats.transfer_count < stats.staged_frames:
        raise AssertionError("GPU transfer timings are incomplete")

    print(
        json.dumps(
            {
                "status": "passed",
                "frames_compared": frame_count,
                "stats": asdict(stats),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
