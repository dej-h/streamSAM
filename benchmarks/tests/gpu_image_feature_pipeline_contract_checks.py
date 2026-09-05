from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from sam2.utils.gpu_frame_stager import GpuFrameLease, GpuFrameStager
from sam2.utils.gpu_image_feature_pipeline import (
    EncodedImageFeature,
    GpuImageFeaturePipeline,
)
from sam2.utils.video_stream import create_video_frame_source


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check bounded CUDA image feature production against decoded frames."
    )
    parser.add_argument("--video", type=Path, required=True)
    arguments = parser.parse_args()
    video_path = Path(arguments.video)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU image feature contract requires CUDA")
    device = torch.device("cuda")
    source = create_video_frame_source(
        video_path=str(video_path),
        image_size=256,
        offload_video_to_cpu=True,
        loading_mode="lazy",
        buffer_capacity=4,
        compute_device=device,
    )
    stager = GpuFrameStager(frame_source=source, device=device, slot_count=2)

    def produce(frame_idx: int) -> EncodedImageFeature:
        lease: GpuFrameLease | None = stager.acquire(frame_idx)
        try:
            stager.prefetch_adjacent(frame_idx)
            image = lease.tensor.unsqueeze(0)
            return EncodedImageFeature(
                frame_idx=frame_idx,
                image=image,
                backbone_out={
                    "backbone_fpn": [image.square()],
                    "vision_pos_enc": [image + 1.0],
                },
                frame_lease=lease,
            )
        except BaseException:
            lease.release()
            raise

    pipeline = GpuImageFeaturePipeline(
        producer=produce,
        frame_count=len(source),
        device=device,
        autocast_enabled=False,
        autocast_dtype=torch.bfloat16,
    )
    compared = 0
    try:
        pipeline.prefetch(0)
        for frame_idx in range(min(len(source), 32)):
            feature = pipeline.acquire(frame_idx)
            pipeline.prefetch(frame_idx + 1)
            expected = source[frame_idx].to(device)
            torch.testing.assert_close(
                feature.backbone_out["backbone_fpn"][0],
                expected.unsqueeze(0).square(),
                rtol=0.0,
                atol=0.0,
            )
            compared += 1
            if feature.frame_lease is not None:
                feature.frame_lease.release()
    finally:
        pipeline.close()
        stager.close()
        source.close()
    stats = pipeline.stats()
    if stats.maximum_pending_depth > 1 or stats.maximum_ready_depth > 1:
        raise AssertionError("image feature pipeline exceeded its bounded capacity")
    print(
        json.dumps(
            {
                "status": "passed",
                "frames_compared": compared,
                "stats": asdict(stats),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
