"""Track one prompted object from an RTSP camera with streamSAM."""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray
import torch

from sam2.build_sam import build_sam2_video_predictor
from sam2.live_video_session import LiveMaskResult, LiveVideoSession
from sam2.utils.mask_prompts import sample_mask_interior_points
from sam2.utils.rtsp_stream import RtspDisconnected, RtspFrameReader, RtspReaderStats
from sam2.utils.video_stream import LiveFrameDeviceChanged, LiveResolutionChanged


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="RTSP camera URL")
    parser.add_argument(
        "--point", type=float, nargs=2, required=True, metavar=("X", "Y")
    )
    parser.add_argument("--config", default="edgetam.yaml")
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/edgetam.pt")
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--mask-directory", type=Path)
    parser.add_argument("--queue-capacity", type=int, default=4)
    parser.add_argument(
        "--decoder", choices=("auto", "cuda", "cpu"), default="auto",
        help="Prefer CUDA automatically, require CUDA, or use software decoding",
    )
    parser.add_argument(
        "--open-timeout-seconds",
        type=float,
        help="Override the connection-open timeout in seconds",
    )
    parser.add_argument(
        "--read-timeout-seconds",
        type=float,
        help="Override the decoder-read timeout in seconds",
    )
    parser.add_argument("--reconnect-attempts", type=int, default=0)
    parser.add_argument(
        "--reconnect-points",
        type=int,
        default=64,
        help="Maximum interior points used to refine the last mask after reconnect",
    )
    arguments = parser.parse_args(argv)
    if arguments.max_frames is not None and arguments.max_frames <= 0:
        parser.error("--max-frames must be positive")
    if arguments.reconnect_attempts < 0:
        parser.error("--reconnect-attempts must not be negative")
    if arguments.reconnect_points <= 0:
        parser.error("--reconnect-points must be positive")
    return arguments


def report_frame(
    result: LiveMaskResult[NDArray[np.bool_]],
    *,
    epoch: int,
    reader_stats: RtspReaderStats,
    retained_outputs: int,
    mask_directory: Path | None,
) -> None:
    if mask_directory is not None:
        epoch_directory = mask_directory / f"epoch-{epoch:03d}"
        epoch_directory.mkdir(parents=True, exist_ok=True)
        mask_path = epoch_directory / f"frame-{result.processed_index:08d}.png"
        mask_image = result.masks[0].astype(np.uint8) * 255
        if not cv2.imwrite(str(mask_path), mask_image):
            raise RuntimeError("could not write live mask image")
    print(
        json.dumps(
            {
                "epoch": epoch,
                "processed_index": result.processed_index,
                "decoded_sequence": result.decoded_sequence,
                "presentation_time_seconds": result.presentation_time_seconds,
                "receive_to_mask_ms": round(result.receive_to_mask_seconds * 1000, 1),
                "foreground_pixels": int(result.masks[0].sum()),
                "decoder_backend": reader_stats.decoder_backend,
                "decoder_fallback_reason": reader_stats.fallback_reason,
                "reader_decoded_frames": reader_stats.decoded_frames,
                "reader_dropped_frames": reader_stats.dropped_frames,
                "retained_outputs": retained_outputs,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("the RTSP example requires a CUDA-capable GPU")
    with redirect_stdout(sys.stderr):
        predictor = build_sam2_video_predictor(
            arguments.config, str(arguments.checkpoint), device="cuda"
        )
    processed_total = 0
    epoch = 0
    last_mask: NDArray[np.bool_] | None = None
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        while arguments.max_frames is None or processed_total < arguments.max_frames:
            try:
                with RtspFrameReader(
                    arguments.url,
                    capacity=arguments.queue_capacity,
                    decoder=arguments.decoder,
                    inference_device=predictor.device,
                    open_timeout_seconds=arguments.open_timeout_seconds,
                    read_timeout_seconds=arguments.read_timeout_seconds,
                ) as reader:
                    first_frame = reader.read()
                    with LiveVideoSession(predictor, first_frame) as session:
                        if processed_total == 0:
                            result = session.prompt_point(*arguments.point)
                        else:
                            if last_mask is None:
                                raise RuntimeError(
                                    "cannot reconnect without a foreground mask"
                                )
                            height, width = first_frame.dimensions
                            mask = (
                                last_mask
                                if last_mask.shape == (height, width)
                                else cv2.resize(
                                    last_mask.astype(np.uint8),
                                    (width, height),
                                    interpolation=cv2.INTER_NEAREST,
                                ).astype(bool)
                            )
                            points = sample_mask_interior_points(
                                mask, max_points=arguments.reconnect_points
                            )
                            if not len(points):
                                raise RuntimeError(
                                    "last foreground mask is empty at the new resolution"
                                )
                            result = session.prompt_points(points, mask=mask)
                        while True:
                            if result.masks[0].any():
                                last_mask = result.masks[0]
                            report_frame(
                                result,
                                epoch=epoch,
                                reader_stats=reader.stats(),
                                retained_outputs=session.retained_output_count,
                                mask_directory=arguments.mask_directory,
                            )
                            processed_total += 1
                            if (
                                arguments.max_frames is not None
                                and processed_total >= arguments.max_frames
                            ):
                                break
                            frame = reader.read()
                            result = session.track(frame)
                return
            except (RtspDisconnected, LiveFrameDeviceChanged, LiveResolutionChanged) as error:
                if epoch >= arguments.reconnect_attempts:
                    raise
                epoch += 1
                delay_seconds = 2 ** min(epoch - 1, 3)
                print(
                    json.dumps(
                        {
                            "event": "new_epoch",
                            "reason": type(error).__name__,
                            "epoch": epoch,
                            "retry_after_seconds": delay_seconds,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                time.sleep(delay_seconds)


if __name__ == "__main__":
    main()
