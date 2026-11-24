"""Compare a SAM 2-family predictor with a separately run reference checkout.

Run this file as a script with either checkout first on ``PYTHONPATH``. Each run
uses the same source frames, checkpoint, point prompt, and dtype, and writes
binary masks for an independent comparison via ``--compare-to``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence, TypedDict, cast

import numpy as np
from numpy.typing import NDArray
import torch

import sam2
from sam2.build_sam import build_sam2_video_predictor


DTypeName = Literal["float32", "bfloat16"]
FrameLoadingMode = Literal["eager", "lazy"]
BinaryMasks = NDArray[np.uint8]


class MaskComparison(TypedDict):
    exact: bool
    different_pixels: int
    mean_iou: float
    minimum_iou: float
    minimum_iou_frame: int


@dataclass(frozen=True)
class Arguments:
    config: str
    checkpoint: Path
    frames: Path
    output: Path
    point_x: float
    point_y: float
    frame_count: int
    dtype: DTypeName
    frame_loading: FrameLoadingMode
    compare_to: Path | None
    require_exact: bool


def parse_arguments(argv: Sequence[str] | None = None) -> Arguments:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--frames", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--point", required=True, type=float, nargs=2, metavar=("X", "Y")
    )
    parser.add_argument("--frame-count", type=int, default=24)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--frame-loading", choices=("eager", "lazy"), default="eager")
    parser.add_argument("--compare-to", type=Path)
    parser.add_argument("--require-exact", action="store_true")
    parsed = parser.parse_args(argv)
    if not 0 < parsed.frame_count <= 64:
        parser.error("--frame-count must be between 1 and 64; this check retains masks")
    if parsed.require_exact and parsed.compare_to is None:
        parser.error("--require-exact requires --compare-to")
    for path, label in ((parsed.checkpoint, "checkpoint"), (parsed.frames, "frames")):
        if not path.exists():
            parser.error(f"{label} does not exist: {path}")
    if parsed.compare_to is not None and not parsed.compare_to.is_file():
        parser.error(f"comparison file does not exist: {parsed.compare_to}")
    return Arguments(
        config=str(parsed.config),
        checkpoint=parsed.checkpoint,
        frames=parsed.frames,
        output=parsed.output,
        point_x=float(parsed.point[0]),
        point_y=float(parsed.point[1]),
        frame_count=int(parsed.frame_count),
        dtype=cast(DTypeName, parsed.dtype),
        frame_loading=cast(FrameLoadingMode, parsed.frame_loading),
        compare_to=parsed.compare_to,
        require_exact=bool(parsed.require_exact),
    )


def collect_masks(arguments: Arguments) -> BinaryMasks:
    if not torch.cuda.is_available():
        raise RuntimeError("this model compatibility check requires CUDA")
    predictor = build_sam2_video_predictor(
        arguments.config,
        str(arguments.checkpoint),
        device="cuda",
        apply_postprocessing=False,
    )
    frame_loading_options = (
        {"frame_loading": "lazy"} if arguments.frame_loading == "lazy" else {}
    )
    masks: list[BinaryMasks] = []
    with (
        torch.inference_mode(),
        torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=arguments.dtype == "bfloat16"
        ),
    ):
        state = predictor.init_state(
            video_path=str(arguments.frames),
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
            **frame_loading_options,
        )
        try:
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,
                obj_id=1,
                points=np.array(
                    [[arguments.point_x, arguments.point_y]], dtype=np.float32
                ),
                labels=np.array([1], dtype=np.int32),
            )
            predictions = predictor.propagate_in_video(state)
            try:
                for frame_idx, object_ids, logits in predictions:
                    if frame_idx != len(masks) or object_ids != [1]:
                        raise RuntimeError(
                            "predictor returned out-of-order or unexpected objects"
                        )
                    masks.append(
                        (logits > 0).to(device="cpu", dtype=torch.uint8).numpy()
                    )
                    if len(masks) == arguments.frame_count:
                        break
            finally:
                predictions.close()
        finally:
            predictor.reset_state(state)
            if hasattr(predictor, "close_video_source"):
                predictor.close_video_source(state)
    if len(masks) != arguments.frame_count:
        raise RuntimeError(
            f"expected {arguments.frame_count} frames, received {len(masks)}"
        )
    return np.stack(masks)


def compare_masks(candidate: BinaryMasks, reference_path: Path) -> MaskComparison:
    with np.load(reference_path) as reference_file:
        reference: BinaryMasks = reference_file["masks"]
    if candidate.shape != reference.shape:
        raise ValueError(
            f"mask shapes differ: {candidate.shape} versus {reference.shape}"
        )
    axes = tuple(range(1, candidate.ndim))
    candidate_bool = candidate.astype(bool)
    reference_bool = reference.astype(bool)
    intersections = np.logical_and(candidate_bool, reference_bool).sum(axis=axes)
    unions = np.logical_or(candidate_bool, reference_bool).sum(axis=axes)
    ious = np.divide(
        intersections,
        unions,
        out=np.ones(len(candidate), dtype=np.float64),
        where=unions > 0,
    )
    return {
        "exact": bool(np.array_equal(candidate, reference)),
        "different_pixels": int(np.count_nonzero(candidate != reference)),
        "mean_iou": float(ious.mean()),
        "minimum_iou": float(ious.min()),
        "minimum_iou_frame": int(ious.argmin()),
    }


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    masks = collect_masks(arguments)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arguments.output, masks=masks)
    report: dict[str, object] = {
        "sam2_package": str(Path(sam2.__file__).resolve()),
        "config": arguments.config,
        "checkpoint": str(arguments.checkpoint.resolve()),
        "frames": str(arguments.frames.resolve()),
        "frame_count": len(masks),
        "dtype": arguments.dtype,
        "frame_loading": arguments.frame_loading,
        "mask_shape": list(masks.shape),
        "output": str(arguments.output.resolve()),
    }
    if arguments.compare_to is not None:
        comparison = compare_masks(masks, arguments.compare_to)
        report["comparison"] = comparison
        if arguments.require_exact and not comparison["exact"]:
            raise AssertionError(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
