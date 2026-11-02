"""Forward-only SAM 2 tracking on frames supplied by a live source."""

from __future__ import annotations

import time
from dataclasses import dataclass
from types import TracebackType
from typing import TypedDict, cast

import numpy as np
from numpy.typing import NDArray
from sam2.sam2_video_predictor import FrameOutput, SAM2VideoPredictor
from sam2.utils.rtsp_stream import CapturedFrame
from sam2.utils.video_stream import LiveVideoFrameSource


class _StoredOutputs(TypedDict):
    cond_frame_outputs: dict[int, FrameOutput]
    non_cond_frame_outputs: dict[int, FrameOutput]


class _LiveInferenceState(TypedDict):
    num_frames: int
    output_dict: _StoredOutputs
    output_dict_per_obj: dict[int, _StoredOutputs]
    frames_already_tracked: dict[int, object]


@dataclass(frozen=True)
class LiveMaskResult:
    processed_index: int
    decoded_sequence: int
    presentation_time_seconds: float | None
    receive_to_mask_seconds: float
    object_ids: tuple[int, ...]
    masks: NDArray[np.bool_]


class LiveVideoSession:
    """Feed arriving frames through the existing SAM 2 predict/commit core.

    A session tracks one prompted object forward. The camera reader owns decoded
    frame drops; this class numbers only frames accepted for inference. Start a
    new session after a disconnect or resolution change.
    """

    def __init__(
        self,
        predictor: SAM2VideoPredictor,
        first_frame: CapturedFrame,
        *,
        offload_state_to_cpu: bool = True,
        enable_gpu_frame_staging: bool = True,
    ) -> None:
        self._predictor = predictor
        self._frame_source = LiveVideoFrameSource(
            first_frame.bgr, image_size=predictor.image_size
        )
        # The legacy predictor still uses a heterogeneous state dictionary.
        # This is the one boundary at which the fields used by the live wrapper
        # are described precisely; model inference stays in the predictor.
        try:
            state = predictor.init_live_state(
                self._frame_source,
                offload_state_to_cpu=offload_state_to_cpu,
                enable_gpu_frame_staging=enable_gpu_frame_staging,
            )
        except BaseException:
            self._frame_source.close()
            raise
        self._state = cast(_LiveInferenceState, state)
        self._current_frame = first_frame
        self._prompted = False
        self._closed = False
        self._failed = False

    def prompt_point(self, x: float, y: float, *, obj_id: int = 1) -> LiveMaskResult:
        """Prompt the first camera frame in its original pixel coordinates."""
        self._require_unprompted()
        try:
            self._predictor.add_new_points_or_box(
                inference_state=self._state,
                frame_idx=0,
                obj_id=obj_id,
                points=np.array([[x, y]], dtype=np.float32),
                labels=np.array([1], dtype=np.int32),
            )
        except BaseException:
            self._failed = True
            raise
        return self._commit_current()

    def prompt_box(
        self, box: tuple[float, float, float, float], *, obj_id: int = 1
    ) -> LiveMaskResult:
        """Prompt the first frame with an (x0, y0, x1, y1) box."""
        self._require_unprompted()
        try:
            self._predictor.add_new_points_or_box(
                inference_state=self._state,
                frame_idx=0,
                obj_id=obj_id,
                box=np.asarray(box, dtype=np.float32),
            )
        except BaseException:
            self._failed = True
            raise
        return self._commit_current()

    def track(self, frame: CapturedFrame) -> LiveMaskResult:
        """Track the next received frame, preserving the previous SAM 2 memory."""
        self._require_open()
        if not self._prompted:
            raise RuntimeError("prompt the first live frame before tracking")
        if frame.decoded_sequence <= self._current_frame.decoded_sequence:
            raise ValueError("decoded frame sequence must increase")
        frame_idx = self._frame_source.append_bgr_frame(frame.bgr)
        self._state["num_frames"] = frame_idx + 1
        self._current_frame = frame
        return self._commit_current()

    @property
    def retained_output_count(self) -> int:
        outputs = self._state["output_dict"]
        return len(outputs["cond_frame_outputs"]) + len(
            outputs["non_cond_frame_outputs"]
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._predictor.close_video_source(self._state)
        finally:
            self._predictor.reset_state(self._state)

    def __enter__(self) -> LiveVideoSession:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _commit_current(self) -> LiveMaskResult:
        frame_idx = self._state["num_frames"] - 1
        predictions = self._predictor.propagate_in_video_predictions(
            self._state,
            start_frame_idx=frame_idx,
            max_frame_num_to_track=0,
            show_progress=False,
        )
        try:
            prediction = next(predictions)
            committed = self._predictor.commit_video_frame(self._state, prediction)
        except BaseException:
            self._failed = True
            raise
        finally:
            predictions.close()
        self._prompted = True
        self._retire_old_outputs(next_frame_idx=frame_idx + 1)
        masks = (committed.mask_logits > 0).squeeze(1).to(device="cpu").numpy().copy()
        return LiveMaskResult(
            processed_index=frame_idx,
            decoded_sequence=self._current_frame.decoded_sequence,
            presentation_time_seconds=self._current_frame.presentation_time_seconds,
            receive_to_mask_seconds=time.monotonic()
            - self._current_frame.received_at_monotonic,
            object_ids=committed.object_ids,
            masks=masks,
        )

    def _retire_old_outputs(self, *, next_frame_idx: int) -> None:
        """Keep every non-conditioning frame the next forward step could read."""
        stride = self._predictor.memory_temporal_stride_for_eval
        memory_lookback = (
            1 + max(0, self._predictor.num_maskmem - 2) * stride
            if self._predictor.num_maskmem > 1
            else 0
        )
        pointer_lookback = (
            self._predictor.max_obj_ptrs_in_encoder - 1
            if self._predictor.use_obj_ptrs_in_encoder
            else 0
        )
        keep_from = max(1, next_frame_idx - max(memory_lookback, pointer_lookback))
        outputs = self._state["output_dict"]
        per_object = self._state["output_dict_per_obj"]
        tracked = self._state["frames_already_tracked"]
        for old_index in tuple(outputs["non_cond_frame_outputs"]):
            if old_index >= keep_from:
                continue
            outputs["non_cond_frame_outputs"].pop(old_index)
            for object_outputs in per_object.values():
                object_outputs["non_cond_frame_outputs"].pop(old_index, None)
            tracked.pop(old_index, None)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("live video session is closed")
        if self._failed:
            raise RuntimeError("live video session failed; start a new session")

    def _require_unprompted(self) -> None:
        self._require_open()
        if self._prompted:
            raise RuntimeError("live video session already has its initial prompt")
