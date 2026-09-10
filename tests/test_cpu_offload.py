"""CUDA storage publication must finish before any CPU reader sees the tensor.

Only learned computation is replaced. The real single-frame inference, memory
offload and adjusted-mask commit paths execute with deliberately delayed copies.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator
import unittest
from unittest.mock import patch

import torch

from sam2.sam2_video_predictor import FrameOutput, SAM2VideoPredictor, VideoFramePrediction, _PendingVideoFrameCommit


@dataclass(frozen=True)
class CpuPublication:
    complete: bool
    snapshot: torch.Tensor


@contextmanager
def delayed_cpu_copies() -> Iterator[list[CpuPublication]]:
    """Delay the real CUDA writer; inspect CPU readiness without synchronizing."""
    publications: list[CpuPublication] = []
    original_to = torch.Tensor.to

    def transfer(tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        destination = kwargs.get("device", args[0] if args else None)
        offloading = (tensor.is_cuda and isinstance(destination, (str, torch.device))
                      and torch.device(destination).type == "cpu")
        if offloading:
            torch.cuda._sleep(20_000_000)
        result = original_to(tensor, *args, **kwargs)
        if offloading:
            publications.append(CpuPublication(torch.cuda.current_stream().query(), result.clone()))
        return result

    try:
        with patch.object(torch.Tensor, "to", transfer):
            yield publications
    finally:
        # Even a failing test must not leave a pending writer using CPU storage.
        torch.cuda.current_stream().synchronize()


@unittest.skipUnless(torch.cuda.is_available(), "requires real CUDA transfers")
class CpuOffloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.predictor = SAM2VideoPredictor.__new__(SAM2VideoPredictor)
        torch.nn.Module.__init__(self.predictor)
        self.predictor.fill_hole_area = 0
        self.predictor.image_size = 8
        self.value = torch.full((1, 1, 2, 2), 7.0, device="cuda")
        self.state: dict[str, Any] = {"device": torch.device("cuda"), "storage_device": torch.device("cpu"),
                                      "constants": {}, "feature_pipeline": None, "num_frames": 3}
        self.features = patch.object(self.predictor, "_get_image_feature", return_value=(None, None, [], [], []))
        self.features.start()
        self.addCleanup(self.features.stop)

    def assert_publications(self, publications: list[CpuPublication], count: int) -> None:
        self.assertEqual(len(publications), count)
        for publication in publications:
            self.assertTrue(publication.complete, "CPU tensor was published before the CUDA copy finished")
            torch.testing.assert_close(publication.snapshot, torch.full_like(publication.snapshot, 7), rtol=0, atol=0)

    def test_single_frame_publishes_ready_masks_memory_pointers_and_scores(self) -> None:
        output: FrameOutput = {name: self.value for name in ("pred_masks", "maskmem_features", "obj_ptr", "object_score_logits")}
        output["maskmem_pos_enc"] = None
        with patch.object(self.predictor, "track_step", return_value=output), delayed_cpu_copies() as publications:
            compact, _ = self.predictor._run_single_frame_inference(
                self.state, {}, 0, 1, True, None, self.value, False, True)
            self.assert_publications(publications, 4)
            for name in ("pred_masks", "maskmem_features", "obj_ptr", "object_score_logits"):
                torch.testing.assert_close(compact[name], torch.full_like(compact[name], 7), rtol=0, atol=0)

    def test_memory_encoder_publishes_ready_cpu_features(self) -> None:
        with patch.object(self.predictor, "_encode_new_memory", return_value=(self.value, None)), delayed_cpu_copies() as publications:
            features, _ = self.predictor._run_memory_encoder(self.state, 0, 1, self.value, self.value, False)
            self.assert_publications(publications, 1)
            torch.testing.assert_close(features, torch.full_like(features, 7), rtol=0, atol=0)

    def test_adjusted_commit_publishes_ready_cpu_mask(self) -> None:
        token = object()
        output = {"pred_masks": torch.zeros(1, 1, 2, 2), "maskmem_features": None, "maskmem_pos_enc": None,
                  "obj_ptr": self.value, "object_score_logits": self.value}
        pending = _PendingVideoFrameCommit(token, 1, (0,), "non_cond_frame_outputs", output, self.value, False, False)
        self.state.update(pending_frame_commit=pending, frames_already_tracked={},
                          output_dict={"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
                          output_dict_per_obj={0: {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}})
        prediction = VideoFramePrediction(1, (0,), self.value, token, self.value)
        # Isolate mask publication: a subsequent blocking memory copy must not
        # accidentally conceal an incomplete earlier mask copy.
        with patch.object(self.predictor, "_run_memory_encoder", return_value=(None, None)), delayed_cpu_copies() as publications:
            self.predictor.commit_video_frame(self.state, prediction, adjusted_mask_logits=self.value)
            self.assert_publications(publications, 1)
            mask = self.state["output_dict"]["non_cond_frame_outputs"][1]["pred_masks"]
            torch.testing.assert_close(mask, torch.full_like(mask, 7), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
