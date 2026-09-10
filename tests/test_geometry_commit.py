"""CPU contract checks through the real commit and mask-to-memory methods.

Only image feature extraction and the learned memory encoder are replaced.
The tests exercise the predictor's actual preparation, policy and publication.
"""

from __future__ import annotations

import unittest
from typing import Any

import torch
from torch import Tensor, nn

from sam2.sam2_video_predictor import (
    SAM2VideoPredictor,
    VideoFramePrediction,
    _PendingVideoFrameCommit,
)
from sam2.utils.geometry_delta import add_geometry_delta


class CaptureEncoder(nn.Module):
    def forward(self, features: Tensor, mask: Tensor, *, skip_mask_sigmoid: bool) -> dict[str, Any]:
        return {"vision_features": mask.clone(), "vision_pos_enc": [torch.zeros_like(mask)]}


class CommitHarness(SAM2VideoPredictor):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.image_size = 8
        self.hidden_dim = 1
        self.non_overlap_masks_for_mem_enc = False
        self.binarize_mask_from_pts_for_mem_enc = True
        self.sigmoid_scale_for_mem_enc = 1.0
        self.sigmoid_bias_for_mem_enc = 0.0
        self.no_obj_embed_spatial = None
        self.spatial_perceiver = None
        self.memory_encoder = CaptureEncoder()
        self.encoded: list[tuple[Tensor, bool]] = []
        self.fail_encoding = False
        self.eval()

    def _run_memory_encoder(self, *, high_res_masks: Tensor, is_mask_from_pts: bool,
                            object_score_logits: Tensor, **kwargs: Any) -> tuple[Tensor, list[Tensor]]:
        if self.fail_encoding:
            raise RuntimeError("injected encoding failure")
        self.encoded.append((high_res_masks.clone(), is_mask_from_pts))
        return self._encode_new_memory(
            current_vision_feats=[torch.ones(4, 1, 1)], feat_sizes=[(2, 2)],
            pred_masks_high_res=high_res_masks, object_score_logits=object_score_logits,
            is_mask_from_pts=is_mask_from_pts,
        )


def pending_frame(*, prompted: bool = False) -> tuple[dict[str, Any], VideoFramePrediction]:
    native = torch.linspace(-2.0, 2.0, 64).reshape(1, 1, 8, 8)
    # Deliberately not resamples of native: ordinary output postprocessing and
    # resolution changes can make these baselines differ.
    low = torch.full((1, 1, 2, 2), 0.37)
    video = torch.full((1, 1, 3, 5), 0.61)
    score = torch.tensor([[0.4]])
    token = object()
    output = {"pred_masks": low, "obj_ptr": torch.tensor([[0.3, 0.8]]),
              "object_score_logits": score, "maskmem_features": None, "maskmem_pos_enc": None}
    pending = _PendingVideoFrameCommit(
        token=token, frame_idx=0 if prompted else 1, object_ids=(7,),
        storage_key="cond_frame_outputs" if prompted else "non_cond_frame_outputs",
        current_out=output, pred_masks_high_res=None if prompted else native,
        memory_already_encoded=prompted, reverse=False,
    )
    state = {"pending_frame_commit": pending, "device": "cpu", "storage_device": "cpu",
             "output_dict": {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
             "output_dict_per_obj": {0: {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}},
             "frames_already_tracked": {}}
    prediction = VideoFramePrediction(
        frame_idx=pending.frame_idx, object_ids=(7,), mask_logits=video, _commit_token=token,
        memory_mask_logits=None if prompted else native.clone(), object_score_logits=score.clone(),
    )
    return state, prediction


class GeometryCommitTests(unittest.TestCase):
    def test_zero_delta_matches_original_at_every_resolution_and_in_memory(self) -> None:
        original, prediction = pending_frame()
        corrected, correction_prediction = pending_frame()
        first, second = CommitHarness(), CommitHarness()
        a = first.commit_video_frame(original, prediction)
        b = second.commit_video_frame(corrected, correction_prediction,
                                     geometry_mask_delta=torch.zeros(1, 1, 8, 8))
        torch.testing.assert_close(a.mask_logits, b.mask_logits, rtol=0, atol=0)
        for name in ("pred_masks", "maskmem_features", "obj_ptr", "object_score_logits"):
            torch.testing.assert_close(original["output_dict"]["non_cond_frame_outputs"][1][name],
                                       corrected["output_dict"]["non_cond_frame_outputs"][1][name],
                                       rtol=0, atol=0)
        self.assertEqual(len(second.encoded), 1)
        self.assertFalse(second.encoded[0][1])

    def test_geometry_remains_soft_while_legacy_replacement_keeps_prompt_policy(self) -> None:
        state, prediction = pending_frame()
        harness = CommitHarness()
        delta = torch.zeros(1, 1, 8, 8)
        delta[..., :4, :4] = -1.0
        native = prediction.memory_mask_logits.clone()
        pending_output = state["pending_frame_commit"].current_out
        preview = add_geometry_delta(prediction.mask_logits, delta)
        committed = harness.commit_video_frame(state, prediction, geometry_mask_delta=delta)
        torch.testing.assert_close(committed.mask_logits, preview, rtol=0, atol=0)
        output = state["output_dict"]["non_cond_frame_outputs"][1]
        torch.testing.assert_close(output["maskmem_features"], (native + delta).sigmoid())
        torch.testing.assert_close(harness.encoded[0][0][..., 4:, 4:], native[..., 4:, 4:])
        torch.testing.assert_close(pending_output["pred_masks"], torch.full((1, 1, 2, 2), 0.37))
        legacy_state, legacy_prediction = pending_frame()
        harness.commit_video_frame(legacy_state, legacy_prediction,
                                  adjusted_mask_logits=legacy_prediction.mask_logits)
        self.assertTrue(harness.encoded[-1][1])
        torch.testing.assert_close(legacy_state["output_dict"]["non_cond_frame_outputs"][1]["maskmem_features"],
                                   torch.ones(1, 1, 8, 8))

    def test_invalid_delta_leaves_pending_and_published_state_unchanged(self) -> None:
        cases = [(torch.zeros(1, 1, 3, 5), ValueError),
                 (torch.zeros(1, 1, 8, 8, dtype=torch.bool), TypeError),
                 (torch.full((1, 1, 8, 8), float("nan")), ValueError)]
        for delta, error in cases:
            with self.subTest(error=error, shape=delta.shape):
                state, prediction = pending_frame()
                pending = state["pending_frame_commit"]
                harness = CommitHarness()
                with self.assertRaises(error):
                    harness.commit_video_frame(state, prediction, geometry_mask_delta=delta)
                self.assertIs(state["pending_frame_commit"], pending)
                self.assertEqual(state["output_dict"]["non_cond_frame_outputs"], {})
                self.assertEqual(harness.encoded, [])

    def test_excludes_mixed_adjustments_and_prompt_geometry(self) -> None:
        state, prediction = pending_frame()
        harness = CommitHarness()
        with self.assertRaises(ValueError):
            harness.commit_video_frame(state, prediction, adjusted_mask_logits=prediction.mask_logits,
                                      geometry_mask_delta=torch.zeros(1, 1, 8, 8))
        state, prediction = pending_frame(prompted=True)
        with self.assertRaises(ValueError):
            harness.commit_video_frame(state, prediction, geometry_mask_delta=torch.zeros(1, 1, 8, 8))
        harness.commit_video_frame(state, prediction)
        self.assertEqual(harness.encoded, [])

    def test_legacy_dtype_overflow_does_not_publish(self) -> None:
        state, prediction = pending_frame()
        harness = CommitHarness()
        with self.assertRaises(ValueError):
            harness.commit_video_frame(state, prediction,
                                      adjusted_mask_logits=torch.full((1, 1, 3, 5), 1e100, dtype=torch.float64))
        self.assertEqual(harness.encoded, [])
        self.assertEqual(state["output_dict"]["non_cond_frame_outputs"], {})
        harness.commit_video_frame(state, prediction)

    def test_encoder_failure_does_not_publish_or_mutate_pending_masks(self) -> None:
        state, prediction = pending_frame()
        pending = state["pending_frame_commit"]
        before = pending.current_out["pred_masks"].clone()
        harness = CommitHarness()
        harness.fail_encoding = True
        with self.assertRaisesRegex(RuntimeError, "injected"):
            harness.commit_video_frame(state, prediction, geometry_mask_delta=torch.ones(1, 1, 8, 8))
        torch.testing.assert_close(before, pending.current_out["pred_masks"], rtol=0, atol=0)
        self.assertEqual(state["output_dict"]["non_cond_frame_outputs"], {})

    def test_commit_cannot_be_repeated_or_use_another_token(self) -> None:
        state, prediction = pending_frame()
        _, other = pending_frame()
        harness = CommitHarness()
        with self.assertRaises(ValueError):
            harness.commit_video_frame(state, other)
        harness.commit_video_frame(state, prediction)
        with self.assertRaises(RuntimeError):
            harness.commit_video_frame(state, prediction)


if __name__ == "__main__":
    unittest.main()
