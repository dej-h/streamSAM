"""Actual memory-selector equivalence and synchronized forward-state eviction."""

from __future__ import annotations

import unittest
from typing import Any

import torch
from torch import Tensor, nn

from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.utils.forward_history import ForwardHistoryPolicy, ForwardHistoryRetention
from test_geometry_commit import CommitHarness, pending_frame


class CaptureAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.observed: tuple[Tensor, Tensor, int, int] | None = None

    def forward(self, *, curr: list[Tensor], curr_pos: list[Tensor], memory: Tensor,
                memory_pos: Tensor, num_obj_ptr_tokens: int, num_spatial_mem: int) -> Tensor:
        self.observed = memory.clone(), memory_pos.clone(), num_obj_ptr_tokens, num_spatial_mem
        return curr[-1]


class SelectorHarness(SAM2VideoPredictor):
    def __init__(self, n: int, stride: int, pointers: bool, pointer_cap: int) -> None:
        nn.Module.__init__(self)
        self.num_maskmem = n
        self.memory_temporal_stride_for_eval = stride
        self.use_obj_ptrs_in_encoder = pointers
        self.max_obj_ptrs_in_encoder = pointer_cap
        self.hidden_dim = self.mem_dim = 4
        self.max_cond_frames_in_attn = -1
        self.only_obj_ptrs_in_the_past_for_eval = True
        self.use_signed_tpos_enc_to_obj_ptrs = False
        self.add_tpos_enc_to_obj_ptrs = False
        self.maskmem_tpos_enc = torch.arange(n, dtype=torch.float32).reshape(n, 1, 1, 1).expand(n, 1, 1, 4)
        self.memory_attention = CaptureAttention()
        self.eval()

    def selected(self, frame_idx: int, outputs: dict[str, Any]) -> tuple[Tensor, Tensor, int, int] | None:
        self.memory_attention.observed = None
        self._prepare_memory_conditioned_features(
            frame_idx, False, [torch.zeros(1, 1, 4)], [torch.zeros(1, 1, 4)],
            [(1, 1)], outputs, num_frames=200)
        return self.memory_attention.observed


def output(index: int) -> dict[str, Any]:
    return {"maskmem_features": torch.full((1, 4, 1, 1), float(index)),
            "maskmem_pos_enc": [torch.zeros(1, 4, 1, 1)],
            "obj_ptr": torch.full((1, 4), 1000. + index)}


def indexed_state(policy: ForwardHistoryPolicy) -> dict[str, Any]:
    return {"forward_history_retention": ForwardHistoryRetention(policy, last_committed_frame=0),
            "output_dict": {"cond_frame_outputs": {0: output(0)}, "non_cond_frame_outputs": {}},
            "output_dict_per_obj": {i: {"cond_frame_outputs": {0: output(0)}, "non_cond_frame_outputs": {}}
                                    for i in range(policy.max_objects)},
            "frames_already_tracked": {0: {"reverse": False}}}


class ForwardHistoryTests(unittest.TestCase):
    def test_retained_history_preserves_actual_future_selector_inputs(self) -> None:
        for n in (0, 1, 2, 7):
            for stride in (1, 2, 5):
                for pointers, cap in ((False, 16), (True, 1), (True, 16)):
                    with self.subTest(n=n, stride=stride, pointers=pointers, cap=cap):
                        model = SelectorHarness(n, stride, pointers, cap)
                        policy = model._forward_policy(2)
                        state = indexed_state(policy)
                        full = {"cond_frame_outputs": {0: output(0)}, "non_cond_frame_outputs": {}}
                        for frame_idx in range(1, 100):
                            expected = model.selected(frame_idx, full)
                            actual = model.selected(frame_idx, state["output_dict"])
                            if expected is None:
                                self.assertIsNone(actual)
                            else:
                                assert actual is not None
                                torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
                                torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
                                self.assertEqual(actual[2:], expected[2:])
                            value = output(frame_idx)
                            full["non_cond_frame_outputs"][frame_idx] = value
                            state["output_dict"]["non_cond_frame_outputs"][frame_idx] = value
                            for per_object in state["output_dict_per_obj"].values():
                                per_object["non_cond_frame_outputs"][frame_idx] = value
                            state["frames_already_tracked"][frame_idx] = {"reverse": False}
                            model._retain_forward_history(state, frame_idx)
                            retained = set(state["output_dict"]["non_cond_frame_outputs"])
                            self.assertLessEqual(len(retained), policy.non_conditioning_capacity)
                            self.assertEqual(set(state["frames_already_tracked"]), retained | {0})
                            for per_object in state["output_dict_per_obj"].values():
                                self.assertEqual(set(per_object["non_cond_frame_outputs"]), retained)
                                self.assertEqual(set(per_object["cond_frame_outputs"]), {0})
                        stats = model.forward_history_stats(state)
                        assert stats is not None
                        self.assertEqual(stats.committed_frames, 100)
                        self.assertEqual(stats.evicted_frames + stats.retained_non_conditioning_frames, 99)

    def test_invalid_start_or_selector_change_fails_before_preflight(self) -> None:
        model = SelectorHarness(7, 5, True, 16)
        for start, reverse in ((0, True), (2, False)):
            state = {"forward_history_retention": ForwardHistoryRetention(model._forward_policy(1)),
                     "tracking_has_started": False}
            with self.assertRaises(ValueError):
                next(model.propagate_in_video_predictions(state, start_frame_idx=start, reverse=reverse))
            self.assertFalse(state["tracking_has_started"])
        state = {"forward_history_retention": ForwardHistoryRetention(model._forward_policy(1))}
        model.memory_temporal_stride_for_eval = 2
        with self.assertRaisesRegex(RuntimeError, "selector"):
            next(model.propagate_in_video_predictions(state))

    def test_prompt_and_object_limits_fail_before_state_mutation(self) -> None:
        model = SelectorHarness(7, 1, True, 16)
        state = {"tracking_has_started": False, "obj_ids": [], "obj_id_to_idx": {}}
        model.enable_forward_history(state)
        with self.assertRaises(ValueError):
            model.add_new_points_or_box(state, 5, 7)
        self.assertEqual(state["obj_ids"], [])
        state["obj_ids"] = [7]
        state["obj_id_to_idx"] = {7: 0}
        with self.assertRaisesRegex(ValueError, "capacity"):
            model.add_new_mask(state, 0, 8, torch.ones(8, 8))
        state["tracking_has_started"] = True
        for action in (lambda: model.add_new_points_or_box(state, 0, 7),
                       lambda: model.clear_all_prompts_in_frame(state, 0, 7),
                       lambda: model.remove_object(state, 7)):
            with self.assertRaisesRegex(ValueError, "initial"):
                action()
        self.assertEqual(state["obj_ids"], [7])

    def test_commit_rejects_nonconsecutive_frame_before_encoding(self) -> None:
        model = CommitHarness()
        model.num_maskmem, model.memory_temporal_stride_for_eval = 7, 1
        model.use_obj_ptrs_in_encoder, model.max_obj_ptrs_in_encoder = True, 16
        state, prediction = pending_frame()
        state["forward_history_retention"] = ForwardHistoryRetention(model._forward_policy(1))
        with self.assertRaisesRegex(ValueError, "consecutive"):
            model.commit_video_frame(state, prediction)
        self.assertEqual(model.encoded, [])
        self.assertIsNotNone(state["pending_frame_commit"])
        self.assertEqual(state["output_dict"]["non_cond_frame_outputs"], {})

    def test_actual_commit_evicts_aggregate_object_and_tracking_entries_together(self) -> None:
        model = CommitHarness()
        model.num_maskmem, model.memory_temporal_stride_for_eval = 2, 1
        model.use_obj_ptrs_in_encoder, model.max_obj_ptrs_in_encoder = False, 16
        state, prediction = pending_frame()
        state["forward_history_retention"] = ForwardHistoryRetention(model._forward_policy(1), last_committed_frame=0)
        # Expired non-conditioning frame zero is distinct from the pinned cond
        # output; only the tracking zero is explicitly pinned by policy.
        state["output_dict"]["non_cond_frame_outputs"][0] = output(0)
        state["output_dict_per_obj"][0]["non_cond_frame_outputs"][0] = output(0)
        state["frames_already_tracked"][0] = {"reverse": False}
        model.commit_video_frame(state, prediction)
        self.assertEqual(set(state["output_dict"]["non_cond_frame_outputs"]), {1})
        self.assertEqual(set(state["output_dict_per_obj"][0]["non_cond_frame_outputs"]), {1})
        self.assertEqual(set(state["frames_already_tracked"]), {0, 1})

    def test_default_retains_unbounded_interactive_history(self) -> None:
        model = SelectorHarness(7, 1, True, 16)
        state = indexed_state(model._forward_policy(1))
        del state["forward_history_retention"]
        for i in range(40):
            state["output_dict"]["non_cond_frame_outputs"][i] = output(i)
        model._retain_forward_history(state, 39)
        self.assertEqual(len(state["output_dict"]["non_cond_frame_outputs"]), 40)
        self.assertIsNone(model.forward_history_stats(state))

    def test_reset_clears_cursor_outputs_and_pending_but_keeps_policy(self) -> None:
        model = SelectorHarness(7, 1, True, 16)
        state = indexed_state(model._forward_policy(1))
        state.update(point_inputs_per_obj={0: {0: object()}}, mask_inputs_per_obj={0: {}},
                     temp_output_dict_per_obj={0: {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}},
                     consolidated_frame_inds={"cond_frame_outputs": {0}, "non_cond_frame_outputs": set()},
                     obj_ids=[7], obj_id_to_idx={7: 0}, obj_idx_to_id={0: 7}, pending_frame_commit=object())
        policy = state["forward_history_retention"].policy
        model.reset_state(state)
        self.assertEqual(state["forward_history_retention"].policy, policy)
        self.assertIsNone(state["pending_frame_commit"])
        stats = model.forward_history_stats(state)
        assert stats is not None
        self.assertEqual(stats.committed_frames, 0)
        self.assertEqual(stats.retained_conditioning_frames, 0)
        self.assertEqual(stats.retained_tracking_frames, 0)
        self.assertFalse(state["tracking_has_started"])


if __name__ == "__main__":
    unittest.main()
