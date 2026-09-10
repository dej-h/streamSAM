"""Borrowed paths, exact preprocessing, frame-local errors and source ownership."""

from __future__ import annotations

import tempfile
import unittest
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from threading import Event, Thread
from typing import Any, overload
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch import nn

from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.utils.gpu_frame_stager import GpuFrameStager
from sam2.utils.misc import _load_img_as_tensor
from sam2.utils.video_stream import (
    EagerFrameSource, SequentialImageDirectoryFrameSource, SequentialImageFrameSource,
)


class RepeatedPaths(Sequence[Path]):
    """An enormous logical sequence which cannot be enumerated or sliced."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.reads = 0

    def __len__(self) -> int:
        return 100_000_000

    @overload
    def __getitem__(self, index: int) -> Path: ...
    @overload
    def __getitem__(self, index: slice) -> Sequence[Path]: ...
    def __getitem__(self, index: int | slice) -> Path | Sequence[Path]:
        if isinstance(index, slice):
            raise AssertionError("source must not slice its paths")
        if not 0 <= index < len(self):
            raise IndexError(index)
        self.reads += 1
        if self.reads > 30:
            raise AssertionError("source enumerated its paths")
        return self.path


class InitHarness(SAM2VideoPredictor):
    def __init__(self, *, fail: bool = False) -> None:
        nn.Module.__init__(self)
        self.register_parameter("anchor", nn.Parameter(torch.zeros(())))
        self.image_size = 8
        self.fail = fail

    def _get_image_feature(self, inference_state: dict[str, Any], frame_idx: int, batch_size: int) -> Any:
        if self.fail:
            raise RuntimeError("warmup failed")
        return inference_state["images"][frame_idx]


class ImageSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for i in range(3):
            data = (np.arange(45).reshape(3, 5, 3) * (i + 1)).astype(np.uint8)
            Image.fromarray(data).save(self.root / f"{i}.jpg")
        self.paths = [self.root / f"{i}.jpg" for i in range(3)]

    def source(self, paths: Sequence[Path] | None = None, capacity: int = 2) -> SequentialImageFrameSource:
        source = SequentialImageFrameSource(self.paths if paths is None else paths, 8, capacity)
        self.addCleanup(source.close)
        return source

    def test_exact_preprocessing_and_seek_match_directory_and_eager_ordering(self) -> None:
        source = self.source()
        directory = SequentialImageDirectoryFrameSource(str(self.root), 8, 2,
                                                        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.addCleanup(directory.close)
        for i in (0, 1, 2, 0, 2, 1):
            expected, _, _ = _load_img_as_tensor(str(self.paths[i]), 8)
            expected = expected.float()
            expected.sub_(torch.tensor((0.485, 0.456, 0.406))[:, None, None])
            expected.div_(torch.tensor((0.229, 0.224, 0.225))[:, None, None])
            torch.testing.assert_close(source[i], expected, rtol=0, atol=0)
            torch.testing.assert_close(directory[i], expected, rtol=0, atol=0)
        self.assertLessEqual(source.stats().maximum_depth, 2)
        source.close()
        self.assertFalse(source._worker.is_alive())
        with self.assertRaisesRegex(RuntimeError, "closed"):
            source[0]

    def test_borrows_enormous_sequence_without_materialization(self) -> None:
        paths = RepeatedPaths(self.paths[0])
        source = self.source(paths)
        self.assertEqual(len(source), len(paths))
        torch.testing.assert_close(source[0], source[99_999_999], rtol=0, atol=0)
        self.assertLessEqual(paths.reads, 10)

    def test_prefetch_failure_preserves_earlier_frames_and_allows_seek(self) -> None:
        source = self.source([*self.paths[:2], self.root / "missing.jpg", self.paths[0]], capacity=4)
        with source._condition:
            self.assertTrue(source._condition.wait_for(lambda: source._worker_error is not None, timeout=5))
        first = source[0]
        self.assertEqual(source[1].shape, first.shape)
        with self.assertRaisesRegex(RuntimeError, "frame 2"):
            source[2]
        torch.testing.assert_close(source[3], first, rtol=0, atol=0)
        torch.testing.assert_close(source[0], first, rtol=0, atol=0)

    def test_wrong_future_dimensions_fail_when_requested(self) -> None:
        Image.new("RGB", (6, 3)).save(self.paths[2])
        source = self.source(capacity=3)
        with source._condition:
            self.assertTrue(source._condition.wait_for(lambda: source._worker_error is not None, timeout=5))
        self.assertEqual(source[0].shape, (3, 8, 8))
        with self.assertRaisesRegex(RuntimeError, "frame 2") as caught:
            source[2]
        self.assertIsInstance(caught.exception.__cause__, ValueError)

    def request_in_thread(self, source: SequentialImageFrameSource, index: int,
                          started: Event | None = None) -> tuple[Event, list[object]]:
        done = Event()
        outcome: list[object] = []

        def consume() -> None:
            try:
                with source._condition:
                    if started is not None:
                        started.set()
                    outcome.append(source[index])
            except BaseException as error:
                outcome.append(error)
            finally:
                done.set()

        thread = Thread(target=consume, daemon=True)
        self.addCleanup(thread.join, 5)
        self.addCleanup(source.close)
        thread.start()
        return done, outcome

    def test_seek_to_next_index_when_older_frames_fill_queue(self) -> None:
        source = self.source(capacity=2)
        with source._condition:
            self.assertTrue(source._condition.wait_for(lambda: len(source._frames) == 2, timeout=5))
        done, outcome = self.request_in_thread(source, 2)
        self.assertTrue(done.wait(5), "seek deadlocked behind full queue")
        self.assertIsInstance(outcome[0], torch.Tensor)

    def test_seek_past_inflight_failure(self) -> None:
        entered, release = Event(), Event()
        original = SequentialImageFrameSource._decode_frame

        def decode(source: SequentialImageFrameSource, index: int) -> torch.Tensor:
            if index == 0:
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("test did not release decoder")
                raise ValueError("failed predecessor")
            return original(source, index)

        with patch.object(SequentialImageFrameSource, "_decode_frame", decode):
            source = self.source()
            self.addCleanup(release.set)
            self.assertTrue(entered.wait(5))
            started = Event()
            done, outcome = self.request_in_thread(source, 1, started)
            # Wait until the consumer has entered its condition wait. Acquiring
            # this lock after a start barrier avoids a timing-dependent sleep.
            self.assertTrue(started.wait(5))
            with source._condition:
                pass
            release.set()
            self.assertTrue(done.wait(5), "request stalled behind failed predecessor")
            self.assertIsInstance(outcome[0], torch.Tensor)

    def test_predictor_accepts_source_and_closes_it_after_warmup_failure(self) -> None:
        predictor = InitHarness()
        source = self.source()
        with patch("sam2.sam2_video_predictor.create_video_frame_source", side_effect=AssertionError("directory scan")):
            state = predictor.init_state(frame_source=source, frame_loading="lazy")
        self.assertIs(state["frame_source"], source)
        self.assertEqual(state["num_frames"], 3)
        predictor.close_video_source(state)
        self.assertTrue(source.stats().closed)
        failing = self.source()
        with self.assertRaisesRegex(RuntimeError, "warmup failed"):
            InitHarness(fail=True).init_state(frame_source=failing)
        self.assertTrue(failing.stats().closed)
        self.assertFalse(failing._worker.is_alive())

    def test_metadata_rejection_closes_owned_source_but_ambiguous_input_does_not(self) -> None:
        source = self.source()
        predictor = InitHarness()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            predictor.init_state(video_path=str(self.root), frame_source=source)
        self.assertFalse(source.stats().closed)
        for metadata in (replace(source.metadata, image_size=16),
                         replace(source.metadata, preprocessing_identity="wrong"),
                         replace(source.metadata, video_width=0)):
            supplied = EagerFrameSource([torch.zeros(3, 8, 8)] * 3, metadata)
            with self.assertRaises(ValueError):
                predictor.init_state(frame_source=supplied)
            self.assertTrue(supplied.stats().closed)

    def test_device_lookup_failure_closes_supplied_source(self) -> None:
        class BrokenDevice(InitHarness):
            @property
            def device(self) -> torch.device:
                raise RuntimeError("device unavailable")

        source = self.source()
        with self.assertRaisesRegex(RuntimeError, "device unavailable"):
            BrokenDevice().init_state(frame_source=source)
        self.assertTrue(source.stats().closed)
        self.assertFalse(source._worker.is_alive())

    def test_cleanup_failure_still_stops_source_worker(self) -> None:
        class FailingStager(GpuFrameStager):
            def __init__(self) -> None:
                pass

            def close(self) -> None:
                raise RuntimeError("CUDA cleanup failed")

        source = self.source()
        with self.assertRaisesRegex(RuntimeError, "CUDA cleanup failed"):
            InitHarness().close_video_source({"frame_source": source, "frame_stager": FailingStager()})
        self.assertTrue(source.stats().closed)
        self.assertFalse(source._worker.is_alive())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for staging failure verification")
    def test_gpu_prefetch_failure_preserves_valid_frames(self) -> None:
        source = self.source([self.paths[0], self.root / "missing.jpg", self.paths[2]])
        stager = GpuFrameStager(frame_source=source, device=torch.device("cuda"))
        self.addCleanup(stager.close)
        stager.prefetch(0)
        with stager._condition:
            self.assertTrue(stager._condition.wait_for(lambda: stager.stats().staged_frames == 1, timeout=5))
        stager.prefetch(1)
        with stager._condition:
            self.assertTrue(stager._condition.wait_for(
                lambda: stager._worker_error is not None or any(slot.state == "failed" for slot in stager._slots),
                timeout=5))
        with stager.acquire(0) as actual:
            expected = source[0].to("cuda")
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            stager.prefetch_adjacent(0)
        with self.assertRaisesRegex(RuntimeError, "frame 1"):
            stager.acquire(1)
        with stager.acquire(2) as actual:
            torch.testing.assert_close(actual, source[2].to("cuda"), rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for prefetch budget verification")
    def test_gpu_pending_requests_are_bounded_and_dropped_request_can_be_acquired(self) -> None:
        source = self.source(RepeatedPaths(self.paths[0]))
        stager = GpuFrameStager(frame_source=source, device=torch.device("cuda"))
        self.addCleanup(stager.close)
        with stager._condition:
            for index in range(100):
                stager.prefetch(index)
            self.assertEqual(stager.stats().maximum_pending_depth, 2)
        with stager.acquire(99) as actual:
            torch.testing.assert_close(actual, source[99].to("cuda"), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
