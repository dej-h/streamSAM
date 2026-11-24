# Live camera and RTSP support proposal

Status: proposed. This document describes work to be implemented and verified;
the current streamSAM API accepts finite video files, not live sources.

## Goal and boundary

Accept frames from a live RTSP camera as they arrive, prompt an object, and
produce ordered masks while retaining useful SAM 2 / EdgeTAM temporal state.
Keep input latency, CPU memory, GPU staging, output, and predictor state bounded
for a long-running session. Report dropped frames and reconnects explicitly.

This targets SAM 2-family **video** predictors. The original image-only SAM has
no video memory to preserve and is outside this proposal. The first release
should support one camera and one prompted object; multi-camera scheduling and
late object insertion can follow after the state lifecycle is proven.

## Why a new live path is needed

- `SequentialMp4FrameSource` reads `CAP_PROP_FRAME_COUNT` before decoding and
  rejects a source with no known frame count. Its `FrameSource` protocol is a
  finite random-access sequence (`__len__`, `__getitem__`).
- `SAM2VideoPredictor.init_state` stores that length, warms frame zero, and
  allocates state around a file-backed source. `propagate_in_video_predictions`
  walks a finite `range`; it cannot wait for future camera frames.
- Committed outputs and `frames_already_tracked` accumulate by frame index.
  Offloading them to CPU reduces VRAM use but does not make a live session's
  memory constant. Existing benchmark output also records one trace per frame.
- SAM 2 memory attention uses recent non-conditioning frames, conditioning
  frames, and a bounded recent object-pointer window. A retention policy must
  preserve exactly the entries future inference will read. Pruning solely by
  age is insufficient when prompts may create conditioning frames.

## Proposed interfaces

Keep `FrameSource` as the file/seek contract. Add a separate pull interface for
live input so callers do not fake a length or random access:

```python
@dataclass(frozen=True)
class CapturedFrame:
    image: torch.Tensor  # normalized CHW frame owned by this delivery
    decoded_sequence: int  # assigned by the reader, not by the camera
    presentation_time_seconds: float | None  # source PTS, if available
    received_at_monotonic: float
    width: int
    height: int

class LiveFrameSource(Protocol):
    def read(self, timeout_seconds: float) -> CapturedFrame: ...
    def stats(self) -> LiveFrameSourceStats: ...
    def close(self) -> None: ...
```

The source owns decoding and a small fixed-capacity queue. Its first adapter
uses an FFmpeg-backed RTSP reader (PyAV is the initial candidate because it
exposes decoded-frame timestamps and transport errors). Make the decoder
replaceable; no RTSP-specific behavior belongs in the predictor. Normalize
frames through the same resize, RGB conversion, and mean/std path as MP4 input.
Treat credentials as source configuration and omit them from logs and artifacts.

Expose a live predictor session with `accept_frame`, `predict_frame`, and
`commit_frame`. Reuse the existing predict-then-commit core, but factor state
initialization so it does not require `video_path`, `len(images)`, or a warm-up
read of frame zero. A frame must be committed or explicitly discarded before
the next inference step. Assign a contiguous **processed-frame index** for SAM
2 temporal positions; retain decoded sequence and presentation timestamps
separately so queue drops are visible. Set any model-facing frame-count value
to the number of processed frames currently known; audit every `num_frames` use
before doing so.

For the first prompt, hold one decoded frame while the caller supplies a point
or box in that frame's original coordinates, then commit its conditioning
output. For a headless camera, allow a configured first-frame prompt. Subsequent
mask corrections should use the same predict/commit transaction; adding new
objects or retroactive edits needs a separate design and test gate.

## Latency and failure policy

The RTSP adapter's default is a **latest-frame** policy: if inference falls
behind, the decoder drops stale queued frames and delivers the newest complete
frame. This bounds end-to-end latency while preserving order among processed
frames. Count every locally dropped decoded frame, record decoded and processed
indices, and expose receive-to-mask latency. Camera-side or network losses
cannot be counted without a source sequence; capture-to-mask latency requires a
trustworthy mapping from source timestamps to local clock time. A
process-every-frame mode may be useful for a finite replay but must not
silently accumulate an unbounded live backlog.

The decoder runs independently of inference, with explicit read/open timeouts,
close semantics, and a bounded retry schedule. A disconnect ends the current
tracking epoch. The default reconnect policy starts a new epoch and requires a
new prompt; carrying state across a discontinuity needs measured mask-quality
evidence. A resolution change also starts a new epoch. An RTSP URL alone does
not imply that OpenCV `read()` will return promptly or preserve useful PTS, so
those cases belong in the adapter's integration checks.

## Bounded temporal state

After each commit, prune only entries no longer reachable by the next forward
step: old non-conditioning outputs, their per-object views, tracking metadata,
and cached features. The retention window must account for both
`num_maskmem`/`memory_temporal_stride_for_eval` and
`max_obj_ptrs_in_encoder`. Test the retention rules against the supported
configuration values; do not repeat a long-running soak for every checkpoint.
Keep the initial conditioning frame. Limit later conditioning frames with an
explicit policy and evaluate the quality cost before enforcing a cap. Record
counts and bytes for all retained state categories; do not infer bounded memory
merely from a bounded decoder queue or GPU offload.

This is a forward-only live contract. File-backed reverse propagation and
retroactive refinement continue to use the finite-video path.

## Implementation and verification sequence

1. **Source contract:** extract shared frame normalization; add `LiveFrameSource`
   and a deterministic in-memory source. Prove ordering, timeouts, bounded
   queue depth, drops, and idempotent close without a network dependency.
2. **Predictor lifecycle:** factor state creation; add one-frame
   predict/commit; compare a deterministic frame replay against the existing
   finite-video path on one representative model. For other supported SAM 2
   configurations, check that their checkpoints load and that a short replay
   follows the same one-frame path with ordered outputs.
3. **Retention:** implement state pruning with invariants over conditioning
   frames, memory frames, object pointers, per-object views, and prompt data.
   Compare masks against unpruned replay, including stride changes, occlusion,
   and a late correction. Measure state counts and RSS/VRAM over at least
   10,000 generated frames on one representative model. The retention code is
   shared; exercise distinct memory-window configurations with short targeted
   checks instead of a separate soak for each model.
4. **RTSP adapter:** add FFmpeg-backed input with timestamps and a bounded
   latest-frame queue. Exercise normal playback, producer faster than consumer,
   disconnect/reconnect, stalled reads, resolution changes, and shutdown using
   a local RTSP server and prerecorded source.
5. **End-to-end gate:** run one long-lived camera replay with frame-level
   receive-to-mask latency, FPS, local drop count, state-size, RSS, and VRAM traces.
   Confirm a memory plateau after warm-up and inspect tracking quality across
   drops and reconnect epochs. Publish the exact model, checkpoint, transport,
   source, and hardware used. Other SAM 2-family checkpoints need compatibility
   checks through the shared path, not repeated long-running memory tests.

The initial public API and example should expose only behaviors that pass these
gates. In particular, no claim of arbitrary-duration constant-memory tracking
follows from the current finite-file benchmark.
