# SAM2 streaming scratchpad

## Intent

Replace eager/chunked video loading with a bounded streaming pipeline without changing segmentation results.

The implementation should:

- Keep VRAM bounded by resolution and object count, not video length.
- Match or beat the current end-to-end FPS.
- Work across the SAM2 family, including EdgeTAM, through explicit adapters.
- Keep frame decode/prefetch, model inference, and checkpoint persistence concurrent.
- Have frame `n + 1` ready before inference finishes frame `n`.
- Prove these claims on long videos with recorded measurements.

## Current path

- [`gradio_app.py`](gradio_app.py#L152) decodes the full upload and re-encodes it into 96-frame temporary MP4 chunks.
- [`load_video_frames`](sam2/utils/misc.py#L172) dispatches to the MP4 or JPEG loader.
- The MP4 loader in [`misc.py`](sam2/utils/misc.py#L282) decodes every frame in the current chunk and stacks them into one tensor.
- [`SAM2VideoPredictor.init_state`](sam2/sam2_video_predictor.py#L44) stores that tensor as `inference_state["images"]`.
- [`_get_image_feature`](sam2/sam2_video_predictor.py#L879) only requires `len(images)` and `images[frame_idx]`, then transfers one frame to the model device.
- [`_run_single_frame_inference`](sam2/sam2_video_predictor.py#L912) stores masks and memory features on the configured storage device. The output dictionaries continue growing with processed frames.
- EdgeTAM uses the older batched SAM2 predictor state. Current upstream SAM2 uses per-object state, so frame loading can be shared but checkpoint capture/restore needs separate adapters.

## Proposed boundaries

### Frame source

Add `sam2/utils/video_stream.py` with a typed `FrameSource` protocol and a sequential implementation for MP4/bytes input.

Required behavior:

- Expose frame count, original dimensions, FPS, and preprocessing identity.
- Return normalized `Tensor[C, H, W]` frames by index.
- Keep one decoder open and decode sequentially during normal propagation.
- Use a bounded pinned-CPU ring buffer.
- Support seeking/re-decoding for corrections on an evicted frame.
- Own no model-specific temporal state.

Keep the existing eager loader as a baseline and compatibility path.

### Predictor state adapter

Add `sam2/video_state.py` with a small adapter contract:

```python
class VideoStateAdapter(Protocol):
    def capture_checkpoint(self, state: InferenceState, frame_idx: int) -> Checkpoint: ...
    def restore_checkpoint(self, checkpoint: Checkpoint) -> InferenceState: ...
    def evict_before(self, state: InferenceState, frame_idx: int) -> None: ...
```

Provide one adapter for this legacy/batched predictor and one for current per-object SAM2 state. Do not make other SAM variants depend on EdgeTAM dictionary keys.

A checkpoint must record the model/config identity, preprocessing identity, frame index and direction, object mapping, prompt log, conditioning state, and temporal memory needed to continue. `cached_features` should be regenerated, not persisted.

### Three-stage pipeline

```text
decoder/prefetch thread       inference/CUDA               checkpoint thread
decode n+1 -> pinned CPU  ->  async H2D -> model n  ->     snapshot -> CPU/disk
bounded ring buffer           fixed GPU staging slots       bounded/coalescing queue
```

- Use a dedicated CUDA stream and event for H2D prefetch.
- Keep fixed current/next GPU staging slots so frame VRAM cannot grow with video length.
- Inference waits on the frame-ready event, not the loader thread.
- After inference, enqueue an immutable CPU-owned checkpoint payload. The writer must never read tensors that inference can still mutate.
- Keep both queues bounded and expose queue depth, wait time, and dropped/coalesced checkpoint counts.
- Keep only the active SAM2 memory window and conditioning state on GPU. Move or persist older recoverable state through the adapter.

The first implementation persists a checkpoint every configurable `X` frames. Historical checkpoints live on disk, not in RAM or VRAM. Memory only holds the active inference state and a bounded number of immutable snapshots waiting to be written. Once a snapshot is durable, its in-memory copy can be released.

Checkpoint cadence defines replay distance. A checkpoint written after frame `c` can resume at `c + 1`; correcting frame `x` restores the nearest checkpoint at or before `x`, seeks and replays the recorded video to `x`, applies the prompt, and writes subsequent state to a new branch. The original checkpoint chain remains unchanged.

Per-frame persistence is unnecessary for the first version and would multiply tensor serialization and disk I/O. Start with periodic checkpoints and choose `X` so the writer keeps up without blocking inference. Do not silently skip a promised checkpoint: if the queue cannot sustain the configured cadence, record the missed deadline and fail the performance requirement. Per-frame persistence can be evaluated later with the same interface.

## Change map

| Area | Change |
| --- | --- |
| [`sam2/utils/misc.py`](sam2/utils/misc.py#L172) | Route streaming sources through `FrameSource`; retain eager loading for the benchmark baseline. |
| [`sam2/sam2_video_predictor.py`](sam2/sam2_video_predictor.py#L44) | Store the frame source and typed state instead of a full video tensor. |
| [`sam2/sam2_video_predictor.py`](sam2/sam2_video_predictor.py#L879) | Consume the prefetched GPU slot and schedule the next frame without reopening the decoder. |
| [`sam2/sam2_video_predictor.py`](sam2/sam2_video_predictor.py#L912) | Enqueue checkpoint snapshots and apply state eviction through the adapter. |
| [`gradio_app.py`](gradio_app.py#L152) | Remove full-video preprocessing and temporary 96-frame chunk generation; pass the original source to the predictor. |
| `benchmarks/benchmark_video_streaming.py` | Compare eager and streaming paths and emit per-frame JSONL plus a summary. |
| `benchmarks/video_streaming_contract_checks.py` | Runtime checks for ordering, bounded buffers, replay after restore, correction on evicted frames, and output equivalence. No pytest dependency is required. |

## Measured baseline

The 2026-08-18 matrix used five distinct videos normalized to 200 frames at
960x540. Every condition processed 1,000 frames on the same RTX 5060 Laptop
GPU, Python 3.12, Torch 2.13, CUDA 13, EdgeTAM checkpoint, BF16 dtype, center
point prompt, and 100 ms memory sampling interval.

Compilation means only the repository's image-encoder `torch.compile` path.
It does not compile the full predictor. Each method used an empty Inductor
cache for video one and reused the disk cache for videos two through five.
Each video still ran in a fresh process, so the warm-cache results include
roughly 9.5-10.6 seconds of graph reconstruction/cache loading that a
long-lived predictor would not pay per video.

| Method | Compiled | GPU FPS | Propagation FPS | E2E FPS | GPU mean ms | Frame p95 ms | Torch peak GiB | NVML device peak GiB | RSS peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| core-eager | no | 19.30 | 18.64 | 14.40 | 51.82 | 66.06 | 2.81 | 3.36 | 6.07 |
| core-eager | yes | 26.64 | 25.57 | 9.06 | 37.54 | 43.44 | 2.56 | 3.28 | 6.23 |
| demo-chunked | no | 20.92 | 19.84 | 11.41 | 47.84 | 83.94 | 0.43 | 0.99 | 5.48 |
| demo-chunked | yes | 27.13 | 25.36 | 7.31 | 36.87 | 69.17 | 0.21 | 0.90 | 5.73 |

Observed compilation effects:

- Warm-cache core propagation improved by 35.9% and mean frame latency fell
  by 26.4%.
- Warm-cache chunked propagation improved by 28.0% and mean frame latency fell
  by 17.3%.
- A cold compile added about 105-109 seconds to the first `init_state` call.
- A fresh process with a warm disk cache still added about seven seconds to
  initialization. Compilation should therefore happen once on a long-lived
  predictor and be warmed before accepting work, not once per video.
- Compiled 200-frame E2E FPS was lower because compilation startup exceeded
  the propagation time saved. This does not contradict the steady propagation
  speedup.
- WSL returned zero for per-process NVML accounting. Use Torch allocation for
  process-attributable tensors and NVML device-used bytes for total device
  pressure.
- The original positional-encoding cache retained a CUDAGraph-owned tensor and
  made compiled video propagation fail on the second encoder invocation. The
  cache is now bypassed while Torch is compiling.
- During propagation, core-eager Torch allocation has a positive fitted slope
  of about 32.7 MiB per 100 frames uncompiled and 29.5 MiB per 100 frames
  compiled. Chunk resets reduce the observed slope to about 2.9 and 0.8 MiB
  per 100 frames respectively, but achieve that by discarding continuity.
- Exact binary-mask checksums differ between compiled and uncompiled execution,
  and between core and demo execution. A checksum mismatch only proves that at
  least one output bit differs. The demo also lossily re-encodes every chunk,
  so it does not consume identical pixels. Add a diagnostic-mask mode and
  report mask IoU, changed-pixel ratio, and logit max/mean error before making
  a correctness claim.

The complete matrix is in
`benchmark_artifacts/video_matrix/20260818T130314Z/`. The machine-readable
aggregate is `matrix_summary.json`; `measurements.json` retains every video,
and each run retains frame and memory JSONL.

### Decision after the baseline

Steps 2-5 remain the right decomposition, with these changes:

- Treat compilation as an orthogonal predictor configuration. Run every
  checkpoint against the same compiled/uncompiled state and report cold
  startup separately. Do not put compiler logic in `FrameSource`, GPU staging,
  or `VideoStateAdapter`.
- Keep one predictor alive across videos in compiled production measurements.
  A subprocess-per-video matrix is useful for startup analysis but understates
  long-lived compiled E2E throughput.
- Compare step 3 propagation against `core-eager` in the same compilation
  state. Compare step 5 output-producing E2E throughput against
  `demo-chunked` in the same compilation state.
- Use core-eager on the original decoded frames as the canonical correctness
  baseline. Demo-chunked is a performance/UI baseline only because its lossy
  temporary MP4 transcode changes model inputs and masks before chunk-boundary
  state differences are considered.
- Add a bounded output writer queue in step 5. Chunked GPU throughput is higher
  than core-eager uncompiled throughput, but its frame-wall p95 is 17.88 ms
  worse because reread, overlay, and MP4 encoding remain on the inference
  thread.
- Report Torch and NVML device memory together. Do not gate on the unsupported
  WSL per-process NVML value.

## Implementation checkpoints

Each checkpoint must run independently and leave a comparison artifact. Do not combine checkpoints when a failure would be ambiguous.

### 1. Capture the eager baseline

This checkpoint changes no loader, predictor, or Gradio behavior. It adds a frozen measurement harness for everything that follows.

Files to add or change:

| File | Direct change |
| --- | --- |
| `benchmarks/benchmark_video_streaming.py` | Typed CLI and orchestration for repeated baseline runs. |
| `benchmarks/video_benchmark_metrics.py` | Typed metric records, CUDA timing, NVML/RSS sampling, mask checksums, JSONL writing, and summary aggregation. |
| [`pyproject.toml`](pyproject.toml) | Define the benchmark dependency group, including `nvidia-ml-py`; resolve it through `uv.lock`. |
| `.gitignore` | Ignore generated `benchmark_artifacts/`; keep only explicitly selected baseline summaries or fixtures. |

Implement these pieces in order:

1. Define frozen dataclasses for `BenchmarkConfig`, `FrameTrace`, `MemorySample`, and `RunSummary`. Do not pass untyped metric dictionaries between modules.
2. Add CLI inputs for video path, model config, checkpoint, prompt JSON, device, dtype, maximum frames, warm-up frames, repeat count, chunk size, and output directory. Store the resolved arguments in every run artifact.
3. Add a deterministic prompt file contract containing `frame_idx`, `obj_id`, point coordinates, and labels. Reject prompts outside the selected frame range.
4. Implement `core-eager` mode. It calls the current `predictor.init_state(video_path)` and `propagate_in_video` path directly, without Gradio or output-video encoding.
5. Implement `demo-chunked` mode as a frozen copy of the current 96-frame preparation and mask-carry behavior. This baseline must remain available after `gradio_app.py` is changed later.
6. Separate model-load time, video preparation, `init_state`, prompt inference, propagation, mask postprocessing, and output encoding. Report core propagation FPS, pipeline FPS, and full demo FPS separately.
7. Measure total throughput with CUDA synchronization only at run boundaries. Use CUDA events for per-frame GPU timings and resolve them in fixed-size batches so timing does not create unbounded event state. Use the same trace batch size in every mode.
8. Sample Torch allocated/reserved/peak VRAM, NVML process VRAM when supported, NVML device-used VRAM, process RSS, and elapsed wall time from a dedicated sampler thread. Record model-loaded, post-init, first-frame, steady-state, and final values alongside the periodic samples.
9. Write `config.json`, `frames.jsonl`, `memory.jsonl`, and `summary.json` under one run directory. Include Python, Torch, CUDA, driver, GPU, model/config hash, video metadata, prompt hash, and repository revision.
10. Hash each binarized output mask with its frame and object ID. Save full masks only behind a diagnostic flag when a checksum differs.
11. Catch OOM and decode failures long enough to write a failed summary with the last frame and memory sample, then exit non-zero. An incomplete run must never be aggregated as a performance result.

Before marking an artifact valid, the benchmark verifies its own invariants: frame counts match the trace, timestamps are monotonic, durations are non-negative, requested warm-up frames are excluded from steady-state summaries, mask records match yielded frame/object IDs, and failed runs are excluded from aggregate results. This validates code owned by the benchmark without adding a test framework to the repository.

The runnable baseline matrix is:

| Mode | Input | Purpose |
| --- | --- | --- |
| `core-eager` | Short video that fits as one tensor | Fastest current model-level reference. |
| `demo-chunked` | Same short video, 96-frame chunks | Quantify current demo overhead. |
| `demo-chunked` | 10k-frame seekable video | Long-video reference for later streaming comparison. |

Pass when the requested repeats produce complete, internally valid artifacts and identical mask checksums. The report records observed FPS variation without imposing a threshold before baseline noise is known. It must show core, pipeline, and demo FPS separately and contain enough raw data to recalculate every summary value.

### 2. Introduce a bounded lazy `FrameSource`

Add the typed interface, first wrap the existing eager tensor without changing behavior, then add the sequential MP4 implementation with a bounded pinned-CPU queue. Keep GPU transfer and predictor state unchanged.

Pass when eager and lazy sources return the same frame count, metadata, ordering, normalized tensors, and random-seek results. The lazy source must process a 10k-frame input without exceeding its configured queue capacity or showing an RSS trend after warm-up.

This remains the next implementation step. The 200-frame eager run already
peaks around 6.07 GiB RSS and 3.36 GiB device-used VRAM, while chunking reduces
device pressure only by resetting continuity and re-encoding temporary files.
The lazy source must remove the length-dependent frame tensor without adopting
those chunk semantics.

### 3. Add current/next GPU staging

Add two fixed GPU frame slots, a dedicated transfer stream, CUDA readiness events, and next-frame scheduling. Keep all existing SAM2 output retention unchanged so this checkpoint isolates frame VRAM and overlap.

Pass when masks are numerically equivalent to core-eager on identical decoded frames using recorded IoU, changed-pixel ratio, and logit-error tolerances; allocated frame-buffer addresses are reused; VRAM shows no growth with video length beyond the variation measured in checkpoint 1; and there are zero prefetch misses after warm-up on the benchmark machine. Compare propagation FPS against `core-eager` with the same compilation state. Compilation must be warmed before the measured frames and its startup must be reported separately.

### 4. Bound SAM2 inference state for forward-only processing

Move retention and eviction behind `VideoStateAdapter`. Preserve conditioning frames and the model-selected active temporal-memory window, discard older recoverable outputs, and use a no-op checkpoint sink. Corrections on evicted frames remain unsupported in this checkpoint.

Pass when output masks remain equivalent to the unbounded run, VRAM and RSS both plateau on 10k and 100k frames, and the same streaming pipeline runs EdgeTAM plus one SAM2.1 configuration without model-specific logic in `FrameSource`. Record active-memory count and evictions per frame.

Before eviction, report frame-tensor bytes, retained output bytes, temporal
memory bytes, conditioning-state bytes, and their counts separately. The matrix
shows that bounding/resetting state can reduce Torch peak allocation from 2.81
GiB to 0.43 GiB, but chunk resets are not a correctness-preserving substitute
for an explicit SAM2 memory-window policy.

### 5. Replace demo chunking and run end to end

Remove temporary 96-frame MP4 generation from `gradio_app.py` and pass the original seekable source into the streaming predictor. Put frame reread/overlay/output-video encoding behind a bounded writer queue so it cannot block inference. Keep the eager path selectable for direct comparison.

Pass when upload, point prompting, full propagation, and output-video writing work on short and long videos; output frame count, FPS, dimensions, and object IDs match the UI baseline; masks are numerically equivalent to core-eager on identical decoded frames; and the long run meets the VRAM, RSS, throughput, and prefetch requirements below. Compare full-pipeline FPS against `demo-chunked` with the same compilation state, and report writer queue depth, writer lag, and inference-side enqueue time.

Periodic disk checkpoints, restore/replay, and branch creation start only after checkpoint 5 passes. Phase one includes the no-op sink boundary but no serialization or checkpoint thread.

## Measurements and acceptance

Run eager and streaming modes on the same model, checkpoint, prompts, video, resolution, dtype, device, and warm-up.

| Requirement | Measurement | Initial pass condition |
| --- | --- | --- |
| Stable VRAM | Per-frame `torch.cuda.memory_allocated`, `memory_reserved`, reset peak after warm-up, and NVML process VRAM | No positive trend with video length at fixed resolution/object count. Set a numerical tolerance only after checkpoint 1 establishes measurement noise. |
| Stable host memory | Process RSS, pinned-buffer bytes, frame/checkpoint queue depth | No upward trend after both bounded queues reach steady state. |
| Throughput | End-to-end frames/wall time and inference-only FPS | Streaming end-to-end FPS is at least the current 96-frame eager/chunk baseline. |
| Frame readiness | Decode time, H2D time, `prefetch_miss`, and inference wait for frame | Zero prefetch misses after warm-up on the benchmark machine; report the observed wait distribution. |
| Non-blocking checkpoints | Snapshot/enqueue time, writer latency, writer lag, queue depth, and missed cadence count | Queue remains bounded, every configured checkpoint reaches disk, and inference throughput does not regress against the no-checkpoint streaming baseline. |
| Correctness | Frame/object IDs, logits or binarized masks against eager mode | Identical binarized masks and numerically equivalent logits within a recorded tolerance. |
| Recovery | Restore checkpoint and replay to a known later frame | Same object IDs and masks as uninterrupted inference. |

Test at least short, 10k-frame, and 100k-frame inputs, with representative resolutions and object counts. Repeat a real clip to isolate video-length scaling, then run a genuinely long encoded video to include decoder behavior.

The benchmark artifact must include hardware/software versions, model/config hash, source metadata, checkpoint policy, buffer sizes, per-stage timings, VRAM/RSS samples, and correctness results. A summary without the per-frame trace is not enough to diagnose stalls or memory growth.

## First implementation scope

- Target seekable recorded videos only. Live streams stay out of scope until this path works and is measured.
- Persist checkpoints every configurable `X` frames. Maximum replay distance is `X - 1` frames.
- Support correction prompts on old frames by restoring from disk, seeking/replaying to the requested frame, and continuing in a new branch.
- Keep the original checkpoint chain immutable so a corrected branch can be discarded or compared with the original run.
- Start with one active branch per correction request. Deduplication, branch merging, and distributed checkpoint storage stay out of scope.
