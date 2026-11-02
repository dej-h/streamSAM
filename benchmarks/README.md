# Benchmark tools

Run these commands from the repository root after `uv sync --group benchmark`.
Generated measurements and replays go under the ignored `benchmark_artifacts/`
directory.

## Measurement

- `benchmark_video_streaming.py` measures individual eager, lazy, chunked, or
  streaming runs and writes frame timings, memory samples, and summaries.
- `video_benchmark_metrics.py` provides shared metric records and collection.
- `run_video_benchmark_matrix.py` compares eager and chunked execution with and
  without image-encoder compilation across five bundled videos. It does not
  measure the streaming mode; use the individual runner or the demo below for
  that comparison.

Inspect the measurement options:

```bash
uv run --group benchmark python3 -m benchmarks.benchmark_video_streaming --help
uv run --group benchmark python3 -m benchmarks.run_video_benchmark_matrix --help
```

Inference measurements require CUDA and the EdgeTAM checkpoint.

## Video creation

`video/run_benchmark_demo.py` runs matched eager, chunked, and streaming
measurements in subprocesses, then creates the comparison replay:

```bash
uv run --group benchmark python3 -m benchmarks.video.run_benchmark_demo \
  --video examples/01_dog.mp4 \
  --prompt examples/prompts/01_dog.json \
  --max-frames 200 \
  --original-rss-safety-limit-gib 4.5 \
  --playback-speed 2
```

`video/render_benchmark_comparison.py` renders existing matched run directories
without running inference. Pass `--original-run`, `--chunked-run`,
`--streaming-run`, and `--output`; use `--help` for layout and playback options.
Rendering requires CPU video decoding and encoding, but no CUDA or checkpoint.
Playback speed changes the replay, not the recorded measurements.

## Published comparison

To repeat the published workload, use an input with at least 1,000 frames and
change `--max-frames` to `1000`. The bundled dog clip has 289 frames. The table
below came from a separate 1,000-frame source, so the repository reproduces the
benchmark method but does not ship the exact source video needed to reproduce
the same numbers byte for byte.

Benchmark setup: NVIDIA GeForce RTX 5060 Laptop GPU, BF16, no compilation.

| Execution strategy | Result | End-to-end FPS | Peak process RSS |
| --- | ---: | ---: | ---: |
| Original eager loading | Safety stop before frame 1 | n/a | 4.5 GiB limit |
| Independent 96-frame batches | 1,000 / 1,000 frames | 17.46 | 4.40 GiB |
| streamSAM | 1,000 / 1,000 frames | **24.49** | **2.52 GiB** |

The eager run was stopped at the configured host RSS limit before it could
materialize the complete input tensor. This was a recorded safety stop, not a
CUDA out-of-memory result.

The implementation and lower-level profiling notes are in
[`SAM2_GPU_STREAMING_PIPELINE.md`](../docs/SAM2_GPU_STREAMING_PIPELINE.md).

## Contract checks

The checks in `tests/` are executable modules. They raise an error on failure.

CPU checks cover frame-source ordering and bounds, and replay rendering:

```bash
uv run --group benchmark python3 -m benchmarks.tests.video_streaming_contract_checks \
  --video examples/01_dog.mp4
uv run --group benchmark python3 -m benchmarks.tests.benchmark_demo_contract_checks
```

CUDA checks cover staging and one-frame-ahead image feature production:

```bash
uv run --group benchmark python3 -m benchmarks.tests.gpu_frame_staging_contract_checks \
  --video examples/01_dog.mp4
uv run --group benchmark python3 -m benchmarks.tests.gpu_image_feature_pipeline_contract_checks \
  --video examples/01_dog.mp4
```

The CUDA checks use decoded frames and synthetic feature operations; they do not
require a model checkpoint. Use a video with at least eight frames for the CPU
frame-source check.

## Live RTSP checks

The reader check uses a prerecorded file as its decoder fixture. The live
predictor check needs CUDA and the bundled EdgeTAM checkpoint. It compares
one-frame-at-a-time input with finite-video masks and checks state retention:

```bash
uv run --group benchmark python3 -m benchmarks.tests.rtsp_reader_contract_checks \
  --video examples/01_dog.mp4
uv run --group benchmark python3 -m benchmarks.tests.live_video_contract_checks \
  --video examples/01_dog.mp4 --frame-count 200
uv run --group benchmark python3 -m benchmarks.tests.live_video_contract_checks \
  --video examples/01_dog.mp4 --frame-count 48 --memory-stride 4
```

The [RTSP live input guide](../docs/RTSP_LIVE_INPUT.md) has the camera example
and current scope.
