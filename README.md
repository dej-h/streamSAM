# streamSAM

Bounded video streaming for the SAM2 model family.



https://github.com/user-attachments/assets/887fefa0-2245-440e-8012-15dd3cb26fb5



streamSAM keeps video decode, CPU queues, GPU staging, temporal inference, and
output writing bounded as a video grows. It retains the existing `sam2` Python
API and puts the streaming work around the predictor instead of introducing a
second model API.

EdgeTAM is the currently verified model. The adapter boundaries are intended to
support other SAM2-family models, but compatibility with those models has not
been validated yet.

## What it adds

- Lazy frame sources with bounded decode queues and backpressure.
- Reusable pinned-memory and GPU staging slots with explicit CUDA ownership.
- Optional one-frame-ahead image feature production.
- A predict-then-commit API for changing a mask before it enters temporal memory.
- Bounded asynchronous video output, GPU timings, NVTX ranges, and Perfetto traces.
- Contract checks and matched benchmark tooling for throughput, memory, and mask
  fidelity.

## Install

streamSAM requires Python 3.10 or newer, PyTorch 2.3.1 or newer, and a CUDA-capable
machine for the measured streaming path. Install PyTorch for your CUDA version
first, then install the repository:

```bash
git clone https://github.com/dej-h/streamSAM.git
cd streamSAM
python3 -m pip install -e .
```

The repository includes the verified EdgeTAM configuration and checkpoint. The
EdgeTAM backbone may download its pretrained TIMM weights on first use.

For the local Gradio demo:

```bash
python3 -m pip install -e ".[gradio]"
python3 gradio_app.py
```

## Reproduce the benchmark

Install the locked benchmark environment with
[uv](https://docs.astral.sh/uv/):

```bash
uv sync --group benchmark
```

Run a 200-frame end-to-end comparison with the bundled example:

```bash
uv run --group benchmark python3 -m benchmarks.run_benchmark_demo \
  --video examples/01_dog.mp4 \
  --prompt examples/prompts/01_dog.json \
  --max-frames 200 \
  --original-rss-safety-limit-gib 4.5 \
  --playback-speed 2
```

The runner executes the original eager loader, independent 96-frame batches,
and streamSAM with the same source, checkpoint, prompt, dtype, and frame limit.
It writes raw frame and memory telemetry, subprocess logs, summaries, and a
synchronized replay under `benchmark_artifacts/demo_comparison/`.

To repeat the published workload, use an input with at least 1,000 frames and
change `--max-frames` to `1000`. The bundled dog clip has 289 frames. The table
below came from a separate 1,000-frame source, so the repository reproduces the
benchmark method but does not ship the exact source video needed to reproduce
the same numbers byte for byte.

One matched local run on an NVIDIA GeForce RTX 5060 Laptop GPU with BF16 and no
compilation produced:

| Execution strategy | Result | End-to-end FPS | Peak process RSS |
| --- | ---: | ---: | ---: |
| Original eager loading | Safety stop before frame 1 | n/a | 4.5 GiB limit |
| Independent 96-frame batches | 1,000 / 1,000 frames | 17.46 | 4.40 GiB |
| streamSAM | 1,000 / 1,000 frames | **24.49** | **2.52 GiB** |

The eager run was stopped at the configured host RSS limit before it could
materialize the complete input tensor. This was a recorded safety stop, not a
CUDA out-of-memory result. The numbers above describe this one matched run, not
a cross-hardware performance claim.

The implementation and lower-level profiling notes are in
[`docs/SAM2_GPU_STREAMING_PIPELINE.md`](docs/SAM2_GPU_STREAMING_PIPELINE.md).

## Attribution

streamSAM is derived from Meta's
[EdgeTAM](https://github.com/facebookresearch/EdgeTAM), which is based on
[SAM 2](https://github.com/facebookresearch/sam2). The inherited code, model,
configuration, checkpoint, copyright notices, and Git history remain attributed
to their original authors.

The original EdgeTAM authors are Chong Zhou, Chenchen Zhu, Yunyang Xiong,
Saksham Suri, Fanyi Xiao, Lemeng Wu, Raghuraman Krishnamoorthi, Bo Dai,
Chen Change Loy, Vikas Chandra, and Bilge Soran. streamSAM is independently
maintained and is not affiliated with or endorsed by Meta.

If you use the EdgeTAM model or checkpoint in research, cite the original work:

```bibtex
@article{zhou2025edgetam,
  title={EdgeTAM: On-Device Track Anything Model},
  author={Zhou, Chong and Zhu, Chenchen and Xiong, Yunyang and Suri, Saksham and Xiao, Fanyi and Wu, Lemeng and Krishnamoorthi, Raghuraman and Dai, Bo and Loy, Chen Change and Chandra, Vikas and Soran, Bilge},
  journal={arXiv preprint arXiv:2501.07256},
  year={2025}
}
```

## License

streamSAM and the inherited EdgeTAM code are licensed under the
[Apache License 2.0](LICENSE).
