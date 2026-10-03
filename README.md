# streamSAM

Track longer videos with SAM 2 without loading every frame into memory first.

https://github.com/user-attachments/assets/887fefa0-2245-440e-8012-15dd3cb26fb5

## What it enables

streamSAM processes finite video files frame by frame. It avoids preloading the
whole file, bounds decode, CPU queues, resizing, GPU staging, and output writing,
and carries SAM 2 temporal state between frames.

In a 1,000-frame benchmark with EdgeTAM (a SAM 2 variant optimized for on-device
video segmentation), streamSAM ran faster than independent 96-frame batches
while using less host memory. The standard eager loader reached the benchmark's
4.5 GiB memory safety limit before inference started.

The predictor still retains per-frame results, so its state can grow as a video
gets longer. streamSAM keeps the existing `sam2` API and checkpoint format.
Existing SAM 2 code can opt into lazy loading with `frame_loading="lazy"`.

streamSAM supports Meta's SAM 2 model family. It has been tested with EdgeTAM,
Meta SAM 2 tiny, and SAM 2.1 tiny. Other SAM 2 models use the same predictor
framework and are expected to work, but have not been directly tested.

## How it is built

A lazy frame source feeds a bounded decode queue with backpressure. Frames pass
through reusable pinned CPU memory and GPU staging slots with explicit ownership.
Optional one-frame-ahead image feature production overlaps work between frames.
Predict-then-commit lets a mask be changed before it enters temporal memory, and
an asynchronous writer bounds video output.

## Try it locally

The repository includes `edgetam.yaml`, an EdgeTAM checkpoint, and example
videos, so you can try the demo before adding streamSAM to another project.
Use Python 3.10 or newer and a CUDA-capable machine for the measured streaming
path. [uv](https://docs.astral.sh/uv/) installs from the committed lockfile:

```bash
git clone https://github.com/dej-h/streamSAM.git
cd streamSAM
uv sync --extra gradio
uv run --extra gradio python3 gradio_app.py
```

Upload a video or choose an example, mark the object with an include point, then
click **Track**. The EdgeTAM backbone may download pretrained TIMM weights on
first use.

If you manage your own Python and PyTorch environment, pip installation also
works through `pyproject.toml`:

```bash
python3 -m pip install -e ".[gradio]"
python3 gradio_app.py
```

## Run the benchmark

The bundled comparison runs eager loading, independent batches, and streamSAM
on the same video and prompt:

```bash
uv run --group benchmark python3 -m benchmarks.video.run_benchmark_demo \
  --video examples/01_dog.mp4 \
  --prompt examples/prompts/01_dog.json \
  --max-frames 200 \
  --original-rss-safety-limit-gib 4.5
```

The [benchmark guide](benchmarks/README.md) has the measured results, workload
details, and other checks.

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
