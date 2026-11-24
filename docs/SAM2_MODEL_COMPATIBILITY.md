# Standard SAM 2 model compatibility check

Status: verified on the **tiny** Meta SAM 2 and SAM 2.1 video checkpoints for
finite files. This is a bounded compatibility result, not a claim about every
checkpoint, live input, or constant-size temporal state.

## What was tested

On 2026-10-03, the current streamSAM checkout (`1290460227a75d3ce6bb9e58ec4f912d2b8ef800`)
was compared with Meta's `facebookresearch/sam2` checkout
(`2b90b9f5ceec907a1c18123530e92e794ad901a4`). The machine used an RTX 5060
Laptop GPU (8 GiB), PyTorch `2.13.0+cu130`, and an NVIDIA 592.15 driver.

The source was the first 24 frames of `examples/01_dog.mp4` (1280 x 720), with
a point at `(480, 270)` on frame 0 for object 1. For the independent predictor
comparison, both checkouts read the *same extracted JPEG files*, offloaded
frames and state to CPU, and ran without compilation or optional
postprocessing. The official tiny checkpoints came from Meta's
[SAM 2 download links](https://github.com/facebookresearch/sam2#download-checkpoints):

| Model | Config | Checkpoint SHA-256 |
| --- | --- | --- |
| SAM 2 tiny | `configs/sam2/sam2_hiera_t.yaml` | `65b50056e05bcb13694174f51bb6da89c894b57b75ccdf0ba6352c597c5d1125` |
| SAM 2.1 tiny | `configs/sam2.1/sam2.1_hiera_t.yaml` | `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69` |

## Results

| Check | SAM 2 tiny | SAM 2.1 tiny |
| --- | --- | --- |
| Official checkpoint loads in streamSAM | Passed | Passed |
| 24 ordered masks from the lazy file source | Passed | Passed |
| streamSAM eager vs lazy, BF16 | 24/24 masks identical | 24/24 masks identical |
| streamSAM lazy vs Meta predictor, float32 | 24/24 masks identical; 0 differing pixels | 24/24 masks identical; 0 differing pixels |
| streamSAM lazy vs Meta predictor, BF16 | Mean IoU 0.9878; minimum 0.8126 at frame 22 | Mean IoU 0.9913; minimum 0.8559 at frame 14 |
| End-to-end MP4 decode, inference, output | 24/24 output frames | 24/24 output frames |

The float32 equality establishes the model and temporal computation path on
this clip. The BF16 difference is **between the two predictors**, not between
streamSAM's eager and lazy loaders. It disappears in float32; the exact
rounding/backend contribution has not been isolated. Do not claim bit-exact
BF16 parity with Meta. The single low-IoU BF16 frame in each model is material
enough to report rather than hide behind the high mean.

The end-to-end runs used the normal `demo-streaming` path with default
postprocessing, GPU frame staging, and bounded output writing on a 24-frame MP4
derived from the same example. Both wrote 24-frame 1280 x 720 MP4s, and the
decode queue reached its configured capacity of four without exceeding it.
Those runs establish that the complete file pipeline executes for both models;
they were **not** compared pixel-for-pixel with Meta's default-postprocessing
predictor. The measured 9.5–10.0 end-to-end FPS is a short-run observation,
not a model-family performance benchmark.

## Reproduce the predictor comparison

The optional [`verify_sam2_family.py`](../benchmarks/verify_sam2_family.py)
script runs either installed predictor with the same inputs, saves binary masks,
and compares them. Set `PYTHONPATH` to a separate Meta checkout for the
reference run. Run the script by its file path so the current working directory
does not select streamSAM's `sam2` package accidentally; the printed
`sam2_package` path confirms which implementation ran.

```bash
mkdir -p /tmp/streamsam-compat-frames
ffmpeg -hide_banner -loglevel error -y -i examples/01_dog.mp4 \
  -frames:v 24 -start_number 0 /tmp/streamsam-compat-frames/%05d.jpg

PYTHONPATH=/tmp/streamsam-upstream-sam2 .venv/bin/python3 \
  benchmarks/verify_sam2_family.py \
  --config configs/sam2.1/sam2.1_hiera_t.yaml \
  --checkpoint /tmp/streamsam-sam2.1-hiera-tiny.pt \
  --frames /tmp/streamsam-compat-frames --point 480 270 \
  --output /tmp/sam21-reference.npz --dtype float32

.venv/bin/python3 benchmarks/verify_sam2_family.py \
  --config configs/sam2.1/sam2.1_hiera_t.yaml \
  --checkpoint /tmp/streamsam-sam2.1-hiera-tiny.pt \
  --frames /tmp/streamsam-compat-frames --point 480 270 \
  --frame-loading lazy --output /tmp/sam21-streamsam.npz \
  --dtype float32 --compare-to /tmp/sam21-reference.npz --require-exact
```

The original measured probe extracted JPEGs from a 24-frame MP4 intermediary
made with FFmpeg 4.4.2. Re-extracting directly from the bundled source as shown
above makes a new but paired input fixture; rerun both sides together. The
bundled source SHA-256 was
`21425eb9b7fce7d3fb23fa17b1c07bcbe99ba9e64e3b5347d8052ba6d9c38924`.

## What remains unverified

- Small, base-plus, and large SAM 2 / SAM 2.1 checkpoints; multiple objects;
  prompts on later frames; mask correction; reverse propagation; compilation;
  and longer clips have not been compared with Meta here. This is the scope of
  the comparison, not a requirement to soak-test every checkpoint. Once a
  checkpoint runs through the shared predictor path, long-running behavior can
  be validated on a representative model, with targeted checks for any
  configuration that changes the retention window.
- File decoding and output queues are bounded, but per-frame predictor output
  records still accumulate. This check does not establish constant memory for
  arbitrarily long video.
- The original image-only SAM model does not expose SAM 2 video memory and is
  outside this predictor's compatibility claim.
- Live RTSP/camera input requires the separate lifecycle and retention work in
  [`LIVE_CAMERA_RTSP_PROPOSAL.md`](LIVE_CAMERA_RTSP_PROPOSAL.md).
