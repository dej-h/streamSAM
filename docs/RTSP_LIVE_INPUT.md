# RTSP live input

streamSAM reads frames from an RTSP camera and tracks one prompted object as
frames arrive. The camera reader supplies decoded frames to the live session.
The session uses `SAM2VideoPredictor.add_new_points_or_box`,
`propagate_in_video_predictions`, and `commit_video_frame`. This path targets
SAM 2-family video predictors, including the bundled EdgeTAM checkpoint.

## Run it

Install the project with `uv sync`, then run the example on a CUDA machine:

```bash
uv run python3 examples/rtsp_camera.py \
  --url 'rtsp://camera.example/live' \
  --point 480 270 \
  --mask-directory camera-masks
```

The point uses the camera frame's original pixel coordinates. The example
prints one JSON record per processed frame. `--mask-directory` is optional;
when set, it writes one PNG per frame under an epoch directory. Use
`--max-frames N` to stop after N processed frames. `--config` and
`--checkpoint` select another compatible SAM 2-family model.

The example accepts `--reconnect-attempts N`. On a disconnect, read timeout, or
resolution change, it closes the current session, opens a new reader, and
prompts the first frame of a new epoch from the last nonempty segmentation.
The initial `--point` is used only before any frames have been processed. It
does not carry temporal state across a gap. Retries use a capped delay of 1,
2, 4, then 8 seconds. The default is zero retries.

Reconnect prompts contain up to 64 positive points; `--reconnect-points N`
sets that budget. Points are drawn from the interior of each connected mask
region, using distance from its boundary rather than a geometric centroid.
This keeps points inside masks with holes and preserves coverage of thin,
disconnected fragments. Larger regions share the remaining points in
proportion to their area. If regions outnumber the budget, the largest are
represented. The mask is resized with nearest-neighbor interpolation when
the new frame reports different dimensions.

The last mask supplies a spatial estimate to SAM's mask decoder, which then
refines it using the interior points and the new frame. This shape context
keeps dense positive point sets from overwhelming the sparse prompt decoder.

The points describe the last nonempty mask's location. They do not estimate
motion during the gap or independently identify an object at a new location.
After processing starts, reconnect requires a previous foreground mask.
Connection failures before the first processed frame retry the initial prompt.

The example and `RtspFrameReader` use OpenCV's connection-open and decoder-read
timeouts. No timeout parameters are passed to the backend unless explicitly
requested with `--open-timeout-seconds` or `--read-timeout-seconds`. An override
must fit in positive OpenCV milliseconds. OpenCV
4.10's FFmpeg backend defaults both to
[30 seconds](https://github.com/opencv/opencv/blob/4.10.0/modules/videoio/src/cap_ffmpeg_impl.hpp#L242-L243).
`reader.read()` waits for a frame or for the decoder to stop; applications can
set a separate caller deadline with `reader.read(timeout_seconds=...)`.
Closing the reader waits for an in-progress decoder operation to finish under
the backend's timeout policy. Streams with long keyframe intervals may need a
larger timeout when joining between keyframes.

For an application that supplies its own prompt or mask consumer, pass frames
from the RTSP reader to the live session.

```python
from sam2.live_video_session import LiveVideoSession
from sam2.utils.rtsp_stream import RtspFrameReader

with RtspFrameReader(camera_url) as reader:
    first = reader.read()
    with LiveVideoSession(predictor, first) as session:
        result = session.prompt_point(480, 270)
        consume(result.masks, result.object_ids)
        while True:
            result = session.track(reader.read())
            consume(result.masks, result.object_ids)
```

`prompt_box((x0, y0, x1, y1))` and `prompt_points(points)` are also available
for the first frame. `points` is an `(N, 2)` float32 array of positive pixel
coordinates for one object. The session needs a predictor built with
`build_sam2_video_predictor`. Pass `mask=previous_mask` to `prompt_points` to
refine a full-resolution boolean mask with the new frame and point prompts.
The example uses CUDA, which is the path exercised here. The caller owns
`torch.inference_mode()` and its desired autocast context, as shown in the
example script.

## How frames move

`RtspFrameReader` decodes on a background thread through OpenCV's FFmpeg
backend. It keeps at most four decoded BGR frames by default. Each `read()`
returns the newest frame and drops older queued frames, so a slow predictor
does not build an ever-growing input delay. `RtspReaderStats` exposes decoded
and locally dropped frame counts and maximum queue depth. It cannot count
camera-side or network losses.

`LiveVideoFrameSource` applies the same resize, RGB conversion, and
normalization as finite MP4 input. It exposes only the latest accepted frame
through the predictor's indexed source interface. The session assigns a
contiguous processed index even when decoded frames are dropped. After each
commit, it keeps the initial conditioning output and the recent outputs that
SAM 2 memory attention or object pointers can still use. It removes older
non-conditioning outputs and their per-object views and tracking metadata.

`LiveMaskResult` contains CPU boolean masks, processed index, decoded sequence,
source presentation time when OpenCV exposes it, and time from local frame
receipt to mask delivery. The tested OpenCV 4.10 build does not expose
`CAP_PROP_PTS`, so presentation time is `None` with that build. The latency
value excludes camera capture and network transport.

## Current scope and checks

The example tracks one object from a first-frame point. The Python session
also accepts a first-frame box or multiple positive points, optionally refining
a mask estimate. It emits mask arrays or optional PNG files;
it does not publish a segmented RTSP stream. Live sessions are forward-only.
Adding objects later, corrections to earlier frames, and state transfer across
camera gaps are outside this first path. The finite-video API still handles
seeking and reverse propagation. The resolution check uses the decoded array
dimensions. Some OpenCV/FFmpeg builds keep returning the initial array size
after a native resolution change. Use fixed-resolution feeds with those builds.

The contract checks compare 200 live-fed EdgeTAM masks with finite-video masks
in FP32, with exact equality. Retained outputs stayed at 16. A 48-frame replay
with memory stride 4 also matched exactly and retained at most 22 outputs.
The reader check verifies bounded queue depth, drop accounting, and disconnect
reporting against a prerecorded video. A local MediaMTX RTSP replay produced
24 masks and PNGs while the decoder dropped stale frames. A controlled
publisher stop and restart produced 26 masks across two epochs, with the second
epoch starting at processed index zero. These checks do not establish an
indefinite memory plateau, tracking quality after large camera gaps, or
performance across other checkpoints. The longer soak and reliability gates
remain in the [original proposal](https://github.com/dej-h/streamSAM/blob/89887ad19444eee3b20d652c20ba9d541f156177/docs/LIVE_CAMERA_RTSP_PROPOSAL.md).

The mask-prompt checks cover holes, concavities, thin disconnected fragments,
image boundaries, empty masks, and a bounded point budget. The CUDA reconnect
check moves the object before starting a fresh session and compares the
mask-derived prompt with the original click. A local moving-scene RTSP replay
also exercised the camera example through a disconnect after the original
click had fallen outside the tracked object.

Run the local checks from the repository root:

```bash
uv run python3 -m benchmarks.tests.rtsp_reader_contract_checks \
  --video examples/01_dog.mp4
uv run python3 -m benchmarks.tests.live_video_contract_checks \
  --video examples/01_dog.mp4 --frame-count 200
uv run python3 -m benchmarks.tests.live_video_contract_checks \
  --video examples/01_dog.mp4 --frame-count 48 --memory-stride 4
uv run python3 -m benchmarks.tests.mask_prompt_contract_checks
uv run python3 -m benchmarks.tests.live_reconnect_contract_checks
```
