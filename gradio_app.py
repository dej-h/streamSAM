# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
from datetime import datetime

import gradio as gr

APP_DIR = os.path.dirname(os.path.abspath(__file__))

os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "0,1,2,3,4,5,6,7"
import tempfile

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
from sam2.utils.video_output_writer import BoundedVideoMaskWriter

# Description
title = "<center><strong><font size='8'>EdgeTAM<font></strong> <a href='https://github.com/facebookresearch/EdgeTAM'><font size='6'>[GitHub]</font></a> </center>"

description_p = """# Instructions
                <ol>
                <li> Upload one video or click one example video</li>
                <li> Click 'include' point type, select the object to segment and track</li>
                <li> Click 'exclude' point type (optional), select the area you want to avoid segmenting and tracking</li>
                <li> Click the 'Track' button to obtain the masked video </li>
                </ol>
              """

# examples
examples = [
    ["examples/01_dog.mp4"],
    ["examples/02_cups.mp4"],
    ["examples/03_blocks.mp4"],
    ["examples/04_coffee.mp4"],
    ["examples/05_default_juggle.mp4"],
    ["examples/01_breakdancer.mp4"],
    ["examples/02_hummingbird.mp4"],
    ["examples/03_skateboarder.mp4"],
    ["examples/04_octopus.mp4"],
    ["examples/05_landing_dog_soccer.mp4"],
    ["examples/06_pingpong.mp4"],
    ["examples/07_snowboarder.mp4"],
    ["examples/08_driving.mp4"],
    ["examples/09_birdcartoon.mp4"],
    ["examples/10_cloth_magic.mp4"],
    ["examples/11_polevault.mp4"],
    ["examples/12_hideandseek.mp4"],
    ["examples/13_butterfly.mp4"],
    ["examples/14_social_dog_training.mp4"],
    ["examples/15_cricket.mp4"],
    ["examples/16_robotarm.mp4"],
    ["examples/17_childrendancing.mp4"],
    ["examples/18_threedogs.mp4"],
    ["examples/19_cyclist.mp4"],
    ["examples/20_doughkneading.mp4"],
    ["examples/21_biker.mp4"],
    ["examples/22_dogskateboarder.mp4"],
    ["examples/23_racecar.mp4"],
    ["examples/24_clownfish.mp4"],
]
examples = [[os.path.join(APP_DIR, sample[0])] for sample in examples]

OBJ_ID = 0
FRAME_BUFFER_SIZE = 4
GPU_STAGING_SLOTS = 2
OUTPUT_QUEUE_SIZE = 4

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
sam2_checkpoint = "checkpoints/edgetam.pt"
model_cfg = "edgetam.yaml"
sam2_checkpoint = os.path.join(APP_DIR, sam2_checkpoint)
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=DEVICE)
print("PREDICTOR LOADED")

# use bfloat16 for the entire notebook
if torch.cuda.is_available():
    torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def get_video_fps(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    # Get the FPS of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    return fps


def cleanup_temp_chunks(session_state):
    """Close bounded video resources retained by the current demo session."""
    inference_state = session_state.get("inference_state")
    if inference_state is not None:
        predictor.close_video_source(inference_state)
    session_state["video_path"] = None
    session_state["video_fps"] = None
    session_state["video_frame_count"] = None
    session_state["video_frame_size"] = None
    session_state["last_output_writer_stats"] = None


def reset(session_state):
    session_state["input_points"] = []
    session_state["input_labels"] = []
    if session_state["inference_state"] is not None:
        predictor.close_video_source(session_state["inference_state"])
        predictor.reset_state(session_state["inference_state"])
    cleanup_temp_chunks(session_state)
    session_state["first_frame"] = None
    session_state["inference_state"] = None
    return (
        None,
        gr.update(open=True),
        None,
        None,
        gr.update(value=None, visible=False),
        session_state,
    )


def clear_points(session_state):
    session_state["input_points"] = []
    session_state["input_labels"] = []
    if session_state["inference_state"]["tracking_has_started"]:
        predictor.reset_state(session_state["inference_state"])
    return (
        session_state["first_frame"],
        None,
        gr.update(value=None, visible=False),
        session_state,
    )


def preprocess_video_in(video_path, session_state):
    if video_path is None:
        return (
            gr.update(open=True),
            None,
            None,
            gr.update(value=None, visible=False),
            session_state,
        )

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise gr.Error("Could not open the input video.")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 30.0
        reported_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, first_frame_bgr = capture.read()
    finally:
        capture.release()
    if not ok:
        raise gr.Error("The input video contains no decodable frames.")

    previous_state = session_state.get("inference_state")
    if previous_state is not None:
        predictor.close_video_source(previous_state)
        predictor.reset_state(previous_state)
    cleanup_temp_chunks(session_state)

    inference_state = predictor.init_state(
        video_path=video_path,
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
        frame_loading="lazy",
        frame_buffer_size=FRAME_BUFFER_SIZE,
        enable_gpu_frame_staging=DEVICE == "cuda",
        gpu_frame_staging_slots=GPU_STAGING_SLOTS,
    )
    metadata = inference_state["frame_source"].metadata
    if reported_frame_count > 0 and reported_frame_count != metadata.frame_count:
        predictor.close_video_source(inference_state)
        raise gr.Error(
            "Video frame-count metadata changed while initializing the stream: "
            f"OpenCV reported {reported_frame_count}, EdgeTAM reported "
            f"{metadata.frame_count}."
        )

    first_frame = cv2.cvtColor(first_frame_bgr, cv2.COLOR_BGR2RGB)
    session_state["first_frame"] = copy.deepcopy(first_frame)
    session_state["video_path"] = video_path
    session_state["video_fps"] = fps
    session_state["video_frame_count"] = metadata.frame_count
    session_state["video_frame_size"] = (
        metadata.video_width,
        metadata.video_height,
    )
    session_state["inference_state"] = inference_state
    session_state["input_points"] = []
    session_state["input_labels"] = []

    print(
        "Prepared one continuous lazy video state: "
        f"{metadata.frame_count} frames, decoder capacity "
        f"{FRAME_BUFFER_SIZE}, GPU slots {GPU_STAGING_SLOTS}."
    )
    return [
        gr.update(open=False),
        gr.update(open=False),
        first_frame,
        None,
        gr.update(value=None, visible=False),
        session_state,
    ]


def segment_with_points(
    point_type,
    session_state,
    evt: gr.SelectData,
):
    session_state["input_points"].append(evt.index)
    print(f"TRACKING INPUT POINT: {session_state['input_points']}")

    if point_type == "include":
        session_state["input_labels"].append(1)
    elif point_type == "exclude":
        session_state["input_labels"].append(0)
    print(f"TRACKING INPUT LABEL: {session_state['input_labels']}")

    # Open the image and get its dimensions
    transparent_background = Image.fromarray(session_state["first_frame"]).convert(
        "RGBA"
    )
    w, h = transparent_background.size

    # Define the circle radius as a fraction of the smaller dimension
    fraction = 0.01  # You can adjust this value as needed
    radius = int(fraction * min(w, h))

    # Create a transparent layer to draw on
    transparent_layer = np.zeros((h, w, 4), dtype=np.uint8)

    for index, track in enumerate(session_state["input_points"]):
        if session_state["input_labels"][index] == 1:
            cv2.circle(transparent_layer, track, radius, (0, 255, 0, 255), -1)
        else:
            cv2.circle(transparent_layer, track, radius, (255, 0, 0, 255), -1)

    # Convert the transparent layer back to an image
    transparent_layer = Image.fromarray(transparent_layer, "RGBA")
    selected_point_map = Image.alpha_composite(
        transparent_background, transparent_layer
    )

    # Let's add a positive click at (x, y) = (210, 350) to get started
    points = np.array(session_state["input_points"], dtype=np.float32)
    # for labels, `1` means positive click and `0` means negative click
    labels = np.array(session_state["input_labels"], np.int32)
    _, _, out_mask_logits = predictor.add_new_points(
        inference_state=session_state["inference_state"],
        frame_idx=0,
        obj_id=OBJ_ID,
        points=points,
        labels=labels,
    )

    mask_image = show_mask((out_mask_logits[0] > 0.0).cpu().numpy())
    first_frame_output = Image.alpha_composite(transparent_background, mask_image)

    torch.cuda.empty_cache()
    return selected_point_map, first_frame_output, session_state


def show_mask(mask, obj_id=None, random_color=False, convert_to_image=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    mask = (mask * 255).astype(np.uint8)
    if convert_to_image:
        mask = Image.fromarray(mask, "RGBA")
    return mask


def propagate_to_all(video_in, session_state):
    inference_state = session_state.get("inference_state")
    if (
        len(session_state["input_points"]) == 0
        or video_in is None
        or inference_state is None
    ):
        return None, session_state

    video_path = session_state.get("video_path")
    frame_count = session_state.get("video_frame_count")
    frame_size = session_state.get("video_frame_size")
    fps = session_state.get("video_fps")
    if video_path is None or frame_count is None or frame_size is None:
        raise gr.Error("The video stream metadata is incomplete; reload the video.")
    if fps is None or fps <= 0:
        fps = 30.0

    unique_id = datetime.now().strftime("%Y%m%d%H%M%S")
    output_path = os.path.join(
        tempfile.gettempdir(), f"output_video_{unique_id}.mp4"
    )
    writer = BoundedVideoMaskWriter(
        input_path=video_path,
        output_path=output_path,
        fps=fps,
        frame_size=frame_size,
        expected_frame_count=frame_count,
        expected_object_ids=(OBJ_ID,),
        capacity=OUTPUT_QUEUE_SIZE,
    )

    try:
        for frame_idx, object_ids, mask_logits in predictor.propagate_in_video(
            inference_state
        ):
            masks = (mask_logits > 0.0).detach().cpu().numpy()
            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks[:, 0]
            writer.submit(frame_idx, tuple(object_ids), masks)
        writer_stats = writer.close()
    except BaseException:
        writer.abort()
        raise

    session_state["last_output_writer_stats"] = writer_stats
    print(
        "continuous propagation complete: "
        f"{writer_stats.written_frames} frames, output queue "
        f"{writer_stats.maximum_depth}/{writer_stats.capacity}"
    )
    return gr.update(value=output_path, visible=True), session_state


def update_ui():
    return gr.update(visible=True)


with gr.Blocks() as demo:
    session_state = gr.State(
        {
            "first_frame": None,
            "video_path": None,
            "video_fps": None,
            "video_frame_count": None,
            "video_frame_size": None,
            "last_output_writer_stats": None,
            "input_points": [],
            "input_labels": [],
            "inference_state": None,
        }
    )

    with gr.Column():
        # Title
        gr.Markdown(title)
        with gr.Row():

            with gr.Column():
                # Instructions
                gr.Markdown(description_p)

                with gr.Accordion("Input Video", open=True) as video_in_drawer:
                    video_in = gr.Video(label="Input Video", format="mp4")

                with gr.Row():
                    point_type = gr.Radio(
                        label="point type",
                        choices=["include", "exclude"],
                        value="include",
                        scale=2,
                    )
                    propagate_btn = gr.Button("Track", scale=1, variant="primary")
                    clear_points_btn = gr.Button("Clear Points", scale=1)
                    reset_btn = gr.Button("Reset", scale=1)

                points_map = gr.Image(
                    label="Frame with Point Prompt", type="numpy", interactive=False
                )

            with gr.Column():
                gr.Markdown("# Try some of the examples below ⬇️")
                gr.Examples(
                    examples=examples,
                    inputs=[
                        video_in,
                    ],
                    examples_per_page=8,
                )
                gr.Markdown("\n\n\n\n\n\n\n\n\n\n\n")
                gr.Markdown("\n\n\n\n\n\n\n\n\n\n\n")
                gr.Markdown("\n\n\n\n\n\n\n\n\n\n\n")
                output_image = gr.Image(label="Reference Mask")

                output_video = gr.Video(visible=False)

    # When new video is uploaded
    video_in.upload(
        fn=preprocess_video_in,
        inputs=[
            video_in,
            session_state,
        ],
        outputs=[
            video_in_drawer,  # Accordion to hide uploaded video player
            points_map,  # Image component where we add new tracking points
            output_image,
            output_video,
            session_state,
        ],
        queue=False,
    )

    video_in.change(
        fn=preprocess_video_in,
        inputs=[
            video_in,
            session_state,
        ],
        outputs=[
            video_in_drawer,  # Accordion to hide uploaded video player
            points_map,  # Image component where we add new tracking points
            output_image,
            output_video,
            session_state,
        ],
        queue=False,
    )

    # triggered when we click on image to add new points
    points_map.select(
        fn=segment_with_points,
        inputs=[
            point_type,  # "include" or "exclude"
            session_state,
        ],
        outputs=[
            points_map,  # updated image with points
            output_image,
            session_state,
        ],
        queue=False,
    )

    # Clear every points clicked and added to the map
    clear_points_btn.click(
        fn=clear_points,
        inputs=session_state,
        outputs=[
            points_map,
            output_image,
            output_video,
            session_state,
        ],
        queue=False,
    )

    reset_btn.click(
        fn=reset,
        inputs=session_state,
        outputs=[
            video_in,
            video_in_drawer,
            points_map,
            output_image,
            output_video,
            session_state,
        ],
        queue=False,
    )

    propagate_btn.click(
        fn=update_ui,
        inputs=[],
        outputs=output_video,
        queue=False,
    ).then(
        fn=propagate_to_all,
        inputs=[
            video_in,
            session_state,
        ],
        outputs=[
            output_video,
            session_state,
        ],
        concurrency_limit=10,
        queue=False,
    )

demo.queue()
demo.launch()
