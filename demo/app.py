"""Gradio demo for GenLit.

Imports `genlit.inference` directly so the CLI and demo share the same code
path — no duplicated forward pass.

Supports `single` and `multi` modes (uploaded photo + auto-trajectory). MIT
mode is excluded here because it needs the MIT Multi-Illumination dataset
that users must download separately and is mostly meant for test set. 

Run locally: `python demo/app.py`
Deploy to HF Space: copy this folder, set hardware to GPU, push to a Space.
"""
import json
import os
import tempfile
from pathlib import Path

import gradio as gr
import torch

from genlit.inference import (
    _build_dataloader,
    _load_pipeline,
    parse_args,
    seed_everything,
)
from genlit.utils.io import video


def run_inference(image, mode, num_inference_steps, trajectory_index):
    """Run a single uploaded image through the selected mode and return a video path."""
    if image is None:
        raise gr.Error("Please upload an image first.")

    tmpdir = Path(tempfile.mkdtemp(prefix="genlit_demo_"))
    img_path = tmpdir / "input.jpg"
    image.save(img_path)
    json_path = tmpdir / "input.json"
    json_path.write_text(json.dumps({"input.jpg": str(img_path)}))

    argv = [
        "--mode", mode,
        "--img_json", str(json_path),
        "--output_dir", str(tmpdir / "out"),
        "--num_inference_steps", str(int(num_inference_steps)),
        "--batch_size", "1",
        "--num_workers", "0",
        "--trajectory_indices", str(int(trajectory_index)),
    ]

    args = parse_args(argv)
    generator = seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = _load_pipeline(args, device)
    dataloader = _build_dataloader(args)

    output_path = None
    with torch.autocast(str(device).replace(":0", ""), enabled=True):
        for batch in dataloader:
            gt = batch["pixel_values"].to(device).squeeze()
            cond = batch["cond_values"].to(device).squeeze()
            pred = pipeline(
                gt[0], cond,
                height=args.height, width=args.width,
                num_frames=args.num_frames,
                decode_chunk_size=args.decode_chunk_size,
                motion_bucket_id=(args.motion_bucket_id if args.motion_bucket_id is not None else batch["motion_values"]),
                fps=6 if args.num_frames == 25 else 7,
                noise_aug_strength=0.02,
                output_type="pt",
                num_inference_steps=args.num_inference_steps,
                generator=generator,
            ).frames
            frames = torch.stack(pred).squeeze().cpu()
            pred_img_frame = torch.concat([gt.cpu()[0:1], frames], axis=0)
            seq_name = batch["sequence_name"]
            video_dir = tmpdir / "video"
            video_dir.mkdir(exist_ok=True)
            video(str(video_dir), pred_img_frame, seq_name)
            # The video() helper saves per-frame PNGs; assemble them into an MP4
            output_path = _frames_to_mp4(video_dir, tmpdir / "output.mp4", args.num_frames)
            break
    return str(output_path)


def _frames_to_mp4(frame_dir: Path, out_mp4: Path, num_frames: int) -> Path:
    """Use ffmpeg to assemble per-frame PNGs in subdirs into a single MP4."""
    import subprocess
    img_subdir = next(frame_dir.iterdir())  # e.g. "0_images"
    fps = 6 if num_frames == 25 else 7
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", str(img_subdir / "%03d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_mp4


with gr.Blocks(title="GenLit — relighting demo") as demo:
    gr.Markdown(
        """# GenLit — object relighting via Stable Video Diffusion + ControlNet
[Paper](https://genlit.is.tue.mpg.de/) · [Code](#) · [Weights](https://huggingface.co/sbharadwaj/genlit) · License: non-commercial research

Upload a photo, pick a mode, and the model will generate a video showing the same object/scene under a varying lighting trajectory.

- **single** — single object, 14 frames at 512×512 (faster)
- **multi** — multi-object scene, 25 frames at 640×448 (slower, but multi objects)

Weights download from HuggingFace on first run (~3GB per mode). Requires GPU.
"""
    )

    with gr.Row():
        with gr.Column():
            img = gr.Image(type="pil", label="Input image", height=320)
            mode = gr.Radio(["single", "multi"], value="single", label="Mode")
            steps = gr.Slider(10, 50, value=30, step=5, label="Inference steps (more = better quality, slower)")
            traj = gr.Number(value=0, precision=0, label="Trajectory index (see grids below)")
            go = gr.Button("Generate", variant="primary")
        with gr.Column():
            out_video = gr.Video(label="Output", height=320)

    gr.Markdown("### Trajectory reference — pick an index from the grid for your mode")
    with gr.Row():
        gr.Image(value="../docs/figures/single_trajectories.png", label="single trajectories (0..19)", interactive=False)
        gr.Image(value="../docs/figures/multi_trajectories.png", label="multi trajectories (0..20)", interactive=False)

    go.click(run_inference, inputs=[img, mode, steps, traj], outputs=[out_video])


if __name__ == "__main__":
    demo.queue().launch()
