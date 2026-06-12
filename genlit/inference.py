"""

Usage:
    python -m genlit.inference --mode {single,mit,multi} ...

"""
from __future__ import annotations

import argparse
import os
import random
from typing import Optional, Sequence

import numpy as np
import torch

from genlit.config import load_config

SEED = 69


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GenLit inference")

    p.add_argument("--mode", required=True, choices=["single", "mit", "multi"])

    p.add_argument("--img_json", type=str, default=None,
                   help="Path to JSON of image paths (single, multi, or mit --in_the_wild)")
    p.add_argument("--sequences_file", type=str, default=None,
                   help="Override trajectory .npy (default from config)")
    p.add_argument("--checkpoint_dir", type=str, default=None,
                   help="Local checkpoint dir; if omitted, downloaded from HF Hub")
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--mit_test_set", type=str, default=None,
                   help="MIT only: path to combined test-set .npy (default: shipped)")
    p.add_argument("--mit_data_root", type=str, default="./mit_data/",
                   help="MIT only: root dir against which paths inside .npy resolve")
    p.add_argument("--trajectory_indices", type=str, default=None,
                   help="single/multi only: comma-separated indices to pick from the "
                        "trajectory .npy (e.g. '0,2,7'). Defaults: single=[0,2,3,4], "
                        "multi=[1,2,3,4,5,6]. See docs/figures/ for trajectory visualizations.")

    p.add_argument("--num_inference_steps", type=int, default=None)
    p.add_argument("--num_frames", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--conditioning_channels", type=int, default=None)
    p.add_argument("--motion_bucket_id", type=int, default=None)
    p.add_argument("--decode_chunk_size", type=int, default=None)
    p.add_argument("--mixed_precision", type=str, default=None)
    p.add_argument("--pretrained_model_name_or_path", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)

    args = p.parse_args(argv)

    cfg = load_config(args.mode)
    for key in [
        "num_inference_steps", "num_frames", "width", "height",
        "conditioning_channels", "motion_bucket_id", "decode_chunk_size",
        "mixed_precision", "pretrained_model_name_or_path",
    ]:
        if getattr(args, key) is None and key in cfg:
            setattr(args, key, cfg[key])

    if args.mode == "single" and args.sequences_file is None:
        args.sequences_file = cfg["default_sequences_file"]
    elif args.mode == "multi" and args.sequences_file is None:
        args.sequences_file = cfg["default_sequences_file"]
    elif args.mode == "mit" and args.mit_test_set is None:
        args.mit_test_set = cfg["default_mit_test_set"]

    # Parse trajectory_indices string -> list[int]
    if args.trajectory_indices is not None:
        try:
            args.trajectory_indices = [int(x) for x in args.trajectory_indices.split(",") if x.strip()]
        except ValueError:
            raise SystemExit(f"--trajectory_indices must be comma-separated integers, got: {args.trajectory_indices!r}")

    return args


def seed_everything(seed: int = SEED) -> torch.Generator:
    """Set all RNG sources for the run. Returns a generator on the active device."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.Generator(device=device).manual_seed(seed)


def _load_pipeline(args: argparse.Namespace, device: torch.device):
    from genlit.models import ControlNetSDVModel, UNetSpatioTemporalConditionControlNetModel
    from genlit.pipeline import StableVideoDiffusionPipelineControlNet
    from genlit.weights import resolve_checkpoint

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    ckpt_path = resolve_checkpoint(args.mode, override=args.checkpoint_dir)
    controlnet = ControlNetSDVModel.from_pretrained(ckpt_path, subfolder="controlnet")
    unet = UNetSpatioTemporalConditionControlNetModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )
    pipeline = StableVideoDiffusionPipelineControlNet.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=unet,
        controlnet=controlnet,
        torch_dtype=weight_dtype,
    )
    return pipeline.to(device)


def _build_dataloader(args: argparse.Namespace) -> torch.utils.data.DataLoader:
    from torch.utils.data import DataLoader

    if args.mode == "single":
        from genlit.data import SingleObjectDataset as Ds
        ds = Ds(
            img_npy=args.img_json,
            sequences_file=args.sequences_file,
            width=args.width,
            height=args.height,
            conditioning_channels=args.conditioning_channels,
            sequence_length=args.num_frames,
            trajectory_indices=args.trajectory_indices,
        )
    elif args.mode == "multi":
        from genlit.data import MultiObjectDataset as Ds
        ds = Ds(
            img_npy=args.img_json,
            sequences_file=args.sequences_file,
            width=args.width,
            height=args.height,
            conditioning_channels=args.conditioning_channels,
            sequence_length=args.num_frames,
            trajectory_indices=args.trajectory_indices,
        )
    elif args.mode == "mit":
        from genlit.data import MITTestSet
        ds = MITTestSet(
            mit_test_set_npy=args.mit_test_set,
            data_root=args.mit_data_root,
            height=args.height,
            width=args.width,
            sequence_length=args.num_frames,
        )
    else:
        raise ValueError(f"Unknown mode {args.mode!r}")

    ds.disable_transforms()
    return DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers)


def main(argv: Optional[Sequence[str]] = None) -> int:
    from genlit.utils.io import save_grid_real, video

    args = parse_args(argv)
    print(f"[genlit] mode={args.mode} steps={args.num_inference_steps}")
    generator = seed_everything(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = _load_pipeline(args, device)
    dataloader = _build_dataloader(args)
    if args.checkpoint_dir:
        checkpoint_tag = os.path.basename(os.path.dirname(args.checkpoint_dir.rstrip("/")))
    else:
        checkpoint_tag = args.mode

    folder_name = f"validation_images_{args.mode}"
    os.makedirs(os.path.join(args.output_dir, folder_name), exist_ok=True)
    checkpoint_folder = os.path.join(args.output_dir, folder_name, checkpoint_tag)
    os.makedirs(checkpoint_folder, exist_ok=True)

    with torch.autocast(str(device).replace(":0", ""), enabled=True):
        for step, batch in enumerate(dataloader):
            gt_frames = batch["pixel_values"].to(device).squeeze()
            cond_frames = batch["cond_values"].to(device).squeeze()
            motion_values = batch["motion_values"]
            seq_name = batch["sequence_name"]
            frame_nums = batch["frame_nums"][0] if "frame_nums" in batch else np.arange(args.num_frames)

            pred_frames = pipeline(
                gt_frames[0],
                cond_frames,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                decode_chunk_size=args.decode_chunk_size,
                motion_bucket_id=(args.motion_bucket_id if args.motion_bucket_id is not None else motion_values),
                fps=6 if args.num_frames == 25 else 7,
                noise_aug_strength=0.02,
                output_type="pt",
                num_inference_steps=args.num_inference_steps,
                generator=generator,
            ).frames

            gt_frames = gt_frames.cpu()
            cond_frames_vis = torch.concat(
                [cond_frames[:, :2, ...], torch.ones_like(cond_frames[:, 0:1, ...])], axis=1
            ).cpu()
            pred_frames = torch.stack(pred_frames).squeeze().cpu()

            scene_label = seq_name[0][0] if isinstance(seq_name[0], (list, tuple)) else seq_name[0]
            grid_dir = os.path.join(checkpoint_folder, "grid_vid", str(scene_label))
            os.makedirs(grid_dir, exist_ok=True)
            out_img = os.path.join(grid_dir, f"val_img_{int(frame_nums[0])}_{int(frame_nums[-1])}.png")
            print(out_img)
            pred_img_frame = torch.concat([gt_frames[0:1], pred_frames], axis=0)
            save_grid_real(gt_frames, cond_frames_vis, pred_img_frame, out_img, seq_name, frame_nums, args.num_frames)

            video_dir = os.path.join(checkpoint_folder, "videos", str(scene_label))
            os.makedirs(video_dir, exist_ok=True)
            video(video_dir, pred_img_frame, seq_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
