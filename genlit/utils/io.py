"""I/O helpers extracted near-verbatim from src/utils/{helper_functions,stack_videos}.py.

Only the functions used by the real-image branch of the original inference scripts are kept:
- list_torchgrid: build a torchvision grid and save as PNG
- save_grid_real: thin wrapper calling list_torchgrid on the predicted frames
- video: save each predicted frame as an individual PNG
"""
import os

import torch
import torchvision.utils as torch_utils
from PIL import Image


def list_torchgrid(
    save_list,
    grid_path: str = "out/",
    save_name: str = "test.png",
    nrow: int = 5,
    save: bool = True,
    scale_factor: int = 255,
    normalize: bool = False,
    scale_each: bool = False,
):
    if not torch.is_tensor(save_list):
        concatenate_imgs = torch.cat(save_list, dim=0)
    else:
        concatenate_imgs = save_list

    grid = torch_utils.make_grid(concatenate_imgs, scale_each=scale_each, normalize=normalize, nrow=nrow)
    grid = grid.permute(1, 2, 0)
    tensor = (grid * scale_factor).to(torch.uint8)

    if save:
        img = Image.fromarray(tensor.detach().cpu().numpy())
        img.save(grid_path)
    else:
        return tensor


def save_grid_real(GT_frames, COND_frames, PRED_frames, save_path, sequence_name, frame_range, num_frames):
    list_torchgrid(PRED_frames, save_path, nrow=PRED_frames.shape[0], scale_factor=255)


def video(video_save, pred_img_frame, seq_name):
    img_save_folder = os.path.join(video_save, f"{seq_name[1][0]}_images")
    os.makedirs(img_save_folder, exist_ok=True)
    for cnt, pred in enumerate(pred_img_frame):
        pred = pred.permute(1, 2, 0)
        pred = (pred * 255).to(torch.uint8)
        pred_pil = Image.fromarray(pred.detach().cpu().numpy())
        pred_pil.save(os.path.join(img_save_folder, f"{cnt:03d}.png"))
