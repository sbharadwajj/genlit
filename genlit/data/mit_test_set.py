"""MIT Multi-Illumination test-set loader.
"""
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, Resize, ToTensor


class MITTestSet(Dataset):
    def __init__(
        self,
        mit_test_set_npy: str,
        data_root: str,
        height: int = 512,
        width: int = 768,
        sequence_length: int = 14,
        motion_value: int = 10,
        transform=None,
    ):
        self.H = height
        self.W = width
        self.sequence_length = sequence_length
        self.motion_value = motion_value
        self.transform = transform or self._default_transforms()
        self.resize = Compose([Resize((height, width))])
        self.totensor = ToTensor()
        self.data_root = data_root

        raw = np.load(mit_test_set_npy, allow_pickle=True)
        if raw.ndim == 4:
            num_scenes, num_traj_per_scene = raw.shape[0], raw.shape[1]
            self.sequences = raw.reshape(-1, sequence_length, 3)
        elif raw.ndim == 3 and raw.shape[1] == sequence_length:
            num_traj_per_scene = 25
            num_scenes = raw.shape[0] // num_traj_per_scene
            self.sequences = raw
        else:
            raise ValueError(f"Unexpected .npy shape {raw.shape}")

        self.seq_num = (
            torch.tensor(np.arange(num_traj_per_scene))
            .unsqueeze(0)
            .repeat(num_scenes, 1)
            .reshape(-1)
        )

    @staticmethod
    def _default_transforms():
        return Compose([Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    def disable_transforms(self):
        self.transform = None

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        slice_ = self.sequences[idx]
        img_frames = []
        cond_frames = []
        img_paths_all = []

        first_rel = slice_[0][0]
        scene = first_rel.split("/")[0] if "/" in first_rel else first_rel.split(".")[0]
        sequence_name = [scene, str(self.seq_num[idx].item())]

        for frame_info in slice_:
            rel_path = frame_info[0]
            abs_path = os.path.join(self.data_root, rel_path)
            img_paths_all.append(int(os.path.basename(rel_path).split("_")[1]))

            cond_data = np.array(frame_info[1].split(" ")).astype(float)
            cond_data[0] = cond_data[0] / 5.0
            cond_data[1] = cond_data[1]
            cond_data[2] = cond_data[2]
            cond_data = cond_data[1:]

            img_frame = self.totensor(Image.open(abs_path)).float()[:3, :, :]
            img_frame = self.resize(img_frame)
            cond_frame = (
                torch.tensor(cond_data).repeat(self.H, self.W, 1).permute(2, 0, 1).float()
            )

            if self.transform:
                img_frame = self.transform(img_frame)

            img_frames.append(img_frame)
            cond_frames.append(cond_frame)

        return {
            "sequence_name": sequence_name,
            "pixel_values": torch.stack(img_frames),
            "cond_values": torch.stack(cond_frames),
            "img_path_frame_num": img_paths_all,
            "motion_values": self.motion_value,
            "frame_nums": np.arange(0, self.sequence_length),
        }
