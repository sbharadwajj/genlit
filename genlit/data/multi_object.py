"""Multi-object dataset loader.
"""
import json

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, Resize, ToTensor


class RealImg(Dataset):
    """Multi-object loader. Default sequence picks: [1, 2, 3, 4, 5, 6].
    Override via `trajectory_indices`.
    """

    DEFAULT_TRAJECTORY_INDICES = [1, 2, 3, 4, 5, 6]

    def __init__(
        self,
        img_npy: str,
        sequences_file: str,
        width: int = 640,
        height: int = 448,
        conditioning_channels: int = 5,
        sequence_length: int = 25,
        motion_value: int = 10,
        step_size: int = 4,
        transform=None,
        mode: str = "train",
        trajectory_indices=None,
    ):
        self.sequence_length = sequence_length
        self.motion_value = motion_value
        self.step_size = step_size
        self.transform = transform or self._default_transforms()
        self.mode = mode
        self.sequences_file = sequences_file
        self.resize = Compose([Resize((height, width))])
        self.conditioning_channels = conditioning_channels
        self.totensor = ToTensor()
        picks = list(trajectory_indices) if trajectory_indices is not None else self.DEFAULT_TRAJECTORY_INDICES
        self.trajectory_indices = picks
        self.sequences = np.load(self.sequences_file, allow_pickle=True)
        self.sequences = self.sequences[picks, :, :]

        self.img_path = img_npy
        with open(self.img_path, "r") as f:
            self.data = json.load(f)
        self.img_names = list(self.data.keys())

        self.render_list = []
        for img_name in self.img_names:
            for cnt, seq in enumerate(self.sequences):
                self.render_list.append([img_name, seq, cnt])

    @staticmethod
    def _default_transforms():
        return Compose([Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    def disable_transforms(self):
        self.transform = None

    def __len__(self):
        return len(self.render_list)

    def __getitem__(self, idx):
        slice_ = self.render_list[idx]
        img_frames = []
        cond_frames = []
        sequence_name = [slice_[0].split(".")[0], str(slice_[2])]

        for frame_info in slice_[1]:
            img_path = self.data[slice_[0]]
            cond_data = np.array(frame_info[0].split(" ") + [frame_info[1]] + [frame_info[2]]).astype(float)
            cond_data[0] = 0.1 + ((cond_data[0] - 0.8) / (1.5 - 0.8)) * 0.8
            theta_360 = cond_data[1] % 360
            cond_data[1] = 0.1 + (theta_360 / 360) * 0.8
            cond_data[2] = cond_data[2] / 90.0
            cond_data[3] = cond_data[3]
            cond_data[4] = cond_data[4] / 75.0
            if self.conditioning_channels == 4:
                cond_data = cond_data[1:]

            img_frame = self.totensor(Image.open(img_path)).float()[:3, :, :]
            img_frame = self.resize(img_frame)
            h, w = img_frame.shape[-2:]
            cond_frame = torch.tensor(cond_data).repeat(h, w, 1).permute(2, 0, 1).float()

            if self.transform:
                img_frame = self.transform(img_frame)

            img_frames.append(img_frame)
            cond_frames.append(cond_frame)

        return {
            "sequence_name": sequence_name,
            "pixel_values": torch.stack(img_frames),
            "cond_values": torch.stack(cond_frames),
            "motion_values": self.motion_value,
            "frame_nums": np.arange(0, self.sequence_length),
        }
