"""
UCF-101 RGB dataset: clip sampling, spatial augmentation, normalization.

Expected folder structure:
  data/UCF-101/<ClassName>/<v_ClassName_gXX_cXX.avi>
  splits/trainlist01.txt  →  "ClassName/v_xxx.avi <label_int>"
  splits/testlist01.txt   →  "ClassName/v_xxx.avi"

Set config.SUBSET = True to use a 10-class subset for quick CPU debugging.
"""

import os
import random

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.io as tvio

import config

# 10 visually distinct classes for fast CPU debugging
_SUBSET_CLASSES = [
    'Basketball', 'Biking', 'Diving', 'GolfSwing', 'HorseRiding',
    'JumpingJack', 'PushUps', 'Skiing', 'TennisSwing', 'Typing',
]


def _read_split(split_file: str, is_train: bool) -> list[tuple[str, int]]:
    samples = []
    with open(split_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            rel_path = parts[0]
            label    = int(parts[1]) - 1 if is_train else -1
            full_path = os.path.join(config.DATA_ROOT, rel_path)
            samples.append((full_path, label))
    return samples


def _sample_clip(video_path: str, clip_len: int,
                 stride: int) -> torch.Tensor:
    """Decode video and uniformly sample clip_len frames.

    Returns a uint8 tensor of shape (T, H, W, C).
    Falls back to a zero tensor when the video cannot be read.
    """
    try:
        vframes, _, _ = tvio.read_video(video_path, pts_unit="sec")
    except Exception:
        return torch.zeros(clip_len, 240, 320, 3, dtype=torch.uint8)

    total = vframes.shape[0]
    if total == 0:
        return torch.zeros(clip_len, 240, 320, 3, dtype=torch.uint8)

    if total >= clip_len * stride:
        max_start = total - clip_len * stride
        start     = random.randint(0, max_start)
        indices   = [start + i * stride for i in range(clip_len)]
    else:
        indices = [i % total for i in range(clip_len)]

    return vframes[indices]


class UCF101Dataset(Dataset):
    def __init__(self, split: str = "train") -> None:
        assert split in ("train", "test")
        self.is_train = split == "train"
        sid    = config.SPLIT_ID
        subset = getattr(config, "SUBSET", False)

        if subset:
            suffix = "_10class"
            self._class_to_idx = {c: i for i, c in enumerate(_SUBSET_CLASSES)}
        else:
            suffix = ""
            self._class_to_idx = {
                c: i for i, c in enumerate(sorted(os.listdir(config.DATA_ROOT)))
            }

        fname = (
            f"trainlist{sid:02d}{suffix}.txt"
            if self.is_train
            else f"testlist{sid:02d}{suffix}.txt"
        )
        self.samples = _read_split(
            os.path.join(config.SPLITS_DIR, fname), self.is_train
        )

        size = config.FRAME_SIZE          # 224 (matches paper)
        if self.is_train:
            self.spatial_tf = T.Compose([
                T.RandomResizedCrop(size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ])
        else:
            # Resize shorter side to 256, then center-crop to 224
            self.spatial_tf = T.Compose([
                T.Resize(int(size * 256 / 224)),
                T.CenterCrop(size),
            ])

        self.normalize = T.Compose([
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=config.MEAN, std=config.STD),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        video_path, label = self.samples[idx]

        if label == -1:
            class_name = os.path.basename(os.path.dirname(video_path))
            label = self._class_to_idx.get(class_name, 0)

        frames = _sample_clip(video_path, config.CLIP_LEN, config.STRIDE)

        processed = []
        for t in range(frames.shape[0]):
            frame = frames[t].permute(2, 0, 1)   # (C, H, W)
            frame = self.spatial_tf(frame)
            frame = self.normalize(frame)
            processed.append(frame)

        clip = torch.stack(processed, dim=1)      # (C, T, H, W)
        return clip, label


def get_loaders() -> tuple[DataLoader, DataLoader]:
    train_ds = UCF101Dataset("train")
    test_ds  = UCF101Dataset("test")

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    return train_loader, test_loader
