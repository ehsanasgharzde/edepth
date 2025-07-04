import os
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Sequence, Any
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

KITTI_MEAN = (0.356, 0.365, 0.376)
KITTI_STD = (0.170, 0.168, 0.173)

class KITTIDepthDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        img_size: Tuple[int, int] = (352, 1216),
        min_depth: float = 1e-3,
        max_depth: float = 80.0,
        depth_scale: float = 256.0,
        augmentation: bool = True,
        cache: bool = False
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = img_size
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.depth_scale = depth_scale
        self.augmentation = augmentation and split == 'train'
        self.rgb_dir = self.data_root / split / 'rgb'
        self.depth_dir = self.data_root / split / 'depth'
        self.samples = self._load_samples()
        self.rgb_transform = self._get_rgb_transform()
        self.depth_transform = self._get_depth_transform()
        self.cache = cache
        self._cache = {} if cache else None

    def _load_samples(self) -> List[Dict[str, Path]]:
        rgb_files = sorted(list(self.rgb_dir.glob('*.jpg'))) + sorted(list(self.rgb_dir.glob('*.png')))
        samples = []
        for rgb_file in rgb_files:
            depth_file = self.depth_dir / (rgb_file.stem + '.png')
            if not depth_file.exists():
                continue
            samples.append({'rgb': rgb_file, 'depth': depth_file})
        if not samples:
            raise RuntimeError(f"No samples found in {self.rgb_dir}")
        return samples

    def _get_rgb_transform(self):
        transforms: Sequence[Any] = [A.Resize(self.img_size[0], self.img_size[1], interpolation=1)]
        if self.augmentation:
            transforms += [
                A.RandomCrop(height=self.img_size[0], width=self.img_size[1]),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
            ]
        transforms += [A.Normalize(mean=KITTI_MEAN, std=KITTI_STD), ToTensorV2()]
        return A.Compose(transforms)  # type: ignore

    def _get_depth_transform(self):
        transforms = [A.Resize(self.img_size[0], self.img_size[1], interpolation=0)]
        transforms += [ToTensorV2()]
        return A.Compose(transforms)

    def _load_rgb(self, path: Path) -> np.ndarray:
        if self.cache and self._cache is not None and path in self._cache:
            return self._cache[path]
        img = np.array(Image.open(path).convert('RGB'))
        if self.cache and self._cache is not None:
            self._cache[path] = img
        return img

    def _load_depth(self, path: Path) -> np.ndarray:
        if self.cache and self._cache is not None and path in self._cache:
            return self._cache[path]
        depth = np.array(Image.open(path)).astype(np.float32)
        depth = depth / self.depth_scale
        depth = np.clip(depth, self.min_depth, self.max_depth)
        if self.cache and self._cache is not None:
            self._cache[path] = depth
        return depth

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        rgb = self._load_rgb(sample['rgb'])
        depth = self._load_depth(sample['depth'])
        rgb = self.rgb_transform(image=rgb)['image']
        depth = self.depth_transform(image=depth)['image'].unsqueeze(0)
        return {'rgb': rgb, 'depth': depth} 