# FILE: datasets/nyu_dataset.py
# ehsanasgharzde - NYU DEPTH V2 DATASET


import os
import glob
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

logger = logging.getLogger(__name__)

NYU_MEAN = (0.413, 0.427, 0.399)
NYU_STD = (0.186, 0.184, 0.186)

class NYUV2Dataset(Dataset):
    def __init__(self, data_root: str, split: str = 'train', img_size: Tuple[int, int] = (480, 640), 
                 min_depth: float = 0.1, max_depth: float = 10.0, depth_scale: float = 1000.0, 
                 augmentation: bool = True, cache: bool = False, validate_data: bool = True):
        super().__init__()
        
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = img_size
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.depth_scale = depth_scale
        self.augmentation = augmentation and split == 'train'
        self.cache = cache
        self.validate_data = validate_data
        
        self._validate_initialization_parameters()
        
        self.rgb_dir = self.data_root / split / 'rgb'
        self.depth_dir = self.data_root / split / 'depth'
        
        if validate_data:
            self._validate_dataset_structure()
            
        self.samples = self._load_samples()
        
        self.rgb_transform = self._get_rgb_transform()
        self.depth_transform = self._get_depth_transform()
        
        self._cache = {} if cache else None
        
        logger.info(f"Initialized NYU Depth V2 dataset: {len(self.samples)} samples for {split} split")
    
    def _validate_initialization_parameters(self):
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root not found: {self.data_root}")
        
        if self.split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split: {self.split}")
        
        if not isinstance(self.img_size, tuple) or len(self.img_size) != 2:
            raise ValueError("img_size must be tuple of (height, width)")
        
        if not (0 < self.min_depth < self.max_depth):
            raise ValueError("min_depth must be less than max_depth")
        
        if self.depth_scale <= 0:
            raise ValueError("depth_scale must be positive")
    
    def _validate_dataset_structure(self):
        if not self.rgb_dir.exists():
            raise FileNotFoundError(f"RGB directory not found: {self.rgb_dir}")
        
        if not self.depth_dir.exists():
            raise FileNotFoundError(f"Depth directory not found: {self.depth_dir}")
        
        rgb_files = list(self.rgb_dir.glob('*.jpg')) + list(self.rgb_dir.glob('*.png'))
        depth_files = list(self.depth_dir.glob('*.png'))
        
        if not rgb_files:
            raise FileNotFoundError(f"No RGB images found in {self.rgb_dir}")
        
        if not depth_files:
            raise FileNotFoundError(f"No depth images found in {self.depth_dir}")
    
    def _load_samples(self) -> List[Dict[str, Path]]:
        rgb_files = sorted(list(self.rgb_dir.glob('*.jpg')) + list(self.rgb_dir.glob('*.png')))
        samples = []
        missing_depth = 0
        
        for rgb_file in rgb_files:
            depth_file = self.depth_dir / (rgb_file.stem + '.png')
            if not depth_file.exists():
                missing_depth += 1
                continue
            
            if self.validate_data and not self._validate_sample_integrity(rgb_file, depth_file):
                continue
                
            samples.append({
                'rgb': rgb_file,
                'depth': depth_file,
                'basename': rgb_file.stem
            })
        
        if missing_depth > 0:
            logger.warning(f"Missing depth files: {missing_depth}")
        
        if not samples:
            raise RuntimeError("No valid RGB-depth pairs found")
        
        logger.info(f"Loaded {len(samples)} valid samples")
        return samples
    
    def _validate_sample_integrity(self, rgb_path: Path, depth_path: Path) -> bool:
        try:
            if rgb_path.stat().st_size < 1024 or depth_path.stat().st_size < 1024:
                logger.warning(f"File too small: {rgb_path} or {depth_path}")
                return False
            
            with Image.open(rgb_path) as rgb_img:
                rgb_img.verify()
            
            with Image.open(depth_path) as depth_img:
                depth_img.verify()
            
            return True
            
        except Exception as e:
            logger.error(f"Sample integrity check failed: {rgb_path}, {depth_path} - {e}")
            return False
    
    def _get_rgb_transform(self):
        transforms = [A.Resize(self.img_size[0], self.img_size[1], interpolation=1)]
        
        if self.augmentation:
            transforms.extend([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2), #type: ignore 
            ])
        
        transforms.extend([
            A.Normalize(mean=NYU_MEAN, std=NYU_STD), #type: ignore 
            ToTensorV2()
        ])
        
        return A.Compose(transforms) #type: ignore 
    
    def _get_depth_transform(self):
        transforms = [A.Resize(self.img_size[0], self.img_size[1], interpolation=0)]
        
        if self.augmentation:
            transforms.append(A.HorizontalFlip(p=0.5)) #type: ignore 
        
        transforms.append(ToTensorV2()) #type: ignore 
        return A.Compose(transforms) #type: ignore 
    
    def _clip_and_scale_depth(self, depth: np.ndarray) -> np.ndarray:
        return np.clip(depth, self.min_depth, self.max_depth)
    
    def _load_rgb(self, path: Path) -> np.ndarray:
        if self.cache and path in self._cache: #type: ignore 
            return self._cache[path] #type: ignore 
        
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Unable to read RGB image: {path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.cache:
            self._cache[path] = img #type: ignore 
        
        return img
    
    def _load_depth(self, path: Path) -> np.ndarray:
        if self.cache and path in self._cache: #type: ignore 
            return self._cache[path] #type: ignore 
        
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise IOError(f"Unable to read depth image: {path}")
        
        depth = depth.astype(np.float32) / self.depth_scale
        depth = np.clip(depth, self.min_depth, self.max_depth)
        
        invalid_mask = (depth <= 0) | np.isnan(depth) | np.isinf(depth)
        depth[invalid_mask] = 0.0
        
        if self.cache:
            self._cache[path] = depth #type: ignore 
        
        return depth
    
    def _create_valid_depth_mask(self, depth: np.ndarray) -> np.ndarray:
        return ((depth > 0) & 
                (~np.isnan(depth)) & 
                (~np.isinf(depth)) & 
                (depth >= self.min_depth) & 
                (depth <= self.max_depth)).astype(bool)
    
    def _apply_synchronized_transforms(self, rgb: np.ndarray, depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.augmentation:
            return rgb, depth
        
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        ], additional_targets={'depth': 'mask'})
        
        try:
            transformed = transform(image=rgb, depth=depth)
            return transformed['image'], transformed['depth']
        except Exception as e:
            logger.error(f"Transform failed: {e}")
            return rgb, depth
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds")
        
        sample = self.samples[idx]
        
        rgb = self._load_rgb(sample['rgb'])
        depth = self._load_depth(sample['depth'])
        
        if self.augmentation:
            rgb, depth = self._apply_synchronized_transforms(rgb, depth)
        
        rgb = self.rgb_transform(image=rgb)['image']
        depth = self.depth_transform(image=depth)['image']
        
        if depth.dim() == 2:
            depth = depth.unsqueeze(0)
        
        valid_mask = self._create_valid_depth_mask(depth.squeeze(0).numpy())
        
        return {
            'rgb': rgb,
            'depth': depth,
            'valid_mask': torch.from_numpy(valid_mask).unsqueeze(0),
            'rgb_path': str(sample['rgb']),
            'depth_path': str(sample['depth']),
            'basename': sample['basename']
        }
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds")
        
        sample = self.samples[idx]
        rgb = self._load_rgb(sample['rgb'])
        depth = self._load_depth(sample['depth'])
        
        return {
            'rgb_path': str(sample['rgb']),
            'depth_path': str(sample['depth']),
            'rgb_shape': rgb.shape,
            'depth_shape': depth.shape,
            'rgb_mean': rgb.mean(axis=(0, 1)).tolist(),
            'rgb_std': rgb.std(axis=(0, 1)).tolist(),
            'depth_min': float(depth.min()),
            'depth_max': float(depth.max()),
            'depth_valid_ratio': float((depth > 0).sum() / depth.size),
            'basename': sample['basename']
        }
    
    def compute_statistics(self) -> Dict[str, Any]:
        rgb_sum = np.zeros(3)
        rgb_sq_sum = np.zeros(3)
        depth_values = []
        pixel_count = 0
        
        logger.info("Computing dataset statistics...")
        
        for i, sample in enumerate(self.samples):
            if i % 100 == 0:
                logger.info(f"Processing sample {i}/{len(self.samples)}")
            
            rgb = self._load_rgb(sample['rgb'])
            depth = self._load_depth(sample['depth'])
            
            rgb_normalized = rgb.astype(np.float32) / 255.0
            rgb_sum += rgb_normalized.reshape(-1, 3).sum(axis=0)
            rgb_sq_sum += (rgb_normalized.reshape(-1, 3) ** 2).sum(axis=0)
            pixel_count += rgb.shape[0] * rgb.shape[1]
            
            valid_mask = self._create_valid_depth_mask(depth)
            depth_values.append(depth[valid_mask])
        
        depth_values = np.concatenate(depth_values) if depth_values else np.array([])
        
        rgb_mean = rgb_sum / pixel_count
        rgb_var = (rgb_sq_sum / pixel_count) - (rgb_mean ** 2)
        rgb_std = np.sqrt(rgb_var)
        
        stats = {
            'rgb_mean': rgb_mean.tolist(),
            'rgb_std': rgb_std.tolist(),
            'num_samples': len(self.samples),
            'num_pixels': int(pixel_count),
            'num_valid_depth_points': int(depth_values.size),
            'depth_min': float(depth_values.min()) if depth_values.size > 0 else None,
            'depth_max': float(depth_values.max()) if depth_values.size > 0 else None,
            'depth_mean': float(depth_values.mean()) if depth_values.size > 0 else None,
            'depth_std': float(depth_values.std()) if depth_values.size > 0 else None,
        }
        
        if depth_values.size > 0:
            for percentile in [25, 50, 75, 95]:
                stats[f'depth_p{percentile}'] = float(np.percentile(depth_values, percentile))
        
        logger.info("Statistics computation completed")
        return stats
    
    def validate_dataset(self) -> Dict[str, Any]:
        logger.info("Validating dataset...")
        
        errors = []
        failed_samples = []
        check_count = min(len(self), 100)
        
        for i in range(check_count):
            try:
                sample = self.__getitem__(i)
                
                if sample['rgb'].shape[0] != 3:
                    raise ValueError(f"Invalid RGB shape: {sample['rgb'].shape}")
                
                if sample['depth'].shape[0] != 1:
                    raise ValueError(f"Invalid depth shape: {sample['depth'].shape}")
                
                if sample['valid_mask'].shape[0] != 1:
                    raise ValueError(f"Invalid mask shape: {sample['valid_mask'].shape}")
                
            except Exception as e:
                errors.append(f"Sample {i}: {e}")
                failed_samples.append(i)
        
        validation_report = {
            'total_checked': check_count,
            'failed_count': len(failed_samples),
            'failed_indices': failed_samples,
            'errors': errors,
            'status': 'passed' if len(failed_samples) == 0 else 'failed'
        }
        
        logger.info(f"Validation completed: {validation_report['status']}")
        return validation_report