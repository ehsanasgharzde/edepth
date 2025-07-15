# FILE: datasets/unreal_dataset.py
# ehsanasgharzde - UNREALSTEREO4K DATASET

import os
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

class UnrealStereo4KDataset(Dataset):
    def __init__(self, data_root: str, split: str = 'train', img_size: Tuple[int, int] = (480, 640),
                 min_depth: float = 0.1, max_depth: float = 100.0, depth_scale: float = 1.0,
                 use_stereo: bool = False, augmentation: bool = True, cache: bool = False,
                 validate_data: bool = True, scene_ids: Optional[List[str]] = None):
        super().__init__()
        
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = img_size
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.depth_scale = depth_scale
        self.use_stereo = use_stereo
        self.augmentation = augmentation and split == 'train'
        self.cache = cache
        self.validate_data = validate_data
        self.scene_ids = scene_ids
        
        self._validate_initialization_parameters()
        
        if validate_data:
            self._validate_dataset_structure()
            
        self.samples = self._load_samples()
        
        self.rgb_transform = self._build_rgb_transform()
        self.depth_transform = self._build_depth_transform()
        
        self._cache_data = {} if cache else None
        
        logger.info(f"Initialized UnrealStereo4K dataset: {len(self.samples)} samples for {split} split")
    
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
        scene_dirs = [d for d in self.data_root.iterdir() if d.is_dir() and d.name.isdigit()]
        if not scene_dirs:
            raise FileNotFoundError(f"No scene directories found in {self.data_root}")
        
        for scene_dir in scene_dirs[:3]:
            required_dirs = ['Image0', 'Disp0']
            if self.use_stereo:
                required_dirs.extend(['Image1', 'Disp1'])
            
            for req_dir in required_dirs:
                if not (scene_dir / req_dir).exists():
                    raise FileNotFoundError(f"Required directory missing: {scene_dir / req_dir}")
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        samples = []
        scene_dirs = sorted([d for d in self.data_root.iterdir() if d.is_dir() and d.name.isdigit()])
        
        if self.scene_ids:
            scene_dirs = [d for d in scene_dirs if d.name in self.scene_ids]
        
        scene_split = self._get_scene_split(scene_dirs)
        missing_files = 0
        
        for scene_dir in scene_split:
            left_dir = scene_dir / 'Image0'
            disp_dir = scene_dir / 'Disp0'
            
            if not left_dir.exists() or not disp_dir.exists():
                logger.warning(f"Missing directories in scene {scene_dir.name}")
                continue
            
            left_files = sorted(left_dir.glob('*.jpg'))
            
            for left_file in left_files:
                disp_file = disp_dir / (left_file.stem + '.npy')
                
                if not disp_file.exists():
                    missing_files += 1
                    continue
                
                sample = {
                    'left_rgb': left_file,
                    'disparity': disp_file,
                    'scene': scene_dir.name,
                    'basename': left_file.stem
                }
                
                if self.use_stereo:
                    right_file = scene_dir / 'Image1' / left_file.name
                    right_disp_file = scene_dir / 'Disp1' / (left_file.stem + '.npy')
                    
                    if right_file.exists() and right_disp_file.exists():
                        sample['right_rgb'] = right_file
                        sample['right_disparity'] = right_disp_file
                    else:
                        continue
                
                if self.validate_data and not self._validate_sample_integrity(sample):
                    continue
                
                samples.append(sample)
        
        if missing_files > 0:
            logger.warning(f"Missing files: {missing_files}")
        
        if not samples:
            raise RuntimeError("No valid samples found")
        
        logger.info(f"Loaded {len(samples)} valid samples from {len(scene_split)} scenes")
        return samples
    
    def _get_scene_split(self, scene_dirs: List[Path]) -> List[Path]:
        if self.split == 'train':
            return scene_dirs[:int(len(scene_dirs) * 0.8)]
        elif self.split == 'val':
            return scene_dirs[int(len(scene_dirs) * 0.8):int(len(scene_dirs) * 0.9)]
        else:
            return scene_dirs[int(len(scene_dirs) * 0.9):]
    
    def _validate_sample_integrity(self, sample: Dict[str, Any]) -> bool:
        try:
            required_files = [sample['left_rgb'], sample['disparity']]
            if self.use_stereo:
                required_files.extend([sample['right_rgb'], sample['right_disparity']])
            
            for file_path in required_files:
                if not file_path.exists() or file_path.stat().st_size < 1024:
                    logger.warning(f"Invalid file: {file_path}")
                    return False
            
            with Image.open(sample['left_rgb']) as img:
                img.verify()
            
            disparity = np.load(sample['disparity'])
            if disparity.size == 0 or np.isnan(disparity).all():
                logger.warning(f"Invalid disparity: {sample['disparity']}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Sample integrity check failed: {e}")
            return False
    
    def _load_rgb(self, path: Path) -> np.ndarray:
        if self.cache and path in self._cache_data: #type: ignore 
            return self._cache_data[path] #type: ignore 
        
        try:
            img = Image.open(path).convert('RGB')
            if img.size != (self.img_size[1], self.img_size[0]):
                img = img.resize((self.img_size[1], self.img_size[0]), Image.LANCZOS) #type: ignore 
            
            img_array = np.array(img)
            
            if self.cache:
                self._cache_data[path] = img_array #type: ignore 
            
            return img_array
        except Exception as e:
            logger.error(f"Failed to load RGB image {path}: {e}")
            raise
    
    def _load_disparity(self, path: Path) -> np.ndarray:
        if self.cache and path in self._cache_data: #type: ignore 
            return self._cache_data[path] #type: ignore 
        
        try:
            disparity = np.load(path).astype(np.float32)
            
            if disparity.shape != self.img_size:
                from scipy.ndimage import zoom
                scale_h = self.img_size[0] / disparity.shape[0]
                scale_w = self.img_size[1] / disparity.shape[1]
                disparity = zoom(disparity, (scale_h, scale_w), order=0)
                disparity = disparity * min(scale_h, scale_w)
            
            depth = self._disparity_to_depth(disparity)
            
            if self.cache:
                self._cache_data[path] = depth #type: ignore 
            
            return depth
        except Exception as e:
            logger.error(f"Failed to load disparity {path}: {e}")
            raise
    
    def _disparity_to_depth(self, disparity: np.ndarray, focal_length: float = 1000.0, 
                           baseline: float = 0.54) -> np.ndarray:
        with np.errstate(divide='ignore', invalid='ignore'):
            depth = (focal_length * baseline) / (disparity + 1e-8)
            depth = np.clip(depth, self.min_depth, self.max_depth)
            depth[disparity <= 0] = 0
        return depth
    
    def _create_valid_depth_mask(self, depth: np.ndarray) -> np.ndarray:
        return ((depth > self.min_depth) & 
                (depth < self.max_depth) & 
                (~np.isnan(depth)) & 
                (~np.isinf(depth))).astype(bool)
    
    def _build_rgb_transform(self):
        transforms = []
        
        if self.augmentation:
            transforms.extend([
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=5),
                T.RandomPerspective(distortion_scale=0.1, p=0.3)
            ])
        
        transforms.extend([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return T.Compose(transforms)
    
    def _build_depth_transform(self):
        return T.Compose([T.ToTensor()])
    
    def _apply_synchronized_transforms(self, left_rgb: np.ndarray, right_rgb: Optional[np.ndarray], 
                                    depth: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        if not self.augmentation:
            return left_rgb, right_rgb, depth
        
        if np.random.rand() > 0.5:
            left_rgb = np.fliplr(left_rgb)
            if right_rgb is not None:
                right_rgb = np.fliplr(right_rgb)
            depth = np.fliplr(depth)
        
        return left_rgb, right_rgb, depth
    
    def compute_statistics(self) -> Dict[str, Any]:
        rgb_sum = np.zeros(3)
        rgb_sq_sum = np.zeros(3)
        depth_values = []
        pixel_count = 0
        
        logger.info("Computing dataset statistics...")
        
        sample_count = min(len(self.samples), 1000)
        
        for i, sample in enumerate(self.samples[:sample_count]):
            if i % 100 == 0:
                logger.info(f"Processing sample {i}/{sample_count}")
            
            rgb = self._load_rgb(sample['left_rgb'])
            depth = self._load_disparity(sample['disparity'])
            
            rgb_normalized = rgb.astype(np.float32) / 255.0
            rgb_sum += rgb_normalized.reshape(-1, 3).sum(axis=0)
            rgb_sq_sum += (rgb_normalized.reshape(-1, 3) ** 2).sum(axis=0)
            pixel_count += rgb.shape[0] * rgb.shape[1]
            
            valid_mask = self._create_valid_depth_mask(depth)
            if valid_mask.any():
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
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds")
        
        sample = self.samples[idx]
        
        left_rgb = self._load_rgb(sample['left_rgb'])
        depth = self._load_disparity(sample['disparity'])
        
        right_rgb = None
        if self.use_stereo:
            right_rgb = self._load_rgb(sample['right_rgb'])
        
        if self.augmentation:
            left_rgb, right_rgb, depth = self._apply_synchronized_transforms(left_rgb, right_rgb, depth)
        
        left_rgb_tensor = self.rgb_transform(left_rgb)
        depth_tensor = self.depth_transform(depth)
        
        if depth_tensor.dim() == 2: #type: ignore 
            depth_tensor = depth_tensor.unsqueeze(0) #type: ignore
        
        valid_mask = self._create_valid_depth_mask(depth)
        
        result = {
            'left_rgb': left_rgb_tensor,
            'depth': depth_tensor,
            'valid_mask': T.ToTensor()(valid_mask.astype(np.float32)), #type: ignore 
            'scene': sample['scene'],
            'basename': sample['basename']
        }
        
        if self.use_stereo and right_rgb is not None:
            right_rgb_tensor = self.rgb_transform(right_rgb)
            result['right_rgb'] = right_rgb_tensor
        
        return result
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds")
        
        sample = self.samples[idx]
        left_rgb = self._load_rgb(sample['left_rgb'])
        depth = self._load_disparity(sample['disparity'])
        
        return {
            'left_rgb_path': str(sample['left_rgb']),
            'disparity_path': str(sample['disparity']),
            'rgb_shape': left_rgb.shape,
            'depth_shape': depth.shape,
            'rgb_mean': left_rgb.mean(axis=(0, 1)).tolist(),
            'rgb_std': left_rgb.std(axis=(0, 1)).tolist(),
            'depth_min': float(depth.min()),
            'depth_max': float(depth.max()),
            'depth_valid_ratio': float((depth > 0).sum() / depth.size),
            'scene': sample['scene'],
            'basename': sample['basename']
        }