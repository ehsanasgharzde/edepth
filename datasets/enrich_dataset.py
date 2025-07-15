# FILE: datasets/enrich_dataset.py
# ehsanasgharzde - ENRICH DATASET

import os
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import random
import cv2

logger = logging.getLogger(__name__)

class ENRICHDataset(Dataset):
    def __init__(self, data_root: str, split: str = 'train', img_size: Tuple[int, int] = (480, 640),
                 min_depth: float = 0.1, max_depth: float = 100.0, depth_scale: float = 1.0,
                 augmentation: bool = True, cache: bool = False, validate_data: bool = True,
                 domain_adaptation: bool = False, dataset_type: str = 'all'):
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
        self.domain_adaptation = domain_adaptation
        self.dataset_type = dataset_type
        
        self._validate_initialization_parameters()
        
        if validate_data:
            self._validate_dataset_structure()
        
        self.samples = self._load_samples()
        
        self.rgb_transform = self._build_rgb_transform()
        self.depth_transform = self._build_depth_transform()
        
        self._cache_data = {} if cache else None
        
        logger.info(f"Initialized ENRICH dataset: {len(self.samples)} samples for {split} split")
    
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
        
        if self.dataset_type not in ['all', 'aerial', 'square', 'statue']:
            raise ValueError(f"Invalid dataset_type: {self.dataset_type}")
    
    def _validate_dataset_structure(self):
        required_datasets = ['ENRICH-Aerial', 'ENRICH-Square', 'ENRICH-Statue']
        
        if self.dataset_type == 'all':
            check_datasets = required_datasets
        else:
            check_datasets = [f'ENRICH-{self.dataset_type.title()}']
        
        for dataset_name in check_datasets:
            dataset_path = self.data_root / dataset_name
            if not dataset_path.exists():
                logger.warning(f"Dataset directory not found: {dataset_path}")
                continue
            
            required_dirs = ['images', 'depth/exr']
            for req_dir in required_dirs:
                if not (dataset_path / req_dir).exists():
                    logger.warning(f"Required directory missing: {dataset_path / req_dir}")
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        samples = []
        
        dataset_dirs = []
        if self.dataset_type == 'all':
            dataset_dirs = [d for d in self.data_root.iterdir() if d.is_dir() and d.name.startswith('ENRICH-')]
        else:
            dataset_path = self.data_root / f'ENRICH-{self.dataset_type.title()}'
            if dataset_path.exists():
                dataset_dirs = [dataset_path]
        
        if not dataset_dirs:
            raise FileNotFoundError(f"No ENRICH dataset directories found in {self.data_root}")
        
        missing_files = 0
        
        for dataset_dir in dataset_dirs:
            dataset_name = dataset_dir.name
            
            images_dir = dataset_dir / 'images'
            depth_dir = dataset_dir / 'depth' / 'exr'
            
            if not images_dir.exists() or not depth_dir.exists():
                logger.warning(f"Missing directories in {dataset_name}")
                continue
            
            image_files = sorted(images_dir.glob('*.jpg'))
            
            for image_file in image_files:
                base_name = image_file.stem
                depth_file = depth_dir / f'{base_name}_depth.exr'
                
                if not depth_file.exists():
                    missing_files += 1
                    continue
                
                sample = {
                    'rgb': image_file,
                    'depth': depth_file,
                    'dataset': dataset_name,
                    'basename': base_name
                }
                
                if self.validate_data and not self._validate_sample_integrity(sample):
                    continue
                
                samples.append(sample)
        
        if missing_files > 0:
            logger.warning(f"Missing depth files: {missing_files}")
        
        if not samples:
            raise RuntimeError("No valid samples found")
        
        split_samples = self._get_split_samples(samples)
        
        logger.info(f"Loaded {len(split_samples)} valid samples from {len(dataset_dirs)} datasets")
        return split_samples
    
    def _get_split_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.split == 'train':
            return samples[:int(len(samples) * 0.8)]
        elif self.split == 'val':
            return samples[int(len(samples) * 0.8):int(len(samples) * 0.9)]
        else:
            return samples[int(len(samples) * 0.9):]
    
    def _validate_sample_integrity(self, sample: Dict[str, Any]) -> bool:
        try:
            required_files = [sample['rgb'], sample['depth']]
            
            for file_path in required_files:
                if not file_path.exists() or file_path.stat().st_size < 1024:
                    logger.warning(f"Invalid file: {file_path}")
                    return False
            
            with Image.open(sample['rgb']) as img:
                img.verify()
            
            return True
            
        except Exception as e:
            logger.error(f"Sample integrity check failed: {e}")
            return False
    
    def _load_rgb(self, path: Path) -> np.ndarray:
        if self.cache and path in self._cache_data: #type: ignore 
            return self._cache_data[path] #type: ignore 
        
        try:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                raise IOError(f"Unable to read RGB image: {path}")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if img.shape[:2] != self.img_size:
                img = cv2.resize(img, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_LINEAR)
            
            if self.cache:
                self._cache_data[path] = img #type: ignore 
            
            return img
        except Exception as e:
            logger.error(f"Failed to load RGB image {path}: {e}")
            raise
    
    def _load_depth(self, path: Path) -> np.ndarray:
        if self.cache and path in self._cache_data: #type: ignore 
            return self._cache_data[path] #type: ignore 
        
        try:
            import OpenEXR #type: ignore 
            import Imath #type: ignore 
            
            exr_file = OpenEXR.InputFile(str(path))
            header = exr_file.header()
            
            dw = header['dataWindow']
            size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
            
            FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
            depth_str = exr_file.channel('R', FLOAT)
            depth = np.frombuffer(depth_str, dtype=np.float32)
            depth = depth.reshape(size[1], size[0])
            
            if depth.shape != self.img_size:
                depth = cv2.resize(depth, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)
            
            depth = np.clip(depth, self.min_depth, self.max_depth)
            
            invalid_mask = (depth <= 0) | np.isnan(depth) | np.isinf(depth)
            depth[invalid_mask] = 0.0
            
            if self.cache:
                self._cache_data[path] = depth #type: ignore 
            
            return depth
        except ImportError:
            logger.warning("OpenEXR not available, trying PIL fallback")
            return self._load_depth_fallback(path)
        except Exception as e:
            logger.error(f"Failed to load depth {path}: {e}")
            raise
    
    def _load_depth_fallback(self, path: Path) -> np.ndarray:
        try:
            depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise IOError(f"Unable to read depth image: {path}")
            
            if depth.dtype != np.float32:
                depth = depth.astype(np.float32)
            
            if depth.shape != self.img_size:
                depth = cv2.resize(depth, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)
            
            depth = np.clip(depth, self.min_depth, self.max_depth)
            
            invalid_mask = (depth <= 0) | np.isnan(depth) | np.isinf(depth)
            depth[invalid_mask] = 0.0
            
            return depth
        except Exception as e:
            logger.error(f"Fallback depth loading failed {path}: {e}")
            raise
    
    def _create_valid_depth_mask(self, depth: np.ndarray) -> np.ndarray:
        return ((depth > self.min_depth) & 
                (depth < self.max_depth) & 
                (~np.isnan(depth)) & 
                (~np.isinf(depth))).astype(bool)
    
    def _apply_domain_adaptation(self, rgb: np.ndarray, depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.domain_adaptation:
            return rgb, depth
        
        noise_std = random.uniform(0.01, 0.05)
        noise = np.random.normal(0, noise_std, depth.shape).astype(np.float32)
        depth = np.clip(depth + noise, self.min_depth, self.max_depth)
        
        brightness_factor = random.uniform(0.8, 1.2)
        contrast_factor = random.uniform(0.8, 1.2)
        
        rgb = rgb.astype(np.float32) / 255.0
        rgb = np.clip(rgb * brightness_factor, 0, 1)
        rgb = np.clip((rgb - 0.5) * contrast_factor + 0.5, 0, 1)
        rgb = (rgb * 255).astype(np.uint8)
        
        return rgb, depth
    
    def _apply_synchronized_transforms(self, rgb: np.ndarray, depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.augmentation:
            return rgb, depth
        
        if random.random() > 0.5:
            rgb = np.fliplr(rgb)
            depth = np.fliplr(depth)
        
        return rgb, depth
    
    def _build_rgb_transform(self):
        transforms = []
        
        if self.augmentation:
            transforms.extend([
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.RandomHorizontalFlip(p=0.5)
            ])
        
        transforms.extend([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return T.Compose(transforms)
    
    def _build_depth_transform(self):
        return T.Compose([T.ToTensor()])
    
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
            
            rgb = self._load_rgb(sample['rgb'])
            depth = self._load_depth(sample['depth'])
            
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
            'dataset': sample['dataset'],
            'basename': sample['basename']
        }
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds")
        
        sample = self.samples[idx]
        
        rgb = self._load_rgb(sample['rgb'])
        depth = self._load_depth(sample['depth'])
        
        if self.domain_adaptation:
            rgb, depth = self._apply_domain_adaptation(rgb, depth)
        
        if self.augmentation:
            rgb, depth = self._apply_synchronized_transforms(rgb, depth)
        
        rgb_tensor = self.rgb_transform(rgb)
        depth_tensor = self.depth_transform(depth)
        
        if depth_tensor.dim() == 2: #type: ignore 
            depth_tensor = depth_tensor.unsqueeze(0) #type: ignore 
        
        valid_mask = self._create_valid_depth_mask(depth)
        
        return {
            'rgb': rgb_tensor,
            'depth': depth_tensor,
            'valid_mask': T.ToTensor()(valid_mask.astype(np.float32)), #type: ignore 
            'dataset': sample['dataset'],
            'basename': sample['basename']
        }