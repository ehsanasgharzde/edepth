# FILE: datasets/kitti_dataset.py
# ehsanasgharzde - KITTI DATASET

import os
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import json
import logging

logger = logging.getLogger(__name__)

class KITTIDataset(Dataset):
    def __init__(self, data_root, split='train', img_size=(352, 1216),
                 min_depth=1e-3, max_depth=80.0, depth_scale=256.0,
                 augmentation=True, cache=False, use_dense_depth=False):
        super().__init__()
        
        self.data_root = data_root
        self.split = split
        self.img_size = img_size
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.depth_scale = depth_scale
        self.augmentation = augmentation
        self.cache = cache
        self.use_dense_depth = use_dense_depth
        
        self._validate_initialization_parameters()
        self._validate_dataset_structure()
        
        self.samples = self._load_samples()
        
        self._cache_data = {} if self.cache else None
        
        self.rgb_transform = self._build_rgb_transform()
        self.depth_transform = self._build_depth_transform()
        
        self.eval_crop_mask = self._create_eval_crop_mask()
        
        logger.info(f"KITTI dataset initialized: {len(self.samples)} samples, split={split}")

    def _validate_initialization_parameters(self):
        if not isinstance(self.data_root, str) or not os.path.isdir(self.data_root):
            raise ValueError(f"'data_root' must be an existing directory. Got: {self.data_root}")
        
        valid_splits = ['train', 'val', 'test']
        if self.split not in valid_splits:
            raise ValueError(f"'split' must be one of {valid_splits}. Got: {self.split}")
        
        if (not isinstance(self.img_size, tuple) or len(self.img_size) != 2 or 
            not all(isinstance(dim, int) and dim > 0 for dim in self.img_size)):
            raise ValueError(f"'img_size' must be a tuple of two positive integers. Got: {self.img_size}")
        
        if not (isinstance(self.min_depth, (int, float)) and 
                isinstance(self.max_depth, (int, float)) and 
                0 < self.min_depth < self.max_depth):
            raise ValueError(f"'min_depth' must be less than 'max_depth' and both must be > 0. Got: min_depth={self.min_depth}, max_depth={self.max_depth}")
        
        if not (isinstance(self.depth_scale, (int, float)) and self.depth_scale > 0):
            raise ValueError(f"'depth_scale' must be a positive number. Got: {self.depth_scale}")

    def _validate_dataset_structure(self):
        sequences_path = os.path.join(self.data_root, 'sequences')
        if not os.path.isdir(sequences_path):
            raise FileNotFoundError(f"Expected KITTI sequences directory not found: {sequences_path}")
        
        calib_path = os.path.join(self.data_root, 'calib')
        if not os.path.isdir(calib_path):
            raise FileNotFoundError(f"Calibration directory missing: {calib_path}")

    def _load_samples(self):
        samples = []
        sequences_path = os.path.join(self.data_root, 'sequences')
        
        if not os.path.isdir(sequences_path):
            logger.error(f"Sequences directory not found: {sequences_path}")
            return samples
        
        split_sequences = self._get_split_sequences()
        missing_depth_count = 0
        
        for seq in split_sequences:
            img_dir = os.path.join(sequences_path, seq, 'image_02')
            
            if self.use_dense_depth:
                depth_dir = os.path.join(sequences_path, seq, 'proj_depth', 'groundtruth')
            else:
                depth_dir = os.path.join(sequences_path, seq, 'proj_depth', 'velodyne_raw')
            
            if not os.path.isdir(img_dir) or not os.path.isdir(depth_dir):
                logger.warning(f"Missing directories for sequence {seq}")
                continue
            
            img_files = sorted(glob.glob(os.path.join(img_dir, '*.png')))
            depth_files = sorted(glob.glob(os.path.join(depth_dir, '*.png')))
            
            for img_path in img_files:
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                depth_path = os.path.join(depth_dir, base_name + '.png')
                
                if not os.path.isfile(depth_path):
                    missing_depth_count += 1
                    continue
                
                calib_path = os.path.join(self.data_root, 'calib', f'{seq}.txt')
                
                samples.append({
                    'rgb': img_path,
                    'depth': depth_path,
                    'calib': calib_path,
                    'sequence': seq,
                    'basename': base_name
                })
        
        logger.info(f"Loaded {len(samples)} samples, missing depth files: {missing_depth_count}")
        return samples

    def _get_split_sequences(self):
        sequences_path = os.path.join(self.data_root, 'sequences')
        all_sequences = sorted([d for d in os.listdir(sequences_path) 
                               if os.path.isdir(os.path.join(sequences_path, d))])
        
        if self.split == 'train':
            return all_sequences[:int(len(all_sequences) * 0.8)]
        elif self.split == 'val':
            return all_sequences[int(len(all_sequences) * 0.8):int(len(all_sequences) * 0.9)]
        else:
            return all_sequences[int(len(all_sequences) * 0.9):]

    def _load_rgb(self, path):
        try:
            if self.cache and path in self._cache_data:
                return self._cache_data[path]['rgb'] #type: ignore 
            
            img = Image.open(path).convert('RGB')
            
            if img.size != (self.img_size[1], self.img_size[0]):
                img = img.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR) #type: ignore 
            
            if self.cache:
                if path not in self._cache_data:
                    self._cache_data[path] = {} #type: ignore 
                self._cache_data[path]['rgb'] = img #type: ignore 
            
            return img
        except Exception as e:
            logger.error(f"Failed to load RGB image at {path}: {e}")
            return None

    def _load_depth(self, path):
        try:
            if self.cache and path in self._cache_data:
                return self._cache_data[path]['depth'] #type: ignore 
            
            depth_png = Image.open(path)
            depth_np = np.array(depth_png).astype(np.float32)
            depth = depth_np / self.depth_scale
            
            if depth.shape != self.img_size:
                depth = Image.fromarray(depth).resize((self.img_size[1], self.img_size[0]), Image.NEAREST) #type: ignore 
                depth = np.array(depth)
            
            if self.cache:
                if path not in self._cache_data:
                    self._cache_data[path] = {} #type: ignore 
                self._cache_data[path]['depth'] = depth #type: ignore 
            
            return depth
        except Exception as e:
            logger.error(f"Failed to load depth image at {path}: {e}")
            return None

    def _create_valid_depth_mask(self, depth):
        if depth is None:
            return None
        
        mask = (
            (depth > self.min_depth) & 
            (depth < self.max_depth) & 
            (~np.isnan(depth)) & 
            (~np.isinf(depth))
        )
        
        if self.eval_crop_mask is not None:
            mask = mask & self.eval_crop_mask
        
        return mask.astype(bool)

    def _create_eval_crop_mask(self):
        crop_h, crop_w = self.img_size
        mask = np.zeros((crop_h, crop_w), dtype=bool)
        
        top = int(crop_h * 0.40810811)
        left = int(crop_w * 0.03594771)
        bottom = int(crop_h * 0.99189189)
        right = int(crop_w * 0.96405229)
        
        mask[top:bottom, left:right] = True
        return mask

    def _build_rgb_transform(self):
        transforms = []
        transforms.append(T.ToTensor())
        
        if self.split == 'train' and self.augmentation:
            transforms.extend([
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.RandomHorizontalFlip(p=0.5)
            ])
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transforms.append(T.Normalize(mean=mean, std=std))
        
        return T.Compose(transforms)

    def _build_depth_transform(self):
        return T.Compose([T.ToTensor()])

    def compute_statistics(self):
        rgb_sums = np.zeros(3)
        rgb_sq_sums = np.zeros(3)
        pixel_count = 0
        depth_values = []
        
        sample_count = min(len(self.samples), 1000)
        
        for i in range(sample_count):
            sample = self.samples[i]
            rgb = self._load_rgb(sample['rgb'])
            depth = self._load_depth(sample['depth'])
            
            if rgb is None or depth is None:
                continue
            
            rgb_np = np.array(rgb) / 255.0
            rgb_sums += np.mean(rgb_np, axis=(0,1))
            rgb_sq_sums += np.mean(rgb_np ** 2, axis=(0,1))
            pixel_count += 1
            
            mask = self._create_valid_depth_mask(depth)
            if mask is not None:
                valid_depth = depth[mask]
                if valid_depth.size > 0:
                    depth_values.append(valid_depth)
        
        if pixel_count > 0:
            rgb_mean = rgb_sums / pixel_count
            rgb_std = np.sqrt(rgb_sq_sums / pixel_count - rgb_mean ** 2)
        else:
            rgb_mean = np.zeros(3)
            rgb_std = np.zeros(3)
        
        if depth_values:
            depth_values = np.concatenate(depth_values)
            depth_min = float(np.min(depth_values))
            depth_max = float(np.max(depth_values))
            depth_percentiles = {
                25: float(np.percentile(depth_values, 25)),
                50: float(np.percentile(depth_values, 50)),
                75: float(np.percentile(depth_values, 75))
            }
        else:
            depth_min = depth_max = 0.0
            depth_percentiles = {25: 0.0, 50: 0.0, 75: 0.0}
        
        return {
            'rgb_mean': rgb_mean.tolist(),
            'rgb_std': rgb_std.tolist(),
            'depth_min': depth_min,
            'depth_max': depth_max,
            'depth_percentiles': depth_percentiles,
            'num_samples': len(self.samples)
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self)}")
        
        sample = self.samples[idx]
        
        rgb = self._load_rgb(sample['rgb'])
        depth = self._load_depth(sample['depth'])
        
        if rgb is None or depth is None:
            logger.error(f"Failed to load sample at index {idx}")
            raise RuntimeError(f"Failed to load sample at index {idx}")
        
        rgb_tensor = self.rgb_transform(rgb)
        depth_tensor = self.depth_transform(depth)
        
        valid_mask = self._create_valid_depth_mask(depth)
        mask_tensor = T.ToTensor()(valid_mask.astype(np.float32)) if valid_mask is not None else None
        
        return {
            'rgb': rgb_tensor,
            'depth': depth_tensor,
            'mask': mask_tensor,
            'sequence': sample['sequence'],
            'basename': sample['basename']
        }