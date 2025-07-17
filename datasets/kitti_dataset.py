# FILE: datasets/kitti_dataset.py
# ehsanasgharzde - KITTI DATASET
# hosseinsolymanzadeh - PROPER COMMENTING

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
        
        self.data_root = data_root  # Path to the dataset root directory
        self.split = split  # Dataset split: 'train', 'val', or 'test'
        self.img_size = img_size  # Desired image size (height, width)
        self.min_depth = min_depth  # Minimum depth value threshold
        self.max_depth = max_depth  # Maximum depth value threshold
        self.depth_scale = depth_scale  # Scale factor for depth values
        self.augmentation = augmentation  # Whether to apply data augmentation
        self.cache = cache  # Whether to cache the loaded data in memory
        self.use_dense_depth = use_dense_depth  # Whether to use dense depth maps
        
        self._validate_initialization_parameters()  # Check if the provided parameters are valid
        self._validate_dataset_structure()  # Check if the dataset directory structure is valid
        
        self.samples = self._load_samples()  # Load the list of samples (e.g., file paths)
        
        self._cache_data = {} if self.cache else None  # Initialize in-memory cache if enabled
        
        self.rgb_transform = self._build_rgb_transform()  # Define transformation pipeline for RGB images
        self.depth_transform = self._build_depth_transform()  # Define transformation pipeline for depth maps
        
        self.eval_crop_mask = self._create_eval_crop_mask()  # Create cropping mask for evaluation
        
        logger.info(f"KITTI dataset initialized: {len(self.samples)} samples, split={split}")  # Log dataset loading info

    def _validate_initialization_parameters(self):
        # Check that data_root is a valid directory
        if not isinstance(self.data_root, str) or not os.path.isdir(self.data_root):
            raise ValueError(f"'data_root' must be an existing directory. Got: {self.data_root}")
        
        valid_splits = ['train', 'val', 'test']  # Allowed dataset splits
        if self.split not in valid_splits:
            raise ValueError(f"'split' must be one of {valid_splits}. Got: {self.split}")
        
        # Ensure img_size is a tuple of two positive integers
        if (not isinstance(self.img_size, tuple) or len(self.img_size) != 2 or 
            not all(isinstance(dim, int) and dim > 0 for dim in self.img_size)):
            raise ValueError(f"'img_size' must be a tuple of two positive integers. Got: {self.img_size}")
        
        # Ensure min_depth and max_depth are valid and min_depth < max_depth
        if not (isinstance(self.min_depth, (int, float)) and 
                isinstance(self.max_depth, (int, float)) and 
                0 < self.min_depth < self.max_depth):
            raise ValueError(f"'min_depth' must be less than 'max_depth' and both must be > 0. Got: min_depth={self.min_depth}, max_depth={self.max_depth}")
        
        # Ensure depth_scale is a positive number
        if not (isinstance(self.depth_scale, (int, float)) and self.depth_scale > 0):
            raise ValueError(f"'depth_scale' must be a positive number. Got: {self.depth_scale}")

    def _validate_dataset_structure(self):
        # Construct the path to the 'sequences' directory
        sequences_path = os.path.join(self.data_root, 'sequences')
        # Check if the 'sequences' directory exists
        if not os.path.isdir(sequences_path):
            raise FileNotFoundError(f"Expected KITTI sequences directory not found: {sequences_path}")
        
        # Construct the path to the 'calib' (calibration) directory
        calib_path = os.path.join(self.data_root, 'calib')
        # Check if the 'calib' directory exists
        if not os.path.isdir(calib_path):
            raise FileNotFoundError(f"Calibration directory missing: {calib_path}")

    def _load_samples(self):
        samples = []  # List to store valid samples
        sequences_path = os.path.join(self.data_root, 'sequences')  # Base path for sequences
        
        # Check again in case sequences directory is missing
        if not os.path.isdir(sequences_path):
            logger.error(f"Sequences directory not found: {sequences_path}")
            return samples  # Return empty list if missing
        
        split_sequences = self._get_split_sequences()  # Get list of sequences for the current split
        missing_depth_count = 0  # Counter for images missing corresponding depth files
        
        for seq in split_sequences:
            # Path to the RGB images for the current sequence
            img_dir = os.path.join(sequences_path, seq, 'image_02')
            
            # Choose the appropriate depth directory based on configuration
            if self.use_dense_depth:
                depth_dir = os.path.join(sequences_path, seq, 'proj_depth', 'groundtruth')
            else:
                depth_dir = os.path.join(sequences_path, seq, 'proj_depth', 'velodyne_raw')
            
            # Skip this sequence if image or depth directories are missing
            if not os.path.isdir(img_dir) or not os.path.isdir(depth_dir):
                logger.warning(f"Missing directories for sequence {seq}")
                continue
            
            # Load and sort all PNG image files
            img_files = sorted(glob.glob(os.path.join(img_dir, '*.png')))
            # Load and sort all PNG depth files (may not be complete)
            depth_files = sorted(glob.glob(os.path.join(depth_dir, '*.png')))
            
            for img_path in img_files:
                base_name = os.path.splitext(os.path.basename(img_path))[0]  # Extract filename without extension
                depth_path = os.path.join(depth_dir, base_name + '.png')  # Expected matching depth file
                
                # Skip if the corresponding depth file does not exist
                if not os.path.isfile(depth_path):
                    missing_depth_count += 1
                    continue
                
                # Calibration file path for the current sequence
                calib_path = os.path.join(self.data_root, 'calib', f'{seq}.txt')
                
                # Append a valid sample dictionary to the list
                samples.append({
                    'rgb': img_path,           # Path to the RGB image
                    'depth': depth_path,       # Path to the depth map
                    'calib': calib_path,       # Path to calibration data
                    'sequence': seq,           # Sequence ID
                    'basename': base_name      # Image base filename
                })
        
        # Log the total number of valid samples and missing depth files
        logger.info(f"Loaded {len(samples)} samples, missing depth files: {missing_depth_count}")
        return samples  # Return the full list of valid samples

    def _get_split_sequences(self):
        # Path to the 'sequences' directory
        sequences_path = os.path.join(self.data_root, 'sequences')
        
        # Get all subdirectories in the 'sequences' directory (i.e., sequence names)
        all_sequences = sorted([d for d in os.listdir(sequences_path) 
                               if os.path.isdir(os.path.join(sequences_path, d))])
        
        # Return 80% of sequences for training
        if self.split == 'train':
            return all_sequences[:int(len(all_sequences) * 0.8)]
        # Return 10% of sequences for validation
        elif self.split == 'val':
            return all_sequences[int(len(all_sequences) * 0.8):int(len(all_sequences) * 0.9)]
        # Return last 10% of sequences for testing
        else:
            return all_sequences[int(len(all_sequences) * 0.9):]

    def _load_rgb(self, path):
        try:
            # Return cached RGB image if available
            if self.cache and path in self._cache_data:
                return self._cache_data[path]['rgb']  # type: ignore 
            
            # Load image and convert to RGB mode
            img = Image.open(path).convert('RGB')
            
            # Resize image if it does not match target dimensions
            if img.size != (self.img_size[1], self.img_size[0]):
                img = img.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)  # type: ignore 
            
            # Cache the loaded RGB image if caching is enabled
            if self.cache:
                if path not in self._cache_data:
                    self._cache_data[path] = {}  # type: ignore 
                self._cache_data[path]['rgb'] = img  # type: ignore 
            
            return img  # Return processed image
        except Exception as e:
            # Log error and return None if loading fails
            logger.error(f"Failed to load RGB image at {path}: {e}")
            return None

    def _load_depth(self, path):
        try:
            # Return cached depth if available
            if self.cache and path in self._cache_data:
                return self._cache_data[path]['depth']  # type: ignore 
            
            # Load depth PNG image and convert to float32 NumPy array
            depth_png = Image.open(path)
            depth_np = np.array(depth_png).astype(np.float32)
            
            # Scale the depth values according to provided scale factor
            depth = depth_np / self.depth_scale
            
            # Resize if depth shape does not match the target dimensions
            if depth.shape != self.img_size:
                depth = Image.fromarray(depth).resize((self.img_size[1], self.img_size[0]), Image.NEAREST)  # type: ignore 
                depth = np.array(depth)
            
            # Cache the loaded depth map if caching is enabled
            if self.cache:
                if path not in self._cache_data:
                    self._cache_data[path] = {}  # type: ignore 
                self._cache_data[path]['depth'] = depth  # type: ignore 
            
            return depth  # Return processed depth array
        except Exception as e:
            # Log error and return None if loading fails
            logger.error(f"Failed to load depth image at {path}: {e}")
            return None

    def _create_valid_depth_mask(self, depth):
        # Return None if depth is not provided
        if depth is None:
            return None
        
        # Create a boolean mask for valid depth values
        mask = (
            (depth > self.min_depth) &             # Must be greater than min_depth
            (depth < self.max_depth) &             # Must be less than max_depth
            (~np.isnan(depth)) &                   # Must not be NaN
            (~np.isinf(depth))                     # Must not be infinite
        )
        
        # Optionally apply evaluation crop mask
        if self.eval_crop_mask is not None:
            mask = mask & self.eval_crop_mask      # Combine with evaluation crop mask
        
        return mask.astype(bool)  # Ensure result is boolean array

    def _create_eval_crop_mask(self):
        # Get image height and width from configuration
        crop_h, crop_w = self.img_size
        
        # Initialize mask with all False
        mask = np.zeros((crop_h, crop_w), dtype=bool)
        
        # Define crop boundaries as fractions of image dimensions
        top = int(crop_h * 0.40810811)
        left = int(crop_w * 0.03594771)
        bottom = int(crop_h * 0.99189189)
        right = int(crop_w * 0.96405229)
        
        # Set the cropped region to True (valid area)
        mask[top:bottom, left:right] = True
        return mask  # Return the evaluation crop mask

    def _build_rgb_transform(self):
        transforms = []  # Initialize list of transformations
        
        # Convert PIL image to tensor
        transforms.append(T.ToTensor())
        
        # Apply data augmentation during training
        if self.split == 'train' and self.augmentation:
            transforms.extend([
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random color jitter
                T.RandomHorizontalFlip(p=0.5)  # 50% chance of horizontal flip
            ])
        
        # Normalize with ImageNet mean and std
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transforms.append(T.Normalize(mean=mean, std=std))
        
        return T.Compose(transforms)  # Compose and return transformation pipeline

    def _build_depth_transform(self):
        # Only convert depth maps to tensor; no augmentation or normalization
        return T.Compose([T.ToTensor()])

    def compute_statistics(self):
        rgb_sums = np.zeros(3)        # Sum of RGB channel means
        rgb_sq_sums = np.zeros(3)     # Sum of squared RGB channel means
        pixel_count = 0               # Counter for number of processed images
        depth_values = []            # List to collect valid depth values
        
        sample_count = min(len(self.samples), 1000)  # Limit to first 1000 samples for speed
        
        for i in range(sample_count):
            sample = self.samples[i]
            rgb = self._load_rgb(sample['rgb'])      # Load RGB image
            depth = self._load_depth(sample['depth'])# Load depth map
            
            if rgb is None or depth is None:
                continue  # Skip if image or depth couldn't be loaded
            
            rgb_np = np.array(rgb) / 255.0           # Normalize RGB to [0, 1]
            rgb_sums += np.mean(rgb_np, axis=(0,1))  # Add per-channel mean
            rgb_sq_sums += np.mean(rgb_np ** 2, axis=(0,1))  # Add per-channel squared mean
            pixel_count += 1
            
            mask = self._create_valid_depth_mask(depth)  # Create valid depth mask
            if mask is not None:
                valid_depth = depth[mask]                # Extract valid depth values
                if valid_depth.size > 0:
                    depth_values.append(valid_depth)     # Add to list
        
        # Compute mean and std if at least one image was processed
        if pixel_count > 0:
            rgb_mean = rgb_sums / pixel_count
            rgb_std = np.sqrt(rgb_sq_sums / pixel_count - rgb_mean ** 2)
        else:
            rgb_mean = np.zeros(3)
            rgb_std = np.zeros(3)
        
        # Compute min, max, and percentiles of depth if data was collected
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
        
        # Return computed statistics as dictionary
        return {
            'rgb_mean': rgb_mean.tolist(),
            'rgb_std': rgb_std.tolist(),
            'depth_min': depth_min,
            'depth_max': depth_max,
            'depth_percentiles': depth_percentiles,
            'num_samples': len(self.samples)
        }

    def __len__(self):
        return len(self.samples)  # Return number of samples in dataset

    def __getitem__(self, idx):
        # Check for out-of-bounds index
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self)}")
        
        sample = self.samples[idx]  # Get sample dictionary by index
        
        rgb = self._load_rgb(sample['rgb'])      # Load RGB image
        depth = self._load_depth(sample['depth'])# Load depth map
        
        # Raise error if loading failed
        if rgb is None or depth is None:
            logger.error(f"Failed to load sample at index {idx}")
            raise RuntimeError(f"Failed to load sample at index {idx}")
        
        rgb_tensor = self.rgb_transform(rgb)       # Apply RGB transform
        depth_tensor = self.depth_transform(depth) # Apply depth transform
        
        valid_mask = self._create_valid_depth_mask(depth)  # Generate valid mask
        mask_tensor = T.ToTensor()(valid_mask.astype(np.float32)) if valid_mask is not None else None
        
        # Return dictionary containing input tensors and metadata
        return {
            'rgb': rgb_tensor,
            'depth': depth_tensor,
            'mask': mask_tensor,
            'sequence': sample['sequence'],
            'basename': sample['basename']
        }