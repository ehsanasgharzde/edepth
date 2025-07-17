# FILE: datasets/nyu_dataset.py
# ehsanasgharzde - NYU DEPTH V2 DATASET
# hosseinsolymanzadeh - PROPER COMMENTING

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
    # Initialize the dataset object with various parameters for data loading and processing
    def __init__(self, data_root: str, split: str = 'train', img_size: Tuple[int, int] = (480, 640), 
                 min_depth: float = 0.1, max_depth: float = 10.0, depth_scale: float = 1000.0, 
                 augmentation: bool = True, cache: bool = False, validate_data: bool = True):
        super().__init__()
        
        # Path to the root directory of the dataset
        self.data_root = Path(data_root)
        # Dataset split: train, val, or test
        self.split = split
        # Target image size (height, width)
        self.img_size = img_size
        # Minimum depth value to consider
        self.min_depth = min_depth
        # Maximum depth value to consider
        self.max_depth = max_depth
        # Scaling factor for depth values
        self.depth_scale = depth_scale
        # Whether to apply data augmentation (only if split is 'train')
        self.augmentation = augmentation and split == 'train'
        # Whether to cache samples in memory
        self.cache = cache
        # Whether to validate dataset files and structure
        self.validate_data = validate_data
        
        # Validate the initial parameters for correctness
        self._validate_initialization_parameters()
        
        # Directory containing RGB images for the given split
        self.rgb_dir = self.data_root / split / 'rgb'
        # Directory containing depth images for the given split
        self.depth_dir = self.data_root / split / 'depth'
        
        # If enabled, check the dataset directory structure and files
        if validate_data:
            self._validate_dataset_structure()
            
        # Load list of all data samples (pairs of RGB and depth images)
        self.samples = self._load_samples()
        
        # Define transformation pipeline for RGB images
        self.rgb_transform = self._get_rgb_transform()
        # Define transformation pipeline for depth images
        self.depth_transform = self._get_depth_transform()
        
        # Initialize cache dictionary if caching is enabled
        self._cache = {} if cache else None
        
        # Log info about dataset initialization
        logger.info(f"Initialized NYU Depth V2 dataset: {len(self.samples)} samples for {split} split")
    
    # Check that initialization parameters are valid and consistent
    def _validate_initialization_parameters(self):
        # Check if the root data directory exists
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root not found: {self.data_root}")
        
        # Validate the split name
        if self.split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split: {self.split}")
        
        # Check that image size is a tuple of length 2
        if not isinstance(self.img_size, tuple) or len(self.img_size) != 2:
            raise ValueError("img_size must be tuple of (height, width)")
        
        # Check that min_depth is positive and less than max_depth
        if not (0 < self.min_depth < self.max_depth):
            raise ValueError("min_depth must be less than max_depth")
        
        # Ensure depth scaling factor is positive
        if self.depth_scale <= 0:
            raise ValueError("depth_scale must be positive")
    
    # Check that the dataset directories for RGB and depth images exist and contain files
    def _validate_dataset_structure(self):
        # Verify RGB directory exists
        if not self.rgb_dir.exists():
            raise FileNotFoundError(f"RGB directory not found: {self.rgb_dir}")
        
        # Verify depth directory exists
        if not self.depth_dir.exists():
            raise FileNotFoundError(f"Depth directory not found: {self.depth_dir}")
        
        # Gather all RGB images with extensions jpg or png
        rgb_files = list(self.rgb_dir.glob('*.jpg')) + list(self.rgb_dir.glob('*.png'))
        # Gather all depth images with extension png
        depth_files = list(self.depth_dir.glob('*.png'))
        
        # Ensure there is at least one RGB image
        if not rgb_files:
            raise FileNotFoundError(f"No RGB images found in {self.rgb_dir}")
        
        # Ensure there is at least one depth image
        if not depth_files:
            raise FileNotFoundError(f"No depth images found in {self.depth_dir}")
    
    # Load paired RGB and depth samples into a list of dictionaries
    def _load_samples(self) -> List[Dict[str, Path]]:
        # Sort RGB files for consistent ordering
        rgb_files = sorted(list(self.rgb_dir.glob('*.jpg')) + list(self.rgb_dir.glob('*.png')))
        samples = []
        missing_depth = 0
        
        # Iterate over all RGB images and find corresponding depth image
        for rgb_file in rgb_files:
            depth_file = self.depth_dir / (rgb_file.stem + '.png')
            # Skip if corresponding depth file does not exist
            if not depth_file.exists():
                missing_depth += 1
                continue
            
            # Validate integrity of sample files if enabled
            if self.validate_data and not self._validate_sample_integrity(rgb_file, depth_file):
                continue
                
            # Append valid sample as dictionary with paths and basename
            samples.append({
                'rgb': rgb_file,
                'depth': depth_file,
                'basename': rgb_file.stem
            })
        
        # Log warning if any depth files were missing
        if missing_depth > 0:
            logger.warning(f"Missing depth files: {missing_depth}")
        
        # Raise error if no valid samples found
        if not samples:
            raise RuntimeError("No valid RGB-depth pairs found")
        
        # Log how many valid samples were loaded
        logger.info(f"Loaded {len(samples)} valid samples")
        return samples
    
    # Check the integrity of RGB and depth image files to ensure they are valid and readable
    def _validate_sample_integrity(self, rgb_path: Path, depth_path: Path) -> bool:
        try:
            # Check that file sizes are above a small threshold (e.g., 1 KB)
            if rgb_path.stat().st_size < 1024 or depth_path.stat().st_size < 1024:
                logger.warning(f"File too small: {rgb_path} or {depth_path}")
                return False
            
            # Verify that RGB image can be opened without error
            with Image.open(rgb_path) as rgb_img:
                rgb_img.verify()
            
            # Verify that depth image can be opened without error
            with Image.open(depth_path) as depth_img:
                depth_img.verify()
            
            return True
            
        except Exception as e:
            # Log error if verification fails
            logger.error(f"Sample integrity check failed: {rgb_path}, {depth_path} - {e}")
            return False
    
    # Define the transformation pipeline applied to RGB images
    def _get_rgb_transform(self):
        # Start with resizing to target image size
        transforms = [A.Resize(self.img_size[0], self.img_size[1], interpolation=1)]
        
        # Add data augmentation transforms only if enabled
        if self.augmentation:
            transforms.extend([
                A.HorizontalFlip(p=0.5),                           # Random horizontal flip with 50% chance
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),  # Random brightness/contrast
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5), # Color jittering
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),         # Gaussian blur with probability 30%
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),       # Add Gaussian noise with probability 20%
            ])
        
        # Normalize using NYU dataset mean and std, then convert to tensor
        transforms.extend([
            A.Normalize(mean=NYU_MEAN, std=NYU_STD), #type: ignore 
            ToTensorV2()
        ])
        
        # Compose all transformations into one pipeline
        return A.Compose(transforms) #type: ignore 
    
    # Define the transformation pipeline applied to depth images
    def _get_depth_transform(self):
        # Resize depth image to target size using nearest neighbor interpolation (interpolation=0)
        transforms = [A.Resize(self.img_size[0], self.img_size[1], interpolation=0)]
        
        # Add horizontal flip augmentation if enabled
        if self.augmentation:
            transforms.append(A.HorizontalFlip(p=0.5)) #type: ignore 
        
        # Convert to tensor after transformations
        transforms.append(ToTensorV2()) #type: ignore 
        return A.Compose(transforms) #type: ignore 
    
    # Clip depth values to the configured min and max depth range
    def _clip_and_scale_depth(self, depth: np.ndarray) -> np.ndarray:
        return np.clip(depth, self.min_depth, self.max_depth)
    
    # Load an RGB image from disk, optionally caching it
    def _load_rgb(self, path: Path) -> np.ndarray:
        # Return cached image if caching enabled and image exists in cache
        if self.cache and path in self._cache: #type: ignore 
            return self._cache[path] #type: ignore 
        
        # Read the image using OpenCV in BGR format
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Unable to read RGB image: {path}")
        
        # Convert from BGR to RGB color space
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Cache the image if caching is enabled
        if self.cache:
            self._cache[path] = img #type: ignore 
        
        return img
    
    # Load a depth image from disk, scale and clip it, optionally caching it
    def _load_depth(self, path: Path) -> np.ndarray:
        # Return cached depth if caching enabled and depth exists in cache
        if self.cache and path in self._cache: #type: ignore 
            return self._cache[path] #type: ignore 
        
        # Read depth image preserving original values (16-bit or 32-bit)
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise IOError(f"Unable to read depth image: {path}")
        
        # Convert to float32 and scale by depth_scale factor
        depth = depth.astype(np.float32) / self.depth_scale
        # Clip depth values to valid range
        depth = np.clip(depth, self.min_depth, self.max_depth)
        
        # Mask invalid values (<=0, NaN, Inf) and set them to 0
        invalid_mask = (depth <= 0) | np.isnan(depth) | np.isinf(depth)
        depth[invalid_mask] = 0.0
        
        # Cache the depth if caching is enabled
        if self.cache:
            self._cache[path] = depth #type: ignore 
        
        return depth
    
    # Create a boolean mask indicating valid depth pixels
    def _create_valid_depth_mask(self, depth: np.ndarray) -> np.ndarray:
        return ((depth > 0) & 
                (~np.isnan(depth)) & 
                (~np.isinf(depth)) & 
                (depth >= self.min_depth) & 
                (depth <= self.max_depth)).astype(bool)
    
    # Apply synchronized augmentations/transforms to both RGB and depth images
    def _apply_synchronized_transforms(self, rgb: np.ndarray, depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # If no augmentation, return inputs unchanged
        if not self.augmentation:
            return rgb, depth
        
        # Define augmentations with additional target 'depth' treated as mask
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),                 # Random horizontal flip
            A.RandomRotate90(p=0.5),                  # Random 90 degree rotations
            A.ShiftScaleRotate(                      # Small shift, scale, and rotation
                shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        ], additional_targets={'depth': 'mask'})
        
        try:
            # Apply transformations to both images
            transformed = transform(image=rgb, depth=depth)
            return transformed['image'], transformed['depth']
        except Exception as e:
            # Log error and return unmodified images if transform fails
            logger.error(f"Transform failed: {e}")
            return rgb, depth
    
    # Return the total number of samples in the dataset
    def __len__(self):
        return len(self.samples)
    
    # Get a dataset sample by index, loading and transforming RGB and depth images
    def __getitem__(self, idx):
        # Check for valid index range
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds")
        
        # Retrieve sample metadata (paths and basename)
        sample = self.samples[idx]
        
        # Load RGB and depth images from disk (or cache)
        rgb = self._load_rgb(sample['rgb'])
        depth = self._load_depth(sample['depth'])
        
        # Apply synchronized augmentation transforms if enabled
        if self.augmentation:
            rgb, depth = self._apply_synchronized_transforms(rgb, depth)
        
        # Apply RGB and depth transformations (resize, normalize, to tensor, etc.)
        rgb = self.rgb_transform(image=rgb)['image']
        depth = self.depth_transform(image=depth)['image']
        
        # Ensure depth tensor has channel dimension (C=1)
        if depth.dim() == 2:
            depth = depth.unsqueeze(0)
        
        # Create a valid mask for depth pixels (boolean mask)
        valid_mask = self._create_valid_depth_mask(depth.squeeze(0).numpy())
        
        # Return dictionary containing processed data and metadata
        return {
            'rgb': rgb,
            'depth': depth,
            'valid_mask': torch.from_numpy(valid_mask).unsqueeze(0),
            'rgb_path': str(sample['rgb']),
            'depth_path': str(sample['depth']),
            'basename': sample['basename']
        }
    
    # Retrieve basic info about a sample without applying transforms
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        # Check for valid index range
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds")
        
        # Load sample metadata and raw images
        sample = self.samples[idx]
        rgb = self._load_rgb(sample['rgb'])
        depth = self._load_depth(sample['depth'])
        
        # Return dictionary with image paths, shapes, statistics, and basename
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
    
    # Compute aggregate statistics (mean, std, min, max, percentiles) for RGB and depth data
    def compute_statistics(self) -> Dict[str, Any]:
        rgb_sum = np.zeros(3)
        rgb_sq_sum = np.zeros(3)
        depth_values = []
        pixel_count = 0
        
        logger.info("Computing dataset statistics...")
        
        # Iterate over all samples to accumulate stats
        for i, sample in enumerate(self.samples):
            if i % 100 == 0:
                logger.info(f"Processing sample {i}/{len(self.samples)}")
            
            rgb = self._load_rgb(sample['rgb'])
            depth = self._load_depth(sample['depth'])
            
            # Normalize RGB pixels to [0,1]
            rgb_normalized = rgb.astype(np.float32) / 255.0
            # Sum RGB values and squared RGB values for variance calculation
            rgb_sum += rgb_normalized.reshape(-1, 3).sum(axis=0)
            rgb_sq_sum += (rgb_normalized.reshape(-1, 3) ** 2).sum(axis=0)
            pixel_count += rgb.shape[0] * rgb.shape[1]
            
            # Collect valid depth pixels for statistics
            valid_mask = self._create_valid_depth_mask(depth)
            depth_values.append(depth[valid_mask])
        
        # Concatenate all valid depth values into one array
        depth_values = np.concatenate(depth_values) if depth_values else np.array([])
        
        # Calculate mean and std deviation for RGB channels
        rgb_mean = rgb_sum / pixel_count
        rgb_var = (rgb_sq_sum / pixel_count) - (rgb_mean ** 2)
        rgb_std = np.sqrt(rgb_var)
        
        # Prepare statistics dictionary
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
        
        # Add percentile statistics if depth values are present
        if depth_values.size > 0:
            for percentile in [25, 50, 75, 95]:
                stats[f'depth_p{percentile}'] = float(np.percentile(depth_values, percentile))
        
        logger.info("Statistics computation completed")
        return stats
    
    # Perform a basic validation check on a subset of dataset samples
    def validate_dataset(self) -> Dict[str, Any]:
        logger.info("Validating dataset...")
        
        errors = []
        failed_samples = []
        # Limit number of samples checked for efficiency
        check_count = min(len(self), 100)
        
        # Iterate over samples and check consistency of data shapes
        for i in range(check_count):
            try:
                sample = self.__getitem__(i)
                
                # Check RGB tensor channel dimension (should be 3)
                if sample['rgb'].shape[0] != 3:
                    raise ValueError(f"Invalid RGB shape: {sample['rgb'].shape}")
                
                # Check depth tensor channel dimension (should be 1)
                if sample['depth'].shape[0] != 1:
                    raise ValueError(f"Invalid depth shape: {sample['depth'].shape}")
                
                # Check valid_mask tensor channel dimension (should be 1)
                if sample['valid_mask'].shape[0] != 1:
                    raise ValueError(f"Invalid mask shape: {sample['valid_mask'].shape}")
                
            except Exception as e:
                # Collect errors and failed sample indices
                errors.append(f"Sample {i}: {e}")
                failed_samples.append(i)
        
        # Compile validation summary report
        validation_report = {
            'total_checked': check_count,
            'failed_count': len(failed_samples),
            'failed_indices': failed_samples,
            'errors': errors,
            'status': 'passed' if len(failed_samples) == 0 else 'failed'
        }
        
        logger.info(f"Validation completed: {validation_report['status']}")
        return validation_report