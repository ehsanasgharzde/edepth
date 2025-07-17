# FILE: datasets/unreal_dataset.py
# ehsanasgharzde - UNREALSTEREO4K DATASET
# hosseinsolymanzadeh - PROPER COMMENTING

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
        # Initialize dataset with configuration parameters
        super().__init__()
        
        # Root directory path of dataset
        self.data_root = Path(data_root)
        # Dataset split: 'train', 'val', or 'test'
        self.split = split
        # Target image size (height, width)
        self.img_size = img_size
        # Minimum depth threshold to consider
        self.min_depth = min_depth
        # Maximum depth threshold to consider
        self.max_depth = max_depth
        # Scale factor to apply on depth values
        self.depth_scale = depth_scale
        # Flag to indicate whether stereo image pairs are used
        self.use_stereo = use_stereo
        # Enable data augmentation only for training split
        self.augmentation = augmentation and split == 'train'
        # Whether to cache data in memory for faster access
        self.cache = cache
        # Flag to enable validation of dataset structure and sample integrity
        self.validate_data = validate_data
        # Optional list of scene IDs to restrict dataset loading
        self.scene_ids = scene_ids
        
        # Validate initialization parameters correctness
        self._validate_initialization_parameters()
        
        # Optionally check dataset directory structure and required files
        if validate_data:
            self._validate_dataset_structure()
            
        # Load dataset samples metadata into memory
        self.samples = self._load_samples()
        
        # Prepare image transformations for RGB images
        self.rgb_transform = self._build_rgb_transform()
        # Prepare transformations for depth data
        self.depth_transform = self._build_depth_transform()
        
        # Initialize cache dictionary if caching enabled, else None
        self._cache_data = {} if cache else None
        
        # Log dataset initialization info
        logger.info(f"Initialized UnrealStereo4K dataset: {len(self.samples)} samples for {split} split")
    
    def _validate_initialization_parameters(self):
        # Check that the dataset root directory exists
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root not found: {self.data_root}")
        
        # Ensure split parameter is one of allowed values
        if self.split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split: {self.split}")
        
        # Verify img_size is a tuple of length 2 (height, width)
        if not isinstance(self.img_size, tuple) or len(self.img_size) != 2:
            raise ValueError("img_size must be tuple of (height, width)")
        
        # Check min_depth is positive and less than max_depth
        if not (0 < self.min_depth < self.max_depth):
            raise ValueError("min_depth must be less than max_depth")
        
        # Validate that depth_scale is positive
        if self.depth_scale <= 0:
            raise ValueError("depth_scale must be positive")
    
    def _validate_dataset_structure(self):
        # List scene directories (folders named as digits) under data root
        scene_dirs = [d for d in self.data_root.iterdir() if d.is_dir() and d.name.isdigit()]
        # Raise error if no scene directories found
        if not scene_dirs:
            raise FileNotFoundError(f"No scene directories found in {self.data_root}")
        
        # For first few scene dirs, check presence of required subdirectories
        for scene_dir in scene_dirs[:3]:
            # Required directories common to all: Image0 and Disp0
            required_dirs = ['Image0', 'Disp0']
            # Add stereo counterparts if stereo mode is used
            if self.use_stereo:
                required_dirs.extend(['Image1', 'Disp1'])
            
            # Check each required directory exists inside the scene directory
            for req_dir in required_dirs:
                if not (scene_dir / req_dir).exists():
                    raise FileNotFoundError(f"Required directory missing: {scene_dir / req_dir}")
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        samples = []
        # Collect and sort scene directories (named as digits)
        scene_dirs = sorted([d for d in self.data_root.iterdir() if d.is_dir() and d.name.isdigit()])
        
        # If restricting to specific scene IDs, filter scenes accordingly
        if self.scene_ids:
            scene_dirs = [d for d in scene_dirs if d.name in self.scene_ids]
        
        # Determine which scenes belong to current split
        scene_split = self._get_scene_split(scene_dirs)
        missing_files = 0
        
        # Iterate over scenes in current split
        for scene_dir in scene_split:
            # Directories for left image and disparity (depth) maps
            left_dir = scene_dir / 'Image0'
            disp_dir = scene_dir / 'Disp0'
            
            # Warn and skip if expected directories missing
            if not left_dir.exists() or not disp_dir.exists():
                logger.warning(f"Missing directories in scene {scene_dir.name}")
                continue
            
            # List all left images in scene (JPEG files)
            left_files = sorted(left_dir.glob('*.jpg'))
            
            # Iterate over each left image file
            for left_file in left_files:
                # Expected disparity file corresponding to left image
                disp_file = disp_dir / (left_file.stem + '.npy')
                
                # Skip if disparity file missing, count for stats
                if not disp_file.exists():
                    missing_files += 1
                    continue
                
                # Create sample dictionary holding file references and metadata
                sample = {
                    'left_rgb': left_file,
                    'disparity': disp_file,
                    'scene': scene_dir.name,
                    'basename': left_file.stem
                }
                
                # If stereo is enabled, add right image and disparity files if exist
                if self.use_stereo:
                    right_file = scene_dir / 'Image1' / left_file.name
                    right_disp_file = scene_dir / 'Disp1' / (left_file.stem + '.npy')
                    
                    # Only include sample if both right image and disparity files exist
                    if right_file.exists() and right_disp_file.exists():
                        sample['right_rgb'] = right_file
                        sample['right_disparity'] = right_disp_file
                    else:
                        # Skip this sample if right stereo files missing
                        continue
                
                # Validate sample integrity if requested; skip if invalid
                if self.validate_data and not self._validate_sample_integrity(sample):
                    continue
                
                # Add validated sample to list
                samples.append(sample)
        
        if missing_files > 0:
            # Log a warning if there are any missing files
            logger.warning(f"Missing files: {missing_files}")
        
        if not samples:
            # Raise an error if no valid samples were found
            raise RuntimeError("No valid samples found")
        
        # Log the number of valid samples and scenes loaded
        logger.info(f"Loaded {len(samples)} valid samples from {len(scene_split)} scenes")
        return samples
    
    def _get_scene_split(self, scene_dirs: List[Path]) -> List[Path]:
        # Return training scenes (first 80%)
        if self.split == 'train':
            return scene_dirs[:int(len(scene_dirs) * 0.8)]
        # Return validation scenes (next 10%)
        elif self.split == 'val':
            return scene_dirs[int(len(scene_dirs) * 0.8):int(len(scene_dirs) * 0.9)]
        # Return test scenes (last 10%)
        else:
            return scene_dirs[int(len(scene_dirs) * 0.9):]
    
    def _validate_sample_integrity(self, sample: Dict[str, Any]) -> bool:
        try:
            # List of required files for validation
            required_files = [sample['left_rgb'], sample['disparity']]
            if self.use_stereo:
                # Add right RGB and disparity if stereo is used
                required_files.extend([sample['right_rgb'], sample['right_disparity']])
            
            # Check if all required files exist and are larger than 1KB
            for file_path in required_files:
                if not file_path.exists() or file_path.stat().st_size < 1024:
                    logger.warning(f"Invalid file: {file_path}")
                    return False
            
            # Verify that the left RGB image can be opened properly
            with Image.open(sample['left_rgb']) as img:
                img.verify()
            
            # Load disparity and check for validity (non-empty and not all NaN)
            disparity = np.load(sample['disparity'])
            if disparity.size == 0 or np.isnan(disparity).all():
                logger.warning(f"Invalid disparity: {sample['disparity']}")
                return False
            
            return True
            
        except Exception as e:
            # Log any errors during sample integrity check
            logger.error(f"Sample integrity check failed: {e}")
            return False
    
    def _load_rgb(self, path: Path) -> np.ndarray:
        # Return cached image if caching enabled and available
        if self.cache and path in self._cache_data: #type: ignore 
            return self._cache_data[path] #type: ignore 
        
        try:
            # Open image and convert to RGB
            img = Image.open(path).convert('RGB')
            # Resize if image size does not match expected size
            if img.size != (self.img_size[1], self.img_size[0]):
                img = img.resize((self.img_size[1], self.img_size[0]), Image.LANCZOS) #type: ignore 
            
            # Convert image to numpy array
            img_array = np.array(img)
            
            # Cache image if caching enabled
            if self.cache:
                self._cache_data[path] = img_array #type: ignore 
            
            return img_array
        except Exception as e:
            # Log failure to load RGB image and re-raise
            logger.error(f"Failed to load RGB image {path}: {e}")
            raise
    
    def _load_disparity(self, path: Path) -> np.ndarray:
        # Return cached disparity if caching enabled and available
        if self.cache and path in self._cache_data: #type: ignore 
            return self._cache_data[path] #type: ignore 
        
        try:
            # Load disparity map as float32 numpy array
            disparity = np.load(path).astype(np.float32)
            
            # Resize disparity if shape does not match expected image size
            if disparity.shape != self.img_size:
                from scipy.ndimage import zoom
                scale_h = self.img_size[0] / disparity.shape[0]
                scale_w = self.img_size[1] / disparity.shape[1]
                # Resize disparity using nearest neighbor interpolation (order=0)
                disparity = zoom(disparity, (scale_h, scale_w), order=0)
                # Scale disparity values accordingly
                disparity = disparity * min(scale_h, scale_w)
            
            # Convert disparity map to depth map
            depth = self._disparity_to_depth(disparity)
            
            # Cache depth if caching enabled
            if self.cache:
                self._cache_data[path] = depth #type: ignore 
            
            return depth
        except Exception as e:
            # Log failure to load disparity and re-raise
            logger.error(f"Failed to load disparity {path}: {e}")
            raise
    
    def _disparity_to_depth(self, disparity: np.ndarray, focal_length: float = 1000.0, 
                           baseline: float = 0.54) -> np.ndarray:
        # Convert disparity to depth using camera parameters
        with np.errstate(divide='ignore', invalid='ignore'):
            depth = (focal_length * baseline) / (disparity + 1e-8)  # Avoid division by zero
            # Clip depth values to valid range
            depth = np.clip(depth, self.min_depth, self.max_depth)
            # Set depth to zero where disparity is zero or negative (invalid)
            depth[disparity <= 0] = 0
        return depth
    
    def _create_valid_depth_mask(self, depth: np.ndarray) -> np.ndarray:
        # Create a boolean mask of valid depth pixels based on thresholds and validity checks
        return ((depth > self.min_depth) & 
                (depth < self.max_depth) & 
                (~np.isnan(depth)) & 
                (~np.isinf(depth))).astype(bool)
    
    def _build_rgb_transform(self):
        transforms = []
        
        if self.augmentation:
            # Add data augmentation transforms if enabled
            transforms.extend([
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=5),
                T.RandomPerspective(distortion_scale=0.1, p=0.3)
            ])
        
        # Add tensor conversion and normalization transforms
        transforms.extend([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Compose all transforms into a single callable
        return T.Compose(transforms)
    
    def _build_depth_transform(self):
        # Compose and return depth transformation pipeline (only convert to tensor here)
        return T.Compose([T.ToTensor()])

    def _apply_synchronized_transforms(self, left_rgb: np.ndarray, right_rgb: Optional[np.ndarray], 
                                    depth: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        # If augmentation disabled, return inputs unchanged
        if not self.augmentation:
            return left_rgb, right_rgb, depth
        
        # Random horizontal flip with 50% chance applied to left RGB, right RGB, and depth
        if np.random.rand() > 0.5:
            left_rgb = np.fliplr(left_rgb)
            if right_rgb is not None:
                right_rgb = np.fliplr(right_rgb)
            depth = np.fliplr(depth)
        
        # Return transformed or original arrays
        return left_rgb, right_rgb, depth

    def compute_statistics(self) -> Dict[str, Any]:
        # Initialize sums and counters for RGB and depth statistics
        rgb_sum = np.zeros(3)
        rgb_sq_sum = np.zeros(3)
        depth_values = []
        pixel_count = 0
        
        logger.info("Computing dataset statistics...")
        
        # Limit statistics calculation to first 1000 samples or fewer
        sample_count = min(len(self.samples), 1000)
        
        for i, sample in enumerate(self.samples[:sample_count]):
            # Log progress every 100 samples
            if i % 100 == 0:
                logger.info(f"Processing sample {i}/{sample_count}")
            
            # Load RGB and depth data for current sample
            rgb = self._load_rgb(sample['left_rgb'])
            depth = self._load_disparity(sample['disparity'])
            
            # Normalize RGB to [0,1]
            rgb_normalized = rgb.astype(np.float32) / 255.0
            
            # Accumulate sum and squared sum of RGB pixels for mean and variance calculation
            rgb_sum += rgb_normalized.reshape(-1, 3).sum(axis=0)
            rgb_sq_sum += (rgb_normalized.reshape(-1, 3) ** 2).sum(axis=0)
            
            # Count total pixels processed
            pixel_count += rgb.shape[0] * rgb.shape[1]
            
            # Create mask of valid depth pixels and accumulate valid depth values
            valid_mask = self._create_valid_depth_mask(depth)
            if valid_mask.any():
                depth_values.append(depth[valid_mask])
        
        # Concatenate all valid depth values or create empty array if none
        depth_values = np.concatenate(depth_values) if depth_values else np.array([])
        
        # Calculate mean and standard deviation of RGB channels
        rgb_mean = rgb_sum / pixel_count
        rgb_var = (rgb_sq_sum / pixel_count) - (rgb_mean ** 2)
        rgb_std = np.sqrt(rgb_var)
        
        # Collect statistics in a dictionary
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
        
        # Add depth percentiles if depth values exist
        if depth_values.size > 0:
            for percentile in [25, 50, 75, 95]:
                stats[f'depth_p{percentile}'] = float(np.percentile(depth_values, percentile))
        
        logger.info("Statistics computation completed")
        return stats

    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.samples)

    def __getitem__(self, idx: int):
        # Check index bounds
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds")
        
        # Get sample at index
        sample = self.samples[idx]
        
        # Load left RGB and depth images
        left_rgb = self._load_rgb(sample['left_rgb'])
        depth = self._load_disparity(sample['disparity'])
        
        right_rgb = None
        # Load right RGB if stereo enabled
        if self.use_stereo:
            right_rgb = self._load_rgb(sample['right_rgb'])
        
        # Apply synchronized augmentations if enabled
        if self.augmentation:
            left_rgb, right_rgb, depth = self._apply_synchronized_transforms(left_rgb, right_rgb, depth)
        
        # Apply RGB and depth transforms (e.g., normalization, tensor conversion)
        left_rgb_tensor = self.rgb_transform(left_rgb)
        depth_tensor = self.depth_transform(depth)
        
        # Add channel dimension to depth tensor if missing
        if depth_tensor.dim() == 2: #type: ignore 
            depth_tensor = depth_tensor.unsqueeze(0) #type: ignore
        
        # Create valid depth mask tensor
        valid_mask = self._create_valid_depth_mask(depth)
        
        # Build result dictionary with tensors and metadata
        result = {
            'left_rgb': left_rgb_tensor,
            'depth': depth_tensor,
            'valid_mask': T.ToTensor()(valid_mask.astype(np.float32)), #type: ignore 
            'scene': sample['scene'],
            'basename': sample['basename']
        }
        
        # Include right RGB tensor if stereo data available
        if self.use_stereo and right_rgb is not None:
            right_rgb_tensor = self.rgb_transform(right_rgb)
            result['right_rgb'] = right_rgb_tensor
        
        return result

    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        # Check index bounds
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds")
        
        # Load sample at index
        sample = self.samples[idx]
        left_rgb = self._load_rgb(sample['left_rgb'])
        depth = self._load_disparity(sample['disparity'])
        
        # Return dictionary with sample metadata and statistics
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
