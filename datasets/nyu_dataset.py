# FILE: datasets/nyu_dataset.py
# ehsanasgharzde - NYU DEPTH V2 DATASET
# hosseinsolymanzadeh - PROPER COMMENTING
# ehsanasgharzde, hosseinsolymanzadeh - FIXED REDUNDANT CODE BY EXTRACTING PURE FUNCTIONS AND BASECLASS LEVEL METHODS

import logging
import numpy as np
from pathlib import Path
import torchvision.transforms as T
from typing import Dict, Tuple, List, Any

from utils.dataset_utils import BaseDataset

logger = logging.getLogger(__name__)


class NYUV2Dataset(BaseDataset):
    # Initialize the dataset object with various parameters for data loading and processing
    def __init__(self, data_root: str, split: str = 'train', img_size: Tuple[int, int] = (480, 640), 
                depth_scale: float = 1000.0, cache: bool = False, validate_data: bool = True):
        super().__init__(data_root, split, img_size, depth_scale, cache, validate_data)

        # Directory containing RGB images for the given split
        self.rgb_dir = self.data_root / split / 'rgb'
        # Directory containing depth images for the given split
        self.depth_dir = self.data_root / split / 'depth'
        
        # Log info about dataset initialization
        logger.info(f"Initialized NYU Depth V2 dataset: {len(self.samples)} samples for {split} split")
    
    # Check that the dataset directories for RGB and depth images exist and contain files
    def validate_dataset_structure(self):
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
    def load_samples(self) -> List[Dict[str, Path]]:
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

            sample = {
                'rgb': rgb_file,
                'depth': depth_file,
                'basename': rgb_file.stem
            }
            
            # Validate integrity of sample files if enabled
            if self.validate_data and not self.validate_sample_integrity(sample):
                continue
                
            # Append valid sample as dictionary with paths and basename
            samples.append(sample)
        
        # Log warning if any depth files were missing
        if missing_depth > 0:
            logger.warning(f"Missing depth files: {missing_depth}")
        
        # Raise error if no valid samples found
        if not samples:
            raise RuntimeError("No valid RGB-depth pairs found")
        
        # Log how many valid samples were loaded
        logger.info(f"Loaded {len(samples)} valid samples")
        return samples
    
    # Retrieve basic info about a sample without applying transforms
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        # Check for valid index range
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds")
        
        # Load sample metadata and raw images
        sample = self.samples[idx]
        rgb = self.load_rgb(sample['rgb'])
        depth = self.load_depth(sample['depth'])
        
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
            
            rgb = self.load_rgb(sample['rgb'])
            depth = self.load_depth(sample['depth'])
            
            # Normalize RGB pixels to [0,1]
            rgb_normalized = rgb.astype(np.float32) / 255.0
            # Sum RGB values and squared RGB values for variance calculation
            rgb_sum += rgb_normalized.reshape(-1, 3).sum(axis=0)
            rgb_sq_sum += (rgb_normalized.reshape(-1, 3) ** 2).sum(axis=0)
            pixel_count += rgb.shape[0] * rgb.shape[1]
            
            # Collect valid depth pixels for statistics
            valid_mask = self.create_default_mask(depth)
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

    def __getitem__(self, idx):
        # Check for valid index range
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds")
        
        # Retrieve sample metadata (paths and basename)
        sample = self.samples[idx]
        
        # Load RGB and depth images from disk (or cache)
        rgb = self.load_rgb(sample['rgb'])
        depth = self.load_depth(sample['depth'])
        
        # Apply RGB and depth transformations (resize, normalize, to tensor, etc.)
        rgb = self.rgb_transform(image=rgb)['image']
        depth = self.depth_transform(image=depth)['image']
        
        # Ensure depth tensor has channel dimension (C=1)
        if depth.dim() == 2:
            depth = depth.unsqueeze(0)
        
        # Create a valid mask for depth pixels (boolean mask)
        valid_mask = self.create_default_mask(depth.squeeze(0).numpy())
        
        # Return dictionary containing processed data and metadata
        return {
            'rgb': rgb,
            'depth': depth,
            'valid_mask': T.ToTensor()(valid_mask.astype(np.float32))
        }
    