# FILE: datasets/kitti_dataset.py
# ehsanasgharzde - KITTI DATASET
# hosseinsolymanzadeh - PROPER COMMENTING
# ehsanasgharzde, hosseinsolymanzadeh - FIXED REDUNDANT CODE BY EXTRACTING PURE FUNCTIONS AND BASECLASS LEVEL METHODS

import os
import glob
import numpy as np
import torchvision.transforms as T
import logging

from utils.dataset import BaseDataset 

logger = logging.getLogger(__name__)

class KITTIDataset(BaseDataset):
    def __init__(self, data_root, split='train', img_size=(352, 1216),
                depth_scale=256.0, cache=False, use_dense_depth=False, validate_data=True):
        super().__init__(data_root, split, img_size, depth_scale, cache, validate_data)
        
        self.use_dense_depth = use_dense_depth  # Whether to use dense depth maps
        
        self.validate_initialization_parameters()  # Check if the provided parameters are valid
        
        if validate_data:
            self.validate_dataset_structure()  # Check if the dataset directory structure is valid
        
        self.samples = self.load_samples()  # Load the list of samples (e.g., file paths)
        
        self.cache_data = {} if self.cache else None  # Initialize in-memory cache if enabled
        
        logger.info(f"KITTI dataset initialized: {len(self.samples)} samples, split={split}")  # Log dataset loading info

    def validate_dataset_structure(self):
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

    def load_samples(self):
        samples = []  # List to store valid samples
        sequences_path = os.path.join(self.data_root, 'sequences')  # Base path for sequences
        
        # Check again in case sequences directory is missing
        if not os.path.isdir(sequences_path):
            logger.error(f"Sequences directory not found: {sequences_path}")
            return samples  # Return empty list if missing
        
        split_sequences = self.get_sequences()  # Get list of sequences for the current split
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

    def get_sequences(self):
        # Path to the 'sequences' directory
        sequences_path = os.path.join(self.data_root, 'sequences')
        
        # Get all subdirectories in the 'sequences' directory (i.e., sequence names)
        all_sequences = sorted([d for d in os.listdir(sequences_path) 
                               if os.path.isdir(os.path.join(sequences_path, d))])
        return all_sequences

    def compute_statistics(self):
        rgb_sums = np.zeros(3)        # Sum of RGB channel means
        rgb_sq_sums = np.zeros(3)     # Sum of squared RGB channel means
        pixel_count = 0               # Counter for number of processed images
        depth_values = []            # List to collect valid depth values
        
        sample_count = min(len(self.samples), 1000)  # Limit to first 1000 samples for speed
        
        for i in range(sample_count):
            sample = self.samples[i]
            rgb = self.load_rgb(sample['rgb'])      # Load RGB image
            depth = self.load_depth(sample['depth'])# Load depth map
            
            if rgb is None or depth is None:
                continue  # Skip if image or depth couldn't be loaded
            
            rgb_np = np.array(rgb) / 255.0           # Normalize RGB to [0, 1]
            rgb_sums += np.mean(rgb_np, axis=(0,1))  # Add per-channel mean
            rgb_sq_sums += np.mean(rgb_np ** 2, axis=(0,1))  # Add per-channel squared mean
            pixel_count += 1
            
            mask = self.create_default_mask(depth)  # Create valid depth mask
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

    def __getitem__(self, idx):
        # Check for out-of-bounds index
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self)}")
        
        sample = self.samples[idx]  # Get sample dictionary by index
        
        rgb = self.load_rgb(sample['rgb'])      # Load RGB image
        depth = self.load_depth(sample['depth'])# Load depth map
        
        # Raise error if loading failed
        if rgb is None or depth is None:
            logger.error(f"Failed to load sample at index {idx}")
            raise RuntimeError(f"Failed to load sample at index {idx}")
        
        rgb_tensor = self.rgb_transform(rgb)       # Apply RGB transform
        depth_tensor = self.depth_transform(depth) # Apply depth transform
        
        valid_mask = self.create_default_mask(depth)  # Generate valid mask
        mask_tensor = T.ToTensor()(valid_mask.astype(np.float32)) if valid_mask is not None else None
        
        # Return dictionary containing input tensors and metadata
        return {
            'rgb': rgb_tensor,
            'depth': depth_tensor,
            'mask': mask_tensor,
            'sequence': sample['sequence'],
            'basename': sample['basename']
        }