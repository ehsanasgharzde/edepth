# FILE: datasets/enrich_dataset.py
# ehsanasgharzde - ENRICH DATASET
# hosseinsolymanzadeh - PROPER COMMENTING
# ehsanasgharzde, hosseinsolymanzadeh - FIXED REDUNDANT CODE BY EXTRACTING PURE FUNCTINOS AND BASECLASS LEVEL METHODS

import numpy as np
import torchvision.transforms as T
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import cv2
import OpenEXR 
import Imath

from utils.dataset import BaseDataset

logger = logging.getLogger(__name__)

class ENRICHDataset(BaseDataset):
    def __init__(self, data_root: str, split: str = 'train', img_size: Tuple[int, int] = (480, 640), 
                depth_scale: float = 1.0, cache: bool = False, validate_data: bool = True):
        super().__init__(data_root, split, img_size, depth_scale, cache, validate_data)
        
        self.dataset_type = 'all'
        
        # Log dataset initialization
        logger.info(f"Initialized ENRICH dataset: {len(self.samples)} samples for {split} split")
    
    def validate_dataset_structure(self):
        # List of expected dataset subfolders
        required_datasets = ['ENRICH-Aerial', 'ENRICH-Square', 'ENRICH-Statue']
        
        # Determine which datasets to validate
        if self.dataset_type == 'all':
            check_datasets = required_datasets
        else:
            check_datasets = [f'ENRICH-{self.dataset_type.title()}']
        
        for dataset_name in check_datasets:
            dataset_path = self.data_root / dataset_name
            if not dataset_path.exists():
                logger.warning(f"Dataset directory not found: {dataset_path}")
                continue
            
            # Check that required subdirectories exist
            required_dirs = ['images', 'depth/exr']
            for req_dir in required_dirs:
                if not (dataset_path / req_dir).exists():
                    logger.warning(f"Required directory missing: {dataset_path / req_dir}")
    
    def load_samples(self) -> List[Dict[str, Any]]:
        # Initialize sample list
        samples = []

        dataset_dirs = []

        # Determine which dataset directories to load based on dataset_type
        if self.dataset_type == 'all':
            dataset_dirs = [d for d in self.data_root.iterdir() if d.is_dir() and d.name.startswith('ENRICH-')]
        else:
            dataset_path = self.data_root / f'ENRICH-{self.dataset_type.title()}'
            if dataset_path.exists():
                dataset_dirs = [dataset_path]

        # Raise error if no dataset directories found
        if not dataset_dirs:
            raise FileNotFoundError(f"No ENRICH dataset directories found in {self.data_root}")

        missing_files = 0  # Counter for missing depth files

        # Iterate through dataset directories
        for dataset_dir in dataset_dirs:
            dataset_name = dataset_dir.name

            images_dir = dataset_dir / 'images'
            depth_dir = dataset_dir / 'depth' / 'exr'

            # Check if required subdirectories exist
            if not images_dir.exists() or not depth_dir.exists():
                logger.warning(f"Missing directories in {dataset_name}")
                continue
            
            # Load and sort all image files (JPG format)
            image_files = sorted(images_dir.glob('*.jpg'))

            # Iterate through image files and check for corresponding depth files
            for image_file in image_files:
                base_name = image_file.stem
                depth_file = depth_dir / f'{base_name}_depth.exr'

                # Skip sample if depth file is missing
                if not depth_file.exists():
                    missing_files += 1
                    continue
                
                # Construct sample dictionary
                sample = {
                    'rgb': image_file,
                    'depth': depth_file,
                    'dataset': dataset_name,
                    'basename': base_name
                }

                # Optionally validate sample integrity
                if self.validate_data and not self.validate_sample_integrity(sample):
                    continue
                
                # Add valid sample to list
                samples.append(sample)

        # Log number of missing depth files, if any
        if missing_files > 0:
            logger.warning(f"Missing depth files: {missing_files}")

        # Raise error if no valid samples were found
        if not samples:
            raise RuntimeError("No valid samples found")

        # Log how many samples were loaded from how many datasets
        logger.info(f"Loaded {len(samples)} valid samples from {len(dataset_dirs)} datasets")
        return samples

    def load_depth(self, path: Path) -> np.ndarray:
        # Return cached depth if available
        if self.cache and path in self.cache_data:  # type: ignore
            return self.cache_data[path]  # type: ignore
        
        try: 
            exr_file = OpenEXR.InputFile(str(path))
            header = exr_file.header()
            
            # Get image dimensions from EXR header
            dw = header['dataWindow']
            size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
            
            # Read the R channel (assuming depth stored in R)
            FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
            depth_str = exr_file.channel('R', FLOAT)
            depth = np.frombuffer(depth_str, dtype=np.float32)
            depth = depth.reshape(size[1], size[0])
            
            # Resize if depth shape mismatches expected size
            if depth.shape != self.img_size:
                depth = cv2.resize(depth, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)
            
            # Replace invalid values (zero, NaN, inf) with 0.0
            invalid_mask = (depth <= 0) | np.isnan(depth) | np.isinf(depth)
            depth[invalid_mask] = 0.0
            
            # Cache if enabled
            if self.cache:
                self.cache_data[path] = depth  # type: ignore
            
            return depth
        except ImportError:
            logger.warning("OpenEXR not available, trying PIL fallback")
            raise
        except Exception as e:
            logger.error(f"Failed to load depth {path}: {e}")
            raise

    def compute_statistics(self) -> Dict[str, Any]:
        # Initialize RGB stats accumulators and depth values list
        rgb_sum = np.zeros(3)
        rgb_sq_sum = np.zeros(3)
        depth_values = []
        pixel_count = 0

        logger.info("Computing dataset statistics...")

        # Limit computation to a sample of the dataset (max 1000 samples)
        sample_count = min(len(self.samples), 1000)

        for i, sample in enumerate(self.samples[:sample_count]):
            if i % 100 == 0:
                logger.info(f"Processing sample {i}/{sample_count}")

            # Load RGB and depth images
            rgb = self.load_rgb(sample['rgb'])
            depth = self.load_depth(sample['depth'])

            # Normalize RGB to [0, 1] and compute running sums
            rgb_normalized = rgb.astype(np.float32) / 255.0
            rgb_sum += rgb_normalized.reshape(-1, 3).sum(axis=0)
            rgb_sq_sum += (rgb_normalized.reshape(-1, 3) ** 2).sum(axis=0)
            pixel_count += rgb.shape[0] * rgb.shape[1]

            # Extract valid depth values using mask
            valid_mask = self.create_default_mask(depth)
            if valid_mask.any():
                depth_values.append(depth[valid_mask])

        # Concatenate all valid depth values
        depth_values = np.concatenate(depth_values) if depth_values else np.array([])

        # Compute mean and std for RGB
        rgb_mean = rgb_sum / pixel_count
        rgb_var = (rgb_sq_sum / pixel_count) - (rgb_mean ** 2)
        rgb_std = np.sqrt(rgb_var)

        # Assemble statistics dictionary
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

        # Compute depth percentiles if values exist
        if depth_values.size > 0:
            for percentile in [25, 50, 75, 95]:
                stats[f'depth_p{percentile}'] = float(np.percentile(depth_values, percentile))

        logger.info("Statistics computation completed")
        return stats

    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        # Check for valid index range
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds")

        # Load sample and corresponding data
        sample = self.samples[idx]
        rgb = self.load_rgb(sample['rgb'])
        depth = self.load_depth(sample['depth'])

        # Return metadata and stats about the sample
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

    def __getitem__(self, idx: int):
        # Validate index
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds")

        # Load RGB and depth data for the sample
        sample = self.samples[idx]
        rgb = self.load_rgb(sample['rgb'])
        depth = self.load_depth(sample['depth'])

        # Apply RGB and depth transforms
        rgb_tensor = self.rgb_transform(rgb)
        depth_tensor = self.depth_transform(depth)

        # Ensure depth has channel dimension
        if depth_tensor.dim() == 2:  # type: ignore
            depth_tensor = depth_tensor.unsqueeze(0)  # type: ignore

        # Generate binary valid depth mask
        valid_mask = self.create_valid_depth_mask(depth_tensor.squeeze(0).numpy()) #type: ignore

        # Return sample as dictionary
        return {
            'rgb': rgb_tensor,
            'depth': depth_tensor,
            'valid_mask': T.ToTensor()(valid_mask.astype(np.float32)),  # type: ignore
        }