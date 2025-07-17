# FILE: datasets/__init__.py
# ehsanasgharzde - DATASET FACTORY AND REGISTRATION SYSTEM
# hosseinsolymanzadeh - PROPER COMMENTING
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import os
import numpy as np
import torch
from PIL import Image
import cv2

logger = logging.getLogger(__name__)

# Attempt to import various dataset classes; log a warning if any import fails
try:
    from .nyu_dataset import NYUV2Dataset
    from .kitti_dataset import KITTIDataset
    from .enrich_dataset import ENRICHDataset
    from .unreal_dataset import UnrealStereo4KDataset
except ImportError:
    logger.warning("Some dataset imports failed")

# Global registry for storing dataset name-class mappings
_DATASET_REGISTRY = {}

# Register a dataset class with a given name
def register_dataset(name: str, dataset_class: type) -> None:
    if name in _DATASET_REGISTRY:
        raise ValueError(f"Dataset '{name}' already registered")
    _DATASET_REGISTRY[name] = dataset_class
    logger.info(f"Registered dataset: {name}")

# Create and return a dataset instance by name, passing in extra keyword arguments
def create_dataset(dataset_name: str, **kwargs) -> torch.utils.data.Dataset: #type: ignore 
    if dataset_name not in _DATASET_REGISTRY:
        raise ValueError(f"Dataset '{dataset_name}' not registered. Available: {list(_DATASET_REGISTRY.keys())}")
    
    dataset_cls = _DATASET_REGISTRY[dataset_name]

    # If a data root is provided, check if it exists
    if 'data_root' in kwargs:
        data_root = Path(kwargs['data_root'])
        if not data_root.exists():
            raise FileNotFoundError(f"Data root does not exist: {data_root}")
    
    try:
        dataset = dataset_cls(**kwargs)  # Instantiate the dataset
        logger.info(f"Created dataset '{dataset_name}' with {len(dataset)} samples")
        return dataset
    except Exception as e:
        logger.error(f"Failed to create dataset '{dataset_name}': {e}")
        raise

def validate_dataset_structure(data_root: str, dataset_type: str) -> Dict[str, Any]: #type: ignore 
    data_path = Path(data_root)  # Convert the root path string to a Path object
    
    # Initialize the report dictionary to store validation results
    report = {
        'exists': data_path.exists(),
        'missing_files': [],
        'corrupt_files': [],
        'total_files': 0,
        'valid_samples': 0
    }
    
    # If the root path doesn't exist, log the error and return the report
    if not report['exists']:
        logger.error(f"Data root does not exist: {data_root}")
        return report
    
    try:
        # Select the appropriate validation function based on dataset type
        if dataset_type == 'nyu':
            report = _validate_nyu_structure(data_path, report)
        elif dataset_type == 'kitti':
            report = _validate_kitti_structure(data_path, report)
        elif dataset_type == 'enrich':
            report = _validate_enrich_structure(data_path, report)
        elif dataset_type == 'unreal':
            report = _validate_unreal_structure(data_path, report)
        else:
            # Handle unknown dataset type
            logger.warning(f"Unknown dataset type: {dataset_type}")
            report['missing_files'].append(f"Unknown dataset type: {dataset_type}")
        
        # Log the number of valid samples after validation
        logger.info(f"Validation completed for {dataset_type}: {report['valid_samples']}/{report['total_files']} valid samples")
        
    except Exception as e:
        # Catch unexpected exceptions during validation
        logger.error(f"Validation failed for {dataset_type}: {e}")
        report['corrupt_files'].append(f"Validation error: {e}")
    
    return report

def _validate_nyu_structure(data_path: Path, report: Dict[str, Any]) -> Dict[str, Any]:
    # Loop through all splits: train, val, test
    for split in ['train', 'val', 'test']:
        rgb_dir = data_path / split / 'rgb'  # RGB images directory
        depth_dir = data_path / split / 'depth'  # Depth maps directory
        
        # Check for existence of RGB directory
        if not rgb_dir.exists():
            report['missing_files'].append(str(rgb_dir))
            continue
        
        # Check for existence of Depth directory
        if not depth_dir.exists():
            report['missing_files'].append(str(depth_dir))
            continue
        
        # Gather list of RGB and Depth files
        rgb_files = list(rgb_dir.glob('*.jpg')) + list(rgb_dir.glob('*.png'))
        depth_files = list(depth_dir.glob('*.png'))
        
        report['total_files'] += len(rgb_files)
        
        # Match each RGB file to its corresponding Depth file
        for rgb_file in rgb_files:
            depth_file = depth_dir / (rgb_file.stem + '.png')
            if depth_file.exists():
                if _validate_file_integrity(rgb_file, depth_file):
                    report['valid_samples'] += 1
                else:
                    report['corrupt_files'].append(str(rgb_file))
            else:
                report['missing_files'].append(str(depth_file))
    
    return report

def _validate_kitti_structure(data_path: Path, report: Dict[str, Any]) -> Dict[str, Any]:
    sequences_dir = data_path / 'sequences'  # Directory for sequence data
    calib_dir = data_path / 'calib'  # Calibration data
    
    # Check existence of required directories
    if not sequences_dir.exists():
        report['missing_files'].append(str(sequences_dir))
        return report
    
    if not calib_dir.exists():
        report['missing_files'].append(str(calib_dir))
    
    # Iterate over each sequence subfolder
    for seq_dir in sequences_dir.iterdir():
        if not seq_dir.is_dir():
            continue
        
        img_dir = seq_dir / 'image_02'  # Left image frames
        depth_dir = seq_dir / 'proj_depth' / 'velodyne_raw'  # Corresponding depth maps
        
        if not img_dir.exists():
            report['missing_files'].append(str(img_dir))
            continue
        
        if not depth_dir.exists():
            report['missing_files'].append(str(depth_dir))
            continue
        
        img_files = list(img_dir.glob('*.png'))
        report['total_files'] += len(img_files)
        
        # Validate each image file against corresponding depth file
        for img_file in img_files:
            depth_file = depth_dir / (img_file.stem + '.png')
            if depth_file.exists():
                if _validate_file_integrity(img_file, depth_file):
                    report['valid_samples'] += 1
                else:
                    report['corrupt_files'].append(str(img_file))
            else:
                report['missing_files'].append(str(depth_file))
    
    return report

def _validate_enrich_structure(data_path: Path, report: Dict[str, Any]) -> Dict[str, Any]:
    # List of enrich sub-dataset names
    enrich_datasets = ['ENRICH-Aerial', 'ENRICH-Square', 'ENRICH-Statue']
    
    # Iterate over each ENRICH sub-dataset
    for dataset_name in enrich_datasets:
        dataset_dir = data_path / dataset_name
        
        if not dataset_dir.exists():
            report['missing_files'].append(str(dataset_dir))
            continue
        
        images_dir = dataset_dir / 'images'  # RGB images
        depth_dir = dataset_dir / 'depth' / 'exr'  # EXR depth maps
        
        if not images_dir.exists():
            report['missing_files'].append(str(images_dir))
            continue
        
        if not depth_dir.exists():
            report['missing_files'].append(str(depth_dir))
            continue
        
        img_files = list(images_dir.glob('*.jpg'))
        report['total_files'] += len(img_files)
        
        # Check matching EXR depth file for each RGB image
        for img_file in img_files:
            depth_file = depth_dir / f'{img_file.stem}_depth.exr'
            if depth_file.exists():
                if _validate_file_integrity(img_file, depth_file):
                    report['valid_samples'] += 1
                else:
                    report['corrupt_files'].append(str(img_file))
            else:
                report['missing_files'].append(str(depth_file))
    
    return report

def _validate_unreal_structure(data_path: Path, report: Dict[str, Any]) -> Dict[str, Any]:
    # Get all scene directories with numeric names
    scene_dirs = [d for d in data_path.iterdir() if d.is_dir() and d.name.isdigit()]
    
    for scene_dir in scene_dirs:
        left_dir = scene_dir / 'Image0'  # Left camera images
        disp_dir = scene_dir / 'Disp0'  # Disparity maps
        
        if not left_dir.exists():
            report['missing_files'].append(str(left_dir))
            continue
        
        if not disp_dir.exists():
            report['missing_files'].append(str(disp_dir))
            continue
        
        img_files = list(left_dir.glob('*.jpg'))
        report['total_files'] += len(img_files)
        
        # Validate each image against corresponding disparity .npy file
        for img_file in img_files:
            disp_file = disp_dir / (img_file.stem + '.npy')
            if disp_file.exists():
                if _validate_file_integrity(img_file, disp_file):
                    report['valid_samples'] += 1
                else:
                    report['corrupt_files'].append(str(img_file))
            else:
                report['missing_files'].append(str(disp_file))
    
    return report

def _validate_file_integrity(file1: Path, file2: Path) -> bool:
    try:
        # Check if either file is smaller than 1KB; if so, consider it invalid
        if file1.stat().st_size < 1024 or file2.stat().st_size < 1024:
            return False
        
        # If file1 is an image (JPG or PNG), try to open and verify it
        if file1.suffix.lower() in ['.jpg', '.png']:
            with Image.open(file1) as img:
                img.verify()  # Verify that file1 is not corrupted
        
        # If file2 is an image (JPG or PNG), try to open and verify it
        if file2.suffix.lower() in ['.jpg', '.png']:
            with Image.open(file2) as img:
                img.verify()  # Verify that file2 is not corrupted
        
        # If all checks pass, return True (files are valid)
        return True

    # Catch any exception (e.g., file not found, corrupted image, etc.) and return False
    except Exception:
        return False

def compute_dataset_statistics(dataset: torch.utils.data.Dataset) -> Dict[str, Any]:  # type: ignore
    # Initialize accumulators for RGB mean and squared mean
    rgb_sum = np.zeros(3)
    rgb_sq_sum = np.zeros(3)
    
    # List to collect valid depth values
    depth_values = []
    
    # Counter for how many samples are processed
    pixel_count = 0
    
    # Limit number of samples to 1000 for efficiency
    sample_count = min(len(dataset), 1000)
    logger.info(f"Computing statistics for {sample_count} samples...")
    
    for i in range(sample_count):
        if i % 100 == 0:
            logger.info(f"Processing sample {i}/{sample_count}")
        
        try:
            # Access sample and extract rgb and depth
            sample = dataset[i]
            rgb = sample['rgb']
            depth = sample['depth']
            
            # Convert RGB to NumPy array if it’s a tensor
            if isinstance(rgb, torch.Tensor):
                rgb_np = rgb.permute(1, 2, 0).numpy()
            else:
                rgb_np = np.array(rgb)
            
            # Convert depth to NumPy array if it’s a tensor
            if isinstance(depth, torch.Tensor):
                depth_np = depth.squeeze().numpy()
            else:
                depth_np = np.array(depth)
            
            # Normalize RGB if values are in 0–255 range
            if rgb_np.max() > 1.0:
                rgb_np = rgb_np / 255.0
            
            # Accumulate RGB mean and squared mean across pixels
            rgb_sum += rgb_np.reshape(-1, 3).mean(axis=0)
            rgb_sq_sum += (rgb_np.reshape(-1, 3) ** 2).mean(axis=0)
            pixel_count += 1
            
            # Collect valid (non-zero) depth values
            valid_depth = depth_np[depth_np > 0]
            if valid_depth.size > 0:
                depth_values.append(valid_depth)
        
        except Exception as e:
            # Log and skip faulty samples
            logger.warning(f"Failed to process sample {i}: {e}")
            continue
    
    # Compute final RGB mean and standard deviation
    if pixel_count > 0:
        rgb_mean = rgb_sum / pixel_count
        rgb_var = (rgb_sq_sum / pixel_count) - (rgb_mean ** 2)
        rgb_std = np.sqrt(np.maximum(rgb_var, 0))
    else:
        rgb_mean = np.zeros(3)
        rgb_std = np.zeros(3)
    
    # Compute depth statistics if any valid depths were found
    if depth_values:
        depth_all = np.concatenate(depth_values)
        depth_stats = {
            'depth_min': float(depth_all.min()),
            'depth_max': float(depth_all.max()),
            'depth_mean': float(depth_all.mean()),
            'depth_std': float(depth_all.std()),
            'depth_median': float(np.median(depth_all)),
        }
        
        # Add depth percentiles (25th, 50th, 75th, 95th)
        for percentile in [25, 50, 75, 95]:
            depth_stats[f'depth_p{percentile}'] = float(np.percentile(depth_all, percentile))
    else:
        # If no valid depth, fill with zeros
        depth_stats = {
            'depth_min': 0.0,
            'depth_max': 0.0,
            'depth_mean': 0.0,
            'depth_std': 0.0,
            'depth_median': 0.0,
        }
    
    # Combine RGB and depth statistics into a single dictionary
    stats = {
        'rgb_mean': rgb_mean.tolist(),
        'rgb_std': rgb_std.tolist(),
        'num_samples': len(dataset),
        'processed_samples': sample_count,
        'num_valid_depth_points': int(sum(len(dv) for dv in depth_values)),
        **depth_stats
    }
    
    logger.info("Statistics computation completed")
    return stats

# Returns a list of all registered dataset names
def get_available_datasets() -> List[str]:
    return list(_DATASET_REGISTRY.keys())

# Retrieves metadata information about a specific dataset by its name
def dataset_info(dataset_name: str) -> Dict[str, Any]:
    # Raise an error if the dataset is not found in the registry
    if dataset_name not in _DATASET_REGISTRY:
        raise ValueError(f"Dataset '{dataset_name}' not registered")
    
    # Retrieve the dataset class from the registry
    dataset_cls = _DATASET_REGISTRY[dataset_name]
    
    # Prepare a dictionary with basic dataset information
    info = {
        'name': dataset_name,                  # Name of the dataset
        'class': dataset_cls.__name__,        # Name of the dataset class
        'module': dataset_cls.__module__,     # Module where the class is defined
    }
    
    # If a docstring is available for the dataset class, include it as a description
    if hasattr(dataset_cls, '__doc__') and dataset_cls.__doc__:
        info['description'] = dataset_cls.__doc__.strip()
    
    return info

# Attempt to register several predefined datasets
try:
    register_dataset('nyu', NYUV2Dataset)                      # Register NYU dataset
    register_dataset('kitti', KITTIDataset)                    # Register KITTI dataset
    register_dataset('enrich', ENRICHDataset)                  # Register ENRICH dataset
    register_dataset('unreal', UnrealStereo4KDataset)          # Register Unreal Stereo 4K dataset
except NameError as e:
    # Log a warning if any dataset class is not defined (NameError)
    logger.warning(f"Failed to register some datasets: {e}")
