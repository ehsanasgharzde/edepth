# FILE: datasets/__init__.py
# ehsanasgharzde - DATASET FACTORY AND REGISTRATION SYSTEM
# hosseinsolymanzadeh - PROPER COMMENTING
# ehsanasgharzde, hosseinsolymanzadeh - FIXED REDUNDANT CODE BY EXTRACTING PURE FUNCTIONS AND BASECLASS LEVEL METHODS

import logging
from pathlib import Path
from typing import Dict, Any, List
import torch

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
DATASET_REGISTRY = {}

# Register a dataset class with a given name
def register_dataset(name: str, dataset_class: type) -> None:
    if name in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{name}' already registered")
    DATASET_REGISTRY[name] = dataset_class
    logger.info(f"Registered dataset: {name}")

# Create and return a dataset instance by name, passing in extra keyword arguments
def create_dataset(dataset_name: str, **kwargs) -> torch.utils.data.Dataset: #type: ignore 
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{dataset_name}' not registered. Available: {list(DATASET_REGISTRY.keys())}")
    
    dataset_cls = DATASET_REGISTRY[dataset_name]

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
# Returns a list of all registered dataset names
def get_available_datasets() -> List[str]:
    return list(DATASET_REGISTRY.keys())

# Retrieves metadata information about a specific dataset by its name
def dataset_info(dataset_name: str) -> Dict[str, Any]:
    # Raise an error if the dataset is not found in the registry
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{dataset_name}' not registered")
    
    # Retrieve the dataset class from the registry
    dataset_cls = DATASET_REGISTRY[dataset_name]
    
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