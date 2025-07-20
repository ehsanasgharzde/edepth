# FILE: utils/dataset.py
# ehsanasgharzde, hosseinsolymanzadeh - FIXED REDUNDANT CODE BY EXTRACTING PURE FUNCTINOS AND BASECLASS LEVEL METHODS

from abc import abstractmethod
from typing import Tuple, Dict, Any, List
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import logging
from albumentations.pytorch import ToTensorV2
import albumentations as A
import cv2
from abc import abstractmethod

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(self, data_root: str, split: str = 'train', img_size: Tuple[int, int] = (480, 640),
                 depth_scale: float = 1.0, cache: bool = False, validate_data: bool = True):
        # Initialize dataset configuration
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = img_size
        self.depth_scale = depth_scale
        self.cache = cache
        self.validate_data = validate_data
        self.rgb_transform = self.get_rgb_transform()
        self.depth_transform = self.get_depth_transform()
        self.samples = []

        # Validate the initial parameters for correctness
        self.validate_initialization_parameters()

        # If enabled, check the dataset directory structure and files
        if validate_data:
            self.validate_dataset_structure()
            
        # Load list of all data samples (pairs of RGB and depth images)
        self.samples = self.load_samples()
        
        # Initialize cache dictionary if caching is enabled
        self.cache_data = {} if cache else None
    
    def validate_initialization_parameters(self):
        # Check if the data root directory exists
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root not found: {self.data_root}")
        
        # Validate the split argument
        if self.split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split: {self.split}")
        
        # Ensure image size is a tuple of two integers
        if not isinstance(self.img_size, tuple) or len(self.img_size) != 2:
            raise ValueError("img_size must be tuple of (height, width)")
        
        # Depth scale must be a positive number
        if self.depth_scale <= 0:
            raise ValueError("depth_scale must be positive")

    # Check that the dataset directories for RGB and depth images exist and contain files
    @abstractmethod
    def validate_dataset_structure(self):
        pass

    def validate_sample_integrity(self, sample: Dict[str, Any]) -> bool:
        try:
            # Ensure RGB and depth files exist and are not empty
            required_files = [sample['rgb'], sample['depth']]
            for file_path in required_files:
                if not file_path.exists() or file_path.stat().st_size < 1024:
                    logger.warning(f"Invalid file: {file_path}")
                    return False

            # Try loading the RGB image with OpenCV
            img = cv2.imread(str(sample['rgb']))
            if img is None or img.size == 0:
                logger.warning(f"Corrupt or unreadable RGB image: {sample['rgb']}")
                return False

            return True

        except Exception as e:
            logger.error(f"Sample integrity check failed: {e}")
            return False

    # Create a boolean mask indicating valid depth pixels
    def create_default_mask(self, target: np.ndarray) -> np.ndarray:
        mask = ((target > 0) & 
                (~np.isnan(target)) & 
                (~np.isinf(target))).astype(bool)

        if mask.sum() == 0:
            logger.warning("No valid pixels found after applying depth range mask")
            return np.zeros_like(target, dtype=bool)
        
        return mask

    # Define the transformation pipeline applied to RGB images
    def get_rgb_transform(self, MEAN: tuple = (0.485, 0.456, 0.406), STD: tuple = (0.229, 0.224, 0.225)):
        # Start with resizing to target image size
        transforms = [A.Resize(self.img_size[0], self.img_size[1], interpolation=1)]
        
        # Normalize using dataset mean and std, then convert to tensor
        transforms.extend([
            A.Normalize(mean=MEAN, std=STD), #type: ignore 
            ToTensorV2()
        ])
        
        # Compose all transformations into one pipeline
        return A.Compose(transforms) #type: ignore 
    
    # Define the transformation pipeline applied to depth images
    def get_depth_transform(self):
        # Resize depth image to target size using nearest neighbor interpolation (interpolation=0)
        transforms = [A.Resize(self.img_size[0], self.img_size[1], interpolation=0)]
        
        # Convert to tensor after transformations
        transforms.append(ToTensorV2()) #type: ignore 
        return A.Compose(transforms) #type: ignore 

    # Load paired RGB and depth samples into a list of dictionaries
    @abstractmethod
    def load_samples(self) -> List[Dict[str, Path]]:
        pass

    def load_rgb(self, path: Path) -> np.ndarray:
        # Return cached RGB image if available
        if self.cache and path in self.cache_data:  # type: ignore
            return self.cache_data[path]  # type: ignore
        
        try:
            # Load image using OpenCV (BGR format)
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                raise IOError(f"Unable to read RGB image: {path}")
            
            # Convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize image if needed
            if img.shape[:2] != self.img_size:
                img = cv2.resize(img, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_LINEAR)
            
            # Cache image if caching enabled
            if self.cache:
                self.cache_data[path] = img  # type: ignore
            
            return img
        except Exception as e:
            logger.error(f"Failed to load RGB image {path}: {e}")
            raise

    # Load a depth image from disk, scale and clip it, optionally caching it
    def load_depth(self, path: Path) -> np.ndarray:
        # Return cached depth if caching enabled and depth exists in cache
        if self.cache and path in self.cache_data: #type: ignore 
            return self.cache_data[path] #type: ignore 
        
        # Read depth image preserving original values (16-bit or 32-bit)
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise IOError(f"Unable to read depth image: {path}")
        
        # Convert to float32 and scale by depth_scale factor
        depth = depth.astype(np.float32) / self.depth_scale
        
        # Mask invalid values (<=0, NaN, Inf) and set them to 0
        invalid_mask = (depth <= 0) | np.isnan(depth) | np.isinf(depth)
        depth[invalid_mask] = 0.0
        
        # Cache the depth if caching is enabled
        if self.cache:
            self.cache_data[path] = depth #type: ignore 
        
        return depth

    # Return the total number of samples in the dataset
    def __len__(self):
        return len(self.samples)