# FILE: tests/test_datasets.py
# ehsanasgharzde - COMPLETE DATASET TEST SUITE
# hosseinsolymanzadeh - PROPER COMMENTING
# hosseinsolymanzadeh - FIXED REDUNDANT CODE BY EXTRACTING PURE FUNCTIONS AND BASECLASS LEVEL METHODS

import os
import cv2
import torch
import pytest
import logging
import tempfile
import numpy as np
from PIL import Image
from typing import List, Tuple

from datasets import create_dataset, register_dataset, get_available_datasets, dataset_info
from datasets.nyu_dataset import NYUV2Dataset
from datasets.kitti_dataset import KITTIDataset
from datasets.enrich_dataset import ENRICHDataset
from utils.dataset_utils import BaseDataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Dataset Factory Registration Tests
def test_register_dataset_new() -> None:
    # Register a new dataset with a unique name
    register_dataset("test_dataset_new", NYUV2Dataset)
    
    # Assert that the dataset is now in the registry
    assert "test_dataset_new" in get_available_datasets()
    
    # Log successful registration
    logger.info("Successfully registered new dataset")

def test_register_dataset_duplicate() -> None:
    # Register the dataset with a given name
    register_dataset("test_dup", NYUV2Dataset)
    
    # Attempt to register again with the same name should raise an error
    with pytest.raises(ValueError, match="already registered"):
        register_dataset("test_dup", NYUV2Dataset)
    
    # Log that the duplicate registration was properly handled
    logger.info("Duplicate registration properly rejected")


def test_create_dataset_valid() -> None:
    # Create a temporary directory for the test dataset
    with tempfile.TemporaryDirectory() as temp_dir:
        # Register a new dataset
        register_dataset("test_valid_create", NYUV2Dataset)
        
        # Create train split directories
        train_dir = os.path.join(temp_dir, "train")
        os.makedirs(os.path.join(train_dir, "rgb"))
        os.makedirs(os.path.join(train_dir, "depth"))
        
        # Create and save a dummy RGB image
        dummy_img = Image.new('RGB', (640, 480), color='red')
        dummy_img.save(os.path.join(train_dir, "rgb", "test.png"))
        
        # Create and save a dummy depth image
        dummy_depth = np.ones((480, 640), dtype=np.uint16) * 1000
        cv2.imwrite(os.path.join(train_dir, "depth", "test.png"), dummy_depth)
        
        # Attempt to create a dataset instance from the prepared data
        dataset = create_dataset("test_valid_create", data_root=temp_dir, split="train")
        
        # Assert the dataset is an instance of the correct class
        assert isinstance(dataset, NYUV2Dataset)
        assert len(dataset) > 0
        
        # Log successful dataset creation
        logger.info("Valid dataset created successfully")

def test_create_dataset_invalid_name() -> None:
    # Try creating a dataset using a name that is not registered
    with pytest.raises(ValueError, match="not registered"):
        create_dataset("nonexistent_dataset", data_root="/tmp")
    
    # Log that the invalid dataset name was correctly rejected
    logger.info("Invalid dataset name properly rejected")

def test_create_dataset_nonexistent_root() -> None:
    # Register a dataset for testing
    register_dataset("test_nonexistent_root", NYUV2Dataset)
    
    # Try creating dataset with non-existent data root
    with pytest.raises(FileNotFoundError, match="Data root does not exist"):
        create_dataset("test_nonexistent_root", data_root="/nonexistent/path")
    
    # Log that the invalid data root was properly rejected
    logger.info("Nonexistent data root properly rejected")

def test_get_available_datasets() -> None:
    # Retrieve list of registered dataset names
    datasets = get_available_datasets()
    
    # Assert it is a list and contains at least one dataset
    assert isinstance(datasets, list)
    assert len(datasets) > 0
    
    # Log how many datasets were found
    logger.info(f"Found {len(datasets)} available datasets")

def test_dataset_info() -> None:
    # Register a dataset for info retrieval
    register_dataset("test_info", NYUV2Dataset)
    
    # Retrieve metadata about the registered dataset
    info = dataset_info("test_info")
    
    # Assert that essential metadata fields are present
    assert "name" in info
    assert "class" in info
    assert "module" in info
    assert info["name"] == "test_info"
    assert info["class"] == "NYUV2Dataset"
    
    # Log successful retrieval of dataset information
    logger.info("Dataset info retrieved successfully")

def test_dataset_info_nonexistent() -> None:
    # Try to get info for a dataset that doesn't exist
    with pytest.raises(ValueError, match="not registered"):
        dataset_info("nonexistent_dataset_info")
    
    # Log that the invalid dataset name was properly rejected
    logger.info("Nonexistent dataset info request properly rejected")

# Base Dataset Tests
def test_base_dataset_initialization_parameters() -> None:
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test invalid split parameter
        with pytest.raises(ValueError, match="Invalid split"):
            BaseDataset(temp_dir, split="invalid")
        
        # Test invalid img_size parameter
        with pytest.raises(ValueError, match="img_size must be tuple"):
            BaseDataset(temp_dir, img_size="invalid")  # type: ignore
        
        # Test invalid depth_scale parameter
        with pytest.raises(ValueError, match="depth_scale must be positive"):
            BaseDataset(temp_dir, depth_scale=-1.0)
        
        logger.info("Base dataset parameter validation working")

def test_base_dataset_nonexistent_root() -> None:
    # Test initialization with non-existent directory
    with pytest.raises(FileNotFoundError, match="Data root not found"):
        BaseDataset("/nonexistent/path")
    
    logger.info("Base dataset nonexistent root properly rejected")

def test_create_default_mask() -> None:
    # Create a temporary directory and dataset structure
    with tempfile.TemporaryDirectory() as temp_dir:
        os.makedirs(os.path.join(temp_dir, "train", "rgb"))
        os.makedirs(os.path.join(temp_dir, "train", "depth"))
        
        # Create dummy files
        dummy_img = Image.new('RGB', (640, 480), color='red')
        dummy_img.save(os.path.join(temp_dir, "train", "rgb", "test.png"))
        dummy_depth = np.ones((480, 640), dtype=np.uint16) * 1000
        cv2.imwrite(os.path.join(temp_dir, "train", "depth", "test.png"), dummy_depth)
        
        dataset = NYUV2Dataset(temp_dir, split="train")
        
        # Test mask creation with various invalid depth values
        depth = np.array([[1.0, 0.0, np.nan], [np.inf, -1.0, 2.0]])
        mask = dataset.create_default_mask(depth)
        expected = np.array([[True, False, False], [False, False, True]])
        np.testing.assert_array_equal(mask, expected)
        
        logger.info("Default mask creation working correctly")

# NYU Dataset Tests
def create_nyu_test_structure(temp_dir: str, num_samples: int = 5) -> Tuple[str, str, str]:
    train_dir = os.path.join(temp_dir, "train")
    rgb_dir = os.path.join(train_dir, "rgb")
    depth_dir = os.path.join(train_dir, "depth")
    
    # Create directories for RGB and depth images
    os.makedirs(rgb_dir)
    os.makedirs(depth_dir)
    
    # Generate dummy RGB and depth image pairs
    for i in range(num_samples):
        # Create dummy RGB image with varying intensity
        dummy_img = Image.new('RGB', (640, 480), color=(i*50, i*50, i*50))
        dummy_img.save(os.path.join(rgb_dir, f"test_{i:04d}.png"))
        
        # Create dummy depth image with random values
        dummy_depth = np.random.randint(100, 5000, (480, 640), dtype=np.uint16)
        cv2.imwrite(os.path.join(depth_dir, f"test_{i:04d}.png"), dummy_depth)
    
    return train_dir, rgb_dir, depth_dir

def test_nyu_initialization_valid() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test structure
        create_nyu_test_structure(temp_dir)
        
        # Initialize dataset with valid settings
        dataset = NYUV2Dataset(data_root=temp_dir, split="train")
        
        # Check that dataset is non-empty and properly configured
        assert len(dataset) > 0
        assert dataset.split == "train"
        assert dataset.img_size == (480, 640)
        assert dataset.depth_scale == 1000.0
        
        logger.info("NYU dataset initialized successfully")

def test_nyu_initialization_invalid_split() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Ensure ValueError is raised for invalid split name
        with pytest.raises(ValueError, match="Invalid split"):
            NYUV2Dataset(data_root=temp_dir, split="invalid")
        
        logger.info("Invalid split properly rejected")


def test_nyu_initialization_missing_directories() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create only train directory without rgb/depth subdirectories
        os.makedirs(os.path.join(temp_dir, "train"))
        
        # Should raise FileNotFoundError for missing RGB/depth directories
        with pytest.raises(FileNotFoundError):
            NYUV2Dataset(data_root=temp_dir, split="train")
        
        logger.info("Missing directories properly detected")


def test_nyu_getitem() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test structure
        create_nyu_test_structure(temp_dir)
        
        # Retrieve a sample from the dataset
        dataset = NYUV2Dataset(data_root=temp_dir, split="train")
        sample = dataset[0]
        
        # Check that the sample contains required fields with correct shapes
        assert "rgb" in sample
        assert "depth" in sample
        assert "valid_mask" in sample
        assert sample["rgb"].shape == (3, 480, 640)
        assert sample["depth"].shape == (1, 480, 640)
        assert sample["valid_mask"].shape == (1, 480, 640)
        
        logger.info("NYU sample retrieved successfully")

def test_nyu_getitem_index_bounds() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test structure
        create_nyu_test_structure(temp_dir, num_samples=3)
        
        dataset = NYUV2Dataset(data_root=temp_dir, split="train")
        
        # Test valid indices
        assert len(dataset) == 3
        sample = dataset[0]
        assert "rgb" in sample
        
        # Test out of bounds indices
        with pytest.raises(IndexError, match="Index .* out of bounds"):
            dataset[10]
        
        with pytest.raises(IndexError, match="Index .* out of bounds"):
            dataset[-1]
        
        logger.info("NYU index bounds properly validated")

def test_nyu_statistics() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test structure
        create_nyu_test_structure(temp_dir)
        
        # Compute dataset statistics
        dataset = NYUV2Dataset(data_root=temp_dir, split="train")
        stats = dataset.compute_statistics()
        
        # Ensure required statistics keys are present
        required_keys = [
            "rgb_mean", "rgb_std", "num_samples", "num_pixels", "num_valid_depth_points",
            "depth_min", "depth_max", "depth_mean", "depth_std"
        ]
        for key in required_keys:
            assert key in stats
        
        # Validate statistical values
        assert len(stats["rgb_mean"]) == 3  # RGB channels
        assert len(stats["rgb_std"]) == 3   # RGB channels
        assert stats["num_samples"] > 0
        assert stats["num_pixels"] > 0
        
        logger.info("NYU statistics computed successfully")

def test_nyu_sample_info() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test structure
        create_nyu_test_structure(temp_dir)
        
        # Retrieve sample metadata
        dataset = NYUV2Dataset(data_root=temp_dir, split="train")
        info = dataset.get_sample_info(0)
        
        # Validate presence of sample metadata keys
        required_keys = [
            "rgb_path", "depth_path", "rgb_shape", "depth_shape", 
            "rgb_mean", "rgb_std", "depth_min", "depth_max", 
            "depth_valid_ratio", "basename"
        ]
        for key in required_keys:
            assert key in info
        
        # Validate data types and ranges
        assert isinstance(info["depth_valid_ratio"], float)
        assert 0.0 <= info["depth_valid_ratio"] <= 1.0
        assert len(info["rgb_mean"]) == 3
        assert len(info["rgb_std"]) == 3
        
        logger.info("NYU sample info retrieved successfully")

# KITTI Dataset Tests
def create_kitti_test_structure(temp_dir: str, num_samples: int = 5) -> Tuple[str, str, str]:
    sequences_dir = os.path.join(temp_dir, "sequences")
    calib_dir = os.path.join(temp_dir, "calib")
    os.makedirs(sequences_dir)
    os.makedirs(calib_dir)
    
    # Create sequence 00 directory with subfolders
    seq_dir = os.path.join(sequences_dir, "00")
    os.makedirs(os.path.join(seq_dir, "image_02"))
    os.makedirs(os.path.join(seq_dir, "proj_depth", "velodyne_raw"))
    os.makedirs(os.path.join(seq_dir, "proj_depth", "groundtruth"))
    
    # Write a dummy calibration file for sequence 00
    calib_file = os.path.join(calib_dir, "00.txt")
    with open(calib_file, 'w') as f:
        f.write("P2: 7.215377e+02 0.000000e+00 6.095593e+02 0.000000e+00 0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00\n")
    
    # Generate dummy image and depth files
    for i in range(num_samples):
        # Create a grayscale RGB image and save it
        dummy_img = Image.new('RGB', (1216, 352), color=(i*50, i*50, i*50))
        dummy_img.save(os.path.join(seq_dir, "image_02", f"{i:010d}.png"))
        
        # Generate and save dummy sparse depth map
        dummy_depth = np.random.randint(100, 5000, (352, 1216), dtype=np.uint16)
        cv2.imwrite(os.path.join(seq_dir, "proj_depth", "velodyne_raw", f"{i:010d}.png"), dummy_depth)
        
        # Generate and save dummy dense depth map
        dummy_dense_depth = np.random.randint(100, 5000, (352, 1216), dtype=np.uint16)
        cv2.imwrite(os.path.join(seq_dir, "proj_depth", "groundtruth", f"{i:010d}.png"), dummy_dense_depth)
    
    return sequences_dir, calib_dir, seq_dir

def test_kitti_initialization_valid() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test structure
        create_kitti_test_structure(temp_dir)
        
        # Test proper dataset initialization with valid structure
        dataset = KITTIDataset(data_root=temp_dir, split="train")
        assert dataset.split == "train"
        assert dataset.img_size == (352, 1216)
        assert dataset.depth_scale == 256.0
        assert not dataset.use_dense_depth  # Default should be False
        
        logger.info("KITTI dataset initialized successfully")

def test_kitti_initialization_invalid_structure() -> None:
    # Test that invalid path raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        KITTIDataset(data_root="/nonexistent/path", split="train")
    
    logger.info("Invalid KITTI structure properly rejected")

def test_kitti_getitem() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test structure
        create_kitti_test_structure(temp_dir)
        
        # Test retrieval of a sample from the dataset
        dataset = KITTIDataset(data_root=temp_dir, split="train")
        
        if len(dataset) > 0:
            sample = dataset[0]
            
            # Check presence of expected keys in the sample
            assert "rgb" in sample
            assert "depth" in sample
            assert "mask" in sample
            assert "sequence" in sample
            assert "basename" in sample
            
            # Validate tensor shapes
            assert sample["rgb"].shape == (3, 352, 1216)
            assert sample["depth"].shape == (1, 352, 1216)
            assert sample["mask"] is not None
            
            logger.info("KITTI sample retrieved successfully")

def test_kitti_statistics() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test structure
        create_kitti_test_structure(temp_dir)
        
        # Test computation of dataset statistics
        dataset = KITTIDataset(data_root=temp_dir, split="train")
        stats = dataset.compute_statistics()
        
        # Check presence of expected statistical keys
        required_keys = ["rgb_mean", "rgb_std", "depth_min", "depth_max", "depth_percentiles", "num_samples"]
        for key in required_keys:
            assert key in stats
        
        # Validate statistical values
        assert len(stats["rgb_mean"]) == 3
        assert len(stats["rgb_std"]) == 3
        assert "25" in stats["depth_percentiles"]
        assert "50" in stats["depth_percentiles"]
        assert "75" in stats["depth_percentiles"]
        
        logger.info("KITTI statistics computed successfully")

def test_kitti_use_dense_depth() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test structure
        create_kitti_test_structure(temp_dir)
        
        # Test KITTI dataset with dense depth option
        dataset = KITTIDataset(data_root=temp_dir, split="train", use_dense_depth=True)
        assert dataset.use_dense_depth
        
        if len(dataset) > 0:
            sample = dataset[0]
            assert "depth" in sample
            
        logger.info("KITTI dense depth option working correctly")

def test_kitti_get_sequences() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test structure
        create_kitti_test_structure(temp_dir)
        
        dataset = KITTIDataset(data_root=temp_dir, split="train")
        sequences = dataset.get_sequences()
        
        # Should find sequence "00"
        assert isinstance(sequences, list)
        assert "00" in sequences
        
        logger.info("KITTI sequences retrieved successfully")

# ENRICH Dataset Tests
def create_enrich_test_structure(temp_dir: str, num_samples: int = 3) -> List[str]:
    datasets = ["ENRICH-Aerial", "ENRICH-Square", "ENRICH-Statue"]
    
    for dataset_name in datasets:
        dataset_path = os.path.join(temp_dir, dataset_name)
        # Create subdirectories for images and depth maps
        os.makedirs(os.path.join(dataset_path, "images"))
        os.makedirs(os.path.join(dataset_path, "depth", "exr"))
        
        for i in range(num_samples):
            # Generate and save dummy RGB image
            dummy_img = Image.new('RGB', (640, 480), color=(i*80, i*80, i*80))
            dummy_img.save(os.path.join(dataset_path, "images", f"image_{i:03d}.jpg"))
            
            # Generate and save dummy depth map as PNG (fallback since EXR might not be available)
            dummy_depth = np.random.rand(480, 640).astype(np.float32) * 1000
            cv2.imwrite(os.path.join(dataset_path, "depth", "exr", f"image_{i:03d}_depth.exr"), dummy_depth.astype(np.uint16))
    
    return datasets

def test_enrich_initialization_valid() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test structure
        create_enrich_test_structure(temp_dir)
        
        # Test valid initialization of ENRICHDataset
        dataset = ENRICHDataset(data_root=temp_dir, split="train")
        assert dataset.split == "train"
        assert dataset.img_size == (480, 640)
        assert dataset.dataset_type == "all"
        assert dataset.depth_scale == 1.0
        
        logger.info("ENRICH dataset initialized successfully")

def test_enrich_getitem() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test structure
        create_enrich_test_structure(temp_dir)
        
        # Test __getitem__ returns valid data sample with expected keys
        dataset = ENRICHDataset(data_root=temp_dir, split="train")
        
        if len(dataset) > 0:
            sample = dataset[0]
            assert "rgb" in sample
            assert "depth" in sample
            assert "valid_mask" in sample
            
            # Validate tensor shapes
            assert sample["rgb"].shape == (3, 480, 640)
            assert sample["depth"].shape == (1, 480, 640)
            assert sample["valid_mask"].shape == (1, 480, 640)
            
            logger.info("ENRICH sample retrieved successfully")

def test_enrich_statistics() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test structure
        create_enrich_test_structure(temp_dir)
        
        # Test that compute_statistics returns expected keys
        dataset = ENRICHDataset(data_root=temp_dir, split="train")
        stats = dataset.compute_statistics()
        
        required_keys = [
            "rgb_mean", "rgb_std", "num_samples", "num_pixels", "num_valid_depth_points"
        ]
        for key in required_keys:
            assert key in stats
        
        # Depth statistics might be None if no valid depth values
        assert "depth_min" in stats
        assert "depth_max" in stats
        assert "depth_mean" in stats
        assert "depth_std" in stats
        
        logger.info("ENRICH statistics computed successfully")

def test_enrich_sample_info() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test structure
        create_enrich_test_structure(temp_dir)
        
        # Test get_sample_info method
        dataset = ENRICHDataset(data_root=temp_dir, split="train")
        
        if len(dataset) > 0:
            info = dataset.get_sample_info(0)
            
            required_keys = [
                "rgb_path", "depth_path", "rgb_shape", "depth_shape",
                "rgb_mean", "rgb_std", "depth_min", "depth_max",
                "depth_valid_ratio", "dataset", "basename"
            ]
            for key in required_keys:
                assert key in info
            
            # Validate specific field types and ranges
            assert isinstance(info["depth_valid_ratio"], float)
            assert 0.0 <= info["depth_valid_ratio"] <= 1.0
            
            logger.info("ENRICH sample info retrieved successfully")

# Error Handling Tests
def test_dataset_index_out_of_bounds() -> None:
    # Create temporary directory structure for empty dataset
    with tempfile.TemporaryDirectory() as temp_dir:
        train_dir = os.path.join(temp_dir, "train")
        os.makedirs(os.path.join(train_dir, "rgb"))
        os.makedirs(os.path.join(train_dir, "depth"))
        
        # Initialize dataset with no images
        dataset = NYUV2Dataset(data_root=temp_dir, split="train", validate_data=False)
        
        # Attempt to access out-of-bounds index, expect IndexError
        with pytest.raises(IndexError, match="Index .* out of bounds"):
            dataset[0]
        
        with pytest.raises(IndexError, match="Index .* out of bounds"):
            dataset[100]

        # Log that the error was properly handled
        logger.info("Out of bounds index properly handled")

def test_dataset_corrupted_files() -> None:
    # Create temporary directory and corrupted image files
    with tempfile.TemporaryDirectory() as temp_dir:
        train_dir = os.path.join(temp_dir, "train")
        os.makedirs(os.path.join(train_dir, "rgb"))
        os.makedirs(os.path.join(train_dir, "depth"))
        
        # Write invalid content to simulate corrupted RGB image
        with open(os.path.join(train_dir, "rgb", "corrupted.png"), 'w') as f:
            f.write("not an image")
        
        # Write invalid content to simulate corrupted depth file
        with open(os.path.join(train_dir, "depth", "corrupted.png"), 'w') as f:
            f.write("not a depth file")
        
        # Initialize dataset with validation enabled
        dataset = NYUV2Dataset(data_root=temp_dir, split="train", validate_data=True)
        
        # Dataset should ignore corrupted files, hence length should be 0
        assert len(dataset) == 0

        # Log that corrupted files were properly handled
        logger.info("Corrupted files properly handled")

def test_dataset_missing_depth_files() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        train_dir = os.path.join(temp_dir, "train")
        os.makedirs(os.path.join(train_dir, "rgb"))
        os.makedirs(os.path.join(train_dir, "depth"))
        
        # Create RGB image but no corresponding depth image
        dummy_img = Image.new('RGB', (640, 480), color='red')
        dummy_img.save(os.path.join(train_dir, "rgb", "test.png"))
        
        # Initialize dataset
        dataset = NYUV2Dataset(data_root=temp_dir, split="train", validate_data=False)
        
        # Dataset should handle missing depth files gracefully
        assert len(dataset) == 0
        
        logger.info("Missing depth files properly handled")

# Caching Tests
def test_dataset_caching_enabled() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test structure
        create_nyu_test_structure(temp_dir, num_samples=1)
        
        # Initialize dataset with caching enabled
        dataset = NYUV2Dataset(data_root=temp_dir, split="train", cache=True)

        # Load the same sample twice
        sample1 = dataset[0]
        sample2 = dataset[0]

        # Ensure the cached samples are equal
        assert torch.equal(sample1["rgb"], sample2["rgb"])
        assert torch.equal(sample1["depth"], sample2["depth"])
        assert torch.equal(sample1["valid_mask"], sample2["valid_mask"])
        
        logger.info("Dataset caching working correctly")

def test_dataset_caching_disabled() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test structure
        create_nyu_test_structure(temp_dir, num_samples=1)
        
        # Initialize dataset with caching disabled
        dataset = NYUV2Dataset(data_root=temp_dir, split="train", cache=False)

        # Load the same sample twice
        sample1 = dataset[0]
        sample2 = dataset[0]

        # Even without caching, the loaded samples should be equal
        assert torch.equal(sample1["rgb"], sample2["rgb"])
        assert torch.equal(sample1["depth"], sample2["depth"])
        assert torch.equal(sample1["valid_mask"], sample2["valid_mask"])
        
        logger.info("Dataset without caching working correctly")

# Transform Tests
def test_dataset_custom_image_size() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test structure
        create_nyu_test_structure(temp_dir, num_samples=1)
        
        # Initialize dataset with custom image size
        custom_size = (256, 320)
        dataset = NYUV2Dataset(data_root=temp_dir, split="train", img_size=custom_size)

        # Load a sample and check shape matches custom size
        sample = dataset[0]
        assert sample["rgb"].shape == (3, custom_size[0], custom_size[1])
        assert sample["depth"].shape == (1, custom_size[0], custom_size[1])
        assert sample["valid_mask"].shape == (1, custom_size[0], custom_size[1])
        
        logger.info("Custom image size working correctly")

def test_dataset_depth_scale() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test structure
        create_nyu_test_structure(temp_dir, num_samples=1)
        
        # Test different depth scale values
        depth_scale = 500.0
        dataset = NYUV2Dataset(data_root=temp_dir, split="train", depth_scale=depth_scale)
        
        # Load a sample
        sample = dataset[0]
        assert sample["depth"].shape == (1, 480, 640)
        assert dataset.depth_scale == depth_scale
        
        logger.info("Depth scale parameter working correctly")

# Integration Tests
def test_dataset_full_pipeline() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test structure
        create_nyu_test_structure(temp_dir, num_samples=3)
        
        # Register dataset
        register_dataset("test_full_pipeline", NYUV2Dataset)
        
        # Create dataset through factory
        dataset = create_dataset("test_full_pipeline", data_root=temp_dir, split="train")
        
        # Verify dataset properties
        assert isinstance(dataset, NYUV2Dataset)
        assert len(dataset) == 3
        
        # Test sample loading
        sample = dataset[0]
        assert "rgb" in sample
        assert "depth" in sample
        assert "valid_mask" in sample
        
        # Test statistics computation
        stats = dataset.compute_statistics()
        assert "rgb_mean" in stats
        assert "num_samples" in stats
        
        # Test sample info retrieval
        info = dataset.get_sample_info(0)
        assert "basename" in info
        
        logger.info("Full dataset pipeline working correctly")

def test_multiple_dataset_types() -> None:
    # Register multiple dataset types
    register_dataset("test_nyu_multi", NYUV2Dataset)
    register_dataset("test_kitti_multi", KITTIDataset)
    register_dataset("test_enrich_multi", ENRICHDataset)
    
    # Verify all are registered
    available = get_available_datasets()
    assert "test_nyu_multi" in available
    assert "test_kitti_multi" in available
    assert "test_enrich_multi" in available
    
    # Test info retrieval for each
    nyu_info = dataset_info("test_nyu_multi")
    kitti_info = dataset_info("test_kitti_multi") 
    enrich_info = dataset_info("test_enrich_multi")
    
    assert nyu_info["class"] == "NYUV2Dataset"
    assert kitti_info["class"] == "KITTIDataset"
    assert enrich_info["class"] == "ENRICHDataset"
    
    logger.info("Multiple dataset types handled correctly")

def test_dataset_thread_safety() -> None:
    import threading
    import time
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test structure
        create_nyu_test_structure(temp_dir, num_samples=5)
        
        # Initialize dataset with caching
        dataset = NYUV2Dataset(data_root=temp_dir, split="train", cache=True)
        
        results = []
        
        def load_samples():
            # Load multiple samples from different threads
            for i in range(len(dataset)):
                sample = dataset[i]
                results.append(sample["rgb"].shape)
                time.sleep(0.01)  # Small delay to encourage race conditions
        
        # Create multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=load_samples)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all results are consistent
        expected_shape = (3, 480, 640)
        for result in results:
            assert result == expected_shape
        
        logger.info("Dataset thread safety verified")
        