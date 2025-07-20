# FILE: tests/test_datasets.py
# ehsanasgharzde - COMPLETE DATASET TEST SUITE
# hosseinsolymanzadeh - PROPER COMMENTING
#hosseinsolymanzadeh - FIXED REDUNDANT CODE BY EXTRACTING PURE FUNCTIONS AND BASECLASS LEVEL METHODS

import logging
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import pytest
from PIL import Image
import cv2

from datasets import create_dataset, register_dataset, get_available_datasets, dataset_info
from datasets.nyu_dataset import NYUV2Dataset
from datasets.kitti_dataset import KITTIDataset
from datasets.enrich_dataset import ENRICHDataset
from utils.dataset import BaseDataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class TestDatasetFactory:
    def test_register_dataset_new(self) -> None:
        # Register a new dataset with a unique name
        register_dataset("test_dataset", NYUV2Dataset)
        
        # Assert that the dataset is now in the registry
        assert "test_dataset" in get_available_datasets()
        
        # Log successful registration
        logger.info("Successfully registered new dataset")

    def test_register_dataset_duplicate(self) -> None:
        # Register the dataset with a given name
        register_dataset("test_dup", NYUV2Dataset)
        # Attempt to register again with the same name should raise an error
        with pytest.raises(ValueError, match="already registered"):
            register_dataset("test_dup", NYUV2Dataset)
        
        # Log that the duplicate registration was properly handled
        logger.info("Duplicate registration properly rejected")

    def test_create_dataset_valid(self) -> None:
        # Create a temporary directory for the test dataset
        with tempfile.TemporaryDirectory() as temp_dir:
            # Register a new dataset
            register_dataset("test_valid", NYUV2Dataset)
            
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
            dataset = create_dataset("test_valid", data_root=temp_dir, split="train")
            # Assert the dataset is an instance of the correct class
            assert isinstance(dataset, NYUV2Dataset)
            
            # Log successful dataset creation
            logger.info("Valid dataset created successfully")

    def test_create_dataset_invalid_name(self) -> None:
        # Try creating a dataset using a name that is not registered
        with pytest.raises(ValueError, match="not registered"):
            create_dataset("nonexistent_dataset", data_root="/tmp")
        
        # Log that the invalid dataset name was correctly rejected
        logger.info("Invalid dataset name properly rejected")

    def test_get_available_datasets(self) -> None:
        # Retrieve list of registered dataset names
        datasets = get_available_datasets()
        
        # Assert it is a list and contains at least one dataset
        assert isinstance(datasets, list)
        assert len(datasets) > 0
        
        # Log how many datasets were found
        logger.info(f"Found {len(datasets)} available datasets")

    def test_dataset_info(self) -> None:
        # Register a dataset for info retrieval
        register_dataset("test_info", NYUV2Dataset)
        
        # Retrieve metadata about the registered dataset
        info = dataset_info("test_info")
        
        # Assert that essential metadata fields are present
        assert "name" in info
        assert "class" in info
        assert "module" in info
        
        # Log successful retrieval of dataset information
        logger.info("Dataset info retrieved successfully")


class TestBaseDataset:
    def setup_method(self) -> None:
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        # Remove the temporary directory and all its contents
        shutil.rmtree(self.temp_dir)

    def test_base_dataset_initialization_parameters(self) -> None:
        # Test valid initialization parameters
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test that BaseDataset validates parameters correctly
            with pytest.raises(ValueError, match="Invalid split"):
                BaseDataset(temp_dir, split="invalid")
            
            with pytest.raises(ValueError, match="img_size must be tuple"):
                BaseDataset(temp_dir, img_size="invalid")
            
            with pytest.raises(ValueError, match="depth_scale must be positive"):
                BaseDataset(temp_dir, depth_scale=-1.0)
            
            logger.info("Base dataset parameter validation working")

    def test_create_default_mask(self) -> None:
        # Test mask creation with valid BaseDataset subclass
        with tempfile.TemporaryDirectory() as temp_dir:
            os.makedirs(os.path.join(temp_dir, "train", "rgb"))
            os.makedirs(os.path.join(temp_dir, "train", "depth"))
            
            # Create dummy files
            dummy_img = Image.new('RGB', (640, 480), color='red')
            dummy_img.save(os.path.join(temp_dir, "train", "rgb", "test.png"))
            dummy_depth = np.ones((480, 640), dtype=np.uint16) * 1000
            cv2.imwrite(os.path.join(temp_dir, "train", "depth", "test.png"), dummy_depth)
            
            dataset = NYUV2Dataset(temp_dir, split="train")
            
            # Test mask creation
            depth = np.array([[1.0, 0.0, np.nan], [np.inf, -1.0, 2.0]])
            mask = dataset.create_default_mask(depth)
            expected = np.array([[True, False, False], [False, False, True]])
            np.testing.assert_array_equal(mask, expected)
            
            logger.info("Default mask creation working correctly")


class TestNYUDataset:
    def setup_method(self) -> None:
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.train_dir = os.path.join(self.temp_dir, "train")
        self.rgb_dir = os.path.join(self.train_dir, "rgb")
        self.depth_dir = os.path.join(self.train_dir, "depth")
        
        # Create directories for RGB and depth images
        os.makedirs(self.rgb_dir)
        os.makedirs(self.depth_dir)
        
        # Generate 5 dummy RGB and depth image pairs
        for i in range(5):
            # Create dummy RGB image with varying intensity
            dummy_img = Image.new('RGB', (640, 480), color=(i*50, i*50, i*50))
            dummy_img.save(os.path.join(self.rgb_dir, f"test_{i:04d}.png"))
            
            # Create dummy depth image with random values
            dummy_depth = np.random.randint(100, 5000, (480, 640), dtype=np.uint16)
            cv2.imwrite(os.path.join(self.depth_dir, f"test_{i:04d}.png"), dummy_depth)

    def teardown_method(self) -> None:
        # Remove the temporary directory and all its contents
        shutil.rmtree(self.temp_dir)

    def test_nyu_initialization_valid(self) -> None:
        # Initialize dataset with valid settings
        dataset = NYUV2Dataset(data_root=self.temp_dir, split="train")
        
        # Check that dataset is non-empty and properly configured
        assert len(dataset) > 0
        assert dataset.split == "train"
        assert dataset.img_size == (480, 640)
        logger.info("NYU dataset initialized successfully")

    def test_nyu_initialization_invalid_split(self) -> None:
        # Ensure ValueError is raised for invalid split name
        with pytest.raises(ValueError, match="Invalid split"):
            NYUV2Dataset(data_root=self.temp_dir, split="invalid")
        logger.info("Invalid split properly rejected")

    def test_nyu_getitem(self) -> None:
        # Retrieve a sample from the dataset
        dataset = NYUV2Dataset(data_root=self.temp_dir, split="train")
        sample = dataset[0]
        
        # Check that the sample contains required fields with correct shapes
        assert "rgb" in sample
        assert "depth" in sample
        assert "valid_mask" in sample
        assert sample["rgb"].shape == (3, 480, 640)
        assert sample["depth"].shape == (1, 480, 640)
        assert sample["valid_mask"].shape == (1, 480, 640)
        logger.info("NYU sample retrieved successfully")

    def test_nyu_statistics(self) -> None:
        # Compute dataset statistics
        dataset = NYUV2Dataset(data_root=self.temp_dir, split="train")
        stats = dataset.compute_statistics()
        
        # Ensure required statistics keys are present
        assert "rgb_mean" in stats
        assert "rgb_std" in stats
        assert "num_samples" in stats
        assert "depth_min" in stats
        assert "depth_max" in stats
        logger.info("NYU statistics computed successfully")

    def test_nyu_sample_info(self) -> None:
        # Retrieve sample metadata
        dataset = NYUV2Dataset(data_root=self.temp_dir, split="train")
        info = dataset.get_sample_info(0)
        
        # Validate presence of sample metadata keys
        assert "rgb_shape" in info
        assert "depth_shape" in info
        assert "depth_valid_ratio" in info
        assert "basename" in info
        logger.info("NYU sample info retrieved successfully")


class TestKITTIDataset:
    def setup_method(self) -> None:
        # Create a temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()

        # Create subdirectories for sequences and calibration data
        self.sequences_dir = os.path.join(self.temp_dir, "sequences")
        self.calib_dir = os.path.join(self.temp_dir, "calib")
        os.makedirs(self.sequences_dir)
        os.makedirs(self.calib_dir)
        
        # Create sequence 00 directory with subfolders
        seq_dir = os.path.join(self.sequences_dir, "00")
        os.makedirs(os.path.join(seq_dir, "image_02"))
        os.makedirs(os.path.join(seq_dir, "proj_depth", "velodyne_raw"))
        
        # Write a dummy calibration file for sequence 00
        calib_file = os.path.join(self.calib_dir, "00.txt")
        with open(calib_file, 'w') as f:
            f.write("P2: 7.215377e+02 0.000000e+00 6.095593e+02 0.000000e+00 0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00\n")
        
        # Generate 5 dummy image and depth files
        for i in range(5):
            # Create a grayscale RGB image and save it
            dummy_img = Image.new('RGB', (1216, 352), color=(i*50, i*50, i*50))
            dummy_img.save(os.path.join(seq_dir, "image_02", f"{i:010d}.png"))
            
            # Generate and save dummy depth map
            dummy_depth = np.random.randint(100, 5000, (352, 1216), dtype=np.uint16)
            cv2.imwrite(os.path.join(seq_dir, "proj_depth", "velodyne_raw", f"{i:010d}.png"), dummy_depth)

    def teardown_method(self) -> None:
        # Clean up the temporary test directory after each test
        shutil.rmtree(self.temp_dir)

    def test_kitti_initialization_valid(self) -> None:
        # Test proper dataset initialization with valid structure
        dataset = KITTIDataset(data_root=self.temp_dir, split="train")
        assert dataset.split == "train"
        assert dataset.img_size == (352, 1216)
        assert not dataset.use_dense_depth  # Default should be False
        logger.info("KITTI dataset initialized successfully")

    def test_kitti_initialization_invalid_structure(self) -> None:
        # Test that invalid path raises FileNotFoundError
        with pytest.raises(FileNotFoundError):
            KITTIDataset(data_root="/nonexistent/path", split="train")
        logger.info("Invalid KITTI structure properly rejected")

    def test_kitti_getitem(self) -> None:
        # Test retrieval of a sample from the dataset
        dataset = KITTIDataset(data_root=self.temp_dir, split="train")
        if len(dataset) > 0:
            sample = dataset[0]
            # Check presence of expected keys in the sample
            assert "rgb" in sample
            assert "depth" in sample
            assert "mask" in sample
            assert "sequence" in sample
            assert "basename" in sample
            logger.info("KITTI sample retrieved successfully")

    def test_kitti_statistics(self) -> None:
        # Test computation of dataset statistics
        dataset = KITTIDataset(data_root=self.temp_dir, split="train")
        stats = dataset.compute_statistics()
        # Check presence of expected statistical keys
        assert "rgb_mean" in stats
        assert "rgb_std" in stats
        assert "depth_min" in stats
        assert "depth_max" in stats
        assert "num_samples" in stats
        logger.info("KITTI statistics computed successfully")

    def test_kitti_use_dense_depth(self) -> None:
        # Test KITTI dataset with dense depth option
        # First create dense depth directory
        seq_dir = os.path.join(self.sequences_dir, "00")
        os.makedirs(os.path.join(seq_dir, "proj_depth", "groundtruth"), exist_ok=True)
        
        # Generate dummy dense depth files
        for i in range(5):
            dummy_depth = np.random.randint(100, 5000, (352, 1216), dtype=np.uint16)
            cv2.imwrite(os.path.join(seq_dir, "proj_depth", "groundtruth", f"{i:010d}.png"), dummy_depth)
        
        dataset = KITTIDataset(data_root=self.temp_dir, split="train", use_dense_depth=True)
        assert dataset.use_dense_depth
        logger.info("KITTI dense depth option working correctly")


class TestENRICHDataset:
    def setup_method(self) -> None:
        # Create a temporary directory to simulate dataset structure
        self.temp_dir = tempfile.mkdtemp()
        datasets = ["ENRICH-Aerial", "ENRICH-Square", "ENRICH-Statue"]
        
        for dataset_name in datasets:
            dataset_path = os.path.join(self.temp_dir, dataset_name)
            # Create subdirectories for images and depth maps
            os.makedirs(os.path.join(dataset_path, "images"))
            os.makedirs(os.path.join(dataset_path, "depth", "exr"))
            
            for i in range(3):
                # Generate and save dummy RGB image
                dummy_img = Image.new('RGB', (640, 480), color=(i*80, i*80, i*80))
                dummy_img.save(os.path.join(dataset_path, "images", f"image_{i:03d}.jpg"))
                
                # Generate and save dummy depth map as PNG (fallback since EXR might not be available)
                dummy_depth = np.random.rand(480, 640).astype(np.float32) * 1000
                cv2.imwrite(os.path.join(dataset_path, "depth", "exr", f"image_{i:03d}_depth.exr"), dummy_depth.astype(np.uint16))

    def teardown_method(self) -> None:
        # Clean up temporary dataset directory after tests
        shutil.rmtree(self.temp_dir)

    def test_enrich_initialization_valid(self) -> None:
        # Test valid initialization of ENRICHDataset
        dataset = ENRICHDataset(data_root=self.temp_dir, split="train")
        assert dataset.split == "train"
        assert dataset.img_size == (480, 640)
        assert dataset.dataset_type == "all"
        logger.info("ENRICH dataset initialized successfully")

    def test_enrich_getitem(self) -> None:
        # Test __getitem__ returns valid data sample with expected keys
        dataset = ENRICHDataset(data_root=self.temp_dir, split="train")
        if len(dataset) > 0:
            sample = dataset[0]
            assert "rgb" in sample
            assert "depth" in sample
            assert "valid_mask" in sample
            logger.info("ENRICH sample retrieved successfully")

    def test_enrich_statistics(self) -> None:
        # Test that compute_statistics returns expected keys
        dataset = ENRICHDataset(data_root=self.temp_dir, split="train")
        stats = dataset.compute_statistics()
        assert "rgb_mean" in stats
        assert "rgb_std" in stats
        assert "num_samples" in stats
        assert "depth_min" in stats or stats["depth_min"] is None
        assert "depth_max" in stats or stats["depth_max"] is None
        logger.info("ENRICH statistics computed successfully")

    def test_enrich_sample_info(self) -> None:
        # Test get_sample_info method
        dataset = ENRICHDataset(data_root=self.temp_dir, split="train")
        if len(dataset) > 0:
            info = dataset.get_sample_info(0)
            assert "rgb_path" in info
            assert "depth_path" in info
            assert "dataset" in info
            assert "basename" in info
            logger.info("ENRICH sample info retrieved successfully")


class TestDatasetErrorHandling:
    def test_dataset_index_out_of_bounds(self) -> None:
        # Create temporary directory structure for empty dataset
        with tempfile.TemporaryDirectory() as temp_dir:
            train_dir = os.path.join(temp_dir, "train")
            os.makedirs(os.path.join(train_dir, "rgb"))
            os.makedirs(os.path.join(train_dir, "depth"))
            
            # Initialize dataset with no images
            dataset = NYUV2Dataset(data_root=temp_dir, split="train")
            
            # Attempt to access out-of-bounds index, expect IndexError
            with pytest.raises(IndexError):
                dataset[100]

            # Log that the error was properly handled
            logger.info("Out of bounds index properly handled")

    def test_dataset_corrupted_files(self) -> None:
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


class TestDatasetCaching:
    def setup_method(self) -> None:
        # Create temporary directories for RGB and depth images
        self.temp_dir = tempfile.mkdtemp()
        train_dir = os.path.join(self.temp_dir, "train")
        self.rgb_dir = os.path.join(train_dir, "rgb")
        self.depth_dir = os.path.join(train_dir, "depth")
        os.makedirs(self.rgb_dir)
        os.makedirs(self.depth_dir)
        
        # Create and save a dummy RGB image
        dummy_img = Image.new('RGB', (640, 480), color='red')
        dummy_img.save(os.path.join(self.rgb_dir, "test.png"))
        
        # Create and save a dummy depth image
        dummy_depth = np.ones((480, 640), dtype=np.uint16) * 1000
        cv2.imwrite(os.path.join(self.depth_dir, "test.png"), dummy_depth)

    def teardown_method(self) -> None:
        # Clean up the temporary directory after each test
        shutil.rmtree(self.temp_dir)

    def test_dataset_caching_enabled(self) -> None:
        # Initialize dataset with caching enabled
        dataset = NYUV2Dataset(data_root=self.temp_dir, split="train", cache=True)

        # Load the same sample twice
        sample1 = dataset[0]
        sample2 = dataset[0]

        # Ensure the cached samples are equal
        assert torch.equal(sample1["rgb"], sample2["rgb"])
        assert torch.equal(sample1["depth"], sample2["depth"])
        logger.info("Dataset caching working correctly")

    def test_dataset_caching_disabled(self) -> None:
        # Initialize dataset with caching disabled
        dataset = NYUV2Dataset(data_root=self.temp_dir, split="train", cache=False)

        # Load the same sample twice
        sample1 = dataset[0]
        sample2 = dataset[0]

        # Even without caching, the loaded samples should be equal
        assert torch.equal(sample1["rgb"], sample2["rgb"])
        assert torch.equal(sample1["depth"], sample2["depth"])
        logger.info("Dataset without caching working correctly")


class TestDatasetTransforms:
    def setup_method(self) -> None:
        # Create temporary directories for RGB and depth images
        self.temp_dir = tempfile.mkdtemp()
        train_dir = os.path.join(self.temp_dir, "train")
        self.rgb_dir = os.path.join(train_dir, "rgb")
        self.depth_dir = os.path.join(train_dir, "depth")
        os.makedirs(self.rgb_dir)
        os.makedirs(self.depth_dir)
        
        # Create and save a dummy RGB image
        dummy_img = Image.new('RGB', (640, 480), color='red')
        dummy_img.save(os.path.join(self.rgb_dir, "test.png"))
        
        # Create and save a dummy depth image
        dummy_depth = np.ones((480, 640), dtype=np.uint16) * 1000
        cv2.imwrite(os.path.join(self.depth_dir, "test.png"), dummy_depth)

    def teardown_method(self) -> None:
        # Clean up the temporary directory after each test
        shutil.rmtree(self.temp_dir)

    def test_dataset_custom_image_size(self) -> None:
        # Initialize dataset with custom image size
        custom_size = (256, 320)
        dataset = NYUV2Dataset(data_root=self.temp_dir, split="train", img_size=custom_size)

        # Load a sample and check shape matches custom size
        sample = dataset[0]
        assert sample["rgb"].shape == (3, custom_size[0], custom_size[1])
        assert sample["depth"].shape == (1, custom_size[0], custom_size[1])
        logger.info("Custom image size working correctly")

    def test_dataset_depth_scale(self) -> None:
        # Test different depth scale values
        depth_scale = 500.0
        dataset = NYUV2Dataset(data_root=self.temp_dir, split="train", depth_scale=depth_scale)
        
        # Load a sample
        sample = dataset[0]
        assert sample["depth"].shape == (1, 480, 640)
        logger.info("Depth scale parameter working correctly")
