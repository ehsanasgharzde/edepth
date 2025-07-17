# FILE: tests/test_datasets.py
# ehsanasgharzde - COMPLETE DATASET TEST SUITE
# hosseinsolymanzadeh - PROPER COMMENTING

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

from datasets import create_dataset, register_dataset, validate_dataset_structure, compute_dataset_statistics, get_available_datasets, dataset_info
from datasets.nyu_dataset import NYUV2Dataset
from datasets.kitti_dataset import KITTIDataset
from datasets.enrich_dataset import ENRICHDataset
from datasets.unreal_dataset import UnrealStereo4KDataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class TestDatasetFactory:
    def test_register_dataset_new(self):
        # Register a new dataset with a unique name
        register_dataset("test_dataset", NYUV2Dataset)
        
        # Assert that the dataset is now in the registry
        assert "test_dataset" in get_available_datasets()
        
        # Log successful registration
        logger.info("Successfully registered new dataset")

    def test_register_dataset_duplicate(self):
        # Register the dataset with a given name
        register_dataset("test_dup", NYUV2Dataset)
        # Attempt to register again with the same name should raise an error
        with pytest.raises(ValueError, match="already registered"):
            register_dataset("test_dup", NYUV2Dataset)
        
        # Log that the duplicate registration was properly handled
        logger.info("Duplicate registration properly rejected")

    def test_create_dataset_valid(self):
        # Create a temporary directory for the test dataset
        with tempfile.TemporaryDirectory() as temp_dir:
            # Register a new dataset
            register_dataset("test_valid", NYUV2Dataset)
            # Create RGB and depth directories
            os.makedirs(os.path.join(temp_dir, "rgb"))
            os.makedirs(os.path.join(temp_dir, "depth"))
            # Create and save a dummy RGB image
            dummy_img = Image.new('RGB', (640, 480), color='red')
            dummy_img.save(os.path.join(temp_dir, "rgb", "test.png"))
            # Create and save a dummy depth image
            dummy_depth = np.ones((480, 640), dtype=np.uint16) * 1000
            cv2.imwrite(os.path.join(temp_dir, "depth", "test.png"), dummy_depth)
            # Attempt to create a dataset instance from the prepared data
            dataset = create_dataset("test_valid", data_root=temp_dir, split="train")
            # Assert the dataset is an instance of the correct class
            assert isinstance(dataset, NYUV2Dataset)
            
            # Log successful dataset creation
            logger.info("Valid dataset created successfully")

    def test_create_dataset_invalid_name(self):
        # Try creating a dataset using a name that is not registered
        with pytest.raises(ValueError, match="not in registry"):
            create_dataset("nonexistent_dataset", data_root="/tmp")
        
        # Log that the invalid dataset name was correctly rejected
        logger.info("Invalid dataset name properly rejected")

    def test_get_available_datasets(self):
        # Retrieve list of registered dataset names
        datasets = get_available_datasets()
        
        # Assert it is a list and contains at least one dataset
        assert isinstance(datasets, list)
        assert len(datasets) > 0
        
        # Log how many datasets were found
        logger.info(f"Found {len(datasets)} available datasets")

    def test_dataset_info(self):
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


class TestNYUDataset:
    def setup_method(self):
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.rgb_dir = os.path.join(self.temp_dir, "rgb")
        self.depth_dir = os.path.join(self.temp_dir, "depth")
        
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

    def teardown_method(self):
        # Remove the temporary directory and all its contents
        shutil.rmtree(self.temp_dir)

    def test_nyu_initialization_valid(self):
        # Initialize dataset with valid settings
        dataset = NYUV2Dataset(data_root=self.temp_dir, split="train")
        
        # Check that dataset is non-empty and properly configured
        assert len(dataset) > 0
        assert dataset.split == "train"
        assert dataset.img_size == (480, 640)
        logger.info("NYU dataset initialized successfully")

    def test_nyu_initialization_invalid_split(self):
        # Ensure ValueError is raised for invalid split name
        with pytest.raises(ValueError, match="Invalid split"):
            NYUV2Dataset(data_root=self.temp_dir, split="invalid")
        logger.info("Invalid split properly rejected")

    def test_nyu_initialization_invalid_depth_range(self):
        # Ensure ValueError is raised when min_depth is greater than max_depth
        with pytest.raises(ValueError, match="min_depth must be less than max_depth"):
            NYUV2Dataset(data_root=self.temp_dir, min_depth=10.0, max_depth=5.0)
        logger.info("Invalid depth range properly rejected")

    def test_nyu_getitem(self):
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

    def test_nyu_statistics(self):
        # Compute dataset statistics
        dataset = NYUV2Dataset(data_root=self.temp_dir, split="train")
        stats = dataset.compute_statistics()
        
        # Ensure required statistics keys are present
        assert "rgb_mean" in stats
        assert "rgb_std" in stats
        assert "depth_stats" in stats
        logger.info("NYU statistics computed successfully")

    def test_nyu_sample_info(self):
        # Retrieve sample metadata
        dataset = NYUV2Dataset(data_root=self.temp_dir, split="train")
        info = dataset.get_sample_info(0)
        
        # Validate presence of sample metadata keys
        assert "rgb_shape" in info
        assert "depth_shape" in info
        assert "valid_ratio" in info
        logger.info("NYU sample info retrieved successfully")


class TestKITTIDataset:
    def setup_method(self):
        # Create a temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()

        # Create subdirectories for sequences and calibration data
        self.sequences_dir = os.path.join(self.temp_dir, "sequences")
        self.calib_dir = os.path.join(self.temp_dir, "calib")
        os.makedirs(self.sequences_dir)
        os.makedirs(self.calib_dir)
        
        # Create sequence 00 directory with subfolders for images and point clouds
        seq_dir = os.path.join(self.sequences_dir, "00")
        os.makedirs(os.path.join(seq_dir, "image_2"))
        os.makedirs(os.path.join(seq_dir, "velodyne"))
        
        # Write a dummy calibration file for sequence 00
        calib_file = os.path.join(self.calib_dir, "00.txt")
        with open(calib_file, 'w') as f:
            f.write("P2: 7.215377e+02 0.000000e+00 6.095593e+02 0.000000e+00 0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00\n")
        
        # Generate 5 dummy image and point cloud files
        for i in range(5):
            # Create a grayscale RGB image and save it
            dummy_img = Image.new('RGB', (1216, 352), color=(i*50, i*50, i*50))
            dummy_img.save(os.path.join(seq_dir, "image_2", f"{i:010d}.png"))
            
            # Generate and save dummy LIDAR point cloud data
            dummy_points = np.random.rand(1000, 4).astype(np.float32)
            dummy_points.tofile(os.path.join(seq_dir, "velodyne", f"{i:010d}.bin"))

    def teardown_method(self):
        # Clean up the temporary test directory after each test
        shutil.rmtree(self.temp_dir)

    def test_kitti_initialization_valid(self):
        # Test proper dataset initialization with valid structure
        dataset = KITTIDataset(data_root=self.temp_dir, split="train")
        assert dataset.split == "train"
        assert dataset.img_size == (352, 1216)
        logger.info("KITTI dataset initialized successfully")

    def test_kitti_initialization_invalid_structure(self):
        # Test that invalid path raises FileNotFoundError
        with pytest.raises(FileNotFoundError):
            KITTIDataset(data_root="/nonexistent/path", split="train")
        logger.info("Invalid KITTI structure properly rejected")

    def test_kitti_getitem(self):
        # Test retrieval of a sample from the dataset
        dataset = KITTIDataset(data_root=self.temp_dir, split="train")
        if len(dataset) > 0:
            sample = dataset[0]
            # Check presence of expected keys in the sample
            assert "rgb" in sample
            assert "depth" in sample
            assert "valid_mask" in sample
            logger.info("KITTI sample retrieved successfully")

    def test_kitti_statistics(self):
        # Test computation of dataset statistics
        dataset = KITTIDataset(data_root=self.temp_dir, split="train")
        stats = dataset.compute_statistics()
        # Check presence of expected statistical keys
        assert "rgb_mean" in stats
        assert "rgb_std" in stats
        logger.info("KITTI statistics computed successfully")


class TestENRICHDataset:
    def setup_method(self):
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
                
                # Generate and save dummy depth map in EXR format
                dummy_depth = np.random.rand(480, 640).astype(np.float32)
                depth_file = os.path.join(dataset_path, "depth", "exr", f"image_{i:03d}.exr")
                cv2.imwrite(depth_file, dummy_depth)

    def teardown_method(self):
        # Clean up temporary dataset directory after tests
        shutil.rmtree(self.temp_dir)

    def test_enrich_initialization_valid(self):
        # Test valid initialization of ENRICHDataset
        dataset = ENRICHDataset(data_root=self.temp_dir, split="train")
        assert dataset.split == "train"
        assert dataset.img_size == (480, 640)
        logger.info("ENRICH dataset initialized successfully")

    def test_enrich_initialization_invalid_dataset_type(self):
        # Test invalid dataset_type raises ValueError
        with pytest.raises(ValueError, match="Invalid dataset_type"):
            ENRICHDataset(data_root=self.temp_dir, dataset_type="invalid")
        logger.info("Invalid ENRICH dataset type properly rejected")

    def test_enrich_getitem(self):
        # Test __getitem__ returns valid data sample with expected keys
        dataset = ENRICHDataset(data_root=self.temp_dir, split="train")
        if len(dataset) > 0:
            sample = dataset[0]
            assert "rgb" in sample
            assert "depth" in sample
            assert "valid_mask" in sample
            assert "dataset" in sample
            logger.info("ENRICH sample retrieved successfully")

    def test_enrich_statistics(self):
        # Test that compute_statistics returns expected keys
        dataset = ENRICHDataset(data_root=self.temp_dir, split="train")
        stats = dataset.compute_statistics()
        assert "rgb_mean" in stats
        assert "rgb_std" in stats
        assert "depth_stats" in stats
        logger.info("ENRICH statistics computed successfully")


class TestUnrealDataset:
    def setup_method(self):
        # Create a temporary directory to simulate the dataset root
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a mock scene directory with subfolders for images and disparities
        scene_dir = os.path.join(self.temp_dir, "Scene001")
        os.makedirs(os.path.join(scene_dir, "Image0"))
        os.makedirs(os.path.join(scene_dir, "Disp0"))
        
        # Generate dummy RGB images and disparity maps
        for i in range(3):
            dummy_img = Image.new('RGB', (640, 480), color=(i*80, i*80, i*80))
            dummy_img.save(os.path.join(scene_dir, "Image0", f"frame_{i:04d}.png"))
            
            dummy_disp = np.random.rand(480, 640).astype(np.float32)
            np.save(os.path.join(scene_dir, "Disp0", f"frame_{i:04d}.npy"), dummy_disp)

    def teardown_method(self):
        # Remove the temporary directory and all its contents
        shutil.rmtree(self.temp_dir)

    def test_unreal_initialization_valid(self):
        # Test dataset initialization with a valid dataset structure
        dataset = UnrealStereo4KDataset(data_root=self.temp_dir, split="train")
        assert dataset.split == "train"                  # Check correct split assignment
        assert dataset.img_size == (480, 640)            # Check that image size is correctly inferred
        logger.info("Unreal dataset initialized successfully")

    def test_unreal_initialization_invalid_structure(self):
        # Test that dataset raises FileNotFoundError when given a nonexistent path
        with pytest.raises(FileNotFoundError):
            UnrealStereo4KDataset(data_root="/nonexistent/path", split="train")
        logger.info("Invalid Unreal structure properly rejected")

    def test_unreal_getitem(self):
        # Test __getitem__ returns a valid sample dictionary with required keys
        dataset = UnrealStereo4KDataset(data_root=self.temp_dir, split="train")
        if len(dataset) > 0:
            sample = dataset[0]
            assert "rgb" in sample                      # Sample should include RGB image
            assert "depth" in sample                    # Sample should include depth map
            assert "valid_mask" in sample               # Sample should include validity mask
            logger.info("Unreal sample retrieved successfully")

    def test_unreal_statistics(self):
        # Test compute_statistics returns dictionary with expected keys
        dataset = UnrealStereo4KDataset(data_root=self.temp_dir, split="train")
        stats = dataset.compute_statistics()
        assert "rgb_mean" in stats                     # Should contain RGB mean
        assert "rgb_std" in stats                      # Should contain RGB standard deviation
        assert "depth_stats" in stats                  # Should contain depth statistics
        logger.info("Unreal statistics computed successfully")


class TestDatasetValidation:
    def setup_method(self):
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        # Remove the temporary directory after test
        shutil.rmtree(self.temp_dir)

    def test_validate_nyu_structure_valid(self):
        # Create required subdirectories for the NYU dataset structure
        os.makedirs(os.path.join(self.temp_dir, "rgb"))
        os.makedirs(os.path.join(self.temp_dir, "depth"))
        
        # Create and save a dummy RGB image
        dummy_img = Image.new('RGB', (640, 480), color='red')
        dummy_img.save(os.path.join(self.temp_dir, "rgb", "test.png"))
        
        # Create and save a dummy depth image with constant value
        dummy_depth = np.ones((480, 640), dtype=np.uint16) * 1000
        cv2.imwrite(os.path.join(self.temp_dir, "depth", "test.png"), dummy_depth)
        
        # Validate the NYU dataset structure and assert that it is valid
        report = validate_dataset_structure(self.temp_dir, "nyu")
        assert report["status"] == "valid"
        logger.info("NYU structure validation passed")

    def test_validate_dataset_structure_invalid_type(self):
        # Expect a ValueError when an unsupported dataset type is passed
        with pytest.raises(ValueError, match="Unsupported dataset type"):
            validate_dataset_structure(self.temp_dir, "invalid_type")
        logger.info("Invalid dataset type properly rejected")


class TestDatasetStatistics:
    def setup_method(self):
        # Create a temporary directory for the dataset
        self.temp_dir = tempfile.mkdtemp()
        self.rgb_dir = os.path.join(self.temp_dir, "rgb")
        self.depth_dir = os.path.join(self.temp_dir, "depth")
        
        # Create subdirectories for RGB and depth images
        os.makedirs(self.rgb_dir)
        os.makedirs(self.depth_dir)
        
        # Generate 10 dummy RGB and depth image pairs
        for i in range(10):
            # Create a dummy RGB image with increasing grayscale intensity
            dummy_img = Image.new('RGB', (640, 480), color=(i*25, i*25, i*25))
            dummy_img.save(os.path.join(self.rgb_dir, f"test_{i:04d}.png"))
            
            # Create a dummy depth image with random values
            dummy_depth = np.random.randint(100, 5000, (480, 640), dtype=np.uint16)
            cv2.imwrite(os.path.join(self.depth_dir, f"test_{i:04d}.png"), dummy_depth)

    def teardown_method(self):
        # Remove the temporary directory and its contents after tests
        shutil.rmtree(self.temp_dir)

    def test_compute_dataset_statistics(self):
        # Initialize the dataset using the temporary directory
        dataset = NYUV2Dataset(data_root=self.temp_dir, split="train")
        
        # Compute dataset statistics (mean, std for RGB; stats for depth)
        stats = compute_dataset_statistics(dataset)
        
        # Check that the expected keys exist in the result
        assert "rgb_mean" in stats
        assert "rgb_std" in stats
        assert "depth_stats" in stats
        
        # Ensure RGB mean and std contain three values (R, G, B)
        assert len(stats["rgb_mean"]) == 3
        assert len(stats["rgb_std"]) == 3

        # Log the success of statistics computation
        logger.info("Dataset statistics computed successfully")


class TestDatasetErrorHandling:
    def test_dataset_index_out_of_bounds(self):
        # Create temporary directory structure for empty dataset
        with tempfile.TemporaryDirectory() as temp_dir:
            os.makedirs(os.path.join(temp_dir, "rgb"))
            os.makedirs(os.path.join(temp_dir, "depth"))
            
            # Initialize dataset with no images
            dataset = NYUV2Dataset(data_root=temp_dir, split="train")
            
            # Attempt to access out-of-bounds index, expect IndexError
            with pytest.raises(IndexError):
                dataset[100]

            # Log that the error was properly handled
            logger.info("Out of bounds index properly handled")

    def test_dataset_corrupted_files(self):
        # Create temporary directory and corrupted image files
        with tempfile.TemporaryDirectory() as temp_dir:
            os.makedirs(os.path.join(temp_dir, "rgb"))
            os.makedirs(os.path.join(temp_dir, "depth"))
            
            # Write invalid content to simulate corrupted RGB image
            with open(os.path.join(temp_dir, "rgb", "corrupted.png"), 'w') as f:
                f.write("not an image")
            
            # Write invalid content to simulate corrupted depth file
            with open(os.path.join(temp_dir, "depth", "corrupted.png"), 'w') as f:
                f.write("not a depth file")
            
            # Initialize dataset with validation enabled
            dataset = NYUV2Dataset(data_root=temp_dir, split="train", validate_data=True)
            
            # Dataset should ignore corrupted files, hence length should be 0
            assert len(dataset) == 0

            # Log that corrupted files were properly handled
            logger.info("Corrupted files properly handled")


class TestDatasetCaching:
    def setup_method(self):
        # Create temporary directories for RGB and depth images
        self.temp_dir = tempfile.mkdtemp()
        self.rgb_dir = os.path.join(self.temp_dir, "rgb")
        self.depth_dir = os.path.join(self.temp_dir, "depth")
        os.makedirs(self.rgb_dir)
        os.makedirs(self.depth_dir)
        
        # Create and save a dummy RGB image
        dummy_img = Image.new('RGB', (640, 480), color='red')
        dummy_img.save(os.path.join(self.rgb_dir, "test.png"))
        
        # Create and save a dummy depth image
        dummy_depth = np.ones((480, 640), dtype=np.uint16) * 1000
        cv2.imwrite(os.path.join(self.depth_dir, "test.png"), dummy_depth)

    def teardown_method(self):
        # Clean up the temporary directory after each test
        shutil.rmtree(self.temp_dir)

    def test_dataset_caching_enabled(self):
        # Initialize dataset with caching enabled
        dataset = NYUV2Dataset(data_root=self.temp_dir, split="train", cache=True)

        # Load the same sample twice
        sample1 = dataset[0]
        sample2 = dataset[0]

        # Ensure the cached samples are equal
        assert torch.equal(sample1["rgb"], sample2["rgb"])
        assert torch.equal(sample1["depth"], sample2["depth"])
        logger.info("Dataset caching working correctly")

    def test_dataset_caching_disabled(self):
        # Initialize dataset with caching disabled
        dataset = NYUV2Dataset(data_root=self.temp_dir, split="train", cache=False)

        # Load the same sample twice
        sample1 = dataset[0]
        sample2 = dataset[0]

        # Even without caching, the loaded samples should be equal
        assert torch.equal(sample1["rgb"], sample2["rgb"])
        assert torch.equal(sample1["depth"], sample2["depth"])
        logger.info("Dataset without caching working correctly")


class TestDatasetAugmentation:
    def setup_method(self):
        # Create temporary directories for RGB and depth images
        self.temp_dir = tempfile.mkdtemp()
        self.rgb_dir = os.path.join(self.temp_dir, "rgb")
        self.depth_dir = os.path.join(self.temp_dir, "depth")
        os.makedirs(self.rgb_dir)
        os.makedirs(self.depth_dir)
        
        # Create and save a dummy RGB image
        dummy_img = Image.new('RGB', (640, 480), color='red')
        dummy_img.save(os.path.join(self.rgb_dir, "test.png"))
        
        # Create and save a dummy depth image
        dummy_depth = np.ones((480, 640), dtype=np.uint16) * 1000
        cv2.imwrite(os.path.join(self.depth_dir, "test.png"), dummy_depth)

    def teardown_method(self):
        # Clean up the temporary directory after each test
        shutil.rmtree(self.temp_dir)

    def test_dataset_augmentation_enabled(self):
        # Initialize dataset with augmentation enabled
        dataset = NYUV2Dataset(data_root=self.temp_dir, split="train", augmentation=True)

        # Load a sample and check shape of RGB and depth
        sample = dataset[0]
        assert sample["rgb"].shape == (3, 480, 640)
        assert sample["depth"].shape == (1, 480, 640)
        logger.info("Dataset augmentation working correctly")

    def test_dataset_augmentation_disabled(self):
        # Initialize dataset with augmentation disabled
        dataset = NYUV2Dataset(data_root=self.temp_dir, split="val", augmentation=False)

        # Load a sample and check shape of RGB and depth
        sample = dataset[0]
        assert sample["rgb"].shape == (3, 480, 640)
        assert sample["depth"].shape == (1, 480, 640)
        logger.info("Dataset without augmentation working correctly")

if __name__ == "__main__":
    # Run the tests using pytest in verbose mode
    pytest.main([__file__, "-v"])
