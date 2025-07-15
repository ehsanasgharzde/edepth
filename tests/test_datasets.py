# FILE: tests/test_datasets.py
# ehsanasgharzde - COMPLETE DATASET TEST SUITE

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
        register_dataset("test_dataset", NYUV2Dataset)
        assert "test_dataset" in get_available_datasets()
        logger.info("Successfully registered new dataset")

    def test_register_dataset_duplicate(self):
        register_dataset("test_dup", NYUV2Dataset)
        with pytest.raises(ValueError, match="already registered"):
            register_dataset("test_dup", NYUV2Dataset)
        logger.info("Duplicate registration properly rejected")

    def test_create_dataset_valid(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            register_dataset("test_valid", NYUV2Dataset)
            os.makedirs(os.path.join(temp_dir, "rgb"))
            os.makedirs(os.path.join(temp_dir, "depth"))
            
            dummy_img = Image.new('RGB', (640, 480), color='red')
            dummy_img.save(os.path.join(temp_dir, "rgb", "test.png"))
            
            dummy_depth = np.ones((480, 640), dtype=np.uint16) * 1000
            cv2.imwrite(os.path.join(temp_dir, "depth", "test.png"), dummy_depth)
            
            dataset = create_dataset("test_valid", data_root=temp_dir, split="train")
            assert isinstance(dataset, NYUV2Dataset)
            logger.info("Valid dataset created successfully")

    def test_create_dataset_invalid_name(self):
        with pytest.raises(ValueError, match="not in registry"):
            create_dataset("nonexistent_dataset", data_root="/tmp")
        logger.info("Invalid dataset name properly rejected")

    def test_get_available_datasets(self):
        datasets = get_available_datasets()
        assert isinstance(datasets, list)
        assert len(datasets) > 0
        logger.info(f"Found {len(datasets)} available datasets")

    def test_dataset_info(self):
        register_dataset("test_info", NYUV2Dataset)
        info = dataset_info("test_info")
        assert "name" in info
        assert "class" in info
        assert "module" in info
        logger.info("Dataset info retrieved successfully")

class TestNYUDataset:
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.rgb_dir = os.path.join(self.temp_dir, "rgb")
        self.depth_dir = os.path.join(self.temp_dir, "depth")
        os.makedirs(self.rgb_dir)
        os.makedirs(self.depth_dir)
        
        for i in range(5):
            dummy_img = Image.new('RGB', (640, 480), color=(i*50, i*50, i*50))
            dummy_img.save(os.path.join(self.rgb_dir, f"test_{i:04d}.png"))
            
            dummy_depth = np.random.randint(100, 5000, (480, 640), dtype=np.uint16)
            cv2.imwrite(os.path.join(self.depth_dir, f"test_{i:04d}.png"), dummy_depth)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir)

    def test_nyu_initialization_valid(self):
        dataset = NYUV2Dataset(data_root=self.temp_dir, split="train")
        assert len(dataset) > 0
        assert dataset.split == "train"
        assert dataset.img_size == (480, 640)
        logger.info("NYU dataset initialized successfully")

    def test_nyu_initialization_invalid_split(self):
        with pytest.raises(ValueError, match="Invalid split"):
            NYUV2Dataset(data_root=self.temp_dir, split="invalid")
        logger.info("Invalid split properly rejected")

    def test_nyu_initialization_invalid_depth_range(self):
        with pytest.raises(ValueError, match="min_depth must be less than max_depth"):
            NYUV2Dataset(data_root=self.temp_dir, min_depth=10.0, max_depth=5.0)
        logger.info("Invalid depth range properly rejected")

    def test_nyu_getitem(self):
        dataset = NYUV2Dataset(data_root=self.temp_dir, split="train")
        sample = dataset[0]
        assert "rgb" in sample
        assert "depth" in sample
        assert "valid_mask" in sample
        assert sample["rgb"].shape == (3, 480, 640)
        assert sample["depth"].shape == (1, 480, 640)
        assert sample["valid_mask"].shape == (1, 480, 640)
        logger.info("NYU sample retrieved successfully")

    def test_nyu_statistics(self):
        dataset = NYUV2Dataset(data_root=self.temp_dir, split="train")
        stats = dataset.compute_statistics()
        assert "rgb_mean" in stats
        assert "rgb_std" in stats
        assert "depth_stats" in stats
        logger.info("NYU statistics computed successfully")

    def test_nyu_sample_info(self):
        dataset = NYUV2Dataset(data_root=self.temp_dir, split="train")
        info = dataset.get_sample_info(0)
        assert "rgb_shape" in info
        assert "depth_shape" in info
        assert "valid_ratio" in info
        logger.info("NYU sample info retrieved successfully")

class TestKITTIDataset:
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.sequences_dir = os.path.join(self.temp_dir, "sequences")
        self.calib_dir = os.path.join(self.temp_dir, "calib")
        os.makedirs(self.sequences_dir)
        os.makedirs(self.calib_dir)
        
        seq_dir = os.path.join(self.sequences_dir, "00")
        os.makedirs(os.path.join(seq_dir, "image_2"))
        os.makedirs(os.path.join(seq_dir, "velodyne"))
        
        calib_file = os.path.join(self.calib_dir, "00.txt")
        with open(calib_file, 'w') as f:
            f.write("P2: 7.215377e+02 0.000000e+00 6.095593e+02 0.000000e+00 0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00\n")
        
        for i in range(5):
            dummy_img = Image.new('RGB', (1216, 352), color=(i*50, i*50, i*50))
            dummy_img.save(os.path.join(seq_dir, "image_2", f"{i:010d}.png"))
            
            dummy_points = np.random.rand(1000, 4).astype(np.float32)
            dummy_points.tofile(os.path.join(seq_dir, "velodyne", f"{i:010d}.bin"))

    def teardown_method(self):
        shutil.rmtree(self.temp_dir)

    def test_kitti_initialization_valid(self):
        dataset = KITTIDataset(data_root=self.temp_dir, split="train")
        assert dataset.split == "train"
        assert dataset.img_size == (352, 1216)
        logger.info("KITTI dataset initialized successfully")

    def test_kitti_initialization_invalid_structure(self):
        with pytest.raises(FileNotFoundError):
            KITTIDataset(data_root="/nonexistent/path", split="train")
        logger.info("Invalid KITTI structure properly rejected")

    def test_kitti_getitem(self):
        dataset = KITTIDataset(data_root=self.temp_dir, split="train")
        if len(dataset) > 0:
            sample = dataset[0]
            assert "rgb" in sample
            assert "depth" in sample
            assert "valid_mask" in sample
            logger.info("KITTI sample retrieved successfully")

    def test_kitti_statistics(self):
        dataset = KITTIDataset(data_root=self.temp_dir, split="train")
        stats = dataset.compute_statistics()
        assert "rgb_mean" in stats
        assert "rgb_std" in stats
        logger.info("KITTI statistics computed successfully")

class TestENRICHDataset:
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        datasets = ["ENRICH-Aerial", "ENRICH-Square", "ENRICH-Statue"]
        
        for dataset_name in datasets:
            dataset_path = os.path.join(self.temp_dir, dataset_name)
            os.makedirs(os.path.join(dataset_path, "images"))
            os.makedirs(os.path.join(dataset_path, "depth", "exr"))
            
            for i in range(3):
                dummy_img = Image.new('RGB', (640, 480), color=(i*80, i*80, i*80))
                dummy_img.save(os.path.join(dataset_path, "images", f"image_{i:03d}.jpg"))
                
                dummy_depth = np.random.rand(480, 640).astype(np.float32)
                depth_file = os.path.join(dataset_path, "depth", "exr", f"image_{i:03d}.exr")
                cv2.imwrite(depth_file, dummy_depth)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir)

    def test_enrich_initialization_valid(self):
        dataset = ENRICHDataset(data_root=self.temp_dir, split="train")
        assert dataset.split == "train"
        assert dataset.img_size == (480, 640)
        logger.info("ENRICH dataset initialized successfully")

    def test_enrich_initialization_invalid_dataset_type(self):
        with pytest.raises(ValueError, match="Invalid dataset_type"):
            ENRICHDataset(data_root=self.temp_dir, dataset_type="invalid")
        logger.info("Invalid ENRICH dataset type properly rejected")

    def test_enrich_getitem(self):
        dataset = ENRICHDataset(data_root=self.temp_dir, split="train")
        if len(dataset) > 0:
            sample = dataset[0]
            assert "rgb" in sample
            assert "depth" in sample
            assert "valid_mask" in sample
            assert "dataset" in sample
            logger.info("ENRICH sample retrieved successfully")

    def test_enrich_statistics(self):
        dataset = ENRICHDataset(data_root=self.temp_dir, split="train")
        stats = dataset.compute_statistics()
        assert "rgb_mean" in stats
        assert "rgb_std" in stats
        assert "depth_stats" in stats
        logger.info("ENRICH statistics computed successfully")

class TestUnrealDataset:
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        scene_dir = os.path.join(self.temp_dir, "Scene001")
        os.makedirs(os.path.join(scene_dir, "Image0"))
        os.makedirs(os.path.join(scene_dir, "Disp0"))
        
        for i in range(3):
            dummy_img = Image.new('RGB', (640, 480), color=(i*80, i*80, i*80))
            dummy_img.save(os.path.join(scene_dir, "Image0", f"frame_{i:04d}.png"))
            
            dummy_disp = np.random.rand(480, 640).astype(np.float32)
            np.save(os.path.join(scene_dir, "Disp0", f"frame_{i:04d}.npy"), dummy_disp)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir)

    def test_unreal_initialization_valid(self):
        dataset = UnrealStereo4KDataset(data_root=self.temp_dir, split="train")
        assert dataset.split == "train"
        assert dataset.img_size == (480, 640)
        logger.info("Unreal dataset initialized successfully")

    def test_unreal_initialization_invalid_structure(self):
        with pytest.raises(FileNotFoundError):
            UnrealStereo4KDataset(data_root="/nonexistent/path", split="train")
        logger.info("Invalid Unreal structure properly rejected")

    def test_unreal_getitem(self):
        dataset = UnrealStereo4KDataset(data_root=self.temp_dir, split="train")
        if len(dataset) > 0:
            sample = dataset[0]
            assert "rgb" in sample
            assert "depth" in sample
            assert "valid_mask" in sample
            logger.info("Unreal sample retrieved successfully")

    def test_unreal_statistics(self):
        dataset = UnrealStereo4KDataset(data_root=self.temp_dir, split="train")
        stats = dataset.compute_statistics()
        assert "rgb_mean" in stats
        assert "rgb_std" in stats
        assert "depth_stats" in stats
        logger.info("Unreal statistics computed successfully")

class TestDatasetValidation:
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.temp_dir)

    def test_validate_nyu_structure_valid(self):
        os.makedirs(os.path.join(self.temp_dir, "rgb"))
        os.makedirs(os.path.join(self.temp_dir, "depth"))
        
        dummy_img = Image.new('RGB', (640, 480), color='red')
        dummy_img.save(os.path.join(self.temp_dir, "rgb", "test.png"))
        
        dummy_depth = np.ones((480, 640), dtype=np.uint16) * 1000
        cv2.imwrite(os.path.join(self.temp_dir, "depth", "test.png"), dummy_depth)
        
        report = validate_dataset_structure(self.temp_dir, "nyu")
        assert report["status"] == "valid"
        logger.info("NYU structure validation passed")

    def test_validate_dataset_structure_invalid_type(self):
        with pytest.raises(ValueError, match="Unsupported dataset type"):
            validate_dataset_structure(self.temp_dir, "invalid_type")
        logger.info("Invalid dataset type properly rejected")

class TestDatasetStatistics:
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.rgb_dir = os.path.join(self.temp_dir, "rgb")
        self.depth_dir = os.path.join(self.temp_dir, "depth")
        os.makedirs(self.rgb_dir)
        os.makedirs(self.depth_dir)
        
        for i in range(10):
            dummy_img = Image.new('RGB', (640, 480), color=(i*25, i*25, i*25))
            dummy_img.save(os.path.join(self.rgb_dir, f"test_{i:04d}.png"))
            
            dummy_depth = np.random.randint(100, 5000, (480, 640), dtype=np.uint16)
            cv2.imwrite(os.path.join(self.depth_dir, f"test_{i:04d}.png"), dummy_depth)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir)

    def test_compute_dataset_statistics(self):
        dataset = NYUV2Dataset(data_root=self.temp_dir, split="train")
        stats = compute_dataset_statistics(dataset)
        assert "rgb_mean" in stats
        assert "rgb_std" in stats
        assert "depth_stats" in stats
        assert len(stats["rgb_mean"]) == 3
        assert len(stats["rgb_std"]) == 3
        logger.info("Dataset statistics computed successfully")

class TestDatasetErrorHandling:
    def test_dataset_index_out_of_bounds(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            os.makedirs(os.path.join(temp_dir, "rgb"))
            os.makedirs(os.path.join(temp_dir, "depth"))
            
            dataset = NYUV2Dataset(data_root=temp_dir, split="train")
            with pytest.raises(IndexError):
                dataset[100]
            logger.info("Out of bounds index properly handled")

    def test_dataset_corrupted_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            os.makedirs(os.path.join(temp_dir, "rgb"))
            os.makedirs(os.path.join(temp_dir, "depth"))
            
            with open(os.path.join(temp_dir, "rgb", "corrupted.png"), 'w') as f:
                f.write("not an image")
            
            with open(os.path.join(temp_dir, "depth", "corrupted.png"), 'w') as f:
                f.write("not a depth file")
            
            dataset = NYUV2Dataset(data_root=temp_dir, split="train", validate_data=True)
            assert len(dataset) == 0
            logger.info("Corrupted files properly handled")

class TestDatasetCaching:
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.rgb_dir = os.path.join(self.temp_dir, "rgb")
        self.depth_dir = os.path.join(self.temp_dir, "depth")
        os.makedirs(self.rgb_dir)
        os.makedirs(self.depth_dir)
        
        dummy_img = Image.new('RGB', (640, 480), color='red')
        dummy_img.save(os.path.join(self.rgb_dir, "test.png"))
        
        dummy_depth = np.ones((480, 640), dtype=np.uint16) * 1000
        cv2.imwrite(os.path.join(self.depth_dir, "test.png"), dummy_depth)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir)

    def test_dataset_caching_enabled(self):
        dataset = NYUV2Dataset(data_root=self.temp_dir, split="train", cache=True)
        sample1 = dataset[0]
        sample2 = dataset[0]
        assert torch.equal(sample1["rgb"], sample2["rgb"])
        assert torch.equal(sample1["depth"], sample2["depth"])
        logger.info("Dataset caching working correctly")

    def test_dataset_caching_disabled(self):
        dataset = NYUV2Dataset(data_root=self.temp_dir, split="train", cache=False)
        sample1 = dataset[0]
        sample2 = dataset[0]
        assert torch.equal(sample1["rgb"], sample2["rgb"])
        assert torch.equal(sample1["depth"], sample2["depth"])
        logger.info("Dataset without caching working correctly")

class TestDatasetAugmentation:
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.rgb_dir = os.path.join(self.temp_dir, "rgb")
        self.depth_dir = os.path.join(self.temp_dir, "depth")
        os.makedirs(self.rgb_dir)
        os.makedirs(self.depth_dir)
        
        dummy_img = Image.new('RGB', (640, 480), color='red')
        dummy_img.save(os.path.join(self.rgb_dir, "test.png"))
        
        dummy_depth = np.ones((480, 640), dtype=np.uint16) * 1000
        cv2.imwrite(os.path.join(self.depth_dir, "test.png"), dummy_depth)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir)

    def test_dataset_augmentation_enabled(self):
        dataset = NYUV2Dataset(data_root=self.temp_dir, split="train", augmentation=True)
        sample = dataset[0]
        assert sample["rgb"].shape == (3, 480, 640)
        assert sample["depth"].shape == (1, 480, 640)
        logger.info("Dataset augmentation working correctly")

    def test_dataset_augmentation_disabled(self):
        dataset = NYUV2Dataset(data_root=self.temp_dir, split="val", augmentation=False)
        sample = dataset[0]
        assert sample["rgb"].shape == (3, 480, 640)
        assert sample["depth"].shape == (1, 480, 640)
        logger.info("Dataset without augmentation working correctly")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])