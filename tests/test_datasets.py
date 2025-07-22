# FILE: tests/test_datasets.py
# ehsanasgharzde - COMPLETE DATASET TEST SUITE
# hosseinsolymanzadeh - PROPER COMMENTING
# hosseinsolymanzadeh - FIXED REDUNDANT CODE BY EXTRACTING PURE FUNCTIONS AND BASECLASS LEVEL METHODS
# ehsanasgharzde - REPLACED PIL WITH CV2 FOR CONSISTENCY

import os
import cv2
import torch
import pytest
import logging
import tempfile
import numpy as np
from typing import List, Tuple

from datasets import create_dataset, register_dataset, get_available_datasets, dataset_info
from datasets.nyu_dataset import NYUV2Dataset
from datasets.kitti_dataset import KITTIDataset
from datasets.enrich_dataset import ENRICHDataset
from utils.dataset_utils import BaseDataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def test_register_dataset_new() -> None:
    register_dataset("test_dataset_new", NYUV2Dataset)
    assert "test_dataset_new" in get_available_datasets()
    logger.info("Successfully registered new dataset")

def test_register_dataset_duplicate() -> None:
    register_dataset("test_dup", NYUV2Dataset)
    with pytest.raises(ValueError, match="already registered"):
        register_dataset("test_dup", NYUV2Dataset)
    logger.info("Duplicate registration properly rejected")

def test_create_dataset_valid() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        register_dataset("test_valid_create", NYUV2Dataset)
        train_dir = os.path.join(temp_dir, "train")
        os.makedirs(os.path.join(train_dir, "rgb"))
        os.makedirs(os.path.join(train_dir, "depth"))
        
        dummy_img = np.full((480, 640, 3), [0, 0, 255], dtype=np.uint8)
        cv2.imwrite(os.path.join(train_dir, "rgb", "test.png"), dummy_img)
        
        dummy_depth = np.ones((480, 640), dtype=np.uint16) * 1000
        cv2.imwrite(os.path.join(train_dir, "depth", "test.png"), dummy_depth)
        
        dataset = create_dataset("test_valid_create", data_root=temp_dir, split="train")
        assert isinstance(dataset, NYUV2Dataset)
        assert len(dataset) > 0
        logger.info("Valid dataset created successfully")

def test_create_dataset_invalid_name() -> None:
    with pytest.raises(ValueError, match="not registered"):
        create_dataset("nonexistent_dataset", data_root="/tmp")
    logger.info("Invalid dataset name properly rejected")

def test_create_dataset_nonexistent_root() -> None:
    register_dataset("test_nonexistent_root", NYUV2Dataset)
    with pytest.raises(FileNotFoundError, match="Data root does not exist"):
        create_dataset("test_nonexistent_root", data_root="/nonexistent/path")
    logger.info("Nonexistent data root properly rejected")

def test_get_available_datasets() -> None:
    datasets = get_available_datasets()
    assert isinstance(datasets, list)
    assert len(datasets) > 0
    logger.info(f"Found {len(datasets)} available datasets")

def test_dataset_info() -> None:
    register_dataset("test_info", NYUV2Dataset)
    info = dataset_info("test_info")
    assert "name" in info
    assert "class" in info
    assert "module" in info
    assert info["name"] == "test_info"
    assert info["class"] == "NYUV2Dataset"
    logger.info("Dataset info retrieved successfully")

def test_dataset_info_nonexistent() -> None:
    with pytest.raises(ValueError, match="not registered"):
        dataset_info("nonexistent_dataset_info")
    logger.info("Nonexistent dataset info request properly rejected")

def test_base_dataset_initialization_parameters() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(ValueError, match="Invalid split"):
            BaseDataset(temp_dir, split="invalid")
        with pytest.raises(ValueError, match="img_size must be tuple"):
            BaseDataset(temp_dir, img_size="invalid")
        with pytest.raises(ValueError, match="depth_scale must be positive"):
            BaseDataset(temp_dir, depth_scale=-1.0)
        logger.info("Base dataset parameter validation working")

def test_base_dataset_nonexistent_root() -> None:
    with pytest.raises(FileNotFoundError, match="Data root not found"):
        BaseDataset("/nonexistent/path")
    logger.info("Base dataset nonexistent root properly rejected")

def test_create_default_mask() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        os.makedirs(os.path.join(temp_dir, "train", "rgb"))
        os.makedirs(os.path.join(temp_dir, "train", "depth"))
        
        dummy_img = np.full((480, 640, 3), [0, 0, 255], dtype=np.uint8)
        cv2.imwrite(os.path.join(temp_dir, "train", "rgb", "test.png"), dummy_img)
        dummy_depth = np.ones((480, 640), dtype=np.uint16) * 1000
        cv2.imwrite(os.path.join(temp_dir, "train", "depth", "test.png"), dummy_depth)
        
        dataset = NYUV2Dataset(temp_dir, split="train")
        depth = np.array([[1.0, 0.0, np.nan], [np.inf, -1.0, 2.0]])
        mask = dataset.create_default_mask(depth)
        expected = np.array([[True, False, False], [False, False, True]])
        np.testing.assert_array_equal(mask, expected)
        logger.info("Default mask creation working correctly")

def create_nyu_test_structure(temp_dir: str, num_samples: int = 5) -> Tuple[str, str, str]:
    train_dir = os.path.join(temp_dir, "train")
    rgb_dir = os.path.join(train_dir, "rgb")
    depth_dir = os.path.join(train_dir, "depth")
    
    os.makedirs(rgb_dir)
    os.makedirs(depth_dir)
    
    for i in range(num_samples):
        color_val = min(i*50, 255)
        dummy_img = np.full((480, 640, 3), [color_val, color_val, color_val], dtype=np.uint8)
        cv2.imwrite(os.path.join(rgb_dir, f"test_{i:04d}.png"), dummy_img)
        
        dummy_depth = np.random.randint(100, 5000, (480, 640), dtype=np.uint16)
        cv2.imwrite(os.path.join(depth_dir, f"test_{i:04d}.png"), dummy_depth)
    
    return train_dir, rgb_dir, depth_dir

def test_nyu_initialization_valid() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        create_nyu_test_structure(temp_dir)
        dataset = NYUV2Dataset(data_root=temp_dir, split="train")
        assert len(dataset) > 0
        assert dataset.split == "train"
        assert dataset.img_size == (480, 640)
        assert dataset.depth_scale == 1000.0
        logger.info("NYU dataset initialized successfully")

def test_nyu_initialization_invalid_split() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(ValueError, match="Invalid split"):
            NYUV2Dataset(data_root=temp_dir, split="invalid")
        logger.info("Invalid split properly rejected")

def test_nyu_initialization_missing_directories() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        os.makedirs(os.path.join(temp_dir, "train"))
        with pytest.raises(FileNotFoundError):
            NYUV2Dataset(data_root=temp_dir, split="train")
        logger.info("Missing directories properly detected")

def test_nyu_getitem() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        create_nyu_test_structure(temp_dir)
        dataset = NYUV2Dataset(data_root=temp_dir, split="train")
        sample = dataset[0]
        assert "rgb" in sample
        assert "depth" in sample
        assert "valid_mask" in sample
        assert sample["rgb"].shape == (3, 480, 640)
        assert sample["depth"].shape == (1, 480, 640)
        assert sample["valid_mask"].shape == (1, 480, 640)
        logger.info("NYU sample retrieved successfully")

def test_nyu_getitem_index_bounds() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        create_nyu_test_structure(temp_dir, num_samples=3)
        dataset = NYUV2Dataset(data_root=temp_dir, split="train")
        assert len(dataset) == 3
        sample = dataset[0]
        assert "rgb" in sample
        with pytest.raises(IndexError, match="Index .* out of bounds"):
            dataset[10]
        with pytest.raises(IndexError, match="Index .* out of bounds"):
            dataset[-1]
        logger.info("NYU index bounds properly validated")

def test_nyu_statistics() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        create_nyu_test_structure(temp_dir)
        dataset = NYUV2Dataset(data_root=temp_dir, split="train")
        stats = dataset.compute_statistics()
        required_keys = [
            "rgb_mean", "rgb_std", "num_samples", "num_pixels", "num_valid_depth_points",
            "depth_min", "depth_max", "depth_mean", "depth_std"
        ]
        for key in required_keys:
            assert key in stats
        assert len(stats["rgb_mean"]) == 3
        assert len(stats["rgb_std"]) == 3
        assert stats["num_samples"] > 0
        assert stats["num_pixels"] > 0
        logger.info("NYU statistics computed successfully")

def test_nyu_sample_info() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        create_nyu_test_structure(temp_dir)
        dataset = NYUV2Dataset(data_root=temp_dir, split="train")
        info = dataset.get_sample_info(0)
        required_keys = [
            "rgb_path", "depth_path", "rgb_shape", "depth_shape", 
            "rgb_mean", "rgb_std", "depth_min", "depth_max", 
            "depth_valid_ratio", "basename"
        ]
        for key in required_keys:
            assert key in info
        assert isinstance(info["depth_valid_ratio"], float)
        assert 0.0 <= info["depth_valid_ratio"] <= 1.0
        assert len(info["rgb_mean"]) == 3
        assert len(info["rgb_std"]) == 3
        logger.info("NYU sample info retrieved successfully")

def create_kitti_test_structure(temp_dir: str, num_samples: int = 5) -> Tuple[str, str, str]:
    sequences_dir = os.path.join(temp_dir, "sequences")
    calib_dir = os.path.join(temp_dir, "calib")
    os.makedirs(sequences_dir)
    os.makedirs(calib_dir)
    
    seq_dir = os.path.join(sequences_dir, "00")
    os.makedirs(os.path.join(seq_dir, "image_02"))
    os.makedirs(os.path.join(seq_dir, "proj_depth", "velodyne_raw"))
    os.makedirs(os.path.join(seq_dir, "proj_depth", "groundtruth"))
    
    calib_file = os.path.join(calib_dir, "00.txt")
    with open(calib_file, 'w') as f:
        f.write("P2: 7.215377e+02 0.000000e+00 6.095593e+02 0.000000e+00 0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00\n")
    
    for i in range(num_samples):
        color_val = min(i*50, 255)
        dummy_img = np.full((352, 1216, 3), [color_val, color_val, color_val], dtype=np.uint8)
        cv2.imwrite(os.path.join(seq_dir, "image_02", f"{i:010d}.png"), dummy_img)
        
        dummy_depth = np.random.randint(100, 5000, (352, 1216), dtype=np.uint16)
        cv2.imwrite(os.path.join(seq_dir, "proj_depth", "velodyne_raw", f"{i:010d}.png"), dummy_depth)
        
        dummy_dense_depth = np.random.randint(100, 5000, (352, 1216), dtype=np.uint16)
        cv2.imwrite(os.path.join(seq_dir, "proj_depth", "groundtruth", f"{i:010d}.png"), dummy_dense_depth)
    
    return sequences_dir, calib_dir, seq_dir

def test_kitti_initialization_valid() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        create_kitti_test_structure(temp_dir)
        dataset = KITTIDataset(data_root=temp_dir, split="train")
        assert dataset.split == "train"
        assert dataset.img_size == (352, 1216)
        assert dataset.depth_scale == 256.0
        assert not dataset.use_dense_depth
        logger.info("KITTI dataset initialized successfully")

def test_kitti_initialization_invalid_structure() -> None:
    with pytest.raises(FileNotFoundError):
        KITTIDataset(data_root="/nonexistent/path", split="train")
    logger.info("Invalid KITTI structure properly rejected")

def test_kitti_getitem() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        create_kitti_test_structure(temp_dir)
        dataset = KITTIDataset(data_root=temp_dir, split="train")
        if len(dataset) > 0:
            sample = dataset[0]
            assert "rgb" in sample
            assert "depth" in sample
            assert "mask" in sample
            assert "sequence" in sample
            assert "basename" in sample
            assert sample["rgb"].shape == (3, 352, 1216)
            assert sample["depth"].shape == (1, 352, 1216)
            assert sample["mask"] is not None
            logger.info("KITTI sample retrieved successfully")

def test_kitti_statistics() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        create_kitti_test_structure(temp_dir)
        dataset = KITTIDataset(data_root=temp_dir, split="train")
        stats = dataset.compute_statistics()
        required_keys = ["rgb_mean", "rgb_std", "depth_min", "depth_max", "depth_percentiles", "num_samples"]
        for key in required_keys:
            assert key in stats
        assert len(stats["rgb_mean"]) == 3
        assert len(stats["rgb_std"]) == 3
        assert "25" in stats["depth_percentiles"]
        assert "50" in stats["depth_percentiles"]
        assert "75" in stats["depth_percentiles"]
        logger.info("KITTI statistics computed successfully")

def test_kitti_use_dense_depth() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        create_kitti_test_structure(temp_dir)
        dataset = KITTIDataset(data_root=temp_dir, split="train", use_dense_depth=True)
        assert dataset.use_dense_depth
        if len(dataset) > 0:
            sample = dataset[0]
            assert "depth" in sample
            logger.info("KITTI dense depth option working correctly")

def test_kitti_get_sequences() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        create_kitti_test_structure(temp_dir)
        dataset = KITTIDataset(data_root=temp_dir, split="train")
        sequences = dataset.get_sequences()
        assert isinstance(sequences, list)
        assert "00" in sequences
        logger.info("KITTI sequences retrieved successfully")

def create_enrich_test_structure(temp_dir: str, num_samples: int = 3) -> List[str]:
    datasets = ["ENRICH-Aerial", "ENRICH-Square", "ENRICH-Statue"]
    for dataset_name in datasets:
        dataset_path = os.path.join(temp_dir, dataset_name)
        os.makedirs(os.path.join(dataset_path, "images"))
        os.makedirs(os.path.join(dataset_path, "depth", "exr"))
        for i in range(num_samples):
            color_val = min(i*80, 255)
            dummy_img = np.full((480, 640, 3), [color_val, color_val, color_val], dtype=np.uint8)
            cv2.imwrite(os.path.join(dataset_path, "images", f"image_{i:03d}.jpg"), dummy_img)
            dummy_depth = np.random.rand(480, 640).astype(np.float32) * 1000
            cv2.imwrite(os.path.join(dataset_path, "depth", "exr", f"image_{i:03d}_depth.exr"), dummy_depth.astype(np.uint16))
    return datasets

def test_enrich_initialization_valid() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        create_enrich_test_structure(temp_dir)
        dataset = ENRICHDataset(data_root=temp_dir, split="train")
        assert dataset.split == "train"
        assert dataset.img_size == (480, 640)
        assert dataset.dataset_type == "all"
        assert dataset.depth_scale == 1.0
        logger.info("ENRICH dataset initialized successfully")

def test_enrich_getitem() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        create_enrich_test_structure(temp_dir)
        dataset = ENRICHDataset(data_root=temp_dir, split="train")
        if len(dataset) > 0:
            sample = dataset[0]
            assert "rgb" in sample
            assert "depth" in sample
            assert "valid_mask" in sample
            assert sample["rgb"].shape == (3, 480, 640)
            assert sample["depth"].shape == (1, 480, 640)
            assert sample["valid_mask"].shape == (1, 480, 640)
            logger.info("ENRICH sample retrieved successfully")

def test_enrich_statistics() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        create_enrich_test_structure(temp_dir)
        dataset = ENRICHDataset(data_root=temp_dir, split="train")
        stats = dataset.compute_statistics()
        required_keys = ["rgb_mean", "rgb_std", "num_samples", "num_pixels", "num_valid_depth_points"]
        for key in required_keys:
            assert key in stats
        assert "depth_min" in stats
        assert "depth_max" in stats
        assert "depth_mean" in stats
        assert "depth_std" in stats
        logger.info("ENRICH statistics computed successfully")

def test_enrich_sample_info() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        create_enrich_test_structure(temp_dir)
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
            assert isinstance(info["depth_valid_ratio"], float)
            assert 0.0 <= info["depth_valid_ratio"] <= 1.0
            logger.info("ENRICH sample info retrieved successfully")

def test_dataset_index_out_of_bounds() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        train_dir = os.path.join(temp_dir, "train")
        os.makedirs(os.path.join(train_dir, "rgb"))
        os.makedirs(os.path.join(train_dir, "depth"))
        dataset = NYUV2Dataset(data_root=temp_dir, split="train", validate_data=False)
        with pytest.raises(IndexError, match="Index .* out of bounds"):
            dataset[0]
        with pytest.raises(IndexError, match="Index .* out of bounds"):
            dataset[100]
        logger.info("Out of bounds index properly handled")

def test_dataset_corrupted_files() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        train_dir = os.path.join(temp_dir, "train")
        os.makedirs(os.path.join(train_dir, "rgb"))
        os.makedirs(os.path.join(train_dir, "depth"))
        with open(os.path.join(train_dir, "rgb", "corrupted.png"), 'w') as f:
            f.write("not an image")
        with open(os.path.join(train_dir, "depth", "corrupted.png"), 'w') as f:
            f.write("not a depth file")
        dataset = NYUV2Dataset(data_root=temp_dir, split="train", validate_data=True)
        assert len(dataset) == 0
        logger.info("Corrupted files properly handled")

def test_dataset_missing_depth_files() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        train_dir = os.path.join(temp_dir, "train")
        os.makedirs(os.path.join(train_dir, "rgb"))
        os.makedirs(os.path.join(train_dir, "depth"))
        dummy_img = np.full((480, 640, 3), [0, 0, 255], dtype=np.uint8)
        cv2.imwrite(os.path.join(train_dir, "rgb", "test.png"), dummy_img)
        dataset = NYUV2Dataset(data_root=temp_dir, split="train", validate_data=False)
        assert len(dataset) == 0
        logger.info("Missing depth files properly handled")

def test_dataset_caching_enabled() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        create_nyu_test_structure(temp_dir, num_samples=1)
        dataset = NYUV2Dataset(data_root=temp_dir, split="train", cache=True)
        sample1 = dataset[0]
        sample2 = dataset[0]
        assert torch.equal(sample1["rgb"], sample2["rgb"])
        assert torch.equal(sample1["depth"], sample2["depth"])
        assert torch.equal(sample1["valid_mask"], sample2["valid_mask"])
        logger.info("Dataset caching working correctly")

def test_dataset_caching_disabled() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        create_nyu_test_structure(temp_dir, num_samples=1)
        dataset = NYUV2Dataset(data_root=temp_dir, split="train", cache=False)
        sample1 = dataset[0]
        sample2 = dataset[0]
        assert torch.equal(sample1["rgb"], sample2["rgb"])
        assert torch.equal(sample1["depth"], sample2["depth"])
        assert torch.equal(sample1["valid_mask"], sample2["valid_mask"])
        logger.info("Dataset without caching working correctly")

def test_dataset_custom_image_size() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        create_nyu_test_structure(temp_dir, num_samples=1)
        custom_size = (256, 320)
        dataset = NYUV2Dataset(data_root=temp_dir, split="train", img_size=custom_size)
        sample = dataset[0]
        assert sample["rgb"].shape == (3, custom_size[0], custom_size[1])
        assert sample["depth"].shape == (1, custom_size[0], custom_size[1])
        assert sample["valid_mask"].shape == (1, custom_size[0], custom_size[1])
        logger.info("Custom image size working correctly")

def test_dataset_depth_scale() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        create_nyu_test_structure(temp_dir, num_samples=1)
        depth_scale = 500.0
        dataset = NYUV2Dataset(data_root=temp_dir, split="train", depth_scale=depth_scale)
        sample = dataset[0]
        assert sample["depth"].shape == (1, 480, 640)
        assert dataset.depth_scale == depth_scale
        logger.info("Depth scale parameter working correctly")

def test_dataset_full_pipeline() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        create_nyu_test_structure(temp_dir, num_samples=3)
        register_dataset("test_full_pipeline", NYUV2Dataset)
        dataset = create_dataset("test_full_pipeline", data_root=temp_dir, split="train")
        assert isinstance(dataset, NYUV2Dataset)
        assert len(dataset) == 3
        sample = dataset[0]
        assert "rgb" in sample
        assert "depth" in sample
        assert "valid_mask" in sample
        stats = dataset.compute_statistics()
        assert "rgb_mean" in stats
        assert "num_samples" in stats
        info = dataset.get_sample_info(0)
        assert "basename" in info
        logger.info("Full dataset pipeline working correctly")

def test_multiple_dataset_types() -> None:
    register_dataset("test_nyu_multi", NYUV2Dataset)
    register_dataset("test_kitti_multi", KITTIDataset)
    register_dataset("test_enrich_multi", ENRICHDataset)
    available = get_available_datasets()
    assert "test_nyu_multi" in available
    assert "test_kitti_multi" in available
    assert "test_enrich_multi" in available
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
        create_nyu_test_structure(temp_dir, num_samples=5)
        dataset = NYUV2Dataset(data_root=temp_dir, split="train", cache=True)
        results = []
        
        def load_samples():
            for i in range(len(dataset)):
                sample = dataset[i]
                results.append(sample["rgb"].shape)
                time.sleep(0.01)
        
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=load_samples)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        expected_shape = (3, 480, 640)
        for result in results:
            assert result == expected_shape
        
        logger.info("Dataset thread safety verified")
