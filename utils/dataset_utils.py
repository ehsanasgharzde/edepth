# FILE: utils/dataset.py
# ehsanasgharzde, hosseinsolymanzadeh - FIXED REDUNDANT CODE BY EXTRACTING PURE FUNCTIONS AND BASECLASS LEVEL METHODS

import cv2
import logging
import hashlib
import threading
import numpy as np
from pathlib import Path
import albumentations as A
from abc import abstractmethod
from cachetools import LRUCache
from torch.utils.data import Dataset, ConcatDataset
from typing import Tuple, Dict, Any, List, Optional
from albumentations.pytorch import ToTensorV2
import torch
import random

logger = logging.getLogger(__name__)

class BaseDataset(Dataset):
    def __init__(self, data_root: str, split: str = 'train', img_size: Tuple[int, int] = (480, 640),
                 depth_scale: float = 1.0, cache: bool = False, validate_data: bool = True, cache_size: int = 1000):
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = img_size
        self.depth_scale = depth_scale
        self.cache = cache
        self.cache_lock = threading.RLock()
        self.validate_data = validate_data
        self.rgb_transform = self.get_rgb_transform()
        self.depth_transform = self.get_depth_transform()
        self.samples = []

        self.validate_initialization_parameters()

        if validate_data:
            self.validate_dataset_structure()
            
        self.samples = self.load_samples()
        
        if cache:
            self.cache_data = LRUCache(maxsize=cache_size)
            self.cache_lock = threading.Lock()
        else:
            self.cache_data = None
            self.cache_lock = None
    
    def validate_initialization_parameters(self):
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root not found: {self.data_root}")
        
        if self.split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split: {self.split}")
        
        if not isinstance(self.img_size, tuple) or len(self.img_size) != 2:
            raise ValueError("img_size must be tuple of (height, width)")
        
        if self.depth_scale <= 0:
            raise ValueError("depth_scale must be positive")

    @abstractmethod
    def validate_dataset_structure(self):
        pass

    def validate_sample_integrity(self, sample: Dict[str, Any]) -> bool:
        try:
            required_files = [sample['rgb'], sample['depth']]
            for file_path in required_files:
                if not file_path.exists() or file_path.stat().st_size < 1024:
                    logger.warning(f"Invalid file: {file_path}")
                    return False

            img = cv2.imread(str(sample['rgb']))
            if img is None or img.size == 0:
                logger.warning(f"Corrupt or unreadable RGB image: {sample['rgb']}")
                return False

            return True

        except Exception as e:
            logger.error(f"Sample integrity check failed: {e}")
            return False

    def create_default_mask(self, target: np.ndarray) -> np.ndarray:
        mask = ((target > 0) & 
                (~np.isnan(target)) & 
                (~np.isinf(target))).astype(bool)

        if mask.sum() == 0:
            logger.warning("No valid pixels found after applying depth range mask")
            return np.zeros_like(target, dtype=bool)
        
        return mask

    def get_rgb_transform(self, MEAN: tuple = (0.485, 0.456, 0.406), STD: tuple = (0.229, 0.224, 0.225)):
        transforms = [A.Resize(self.img_size[0], self.img_size[1], interpolation=1)]
        
        transforms.extend([
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2()
        ]) # type: ignore
        
        return A.Compose(transforms) # type: ignore
    
    def get_depth_transform(self):
        transforms = [A.Resize(self.img_size[0], self.img_size[1], interpolation=0)]
        transforms.append(ToTensorV2()) # type: ignore
        return A.Compose(transforms) # type: ignore

    @abstractmethod
    def load_samples(self) -> List[Dict[str, Path]]:
        pass

    def _get_cache_key(self, path: Path, method: str) -> str:
        path_str = str(path.resolve())
        return f"{method}:{hashlib.md5(path_str.encode()).hexdigest()}"

    def load_rgb(self, path: Path) -> np.ndarray:
        if self.cache_data is not None:
            cache_key = self._get_cache_key(path, 'rgb')
            with self.cache_lock: # type: ignore
                if cache_key in self.cache_data:
                    return self.cache_data[cache_key]
        
        try:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                raise IOError(f"Unable to read RGB image: {path}")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if img.shape[:2] != self.img_size:
                img = cv2.resize(img, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_LINEAR)
            
            if self.cache_data is not None:
                with self.cache_lock: # type: ignore
                    self.cache_data[cache_key] = img.copy()
            
            return img
        except Exception as e:
            logger.error(f"Failed to load RGB image {path}: {e}")
            raise

    def load_depth(self, path: Path) -> np.ndarray:
        if self.cache_data is not None:
            cache_key = self._get_cache_key(path, 'depth')
            with self.cache_lock: # type: ignore
                if cache_key in self.cache_data:
                    return self.cache_data[cache_key]
        
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise IOError(f"Unable to read depth image: {path}")
        
        depth = depth.astype(np.float32) / self.depth_scale
        
        invalid_mask = (depth <= 0) | np.isnan(depth) | np.isinf(depth)
        depth[invalid_mask] = 0.0
        
        if self.cache_data is not None:
            with self.cache_lock: # type: ignore
                self.cache_data[cache_key] = depth.copy()
        
        return depth

    def __len__(self):
        return len(self.samples)


class WeightedCombinedDataset(Dataset):
    def __init__(self, datasets: List[Dataset], weights: Optional[List[float]] = None, 
                 strategy: str = 'weighted_random', seed: int = 42):
        self.datasets = datasets
        self.strategy = strategy
        self.rng = random.Random(seed)
        
        if not datasets:
            raise ValueError("At least one dataset must be provided")
        
        self.dataset_lengths = [len(dataset) for dataset in datasets] # type: ignore
        self.total_length = sum(self.dataset_lengths)
        
        if weights is None:
            self.weights = [length / self.total_length for length in self.dataset_lengths]
        else:
            if len(weights) != len(datasets):
                raise ValueError("Number of weights must match number of datasets")
            total_weight = sum(weights)
            self.weights = [w / total_weight for w in weights]
        
        self.cumulative_lengths = []
        cumsum = 0
        for length in self.dataset_lengths:
            cumsum += length
            self.cumulative_lengths.append(cumsum)
        
        if strategy == 'weighted_random':
            self.epoch_length = max(self.dataset_lengths)
        elif strategy == 'round_robin':
            self.epoch_length = max(self.dataset_lengths)
        elif strategy == 'balanced':
            self.epoch_length = max(self.dataset_lengths) * len(datasets)
        else:
            self.epoch_length = self.total_length

    def __len__(self):
        return self.epoch_length

    def __getitem__(self, idx):
        if self.strategy == 'weighted_random':
            dataset_idx = self.rng.choices(range(len(self.datasets)), weights=self.weights)[0]
            sample_idx = self.rng.randint(0, len(self.datasets[dataset_idx]) - 1) # type: ignore
            return self.datasets[dataset_idx][sample_idx]
        
        elif self.strategy == 'round_robin':
            dataset_idx = idx % len(self.datasets)
            sample_idx = idx // len(self.datasets)
            if sample_idx >= len(self.datasets[dataset_idx]):
                sample_idx = sample_idx % len(self.datasets[dataset_idx])
            return self.datasets[dataset_idx][sample_idx]
        
        elif self.strategy == 'balanced':
            cycle_length = max(self.dataset_lengths)
            cycle_idx = idx // cycle_length
            dataset_idx = cycle_idx % len(self.datasets)
            sample_idx = idx % cycle_length
            if sample_idx >= len(self.datasets[dataset_idx]):
                sample_idx = sample_idx % len(self.datasets[dataset_idx])
            return self.datasets[dataset_idx][sample_idx]
        
        else:  # sequential
            for i, cumulative_length in enumerate(self.cumulative_lengths):
                if idx < cumulative_length:
                    dataset_idx = i
                    sample_idx = idx - (cumulative_length - self.dataset_lengths[i])
                    return self.datasets[dataset_idx][sample_idx]


class StratifiedCombinedDataset(Dataset):
    def __init__(self, datasets: List[Dataset], strata_sizes: Optional[List[int]] = None, 
                 shuffle: bool = True, seed: int = 42):
        self.datasets = datasets
        self.shuffle = shuffle
        self.rng = random.Random(seed)
        
        if not datasets:
            raise ValueError("At least one dataset must be provided")
        
        self.dataset_lengths = [len(dataset) for dataset in datasets] # type: ignore
        
        if strata_sizes is None:
            min_size = min(self.dataset_lengths)
            self.strata_sizes = [min_size] * len(datasets)
        else:
            if len(strata_sizes) != len(datasets):
                raise ValueError("Number of strata sizes must match number of datasets")
            self.strata_sizes = strata_sizes
        
        self.total_length = sum(self.strata_sizes)
        self.indices = self._generate_indices()

    def _generate_indices(self):
        indices = []
        for dataset_idx, strata_size in enumerate(self.strata_sizes):
            dataset_indices = list(range(len(self.datasets[dataset_idx]))) # type: ignore
            if self.shuffle:
                self.rng.shuffle(dataset_indices)
            
            selected_indices = dataset_indices[:strata_size]
            if len(selected_indices) < strata_size:
                selected_indices = (selected_indices * (strata_size // len(selected_indices) + 1))[:strata_size]
            
            for sample_idx in selected_indices:
                indices.append((dataset_idx, sample_idx))
        
        if self.shuffle:
            self.rng.shuffle(indices)
        
        return indices

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        dataset_idx, sample_idx = self.indices[idx]
        return self.datasets[dataset_idx][sample_idx]


class AdaptiveCombinedDataset(Dataset):
    def __init__(self, datasets: List[Dataset], performance_weights: Optional[List[float]] = None,
                 adaptation_rate: float = 0.1, min_weight: float = 0.05):
        self.datasets = datasets
        self.adaptation_rate = adaptation_rate
        self.min_weight = min_weight
        
        if not datasets:
            raise ValueError("At least one dataset must be provided")
        
        self.dataset_lengths = [len(dataset) for dataset in datasets] # type: ignore
        
        if performance_weights is None:
            self.weights = [1.0 / len(datasets)] * len(datasets)
        else:
            if len(performance_weights) != len(datasets):
                raise ValueError("Number of performance weights must match number of datasets")
            total_weight = sum(performance_weights)
            self.weights = [w / total_weight for w in performance_weights]
        
        self.epoch_length = max(self.dataset_lengths)
        self.performance_history = [[] for _ in range(len(datasets))]

    def update_performance(self, dataset_idx: int, performance_score: float):
        self.performance_history[dataset_idx].append(performance_score)
        
        if len(self.performance_history[dataset_idx]) > 100:
            self.performance_history[dataset_idx] = self.performance_history[dataset_idx][-100:]
        
        self._adapt_weights()

    def _adapt_weights(self):
        avg_performances = []
        for history in self.performance_history:
            if history:
                avg_performances.append(sum(history[-10:]) / len(history[-10:]))
            else:
                avg_performances.append(0.5)
        
        if max(avg_performances) > 0:
            normalized_performances = [p / max(avg_performances) for p in avg_performances]
            
            for i in range(len(self.weights)):
                target_weight = max(normalized_performances[i], self.min_weight)
                self.weights[i] += self.adaptation_rate * (target_weight - self.weights[i])
            
            total_weight = sum(self.weights)
            self.weights = [w / total_weight for w in self.weights]

    def __len__(self):
        return self.epoch_length

    def __getitem__(self, idx):
        dataset_idx = random.choices(range(len(self.datasets)), weights=self.weights)[0]
        sample_idx = random.randint(0, len(self.datasets[dataset_idx]) - 1) # type: ignore
        return self.datasets[dataset_idx][sample_idx]


def combine_datasets(datasets: List[Dataset], 
                    method: str = 'concat',
                    weights: Optional[List[float]] = None,
                    strategy: str = 'weighted_random',
                    strata_sizes: Optional[List[int]] = None,
                    performance_weights: Optional[List[float]] = None,
                    adaptation_rate: float = 0.1,
                    min_weight: float = 0.05,
                    shuffle: bool = True,
                    seed: int = 42) -> Dataset:
    
    if not datasets:
        raise ValueError("At least one dataset must be provided")
    
    if len(datasets) == 1:
        return datasets[0]
    
    if method == 'concat':
        return ConcatDataset(datasets)
    
    elif method == 'weighted':
        return WeightedCombinedDataset(
            datasets=datasets,
            weights=weights,
            strategy=strategy,
            seed=seed
        )
    
    elif method == 'stratified':
        return StratifiedCombinedDataset(
            datasets=datasets,
            strata_sizes=strata_sizes,
            shuffle=shuffle,
            seed=seed
        )
    
    elif method == 'adaptive':
        return AdaptiveCombinedDataset(
            datasets=datasets,
            performance_weights=performance_weights,
            adaptation_rate=adaptation_rate,
            min_weight=min_weight
        )
    
    else:
        raise ValueError(f"Unknown combination method: {method}")


def create_multi_dataset_loader(dataset_configs: List[Dict[str, Any]], 
                               dataset_class,
                               combination_method: str = 'concat',
                               combination_params: Optional[Dict[str, Any]] = None) -> Dataset:
    
    datasets = []
    for config in dataset_configs:
        dataset = dataset_class(**config)
        datasets.append(dataset)
        logger.info(f"Loaded dataset with {len(dataset)} samples from {config.get('data_root', 'unknown')}")
    
    if combination_params is None:
        combination_params = {}
    
    combined_dataset = combine_datasets(
        datasets=datasets,
        method=combination_method,
        **combination_params
    )
    
    total_samples = len(combined_dataset) # type: ignore
    logger.info(f"Combined {len(datasets)} datasets into single dataset with {total_samples} samples using {combination_method} method")
    
    return combined_dataset


class DatasetMixer:
    def __init__(self, datasets: List[Dataset], mix_probability: float = 0.5, seed: int = 42):
        self.datasets = datasets
        self.mix_probability = mix_probability
        self.rng = random.Random(seed)
        
        if len(datasets) < 2:
            raise ValueError("At least 2 datasets required for mixing")

    def mix_samples(self, samples: List[Dict[str, torch.Tensor]], 
                   alpha: float = 0.5) -> Dict[str, torch.Tensor]:
        
        if len(samples) != 2:
            raise ValueError("Exactly 2 samples required for mixing")
        
        mixed_sample = {}
        for key in samples[0].keys():
            if key in ['image', 'depth']:
                mixed_sample[key] = alpha * samples[0][key] + (1 - alpha) * samples[1][key]
            else:
                mixed_sample[key] = samples[0][key]
        
        return mixed_sample

    def get_mixed_sample(self, dataset_indices: List[int], 
                        sample_indices: List[int]) -> Dict[str, torch.Tensor]:
        
        samples = []
        for dataset_idx, sample_idx in zip(dataset_indices, sample_indices):
            sample = self.datasets[dataset_idx][sample_idx]
            samples.append(sample)
        
        alpha = self.rng.uniform(0.2, 0.8)
        return self.mix_samples(samples, alpha)


def create_balanced_sampler_weights(datasets: List[Dataset]) -> List[float]:
    lengths = [len(dataset) for dataset in datasets] # type: ignore
    max_length = max(lengths)
    weights = [max_length / length for length in lengths]
    total_weight = sum(weights)
    return [w / total_weight for w in weights]


def validate_dataset_compatibility(datasets: List[Dataset]) -> bool:
    if not datasets:
        return True
    
    first_sample = datasets[0][0]
    reference_keys = set(first_sample.keys()) if isinstance(first_sample, dict) else None
    
    for i, dataset in enumerate(datasets[1:], 1):
        try:
            sample = dataset[0]
            if isinstance(sample, dict) and reference_keys:
                if set(sample.keys()) != reference_keys:
                    logger.warning(f"Dataset {i} has different keys: {set(sample.keys())} vs {reference_keys}")
                    return False
            elif not isinstance(sample, dict) and reference_keys:
                logger.warning(f"Dataset {i} returns different format than dataset 0")
                return False
        except Exception as e:
            logger.error(f"Failed to validate dataset {i}: {e}")
            return False
    
    return True


def get_dataset_statistics(datasets: List[Dataset]) -> Dict[str, Any]:
    stats = {
        'num_datasets': len(datasets),
        'total_samples': sum(len(dataset) for dataset in datasets), # type: ignore
        'individual_lengths': [len(dataset) for dataset in datasets], # type: ignore
        'min_length': min(len(dataset) for dataset in datasets) if datasets else 0, # type: ignore
        'max_length': max(len(dataset) for dataset in datasets) if datasets else 0, # type: ignore
        'avg_length': sum(len(dataset) for dataset in datasets) / len(datasets) if datasets else 0 # type: ignore
    }
    
    return stats