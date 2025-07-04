from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class ModelConfig:
    name: str = 'fixed_dpt'
    backbone: str = 'vit_base_patch16_224'
    extract_layers: List[int] = field(default_factory=lambda: [3, 6, 9, 12])
    decoder_channels: List[int] = field(default_factory=lambda: [256, 512, 1024, 1024])
    patch_size: int = 16
    num_classes: int = 1
    pretrained: bool = True

@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    amp: bool = True
    grad_clip: float = 1.0
    scheduler: str = 'cosine'
    checkpoint_dir: str = 'checkpoints/'
    log_dir: str = 'logs/'
    seed: int = 42

@dataclass
class DataConfig:
    dataset: str = 'nyu_depth_v2'
    data_root: str = './data'
    img_size: Tuple[int, int] = (480, 640)
    num_workers: int = 4
    pin_memory: bool = True
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    min_depth: float = 0.1
    max_depth: float = 10.0
    depth_scale: float = 1000.0
    augmentation: bool = True

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig) 