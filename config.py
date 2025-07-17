# FILE: config_fixed.py
# ehsanasgharzadeh - CONFIGURATION
# hosseinsolymanzadeh - PROPER COMMENTING

from dataclasses import dataclass, field, asdict, fields
from typing import List, Tuple, Optional, Dict, Any, Union
from pathlib import Path
import os
import logging
from omegaconf import OmegaConf
from abc import ABC, abstractmethod
from enum import Enum
import json
import yaml
import datetime
import copy

logger = logging.getLogger(__name__)

# Custom exception for general configuration errors
class ConfigError(Exception):
    pass


# Custom exception for specific configuration validation errors
class ConfigValidationError(ConfigError):
    pass


# Enum representing different types of learning rate schedulers
class SchedulerType(Enum):
    COSINE = "cosine"
    STEP = "step"
    EXPONENTIAL = "exponential"
    PLATEAU = "plateau"


# Enum representing supported dataset types
class DatasetType(Enum):
    NYU_DEPTH_V2 = "nyu_depth_v2"
    KITTI = "kitti"
    CITYSCAPES = "cityscapes"


# Enum representing supported backbone network architectures
class BackboneType(Enum):
    VIT_BASE_PATCH16_224 = "vit_base_patch16_224"
    VIT_LARGE_PATCH16_224 = "vit_large_patch16_224"
    RESNET50 = "resnet50"
    RESNET101 = "resnet101"


# Base class for configuration validators
class BaseValidator:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)  # Initialize a logger for the validator
        self.errors = []  # Store validation error messages

    # Helper method to log and store an error message
    def _add_error(self, message: str):
        self.errors.append(message)
        self.logger.error(message)

    # Validate that a value is a positive integer
    def _validate_positive_int(self, value: Any, name: str) -> bool:
        if not isinstance(value, int) or value <= 0:
            self._add_error(f"{name} must be a positive integer")
            return False
        return True

    # Validate that a value is a positive float or int
    def _validate_positive_float(self, value: Any, name: str) -> bool:
        if not isinstance(value, (int, float)) or value <= 0:
            self._add_error(f"{name} must be a positive number")
            return False
        return True

    # Validate that a number is within a specific range
    def _validate_range(self, value: Any, name: str, min_val: float, max_val: float) -> bool:
        if not isinstance(value, (int, float)) or not (min_val <= value <= max_val):
            self._add_error(f"{name} must be between {min_val} and {max_val}")
            return False
        return True

    # Validate that a value is one of the values defined in a given Enum
    def _validate_enum(self, value: Any, name: str, enum_class: type) -> bool:
        if value not in [e.value for e in enum_class]:  # type: ignore
            self._add_error(f"{name} must be one of {[e.value for e in enum_class]}")  # type: ignore
            return False
        return True

    # Validate that a string is not empty or only whitespace
    def _validate_non_empty_string(self, value: Any, name: str) -> bool:
        if not isinstance(value, str) or not value.strip():
            self._add_error(f"{name} must be a non-empty string")
            return False
        return True

    # Validate that a list contains only items of a specific type
    def _validate_list_type(self, value: Any, name: str, expected_type: type) -> bool:
        if not isinstance(value, list):
            self._add_error(f"{name} must be a list")
            return False
        if not all(isinstance(item, expected_type) for item in value):
            self._add_error(f"All items in {name} must be of type {expected_type.__name__}")
            return False
        return True

    # Validate that a tuple has a specific length and item types
    def _validate_tuple_type(self, value: Any, name: str, expected_length: int, expected_type: type) -> bool:
        if not isinstance(value, tuple) or len(value) != expected_length:
            self._add_error(f"{name} must be a tuple of length {expected_length}")
            return False
        if not all(isinstance(item, expected_type) for item in value):
            self._add_error(f"All items in {name} must be of type {expected_type.__name__}")
            return False
        return True

    # Return a copy of the current list of error messages
    def get_errors(self) -> List[str]:
        return self.errors.copy()

    # Clear all stored validation errors
    def clear_errors(self):
        self.errors.clear()


# Base configuration class that provides validation functionality via BaseValidator
@dataclass
class BaseConfig(ABC):
    def __init_subclass__(cls, **kwargs):
        # Automatically assigns a validator instance to subclasses
        super().__init_subclass__(**kwargs)
        cls._validator = BaseValidator()

    def __post_init__(self):
        # Clears previous validation errors before running validation
        self._validator.clear_errors()
        self.validate()
        # If validation errors exist, raise an exception with all error messages
        if self._validator.errors:
            raise ConfigValidationError(f"Validation failed: {'; '.join(self._validator.errors)}")

    @abstractmethod
    def validate(self):
        # Abstract method that must be implemented in subclasses to define validation logic
        pass

    def to_dict(self) -> Dict[str, Any]:
        # Converts dataclass fields into a dictionary
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        # Filters input dict to include only valid fields and returns an instance of the class
        init_keys = {f.name for f in fields(cls)}
        filtered_dict = {k: v for k, v in config_dict.items() if k in init_keys}
        return cls(**filtered_dict)


# Configuration for the model with validation rules defined in validate()
@dataclass
class ModelConfig(BaseConfig):
    name: str = 'fixed_dpt'  # Name of the model
    backbone: str = 'vit_base_patch16_224'  # Backbone model name
    extract_layers: List[int] = field(default_factory=lambda: [3, 6, 9, 12])  # Transformer layers to extract features from
    decoder_channels: List[int] = field(default_factory=lambda: [256, 512, 1024, 1024])  # Decoder layer channel sizes
    patch_size: int = 16  # Patch size for ViT
    num_classes: int = 1  # Number of output classes
    pretrained: bool = True  # Whether to use a pretrained backbone
    dropout_rate: float = 0.1  # Dropout rate for general layers
    attention_dropout: float = 0.1  # Dropout rate for attention layers
    hidden_dim: int = 768  # Hidden dimension of transformer
    num_heads: int = 12  # Number of attention heads
    num_layers: int = 12  # Number of transformer layers

    def validate(self):
        # Validate each field using corresponding methods from the validator
        self._validator._validate_non_empty_string(self.name, "name")
        self._validator._validate_enum(self.backbone, "backbone", BackboneType)
        self._validator._validate_list_type(self.extract_layers, "extract_layers", int)
        self._validator._validate_list_type(self.decoder_channels, "decoder_channels", int)

        # Ensure all layer indices are within valid range
        if self.extract_layers and not all(1 <= layer <= 12 for layer in self.extract_layers):
            self._validator._add_error("All extract_layers must be between 1 and 12")

        # Ensure decoder channels align with number of extracted layers
        if len(self.decoder_channels) != len(self.extract_layers):
            self._validator._add_error("decoder_channels length must match extract_layers length")

        # Patch size must be one of the predefined valid values
        if self.patch_size not in [8, 16, 32]:
            self._validator._add_error("patch_size must be 8, 16, or 32")

        # Basic numeric and range validations for remaining fields
        self._validator._validate_positive_int(self.num_classes, "num_classes")
        self._validator._validate_range(self.dropout_rate, "dropout_rate", 0.0, 1.0)
        self._validator._validate_range(self.attention_dropout, "attention_dropout", 0.0, 1.0)
        self._validator._validate_positive_int(self.hidden_dim, "hidden_dim")
        self._validator._validate_positive_int(self.num_heads, "num_heads")
        self._validator._validate_positive_int(self.num_layers, "num_layers")

        # Ensure hidden_dim is divisible by number of heads (important for attention mechanism)
        if self.hidden_dim % self.num_heads != 0:
            self._validator._add_error("hidden_dim must be divisible by num_heads")


# Configuration class for training settings
@dataclass
class TrainingConfig(BaseConfig):
    epochs: int = 100  # Number of training epochs
    batch_size: int = 8  # Number of samples per training batch
    learning_rate: float = 1e-4  # Initial learning rate
    weight_decay: float = 0.01  # L2 regularization factor
    amp: bool = True  # Use Automatic Mixed Precision (AMP) if True
    grad_clip: float = 1.0  # Maximum norm for gradient clipping
    scheduler: str = 'cosine'  # Learning rate scheduler type
    checkpoint_dir: str = 'checkpoints/'  # Directory to save model checkpoints
    log_dir: str = 'logs/'  # Directory for training logs
    seed: int = 42  # Random seed for reproducibility
    warmup_epochs: int = 5  # Number of warm-up epochs before applying full learning rate
    min_lr: float = 1e-6  # Minimum learning rate for scheduler
    patience: int = 10  # Patience for early stopping
    save_frequency: int = 10  # Save model every N epochs
    validation_frequency: int = 1  # Run validation every N epochs
    early_stopping: bool = True  # Enable early stopping if True
    gradient_accumulation_steps: int = 1  # Number of steps to accumulate gradients before update
    optimizer: str = 'adamw'  # Optimizer type
    betas: Tuple[float, float] = (0.9, 0.999)  # Beta coefficients for Adam-like optimizers
    eps: float = 1e-8  # Epsilon value for numerical stability in optimizer

    def validate(self):
        # Validate all training configuration parameters
        self._validator._validate_positive_int(self.epochs, "epochs")
        self._validator._validate_positive_int(self.batch_size, "batch_size")
        self._validator._validate_positive_float(self.learning_rate, "learning_rate")
        self._validator._validate_range(self.weight_decay, "weight_decay", 0.0, float('inf'))
        self._validator._validate_positive_float(self.grad_clip, "grad_clip")
        self._validator._validate_enum(self.scheduler, "scheduler", SchedulerType)
        self._validator._validate_non_empty_string(self.checkpoint_dir, "checkpoint_dir")
        self._validator._validate_non_empty_string(self.log_dir, "log_dir")

        # Seed and warm-up must be non-negative
        if self.seed < 0:
            self._validator._add_error("seed must be non-negative")

        if self.warmup_epochs < 0:
            self._validator._add_error("warmup_epochs must be non-negative")

        if self.warmup_epochs >= self.epochs:
            self._validator._add_error("warmup_epochs must be less than epochs")

        self._validator._validate_positive_float(self.min_lr, "min_lr")

        # Ensure min_lr is less than base learning rate
        if self.min_lr >= self.learning_rate:
            self._validator._add_error("min_lr must be less than learning_rate")

        self._validator._validate_positive_int(self.patience, "patience")
        self._validator._validate_positive_int(self.save_frequency, "save_frequency")
        self._validator._validate_positive_int(self.validation_frequency, "validation_frequency")
        self._validator._validate_positive_int(self.gradient_accumulation_steps, "gradient_accumulation_steps")

        # Optimizer must be a known supported type
        if self.optimizer.lower() not in ['adam', 'adamw', 'sgd', 'rmsprop']:
            self._validator._add_error("optimizer must be one of ['adam', 'adamw', 'sgd', 'rmsprop']")

        self._validator._validate_tuple_type(self.betas, "betas", 2, float)

        # Betas must lie within (0, 1) for optimizer stability
        if self.betas and not (0.0 < self.betas[0] < 1.0 and 0.0 < self.betas[1] < 1.0):
            self._validator._add_error("betas values must be between 0.0 and 1.0")

        self._validator._validate_positive_float(self.eps, "eps")


# Configuration class for data loading and preprocessing
@dataclass
class DataConfig(BaseConfig):
    dataset: str = 'nyu_depth_v2'  # Dataset identifier
    data_root: str = './data'  # Root directory for data
    img_size: Tuple[int, int] = (480, 640)  # Target input image size (H, W)
    num_workers: int = 4  # Number of subprocesses for data loading
    pin_memory: bool = True  # Pin memory during data loading
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)  # Mean for image normalization
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)  # Std for image normalization
    min_depth: float = 0.1  # Minimum valid depth value
    max_depth: float = 10.0  # Maximum valid depth value
    depth_scale: float = 1000.0  # Scale factor to convert depth units
    augmentation: bool = True  # Enable/disable data augmentation
    train_split: float = 0.8  # Fraction of data used for training
    val_split: float = 0.1  # Fraction of data used for validation
    test_split: float = 0.1  # Fraction of data used for testing
    shuffle: bool = True  # Shuffle dataset before splitting
    drop_last: bool = True  # Drop last incomplete batch
    horizontal_flip: bool = True  # Apply horizontal flip during augmentation
    vertical_flip: bool = False  # Apply vertical flip during augmentation
    rotation_range: float = 10.0  # Max rotation angle in degrees
    brightness_range: float = 0.2  # Range for brightness variation
    contrast_range: float = 0.2  # Range for contrast variation
    saturation_range: float = 0.2  # Range for saturation variation
    hue_range: float = 0.1  # Range for hue variation

    def validate(self):
        # Validate dataset-related parameters
        self._validator._validate_enum(self.dataset, "dataset", DatasetType)
        self._validator._validate_non_empty_string(self.data_root, "data_root")
        self._validator._validate_tuple_type(self.img_size, "img_size", 2, int)

        # Ensure image dimensions are positive and within a reasonable range
        if self.img_size and not all(0 < size <= 4096 for size in self.img_size):
            self._validator._add_error("img_size values must be positive and reasonable")

        if self.num_workers < 0:
            self._validator._add_error("num_workers must be non-negative")

        self._validator._validate_tuple_type(self.normalize_mean, "normalize_mean", 3, float)
        self._validator._validate_tuple_type(self.normalize_std, "normalize_std", 3, float)

        # Check normalization values are within valid bounds
        if self.normalize_mean and not all(0.0 <= val <= 1.0 for val in self.normalize_mean):
            self._validator._add_error("normalize_mean values must be between 0.0 and 1.0")

        if self.normalize_std and not all(val > 0.0 for val in self.normalize_std):
            self._validator._add_error("normalize_std values must be positive")

        self._validator._validate_positive_float(self.min_depth, "min_depth")
        self._validator._validate_positive_float(self.max_depth, "max_depth")

        # Ensure max depth is greater than min depth
        if self.max_depth <= self.min_depth:
            self._validator._add_error("max_depth must be greater than min_depth")

        self._validator._validate_positive_float(self.depth_scale, "depth_scale")
        self._validator._validate_range(self.train_split, "train_split", 0.0, 1.0)
        self._validator._validate_range(self.val_split, "val_split", 0.0, 1.0)
        self._validator._validate_range(self.test_split, "test_split", 0.0, 1.0)

        # Ensure that the sum of dataset splits equals 1.0
        if abs(self.train_split + self.val_split + self.test_split - 1.0) > 1e-6:
            self._validator._add_error("train_split + val_split + test_split must equal 1.0")

        # Validate ranges for augmentation parameters
        self._validator._validate_range(self.rotation_range, "rotation_range", 0.0, 180.0)
        self._validator._validate_range(self.brightness_range, "brightness_range", 0.0, 1.0)
        self._validator._validate_range(self.contrast_range, "contrast_range", 0.0, 1.0)
        self._validator._validate_range(self.saturation_range, "saturation_range", 0.0, 1.0)
        self._validator._validate_range(self.hue_range, "hue_range", 0.0, 1.0)

        # Warn if augmentation is enabled but all augmentation values are effectively zero or disabled
        if self.augmentation:
            if (self.rotation_range == 0 and self.brightness_range == 0 and 
                self.contrast_range == 0 and self.saturation_range == 0 and 
                self.hue_range == 0 and not self.horizontal_flip and not self.vertical_flip):
                self._validator._add_error("augmentation enabled but no augmentation parameters set")


# Combined configuration class that encapsulates model, training, and data settings
@dataclass
class Config(BaseConfig):
    model: ModelConfig = field(default_factory=ModelConfig)  # Model-specific configuration
    training: TrainingConfig = field(default_factory=TrainingConfig)  # Training-specific configuration
    data: DataConfig = field(default_factory=DataConfig)  # Data-specific configuration
    version: str = "1.0.0"  # Configuration version
    name: str = "edepth_config"  # Name identifier for this config
    description: str = "Configuration for EDepth monocular depth estimation"  # Short description
    created_at: Optional[str] = None  # ISO timestamp for creation time
    modified_at: Optional[str] = None  # ISO timestamp for last modification

    def validate(self):
        # Validate that sub-configs are of correct type
        if not isinstance(self.model, ModelConfig):
            self._validator._add_error("model must be ModelConfig instance")

        if not isinstance(self.training, TrainingConfig):
            self._validator._add_error("training must be TrainingConfig instance")

        if not isinstance(self.data, DataConfig):
            self._validator._add_error("data must be DataConfig instance")

        self._validator._validate_non_empty_string(self.version, "version")
        self._validator._validate_non_empty_string(self.name, "name")

        # Warn (not error) if batch size and image size are both large — may cause memory issues
        if (self.training.batch_size > 32 and 
            self.data.img_size[0] * self.data.img_size[1] > 640 * 480):
            logger.warning("Large batch size with large image size may cause memory issues")

    def update_timestamps(self):
        # Set modified_at to current time, set created_at if not already set
        now = datetime.datetime.now().isoformat()
        self.modified_at = now
        if self.created_at is None:
            self.created_at = now

    def get_summary(self) -> Dict[str, Any]:
        # Return a summarized dictionary of key configuration settings
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "model_backbone": self.model.backbone,
            "dataset": self.data.dataset,
            "epochs": self.training.epochs,
            "batch_size": self.training.batch_size,
            "learning_rate": self.training.learning_rate,
            "image_size": self.data.img_size,
            "augmentation_enabled": self.data.augmentation,
            "created_at": self.created_at,
            "modified_at": self.modified_at
        }

# Factory class for creating pre-defined config presets
class ConfigFactory:
    @staticmethod
    def create_nyu_config() -> Config:
        # Creates a config preset for the NYU Depth V2 dataset
        config = Config(
            data=DataConfig(
                dataset='nyu_depth_v2',
                data_root='./data/nyu',
                img_size=(480, 640),
                min_depth=0.1,
                max_depth=10.0,
                depth_scale=1000.0
            ),
            model=ModelConfig(
                backbone='vit_base_patch16_224',
                pretrained=True,
                dropout_rate=0.1
            ),
            training=TrainingConfig(
                epochs=100,
                batch_size=8,
                learning_rate=1e-4,
                scheduler='cosine'
            )
        )
        config.update_timestamps()
        logger.info("Created NYU configuration")
        return config

    @staticmethod
    def create_kitti_config() -> Config:
        # Creates a config preset for the KITTI dataset
        config = Config(
            data=DataConfig(
                dataset='kitti',
                data_root='./data/kitti',
                img_size=(352, 1216),
                min_depth=0.1,
                max_depth=80.0,
                depth_scale=256.0
            ),
            model=ModelConfig(
                backbone='resnet50',
                pretrained=True,
                dropout_rate=0.2
            ),
            training=TrainingConfig(
                epochs=150,
                batch_size=4,
                learning_rate=1e-4,
                scheduler='step'
            )
        )
        config.update_timestamps()
        logger.info("Created KITTI configuration")
        return config

    @staticmethod
    def create_debug_config() -> Config:
        # Creates a lightweight config for debugging and fast tests
        config = Config(
            data=DataConfig(
                dataset='nyu_depth_v2',
                data_root='./data/debug',
                img_size=(64, 64),
                num_workers=0,
                augmentation=False
            ),
            model=ModelConfig(
                backbone='resnet50',
                pretrained=False,
                dropout_rate=0.0
            ),
            training=TrainingConfig(
                epochs=1,
                batch_size=2,
                learning_rate=0.01,
                amp=False
            )
        )
        config.update_timestamps()
        logger.info("Created debug configuration")
        return config

    @staticmethod
    def create_production_config() -> Config:
        # Creates a high-capacity config for production deployments
        config = Config(
            data=DataConfig(
                dataset='nyu_depth_v2',
                data_root='/mnt/production_data/nyu',
                img_size=(480, 640),
                num_workers=8,
                pin_memory=True
            ),
            model=ModelConfig(
                backbone='vit_large_patch16_224',
                pretrained=True,
                dropout_rate=0.1
            ),
            training=TrainingConfig(
                epochs=200,
                batch_size=16,
                learning_rate=1e-4,
                scheduler='cosine',
                amp=True,
                weight_decay=1e-4
            )
        )
        config.update_timestamps()
        logger.info("Created production configuration")
        return config

# Class for managing loading, saving, merging, and validating configurations
class ConfigManager:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_config_from_file(self, config_path: str) -> Config:
        # Load configuration from a YAML or JSON file
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        try:
            if path.suffix.lower() in ['.yaml', '.yml']:
                with open(path, 'r') as f:
                    config_dict = yaml.safe_load(f)
            elif path.suffix.lower() == '.json':
                with open(path, 'r') as f:
                    config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {path.suffix}")

            config = Config.from_dict(config_dict)
            self.logger.info(f"Loaded config from {config_path}")
            return config
        except Exception as e:
            self.logger.error(f"Failed to load config from {config_path}: {e}")
            raise

    def save_config_to_file(self, config: Config, config_path: str):
        # Save configuration to YAML or JSON file
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            config_dict = config.to_dict()

            if path.suffix.lower() in ['.yaml', '.yml']:
                with open(path, 'w') as f:
                    yaml.dump(config_dict, f, default_flow_style=False)
            elif path.suffix.lower() == '.json':
                with open(path, 'w') as f:
                    json.dump(config_dict, f, indent=4)
            else:
                raise ValueError(f"Unsupported config file format: {path.suffix}")

            self.logger.info(f"Saved config to {config_path}")
        except Exception as e:
            self.logger.error(f"Failed to save config to {config_path}: {e}")
            raise

    def merge_configs(self, base_config: Config, override_config: Config) -> Config:
        # Merge two configurations (override overrides base)
        base_dict = base_config.to_dict()
        override_dict = override_config.to_dict()

        merged_dict = self._deep_merge(base_dict, override_dict)
        merged_config = Config.from_dict(merged_dict)
        merged_config.update_timestamps()

        self.logger.info("Merged configurations")
        return merged_config

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        # Recursively merge dictionaries
        merged = base.copy()
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    def validate_config_compatibility(self, config: Config) -> bool:
        # Validates full configuration and logs outcome
        try:
            config.validate()
            self.logger.info("Config validation passed")
            return True
        except ConfigValidationError as e:
            self.logger.error(f"Config validation failed: {e}")
            return False

# Utility function for global logging setup
def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s'
    )

# Initialize a global instance of ConfigManager
config_manager = ConfigManager()
