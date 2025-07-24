# FILE: config_fixed.py
# ehsanasgharzadeh - CONFIGURATION
# hosseinsolymanzadeh - PROPER COMMENTING
# ehsanasgharzde - FIXED REDUNDANT CODE BY EXTRACTING PURE FUNCTIONS AND BASECLASS LEVEL METHODS

from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from enum import Enum
import logging
from abc import ABC
import copy
import os

logger = logging.getLogger(__name__)

class ConfigError(Exception):
    pass

class ConfigValidationError(ConfigError):
    pass

class SchedulerType(Enum):
    COSINE = "cosine"
    STEP = "step"
    EXPONENTIAL = "exponential"
    PLATEAU = "plateau"
    LINEAR = "linear"
    POLYNOMIAL = "polynomial"

class OptimizerType(Enum):
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    RMSPROP = "rmsprop"
    ADAGRAD = "adagrad"

class DatasetType(Enum):
    NYU_DEPTH_V2 = "nyu_depth_v2"
    KITTI = "kitti"
    ENRICH = "enrich"
    UNREAL_STEREO4K = "unreal_stereo4k"

class BackboneType(Enum):
    VIT_SMALL_PATCH16_224 = "vit_small_patch16_224"
    VIT_BASE_PATCH16_224 = "vit_base_patch16_224"
    VIT_BASE_PATCH16_384 = "vit_base_patch16_384"
    VIT_BASE_PATCH8_224 = "vit_base_patch8_224"
    VIT_LARGE_PATCH16_224 = "vit_large_patch16_224"
    VIT_LARGE_PATCH16_384 = "vit_large_patch16_384"

class LossType(Enum):
    SILOG = "silog"
    BERHU = "berhu"
    RMSE = "rmse"
    MAE = "mae"
    MULTI_SCALE = "multi_scale"
    EDGE_AWARE_SMOOTHNESS = "edge_aware_smoothness"
    GRADIENT_CONSISTENCY = "gradient_consistency"
    MULTI_LOSS = "multi_loss"

class DeviceType(Enum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"

class PrecisionType(Enum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"

class BaseConfig(ABC):
    def validate(self) -> None:
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self) # type: ignore
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        return cls(**config_dict)

@dataclass
class TestConfig(BaseConfig):
    test_directory: str = "tests"
    output_directory: str = "test_results"
    parallel_workers: int = 4
    timeout: int = 300
    coverage_enabled: bool = True
    html_report: bool = True
    json_report: bool = True
    verbose: bool = True
    fail_fast: bool = False
    include_patterns: List[str] = field(default_factory=lambda: ["test_*.py", "*_test.py"])
    exclude_patterns: List[str] = field(default_factory=lambda: ["__pycache__", "*.pyc", "*.txt"])
    email_notifications: bool = False
    email_config: Dict[str, str] = field(default_factory=dict)
    memory_monitoring: bool = True
    performance_profiling: bool = False
    test_isolation: bool = True
    retry_failed: int = 0
    component_dirs: List[str] = field(default_factory=lambda: ["components"])

    def validate(self) -> None:
        if not os.path.isdir(self.test_directory):
            raise ConfigValidationError(f"test_directory does not exist: {self.test_directory}")

        if not os.path.isdir(self.output_directory):
            raise ConfigValidationError(f"output_directory does not exist: {self.output_directory}")

        if self.parallel_workers < 1:
            raise ConfigValidationError("parallel_workers must be at least 1")

        if self.timeout <= 0:
            raise ConfigValidationError("timeout must be a positive integer")

        if self.retry_failed < 0:
            raise ConfigValidationError("retry_failed must be zero or positive")

        if not isinstance(self.include_patterns, list) or not all(isinstance(p, str) for p in self.include_patterns):
            raise ConfigValidationError("include_patterns must be a list of strings")

        if not isinstance(self.exclude_patterns, list) or not all(isinstance(p, str) for p in self.exclude_patterns):
            raise ConfigValidationError("exclude_patterns must be a list of strings")

        if self.email_notifications:
            required_keys = {"smtp_server", "port", "sender", "recipient"}
            missing = required_keys - self.email_config.keys()
            if missing:
                raise ConfigValidationError(f"Missing email_config keys: {', '.join(missing)}")

        if not isinstance(self.component_dirs, list) or not all(isinstance(p, str) for p in self.component_dirs):
            raise ConfigValidationError("component_dirs must be a list of strings")

        if not isinstance(self.memory_monitoring, bool):
            raise ConfigValidationError("memory_monitoring must be a boolean")

        if not isinstance(self.performance_profiling, bool):
            raise ConfigValidationError("performance_profiling must be a boolean")

        if not isinstance(self.test_isolation, bool):
            raise ConfigValidationError("test_isolation must be a boolean")

        if not isinstance(self.coverage_enabled, bool):
            raise ConfigValidationError("coverage_enabled must be a boolean")

        if not isinstance(self.html_report, bool):
            raise ConfigValidationError("html_report must be a boolean")

        if not isinstance(self.json_report, bool):
            raise ConfigValidationError("json_report must be a boolean")

        if not isinstance(self.verbose, bool):
            raise ConfigValidationError("verbose must be a boolean")

        if not isinstance(self.fail_fast, bool):
            raise ConfigValidationError("fail_fast must be a boolean")

@dataclass
class ModelConfig(BaseConfig):
    backbone: str = "vit_base_patch16_224"
    extract_layers: List[int] = field(default_factory=lambda: [2, 5, 8, 11])
    decoder_channels: List[int] = field(default_factory=lambda: [256, 512, 1024, 1024])
    output_channels: int = 1
    pretrained: bool = True
    freeze_backbone: bool = False
    use_gradient_checkpointing: bool = False
    dropout_rate: float = 0.1
    attention_dropout: float = 0.1
    path_dropout: float = 0.1
    
    def validate(self) -> None:
        if self.backbone not in [b.value for b in BackboneType]:
            raise ConfigValidationError(f"Invalid backbone: {self.backbone}")
        
        if not self.extract_layers or len(self.extract_layers) != 4:
            raise ConfigValidationError("extract_layers must contain exactly 4 elements")
        
        if not self.decoder_channels or len(self.decoder_channels) != 4:
            raise ConfigValidationError("decoder_channels must contain exactly 4 elements")
        
        if self.output_channels <= 0:
            raise ConfigValidationError("output_channels must be positive")
        
        if not 0.0 <= self.dropout_rate <= 1.0:
            raise ConfigValidationError("dropout_rate must be between 0.0 and 1.0")
        
        if not 0.0 <= self.attention_dropout <= 1.0:
            raise ConfigValidationError("attention_dropout must be between 0.0 and 1.0")
        
        if not 0.0 <= self.path_dropout <= 1.0:
            raise ConfigValidationError("path_dropout must be between 0.0 and 1.0")

@dataclass
class LossConfig(BaseConfig):
    loss_type: str = "silog"
    loss_weights: Dict[str, float] = field(default_factory=lambda: {"silog": 1.0})
    silog_lambda: float = 0.85
    berhu_threshold: float = 0.2
    multi_scale_weights: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.25])
    edge_aware_weight: float = 0.1
    gradient_consistency_weight: float = 0.1
    
    def validate(self) -> None:
        if self.loss_type not in [l.value for l in LossType]:
            raise ConfigValidationError(f"Invalid loss_type: {self.loss_type}")
        
        if not 0.0 <= self.silog_lambda <= 1.0:
            raise ConfigValidationError("silog_lambda must be between 0.0 and 1.0")
        
        if self.berhu_threshold <= 0.0:
            raise ConfigValidationError("berhu_threshold must be positive")
        
        if not self.multi_scale_weights:
            raise ConfigValidationError("multi_scale_weights cannot be empty")
        
        if any(w <= 0 for w in self.multi_scale_weights):
            raise ConfigValidationError("All multi_scale_weights must be positive")

@dataclass
class OptimizerConfig(BaseConfig):
    optimizer_type: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    momentum: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    amsgrad: bool = False
    
    def validate(self) -> None:
        if self.optimizer_type not in [o.value for o in OptimizerType]:
            raise ConfigValidationError(f"Invalid optimizer_type: {self.optimizer_type}")
        
        if self.learning_rate <= 0:
            raise ConfigValidationError("learning_rate must be positive")
        
        if self.weight_decay < 0:
            raise ConfigValidationError("weight_decay must be non-negative")
        
        if not 0.0 <= self.momentum <= 1.0:
            raise ConfigValidationError("momentum must be between 0.0 and 1.0")
        
        if not 0.0 <= self.beta1 <= 1.0:
            raise ConfigValidationError("beta1 must be between 0.0 and 1.0")
        
        if not 0.0 <= self.beta2 <= 1.0:
            raise ConfigValidationError("beta2 must be between 0.0 and 1.0")
        
        if self.eps <= 0:
            raise ConfigValidationError("eps must be positive")

@dataclass
class SchedulerConfig(BaseConfig):
    scheduler_type: str = "cosine"
    step_size: int = 10
    gamma: float = 0.1
    T_max: int = 100
    eta_min: float = 1e-8
    patience: int = 10
    factor: float = 0.5
    threshold: float = 1e-4
    cooldown: int = 0
    min_lr: float = 1e-8
    warmup_epochs: int = 0
    warmup_start_lr: float = 1e-8
    
    def validate(self) -> None:
        if self.scheduler_type not in [s.value for s in SchedulerType]:
            raise ConfigValidationError(f"Invalid scheduler_type: {self.scheduler_type}")
        
        if self.step_size <= 0:
            raise ConfigValidationError("step_size must be positive")
        
        if not 0.0 < self.gamma <= 1.0:
            raise ConfigValidationError("gamma must be between 0.0 and 1.0")
        
        if self.T_max <= 0:
            raise ConfigValidationError("T_max must be positive")
        
        if self.eta_min < 0:
            raise ConfigValidationError("eta_min must be non-negative")
        
        if self.patience <= 0:
            raise ConfigValidationError("patience must be positive")
        
        if not 0.0 < self.factor <= 1.0:
            raise ConfigValidationError("factor must be between 0.0 and 1.0")
        
        if self.threshold <= 0:
            raise ConfigValidationError("threshold must be positive")
        
        if self.cooldown < 0:
            raise ConfigValidationError("cooldown must be non-negative")
        
        if self.min_lr < 0:
            raise ConfigValidationError("min_lr must be non-negative")
        
        if self.warmup_epochs < 0:
            raise ConfigValidationError("warmup_epochs must be non-negative")
        
        if self.warmup_start_lr < 0:
            raise ConfigValidationError("warmup_start_lr must be non-negative")

@dataclass
class TrainingConfig(BaseConfig):
    epochs: int = 100
    batch_size: int = 8
    accumulation_steps: int = 1
    val_batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    
    use_mixed_precision: bool = True
    precision_type: str = "float16"
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = False
    
    save_interval: int = 10
    eval_interval: int = 5
    log_interval: int = 100
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    resume_from: Optional[str] = None
    
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-4
    save_best_only: bool = True
    monitor_metric: str = "val_rmse"
    
    def validate(self) -> None:
        if self.epochs <= 0:
            raise ConfigValidationError("epochs must be positive")
        
        if self.batch_size <= 0:
            raise ConfigValidationError("batch_size must be positive")
        
        if self.accumulation_steps <= 0:
            raise ConfigValidationError("accumulation_steps must be positive")
        
        if self.val_batch_size <= 0:
            raise ConfigValidationError("val_batch_size must be positive")
        
        if self.num_workers < 0:
            raise ConfigValidationError("num_workers must be non-negative")
        
        if self.prefetch_factor <= 0:
            raise ConfigValidationError("prefetch_factor must be positive")
        
        if self.precision_type not in [p.value for p in PrecisionType]:
            raise ConfigValidationError(f"Invalid precision_type: {self.precision_type}")
        
        if self.max_grad_norm <= 0:
            raise ConfigValidationError("max_grad_norm must be positive")
        
        if self.save_interval <= 0:
            raise ConfigValidationError("save_interval must be positive")
        
        if self.eval_interval <= 0:
            raise ConfigValidationError("eval_interval must be positive")
        
        if self.log_interval <= 0:
            raise ConfigValidationError("log_interval must be positive")
        
        if self.early_stopping_patience <= 0:
            raise ConfigValidationError("early_stopping_patience must be positive")
        
        if self.early_stopping_min_delta < 0:
            raise ConfigValidationError("early_stopping_min_delta must be non-negative")
        
        self.optimizer.validate()
        self.scheduler.validate()
        self.loss.validate()

@dataclass
class AugmentationConfig(BaseConfig):
    horizontal_flip: bool = True
    horizontal_flip_prob: float = 0.5
    color_jitter: bool = True
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    color_jitter_saturation: float = 0.2
    color_jitter_hue: float = 0.1
    random_crop: bool = False
    random_crop_scale: Tuple[float, float] = (0.8, 1.0)
    normalize: bool = True
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    def validate(self) -> None:
        if not 0.0 <= self.horizontal_flip_prob <= 1.0:
            raise ConfigValidationError("horizontal_flip_prob must be between 0.0 and 1.0")
        
        if not 0.0 <= self.color_jitter_brightness <= 1.0:
            raise ConfigValidationError("color_jitter_brightness must be between 0.0 and 1.0")
        
        if not 0.0 <= self.color_jitter_contrast <= 1.0:
            raise ConfigValidationError("color_jitter_contrast must be between 0.0 and 1.0")
        
        if not 0.0 <= self.color_jitter_saturation <= 1.0:
            raise ConfigValidationError("color_jitter_saturation must be between 0.0 and 1.0")
        
        if not 0.0 <= self.color_jitter_hue <= 0.5:
            raise ConfigValidationError("color_jitter_hue must be between 0.0 and 0.5")
        
        if len(self.random_crop_scale) != 2 or self.random_crop_scale[0] > self.random_crop_scale[1]:
            raise ConfigValidationError("random_crop_scale must be a tuple of two values where first <= second")
        
        if len(self.mean) != 3 or len(self.std) != 3:
            raise ConfigValidationError("mean and std must be tuples of 3 values")

@dataclass
class DataConfig(BaseConfig):
    dataset_type: str = "nyu_depth_v2"
    data_root: str = "data"
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    
    img_size: Tuple[int, int] = (480, 640)
    depth_scale: float = 1000.0
    max_depth: float = 10.0
    min_depth: float = 0.1
    
    cache_dataset: bool = False
    validate_data: bool = True
    num_samples: Optional[int] = None
    
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    
    def validate(self) -> None:
        if self.dataset_type not in [d.value for d in DatasetType]:
            raise ConfigValidationError(f"Invalid dataset_type: {self.dataset_type}")
        
        if not Path(self.data_root).exists():
            logger.warning(f"Data root path does not exist: {self.data_root}")
        
        if len(self.img_size) != 2 or any(s <= 0 for s in self.img_size):
            raise ConfigValidationError("img_size must be a tuple of two positive integers")
        
        if self.depth_scale <= 0:
            raise ConfigValidationError("depth_scale must be positive")
        
        if self.max_depth <= self.min_depth:
            raise ConfigValidationError("max_depth must be greater than min_depth")
        
        if self.min_depth <= 0:
            raise ConfigValidationError("min_depth must be positive")
        
        if self.num_samples is not None and self.num_samples <= 0:
            raise ConfigValidationError("num_samples must be positive if specified")
        
        self.augmentation.validate()

@dataclass
class InferenceConfig(BaseConfig):
    batch_size: int = 1
    device: str = "auto"
    use_mixed_precision: bool = True
    precision_type: str = "float16"
    
    output_dir: str = "outputs"
    output_format: str = "png"
    save_colorized: bool = True
    save_raw: bool = True
    colormap: str = "plasma"
    
    resize_output: bool = False
    output_size: Optional[Tuple[int, int]] = None
    
    benchmark_mode: bool = False
    warmup_runs: int = 5
    timing_runs: int = 50
    
    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ConfigValidationError("batch_size must be positive")
        
        if self.device not in [d.value for d in DeviceType]:
            raise ConfigValidationError(f"Invalid device: {self.device}")
        
        if self.precision_type not in [p.value for p in PrecisionType]:
            raise ConfigValidationError(f"Invalid precision_type: {self.precision_type}")
        
        if self.output_format not in ["png", "jpg", "tiff", "exr", "npy"]:
            raise ConfigValidationError(f"Invalid output_format: {self.output_format}")
        
        if self.colormap not in ["plasma", "viridis", "jet", "gray", "magma"]:
            raise ConfigValidationError(f"Invalid colormap: {self.colormap}")
        
        if self.output_size is not None:
            if len(self.output_size) != 2 or any(s <= 0 for s in self.output_size):
                raise ConfigValidationError("output_size must be a tuple of two positive integers")
        
        if self.warmup_runs < 0:
            raise ConfigValidationError("warmup_runs must be non-negative")
        
        if self.timing_runs <= 0:
            raise ConfigValidationError("timing_runs must be positive")

@dataclass
class MonitoringConfig(BaseConfig):
    # Model monitoring
    enable_model_monitoring: bool = True
    model_history_size: int = 1000

    # System monitoring
    enable_system_monitoring: bool = True
    system_update_interval: float = 1.0
    system_history_size: int = 1000

    # Alert thresholds
    cpu_alert_threshold: float = 80.0
    memory_alert_threshold: float = 85.0
    gpu_utilization_threshold: float = 90.0
    gpu_temperature_threshold: float = 80.0
    disk_usage_threshold: float = 90.0

    # Training monitoring
    gradient_monitoring: bool = True
    activation_monitoring: bool = False

    # Streamlit dashboard
    dashboard_update_interval: int = 2  # seconds
    dashboard_port: int = 8501
    dashboard_host: str = "localhost"

    # Export settings
    auto_export: bool = False
    export_interval: int = 300  # seconds
    export_format: str = "json"
    export_directory: str = "monitoring_exports"

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = "monitoring.log"

    def get_alert_thresholds(self) -> Dict[str, float]:
        return {
            'cpu_percent': self.cpu_alert_threshold,
            'memory_percent': self.memory_alert_threshold,
            'gpu_utilization': self.gpu_utilization_threshold,
            'gpu_temperature': self.gpu_temperature_threshold,
            'disk_usage': self.disk_usage_threshold
        }

@dataclass
class ExperimentConfig(BaseConfig):
    name: str = "depth_estimation_experiment"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    
    seed: int = 42
    deterministic: bool = True
    benchmark: bool = True
    
    logging_level: str = "INFO"
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    tensorboard_log_dir: Optional[str] = None
    
    def validate(self) -> None:
        if not self.name:
            raise ConfigValidationError("experiment name cannot be empty")
        
        if self.seed < 0:
            raise ConfigValidationError("seed must be non-negative")
        
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.logging_level not in valid_levels:
            raise ConfigValidationError(f"logging_level must be one of {valid_levels}")

@dataclass
class Config(BaseConfig):
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    version: str = "1.0.0"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    
    def validate(self) -> None:
        self.model.validate()
        self.training.validate()
        self.data.validate()
        self.inference.validate()
        self.experiment.validate()
        
        if self.training.scheduler.T_max < self.training.epochs:
            logger.warning("scheduler T_max is less than training epochs")

class ConfigFactory:
    @staticmethod
    def create_nyu_config() -> Config:
        config = Config()
        config.data.dataset_type = DatasetType.NYU_DEPTH_V2.value
        config.data.img_size = (480, 640)
        config.data.depth_scale = 1000.0
        config.data.max_depth = 10.0
        config.model.backbone = BackboneType.VIT_BASE_PATCH16_224.value
        config.training.epochs = 50
        config.training.batch_size = 8
        config.experiment.name = "nyu_depth_v2_experiment"
        return config
    
    @staticmethod
    def create_kitti_config() -> Config:
        config = Config()
        config.data.dataset_type = DatasetType.KITTI.value
        config.data.img_size = (352, 1216)
        config.data.depth_scale = 256.0
        config.data.max_depth = 80.0
        config.model.backbone = BackboneType.VIT_BASE_PATCH16_384.value
        config.training.epochs = 100
        config.training.batch_size = 4
        config.experiment.name = "kitti_experiment"
        return config
    
    @staticmethod
    def create_debug_config() -> Config:
        config = Config()
        config.training.epochs = 2
        config.training.batch_size = 2
        config.training.val_batch_size = 2
        config.training.save_interval = 1
        config.training.eval_interval = 1
        config.training.log_interval = 1
        config.data.num_samples = 10
        config.experiment.name = "debug_experiment"
        config.experiment.logging_level = "DEBUG"
        return config
    
    @staticmethod
    def create_production_config() -> Config:
        config = Config()
        config.training.epochs = 200
        config.training.batch_size = 16
        config.training.use_mixed_precision = True
        config.training.gradient_checkpointing = True
        config.model.use_gradient_checkpointing = True
        config.data.cache_dataset = True
        config.experiment.name = "production_experiment"
        config.experiment.deterministic = False
        config.experiment.benchmark = True
        return config
    
    @staticmethod
    def create_inference_config() -> InferenceConfig:
        return InferenceConfig()

class ConfigPresets:
    SMALL_MODEL = {
        "backbone": BackboneType.VIT_SMALL_PATCH16_224.value,
        "extract_layers": [2, 5, 8, 11],
        "decoder_channels": [128, 256, 512, 512]
    }
    
    BASE_MODEL = {
        "backbone": BackboneType.VIT_BASE_PATCH16_224.value,
        "extract_layers": [2, 5, 8, 11], 
        "decoder_channels": [256, 512, 1024, 1024]
    }
    
    LARGE_MODEL = {
        "backbone": BackboneType.VIT_LARGE_PATCH16_224.value,
        "extract_layers": [4, 11, 17, 23],
        "decoder_channels": [512, 1024, 1024, 1024]
    }

def create_config_from_preset(preset_name: str, **overrides) -> Config:
    if preset_name == "nyu":
        config = ConfigFactory.create_nyu_config()
    elif preset_name == "kitti":
        config = ConfigFactory.create_kitti_config()
    elif preset_name == "debug":
        config = ConfigFactory.create_debug_config()
    elif preset_name == "production":
        config = ConfigFactory.create_production_config()
    else:
        raise ConfigValidationError(f"Unknown preset: {preset_name}")
    
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logger.warning(f"Unknown config attribute: {key}")
    
    config.validate()
    return config

def validate_config_compatibility(config: Config) -> List[str]:
    warnings = []
    
    if config.training.use_mixed_precision and config.training.precision_type == PrecisionType.FLOAT32.value:
        warnings.append("Mixed precision enabled but precision_type is float32")
    
    if config.model.use_gradient_checkpointing and not config.training.gradient_checkpointing:
        warnings.append("Model gradient checkpointing enabled but training gradient checkpointing disabled")
    
    if config.training.batch_size * config.training.accumulation_steps > 64:
        warnings.append("Effective batch size is very large, may cause memory issues")
    
    if config.data.img_size[0] * config.data.img_size[1] > 1024 * 1024:
        warnings.append("Image size is very large, may cause memory issues")
    
    return warnings

def merge_configs(base_config: Config, override_config: Config) -> Config:
    base_dict = base_config.to_dict()
    override_dict = override_config.to_dict()
    
    def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        result = copy.deepcopy(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    merged_dict = deep_merge(base_dict, override_dict)
    merged_config = Config.from_dict(merged_dict)
    merged_config.validate()
    
    return merged_config