#FILE: monitoring/config.py
# ehsanasgharzde - CONFIGURATION

from dataclasses import dataclass, field
from typing import Dict, Any
from pathlib import Path
import os
import logging
import json
from datetime import datetime

@dataclass
class SystemThresholds:
    cpu_usage: float = 85.0
    memory_usage: float = 90.0
    gpu_usage: float = 90.0
    gpu_temperature: float = 80.0
    disk_usage: float = 95.0
    network_io_bytes: float = 1000000.0

@dataclass
class TrainingThresholds:
    loss_increase: float = 0.1
    gradient_norm: float = 10.0
    gradient_vanish: float = 1e-5
    weight_std_threshold: float = 1e-5
    convergence_window: int = 100
    plateau_threshold: float = 1e-4

@dataclass
class DataThresholds:
    nan_tolerance: int = 0
    inf_tolerance: int = 0
    distribution_drift_sigma: float = 3.0
    memory_usage_mb: float = 1000.0
    tensor_size_mb: float = 500.0

@dataclass
class LoggingConfig:
    log_level: str = "INFO"
    log_dir: str = "logs"
    structured_logging: bool = True
    log_rotation_mb: int = 5
    log_retention_days: int = 30
    enable_console_logging: bool = True

@dataclass
class ProfilingConfig:
    enable_profiling: bool = False
    profiling_interval: int = 100
    profile_memory: bool = True
    profile_cuda: bool = True
    profile_warmup_steps: int = 10
    profile_active_steps: int = 20

@dataclass
class MonitoringConfig:
    system_monitoring_interval: float = 1.0
    data_monitoring_sample_rate: float = 1.0
    training_log_interval: int = 10
    training_checkpoint_interval: int = 100
    history_size: int = 1000
    
    output_directory: str = "monitoring_output"
    
    enable_gpu_monitoring: bool = True
    enable_gradient_monitoring: bool = True
    enable_activation_monitoring: bool = True
    enable_visualization: bool = True
    enable_alerts: bool = True
    
    system_thresholds: SystemThresholds = field(default_factory=SystemThresholds)
    training_thresholds: TrainingThresholds = field(default_factory=TrainingThresholds)
    data_thresholds: DataThresholds = field(default_factory=DataThresholds)
    logging_config: LoggingConfig = field(default_factory=LoggingConfig)
    profiling_config: ProfilingConfig = field(default_factory=ProfilingConfig)
    
    def __post_init__(self):
        self._setup_directories()
        self._setup_logging()
        self._validate_configuration()
    
    def _setup_directories(self):
        directories = [
            self.output_directory,
            self.logging_config.log_dir,
            os.path.join(self.output_directory, "plots"),
            os.path.join(self.output_directory, "checkpoints"),
            os.path.join(self.output_directory, "profiles")
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        log_level = getattr(logging, self.logging_config.log_level.upper())
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(
                    os.path.join(self.logging_config.log_dir, 'monitoring.log')
                ),
                logging.StreamHandler() if self.logging_config.enable_console_logging else logging.NullHandler()
            ]
        )
        self.logger = logging.getLogger('MonitoringConfig')
        self.logger.info("Monitoring configuration initialized")
    
    def _validate_configuration(self):
        validations = [
            (self.system_monitoring_interval > 0, "system_monitoring_interval must be positive"),
            (0 < self.data_monitoring_sample_rate <= 1, "data_monitoring_sample_rate must be between 0 and 1"),
            (self.training_log_interval > 0, "training_log_interval must be positive"),
            (self.training_checkpoint_interval > 0, "training_checkpoint_interval must be positive"),
            (self.history_size > 0, "history_size must be positive"),
            (self.system_thresholds.cpu_usage > 0, "cpu_usage threshold must be positive"),
            (self.system_thresholds.memory_usage > 0, "memory_usage threshold must be positive"),
            (self.training_thresholds.gradient_norm > 0, "gradient_norm threshold must be positive"),
            (self.data_thresholds.distribution_drift_sigma > 0, "distribution_drift_sigma must be positive")
        ]
        
        for condition, message in validations:
            if not condition:
                self.logger.error(f"Configuration validation failed: {message}")
                raise ValueError(message)
    
    def get_alert_thresholds(self) -> Dict[str, float]:
        return {
            "cpu_usage": self.system_thresholds.cpu_usage,
            "memory_usage": self.system_thresholds.memory_usage,
            "gpu_usage": self.system_thresholds.gpu_usage,
            "gpu_temperature": self.system_thresholds.gpu_temperature,
            "disk_usage": self.system_thresholds.disk_usage,
            "loss_increase": self.training_thresholds.loss_increase,
            "gradient_norm": self.training_thresholds.gradient_norm,
            "nan_tolerance": self.data_thresholds.nan_tolerance,
            "inf_tolerance": self.data_thresholds.inf_tolerance
        }
    
    def update_thresholds(self, category: str, updates: Dict[str, Any]):
        if category == "system":
            for key, value in updates.items():
                if hasattr(self.system_thresholds, key):
                    setattr(self.system_thresholds, key, value)
                    self.logger.info(f"Updated system threshold {key} to {value}")
        elif category == "training":
            for key, value in updates.items():
                if hasattr(self.training_thresholds, key):
                    setattr(self.training_thresholds, key, value)
                    self.logger.info(f"Updated training threshold {key} to {value}")
        elif category == "data":
            for key, value in updates.items():
                if hasattr(self.data_thresholds, key):
                    setattr(self.data_thresholds, key, value)
                    self.logger.info(f"Updated data threshold {key} to {value}")
        else:
            self.logger.warning(f"Unknown threshold category: {category}")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "system_monitoring_interval": self.system_monitoring_interval,
            "data_monitoring_sample_rate": self.data_monitoring_sample_rate,
            "training_log_interval": self.training_log_interval,
            "training_checkpoint_interval": self.training_checkpoint_interval,
            "history_size": self.history_size,
            "output_directory": self.output_directory,
            "enable_gpu_monitoring": self.enable_gpu_monitoring,
            "enable_gradient_monitoring": self.enable_gradient_monitoring,
            "enable_activation_monitoring": self.enable_activation_monitoring,
            "enable_visualization": self.enable_visualization,
            "enable_alerts": self.enable_alerts,
            "system_thresholds": self.system_thresholds.__dict__,
            "training_thresholds": self.training_thresholds.__dict__,
            "data_thresholds": self.data_thresholds.__dict__,
            "logging_config": self.logging_config.__dict__,
            "profiling_config": self.profiling_config.__dict__
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MonitoringConfig':
        system_thresholds = SystemThresholds(**config_dict.pop("system_thresholds", {}))
        training_thresholds = TrainingThresholds(**config_dict.pop("training_thresholds", {}))
        data_thresholds = DataThresholds(**config_dict.pop("data_thresholds", {}))
        logging_config = LoggingConfig(**config_dict.pop("logging_config", {}))
        profiling_config = ProfilingConfig(**config_dict.pop("profiling_config", {}))
        
        return cls(
            system_thresholds=system_thresholds,
            training_thresholds=training_thresholds,
            data_thresholds=data_thresholds,
            logging_config=logging_config,
            profiling_config=profiling_config,
            **config_dict
        )
    
    @classmethod
    def from_json_file(cls, file_path: str) -> 'MonitoringConfig':
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def save_to_json_file(self, file_path: str):
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        self.logger.info(f"Configuration saved to {file_path}")
    
    def get_component_config(self, component: str) -> Dict[str, Any]:
        component_configs = {
            "system_monitor": {
                "monitoring_interval": self.system_monitoring_interval,
                "history_size": self.history_size,
                "alert_thresholds": self.system_thresholds.__dict__,
                "enable_gpu_monitoring": self.enable_gpu_monitoring
            },
            "data_monitor": {
                "track_gradients": self.enable_gradient_monitoring,
                "track_activations": self.enable_activation_monitoring,
                "sample_rate": self.data_monitoring_sample_rate,
                "thresholds": self.data_thresholds.__dict__
            },
            "training_monitor": {
                "log_interval": self.training_log_interval,
                "checkpoint_interval": self.training_checkpoint_interval,
                "thresholds": self.training_thresholds.__dict__
            },
            "logger": {
                "log_dir": self.logging_config.log_dir,
                "log_level": getattr(logging, self.logging_config.log_level.upper()),
                "structured_logging": self.logging_config.structured_logging,
                "rotation_mb": self.logging_config.log_rotation_mb
            },
            "profiler": {
                "enable_profiling": self.profiling_config.enable_profiling,
                "profiling_interval": self.profiling_config.profiling_interval,
                "profile_memory": self.profiling_config.profile_memory,
                "profile_cuda": self.profiling_config.profile_cuda
            },
            "visualizer": {
                "output_dir": os.path.join(self.output_directory, "plots"),
                "interactive": self.enable_visualization
            }
        }
        return component_configs.get(component, {})
    
    def create_runtime_config(self) -> Dict[str, Any]:
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "monitoring_active": True,
            "components": {
                "system_monitor": self.enable_gpu_monitoring,
                "data_monitor": self.enable_gradient_monitoring or self.enable_activation_monitoring,
                "training_monitor": True,
                "logger": True,
                "profiler": self.profiling_config.enable_profiling,
                "visualizer": self.enable_visualization,
                "alerts": self.enable_alerts
            },
            "resource_limits": {
                "max_memory_mb": self.data_thresholds.memory_usage_mb,
                "max_tensor_size_mb": self.data_thresholds.tensor_size_mb,
                "history_retention": self.history_size
            }
        }