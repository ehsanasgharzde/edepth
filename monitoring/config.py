#FILE: monitoring/config.py
# ehsanasgharzde - CONFIGURATION
# hosseinsolymanzadeh - PROPER COMMENTING

from dataclasses import dataclass, field
from typing import Dict, Any
from pathlib import Path
import os
import logging
import json
from datetime import datetime

@dataclass
class SystemThresholds:
    # Threshold for CPU usage percentage
    cpu_usage: float = 85.0
    # Threshold for memory usage percentage
    memory_usage: float = 90.0
    # Threshold for GPU usage percentage
    gpu_usage: float = 90.0
    # Threshold for GPU temperature in Celsius
    gpu_temperature: float = 80.0
    # Threshold for disk usage percentage
    disk_usage: float = 95.0
    # Threshold for network I/O bytes
    network_io_bytes: float = 1000000.0


@dataclass
class TrainingThresholds:
    # Allowed increase in loss before triggering alert
    loss_increase: float = 0.1
    # Maximum gradient norm allowed
    gradient_norm: float = 10.0
    # Threshold to detect vanishing gradients
    gradient_vanish: float = 1e-5
    # Threshold for standard deviation of weights
    weight_std_threshold: float = 1e-5
    # Number of steps to consider for convergence check
    convergence_window: int = 100
    # Threshold for plateau detection in training
    plateau_threshold: float = 1e-4


@dataclass
class DataThresholds:
    # Number of allowed NaN values tolerated
    nan_tolerance: int = 0
    # Number of allowed infinite values tolerated
    inf_tolerance: int = 0
    # Sigma threshold for detecting distribution drift
    distribution_drift_sigma: float = 3.0
    # Memory usage threshold in megabytes
    memory_usage_mb: float = 1000.0
    # Tensor size threshold in megabytes
    tensor_size_mb: float = 500.0


@dataclass
class LoggingConfig:
    # Logging level (e.g. DEBUG, INFO, WARNING)
    log_level: str = "INFO"
    # Directory path to store log files
    log_dir: str = "logs"
    # Flag for structured logging format
    structured_logging: bool = True
    # Maximum size of a log file in megabytes before rotation
    log_rotation_mb: int = 5
    # Number of days to retain old log files
    log_retention_days: int = 30
    # Enable or disable console logging output
    enable_console_logging: bool = True


@dataclass
class ProfilingConfig:
    # Enable or disable profiling
    enable_profiling: bool = False
    # Interval in steps between profiling snapshots
    profiling_interval: int = 100
    # Whether to profile memory usage
    profile_memory: bool = True
    # Whether to profile CUDA (GPU) operations
    profile_cuda: bool = True
    # Number of warmup steps before active profiling
    profile_warmup_steps: int = 10
    # Number of steps to actively profile
    profile_active_steps: int = 20


@dataclass
class MonitoringConfig:
    # Interval in seconds for system monitoring
    system_monitoring_interval: float = 1.0
    # Sampling rate for data monitoring
    data_monitoring_sample_rate: float = 1.0
    # Interval in steps to log training information
    training_log_interval: int = 10
    # Interval in steps to save training checkpoints
    training_checkpoint_interval: int = 100
    # Size of history buffer for monitoring data
    history_size: int = 1000
    
    # Directory to output monitoring results and data
    output_directory: str = "monitoring_output"
    
    # Flags to enable/disable monitoring features
    enable_gpu_monitoring: bool = True
    enable_gradient_monitoring: bool = True
    enable_activation_monitoring: bool = True
    enable_visualization: bool = True
    enable_alerts: bool = True
    
    # Nested threshold and config classes with default instances
    system_thresholds: SystemThresholds = field(default_factory=SystemThresholds)
    training_thresholds: TrainingThresholds = field(default_factory=TrainingThresholds)
    data_thresholds: DataThresholds = field(default_factory=DataThresholds)
    logging_config: LoggingConfig = field(default_factory=LoggingConfig)
    profiling_config: ProfilingConfig = field(default_factory=ProfilingConfig)
    
    def __post_init__(self):
        # Initialize directories, logging, and validate configs after object creation
        self._setup_directories()
        self._setup_logging()
        self._validate_configuration()
    
    def _setup_directories(self):
        # List of directories to ensure exist for monitoring outputs and logs
        directories = [
            self.output_directory,
            self.logging_config.log_dir,
            os.path.join(self.output_directory, "plots"),
            os.path.join(self.output_directory, "checkpoints"),
            os.path.join(self.output_directory, "profiles")
        ]
        # Create directories if they don't exist, including parents
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        # Get the logging level constant from string name
        log_level = getattr(logging, self.logging_config.log_level.upper())
        # Configure root logger with file and optional console handlers
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                # Log to a file in the log directory
                logging.FileHandler(
                    os.path.join(self.logging_config.log_dir, 'monitoring.log')
                ),
                # Conditionally add console output handler or null handler
                logging.StreamHandler() if self.logging_config.enable_console_logging else logging.NullHandler()
            ]
        )
        # Create a logger instance for this config class
        self.logger = logging.getLogger('MonitoringConfig')
        # Log that monitoring config initialization is complete
        self.logger.info("Monitoring configuration initialized")
    
    def _validate_configuration(self):
        # List of validation checks as (condition, error_message) tuples
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
        
        # Iterate over validations and raise error if any condition fails
        for condition, message in validations:
            if not condition:
                self.logger.error(f"Configuration validation failed: {message}")
                raise ValueError(message)
    
    def get_alert_thresholds(self) -> Dict[str, float]:
        # Return a dictionary of key alert thresholds from various configs
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
        # Update threshold values in specified category with provided key-value pairs
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
            # Warn if an unknown category is provided
            self.logger.warning(f"Unknown threshold category: {category}")
    
    def to_dict(self) -> Dict[str, Any]:
        # Serialize the full MonitoringConfig to a dictionary, including nested configs
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
        # Construct MonitoringConfig from a dictionary, unpacking nested config dicts
        system_thresholds = SystemThresholds(**config_dict.pop("system_thresholds", {}))
        training_thresholds = TrainingThresholds(**config_dict.pop("training_thresholds", {}))
        data_thresholds = DataThresholds(**config_dict.pop("data_thresholds", {}))
        logging_config = LoggingConfig(**config_dict.pop("logging_config", {}))
        profiling_config = ProfilingConfig(**config_dict.pop("profiling_config", {}))
        
        # Return a new instance with nested configs and remaining kwargs
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
        # Load configuration dictionary from a JSON file and create MonitoringConfig instance
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def save_to_json_file(self, file_path: str):
        # Save the current MonitoringConfig as a JSON file with pretty indentation
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        # Log that the config was successfully saved
        self.logger.info(f"Configuration saved to {file_path}")
    def get_component_config(self, component: str) -> Dict[str, Any]:
        # Dictionary mapping each component to its configuration dictionary
        component_configs = {
            "system_monitor": {
                # Interval between system monitoring checks
                "monitoring_interval": self.system_monitoring_interval,
                # Number of historical data points to keep
                "history_size": self.history_size,
                # Alert thresholds as a dictionary
                "alert_thresholds": self.system_thresholds.__dict__,
                # Flag to enable GPU monitoring
                "enable_gpu_monitoring": self.enable_gpu_monitoring
            },
            "data_monitor": {
                # Whether to track gradients during monitoring
                "track_gradients": self.enable_gradient_monitoring,
                # Whether to track activations during monitoring
                "track_activations": self.enable_activation_monitoring,
                # Sampling rate for data monitoring
                "sample_rate": self.data_monitoring_sample_rate,
                # Thresholds for data monitoring as a dictionary
                "thresholds": self.data_thresholds.__dict__
            },
            "training_monitor": {
                # Interval for logging training info
                "log_interval": self.training_log_interval,
                # Interval for saving checkpoints
                "checkpoint_interval": self.training_checkpoint_interval,
                # Thresholds for training monitoring as a dictionary
                "thresholds": self.training_thresholds.__dict__
            },
            "logger": {
                # Directory to save logs
                "log_dir": self.logging_config.log_dir,
                # Log level (converted from string to logging level)
                "log_level": getattr(logging, self.logging_config.log_level.upper()),
                # Flag for structured logging
                "structured_logging": self.logging_config.structured_logging,
                # Log rotation size in megabytes
                "rotation_mb": self.logging_config.log_rotation_mb
            },
            "profiler": {
                # Flag to enable profiling
                "enable_profiling": self.profiling_config.enable_profiling,
                # Interval between profiling sessions
                "profiling_interval": self.profiling_config.profiling_interval,
                # Whether to profile memory usage
                "profile_memory": self.profiling_config.profile_memory,
                # Whether to profile CUDA operations
                "profile_cuda": self.profiling_config.profile_cuda
            },
            "visualizer": {
                # Directory to save output visualizations (plots)
                "output_dir": os.path.join(self.output_directory, "plots"),
                # Flag for enabling interactive visualization
                "interactive": self.enable_visualization
            }
        }
        # Return the config dictionary for the requested component, or empty dict if not found
        return component_configs.get(component, {})
    
    def create_runtime_config(self) -> Dict[str, Any]:
        # Create a runtime configuration dictionary reflecting current state/settings
        return {
            # Current UTC timestamp in ISO format
            "timestamp": datetime.utcnow().isoformat(),
            # Flag indicating if monitoring is active
            "monitoring_active": True,
            # Dictionary indicating which components are enabled at runtime
            "components": {
                "system_monitor": self.enable_gpu_monitoring,  # system monitor enabled if GPU monitoring enabled
                "data_monitor": self.enable_gradient_monitoring or self.enable_activation_monitoring,  # enabled if either monitoring type active
                "training_monitor": True,  # always enabled
                "logger": True,  # always enabled
                "profiler": self.profiling_config.enable_profiling,  # enabled if profiling enabled
                "visualizer": self.enable_visualization,  # enabled if visualization enabled
                "alerts": self.enable_alerts  # enabled if alerts enabled
            },
            # Resource limits to enforce during runtime
            "resource_limits": {
                # Max memory usage in MB
                "max_memory_mb": self.data_thresholds.memory_usage_mb,
                # Max tensor size allowed in MB
                "max_tensor_size_mb": self.data_thresholds.tensor_size_mb,
                # Number of historical data points to retain
                "history_retention": self.history_size
            }
        }