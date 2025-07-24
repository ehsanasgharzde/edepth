# FILE: monitoring/model_integration.py
# ehsanasgharzde - SYSTEM MONITOR

import torch
import torch.nn as nn
import time
import logging
from typing import Dict, Any, Optional, Callable
from contextlib import contextmanager
from functools import wraps

from model_monitor import ModelPerformanceMonitor
from system_monitor import SystemResourceMonitor
from configs.config import MonitoringConfig

logger = logging.getLogger(__name__)

class ModelMonitoringIntegration:
    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()

        # Initialize monitors
        self.model_monitor = ModelPerformanceMonitor(
            max_history=self.config.model_history_size
        )

        self.system_monitor = SystemResourceMonitor(
            update_interval=self.config.system_update_interval,
            max_history=self.config.system_history_size
        )

        # Set alert thresholds
        if self.config.enable_system_monitoring:
            self.system_monitor.set_thresholds(**self.config.get_alert_thresholds())

        self.is_monitoring = False

    def start_monitoring(self) -> None:
        if self.is_monitoring:
            logger.warning("Monitoring already started")
            return

        if self.config.enable_system_monitoring:
            self.system_monitor.start_monitoring()

        self.is_monitoring = True
        logger.info("Model monitoring integration started")

    def stop_monitoring(self) -> None:
        if not self.is_monitoring:
            return

        if self.config.enable_system_monitoring:
            self.system_monitor.stop_monitoring()

        self.is_monitoring = False
        logger.info("Model monitoring integration stopped")

    def register_model(self, model: nn.Module, name: str) -> None:
        """Register a model for monitoring"""
        if self.config.enable_model_monitoring:
            self.model_monitor.register_model(model, name)

    @contextmanager
    def monitor_inference(self, model_name: str, input_tensor: torch.Tensor):
        start_time = time.time()

        try:
            yield

        finally:
            if self.config.enable_model_monitoring:
                # This will be called after the model forward pass
                inference_time = time.time() - start_time

                # We need to get the output tensor from the calling context
                # This is a limitation of the context manager approach
                # Better to use the decorator approach below
                pass

    def track_inference_result(self, model_name: str, input_tensor: torch.Tensor, 
                             output_tensor: torch.Tensor, inference_time: float) -> None:
        if self.config.enable_model_monitoring:
            self.model_monitor.track_inference(
                model_name, input_tensor, output_tensor, inference_time
            )

    def track_training_step(self, epoch: int, step: int, loss: float, 
                          learning_rate: float, **kwargs) -> None:
        if self.config.enable_model_monitoring:

            # Calculate gradient norm if model is provided
            if 'model' in kwargs:
                model = kwargs.pop('model')
                grad_norm = self.calculate_gradient_norm(model)
                kwargs['gradient_norm'] = grad_norm

            self.model_monitor.track_training_step(
                epoch, step, loss, learning_rate, **kwargs
            )

    def calculate_gradient_norm(self, model: nn.Module) -> Optional[float]:
        if not self.config.gradient_monitoring:
            return None

        try:
            total_norm = 0.0
            param_count = 0

            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1

            if param_count > 0:
                return (total_norm ** 0.5)
            else:
                return None

        except Exception as e:
            logger.warning(f"Failed to calculate gradient norm: {e}")
            return None

    def get_monitoring_summary(self) -> Dict[str, Any]:
        summary = {
            'monitoring_active': self.is_monitoring,
            'config': self.config.to_dict()
        }

        if self.config.enable_model_monitoring:
            summary['model_summaries'] = {}
            for model_name in self.model_monitor.active_models.keys():
                summary['model_summaries'][model_name] = self.model_monitor.get_model_summary(model_name)

            summary['training_summary'] = self.model_monitor.get_training_summary()

        if self.config.enable_system_monitoring:
            summary['system_summary'] = self.system_monitor.get_summary_stats()
            summary['current_system_metrics'] = self.system_monitor.get_current_metrics()

        return summary

    def export_all_metrics(self, filepath: str, format: str = "json") -> bool:
        try:
            # Export model metrics
            if self.config.enable_model_monitoring:
                model_path = filepath.replace('.json', '_model.json').replace('.csv', '_model.csv')
                self.model_monitor.export_metrics(model_path, format)

            # Export system metrics  
            if self.config.enable_system_monitoring:
                system_path = filepath.replace('.json', '_system.json').replace('.csv', '_system.csv')
                self.system_monitor.export_metrics(system_path, format)

            logger.info(f"All metrics exported with base path: {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            return False

    def add_alert_callback(self, callback: Callable[[str], None]) -> None:
        if self.config.enable_system_monitoring:
            self.system_monitor.add_alert_callback(callback)

def monitor_model_inference(integration: ModelMonitoringIntegration, model_name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Assume first argument is input tensor
            input_tensor = args[0] if args else None

            start_time = time.time()
            result = func(*args, **kwargs)
            inference_time = time.time() - start_time

            if input_tensor is not None and result is not None:
                if isinstance(result, torch.Tensor):
                    output_tensor = result
                elif isinstance(result, (tuple, list)) and len(result) > 0:
                    output_tensor = result[0]  # Assume first element is the main output
                else:
                    output_tensor = torch.tensor([0.0])  # Fallback

                integration.track_inference_result(
                    model_name, input_tensor, output_tensor, inference_time
                )

            return result
        return wrapper
    return decorator