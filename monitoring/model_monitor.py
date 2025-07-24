# FILE: monitoring/model_monitor.py
# ehsanasgharzde - SYSTEM MONITOR

import torch
import torch.nn as nn
import json
from datetime import datetime
from typing import Dict, Any, Optional
from collections import defaultdict, deque
from dataclasses import dataclass, field
import logging
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ModelMetrics:
    timestamp: datetime = field(default_factory=datetime.utcnow)
    model_name: str = ""
    inference_time: float = 0.0
    memory_usage: float = 0.0
    input_shape: tuple = ()
    output_shape: tuple = ()
    parameters_count: int = 0
    flops: Optional[float] = None
    throughput: float = 0.0  # samples per second
    batch_size: int = 1

@dataclass
class TrainingMetrics:
    timestamp: datetime = field(default_factory=datetime.utcnow)
    epoch: int = 0
    step: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    gradient_norm: Optional[float] = None
    accuracy: Optional[float] = None
    validation_loss: Optional[float] = None
    batch_time: float = 0.0
    data_time: float = 0.0

class ModelPerformanceMonitor:
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.model_metrics = defaultdict(lambda: deque(maxlen=max_history))
        self.training_metrics = deque(maxlen=max_history)
        self.active_models = {}
        self.hooks = {}
        self.start_time = None
        self.is_monitoring = False

    def register_model(self, model: nn.Module, name: str) -> None:
        self.active_models[name] = {
            'model': model,
            'parameters': sum(p.numel() for p in model.parameters()),
            'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'device': next(model.parameters()).device if list(model.parameters()) else 'cpu'
        }
        logger.info(f"Model '{name}' registered for monitoring")

    def track_inference(self, model_name: str, input_tensor: torch.Tensor, 
                       output_tensor: torch.Tensor, inference_time: float) -> None:
        if model_name not in self.active_models:
            logger.warning(f"Model '{model_name}' not registered")
            return

        # Calculate memory usage
        memory_usage = 0.0
        if torch.cuda.is_available():
            memory_usage = torch.cuda.memory_allocated() / 1024**2  # MB

        # Calculate throughput
        batch_size = input_tensor.shape[0] if len(input_tensor.shape) > 0 else 1
        throughput = batch_size / inference_time if inference_time > 0 else 0.0

        metrics = ModelMetrics(
            model_name=model_name,
            inference_time=inference_time,
            memory_usage=memory_usage,
            input_shape=tuple(input_tensor.shape),
            output_shape=tuple(output_tensor.shape),
            parameters_count=self.active_models[model_name]['parameters'],
            throughput=throughput,
            batch_size=batch_size
        )

        self.model_metrics[model_name].append(metrics)

    def track_training_step(self, epoch: int, step: int, loss: float, 
                           learning_rate: float, **kwargs) -> None:
        metrics = TrainingMetrics(
            epoch=epoch,
            step=step,
            loss=loss,
            learning_rate=learning_rate,
            **{k: v for k, v in kwargs.items() if hasattr(TrainingMetrics, k)}
        )

        self.training_metrics.append(metrics)

    def get_model_summary(self, model_name: str) -> Dict[str, Any]:
        if model_name not in self.model_metrics:
            return {}

        metrics = list(self.model_metrics[model_name])
        if not metrics:
            return {}

        recent_metrics = metrics[-10:]  # Last 10 measurements

        return {
            'model_name': model_name,
            'total_measurements': len(metrics),
            'avg_inference_time': np.mean([m.inference_time for m in recent_metrics]),
            'min_inference_time': min(m.inference_time for m in recent_metrics),
            'max_inference_time': max(m.inference_time for m in recent_metrics),
            'avg_memory_usage': np.mean([m.memory_usage for m in recent_metrics]),
            'avg_throughput': np.mean([m.throughput for m in recent_metrics]),
            'parameters_count': recent_metrics[0].parameters_count,
            'last_updated': recent_metrics[-1].timestamp.isoformat()
        }

    def get_training_summary(self) -> Dict[str, Any]:
        if not self.training_metrics:
            return {}

        metrics = list(self.training_metrics)
        recent_metrics = metrics[-10:]

        return {
            'total_steps': len(metrics),
            'current_epoch': metrics[-1].epoch,
            'current_step': metrics[-1].step,
            'current_loss': metrics[-1].loss,
            'avg_loss': np.mean([m.loss for m in recent_metrics]),
            'min_loss': min(m.loss for m in metrics),
            'current_lr': metrics[-1].learning_rate,
            'gradient_norm': metrics[-1].gradient_norm,
            'last_updated': metrics[-1].timestamp.isoformat()
        }

    def get_all_metrics(self) -> Dict[str, Any]:
        return {
            'model_metrics': {
                name: [
                    {
                        'timestamp': m.timestamp.isoformat(),
                        'inference_time': m.inference_time,
                        'memory_usage': m.memory_usage,
                        'throughput': m.throughput,
                        'batch_size': m.batch_size
                    } for m in list(metrics)
                ] for name, metrics in self.model_metrics.items()
            },
            'training_metrics': [
                {
                    'timestamp': m.timestamp.isoformat(),
                    'epoch': m.epoch,
                    'step': m.step,
                    'loss': m.loss,
                    'learning_rate': m.learning_rate,
                    'gradient_norm': m.gradient_norm
                } for m in list(self.training_metrics)
            ]
        }

    def export_metrics(self, filepath: str, format: str = "json") -> bool:
        try:
            metrics = self.get_all_metrics()

            if format == "json":
                with open(filepath, 'w') as f:
                    json.dump(metrics, f, indent=2, default=str)
            elif format == "csv":
                import pandas as pd

                # Export training metrics
                if metrics['training_metrics']:
                    df_training = pd.DataFrame(metrics['training_metrics'])
                    training_path = filepath.replace('.csv', '_training.csv')
                    df_training.to_csv(training_path, index=False)

                # Export model metrics
                for model_name, model_data in metrics['model_metrics'].items():
                    if model_data:
                        df_model = pd.DataFrame(model_data)
                        model_path = filepath.replace('.csv', f'_{model_name}.csv')
                        df_model.to_csv(model_path, index=False)

            logger.info(f"Metrics exported to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            return False

    def clear_metrics(self) -> None:
        self.model_metrics.clear()
        self.training_metrics.clear()
        logger.info("All metrics cleared")
