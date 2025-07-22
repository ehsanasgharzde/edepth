#FILE: monitoring/training_monitor.py
# ehsanasgharzde - TRAINING MONITOR
# hosseinsolymanzadeh - PROPER COMMENTING

import torch
import numpy as np
from typing import Dict, List, Any, Optional
from collections import deque
import logging
import hashlib
import json
from datetime import datetime, timedelta
import time
from dataclasses import dataclass, field

@dataclass
class TrainingMetrics:
    # Holds training metrics for each step
    step: int
    loss: float
    learning_rate: float
    metrics: Dict[str, float]
    # Timestamp defaults to current UTC time
    timestamp: datetime = field(default_factory=datetime.utcnow)
    convergence_delta: Optional[float] = None
    weight_stats: Optional[Dict[str, Any]] = None


class TrainingMonitor:
    
    def __init__(self, log_interval: int = 10, checkpoint_interval: int = 100, 
                 logger: Optional[logging.Logger] = None, config: Optional[Any] = None):
        # Interval for logging training info
        self.log_interval = log_interval
        # Interval for saving checkpoints
        self.checkpoint_interval = checkpoint_interval
        # Logger instance, fallback to module logger if none provided
        self.logger = logger or logging.getLogger(__name__)
        # Optional external config
        self.config = config
        
        # List to store historical training metrics
        self.training_history: List[TrainingMetrics] = []
        # Buffer to track recent convergence deltas (max length 1000)
        self.convergence_buffer = deque(maxlen=1000)
        # Track batch processing times (max length 100)
        self.batch_processing_times = deque(maxlen=100)
        # Track data loading times (max length 100)
        self.data_loading_times = deque(maxlen=100)
        
        # Registry to track weights (e.g. for monitoring)
        self.weight_tracking_registry = {}
        # List to hold gradient hook references
        self.gradient_hooks = []
        # Dictionary of alert thresholds initialized from config or defaults
        self.alert_thresholds = self._initialize_alert_thresholds()
        
        # Current training step counter
        self.current_step = 0
        # Timestamp for when training started
        self.training_start_time = None
        # Timestamp for last checkpoint saved
        self.last_checkpoint_time = None

    def _initialize_alert_thresholds(self) -> Dict[str, float]:
        # Initialize alert thresholds either from config or set defaults
        if self.config and hasattr(self.config, 'alert_thresholds'):
            return self.config.alert_thresholds
        return {
            # Allowed increase in loss before alert
            "loss_increase": 0.1,
            # Maximum gradient norm allowed before alert
            "gradient_norm": 10.0,
            # Threshold for detecting stagnation
            "stagnation_threshold": 1e-6,
            # Factor to detect divergence
            "divergence_factor": 2.0
        }

    def _generate_tensor_id(self, tensor: torch.Tensor) -> str:
        # Create a unique ID for tensor based on shape, dtype, device
        tensor_info = f"{tensor.shape}_{tensor.dtype}_{tensor.device}"
        # MD5 hash and shorten to first 8 chars
        return hashlib.md5(tensor_info.encode()).hexdigest()[:8]

    def start_training_session(self) -> None:
        # Mark start time of training session
        self.training_start_time = time.time()
        # Reset step count to zero
        self.current_step = 0
        # Log info about session start
        self.logger.info("Training session started")

    def log_training_step(self, step: int, loss: float, metrics: Dict[str, float], 
                         lr: float, model: Optional[torch.nn.Module] = None) -> None:
        # Update the current step number
        self.current_step = step
        # Calculate the change in loss compared to the previous step
        convergence_delta = self._calculate_convergence_delta(loss)
        
        weight_stats = None
        # Compute weight statistics if model is provided and at checkpoint interval
        if model and step % self.checkpoint_interval == 0:
            weight_stats = self._compute_weight_statistics(model)
            
        # Create a record of training metrics for this step
        training_metric = TrainingMetrics(
            step=step,
            loss=loss,
            learning_rate=lr,
            metrics=metrics,
            convergence_delta=convergence_delta,
            weight_stats=weight_stats
        )
        
        # Append this training metric to history
        self.training_history.append(training_metric)
        # Append convergence delta to buffer if it exists
        self.convergence_buffer.append(convergence_delta) if convergence_delta else None
        
        # Log training progress if at the designated log interval
        if step % self.log_interval == 0:
            self._log_training_progress(training_metric)
            
        # Check for any alerts based on the current training metric
        self._check_training_alerts(training_metric)

    def _calculate_convergence_delta(self, current_loss: float) -> Optional[float]:
        # Calculate absolute difference between current loss and previous loss if history exists
        if len(self.training_history) > 0:
            previous_loss = self.training_history[-1].loss
            return abs(current_loss - previous_loss)
        # Return None if no previous loss to compare
        return None

    def _compute_weight_statistics(self, model: torch.nn.Module) -> Dict[str, Any]:
        weight_stats = {}
        # Iterate through model parameters and collect statistics for those requiring gradients
        for name, param in model.named_parameters():
            if param.requires_grad and param.data is not None:
                tensor_data = param.data.cpu().numpy()
                # Compute various statistics for weights of the parameter
                weight_stats[name] = {
                    "mean": float(np.mean(tensor_data)),
                    "std": float(np.std(tensor_data)),
                    "min": float(np.min(tensor_data)),
                    "max": float(np.max(tensor_data)),
                    "norm": float(torch.norm(param.data).item()),
                    "sparsity": float(np.mean(tensor_data == 0)),
                    # Gradient norm if gradients are available, else None
                    "gradient_norm": float(torch.norm(param.grad).item()) if param.grad is not None else None
                }
        return weight_stats

    def _log_training_progress(self, metric: TrainingMetrics) -> None:
        # Format metrics into a string for logging
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metric.metrics.items()])
        # Log step number, loss, learning rate, and additional metrics
        self.logger.info(f"Step {metric.step}: Loss={metric.loss:.4f}, LR={metric.learning_rate:.6f}, {metrics_str}")

    def track_batch_processing(self, batch_time: float, data_time: float) -> None:
        # Append batch processing and data loading times to respective lists
        self.batch_processing_times.append(batch_time)
        self.data_loading_times.append(data_time)
        
        # Log average times every log_interval batches
        if len(self.batch_processing_times) % self.log_interval == 0:
            avg_batch_time = np.mean(list(self.batch_processing_times))
            avg_data_time = np.mean(list(self.data_loading_times))
            self.logger.info(f"Batch processing: {avg_batch_time:.3f}s, Data loading: {avg_data_time:.3f}s")

    def monitor_gradient_flow(self, model: torch.nn.Module) -> Dict[str, float]:
        gradient_norms = {}
        # Collect gradient norms for all parameters that have gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = torch.norm(param.grad).item()
                gradient_norms[name] = grad_norm
        return gradient_norms

    def analyze_loss_convergence(self, window_size: int = 100) -> Dict[str, Any]:
        # If not enough data in history, return status indicating insufficient data
        if len(self.training_history) < window_size:
            return {"status": "insufficient_data", "samples": len(self.training_history)}
            
        # Extract recent loss values from the training history
        recent_losses = [m.loss for m in self.training_history[-window_size:]]
        loss_array = np.array(recent_losses)
        
        # Smooth losses using moving average with window size min(10, number of samples)
        smoothed_losses = np.convolve(loss_array, np.ones(min(10, len(loss_array)))/min(10, len(loss_array)), mode='valid')
        
        # Calculate slope of loss trend over the smoothed window
        convergence_slope = (smoothed_losses[-1] - smoothed_losses[0]) / len(smoothed_losses) if len(smoothed_losses) > 1 else 0
        # Calculate average absolute change between consecutive smoothed losses
        convergence_rate = np.mean(np.abs(np.diff(smoothed_losses))) if len(smoothed_losses) > 1 else 0
        # Calculate variance of smoothed losses
        loss_variance = np.var(smoothed_losses)
        
        # Determine convergence or divergence based on thresholds
        is_converged = loss_variance < self.alert_thresholds["stagnation_threshold"]
        is_diverging = convergence_slope > self.alert_thresholds["loss_increase"]
        
        # Return summary of convergence analysis
        return {
            "status": "converged" if is_converged else "diverging" if is_diverging else "training",
            "slope": convergence_slope,
            "convergence_rate": convergence_rate,
            "variance": loss_variance,
            "window_size": window_size,
            "samples_analyzed": len(recent_losses)
        }

    def detect_training_anomalies(self) -> List[str]:
        anomalies = []
        
        # If less than two history points, return empty anomaly list
        if len(self.training_history) < 2:
            return anomalies
            
        # Get the latest training metric
        current_metric = self.training_history[-1]
        
        # Check for NaN or Inf in the loss (numerical instability)
        if np.isnan(current_metric.loss) or np.isinf(current_metric.loss):
            anomalies.append("Loss is NaN or Inf - numerical instability detected")
            
        # Check for loss stagnation if enough history is available
        if len(self.training_history) >= 10:
            recent_losses = [m.loss for m in self.training_history[-10:]]
            if np.std(recent_losses) < self.alert_thresholds["stagnation_threshold"]:
                anomalies.append("Loss has plateaued - potential training stagnation")
                
        # Check for divergence by comparing recent losses to historical average
        if len(self.training_history) >= 50:
            recent_trend = np.mean([m.loss for m in self.training_history[-10:]])
            historical_avg = np.mean([m.loss for m in self.training_history[-50:-10]])
            if recent_trend > historical_avg * self.alert_thresholds["divergence_factor"]:
                anomalies.append("Loss is diverging - training may be unstable")
                
        # Analyze weight statistics for signs of vanishing weights or exploding gradients
        if current_metric.weight_stats:
            for layer_name, stats in current_metric.weight_stats.items():
                # Detect vanishing weights: mean and std very close to zero
                if abs(stats["mean"]) < 1e-7 and stats["std"] < 1e-7:
                    anomalies.append(f"Vanishing weights detected in layer: {layer_name}")
                # Detect exploding gradients: gradient norm above threshold
                if stats["gradient_norm"] and stats["gradient_norm"] > self.alert_thresholds["gradient_norm"]:
                    anomalies.append(f"Exploding gradients detected in layer: {layer_name}")
                    
        # Return list of detected anomalies
        return anomalies

    def _check_training_alerts(self, metric: TrainingMetrics) -> None:
        # Detect any anomalies in the training process
        anomalies = self.detect_training_anomalies()
        # Log each detected anomaly as a warning
        for anomaly in anomalies:
            self.logger.warning(f"Training Alert: {anomaly}")

    def get_training_summary(self, time_window: int = 300) -> Dict[str, Any]:
        # Return status if no training history is present
        if not self.training_history:
            return {"status": "no_data"}
            
        # Calculate cutoff time based on time window in seconds
        cutoff_time = datetime.utcnow() - timedelta(seconds=time_window) 
        # Filter metrics within the cutoff time window
        recent_metrics = [m for m in self.training_history if m.timestamp >= cutoff_time]
        
        # If no recent metrics, fallback to last 10 or all metrics available
        if not recent_metrics:
            recent_metrics = self.training_history[-10:] if len(self.training_history) >= 10 else self.training_history
            
        # Extract losses and learning rates from recent metrics
        losses = [m.loss for m in recent_metrics]
        learning_rates = [m.learning_rate for m in recent_metrics]
        
        # Build summary dictionary with various statistics and analysis
        summary = {
            "total_steps": len(self.training_history),
            "recent_steps": len(recent_metrics),
            "current_loss": losses[-1] if losses else None,
            "loss_trend": {
                "mean": np.mean(losses),
                "std": np.std(losses),
                "min": np.min(losses),
                "max": np.max(losses)
            },
            "learning_rate_trend": {
                "current": learning_rates[-1] if learning_rates else None,
                "mean": np.mean(learning_rates),
                "range": [np.min(learning_rates), np.max(learning_rates)]
            },
            # Include detailed convergence analysis results
            "convergence_analysis": self.analyze_loss_convergence(),
            # Include any anomalies detected in recent training
            "recent_anomalies": self.detect_training_anomalies(),
            # Average batch processing and data loading times
            "batch_processing": {
                "avg_batch_time": np.mean(list(self.batch_processing_times)) if self.batch_processing_times else None,
                "avg_data_time": np.mean(list(self.data_loading_times)) if self.data_loading_times else None
            }
        }
        
        return summary

    def save_checkpoint(self, filepath: str, additional_data: Optional[Dict[str, Any]] = None) -> None:
        # Prepare checkpoint data dictionary for saving
        checkpoint_data = {
            "training_history": [
                {
                    "step": m.step,
                    "loss": m.loss,
                    "learning_rate": m.learning_rate,
                    "metrics": m.metrics,
                    "timestamp": m.timestamp.isoformat(),
                    "convergence_delta": m.convergence_delta
                }
                for m in self.training_history
            ],
            "current_step": self.current_step,
            "training_start_time": self.training_start_time,
            "alert_thresholds": self.alert_thresholds,
            "config": self.config.__dict__ if self.config else None
        }
        
        # Update checkpoint with any additional user-provided data
        if additional_data:
            checkpoint_data.update(additional_data)
            
        # Write checkpoint data to JSON file with indentation and proper string serialization
        with open(filepath, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
            
        # Log successful checkpoint saving
        self.logger.info(f"Training checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        # Load checkpoint data from JSON file
        with open(filepath, 'r') as f:
            checkpoint_data = json.load(f)
            
        # Restore internal state from checkpoint data
        self.current_step = checkpoint_data.get("current_step", 0)
        self.training_start_time = checkpoint_data.get("training_start_time")
        self.alert_thresholds = checkpoint_data.get("alert_thresholds", self._initialize_alert_thresholds())
        
        # Log successful checkpoint loading
        self.logger.info(f"Training checkpoint loaded from {filepath}")
        return checkpoint_data

    def reset_monitoring(self) -> None:
        # Clear all stored monitoring data and reset state variables
        self.training_history.clear()
        self.convergence_buffer.clear()
        self.batch_processing_times.clear()
        self.data_loading_times.clear()
        self.weight_tracking_registry.clear()
        self.current_step = 0
        self.training_start_time = None
        # Log that monitoring has been reset
        self.logger.info("Training monitor reset")