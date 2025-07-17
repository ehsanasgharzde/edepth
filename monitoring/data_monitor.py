#FILE: monitoring/data_monitor.py
# ehsanasgharzde - DATA MONITOR
# hosseinsolymanzadeh - PROPER COMMENTING

import torch
import numpy as np
from typing import Union, Dict, List, Any, Optional, Callable, Tuple
from collections import defaultdict, deque
import logging
from dataclasses import dataclass, field
import hashlib
import json
from datetime import datetime
import time

@dataclass
class DataMetrics:
    tensor_id: str  # Unique identifier for the tensor
    shape: Tuple[int, ...]  # Shape of the tensor
    dtype: str  # Data type of the tensor
    device: str  # Device on which the tensor is stored (e.g., 'cpu' or 'cuda')
    memory_usage: float  # Memory usage in MB
    min_value: float  # Minimum value in the tensor
    max_value: float  # Maximum value in the tensor
    mean_value: float  # Mean value of the tensor
    std_value: float  # Standard deviation of the tensor
    nan_count: int  # Number of NaN values in the tensor
    inf_count: int  # Number of Inf values in the tensor
    gradient_norm: Optional[float]  # Norm of the gradient if available
    timestamp: datetime = field(default_factory=datetime.utcnow)  # Timestamp of metric capture

class DataMonitor:
    
    def __init__(self, track_gradients: bool = True, 
                 track_activations: bool = True,
                 sample_rate: float = 1.0,
                 log_level: int = logging.INFO):
        self.track_gradients = track_gradients  # Flag to track gradient stats
        self.track_activations = track_activations  # Flag to track activation stats
        self.sample_rate = sample_rate  # Rate at which tensors are sampled
        
        self.logger = logging.getLogger("data_monitor")  # Logger for monitoring
        self.logger.setLevel(log_level)  # Set log level
        if not self.logger.handlers:
            handler = logging.StreamHandler()  # Stream handler for console output
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)  # Add handler if not already attached
        
        self.tensor_registry = {}  # Registry of tracked tensors
        self.gradient_stats = defaultdict(list)  # Stores gradient norms over time
        self.activation_stats = defaultdict(list)  # Stores activation stats over time
        self.flow_history = deque(maxlen=1000)  # History of tensor flows
        self.tensor_metadata = {}  # Metadata for tensors
        self.hooks = []  # List of registered hooks
        
        self.convergence_info = deque(maxlen=1000)  # Track convergence events
        self.anomaly_history = deque(maxlen=500)  # Track detected anomalies
        
    def _generate_tensor_id(self, tensor: torch.Tensor) -> str:
        # Generate a unique hash for a tensor based on its shape, dtype, and id
        tensor_hash = hashlib.md5(str(tensor.shape).encode() + str(tensor.dtype).encode() + str(id(tensor)).encode()).hexdigest()
        return tensor_hash[:16]  # Return first 16 characters as ID

    def _calculate_tensor_stats(self, tensor: torch.Tensor) -> Dict[str, Any]:
        # Compute statistics for a given tensor
        with torch.no_grad():
            stats = {
                'memory_usage': tensor.element_size() * tensor.nelement() / (1024 ** 2),  # Size in MB
                'min_value': float(torch.min(tensor).item()),  # Minimum value
                'max_value': float(torch.max(tensor).item()),  # Maximum value
                'mean_value': float(torch.mean(tensor).item()),  # Mean
                'std_value': float(torch.std(tensor).item()),  # Standard deviation
                'nan_count': int(torch.isnan(tensor).sum().item()),  # Count of NaNs
                'inf_count': int(torch.isinf(tensor).sum().item()),  # Count of Infs
                'sparsity': float((tensor == 0).float().mean().item()),  # Fraction of zero values
                'shape': tuple(tensor.shape),  # Tensor shape
                'dtype': str(tensor.dtype),  # Data type as string
                'device': str(tensor.device)  # Device as string
            }
        return stats

    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        # Log an event with timestamp and data
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'data': data
        }
        self.logger.info(json.dumps(log_entry))  # Log as JSON

    def register_tensor(self, tensor: torch.Tensor, 
                        name: str, 
                        category: str = "unknown") -> str:
        # Register a tensor for monitoring
        tensor_id = self._generate_tensor_id(tensor)  # Create ID
        
        stats = self._calculate_tensor_stats(tensor)  # Compute stats
        
        # Store tensor metadata
        self.tensor_registry[tensor_id] = {
            "name": name,
            "category": category,
            "stats": stats,
            "registered_at": datetime.utcnow()
        }
        
        # Optionally track gradient norms
        if self.track_gradients and tensor.requires_grad:
            def grad_hook(grad, tensor_name=name):
                if grad is not None:
                    grad_norm = float(grad.norm().item())  # Compute gradient norm
                    self.gradient_stats[tensor_name].append({
                        'norm': grad_norm,
                        'timestamp': datetime.utcnow()
                    })
                    # Keep only the latest 100 entries
                    if len(self.gradient_stats[tensor_name]) > 100:
                        self.gradient_stats[tensor_name].pop(0)
            
            hook = tensor.register_hook(grad_hook)  # Register gradient hook
            self.hooks.append(hook)  # Store hook for potential cleanup
        
        # Log the registration event
        self._log_event("tensor_registered", {
            "tensor_id": tensor_id,
            "name": name,
            "category": category,
            "shape": stats['shape'],
            "memory_mb": stats['memory_usage']
        })
        
        return tensor_id  # Return the generated tensor ID

    def track_tensor_stats(self, tensor: torch.Tensor, name: str) -> DataMetrics:
        # Generate a unique identifier for the tensor
        tensor_id = self._generate_tensor_id(tensor)

        # Calculate basic statistics about the tensor
        stats = self._calculate_tensor_stats(tensor)

        gradient_norm = None
        # If the tensor has gradients, compute the L2 norm of the gradient
        if tensor.requires_grad and tensor.grad is not None:
            gradient_norm = float(tensor.grad.norm().item())

        # Package all collected statistics into a DataMetrics object
        metrics = DataMetrics(
            tensor_id=tensor_id,
            shape=stats['shape'],
            dtype=stats['dtype'],
            device=stats['device'],
            memory_usage=stats['memory_usage'],
            min_value=stats['min_value'],
            max_value=stats['max_value'],
            mean_value=stats['mean_value'],
            std_value=stats['std_value'],
            nan_count=stats['nan_count'],
            inf_count=stats['inf_count'],
            gradient_norm=gradient_norm
        )

        # Log the statistics event for future analysis or tracking
        self._log_event("tensor_stats", {
            "tensor_id": tensor_id,
            "name": name,
            "stats": stats,
            "gradient_norm": gradient_norm
        })

        return metrics

    def monitor_gradient_flow(self, model: torch.nn.Module) -> Dict[str, Any]:
        monitored_params = 0  # Counter for parameters being monitored

        # Iterate over all parameters that require gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Define a hook to capture gradient statistics during backprop
                def create_hook(param_name):
                    def hook(grad):
                        if grad is not None:
                            # Record the norm of the gradient and timestamp
                            grad_norm = float(grad.norm().item())
                            self.gradient_stats[param_name].append({
                                'norm': grad_norm,
                                'timestamp': datetime.utcnow()
                            })

                            # Limit stored history to 100 entries
                            if len(self.gradient_stats[param_name]) > 100:
                                self.gradient_stats[param_name].pop(0)
                    return hook

                # Register the hook and store its handle
                hook = param.register_hook(create_hook(name))
                self.hooks.append(hook)
                monitored_params += 1

        result = {
            "status": "gradient_hooks_registered",
            "params_monitored": monitored_params,
            "total_hooks": len(self.hooks)
        }

        # Log the gradient monitoring setup
        self._log_event("gradient_monitoring_started", result)
        return result

    def monitor_activation_patterns(self, model: torch.nn.Module) -> Dict[str, Any]:
        monitored_modules = 0  # Counter for monitored modules

        # Iterate through all modules in the model
        for name, module in model.named_modules():
            # Skip sequential containers to avoid redundancy
            if not isinstance(module, torch.nn.Sequential):
                # Define a hook to monitor the module's output
                def create_forward_hook(module_name):
                    def hook(module, input, output):
                        if isinstance(output, torch.Tensor):
                            with torch.no_grad():
                                # Collect various activation statistics
                                activation_stats = {
                                    "mean": float(output.mean().item()),
                                    "std": float(output.std().item()),
                                    "sparsity": float((output == 0).float().mean().item()),
                                    "min": float(output.min().item()),
                                    "max": float(output.max().item()),
                                    "timestamp": datetime.utcnow()
                                }

                                # Store the stats for the module
                                self.activation_stats[module_name].append(activation_stats)

                                # Limit stored history to 50 entries
                                if len(self.activation_stats[module_name]) > 50:
                                    self.activation_stats[module_name].pop(0)
                    return hook

                # Register the forward hook and store its handle
                hook = module.register_forward_hook(create_forward_hook(name))
                self.hooks.append(hook)
                monitored_modules += 1

        result = {
            "status": "activation_hooks_registered",
            "modules_monitored": monitored_modules,
            "total_hooks": len(self.hooks)
        }

        # Log the activation monitoring setup
        self._log_event("activation_monitoring_started", result)
        return result

    def validate_data_integrity(self, data: Union[torch.Tensor, np.ndarray],
                                schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Determine whether the data is a PyTorch tensor
        is_tensor = isinstance(data, torch.Tensor)

        # Count NaN and Inf values depending on the data type
        if is_tensor:
            nan_count = int(torch.isnan(data).sum().item())
            inf_count = int(torch.isinf(data).sum().item())
        else:
            nan_count = int(np.isnan(data).sum())
            inf_count = int(np.isinf(data).sum())

        # Construct the initial integrity report
        result = {
            "is_tensor": is_tensor,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "shape": tuple(data.shape),
            "dtype": str(data.dtype),
            "valid": nan_count == 0 and inf_count == 0
        }

        # If a schema is provided, validate shape and dtype
        if schema:
            if "shape" in schema and tuple(data.shape) != tuple(schema["shape"]):
                result["shape_mismatch"] = True
                result["valid"] = False
            if "dtype" in schema and str(data.dtype) != str(schema["dtype"]):
                result["dtype_mismatch"] = True
                result["valid"] = False

        # Log an event if data integrity is violated
        if not result["valid"]:
            self._log_event("data_integrity_violation", result)

        return result

    def track_data_flow(self, data: Any, stage: str, operation: str) -> str:
        # Generate a unique trace ID using stage, operation, and current timestamp
        trace_id = hashlib.md5(f"{stage}_{operation}_{time.time()}".encode()).hexdigest()[:16]
    
        # Create a flow entry dictionary capturing metadata about the data
        flow_entry = {
            "trace_id": trace_id,
            "stage": stage,
            "operation": operation,
            "timestamp": datetime.utcnow(),
            "data_type": str(type(data)),
            "data_shape": getattr(data, 'shape', None) if hasattr(data, 'shape') else None
        }
    
        # Append the flow entry to history
        self.flow_history.append(flow_entry)
    
        # Log the data flow tracking event
        self._log_event("data_flow_tracked", flow_entry)
    
        # Return the generated trace ID
        return trace_id
    
    def detect_data_anomalies(self, data: torch.Tensor,
                              reference_stats: Optional[Dict[str, Any]] = None) -> List[str]:
        anomalies = []
    
        # Check for NaN values
        if torch.isnan(data).any():
            anomalies.append("NaN_values_detected")
    
        # Check for Inf values
        if torch.isinf(data).any():
            anomalies.append("Inf_values_detected")
    
        # Calculate current tensor statistics
        current_stats = self._calculate_tensor_stats(data)
    
        if reference_stats:
            # Compute difference in means and check against threshold
            mean_diff = abs(current_stats['mean_value'] - reference_stats.get('mean', 0))
            std_threshold = 3 * reference_stats.get('std', 1)
    
            # Check for significant mean shift (distribution drift)
            if mean_diff > std_threshold:
                anomalies.append("distribution_drift_detected")
    
            # Check for unusually high variance
            if current_stats['std_value'] > 5 * reference_stats.get('std', 1):
                anomalies.append("high_variance_detected")
    
        # Detect high sparsity (mostly zero values)
        if current_stats['sparsity'] > 0.95:
            anomalies.append("high_sparsity_detected")
    
        if anomalies:
            # Create and store anomaly log entry
            anomaly_entry = {
                'timestamp': datetime.utcnow(),
                'anomalies': anomalies,
                'stats': current_stats
            }
            self.anomaly_history.append(anomaly_entry)
    
            # Log the anomaly detection event
            self._log_event("anomalies_detected", anomaly_entry)
    
        # Return list of detected anomalies
        return anomalies
    
    def get_data_summary(self, time_window: int = 300) -> Dict[str, Any]:
        now = datetime.utcnow()
    
        # Filter flow events within the time window
        recent_flows = [f for f in self.flow_history 
                       if (now - f["timestamp"]).total_seconds() <= time_window]
    
        # Filter anomaly events within the time window
        recent_anomalies = [a for a in self.anomaly_history 
                           if (now - a["timestamp"]).total_seconds() <= time_window]
    
        gradient_summary = {}
        for param_name, grad_history in self.gradient_stats.items():
            # Filter recent gradient updates
            recent_grads = [g for g in grad_history 
                           if (now - g["timestamp"]).total_seconds() <= time_window]
            if recent_grads:
                norms = [g['norm'] for g in recent_grads]
                gradient_summary[param_name] = {
                    'max_norm': max(norms),
                    'min_norm': min(norms),
                    'avg_norm': sum(norms) / len(norms),
                    'count': len(norms)
                }
    
        activation_summary = {}
        for module_name, activation_history in self.activation_stats.items():
            # Filter recent activations
            recent_acts = [a for a in activation_history 
                          if (now - a["timestamp"]).total_seconds() <= time_window]
            if recent_acts:
                activation_summary[module_name] = {
                    'avg_mean': sum(a['mean'] for a in recent_acts) / len(recent_acts),
                    'avg_sparsity': sum(a['sparsity'] for a in recent_acts) / len(recent_acts),
                    'count': len(recent_acts)
                }
    
        # Build final summary dictionary
        summary = {
            "tensor_count": len(self.tensor_registry),
            "recent_flows": len(recent_flows),
            "recent_anomalies": len(recent_anomalies),
            "gradient_stats": gradient_summary,
            "activation_stats": activation_summary,
            "active_hooks": len(self.hooks),
            "time_window_seconds": time_window
        }
    
        # Log the generated summary
        self._log_event("data_summary_generated", summary)
    
        # Return summary report
        return summary
    
    def cleanup_hooks(self) -> None:
        # Remove all registered hooks
        for hook in self.hooks:
            hook.remove()
    
        # Clear the hook list
        self.hooks.clear()
    
        # Log the cleanup event
        self._log_event("hooks_cleaned", {"removed_count": len(self.hooks)})
    
    def reset_monitoring(self) -> None:
        # Reset all monitoring-related components and logs
        self.cleanup_hooks()
        self.tensor_registry.clear()
        self.gradient_stats.clear()
        self.activation_stats.clear()
        self.flow_history.clear()
        self.anomaly_history.clear()
    
        # Log the reset event
        self._log_event("monitoring_reset", {"timestamp": datetime.utcnow()})
    