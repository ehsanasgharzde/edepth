#FILE: monitoring/data_monitor.py
# ehsanasgharzde - DATA MONITOR

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
    tensor_id: str
    shape: Tuple[int, ...]
    dtype: str
    device: str
    memory_usage: float
    min_value: float
    max_value: float
    mean_value: float
    std_value: float
    nan_count: int
    inf_count: int
    gradient_norm: Optional[float]
    timestamp: datetime = field(default_factory=datetime.utcnow)

class DataMonitor:
    
    def __init__(self, track_gradients: bool = True, 
                 track_activations: bool = True,
                 sample_rate: float = 1.0,
                 log_level: int = logging.INFO):
        self.track_gradients = track_gradients
        self.track_activations = track_activations
        self.sample_rate = sample_rate
        
        self.logger = logging.getLogger("data_monitor")
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.tensor_registry = {}
        self.gradient_stats = defaultdict(list)
        self.activation_stats = defaultdict(list)
        self.flow_history = deque(maxlen=1000)
        self.tensor_metadata = {}
        self.hooks = []
        
        self.convergence_info = deque(maxlen=1000)
        self.anomaly_history = deque(maxlen=500)
        
    def _generate_tensor_id(self, tensor: torch.Tensor) -> str:
        tensor_hash = hashlib.md5(str(tensor.shape).encode() + str(tensor.dtype).encode() + str(id(tensor)).encode()).hexdigest()
        return tensor_hash[:16]

    def _calculate_tensor_stats(self, tensor: torch.Tensor) -> Dict[str, Any]:
        with torch.no_grad():
            stats = {
                'memory_usage': tensor.element_size() * tensor.nelement() / (1024 ** 2),
                'min_value': float(torch.min(tensor).item()),
                'max_value': float(torch.max(tensor).item()),
                'mean_value': float(torch.mean(tensor).item()),
                'std_value': float(torch.std(tensor).item()),
                'nan_count': int(torch.isnan(tensor).sum().item()),
                'inf_count': int(torch.isinf(tensor).sum().item()),
                'sparsity': float((tensor == 0).float().mean().item()),
                'shape': tuple(tensor.shape),
                'dtype': str(tensor.dtype),
                'device': str(tensor.device)
            }
        return stats

    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'data': data
        }
        self.logger.info(json.dumps(log_entry))

    def register_tensor(self, tensor: torch.Tensor, 
                        name: str, 
                        category: str = "unknown") -> str:
        tensor_id = self._generate_tensor_id(tensor)
        
        stats = self._calculate_tensor_stats(tensor)
        
        self.tensor_registry[tensor_id] = {
            "name": name,
            "category": category,
            "stats": stats,
            "registered_at": datetime.utcnow()
        }
        
        if self.track_gradients and tensor.requires_grad:
            def grad_hook(grad, tensor_name=name):
                if grad is not None:
                    grad_norm = float(grad.norm().item())
                    self.gradient_stats[tensor_name].append({
                        'norm': grad_norm,
                        'timestamp': datetime.utcnow()
                    })
                    
                    if len(self.gradient_stats[tensor_name]) > 100:
                        self.gradient_stats[tensor_name].pop(0)
                    
            hook = tensor.register_hook(grad_hook)
            self.hooks.append(hook)
        
        self._log_event("tensor_registered", {
            "tensor_id": tensor_id,
            "name": name,
            "category": category,
            "shape": stats['shape'],
            "memory_mb": stats['memory_usage']
        })
        
        return tensor_id

    def track_tensor_stats(self, tensor: torch.Tensor, name: str) -> DataMetrics:
        tensor_id = self._generate_tensor_id(tensor)
        stats = self._calculate_tensor_stats(tensor)
        
        gradient_norm = None
        if tensor.requires_grad and tensor.grad is not None:
            gradient_norm = float(tensor.grad.norm().item())
        
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
        
        self._log_event("tensor_stats", {
            "tensor_id": tensor_id,
            "name": name,
            "stats": stats,
            "gradient_norm": gradient_norm
        })
        
        return metrics

    def monitor_gradient_flow(self, model: torch.nn.Module) -> Dict[str, Any]:
        monitored_params = 0
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                def create_hook(param_name):
                    def hook(grad):
                        if grad is not None:
                            grad_norm = float(grad.norm().item())
                            self.gradient_stats[param_name].append({
                                'norm': grad_norm,
                                'timestamp': datetime.utcnow()
                            })
                            
                            if len(self.gradient_stats[param_name]) > 100:
                                self.gradient_stats[param_name].pop(0)
                    return hook
                
                hook = param.register_hook(create_hook(name))
                self.hooks.append(hook)
                monitored_params += 1
        
        result = {
            "status": "gradient_hooks_registered",
            "params_monitored": monitored_params,
            "total_hooks": len(self.hooks)
        }
        
        self._log_event("gradient_monitoring_started", result)
        return result

    def monitor_activation_patterns(self, model: torch.nn.Module) -> Dict[str, Any]:
        monitored_modules = 0
        
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.Sequential):
                def create_forward_hook(module_name):
                    def hook(module, input, output):
                        if isinstance(output, torch.Tensor):
                            with torch.no_grad():
                                activation_stats = {
                                    "mean": float(output.mean().item()),
                                    "std": float(output.std().item()),
                                    "sparsity": float((output == 0).float().mean().item()),
                                    "min": float(output.min().item()),
                                    "max": float(output.max().item()),
                                    "timestamp": datetime.utcnow()
                                }
                                
                                self.activation_stats[module_name].append(activation_stats)
                                
                                if len(self.activation_stats[module_name]) > 50:
                                    self.activation_stats[module_name].pop(0)
                    return hook
                
                hook = module.register_forward_hook(create_forward_hook(name))
                self.hooks.append(hook)
                monitored_modules += 1
        
        result = {
            "status": "activation_hooks_registered",
            "modules_monitored": monitored_modules,
            "total_hooks": len(self.hooks)
        }
        
        self._log_event("activation_monitoring_started", result)
        return result

    def validate_data_integrity(self, data: Union[torch.Tensor, np.ndarray],
                                schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        is_tensor = isinstance(data, torch.Tensor)
        
        if is_tensor:
            nan_count = int(torch.isnan(data).sum().item())
            inf_count = int(torch.isinf(data).sum().item())
        else:
            nan_count = int(np.isnan(data).sum())
            inf_count = int(np.isinf(data).sum())
        
        result = {
            "is_tensor": is_tensor,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "shape": tuple(data.shape),
            "dtype": str(data.dtype),
            "valid": nan_count == 0 and inf_count == 0
        }
        
        if schema:
            if "shape" in schema and tuple(data.shape) != tuple(schema["shape"]):
                result["shape_mismatch"] = True
                result["valid"] = False
            if "dtype" in schema and str(data.dtype) != str(schema["dtype"]):
                result["dtype_mismatch"] = True
                result["valid"] = False
        
        if not result["valid"]:
            self._log_event("data_integrity_violation", result)
        
        return result

    def track_data_flow(self, data: Any, stage: str, operation: str) -> str:
        trace_id = hashlib.md5(f"{stage}_{operation}_{time.time()}".encode()).hexdigest()[:16]
        
        flow_entry = {
            "trace_id": trace_id,
            "stage": stage,
            "operation": operation,
            "timestamp": datetime.utcnow(),
            "data_type": str(type(data)),
            "data_shape": getattr(data, 'shape', None) if hasattr(data, 'shape') else None
        }
        
        self.flow_history.append(flow_entry)
        
        self._log_event("data_flow_tracked", flow_entry)
        return trace_id

    def detect_data_anomalies(self, data: torch.Tensor,
                              reference_stats: Optional[Dict[str, Any]] = None) -> List[str]:
        anomalies = []
        
        if torch.isnan(data).any():
            anomalies.append("NaN_values_detected")
        if torch.isinf(data).any():
            anomalies.append("Inf_values_detected")
        
        current_stats = self._calculate_tensor_stats(data)
        
        if reference_stats:
            mean_diff = abs(current_stats['mean_value'] - reference_stats.get('mean', 0))
            std_threshold = 3 * reference_stats.get('std', 1)
            
            if mean_diff > std_threshold:
                anomalies.append("distribution_drift_detected")
            
            if current_stats['std_value'] > 5 * reference_stats.get('std', 1):
                anomalies.append("high_variance_detected")
        
        if current_stats['sparsity'] > 0.95:
            anomalies.append("high_sparsity_detected")
        
        if anomalies:
            anomaly_entry = {
                'timestamp': datetime.utcnow(),
                'anomalies': anomalies,
                'stats': current_stats
            }
            self.anomaly_history.append(anomaly_entry)
            
            self._log_event("anomalies_detected", anomaly_entry)
        
        return anomalies

    def get_data_summary(self, time_window: int = 300) -> Dict[str, Any]:
        now = datetime.utcnow()
        
        recent_flows = [f for f in self.flow_history 
                       if (now - f["timestamp"]).total_seconds() <= time_window]
        
        recent_anomalies = [a for a in self.anomaly_history 
                           if (now - a["timestamp"]).total_seconds() <= time_window]
        
        gradient_summary = {}
        for param_name, grad_history in self.gradient_stats.items():
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
            recent_acts = [a for a in activation_history 
                          if (now - a["timestamp"]).total_seconds() <= time_window]
            if recent_acts:
                activation_summary[module_name] = {
                    'avg_mean': sum(a['mean'] for a in recent_acts) / len(recent_acts),
                    'avg_sparsity': sum(a['sparsity'] for a in recent_acts) / len(recent_acts),
                    'count': len(recent_acts)
                }
        
        summary = {
            "tensor_count": len(self.tensor_registry),
            "recent_flows": len(recent_flows),
            "recent_anomalies": len(recent_anomalies),
            "gradient_stats": gradient_summary,
            "activation_stats": activation_summary,
            "active_hooks": len(self.hooks),
            "time_window_seconds": time_window
        }
        
        self._log_event("data_summary_generated", summary)
        return summary

    def cleanup_hooks(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        
        self._log_event("hooks_cleaned", {"removed_count": len(self.hooks)})

    def reset_monitoring(self) -> None:
        self.cleanup_hooks()
        self.tensor_registry.clear()
        self.gradient_stats.clear()
        self.activation_stats.clear()
        self.flow_history.clear()
        self.anomaly_history.clear()
        
        self._log_event("monitoring_reset", {"timestamp": datetime.utcnow()})