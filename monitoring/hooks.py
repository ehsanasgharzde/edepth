#FILE: monitoring/hooks.py
# ehsanasgharzde - HOOKS SYSTEM

from typing import Dict, Any, List
import torch.nn as nn
import torch
from collections import defaultdict
import logging
from datetime import datetime

class MonitoringHooks:
    
    def __init__(self, data_monitor, system_monitor, logger=None):
        self.data_monitor = data_monitor
        self.system_monitor = system_monitor
        self.logger = logger or logging.getLogger(__name__)
        
        self.forward_hooks = {}
        self.backward_hooks = {}
        self.training_hooks = []
        self.inference_hooks = []
        self.hook_data = defaultdict(list)
        
    def register_forward_hook(self, module: nn.Module, hook_name: str) -> None:
        def forward_hook(module, input, output):
            if torch.is_tensor(output):
                stats = self.data_monitor.track_tensor_stats(output, hook_name)
                self.hook_data[hook_name].append({
                    'timestamp': datetime.utcnow(),
                    'input_shape': input[0].shape if isinstance(input, tuple) else input.shape,
                    'output_shape': output.shape,
                    'output_mean': float(output.detach().mean().item()),
                    'output_std': float(output.detach().std().item()),
                    'memory_usage': stats.memory_usage,
                    'nan_count': stats.nan_count,
                    'inf_count': stats.inf_count
                })
                
                if stats.nan_count > 0 or stats.inf_count > 0:
                    self.logger.warning(f"Anomaly detected in {hook_name}: NaN={stats.nan_count}, Inf={stats.inf_count}")
        
        hook_handle = module.register_forward_hook(forward_hook)
        self.forward_hooks[hook_name] = hook_handle
        self.logger.info(f"Forward hook registered: {hook_name}")
        
    def register_backward_hook(self, module: nn.Module, hook_name: str) -> None:
        def backward_hook(module, grad_input, grad_output):
            if grad_output and grad_output[0] is not None:
                grad_tensor = grad_output[0]
                stats = self.data_monitor.track_tensor_stats(grad_tensor, f"{hook_name}_grad")
                self.hook_data[f"{hook_name}_grad"].append({
                    'timestamp': datetime.utcnow(),
                    'grad_norm': float(grad_tensor.detach().norm().item()),
                    'grad_mean': float(grad_tensor.detach().mean().item()),
                    'grad_std': float(grad_tensor.detach().std().item()),
                    'memory_usage': stats.memory_usage,
                    'nan_count': stats.nan_count,
                    'inf_count': stats.inf_count
                })
                
                if stats.nan_count > 0 or stats.inf_count > 0:
                    self.logger.warning(f"Gradient anomaly in {hook_name}: NaN={stats.nan_count}, Inf={stats.inf_count}")
        
        hook_handle = module.register_full_backward_hook(backward_hook)
        self.backward_hooks[hook_name] = hook_handle
        self.logger.info(f"Backward hook registered: {hook_name}")
        
    def register_training_hooks(self, trainer) -> None:
        def on_epoch_start(epoch):
            metrics = self.system_monitor.collect_system_metrics()
            self.logger.info(f"Epoch {epoch} started - System metrics captured")
            
        def on_epoch_end(epoch):
            metrics = self.system_monitor.collect_system_metrics()
            alerts = self.system_monitor.check_resource_alerts(metrics)
            if alerts:
                self.logger.warning(f"Resource alerts after epoch {epoch}: {alerts}")
            self.logger.info(f"Epoch {epoch} ended - System metrics captured")
            
        def on_batch_start(batch_idx):
            self.data_monitor.track_data_flow(batch_idx, "training", "batch_start")
            
        def on_batch_end(batch_idx):
            metrics = self.system_monitor.collect_system_metrics()
            if batch_idx % 100 == 0:
                summary = self.system_monitor.get_performance_summary()
                self.logger.info(f"Batch {batch_idx} processed - Performance: {summary}")
                
        hooks = [on_epoch_start, on_epoch_end, on_batch_start, on_batch_end]
        
        if hasattr(trainer, 'on_epoch_start'):
            trainer.on_epoch_start.extend(hooks[:2])
        if hasattr(trainer, 'on_batch_start'):
            trainer.on_batch_start.extend(hooks[2:])
            
        self.training_hooks.extend(hooks)
        self.logger.info(f"Training hooks registered: {len(hooks)}")
        
    def register_inference_hooks(self, inference_engine) -> None:
        def before_preprocessing(data):
            if torch.is_tensor(data):
                stats = self.data_monitor.track_tensor_stats(data, "inference_input")
                anomalies = self.data_monitor.detect_data_anomalies(data)
                if anomalies:
                    self.logger.warning(f"Input anomalies detected: {anomalies}")
            self.logger.info("Preprocessing started")
            
        def after_postprocessing(output):
            if torch.is_tensor(output):
                stats = self.data_monitor.track_tensor_stats(output, "inference_output")
                anomalies = self.data_monitor.detect_data_anomalies(output)
                if anomalies:
                    self.logger.warning(f"Output anomalies detected: {anomalies}")
            metrics = self.system_monitor.collect_system_metrics()
            self.logger.info("Postprocessing completed")
            
        hooks = [before_preprocessing, after_postprocessing]
        
        if hasattr(inference_engine, 'on_preprocess'):
            inference_engine.on_preprocess.extend(hooks[:1])
        if hasattr(inference_engine, 'on_postprocess'):
            inference_engine.on_postprocess.extend(hooks[1:])
            
        self.inference_hooks.extend(hooks)
        self.logger.info(f"Inference hooks registered: {len(hooks)}")
        
    def monitor_gradient_flow(self, model: nn.Module) -> Dict[str, Any]:
        return self.data_monitor.monitor_gradient_flow(model)
        
    def monitor_activations(self, model: nn.Module) -> Dict[str, Any]:
        return self.data_monitor.monitor_activation_patterns(model)
        
    def get_hook_summary(self, time_window: int = 300) -> Dict[str, Any]:
        now = datetime.utcnow()
        summary = {
            'forward_hooks': len(self.forward_hooks),
            'backward_hooks': len(self.backward_hooks),
            'training_hooks': len(self.training_hooks),
            'inference_hooks': len(self.inference_hooks),
            'recent_data': {}
        }
        
        for hook_name, data_list in self.hook_data.items():
            recent_data = [d for d in data_list if (now - d['timestamp']).total_seconds() <= time_window]
            if recent_data:
                summary['recent_data'][hook_name] = {
                    'count': len(recent_data),
                    'avg_memory': sum(d.get('memory_usage', 0) for d in recent_data) / len(recent_data),
                    'total_anomalies': sum(d.get('nan_count', 0) + d.get('inf_count', 0) for d in recent_data)
                }
        
        return summary
        
    def remove_all_hooks(self) -> None:
        for name, hook in self.forward_hooks.items():
            hook.remove()
        for name, hook in self.backward_hooks.items():
            hook.remove()
            
        self.forward_hooks.clear()
        self.backward_hooks.clear()
        self.training_hooks.clear()
        self.inference_hooks.clear()
        self.hook_data.clear()
        
        self.logger.info("All hooks removed")
        
    def check_hook_health(self) -> List[str]:
        issues = []
        
        for hook_name, data_list in self.hook_data.items():
            if not data_list:
                continue
                
            recent_data = data_list[-10:]
            total_anomalies = sum(d.get('nan_count', 0) + d.get('inf_count', 0) for d in recent_data)
            
            if total_anomalies > 0:
                issues.append(f"Hook {hook_name} detected {total_anomalies} anomalies in recent data")
                
            avg_memory = sum(d.get('memory_usage', 0) for d in recent_data) / len(recent_data)
            if avg_memory > 1000:
                issues.append(f"Hook {hook_name} showing high memory usage: {avg_memory:.2f}MB")
                
        return issues