#FILE: monitoring/integration.py
# ehsanasgharzde - INTEGRATION MANAGER

from typing import Dict, Any, List, Optional, Callable
import torch
import torch.nn as nn
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from collections import defaultdict, deque

from .data_monitor import DataMonitor
from .system_monitor import SystemMonitor
from .training_monitor import TrainingMonitor
from .profiler import PerformanceProfiler
from .visualization import MonitoringVisualizer
from .hooks import MonitoringHooks
from .logger import MonitoringLogger
from .config import MonitoringConfig

@dataclass
class MonitoringState:
    is_active: bool = False
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_duration: float = 0.0
    collected_metrics: Dict[str, List[Any]] = field(default_factory=dict)
    alerts_triggered: List[str] = field(default_factory=list)

class MonitoringIntegration:
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.logger = MonitoringLogger(config.output_directory)
        self.state = MonitoringState()
        
        self.data_monitor = DataMonitor(
            track_gradients=config.enable_gradient_monitoring,
            track_activations=config.enable_activation_monitoring,
            sample_rate=config.data_monitoring_sample_rate
        )
        
        self.system_monitor = SystemMonitor(
            monitoring_interval=config.system_monitoring_interval,
            history_size=1000
        )
        
        self.training_monitor = TrainingMonitor(
            log_interval=10,
            checkpoint_interval=100
        )
        
        self.profiler = PerformanceProfiler(
            profile_memory=True,
            profile_cuda=config.enable_gpu_monitoring
        )
        
        self.visualizer = MonitoringVisualizer(
            config=config,
            logger=self.logger
        )
        
        self.hooks_manager = MonitoringHooks(
            self.data_monitor,
            self.system_monitor
        )
        
        self.alert_thresholds = config.alert_thresholds or {} #type: ignore 
        self.monitoring_thread = None
        self.stop_event = threading.Event()
        
        logging.basicConfig(level=getattr(logging, config.log_level.upper())) #type: ignore 
        self.log = logging.getLogger(__name__)

    def start(self) -> None:
        if self.state.is_active:
            self.log.warning("Monitoring already active")
            return
            
        self.state.is_active = True
        self.state.start_time = datetime.utcnow()
        self.stop_event.clear()
        
        self.system_monitor.start_monitoring()
        
        if self.config.enable_profiling: #type: ignore 
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True
            )
            self.monitoring_thread.start()
            
        self.log.info("Monitoring system started")
        self.logger.log_system_metrics({"event": "monitoring_started"})

    def stop(self) -> None:
        if not self.state.is_active:
            self.log.warning("Monitoring not active")
            return
            
        self.state.is_active = False
        self.state.end_time = datetime.utcnow()
        
        if self.state.start_time:
            self.state.total_duration = (
                self.state.end_time - self.state.start_time
            ).total_seconds()
        
        self.stop_event.set()
        self.system_monitor.stop_monitoring()
        
        if self.monitoring_thread:
            self.monitoring_thread.join()
            
        self.hooks_manager.remove_all_hooks()
        self._generate_final_report()
        
        self.log.info("Monitoring system stopped")
        self.logger.log_system_metrics({"event": "monitoring_stopped"})

    def integrate_with_trainer(self, trainer) -> None:
        self.hooks_manager.register_training_hooks(trainer)
        
        if hasattr(trainer, 'on_step_end'):
            trainer.on_step_end.append(self._on_training_step)
        if hasattr(trainer, 'on_epoch_end'):
            trainer.on_epoch_end.append(self._on_training_epoch)
        if hasattr(trainer, 'on_loss_computed'):
            trainer.on_loss_computed.append(self._on_loss_computed)
        if hasattr(trainer, 'model'):
            self._register_model_hooks(trainer.model)
            
        self.log.info("Training integration completed")

    def integrate_with_model(self, model: nn.Module) -> None:
        self._register_model_hooks(model)
        
        if self.config.enable_gradient_monitoring:
            self.data_monitor.monitor_gradient_flow(model)
        if self.config.enable_activation_monitoring:
            self.data_monitor.monitor_activation_patterns(model)
            
        self.log.info("Model integration completed")

    def integrate_with_dataloader(self, dataloader) -> None:
        if hasattr(dataloader, 'dataset'):
            self._register_dataloader_hooks(dataloader)
            
        self.log.info("Dataloader integration completed")

    def _register_model_hooks(self, model: nn.Module) -> None:
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.LSTM)):
                self.hooks_manager.register_forward_hook(module, f"{name}_forward")
                self.hooks_manager.register_backward_hook(module, f"{name}_backward")

    def _register_dataloader_hooks(self, dataloader) -> None:
        original_getitem = dataloader.dataset.__getitem__
        
        def monitored_getitem(idx):
            start_time = time.time()
            data = original_getitem(idx)
            end_time = time.time()
            
            self.logger.log_data_metrics({
                "data_load_time": end_time - start_time,
                "data_index": idx
            })
            
            return data
            
        dataloader.dataset.__getitem__ = monitored_getitem

    def _on_training_step(self, step: int, loss: float, lr: float) -> None:
        self.training_monitor.log_training_step(step, loss, {}, lr)
        
        if step % self.config.profiling_interval == 0 and self.config.enable_profiling: #type: ignore 
            self._capture_performance_metrics()
            
        self._check_alerts(loss)

    def _on_training_epoch(self, epoch: int, metrics: Dict[str, Any]) -> None:
        self.logger.log_training_event("epoch_completed", {
            "epoch": epoch,
            "metrics": metrics
        })
        
        if self.config.enable_visualization:
            self._update_visualizations()

    def _on_loss_computed(self, loss: float) -> None:
        self.state.collected_metrics.setdefault("losses", []).append(loss)
        
        if len(self.state.collected_metrics["losses"]) > 100:
            convergence_info = self.training_monitor.monitor_loss_convergence( #type: ignore 
                self.state.collected_metrics["losses"]
            )
            if convergence_info.get("plateau_detected"):
                self.state.alerts_triggered.append("Training plateau detected")

    def _capture_performance_metrics(self) -> None:
        system_metrics = self.system_monitor.collect_system_metrics()
        self.logger.log_system_metrics(system_metrics)
        
        self.state.collected_metrics.setdefault("system_metrics", []).append(system_metrics)

    def _check_alerts(self, current_loss: float) -> None:
        if "loss_increase" in self.alert_thresholds:
            losses = self.state.collected_metrics.get("losses", [])
            if len(losses) >= 2:
                increase = (current_loss - losses[-2]) / losses[-2]
                if increase > self.alert_thresholds["loss_increase"]:
                    alert_msg = f"Loss increased by {increase:.2%}"
                    self.state.alerts_triggered.append(alert_msg)
                    self.log.warning(alert_msg)

    def _update_visualizations(self) -> None:
        if not self.config.enable_visualization:
            return
            
        losses = self.state.collected_metrics.get("losses", [])
        if losses:
            self.visualizer.plot_training_progress(losses, {}) #type: ignore 
            
        system_metrics = self.state.collected_metrics.get("system_metrics", [])
        if system_metrics:
            metrics_dict = self._convert_system_metrics_to_dict(system_metrics)
            self.visualizer.plot_system_metrics(metrics_dict) #type: ignore 

    def _convert_system_metrics_to_dict(self, system_metrics: List[Any]) -> List[Dict[str, Any]]:
        converted = []
        for metric in system_metrics:
            converted.append({
                "time": metric.timestamp,
                "cpu": metric.cpu_percent,
                "memory": metric.memory_percent,
                "gpu": metric.gpu_utilization[0] if metric.gpu_utilization else 0,
                "temperature": metric.gpu_temperature[0] if metric.gpu_temperature else 0
            })
        return converted

    def _monitoring_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                self._capture_performance_metrics()
                time.sleep(self.config.system_monitoring_interval)
            except Exception as e:
                self.logger.log_error_event(e, {"context": "monitoring_loop"})
                self.log.error(f"Error in monitoring loop: {e}")

    def _generate_final_report(self) -> Dict[str, Any]:
        report = {
            "monitoring_duration": self.state.total_duration,
            "total_alerts": len(self.state.alerts_triggered),
            "alerts": self.state.alerts_triggered,
            "metrics_collected": {
                k: len(v) for k, v in self.state.collected_metrics.items()
            }
        }
        
        if self.config.enable_visualization:
            self._generate_final_visualizations()
            
        self.logger.log_system_metrics({"event": "final_report", "report": report})
        return report

    def _generate_final_visualizations(self) -> None:
        losses = self.state.collected_metrics.get("losses", [])
        if losses:
            self.visualizer.plot_training_progress(losses, {}) #type: ignore 
            
        system_metrics = self.state.collected_metrics.get("system_metrics", [])
        if system_metrics:
            metrics_dict = self._convert_system_metrics_to_dict(system_metrics)
            dashboard_path = self.visualizer.create_monitoring_dashboard({ #type: ignore    
                "time": [m["time"] for m in metrics_dict],
                "cpu": [m["cpu"] for m in metrics_dict],
                "memory": [m["memory"] for m in metrics_dict],
                "gpu": [m["gpu"] for m in metrics_dict],
                "temperature": [m["temperature"] for m in metrics_dict]
            })
            self.log.info(f"Dashboard created at: {dashboard_path}")

    def get_current_status(self) -> Dict[str, Any]:
        return {
            "is_active": self.state.is_active,
            "start_time": self.state.start_time,
            "duration": (
                datetime.utcnow() - self.state.start_time
            ).total_seconds() if self.state.start_time else 0,
            "alerts_count": len(self.state.alerts_triggered),
            "metrics_collected": {
                k: len(v) for k, v in self.state.collected_metrics.items()
            }
        }