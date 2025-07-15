#FILE: monitoring/__init__.py
# ehsanasgharzde - MAIN MONITORING CLASS

from typing import Optional, Dict, Any, List, Callable
from pathlib import Path
import logging
import threading
import time
from datetime import datetime

from .config import MonitoringConfig
from .data_monitor import DataMonitor
from .system_monitor import SystemMonitor
from .training_monitor import TrainingMonitor
from .profiler import PerformanceProfiler
from .visualization import MonitoringVisualizer
from .hooks import MonitoringHooks
from .logger import MonitoringLogger
from .integration import MonitoringIntegration, MonitoringState

class ComprehensiveMonitor:
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.state = MonitoringState()
        
        self.monitoring_logger = MonitoringLogger(
            log_dir=config.logging_config.log_dir,
            log_level=getattr(logging, config.logging_config.log_level.upper()),
            structured_logging=config.logging_config.structured_logging
        )
        
        self.data_monitor = DataMonitor(
            track_gradients=config.enable_gradient_monitoring,
            track_activations=config.enable_activation_monitoring,
            sample_rate=config.data_monitoring_sample_rate,
            log_level=getattr(logging, config.logging_config.log_level.upper())
        )
        
        self.system_monitor = SystemMonitor(
            monitoring_interval=config.system_monitoring_interval,
            history_size=config.history_size
        )
        
        self.training_monitor = TrainingMonitor(
            log_interval=config.training_log_interval,
            checkpoint_interval=config.training_checkpoint_interval,
            logger=self.logger,
            config=config
        )
        
        self.profiler = PerformanceProfiler(
            profile_memory=config.profiling_config.profile_memory,
            profile_cuda=config.profiling_config.profile_cuda,
            log_interval=config.profiling_config.profiling_interval,
            output_dir=str(Path(config.output_directory) / "profiler")
        )
        
        self.visualizer = MonitoringVisualizer(
            config=config,
            logger=self.monitoring_logger
        )
        
        self.hooks_manager = MonitoringHooks(
            data_monitor=self.data_monitor,
            system_monitor=self.system_monitor,
            logger=self.logger
        )
        
        self.integration = MonitoringIntegration(config)
        
        self.monitoring_thread = None
        self.stop_event = threading.Event()
        self.alert_handlers = {}
        self.metrics_cache = {}
        
        self.logger.info("ComprehensiveMonitor initialized successfully")

    def start_monitoring(self, 
                        trainer: Optional[Any] = None, 
                        model: Optional[Any] = None,
                        dataloader: Optional[Any] = None) -> None:
        if self.state.is_active:
            self.logger.warning("Monitoring is already active")
            return
        
        self.state.is_active = True
        self.state.start_time = datetime.utcnow()
        self.stop_event.clear()
        
        self.system_monitor.start_monitoring()
        self.training_monitor.start_training_session()
        
        if trainer:
            self.integration.integrate_with_trainer(trainer)
            
        if model:
            self.integration.integrate_with_model(model)
            
        if dataloader:
            self.integration.integrate_with_dataloader(dataloader)
        
        self.integration.start()
        
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Comprehensive monitoring started")
        self.monitoring_logger.log_system_metrics({"event": "monitoring_started"})

    def stop_monitoring(self) -> Dict[str, Any]:
        if not self.state.is_active:
            self.logger.warning("Monitoring is not active")
            return {}
        
        self.state.is_active = False
        self.state.end_time = datetime.utcnow()
        
        if self.state.start_time:
            self.state.total_duration = (
                self.state.end_time - self.state.start_time
            ).total_seconds()
        
        self.stop_event.set()
        
        self.system_monitor.stop_monitoring()
        self.integration.stop()
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
            
        self.hooks_manager.remove_all_hooks()
        self.data_monitor.cleanup_hooks()
        
        report = self._generate_final_report()
        
        self.logger.info("Comprehensive monitoring stopped")
        self.monitoring_logger.log_system_metrics({
            "event": "monitoring_stopped",
            "duration": self.state.total_duration
        })
        
        return report

    def _monitoring_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                self._collect_metrics()
                self._check_alerts()
                self._update_cache()
                time.sleep(self.config.system_monitoring_interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                self.monitoring_logger.log_error_event(e, {"context": "monitoring_loop"})
                time.sleep(self.config.system_monitoring_interval)

    def _collect_metrics(self) -> None:
        system_metrics = self.system_monitor.get_current_metrics()
        data_summary = self.data_monitor.get_data_summary()
        training_summary = self.training_monitor.get_training_summary()
        
        self.state.collected_metrics["system"] = self.state.collected_metrics.get("system", [])
        self.state.collected_metrics["system"].append(system_metrics)
        
        self.state.collected_metrics["data"] = self.state.collected_metrics.get("data", [])
        self.state.collected_metrics["data"].append(data_summary)
        
        self.state.collected_metrics["training"] = self.state.collected_metrics.get("training", [])
        self.state.collected_metrics["training"].append(training_summary)
        
        for key in ["system", "data", "training"]:
            if len(self.state.collected_metrics[key]) > self.config.history_size:
                self.state.collected_metrics[key] = self.state.collected_metrics[key][-self.config.history_size:]

    def _check_alerts(self) -> None:
        if not self.config.enable_alerts:
            return
            
        alert_thresholds = self.config.get_alert_thresholds()
        
        if self.state.collected_metrics.get("system"):
            latest_system = self.state.collected_metrics["system"][-1]
            
            if latest_system.cpu_percent > alert_thresholds["cpu_usage"]:
                alert_msg = f"High CPU usage: {latest_system.cpu_percent:.1f}%"
                self._trigger_alert("cpu_usage", alert_msg)
                
            if latest_system.memory_percent > alert_thresholds["memory_usage"]:
                alert_msg = f"High memory usage: {latest_system.memory_percent:.1f}%"
                self._trigger_alert("memory_usage", alert_msg)
                
            if latest_system.gpu_utilization:
                max_gpu_usage = max(latest_system.gpu_utilization)
                if max_gpu_usage > alert_thresholds["gpu_usage"]:
                    alert_msg = f"High GPU usage: {max_gpu_usage:.1f}%"
                    self._trigger_alert("gpu_usage", alert_msg)
                    
            if latest_system.gpu_temperature:
                max_gpu_temp = max(latest_system.gpu_temperature)
                if max_gpu_temp > alert_thresholds["gpu_temperature"]:
                    alert_msg = f"High GPU temperature: {max_gpu_temp:.1f}°C"
                    self._trigger_alert("gpu_temperature", alert_msg)

    def _trigger_alert(self, alert_type: str, message: str) -> None:
        if alert_type not in self.state.alerts_triggered:
            self.state.alerts_triggered.append(alert_type)
            
        self.logger.warning(f"Alert: {message}")
        self.monitoring_logger.log_alert(alert_type, message)
        
        if alert_type in self.alert_handlers:
            try:
                self.alert_handlers[alert_type](message)
            except Exception as e:
                self.logger.error(f"Error in alert handler for {alert_type}: {e}")

    def _update_cache(self) -> None:
        self.metrics_cache = {
            "system": self.state.collected_metrics.get("system", [])[-10:],
            "data": self.state.collected_metrics.get("data", [])[-10:],
            "training": self.state.collected_metrics.get("training", [])[-10:],
            "timestamp": datetime.utcnow()
        }

    def _generate_final_report(self) -> Dict[str, Any]:
        report = {
            "monitoring_session": {
                "duration": self.state.total_duration,
                "start_time": self.state.start_time,
                "end_time": self.state.end_time,
                "total_alerts": len(self.state.alerts_triggered),
                "alert_types": list(set(self.state.alerts_triggered))
            },
            "metrics_collected": {
                k: len(v) for k, v in self.state.collected_metrics.items()
            },
            "system_summary": self.system_monitor.get_performance_summary(),
            "training_summary": self.training_monitor.get_training_summary(),
            "data_summary": self.data_monitor.get_data_summary(),
            "profiling_summary": self.profiler.generate_performance_report() if self.config.profiling_config.enable_profiling else None
        }
        
        if self.config.enable_visualization:
            dashboard_path = self.visualizer.generate_comprehensive_report(
                self.state.collected_metrics.get("system", []),
                self.training_monitor,
                self.state.collected_metrics.get("data", [])
            )
            report["dashboard_path"] = dashboard_path
        
        self.monitoring_logger.log_system_metrics({"event": "final_report", "report": report})
        return report

    def get_real_time_status(self) -> Dict[str, Any]:
        return {
            "is_active": self.state.is_active,
            "start_time": self.state.start_time,
            "duration": (
                datetime.utcnow() - self.state.start_time
            ).total_seconds() if self.state.start_time else 0,
            "current_metrics": self.metrics_cache,
            "alerts_count": len(self.state.alerts_triggered),
            "recent_alerts": self.state.alerts_triggered[-5:] if self.state.alerts_triggered else [],
            "components_status": {
                "system_monitor": self.system_monitor.monitoring_thread.is_alive() if self.system_monitor.monitoring_thread else False,
                "data_monitor": len(self.data_monitor.hooks),
                "training_monitor": len(self.training_monitor.training_history),
                "profiler": self.profiler.step_counter,
                "visualizer": self.config.enable_visualization
            }
        }

    def register_alert_handler(self, alert_type: str, handler: Callable) -> None:
        self.alert_handlers[alert_type] = handler
        self.logger.info(f"Alert handler registered for {alert_type}")

    def update_config(self, config_updates: Dict[str, Any]) -> None:
        for key, value in config_updates.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.info(f"Configuration updated: {key} = {value}")
            else:
                self.logger.warning(f"Invalid configuration key: {key}")

    def export_metrics(self, filepath: str, format: str = "json") -> bool:
        try:
            if format == "json":
                import json
                with open(filepath, 'w') as f:
                    json.dump(self.state.collected_metrics, f, indent=2, default=str)
            elif format == "csv":
                import pandas as pd
                df = pd.DataFrame(self.state.collected_metrics)
                df.to_csv(filepath, index=False)
            else:
                self.logger.error(f"Unsupported export format: {format}")
                return False
                
            self.logger.info(f"Metrics exported to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error exporting metrics: {e}")
            return False

    def create_monitoring_dashboard(self) -> str:
        if not self.config.enable_visualization:
            self.logger.warning("Visualization is disabled")
            return ""
            
        return self.visualizer.create_interactive_dashboard(
            self.state.collected_metrics.get("system", []),
            self.training_monitor
        )

    def profile_training_step(self, model, loss_fn, optimizer, input_data, targets) -> Dict[str, Any]:
        if not self.config.profiling_config.enable_profiling:
            self.logger.warning("Profiling is disabled")
            return {}
            
        return self.profiler.profile_training_step(model, loss_fn, optimizer, input_data, targets)

    def cleanup(self) -> None:
        if self.state.is_active:
            self.stop_monitoring()
            
        self.visualizer.cleanup_old_plots()
        self.system_monitor.clear_history()
        self.data_monitor.reset_monitoring()
        self.training_monitor.reset_monitoring()
        self.profiler.reset_profiler()
        
        self.logger.info("Monitoring system cleaned up")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()