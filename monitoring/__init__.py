#FILE: monitoring/__init__.py
# ehsanasgharzde - MAIN MONITORING CLASS
# hosseinsolymanzadeh - PROPER COMMENTING

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
        # Store configuration object
        self.config = config
        # Initialize logger for this module
        self.logger = logging.getLogger(__name__)
        # Initialize the current monitoring state
        self.state = MonitoringState()
        
        # Set up a dedicated logger for monitoring with given directory, level, and structured logging
        self.monitoring_logger = MonitoringLogger(
            log_dir=config.logging_config.log_dir,
            log_level=getattr(logging, config.logging_config.log_level.upper()),
            structured_logging=config.logging_config.structured_logging
        )
        
        # Initialize data monitor to track gradients and activations based on config with a sample rate and log level
        self.data_monitor = DataMonitor(
            track_gradients=config.enable_gradient_monitoring,
            track_activations=config.enable_activation_monitoring,
            sample_rate=config.data_monitoring_sample_rate,
            log_level=getattr(logging, config.logging_config.log_level.upper())
        )
        
        # Initialize system monitor with interval and history size
        self.system_monitor = SystemMonitor(
            monitoring_interval=config.system_monitoring_interval,
            history_size=config.history_size
        )
        
        # Initialize training monitor with log/checkpoint intervals, logger, and config
        self.training_monitor = TrainingMonitor(
            log_interval=config.training_log_interval,
            checkpoint_interval=config.training_checkpoint_interval,
            logger=self.logger,
            config=config
        )
        
        # Initialize performance profiler with memory and CUDA profiling flags, log interval, and output directory
        self.profiler = PerformanceProfiler(
            profile_memory=config.profiling_config.profile_memory,
            profile_cuda=config.profiling_config.profile_cuda,
            log_interval=config.profiling_config.profiling_interval,
            output_dir=str(Path(config.output_directory) / "profiler")
        )
        
        # Initialize visualizer with config and monitoring logger
        self.visualizer = MonitoringVisualizer(
            config=config,
            logger=self.monitoring_logger
        )
        
        # Set up hooks manager to coordinate data and system monitors with logging
        self.hooks_manager = MonitoringHooks(
            data_monitor=self.data_monitor,
            system_monitor=self.system_monitor,
            logger=self.logger
        )
        
        # Initialize integration module with config to integrate monitoring into training workflow
        self.integration = MonitoringIntegration(config)
        
        # Placeholder for the monitoring background thread
        self.monitoring_thread = None
        # Event to signal stopping of monitoring thread
        self.stop_event = threading.Event()
        # Dictionary for alert handlers keyed by alert type
        self.alert_handlers = {}
        # Cache dictionary to store metrics temporarily
        self.metrics_cache = {}
        
        # Log that initialization completed successfully
        self.logger.info("ComprehensiveMonitor initialized successfully")

    def start_monitoring(self, 
                        trainer: Optional[Any] = None, 
                        model: Optional[Any] = None,
                        dataloader: Optional[Any] = None) -> None:
        # Prevent starting monitoring if it's already active
        if self.state.is_active:
            self.logger.warning("Monitoring is already active")
            return
        
        # Mark monitoring as active
        self.state.is_active = True
        # Record start time in UTC
        self.state.start_time = datetime.utcnow()
        # Clear any previous stop event
        self.stop_event.clear()
        
        # Start system resource monitoring
        self.system_monitor.start_monitoring()
        # Start tracking training session metrics
        self.training_monitor.start_training_session()
        
        # Integrate monitoring with trainer object if provided
        if trainer:
            self.integration.integrate_with_trainer(trainer)
            
        # Integrate monitoring with model object if provided
        if model:
            self.integration.integrate_with_model(model)
            
        # Integrate monitoring with dataloader object if provided
        if dataloader:
            self.integration.integrate_with_dataloader(dataloader)
        
        # Start integration mechanisms (e.g., hooks, callbacks)
        self.integration.start()
        
        # Launch monitoring loop in a daemon thread for background execution
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        # Log that monitoring has started
        self.logger.info("Comprehensive monitoring started")
        # Log the event in the monitoring logger for system-wide metrics tracking
        self.monitoring_logger.log_system_metrics({"event": "monitoring_started"})

    def stop_monitoring(self) -> Dict[str, Any]:
        # Check if monitoring is currently inactive; if so, log warning and return empty report
        if not self.state.is_active:
            self.logger.warning("Monitoring is not active")
            return {}
        
        # Mark monitoring as inactive
        self.state.is_active = False
        # Record the monitoring end time
        self.state.end_time = datetime.utcnow()
        
        # Calculate total monitoring duration if start time exists
        if self.state.start_time:
            self.state.total_duration = (
                self.state.end_time - self.state.start_time
            ).total_seconds()
        
        # Signal the monitoring loop to stop
        self.stop_event.set()
        
        # Stop system monitoring processes
        self.system_monitor.stop_monitoring()
        # Stop integration processes like hooks or callbacks
        self.integration.stop()
        
        # Wait for monitoring thread to finish with a timeout of 5 seconds
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
            
        # Remove all registered hooks from hooks manager
        self.hooks_manager.remove_all_hooks()
        # Cleanup hooks related to data monitoring
        self.data_monitor.cleanup_hooks()
        
        # Generate a final report summarizing collected metrics and results
        report = self._generate_final_report()
        
        # Log that monitoring has been stopped
        self.logger.info("Comprehensive monitoring stopped")
        # Log monitoring stopped event with duration to monitoring logger
        self.monitoring_logger.log_system_metrics({
            "event": "monitoring_stopped",
            "duration": self.state.total_duration
        })
        
        # Return the final monitoring report
        return report

    def _monitoring_loop(self) -> None:
        # Continuously run until stop event is set
        while not self.stop_event.is_set():
            try:
                # Collect current metrics from all monitors
                self._collect_metrics()
                # Check if any alerts need to be triggered based on metrics
                self._check_alerts()
                # Update internal cache with latest metrics
                self._update_cache()
                # Sleep for configured system monitoring interval before next iteration
                time.sleep(self.config.system_monitoring_interval)
            except Exception as e:
                # Log any errors encountered during the loop
                self.logger.error(f"Error in monitoring loop: {e}")
                # Log error event to monitoring logger with context
                self.monitoring_logger.log_error_event(e, {"context": "monitoring_loop"})
                # Sleep to avoid rapid retry in case of persistent errors
                time.sleep(self.config.system_monitoring_interval)

    def _collect_metrics(self) -> None:
        # Get current system metrics from system monitor
        system_metrics = self.system_monitor.get_current_metrics()
        # Get summarized data metrics (e.g., activations, gradients) from data monitor
        data_summary = self.data_monitor.get_data_summary()
        # Get current training session summary metrics from training monitor
        training_summary = self.training_monitor.get_training_summary()
        
        # Initialize or retrieve list for system metrics collection and append latest
        self.state.collected_metrics["system"] = self.state.collected_metrics.get("system", [])
        self.state.collected_metrics["system"].append(system_metrics)
        
        # Initialize or retrieve list for data metrics collection and append latest
        self.state.collected_metrics["data"] = self.state.collected_metrics.get("data", [])
        self.state.collected_metrics["data"].append(data_summary)
        
        # Initialize or retrieve list for training metrics collection and append latest
        self.state.collected_metrics["training"] = self.state.collected_metrics.get("training", [])
        self.state.collected_metrics["training"].append(training_summary)
        
        # Trim collected metrics history to configured maximum size to limit memory use
        for key in ["system", "data", "training"]:
            if len(self.state.collected_metrics[key]) > self.config.history_size:
                self.state.collected_metrics[key] = self.state.collected_metrics[key][-self.config.history_size:]

    def _check_alerts(self) -> None:
        # Return early if alerting is disabled in configuration
        if not self.config.enable_alerts:
            return
            
        # Retrieve alert threshold values from configuration
        alert_thresholds = self.config.get_alert_thresholds()
        
        # Check if there are any system metrics collected to evaluate
        if self.state.collected_metrics.get("system"):
            # Get the latest system metrics entry
            latest_system = self.state.collected_metrics["system"][-1]
            
            # Check CPU usage against threshold and trigger alert if exceeded
            if latest_system.cpu_percent > alert_thresholds["cpu_usage"]:
                alert_msg = f"High CPU usage: {latest_system.cpu_percent:.1f}%"
                self._trigger_alert("cpu_usage", alert_msg)
                
            # Check memory usage against threshold and trigger alert if exceeded
            if latest_system.memory_percent > alert_thresholds["memory_usage"]:
                alert_msg = f"High memory usage: {latest_system.memory_percent:.1f}%"
                self._trigger_alert("memory_usage", alert_msg)
                
            # If GPU utilization data exists, check max usage and trigger alert if exceeded
            if latest_system.gpu_utilization:
                max_gpu_usage = max(latest_system.gpu_utilization)
                if max_gpu_usage > alert_thresholds["gpu_usage"]:
                    alert_msg = f"High GPU usage: {max_gpu_usage:.1f}%"
                    self._trigger_alert("gpu_usage", alert_msg)
                    
            # If GPU temperature data exists, check max temperature and trigger alert if exceeded
            if latest_system.gpu_temperature:
                max_gpu_temp = max(latest_system.gpu_temperature)
                if max_gpu_temp > alert_thresholds["gpu_temperature"]:
                    alert_msg = f"High GPU temperature: {max_gpu_temp:.1f}°C"
                    self._trigger_alert("gpu_temperature", alert_msg)

    def _trigger_alert(self, alert_type: str, message: str) -> None:
        # Track that this alert type has been triggered (avoid duplicates if needed)
        if alert_type not in self.state.alerts_triggered:
            self.state.alerts_triggered.append(alert_type)
            
        # Log a warning about the alert
        self.logger.warning(f"Alert: {message}")
        # Log the alert through the monitoring logger system
        self.monitoring_logger.log_alert(alert_type, message)
        
        # If a handler function exists for this alert type, execute it safely
        if alert_type in self.alert_handlers:
            try:
                self.alert_handlers[alert_type](message)
            except Exception as e:
                # Log any error raised by the alert handler
                self.logger.error(f"Error in alert handler for {alert_type}: {e}")

    def _update_cache(self) -> None:
        # Update the metrics cache with the last 10 entries from each metric category and current timestamp
        self.metrics_cache = {
            "system": self.state.collected_metrics.get("system", [])[-10:],
            "data": self.state.collected_metrics.get("data", [])[-10:],
            "training": self.state.collected_metrics.get("training", [])[-10:],
            "timestamp": datetime.utcnow()
        }

    def _generate_final_report(self) -> Dict[str, Any]:
        # Construct a dictionary report summarizing the monitoring session and collected data
        report = {
            "monitoring_session": {
                "duration": self.state.total_duration,
                "start_time": self.state.start_time,
                "end_time": self.state.end_time,
                "total_alerts": len(self.state.alerts_triggered),
                "alert_types": list(set(self.state.alerts_triggered))
            },
            "metrics_collected": {
                # Report the count of collected metrics per category
                k: len(v) for k, v in self.state.collected_metrics.items()
            },
            # Include summaries generated by system, training, and data monitors
            "system_summary": self.system_monitor.get_performance_summary(),
            "training_summary": self.training_monitor.get_training_summary(),
            "data_summary": self.data_monitor.get_data_summary(),
            # Include profiling summary if profiling is enabled in config
            "profiling_summary": self.profiler.generate_performance_report() if self.config.profiling_config.enable_profiling else None
        }
        
        # If visualization is enabled, generate and include a dashboard report path
        if self.config.enable_visualization:
            dashboard_path = self.visualizer.generate_comprehensive_report(
                self.state.collected_metrics.get("system", []),
                self.training_monitor,
                self.state.collected_metrics.get("data", [])
            )
            report["dashboard_path"] = dashboard_path
        
        # Log the final report event and data to the monitoring logger
        self.monitoring_logger.log_system_metrics({"event": "final_report", "report": report})
        # Return the constructed report dictionary
        return report

    def get_real_time_status(self) -> Dict[str, Any]:
        # Return a dictionary with the current real-time monitoring status and metrics
        return {
            "is_active": self.state.is_active,
            "start_time": self.state.start_time,
            # Calculate current duration since start, or 0 if not started
            "duration": (
                datetime.utcnow() - self.state.start_time
            ).total_seconds() if self.state.start_time else 0,
            # Include the cached recent metrics snapshots
            "current_metrics": self.metrics_cache,
            # Number of alerts triggered so far
            "alerts_count": len(self.state.alerts_triggered),
            # List the last 5 triggered alerts if any
            "recent_alerts": self.state.alerts_triggered[-5:] if self.state.alerts_triggered else [],
            # Status info of main monitoring components
            "components_status": {
                # Whether the system monitor's internal thread is alive
                "system_monitor": self.system_monitor.monitoring_thread.is_alive() if self.system_monitor.monitoring_thread else False,
                # Count of hooks registered in data monitor
                "data_monitor": len(self.data_monitor.hooks),
                # Count of training history entries in training monitor
                "training_monitor": len(self.training_monitor.training_history),
                # Number of profiling steps recorded by profiler
                "profiler": self.profiler.step_counter,
                # Whether visualization feature is enabled
                "visualizer": self.config.enable_visualization
            }
        }

    def register_alert_handler(self, alert_type: str, handler: Callable) -> None:
        # Register a callback function to handle a specific alert type
        self.alert_handlers[alert_type] = handler
        # Log that the alert handler was successfully registered
        self.logger.info(f"Alert handler registered for {alert_type}")

    def update_config(self, config_updates: Dict[str, Any]) -> None:
        # Update monitoring configuration with provided key-value pairs
        for key, value in config_updates.items():
            if hasattr(self.config, key):
                # Update the attribute on the config object
                setattr(self.config, key, value)
                self.logger.info(f"Configuration updated: {key} = {value}")
            else:
                # Warn if an invalid or unknown config key was provided
                self.logger.warning(f"Invalid configuration key: {key}")

    def export_metrics(self, filepath: str, format: str = "json") -> bool:
        try:
            # Export collected metrics to JSON file
            if format == "json":
                import json
                with open(filepath, 'w') as f:
                    json.dump(self.state.collected_metrics, f, indent=2, default=str)
            # Export collected metrics to CSV file using pandas
            elif format == "csv":
                import pandas as pd
                df = pd.DataFrame(self.state.collected_metrics)
                df.to_csv(filepath, index=False)
            else:
                # Log error for unsupported export format and return failure
                self.logger.error(f"Unsupported export format: {format}")
                return False
                
            # Log successful export and return success
            self.logger.info(f"Metrics exported to {filepath}")
            return True
        except Exception as e:
            # Log exception on failure to export and return failure
            self.logger.error(f"Error exporting metrics: {e}")
            return False

    def create_monitoring_dashboard(self) -> str:
        # Check if visualization feature is enabled before proceeding
        if not self.config.enable_visualization:
            self.logger.warning("Visualization is disabled")
            return ""
            
        # Generate and return the path or URL to an interactive monitoring dashboard
        return self.visualizer.create_interactive_dashboard(
            self.state.collected_metrics.get("system", []),
            self.training_monitor
        )

    def profile_training_step(self, model, loss_fn, optimizer, input_data, targets) -> Dict[str, Any]:
        # Return empty dict and log warning if profiling is disabled
        if not self.config.profiling_config.enable_profiling:
            self.logger.warning("Profiling is disabled")
            return {}
            
        # Use the profiler to profile a single training step and return results
        return self.profiler.profile_training_step(model, loss_fn, optimizer, input_data, targets)

    def cleanup(self) -> None:
        # If monitoring is active, stop it first to ensure clean shutdown
        if self.state.is_active:
            self.stop_monitoring()
            
        # Clean up visualization artifacts (e.g., old plots)
        self.visualizer.cleanup_old_plots()
        # Clear system monitor's stored history
        self.system_monitor.clear_history()
        # Reset data monitor state and hooks
        self.data_monitor.reset_monitoring()
        # Reset training monitor state and history
        self.training_monitor.reset_monitoring()
        # Reset profiler internal state and counters
        self.profiler.reset_profiler()
        
        # Log that cleanup completed successfully
        self.logger.info("Monitoring system cleaned up")

    def __enter__(self):
        # Context manager enter method returns self for 'with' statements
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # On exiting context, perform cleanup to free resources
        self.cleanup()
