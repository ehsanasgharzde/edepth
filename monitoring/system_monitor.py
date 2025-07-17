#FILE: monitoring/system_monitor.py
# ehsanasgharzde - SYSTEM MONITOR
# hosseinsolymanzadeh - PROPER COMMENTING

import psutil
import GPUtil #type: ignore 
import torch
import threading
import time
import logging
from typing import Dict, List, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import pickle
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    # Timestamp of metric collection (default: current UTC time)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    # CPU usage percentage
    cpu_percent: float = 0.0
    # Memory usage percentage
    memory_percent: float = 0.0
    # Available memory in MB
    memory_available: float = 0.0
    # List of GPU utilization percentages per GPU
    gpu_utilization: List[float] = field(default_factory=list)
    # List of GPU memory used in MB per GPU
    gpu_memory_used: List[float] = field(default_factory=list)
    # List of total GPU memory in MB per GPU
    gpu_memory_total: List[float] = field(default_factory=list)
    # List of GPU temperatures per GPU in Celsius
    gpu_temperature: List[float] = field(default_factory=list)
    # Disk usage percentage for root partition
    disk_usage: float = 0.0
    # Network I/O rates: bytes sent and received per second
    network_io: Dict[str, float] = field(default_factory=lambda: {'bytes_sent': 0.0, 'bytes_recv': 0.0})
    # Number of running processes
    process_count: int = 0
    # System load averages over 1, 5, and 15 minutes
    load_average: List[float] = field(default_factory=list)

class SystemMonitor:
    def __init__(self, monitoring_interval: float = 1.0, history_size: int = 1000):
        # Interval in seconds between metric collections
        self.monitoring_interval = monitoring_interval
        # Maximum number of metrics to keep in history
        self.history_size = history_size
        # Deque to store history of collected metrics, with max length
        self.metrics_history = deque(maxlen=history_size)
        # Thread that runs monitoring loop
        self.monitoring_thread = None
        # Event to signal monitoring thread to stop
        self._stop_event = threading.Event()
        # Thresholds for triggering alerts on system resources
        self.alert_thresholds = {
            'cpu': 90.0,       # CPU usage percent
            'memory': 90.0,    # Memory usage percent
            'gpu_util': 90.0,  # GPU utilization percent
            'gpu_temp': 80.0,  # GPU temperature in Celsius
            'disk': 95.0,      # Disk usage percent
        }
        # Initial network counters for rate calculation
        self._initial_net = psutil.net_io_counters()
        # Initial timestamp for network rate calculation
        self._initial_time = time.time()

    def start_monitoring(self):
        # Start the monitoring thread if not already running
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            self._stop_event.clear()  # Clear stop signal
            # Create and start daemon thread running the monitor loop
            self.monitoring_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitoring_thread.start()
            logger.info("System monitoring started")

    def stop_monitoring(self):
        # Signal the monitoring thread to stop
        self._stop_event.set()
        if self.monitoring_thread:
            # Wait for the thread to finish
            self.monitoring_thread.join()
            logger.info("System monitoring stopped")

    def _monitor_loop(self):
        # Loop that collects metrics periodically until stopped
        while not self._stop_event.is_set():
            try:
                # Collect current system metrics
                metrics = self.collect_system_metrics()
                # Append metrics to history deque
                self.metrics_history.append(metrics)
                # Check if any resource usage exceeds alert thresholds
                alerts = self.check_resource_alerts(metrics)
                if alerts:
                    # Log warnings for each triggered alert
                    for alert in alerts:
                        logger.warning(f"Resource alert: {alert}")
                # Sleep until next collection
                time.sleep(self.monitoring_interval)
            except Exception as e:
                # Log any errors in the monitoring loop and continue
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)

    def collect_system_metrics(self) -> SystemMetrics:
        # Get current UTC timestamp
        timestamp = datetime.utcnow()
        
        try:
            # Get CPU usage percentage instantly
            cpu_percent = psutil.cpu_percent(interval=None)
            # Get system load averages (1, 5, 15 minutes)
            load_avg = list(psutil.getloadavg())
            
            # Get memory stats
            mem = psutil.virtual_memory()
            memory_percent = mem.percent
            # Convert available memory bytes to megabytes
            memory_available = mem.available / (1024 ** 2)
            
            # Get disk usage percentage for root '/'
            disk = psutil.disk_usage('/')
            disk_usage = disk.percent
            
            # Get network I/O counters
            net_io = psutil.net_io_counters()
            now = time.time()
            # Calculate elapsed time since last network measurement
            time_diff = now - self._initial_time
            if time_diff > 0:
                # Calculate bytes sent and received per second
                net_sent_rate = (net_io.bytes_sent - self._initial_net.bytes_sent) / time_diff
                net_recv_rate = (net_io.bytes_recv - self._initial_net.bytes_recv) / time_diff
            else:
                net_sent_rate = net_recv_rate = 0.0
            
            # Store network I/O rates
            network_io = {'bytes_sent': net_sent_rate, 'bytes_recv': net_recv_rate}
            # Update initial counters for next calculation
            self._initial_net = net_io
            self._initial_time = now
            
            # Count the number of running processes
            process_count = len(psutil.pids())
            
            # Initialize GPU metrics lists
            gpu_util, gpu_mem_used, gpu_mem_total, gpu_temp = [], [], [], []
            try:
                # Try to get GPU stats using GPUtil
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    # Append GPU utilization percentage
                    gpu_util.append(gpu.load * 100)
                    # Append GPU memory used in MB
                    gpu_mem_used.append(gpu.memoryUsed)
                    # Append total GPU memory in MB
                    gpu_mem_total.append(gpu.memoryTotal)
                    # Append GPU temperature in Celsius
                    gpu_temp.append(gpu.temperature)
            except Exception as e:
                # GPU monitoring not available or failed
                logger.debug(f"GPU monitoring not available: {e}")
            
            # Return a SystemMetrics instance with all collected data
            return SystemMetrics(
                timestamp=timestamp,
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_available=memory_available,
                gpu_utilization=gpu_util,
                gpu_memory_used=gpu_mem_used,
                gpu_memory_total=gpu_mem_total,
                gpu_temperature=gpu_temp,
                disk_usage=disk_usage,
                network_io=network_io,
                process_count=process_count,
                load_average=load_avg
            )
        except Exception as e:
            # Log error and return metrics with just the timestamp
            logger.error(f"Error collecting system metrics: {e}")
            return SystemMetrics(timestamp=timestamp)

    def get_current_metrics(self) -> SystemMetrics:
        # Retrieve current system metrics by collecting fresh data
        return self.collect_system_metrics()

    def get_gpu_metrics(self) -> Dict[str, Any]:
        try:
            # Attempt to get detailed GPU metrics using GPUtil
            gpus = GPUtil.getGPUs()
            gpu_data = {}
            for gpu in gpus:
                # Store utilization, memory usage, temperature, and name for each GPU
                gpu_data[f'GPU_{gpu.id}'] = {
                    'utilization': gpu.load * 100,            # GPU load in percent
                    'memory_used_MB': gpu.memoryUsed,         # Used memory in MB
                    'memory_total_MB': gpu.memoryTotal,       # Total memory in MB
                    'temperature_C': gpu.temperature,         # Temperature in Celsius
                    'name': gpu.name,                         # GPU model/name
                }
            return gpu_data
        except Exception as e:
            # Log error and return fallback message if GPU monitoring fails
            logger.error(f"Error getting GPU metrics: {e}")
            return {'error': 'GPU monitoring not available'}

    def get_memory_breakdown(self) -> Dict[str, Any]:
        try:
            # Get system memory statistics
            mem = psutil.virtual_memory()
            # Get GPU stats from previous method
            gpu_stats = self.get_gpu_metrics()

            # Dictionary to store PyTorch GPU memory usage per device
            pytorch_gpu_mem = {}
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    try:
                        # Set the current CUDA device
                        torch.cuda.set_device(i)
                        # Record PyTorch allocated and reserved memory in MB for device i
                        pytorch_gpu_mem[f'GPU_{i}'] = {
                            'torch_allocated_MB': torch.cuda.memory_allocated(i) / (1024 ** 2),
                            'torch_reserved_MB': torch.cuda.memory_reserved(i) / (1024 ** 2)
                        }
                    except Exception as e:
                        # Log errors getting PyTorch GPU memory for each device
                        logger.error(f"Error getting PyTorch GPU memory for device {i}: {e}")

            # Return combined system and GPU memory breakdown
            return {
                'system_memory_percent': mem.percent,                  # System memory usage percent
                'available_MB': mem.available / (1024 ** 2),           # Available system memory MB
                'used_MB': mem.used / (1024 ** 2),                     # Used system memory MB
                'total_MB': mem.total / (1024 ** 2),                   # Total system memory MB
                'pytorch_gpu_memory': pytorch_gpu_mem,                 # PyTorch GPU memory usage
                'gpu_memory': gpu_stats                                 # GPU stats from GPUtil
            }
        except Exception as e:
            # Log error and return fallback message if memory monitoring fails
            logger.error(f"Error getting memory breakdown: {e}")
            return {'error': 'Memory monitoring failed'}

    def check_resource_alerts(self, metrics: SystemMetrics) -> List[str]:
        alerts = []

        try:
            # Check if CPU usage exceeds threshold
            if metrics.cpu_percent > self.alert_thresholds['cpu']:
                alerts.append(f"High CPU usage: {metrics.cpu_percent:.2f}%")

            # Check if memory usage exceeds threshold
            if metrics.memory_percent > self.alert_thresholds['memory']:
                alerts.append(f"High memory usage: {metrics.memory_percent:.2f}%")

            # Check if disk usage exceeds threshold
            if metrics.disk_usage > self.alert_thresholds['disk']:
                alerts.append(f"Low disk space: {metrics.disk_usage:.2f}% used")

            # Check GPU utilizations against threshold for each GPU
            for i, gpu_util in enumerate(metrics.gpu_utilization):
                if gpu_util > self.alert_thresholds['gpu_util']:
                    alerts.append(f"High GPU {i} utilization: {gpu_util:.2f}%")

            # Check GPU temperatures against threshold for each GPU
            for i, temp in enumerate(metrics.gpu_temperature):
                if temp > self.alert_thresholds['gpu_temp']:
                    alerts.append(f"High GPU {i} temperature: {temp:.2f}C")
        except Exception as e:
            # Log any errors during alert checking
            logger.error(f"Error checking resource alerts: {e}")

        # Return list of alert messages triggered
        return alerts

    def get_performance_summary(self, time_window: int = 300) -> Dict[str, Any]:
        # Get current UTC time
        try:
            now = datetime.utcnow()
            # Filter metrics within the time window (last `time_window` seconds)
            filtered_metrics = [m for m in self.metrics_history if (now - m.timestamp).total_seconds() <= time_window]
            
            # Return warning if no metrics in the window
            if not filtered_metrics:
                return {"warning": "No data in selected time window"}
            
            # Helper function to safely calculate average of a field
            def safe_avg(field):
                try:
                    # Extract field values if the metric has that attribute
                    values = [getattr(m, field) for m in filtered_metrics if hasattr(m, field)]
                    # Compute average or return 0.0 if no values
                    return sum(values) / len(values) if values else 0.0
                except:
                    # In case of any error, return 0.0
                    return 0.0
            
            # Prepare the summary dictionary with averages and counts
            summary = {
                'time_window_seconds': time_window,
                'samples_count': len(filtered_metrics),
                'avg_cpu_percent': safe_avg('cpu_percent'),
                'avg_memory_percent': safe_avg('memory_percent'),
                'avg_memory_available_MB': safe_avg('memory_available'),
                'avg_disk_usage_percent': safe_avg('disk_usage'),
                'avg_process_count': safe_avg('process_count'),
            }
            
            # If GPU utilization data is available, calculate its average
            if filtered_metrics[0].gpu_utilization:
                gpu_utils = []
                for m in filtered_metrics:
                    if m.gpu_utilization:
                        # Average GPU utilization across all GPUs for this sample
                        gpu_utils.append(sum(m.gpu_utilization) / len(m.gpu_utilization))
                # Average GPU utilization across all filtered samples or 0.0 if none
                summary['avg_gpu_utilization_percent'] = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0.0
            
            # Return the performance summary dictionary
            return summary
        except Exception as e:
            # Log error and return error message
            logger.error(f"Error getting performance summary: {e}")
            return {"error": "Failed to generate performance summary"}

    def get_metrics_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        try:
            # Get the most recent `limit` metrics from history
            recent_metrics = list(self.metrics_history)[-limit:]
            # Convert each metric object to dict representation
            return [self._metrics_to_dict(m) for m in recent_metrics]
        except Exception as e:
            # Log error and return empty list on failure
            logger.error(f"Error getting metrics history: {e}")
            return []

    def _metrics_to_dict(self, metrics: SystemMetrics) -> Dict[str, Any]:
        # Convert a SystemMetrics instance to a dictionary of relevant fields
        return {
            'timestamp': metrics.timestamp.isoformat(),
            'cpu_percent': metrics.cpu_percent,
            'memory_percent': metrics.memory_percent,
            'memory_available': metrics.memory_available,
            'gpu_utilization': metrics.gpu_utilization,
            'gpu_memory_used': metrics.gpu_memory_used,
            'gpu_memory_total': metrics.gpu_memory_total,
            'gpu_temperature': metrics.gpu_temperature,
            'disk_usage': metrics.disk_usage,
            'network_io': metrics.network_io,
            'process_count': metrics.process_count,
            'load_average': metrics.load_average
        }

    def export_metrics(self, filename: str, format: str = 'json') -> bool:
        try:
            # Retrieve all metrics data
            metrics_data = self.get_metrics_history()
            
            # Export metrics to requested file format
            if format == 'json':
                # Write as JSON file with indentation
                with open(filename, 'w') as f:
                    json.dump(metrics_data, f, indent=2)
            elif format == 'pickle':
                # Write as pickle binary file
                with open(filename, 'wb') as f:
                    pickle.dump(metrics_data, f)
            else:
                # Unsupported format error
                logger.error(f"Unsupported export format: {format}")
                return False
            
            # Log successful export
            logger.info(f"Metrics exported to {filename}")
            return True
        except Exception as e:
            # Log error and return failure
            logger.error(f"Error exporting metrics: {e}")
            return False

    def set_alert_thresholds(self, thresholds: Dict[str, float]):
        # Update alert thresholds dictionary with new values
        self.alert_thresholds.update(thresholds)
        # Log the update
        logger.info(f"Alert thresholds updated: {thresholds}")

    def clear_history(self):
        # Clear the metrics history list
        self.metrics_history.clear()
        # Log clearing of history
        logger.info("Metrics history cleared")

    def get_system_info(self) -> Dict[str, Any]:
        try:
            # Return a dictionary with basic system info and configuration
            return {
                'platform': psutil.WINDOWS if hasattr(psutil, 'WINDOWS') else 'linux',
                'cpu_count': psutil.cpu_count(),
                'memory_total_MB': psutil.virtual_memory().total / (1024 ** 2),
                'disk_total_GB': psutil.disk_usage('/').total / (1024 ** 3),
                'gpu_count': len(GPUtil.getGPUs()) if GPUtil else 0,
                'python_version': f"{torch.__version__}",
                'monitoring_interval': self.monitoring_interval,
                'history_size': self.history_size
            }
        except Exception as e:
            # Log error and return error message dictionary
            logger.error(f"Error getting system info: {e}")
            return {'error': 'Failed to get system info'}
