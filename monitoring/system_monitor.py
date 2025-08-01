# FILE: monitoring/system_monitor.py
# ehsanasgharzde - SYSTEM MONITOR

import psutil
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Callable
from collections import deque
from dataclasses import dataclass, field
import json

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from logger.logger import setup_logging 

# Setup logger for factory operations
logger = setup_logging(__file__)

@dataclass
class SystemMetrics:
    timestamp: datetime = field(default_factory=datetime.utcnow)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_available: float = 0.0
    disk_usage: float = 0.0
    gpu_utilization: List[float] = field(default_factory=list)
    gpu_memory_used: List[float] = field(default_factory=list)
    gpu_memory_total: List[float] = field(default_factory=list)
    gpu_temperature: List[float] = field(default_factory=list)
    network_io: Dict[str, float] = field(default_factory=dict)

class SystemResourceMonitor:
    def __init__(self, update_interval: float = 1.0, max_history: int = 1000):
        self.update_interval = update_interval
        self.max_history = max_history
        self.metrics_history = deque(maxlen=max_history)
        self.is_monitoring = False
        self.monitor_thread = None
        self.stop_event = threading.Event()

        # Alert thresholds
        self.thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_usage': 90.0,
            'gpu_utilization': 90.0,
            'gpu_temperature': 80.0
        }

        self.alert_callbacks = []

    def start_monitoring(self) -> None:
        if self.is_monitoring:
            logger.warning("Monitoring already started")
            return

        self.is_monitoring = True
        self.stop_event.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("System monitoring started")

    def stop_monitoring(self) -> None:
        if not self.is_monitoring:
            return

        self.is_monitoring = False
        self.stop_event.set()

        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)

        logger.info("System monitoring stopped")

    def _monitor_loop(self) -> None:
        while not self.stop_event.wait(self.update_interval):
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                self._check_alerts(metrics)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

    def _collect_metrics(self) -> SystemMetrics:
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # Network IO
        net_io = psutil.net_io_counters()
        network_io = {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv,
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv
        }

        metrics = SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_available=memory.available / (1024**3),  # GB
            disk_usage=disk.percent,
            network_io=network_io # type: ignore
        )

        # GPU metrics
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    metrics.gpu_utilization.append(gpu.load * 100)
                    metrics.gpu_memory_used.append(gpu.memoryUsed)
                    metrics.gpu_memory_total.append(gpu.memoryTotal)
                    metrics.gpu_temperature.append(gpu.temperature)
            except Exception as e:
                logger.warning(f"Failed to collect GPU metrics: {e}")

        # PyTorch GPU metrics
        elif TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                for i in range(torch.cuda.device_count()):
                    memory_allocated = torch.cuda.memory_allocated(i) / (1024**2)  # MB
                    memory_reserved = torch.cuda.memory_reserved(i) / (1024**2)  # MB
                    metrics.gpu_memory_used.append(memory_allocated)
                    metrics.gpu_memory_total.append(memory_reserved)
                    metrics.gpu_utilization.append(0.0)  # PyTorch doesn't provide utilization
                    metrics.gpu_temperature.append(0.0)
            except Exception as e:
                logger.warning(f"Failed to collect PyTorch GPU metrics: {e}")

        return metrics

    def _check_alerts(self, metrics: SystemMetrics) -> None:
        alerts = []

        if metrics.cpu_percent > self.thresholds['cpu_percent']:
            alerts.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")

        if metrics.memory_percent > self.thresholds['memory_percent']:
            alerts.append(f"High memory usage: {metrics.memory_percent:.1f}%")

        if metrics.disk_usage > self.thresholds['disk_usage']:
            alerts.append(f"High disk usage: {metrics.disk_usage:.1f}%")

        for i, gpu_util in enumerate(metrics.gpu_utilization):
            if gpu_util > self.thresholds['gpu_utilization']:
                alerts.append(f"High GPU {i} utilization: {gpu_util:.1f}%")

        for i, gpu_temp in enumerate(metrics.gpu_temperature):
            if gpu_temp > self.thresholds['gpu_temperature']:
                alerts.append(f"High GPU {i} temperature: {gpu_temp:.1f}°C")

        # Trigger alert callbacks
        for alert in alerts:
            logger.warning(f"System Alert: {alert}")
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")

    def get_current_metrics(self) -> Optional[SystemMetrics]:
        if self.metrics_history:
            return self.metrics_history[-1]
        return None

    def get_metrics_history(self, minutes: int = 10) -> List[SystemMetrics]:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        return [m for m in self.metrics_history if m.timestamp >= cutoff]

    def get_summary_stats(self, minutes: int = 10) -> Dict[str, Any]:
        recent_metrics = self.get_metrics_history(minutes)

        if not recent_metrics:
            return {}

        return {
            'cpu_usage': {
                'current': recent_metrics[-1].cpu_percent,
                'average': sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics),
                'max': max(m.cpu_percent for m in recent_metrics),
                'min': min(m.cpu_percent for m in recent_metrics)
            },
            'memory_usage': {
                'current': recent_metrics[-1].memory_percent,
                'average': sum(m.memory_percent for m in recent_metrics) / len(recent_metrics),
                'max': max(m.memory_percent for m in recent_metrics),
                'min': min(m.memory_percent for m in recent_metrics)
            },
            'gpu_stats': self._get_gpu_summary(recent_metrics),
            'timestamp_range': {
                'start': recent_metrics[0].timestamp.isoformat(),
                'end': recent_metrics[-1].timestamp.isoformat(),
                'duration_minutes': minutes
            }
        }

    def _get_gpu_summary(self, metrics: List[SystemMetrics]) -> Dict[str, Any]:
        if not metrics or not metrics[0].gpu_utilization:
            return {}

        gpu_count = len(metrics[0].gpu_utilization)
        gpu_stats = {}

        for gpu_id in range(gpu_count):
            utilizations = [m.gpu_utilization[gpu_id] for m in metrics if len(m.gpu_utilization) > gpu_id]
            memory_used = [m.gpu_memory_used[gpu_id] for m in metrics if len(m.gpu_memory_used) > gpu_id]
            temperatures = [m.gpu_temperature[gpu_id] for m in metrics if len(m.gpu_temperature) > gpu_id]

            gpu_stats[f'gpu_{gpu_id}'] = {
                'utilization': {
                    'current': utilizations[-1] if utilizations else 0,
                    'average': sum(utilizations) / len(utilizations) if utilizations else 0,
                    'max': max(utilizations) if utilizations else 0
                },
                'memory_used': {
                    'current': memory_used[-1] if memory_used else 0,
                    'average': sum(memory_used) / len(memory_used) if memory_used else 0,
                    'max': max(memory_used) if memory_used else 0
                },
                'temperature': {
                    'current': temperatures[-1] if temperatures else 0,
                    'average': sum(temperatures) / len(temperatures) if temperatures else 0,
                    'max': max(temperatures) if temperatures else 0
                }
            }

        return gpu_stats

    def add_alert_callback(self, callback: Callable) -> None:
        self.alert_callbacks.append(callback)

    def set_thresholds(self, **thresholds) -> None:
        for key, value in thresholds.items():
            if key in self.thresholds:
                self.thresholds[key] = value
                logger.info(f"Updated threshold {key}: {value}")

    def export_metrics(self, filepath: str, format: str = "json") -> bool:
        try:
            metrics_data = [
                {
                    'timestamp': m.timestamp.isoformat(),
                    'cpu_percent': m.cpu_percent,
                    'memory_percent': m.memory_percent,
                    'memory_available': m.memory_available,
                    'disk_usage': m.disk_usage,
                    'gpu_utilization': m.gpu_utilization,
                    'gpu_memory_used': m.gpu_memory_used,
                    'gpu_temperature': m.gpu_temperature,
                    'network_io': m.network_io
                } for m in self.metrics_history
            ]

            if format == "json":
                with open(filepath, 'w') as f:
                    json.dump(metrics_data, f, indent=2)
            elif format == "csv":
                import pandas as pd
                df = pd.DataFrame(metrics_data)
                df.to_csv(filepath, index=False)

            logger.info(f"System metrics exported to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to export system metrics: {e}")
            return False

    def clear_history(self) -> None:
        self.metrics_history.clear()
        logger.info("System metrics history cleared")
