#FILE: monitoring/system_monitor.py
# ehsanasgharzde - SYSTEM MONITOR

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
    timestamp: datetime = field(default_factory=datetime.utcnow)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_available: float = 0.0
    gpu_utilization: List[float] = field(default_factory=list)
    gpu_memory_used: List[float] = field(default_factory=list)
    gpu_memory_total: List[float] = field(default_factory=list)
    gpu_temperature: List[float] = field(default_factory=list)
    disk_usage: float = 0.0
    network_io: Dict[str, float] = field(default_factory=lambda: {'bytes_sent': 0.0, 'bytes_recv': 0.0})
    process_count: int = 0
    load_average: List[float] = field(default_factory=list)

class SystemMonitor:
    def __init__(self, monitoring_interval: float = 1.0, history_size: int = 1000):
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        self.metrics_history = deque(maxlen=history_size)
        self.monitoring_thread = None
        self._stop_event = threading.Event()
        self.alert_thresholds = {
            'cpu': 90.0,
            'memory': 90.0,
            'gpu_util': 90.0,
            'gpu_temp': 80.0,
            'disk': 95.0,
        }
        self._initial_net = psutil.net_io_counters()
        self._initial_time = time.time()

    def start_monitoring(self):
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            self._stop_event.clear()
            self.monitoring_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitoring_thread.start()
            logger.info("System monitoring started")

    def stop_monitoring(self):
        self._stop_event.set()
        if self.monitoring_thread:
            self.monitoring_thread.join()
            logger.info("System monitoring stopped")

    def _monitor_loop(self):
        while not self._stop_event.is_set():
            try:
                metrics = self.collect_system_metrics()
                self.metrics_history.append(metrics)
                alerts = self.check_resource_alerts(metrics)
                if alerts:
                    for alert in alerts:
                        logger.warning(f"Resource alert: {alert}")
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)

    def collect_system_metrics(self) -> SystemMetrics:
        timestamp = datetime.utcnow()
        
        try:
            cpu_percent = psutil.cpu_percent(interval=None)
            load_avg = list(psutil.getloadavg())
            
            mem = psutil.virtual_memory()
            memory_percent = mem.percent
            memory_available = mem.available / (1024 ** 2)
            
            disk = psutil.disk_usage('/')
            disk_usage = disk.percent
            
            net_io = psutil.net_io_counters()
            now = time.time()
            time_diff = now - self._initial_time
            if time_diff > 0:
                net_sent_rate = (net_io.bytes_sent - self._initial_net.bytes_sent) / time_diff
                net_recv_rate = (net_io.bytes_recv - self._initial_net.bytes_recv) / time_diff
            else:
                net_sent_rate = net_recv_rate = 0.0
            
            network_io = {'bytes_sent': net_sent_rate, 'bytes_recv': net_recv_rate}
            self._initial_net = net_io
            self._initial_time = now
            
            process_count = len(psutil.pids())
            
            gpu_util, gpu_mem_used, gpu_mem_total, gpu_temp = [], [], [], []
            try:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    gpu_util.append(gpu.load * 100)
                    gpu_mem_used.append(gpu.memoryUsed)
                    gpu_mem_total.append(gpu.memoryTotal)
                    gpu_temp.append(gpu.temperature)
            except Exception as e:
                logger.debug(f"GPU monitoring not available: {e}")
            
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
            logger.error(f"Error collecting system metrics: {e}")
            return SystemMetrics(timestamp=timestamp)

    def get_current_metrics(self) -> SystemMetrics:
        return self.collect_system_metrics()

    def get_gpu_metrics(self) -> Dict[str, Any]:
        try:
            gpus = GPUtil.getGPUs()
            gpu_data = {}
            for gpu in gpus:
                gpu_data[f'GPU_{gpu.id}'] = {
                    'utilization': gpu.load * 100,
                    'memory_used_MB': gpu.memoryUsed,
                    'memory_total_MB': gpu.memoryTotal,
                    'temperature_C': gpu.temperature,
                    'name': gpu.name,
                }
            return gpu_data
        except Exception as e:
            logger.error(f"Error getting GPU metrics: {e}")
            return {'error': 'GPU monitoring not available'}

    def get_memory_breakdown(self) -> Dict[str, Any]:
        try:
            mem = psutil.virtual_memory()
            gpu_stats = self.get_gpu_metrics()
            
            pytorch_gpu_mem = {}
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    try:
                        torch.cuda.set_device(i)
                        pytorch_gpu_mem[f'GPU_{i}'] = {
                            'torch_allocated_MB': torch.cuda.memory_allocated(i) / (1024 ** 2),
                            'torch_reserved_MB': torch.cuda.memory_reserved(i) / (1024 ** 2)
                        }
                    except Exception as e:
                        logger.error(f"Error getting PyTorch GPU memory for device {i}: {e}")
            
            return {
                'system_memory_percent': mem.percent,
                'available_MB': mem.available / (1024 ** 2),
                'used_MB': mem.used / (1024 ** 2),
                'total_MB': mem.total / (1024 ** 2),
                'pytorch_gpu_memory': pytorch_gpu_mem,
                'gpu_memory': gpu_stats
            }
        except Exception as e:
            logger.error(f"Error getting memory breakdown: {e}")
            return {'error': 'Memory monitoring failed'}

    def check_resource_alerts(self, metrics: SystemMetrics) -> List[str]:
        alerts = []
        
        try:
            if metrics.cpu_percent > self.alert_thresholds['cpu']:
                alerts.append(f"High CPU usage: {metrics.cpu_percent:.2f}%")
            
            if metrics.memory_percent > self.alert_thresholds['memory']:
                alerts.append(f"High memory usage: {metrics.memory_percent:.2f}%")
            
            if metrics.disk_usage > self.alert_thresholds['disk']:
                alerts.append(f"Low disk space: {metrics.disk_usage:.2f}% used")
            
            for i, gpu_util in enumerate(metrics.gpu_utilization):
                if gpu_util > self.alert_thresholds['gpu_util']:
                    alerts.append(f"High GPU {i} utilization: {gpu_util:.2f}%")
            
            for i, temp in enumerate(metrics.gpu_temperature):
                if temp > self.alert_thresholds['gpu_temp']:
                    alerts.append(f"High GPU {i} temperature: {temp:.2f}C")
        except Exception as e:
            logger.error(f"Error checking resource alerts: {e}")
        
        return alerts

    def get_performance_summary(self, time_window: int = 300) -> Dict[str, Any]:
        try:
            now = datetime.utcnow()
            filtered_metrics = [m for m in self.metrics_history if (now - m.timestamp).total_seconds() <= time_window]
            
            if not filtered_metrics:
                return {"warning": "No data in selected time window"}
            
            def safe_avg(field):
                try:
                    values = [getattr(m, field) for m in filtered_metrics if hasattr(m, field)]
                    return sum(values) / len(values) if values else 0.0
                except:
                    return 0.0
            
            summary = {
                'time_window_seconds': time_window,
                'samples_count': len(filtered_metrics),
                'avg_cpu_percent': safe_avg('cpu_percent'),
                'avg_memory_percent': safe_avg('memory_percent'),
                'avg_memory_available_MB': safe_avg('memory_available'),
                'avg_disk_usage_percent': safe_avg('disk_usage'),
                'avg_process_count': safe_avg('process_count'),
            }
            
            if filtered_metrics[0].gpu_utilization:
                gpu_utils = []
                for m in filtered_metrics:
                    if m.gpu_utilization:
                        gpu_utils.append(sum(m.gpu_utilization) / len(m.gpu_utilization))
                summary['avg_gpu_utilization_percent'] = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0.0
            
            return summary
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {"error": "Failed to generate performance summary"}

    def get_metrics_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        try:
            recent_metrics = list(self.metrics_history)[-limit:]
            return [self._metrics_to_dict(m) for m in recent_metrics]
        except Exception as e:
            logger.error(f"Error getting metrics history: {e}")
            return []

    def _metrics_to_dict(self, metrics: SystemMetrics) -> Dict[str, Any]:
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
            metrics_data = self.get_metrics_history()
            
            if format == 'json':
                with open(filename, 'w') as f:
                    json.dump(metrics_data, f, indent=2)
            elif format == 'pickle':
                with open(filename, 'wb') as f:
                    pickle.dump(metrics_data, f)
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
            
            logger.info(f"Metrics exported to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            return False

    def set_alert_thresholds(self, thresholds: Dict[str, float]):
        self.alert_thresholds.update(thresholds)
        logger.info(f"Alert thresholds updated: {thresholds}")

    def clear_history(self):
        self.metrics_history.clear()
        logger.info("Metrics history cleared")

    def get_system_info(self) -> Dict[str, Any]:
        try:
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
            logger.error(f"Error getting system info: {e}")
            return {'error': 'Failed to get system info'}