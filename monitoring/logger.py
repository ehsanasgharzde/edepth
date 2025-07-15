#FILE: monitoring/logger.py
# ehsanasgharzde - LOGGER

import logging
import json
import gzip
import shutil
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import asdict
import threading
import time
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class MonitoringLogger:
    def __init__(self, log_dir: str = "logs", log_level: int = logging.INFO, structured_logging: bool = True):
        self.log_dir = Path(log_dir)
        self.structured_logging = structured_logging
        self.log_level = log_level
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.system_log = self.log_dir / "system.log"
        self.data_log = self.log_dir / "data.log"
        self.training_log = self.log_dir / "training.log"
        self.error_log = self.log_dir / "error.log"
        
        self.loggers = {}
        self.log_stats = defaultdict(int)
        self.error_cache = deque(maxlen=100)
        self.alert_cache = deque(maxlen=50)
        
        self._setup_loggers()
        self._start_background_cleanup()

    def _setup_loggers(self):
        log_configs = [
            ("system", self.system_log),
            ("data", self.data_log),
            ("training", self.training_log),
            ("error", self.error_log)
        ]
        
        for name, log_file in log_configs:
            logger_instance = logging.getLogger(name)
            logger_instance.setLevel(self.log_level)
            logger_instance.handlers.clear()
            
            handler = logging.FileHandler(log_file, mode='a')
            formatter = logging.Formatter('%(message)s')
            handler.setFormatter(formatter)
            logger_instance.addHandler(handler)
            logger_instance.propagate = False
            
            self.loggers[name] = logger_instance

    def _start_background_cleanup(self):
        cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        cleanup_thread.start()

    def _cleanup_worker(self):
        while True:
            time.sleep(3600)
            self._cleanup_old_logs()
            self._rotate_large_logs()

    def _write_log(self, logger_name: str, record: Dict[str, Any]):
        record["timestamp"] = datetime.utcnow().isoformat()
        msg = json.dumps(record) if self.structured_logging else str(record)
        
        try:
            self.loggers[logger_name].info(msg)
            self.log_stats[logger_name] += 1
        except Exception as e:
            logger.error(f"Failed to write log: {e}")

    def log_system_metrics(self, metrics: Any):
        metrics_dict = self._normalize_metrics(metrics)
        self._write_log("system", {"type": "system_metrics", "metrics": metrics_dict})

    def log_data_metrics(self, metrics: Any):
        metrics_dict = self._normalize_metrics(metrics)
        self._write_log("data", {"type": "data_metrics", "metrics": metrics_dict})

    def log_training_event(self, event: str, data: Dict[str, Any]):
        self._write_log("training", {"event": event, "data": data})

    def log_error_event(self, error: Exception, context: Dict[str, Any]):
        error_info = {
            "error": str(error),
            "type": type(error).__name__,
            "context": context
        }
        self._write_log("error", error_info)
        self.error_cache.append(error_info)

    def log_alert(self, alert_type: str, message: str, severity: str = "warning"):
        alert_record = {
            "alert_type": alert_type,
            "message": message,
            "severity": severity,
            "timestamp": datetime.utcnow().isoformat()
        }
        self._write_log("system", {"type": "alert", "alert": alert_record})
        self.alert_cache.append(alert_record)

    def log_performance_event(self, operation: str, duration: float, metadata: Optional[Dict[str, Any]] = None):
        perf_record = {
            "operation": operation,
            "duration_ms": duration * 1000,
            "metadata": metadata or {}
        }
        self._write_log("system", {"type": "performance", "performance": perf_record})

    def _normalize_metrics(self, metrics: Any) -> Dict[str, Any]:
        if hasattr(metrics, "__dict__"):
            return vars(metrics)
        elif hasattr(metrics, "__dataclass_fields__"):
            return asdict(metrics)
        elif isinstance(metrics, dict):
            return metrics
        else:
            return {"value": str(metrics)}

    def _rotate_large_logs(self, max_size_mb: int = 10):
        for log_file in [self.system_log, self.data_log, self.training_log, self.error_log]:
            if log_file.exists() and log_file.stat().st_size > max_size_mb * 1024 * 1024:
                timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
                compressed_file = log_file.with_name(f"{log_file.stem}_{timestamp}.log.gz")
                
                with open(log_file, 'rb') as f_in:
                    with gzip.open(compressed_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                log_file.unlink()
                log_file.touch()

    def _cleanup_old_logs(self, days_to_keep: int = 30):
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        
        for log_file in self.log_dir.glob("*.log.gz"):
            if log_file.stat().st_mtime < cutoff_date.timestamp():
                log_file.unlink()

    def create_log_summary(self, time_window: int = 3600) -> Dict[str, Any]:
        cutoff = datetime.utcnow() - timedelta(seconds=time_window)
        summary = {
            "system": 0,
            "data": 0,
            "training": 0,
            "errors": 0,
            "alerts": 0,
            "performance_events": 0
        }

        for name, log_file in [("system", self.system_log), ("data", self.data_log), 
                               ("training", self.training_log), ("error", self.error_log)]:
            if not log_file.exists():
                continue

            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        try:
                            record = json.loads(line.strip())
                            ts = datetime.fromisoformat(record.get("timestamp", ""))
                            if ts > cutoff:
                                if record.get("type") == "alert":
                                    summary["alerts"] += 1
                                elif record.get("type") == "performance":
                                    summary["performance_events"] += 1
                                else:
                                    summary[name] += 1
                        except (json.JSONDecodeError, ValueError):
                            continue
            except Exception as e:
                logger.error(f"Error reading log file {log_file}: {e}")

        return summary

    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        return list(self.error_cache)[-limit:]

    def get_recent_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        return list(self.alert_cache)[-limit:]

    def export_logs(self, output_file: str, time_window: int = 86400):
        cutoff = datetime.utcnow() - timedelta(seconds=time_window)
        exported_logs = []

        for log_file in [self.system_log, self.data_log, self.training_log, self.error_log]:
            if not log_file.exists():
                continue

            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        try:
                            record = json.loads(line.strip())
                            ts = datetime.fromisoformat(record.get("timestamp", ""))
                            if ts > cutoff:
                                record["source_log"] = log_file.name
                                exported_logs.append(record)
                        except (json.JSONDecodeError, ValueError):
                            continue
            except Exception as e:
                logger.error(f"Error reading log file {log_file}: {e}")

        exported_logs.sort(key=lambda x: x.get("timestamp", ""))
        
        with open(output_file, 'w') as f:
            json.dump(exported_logs, f, indent=2)

    def get_log_statistics(self) -> Dict[str, Any]:
        return {
            "total_logs": dict(self.log_stats),
            "error_count": len(self.error_cache),
            "alert_count": len(self.alert_cache),
            "log_files": {
                "system": self.system_log.stat().st_size if self.system_log.exists() else 0,
                "data": self.data_log.stat().st_size if self.data_log.exists() else 0,
                "training": self.training_log.stat().st_size if self.training_log.exists() else 0,
                "error": self.error_log.stat().st_size if self.error_log.exists() else 0
            }
        }