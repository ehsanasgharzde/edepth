#FILE: monitoring/logger.py
# ehsanasgharzde - LOGGER
# hosseinsolymanzadeh - PROPER COMMENTING

import logging
import json
import gzip
import shutil
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import asdict
import threading
import time
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class MonitoringLogger:
    def __init__(self, log_dir: str = "logs", log_level: int = logging.INFO, structured_logging: bool = True):
        # Set up logging directory and config flags
        self.log_dir = Path(log_dir)
        self.structured_logging = structured_logging
        self.log_level = log_level
        self.log_dir.mkdir(parents=True, exist_ok=True)  # Ensure log directory exists

        # Define log file paths
        self.system_log = self.log_dir / "system.log"
        self.data_log = self.log_dir / "data.log"
        self.training_log = self.log_dir / "training.log"
        self.error_log = self.log_dir / "error.log"

        # Containers for logger instances and statistics
        self.loggers = {}
        self.log_stats = defaultdict(int)  # Tracks number of logs per logger
        self.error_cache = deque(maxlen=100)  # Stores recent error logs
        self.alert_cache = deque(maxlen=50)  # Stores recent alert logs

        # Initialize loggers and start cleanup thread
        self._setup_loggers()
        self._start_background_cleanup()

    def _setup_loggers(self):
        # Define configurations for different log categories
        log_configs = [
            ("system", self.system_log),
            ("data", self.data_log),
            ("training", self.training_log),
            ("error", self.error_log)
        ]

        # Set up individual logger for each category
        for name, log_file in log_configs:
            logger_instance = logging.getLogger(name)
            logger_instance.setLevel(self.log_level)
            logger_instance.handlers.clear()  # Remove any existing handlers

            handler = logging.FileHandler(log_file, mode='a')  # Append mode
            formatter = logging.Formatter('%(message)s')
            handler.setFormatter(formatter)
            logger_instance.addHandler(handler)
            logger_instance.propagate = False  # Prevent propagation to root logger

            self.loggers[name] = logger_instance

    def _start_background_cleanup(self):
        # Start a daemon thread to handle periodic log cleanup
        cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        cleanup_thread.start()

    def _cleanup_worker(self):
        # Periodically clean up old and large log files
        while True:
            time.sleep(3600)  # Wait for 1 hour
            self._cleanup_old_logs()
            self._rotate_large_logs()

    def _write_log(self, logger_name: str, record: Dict[str, Any]):
        # Add timestamp and serialize log record
        record["timestamp"] = datetime.utcnow().isoformat()
        msg = json.dumps(record) if self.structured_logging else str(record)

        try:
            # Write log to file and update statistics
            self.loggers[logger_name].info(msg)
            self.log_stats[logger_name] += 1
        except Exception as e:
            logger.error(f"Failed to write log: {e}")  # Log writing failure

    def log_system_metrics(self, metrics: Any):
        # Log system-level metrics
        metrics_dict = self._normalize_metrics(metrics)
        self._write_log("system", {"type": "system_metrics", "metrics": metrics_dict})

    def log_data_metrics(self, metrics: Any):
        # Log data-related metrics
        metrics_dict = self._normalize_metrics(metrics)
        self._write_log("data", {"type": "data_metrics", "metrics": metrics_dict})

    def log_training_event(self, event: str, data: Dict[str, Any]):
        # Log an event related to model training
        self._write_log("training", {"event": event, "data": data})

    def log_error_event(self, error: Exception, context: Dict[str, Any]):
        # Log error with contextual information
        error_info = {
            "error": str(error),
            "type": type(error).__name__,
            "context": context
        }
        self._write_log("error", error_info)
        self.error_cache.append(error_info)  # Store in error cache

    def log_alert(self, alert_type: str, message: str, severity: str = "warning"):
        # Log a system alert with severity level
        alert_record = {
            "alert_type": alert_type,
            "message": message,
            "severity": severity,
            "timestamp": datetime.utcnow().isoformat()
        }
        self._write_log("system", {"type": "alert", "alert": alert_record})
        self.alert_cache.append(alert_record)  # Store in alert cache

    def log_performance_event(self, operation: str, duration: float, metadata: Optional[Dict[str, Any]] = None):
        # Log performance metrics for an operation
        perf_record = {
            "operation": operation,
            "duration_ms": duration * 1000,
            "metadata": metadata or {}
        }
        self._write_log("system", {"type": "performance", "performance": perf_record})

    def _normalize_metrics(self, metrics: Any) -> Dict[str, Any]:
        # Convert metrics object into dictionary format
        if hasattr(metrics, "__dict__"):
            return vars(metrics)
        elif hasattr(metrics, "__dataclass_fields__"):
            return asdict(metrics)
        elif isinstance(metrics, dict):
            return metrics
        else:
            return {"value": str(metrics)}  # Fallback to string representation

    def _rotate_large_logs(self, max_size_mb: int = 10):
        # Check each log file's size and compress it if it exceeds the max size
        for log_file in [self.system_log, self.data_log, self.training_log, self.error_log]:
            if log_file.exists() and log_file.stat().st_size > max_size_mb * 1024 * 1024:
                # Generate timestamped filename for compressed log
                timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
                compressed_file = log_file.with_name(f"{log_file.stem}_{timestamp}.log.gz")

                # Compress the existing log file using gzip
                with open(log_file, 'rb') as f_in:
                    with gzip.open(compressed_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

                # Remove the original log file and create a fresh empty one
                log_file.unlink()
                log_file.touch()

    def _cleanup_old_logs(self, days_to_keep: int = 30):
        # Delete compressed log files older than the specified number of days
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)

        for log_file in self.log_dir.glob("*.log.gz"):
            if log_file.stat().st_mtime < cutoff_date.timestamp():
                log_file.unlink()

    def create_log_summary(self, time_window: int = 3600) -> Dict[str, Any]:
        # Generate summary of log activity within the given time window
        cutoff = datetime.utcnow() - timedelta(seconds=time_window)
        summary = {
            "system": 0,
            "data": 0,
            "training": 0,
            "errors": 0,
            "alerts": 0,
            "performance_events": 0
        }

        # Iterate over each log file and parse log records
        for name, log_file in [("system", self.system_log), ("data", self.data_log), 
                               ("training", self.training_log), ("error", self.error_log)]:
            if not log_file.exists():
                continue

            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        try:
                            # Parse JSON log record and extract timestamp
                            record = json.loads(line.strip())
                            ts = datetime.fromisoformat(record.get("timestamp", ""))
                            if ts > cutoff:
                                # Categorize log types for counting
                                if record.get("type") == "alert":
                                    summary["alerts"] += 1
                                elif record.get("type") == "performance":
                                    summary["performance_events"] += 1
                                else:
                                    summary[name] += 1
                        except (json.JSONDecodeError, ValueError):
                            continue  # Skip malformed lines
            except Exception as e:
                logger.error(f"Error reading log file {log_file}: {e}")

        return summary

    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        # Return most recent error entries up to the specified limit
        return list(self.error_cache)[-limit:]

    def get_recent_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        # Return most recent alert entries up to the specified limit
        return list(self.alert_cache)[-limit:]

    def export_logs(self, output_file: str, time_window: int = 86400):
        # Export logs within a time window to a single JSON file
        cutoff = datetime.utcnow() - timedelta(seconds=time_window)
        exported_logs = []

        for log_file in [self.system_log, self.data_log, self.training_log, self.error_log]:
            if not log_file.exists():
                continue

            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        try:
                            # Parse each line as JSON and filter by timestamp
                            record = json.loads(line.strip())
                            ts = datetime.fromisoformat(record.get("timestamp", ""))
                            if ts > cutoff:
                                record["source_log"] = log_file.name
                                exported_logs.append(record)
                        except (json.JSONDecodeError, ValueError):
                            continue  # Skip malformed records
            except Exception as e:
                logger.error(f"Error reading log file {log_file}: {e}")

        # Sort exported logs by timestamp before writing to file
        exported_logs.sort(key=lambda x: x.get("timestamp", ""))
        
        with open(output_file, 'w') as f:
            json.dump(exported_logs, f, indent=2)

    def get_log_statistics(self) -> Dict[str, Any]:
        # Return statistics about log usage and cache sizes
        return {
            "total_logs": dict(self.log_stats),  # Total number of logs per category
            "error_count": len(self.error_cache),  # Number of errors cached
            "alert_count": len(self.alert_cache),  # Number of alerts cached
            "log_files": {  # File sizes for each log file
                "system": self.system_log.stat().st_size if self.system_log.exists() else 0,
                "data": self.data_log.stat().st_size if self.data_log.exists() else 0,
                "training": self.training_log.stat().st_size if self.training_log.exists() else 0,
                "error": self.error_log.stat().st_size if self.error_log.exists() else 0
            }
        }
