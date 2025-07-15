#FILE: tests/test_monitoring.py
# ehsanasgharzde - TESTING MONITORING SYSTEM

import unittest
from unittest.mock import MagicMock, patch, Mock
import logging
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitoring.config import MonitoringConfig
from monitoring.data_monitor import DataMonitor
from monitoring.system_monitor import SystemMonitor
from monitoring.training_monitor import TrainingMonitor
from monitoring.profiler import PerformanceProfiler
from monitoring.hooks import MonitoringHooks
from monitoring.logger import MonitoringLogger
from monitoring.integration import MonitoringIntegration
from monitoring.visualization import MonitoringVisualizer
from monitoring import ComprehensiveMonitor

class TestMonitoringSystem(unittest.TestCase):
    
    def setUp(self):
        self.config = MonitoringConfig()
        self.logger = logging.getLogger("test_monitor")
        self.logger.setLevel(logging.INFO)

    def test_comprehensive_monitor_initialization(self):
        monitor = ComprehensiveMonitor(self.config)
        
        self.assertIsNotNone(monitor.data_monitor)
        self.assertIsNotNone(monitor.system_monitor)
        self.assertIsNotNone(monitor.training_monitor)
        self.assertIsNotNone(monitor.profiler)
        self.assertIsNotNone(monitor.visualizer)
        self.assertIsNotNone(monitor.hooks_manager)
        self.assertIsNotNone(monitor.integration)
        self.assertFalse(monitor.state.is_active)

    def test_monitoring_start_stop_cycle(self):
        monitor = ComprehensiveMonitor(self.config)
        
        self.assertFalse(monitor.state.is_active)
        
        monitor.start_monitoring()
        self.assertTrue(monitor.state.is_active)
        self.assertIsNotNone(monitor.state.start_time)
        
        report = monitor.stop_monitoring()
        self.assertFalse(monitor.state.is_active)
        self.assertIsNotNone(monitor.state.end_time)
        self.assertIsInstance(report, dict)
        self.assertIn("monitoring_session", report)

    def test_system_monitor_functionality(self):
        system_monitor = SystemMonitor(monitoring_interval=0.1, history_size=10)
        
        metrics = system_monitor.collect_system_metrics()
        self.assertIsNotNone(metrics.cpu_percent)
        self.assertIsNotNone(metrics.memory_percent)
        self.assertIsNotNone(metrics.timestamp)
        
        alerts = system_monitor.check_resource_alerts(metrics)
        self.assertIsInstance(alerts, list)
        
        summary = system_monitor.get_performance_summary()
        self.assertIsInstance(summary, dict)
        self.assertIn("avg_cpu_percent", summary)

    def test_data_monitor_tensor_tracking(self):
        data_monitor = DataMonitor(
            track_gradients=True,
            track_activations=True,
            sample_rate=1.0
        )
        
        test_tensor = torch.randn(10, 10, requires_grad=True)
        tensor_id = data_monitor.register_tensor(test_tensor, "test_tensor", "activation")
        
        self.assertIsNotNone(tensor_id)
        self.assertIn(tensor_id, data_monitor.tensor_registry)
        
        metrics = data_monitor.track_tensor_stats(test_tensor, "test_tensor")
        self.assertIsNotNone(metrics.mean_value)
        self.assertIsNotNone(metrics.std_value)
        self.assertEqual(metrics.tensor_id, tensor_id)
        
        data_monitor.cleanup_hooks()

    def test_training_monitor_step_logging(self):
        training_monitor = TrainingMonitor(
            log_interval=1,
            checkpoint_interval=5,
            logger=self.logger
        )
        
        training_monitor.start_training_session()
        
        for step in range(10):
            loss = 1.0 / (step + 1)
            lr = 0.01 * (0.9 ** step)
            metrics = {"accuracy": 0.8 + step * 0.02}
            
            training_monitor.log_training_step(step, loss, metrics, lr)
        
        self.assertEqual(len(training_monitor.training_history), 10)
        self.assertEqual(training_monitor.current_step, 9)
        
        convergence_analysis = training_monitor.analyze_loss_convergence()
        self.assertIsInstance(convergence_analysis, dict)
        self.assertIn("status", convergence_analysis)
        
        anomalies = training_monitor.detect_training_anomalies()
        self.assertIsInstance(anomalies, list)

    def test_profiler_context_manager(self):
        profiler = PerformanceProfiler(
            profile_memory=True,
            profile_cuda=torch.cuda.is_available(),
            output_dir="test_profiler_output"
        )
        
        with profiler.profile_context("test_section"):
            result = torch.randn(100, 100).sum()
        
        self.assertIn("test_section", profiler.profiler_results)
        self.assertIsNotNone(profiler.profiler_results["test_section"]["duration"])

    def test_hooks_manager_model_integration(self):
        data_monitor = DataMonitor()
        system_monitor = SystemMonitor()
        hooks_manager = MonitoringHooks(data_monitor, system_monitor, self.logger)
        
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
        )
        
        for i, module in enumerate(model):
            if isinstance(module, torch.nn.Linear):
                hooks_manager.register_forward_hook(module, f"linear_{i}")
        
        self.assertGreater(len(hooks_manager.forward_hooks), 0)
        
        input_tensor = torch.randn(1, 10)
        output = model(input_tensor)
        
        summary = hooks_manager.get_hook_summary()
        self.assertIn("forward_hooks", summary)
        
        hooks_manager.remove_all_hooks()
        self.assertEqual(len(hooks_manager.forward_hooks), 0)

    def test_monitoring_logger_functionality(self):
        logger = MonitoringLogger(
            log_dir="test_logs",
            log_level=logging.INFO,
            structured_logging=True
        )
        
        test_metrics = {"cpu": 50.0, "memory": 60.0}
        logger.log_system_metrics(test_metrics)
        
        test_error = Exception("Test error")
        logger.log_error_event(test_error, {"context": "test"})
        
        logger.log_alert("cpu_usage", "High CPU usage detected", "warning")
        
        recent_errors = logger.get_recent_errors()
        self.assertIsInstance(recent_errors, list)
        
        recent_alerts = logger.get_recent_alerts()
        self.assertIsInstance(recent_alerts, list)

    def test_integration_with_model(self):
        integration = MonitoringIntegration(self.config)
        
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
        )
        
        integration.integrate_with_model(model)
        
        self.assertGreater(len(integration.hooks_manager.forward_hooks), 0)
        
        input_tensor = torch.randn(1, 10)
        output = model(input_tensor)
        
        status = integration.get_current_status()
        self.assertIn("is_active", status)
        self.assertIn("metrics_collected", status)

    def test_data_validation_and_anomaly_detection(self):
        data_monitor = DataMonitor()
        
        valid_tensor = torch.randn(10, 10)
        validation_result = data_monitor.validate_data_integrity(valid_tensor)
        self.assertTrue(validation_result["valid"])
        
        invalid_tensor = torch.tensor([1.0, 2.0, float('nan'), 4.0])
        validation_result = data_monitor.validate_data_integrity(invalid_tensor)
        self.assertFalse(validation_result["valid"])
        self.assertGreater(validation_result["nan_count"], 0)
        
        anomalies = data_monitor.detect_data_anomalies(valid_tensor)
        self.assertIsInstance(anomalies, list)

    def test_memory_usage_monitoring(self):
        profiler = PerformanceProfiler()
        
        memory_stats = profiler.profile_memory_usage()
        self.assertIsInstance(memory_stats, dict)
        self.assertIn("system", memory_stats)
        
        if torch.cuda.is_available():
            self.assertIn("cuda_0", memory_stats)

    def test_gradient_flow_monitoring(self):
        data_monitor = DataMonitor(track_gradients=True)
        
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
        )
        
        gradient_info = data_monitor.monitor_gradient_flow(model)
        self.assertIn("params_monitored", gradient_info)
        
        input_tensor = torch.randn(1, 10)
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()
        
        data_monitor.cleanup_hooks()

    def test_configuration_validation(self):
        config = MonitoringConfig()
        
        self.assertIsNotNone(config.system_thresholds)
        self.assertIsNotNone(config.training_thresholds)
        self.assertIsNotNone(config.data_thresholds)
        self.assertIsNotNone(config.logging_config)
        self.assertIsNotNone(config.profiling_config)
        
        alert_thresholds = config.get_alert_thresholds()
        self.assertIsInstance(alert_thresholds, dict)
        self.assertIn("cpu_usage", alert_thresholds)

    def test_visualization_integration(self):
        visualizer = MonitoringVisualizer(self.config, MonitoringLogger())
        
        system_metrics = [
            SystemMonitor().collect_system_metrics()
            for _ in range(5)
        ]
        
        training_monitor = TrainingMonitor()
        training_monitor.start_training_session()
        
        for i in range(10):
            training_monitor.log_training_step(i, 1.0/(i+1), {}, 0.01)
        
        plot_path = visualizer.plot_system_metrics(system_metrics)
        self.assertTrue(os.path.exists(plot_path) or plot_path == "")

    def test_comprehensive_workflow(self):
        monitor = ComprehensiveMonitor(self.config)
        
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
        )
        
        monitor.start_monitoring(model=model)
        
        for step in range(5):
            input_tensor = torch.randn(1, 10)
            output = model(input_tensor)
            loss = output.sum()
            loss.backward()
            
            monitor.training_monitor.log_training_step(
                step, loss.item(), {"accuracy": 0.9}, 0.01
            )
        
        status = monitor.get_real_time_status()
        self.assertTrue(status["is_active"])
        self.assertGreater(status["duration"], 0)
        
        report = monitor.stop_monitoring()
        self.assertIsInstance(report, dict)
        self.assertIn("monitoring_session", report)

    def test_error_handling_and_recovery(self):
        monitor = ComprehensiveMonitor(self.config)
        
        try:
            monitor.start_monitoring()
            
            invalid_tensor = torch.tensor([float('nan')])
            monitor.data_monitor.validate_data_integrity(invalid_tensor)
            
            monitor.stop_monitoring()
            
        except Exception as e:
            self.logger.error(f"Error in monitoring: {e}")
            monitor.cleanup()

    def test_multi_gpu_support(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        system_monitor = SystemMonitor()
        metrics = system_monitor.collect_system_metrics()
        
        if metrics.gpu_utilization:
            self.assertIsInstance(metrics.gpu_utilization, list)
            self.assertIsInstance(metrics.gpu_temperature, list)
            self.assertIsInstance(metrics.gpu_memory_used, list)

    def tearDown(self):
        import shutil
        
        cleanup_dirs = [
            "test_logs",
            "test_profiler_output",
            "monitoring_output",
            "logs"
        ]
        
        for dir_name in cleanup_dirs:
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name, ignore_errors=True)

if __name__ == "__main__":
    unittest.main()