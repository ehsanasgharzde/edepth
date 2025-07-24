#FILE: tests/test_monitoring.py
# ehsanasgharzde - TESTING MONITORING SYSTEM
# hosseinsolymanzadeh - PROPER COMMENTING
# ehsanasgharzde -  FIXED REDUNDANT CODE BY EXTRACTING PURE FUNCTIONS AND BASECLASS LEVEL METHODS

import unittest
import time
import torch
import torch.nn as nn
import os
import shutil
from datetime import datetime
from unittest.mock import Mock, patch

from monitoring.system_monitor import SystemResourceMonitor, SystemMetrics
from monitoring.model_monitor import ModelPerformanceMonitor, ModelMetrics, TrainingMetrics
from monitoring.model_integration import ModelMonitoringIntegration
from configs.config import MonitoringConfig

class TestMonitoringSystem(unittest.TestCase):
    
    def setUp(self):
        self.config = MonitoringConfig()
        self.test_dirs = ['test_logs', 'test_exports', 'monitoring_output']
        for dir_name in self.test_dirs:
            os.makedirs(dir_name, exist_ok=True)

    def test_system_resource_monitor_initialization(self):
        monitor = SystemResourceMonitor(update_interval=0.5, max_history=100)
        
        self.assertEqual(monitor.update_interval, 0.5)
        self.assertEqual(monitor.max_history, 100)
        self.assertFalse(monitor.is_monitoring)
        self.assertIsNotNone(monitor.thresholds)
        self.assertEqual(len(monitor.metrics_history), 0)

    def test_system_metrics_collection(self):
        monitor = SystemResourceMonitor()
        metrics = monitor._collect_metrics()
        
        self.assertIsInstance(metrics, SystemMetrics)
        self.assertGreaterEqual(metrics.cpu_percent, 0.0)
        self.assertGreaterEqual(metrics.memory_percent, 0.0)
        self.assertGreaterEqual(metrics.memory_available, 0.0)
        self.assertGreaterEqual(metrics.disk_usage, 0.0)
        self.assertIsInstance(metrics.gpu_utilization, list)
        self.assertIsInstance(metrics.network_io, dict)
        self.assertIsInstance(metrics.timestamp, datetime)

    def test_system_monitor_lifecycle(self):
        monitor = SystemResourceMonitor(update_interval=0.1)
        
        self.assertFalse(monitor.is_monitoring)
        
        monitor.start_monitoring()
        self.assertTrue(monitor.is_monitoring)
        self.assertIsNotNone(monitor.monitor_thread)
        
        time.sleep(0.3)
        self.assertGreater(len(monitor.metrics_history), 0)
        
        monitor.stop_monitoring()
        self.assertFalse(monitor.is_monitoring)

    def test_system_alert_thresholds(self):
        monitor = SystemResourceMonitor()
        
        original_cpu_threshold = monitor.thresholds['cpu_percent']
        monitor.set_thresholds(cpu_percent=90.0)
        self.assertEqual(monitor.thresholds['cpu_percent'], 90.0)
        
        alert_callback = Mock()
        monitor.add_alert_callback(alert_callback)
        
        high_cpu_metrics = SystemMetrics(cpu_percent=95.0)
        monitor._check_alerts(high_cpu_metrics)

    def test_system_metrics_history_and_summary(self):
        monitor = SystemResourceMonitor(max_history=5)
        
        for i in range(3):
            metrics = SystemMetrics(cpu_percent=50.0 + i * 10)
            monitor.metrics_history.append(metrics)
        
        current = monitor.get_current_metrics()
        self.assertIsNotNone(current)
        self.assertEqual(current.cpu_percent, 70.0) # type: ignore
        
        recent = monitor.get_metrics_history(minutes=10)
        self.assertEqual(len(recent), 3)
        
        summary = monitor.get_summary_stats(minutes=10)
        self.assertIn('cpu_usage', summary)
        self.assertIn('memory_usage', summary)

    def test_model_performance_monitor_initialization(self):
        monitor = ModelPerformanceMonitor(max_history=500)
        
        self.assertEqual(monitor.max_history, 500)
        self.assertEqual(len(monitor.active_models), 0)
        self.assertEqual(len(monitor.training_metrics), 0)
        self.assertFalse(monitor.is_monitoring)

    def test_model_registration_and_tracking(self):
        monitor = ModelPerformanceMonitor()
        
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        
        monitor.register_model(model, "test_model")
        
        self.assertIn("test_model", monitor.active_models)
        self.assertGreater(monitor.active_models["test_model"]["parameters"], 0)
        self.assertEqual(monitor.active_models["test_model"]["trainable_params"], 
                        monitor.active_models["test_model"]["parameters"])

    def test_inference_tracking(self):
        monitor = ModelPerformanceMonitor()
        model = nn.Linear(10, 1)
        monitor.register_model(model, "linear_model")
        
        input_tensor = torch.randn(5, 10)
        output_tensor = torch.randn(5, 1)
        inference_time = 0.05
        
        monitor.track_inference("linear_model", input_tensor, output_tensor, inference_time)
        
        self.assertEqual(len(monitor.model_metrics["linear_model"]), 1)
        
        metrics = monitor.model_metrics["linear_model"][0]
        self.assertEqual(metrics.model_name, "linear_model")
        self.assertEqual(metrics.inference_time, inference_time)
        self.assertEqual(metrics.batch_size, 5)
        self.assertEqual(metrics.input_shape, (5, 10))

    def test_training_step_tracking(self):
        monitor = ModelPerformanceMonitor()
        
        monitor.track_training_step(
            epoch=1, 
            step=100, 
            loss=0.5, 
            learning_rate=0.001,
            accuracy=0.85,
            gradient_norm=1.2
        )
        
        self.assertEqual(len(monitor.training_metrics), 1)
        
        metrics = monitor.training_metrics[0]
        self.assertEqual(metrics.epoch, 1)
        self.assertEqual(metrics.step, 100)
        self.assertEqual(metrics.loss, 0.5)
        self.assertEqual(metrics.learning_rate, 0.001)

    def test_model_summary_generation(self):
        monitor = ModelPerformanceMonitor()
        model = nn.Linear(10, 1)
        monitor.register_model(model, "test_model")
        
        for i in range(5):
            input_tensor = torch.randn(2, 10)
            output_tensor = torch.randn(2, 1)
            monitor.track_inference("test_model", input_tensor, output_tensor, 0.01 * (i + 1))
        
        summary = monitor.get_model_summary("test_model")
        
        self.assertIn("model_name", summary)
        self.assertIn("total_measurements", summary)
        self.assertIn("avg_inference_time", summary)
        self.assertIn("parameters_count", summary)
        self.assertEqual(summary["total_measurements"], 5)

    def test_training_summary_generation(self):
        monitor = ModelPerformanceMonitor()
        
        for step in range(10):
            monitor.track_training_step(
                epoch=0,
                step=step,
                loss=1.0 / (step + 1),
                learning_rate=0.01
            )
        
        summary = monitor.get_training_summary()
        
        self.assertIn("total_steps", summary)
        self.assertIn("current_step", summary)
        self.assertIn("current_loss", summary)
        self.assertIn("min_loss", summary)
        self.assertEqual(summary["total_steps"], 10)
        self.assertEqual(summary["current_step"], 9)

    def test_monitoring_integration_initialization(self):
        config = MonitoringConfig()
        integration = ModelMonitoringIntegration(config)
        
        self.assertIsNotNone(integration.model_monitor)
        self.assertIsNotNone(integration.system_monitor)
        self.assertEqual(integration.config, config)
        self.assertFalse(integration.is_monitoring)

    def test_monitoring_integration_lifecycle(self):
        integration = ModelMonitoringIntegration()
        
        self.assertFalse(integration.is_monitoring)
        
        integration.start_monitoring()
        self.assertTrue(integration.is_monitoring)
        
        integration.stop_monitoring()
        self.assertFalse(integration.is_monitoring)

    def test_integration_model_registration(self):
        integration = ModelMonitoringIntegration()
        
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        
        integration.register_model(model, "integration_model")
        
        self.assertIn("integration_model", integration.model_monitor.active_models)

    def test_integration_inference_tracking(self):
        integration = ModelMonitoringIntegration()
        
        input_tensor = torch.randn(3, 5)
        output_tensor = torch.randn(3, 2)
        inference_time = 0.02
        
        integration.track_inference_result("test_model", input_tensor, output_tensor, inference_time)
        
        if integration.config.enable_model_monitoring:
            self.assertGreater(len(integration.model_monitor.model_metrics), 0)

    def test_integration_training_tracking(self):
        integration = ModelMonitoringIntegration()
        
        integration.track_training_step(
            epoch=2,
            step=50,
            loss=0.3,
            learning_rate=0.005,
            validation_loss=0.35
        )
        
        if integration.config.enable_model_monitoring:
            self.assertGreater(len(integration.model_monitor.training_metrics), 0)

    def test_gradient_norm_calculation(self):
        integration = ModelMonitoringIntegration()
        
        model = nn.Linear(5, 1)
        input_tensor = torch.randn(1, 5)
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()
        
        grad_norm = integration.calculate_gradient_norm(model)
        
        if integration.config.gradient_monitoring:
            self.assertIsInstance(grad_norm, (float, type(None)))
            if grad_norm is not None:
                self.assertGreaterEqual(grad_norm, 0.0)

    def test_monitoring_summary_generation(self):
        integration = ModelMonitoringIntegration()
        integration.start_monitoring()
        
        model = nn.Linear(3, 1)
        integration.register_model(model, "summary_model")
        
        summary = integration.get_monitoring_summary()
        
        self.assertIn("monitoring_active", summary)
        self.assertIn("config", summary)
        self.assertTrue(summary["monitoring_active"])
        
        if integration.config.enable_model_monitoring:
            self.assertIn("model_summaries", summary)
            self.assertIn("training_summary", summary)
        
        if integration.config.enable_system_monitoring:
            self.assertIn("system_summary", summary)

    def test_metrics_export_functionality(self):
        monitor = ModelPerformanceMonitor()
        
        for i in range(3):
            monitor.track_training_step(i, i, 0.5 - i * 0.1, 0.01)
        
        export_path = os.path.join("test_exports", "test_metrics.json")
        success = monitor.export_metrics(export_path, format="json")
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(export_path))
        
        with open(export_path, 'r') as f:
            import json
            data = json.load(f)
            self.assertIn("training_metrics", data)

    def test_system_metrics_export(self):
        monitor = SystemResourceMonitor()
        
        for i in range(3):
            metrics = SystemMetrics(cpu_percent=50.0 + i * 5)
            monitor.metrics_history.append(metrics)
        
        export_path = os.path.join("test_exports", "system_metrics.json")
        success = monitor.export_metrics(export_path, format="json")
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(export_path))

    def test_configuration_integration(self):
        config = MonitoringConfig()
        
        self.assertIsNotNone(config.enable_model_monitoring)
        self.assertIsNotNone(config.enable_system_monitoring)
        self.assertIsNotNone(config.model_history_size)
        self.assertIsNotNone(config.system_update_interval)

    def test_alert_callback_system(self):
        monitor = SystemResourceMonitor()
        
        callback_executed = Mock()
        monitor.add_alert_callback(callback_executed)
        
        high_memory_metrics = SystemMetrics(memory_percent=90.0)
        monitor._check_alerts(high_memory_metrics)

    def test_data_integrity_and_cleanup(self):
        monitor = ModelPerformanceMonitor(max_history=3)
        
        for i in range(5):
            monitor.track_training_step(0, i, 0.1, 0.01)
        
        self.assertLessEqual(len(monitor.training_metrics), 3)
        
        monitor.clear_metrics()
        self.assertEqual(len(monitor.training_metrics), 0)

    def test_concurrent_monitoring_access(self):
        monitor = SystemResourceMonitor(update_interval=0.1)
        monitor.start_monitoring()
        
        time.sleep(0.2)
        
        metrics1 = monitor.get_current_metrics()
        metrics2 = monitor.get_metrics_history(minutes=1)
        summary = monitor.get_summary_stats(minutes=1)
        
        self.assertIsNotNone(metrics1)
        self.assertIsInstance(metrics2, list)
        self.assertIsInstance(summary, dict)
        
        monitor.stop_monitoring()

    def test_gpu_metrics_handling(self):
        monitor = SystemResourceMonitor()
        metrics = monitor._collect_metrics()
        
        self.assertIsInstance(metrics.gpu_utilization, list)
        self.assertIsInstance(metrics.gpu_memory_used, list)
        self.assertIsInstance(metrics.gpu_temperature, list)

    def test_network_io_tracking(self):
        monitor = SystemResourceMonitor()
        metrics = monitor._collect_metrics()
        
        self.assertIn('bytes_sent', metrics.network_io)
        self.assertIn('bytes_recv', metrics.network_io)
        self.assertIn('packets_sent', metrics.network_io)
        self.assertIn('packets_recv', metrics.network_io)

    def test_threshold_customization(self):
        monitor = SystemResourceMonitor()
        
        original_thresholds = monitor.thresholds.copy()
        
        monitor.set_thresholds(
            cpu_percent=75.0,
            memory_percent=80.0,
            gpu_utilization=85.0
        )
        
        self.assertEqual(monitor.thresholds['cpu_percent'], 75.0)
        self.assertEqual(monitor.thresholds['memory_percent'], 80.0)
        self.assertEqual(monitor.thresholds['gpu_utilization'], 85.0)

    def test_integration_export_all_metrics(self):
        integration = ModelMonitoringIntegration()
        integration.start_monitoring()
        
        model = nn.Linear(2, 1)
        integration.register_model(model, "export_test")
        integration.track_training_step(0, 1, 0.4, 0.01)
        
        export_base = os.path.join("test_exports", "integration_export")
        success = integration.export_all_metrics(f"{export_base}.json")
        
        integration.stop_monitoring()

    def test_model_metrics_dataclass_functionality(self):
        metrics = ModelMetrics(
            model_name="test",
            inference_time=0.05,
            memory_usage=128.0,
            input_shape=(1, 10),
            parameters_count=100
        )
        
        self.assertEqual(metrics.model_name, "test")
        self.assertEqual(metrics.inference_time, 0.05)
        self.assertEqual(metrics.memory_usage, 128.0)
        self.assertIsInstance(metrics.timestamp, datetime)

    def test_training_metrics_dataclass_functionality(self):
        metrics = TrainingMetrics(
            epoch=5,
            step=100,
            loss=0.25,
            learning_rate=0.001,
            gradient_norm=0.8
        )
        
        self.assertEqual(metrics.epoch, 5)
        self.assertEqual(metrics.step, 100)
        self.assertEqual(metrics.loss, 0.25)
        self.assertEqual(metrics.gradient_norm, 0.8)
        self.assertIsInstance(metrics.timestamp, datetime)

    def test_monitoring_decorator_integration(self):
        from monitoring.model_integration import monitor_model_inference
        
        integration = ModelMonitoringIntegration()
        
        @monitor_model_inference(integration, "decorated_model")
        def dummy_inference(input_tensor):
            return torch.randn(1, 2)
        
        input_data = torch.randn(1, 5)
        result = dummy_inference(input_data)
        
        self.assertIsNotNone(result)

    def test_error_handling_in_monitoring(self):
        monitor = SystemResourceMonitor()
        
        with patch('psutil.cpu_percent', side_effect=Exception("Test error")):
            try:
                monitor.start_monitoring()
                time.sleep(0.1)
                monitor.stop_monitoring()
            except Exception:
                self.fail("Monitoring should handle internal errors gracefully")

    def test_memory_management_in_long_running_monitor(self):
        monitor = ModelPerformanceMonitor(max_history=5)
        
        for i in range(10):
            monitor.track_training_step(0, i, 0.1, 0.01)
        
        self.assertEqual(len(monitor.training_metrics), 5)

    def tearDown(self):
        for dir_name in self.test_dirs:
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name, ignore_errors=True)