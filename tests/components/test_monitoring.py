#FILE: tests/test_monitoring.py
# ehsanasgharzde - TESTING MONITORING SYSTEM
# hosseinsolymanzadeh - PROPER COMMENTING
# ehsanasgharzde -  FIXED REDUNDANT CODE BY EXTRACTING PURE FUNCTIONS AND BASECLASS LEVEL METHODS

import time
import torch
import pytest
import torch.nn as nn
from datetime import datetime

from monitoring.system_monitor import SystemResourceMonitor, SystemMetrics
from monitoring.model_monitor import ModelPerformanceMonitor, ModelMetrics, TrainingMetrics
from monitoring.model_integration import ModelMonitoringIntegration
from configs.config import MonitoringConfig


# Fixture for config and directories management
@pytest.fixture
def setup_monitoring_env() -> MonitoringConfig:
    # Create a standard config for testing
    config = MonitoringConfig()
    
    # pytest uses its own temporary directory management
    # and fixtures for cleanup, so we don't need explicit directory creation
    
    return config


# System monitor tests
def test_system_resource_monitor_initialization() -> None:
    # Test that monitor initializes with correct parameters
    monitor = SystemResourceMonitor(update_interval=0.5, max_history=100)
    
    assert monitor.update_interval == 0.5
    assert monitor.max_history == 100
    assert not monitor.is_monitoring
    assert monitor.thresholds is not None
    assert len(monitor.metrics_history) == 0


def test_system_metrics_collection() -> None:
    # Test that monitor collects valid metrics
    monitor = SystemResourceMonitor()
    metrics = monitor._collect_metrics()
    
    assert isinstance(metrics, SystemMetrics)
    assert metrics.cpu_percent >= 0.0
    assert metrics.memory_percent >= 0.0
    assert metrics.memory_available >= 0.0
    assert metrics.disk_usage >= 0.0
    assert isinstance(metrics.gpu_utilization, list)
    assert isinstance(metrics.network_io, dict)
    assert isinstance(metrics.timestamp, datetime)


def test_system_monitor_lifecycle() -> None:
    # Test the start/stop functionality of the system monitor
    monitor = SystemResourceMonitor(update_interval=0.1)
    
    assert not monitor.is_monitoring
    
    monitor.start_monitoring()
    assert monitor.is_monitoring
    assert monitor.monitor_thread is not None
    
    time.sleep(0.3)  # Allow collection of some metrics
    assert len(monitor.metrics_history) > 0
    
    monitor.stop_monitoring()
    assert not monitor.is_monitoring


def test_system_alert_thresholds() -> None:
    # Test setting and checking alert thresholds
    monitor = SystemResourceMonitor()
    
    original_cpu_threshold = monitor.thresholds['cpu_percent']
    monitor.set_thresholds(cpu_percent=90.0)
    assert monitor.thresholds['cpu_percent'] == 90.0
    
    # Create a metrics object with high CPU
    high_cpu_metrics = SystemMetrics(cpu_percent=95.0)
    
    # Create a callback to track alerts
    alert_received = False
    
    def alert_callback(message: str) -> None:
        nonlocal alert_received
        alert_received = True
    
    monitor.add_alert_callback(alert_callback)
    monitor._check_alerts(high_cpu_metrics)
    
    assert alert_received  # Alert should be triggered


def test_system_metrics_history_and_summary() -> None:
    # Test metrics history collection and summary stats
    monitor = SystemResourceMonitor(max_history=5)
    
    for i in range(3):
        metrics = SystemMetrics(cpu_percent=50.0 + i * 10)
        monitor.metrics_history.append(metrics)
    
    current = monitor.get_current_metrics()
    assert current is not None
    assert current.cpu_percent == 70.0
    
    recent = monitor.get_metrics_history(minutes=10)
    assert len(recent) == 3
    
    summary = monitor.get_summary_stats(minutes=10)
    assert 'cpu_usage' in summary
    assert 'memory_usage' in summary


# Model monitor tests
def test_model_performance_monitor_initialization() -> None:
    # Test model monitor initialization
    monitor = ModelPerformanceMonitor(max_history=500)
    
    assert monitor.max_history == 500
    assert len(monitor.active_models) == 0
    assert len(monitor.training_metrics) == 0
    assert not monitor.is_monitoring


def test_model_registration_and_tracking() -> None:
    # Test registering a model for monitoring
    monitor = ModelPerformanceMonitor()
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1)
    )
    
    monitor.register_model(model, "test_model")
    
    assert "test_model" in monitor.active_models
    assert monitor.active_models["test_model"]["parameters"] > 0
    assert monitor.active_models["test_model"]["trainable_params"] == monitor.active_models["test_model"]["parameters"]


def test_inference_tracking() -> None:
    # Test tracking model inference performance
    monitor = ModelPerformanceMonitor()
    model = nn.Linear(10, 1)
    monitor.register_model(model, "linear_model")
    
    input_tensor = torch.randn(5, 10)
    output_tensor = torch.randn(5, 1)
    inference_time = 0.05
    
    monitor.track_inference("linear_model", input_tensor, output_tensor, inference_time)
    
    assert len(monitor.model_metrics["linear_model"]) == 1
    
    metrics = monitor.model_metrics["linear_model"][0]
    assert metrics.model_name == "linear_model"
    assert metrics.inference_time == inference_time
    assert metrics.batch_size == 5
    assert metrics.input_shape == (5, 10)


def test_training_step_tracking() -> None:
    # Test tracking training steps
    monitor = ModelPerformanceMonitor()
    
    monitor.track_training_step(
        epoch=1, 
        step=100, 
        loss=0.5, 
        learning_rate=0.001,
        accuracy=0.85,
        gradient_norm=1.2
    )
    
    assert len(monitor.training_metrics) == 1
    
    metrics = monitor.training_metrics[0]
    assert metrics.epoch == 1
    assert metrics.step == 100
    assert metrics.loss == 0.5
    assert metrics.learning_rate == 0.001


def test_model_summary_generation() -> None:
    # Test generating model performance summary
    monitor = ModelPerformanceMonitor()
    model = nn.Linear(10, 1)
    monitor.register_model(model, "test_model")
    
    # Generate multiple inference records
    for i in range(5):
        input_tensor = torch.randn(2, 10)
        output_tensor = torch.randn(2, 1)
        monitor.track_inference("test_model", input_tensor, output_tensor, 0.01 * (i + 1))
    
    summary = monitor.get_model_summary("test_model")
    
    assert "model_name" in summary
    assert "total_measurements" in summary
    assert "avg_inference_time" in summary
    assert "parameters_count" in summary
    assert summary["total_measurements"] == 5


def test_training_summary_generation() -> None:
    # Test generating training summary
    monitor = ModelPerformanceMonitor()
    
    # Generate training steps
    for step in range(10):
        monitor.track_training_step(
            epoch=0,
            step=step,
            loss=1.0 / (step + 1),
            learning_rate=0.01
        )
    
    summary = monitor.get_training_summary()
    
    assert "total_steps" in summary
    assert "current_step" in summary
    assert "current_loss" in summary
    assert "min_loss" in summary
    assert summary["total_steps"] == 10
    assert summary["current_step"] == 9


# Integration tests
def test_monitoring_integration_initialization() -> None:
    # Test integration component initialization
    config = MonitoringConfig()
    integration = ModelMonitoringIntegration(config)
    
    assert integration.model_monitor is not None
    assert integration.system_monitor is not None
    assert integration.config == config
    assert not integration.is_monitoring


def test_monitoring_integration_lifecycle() -> None:
    # Test integration start/stop functionality
    integration = ModelMonitoringIntegration()
    
    assert not integration.is_monitoring
    
    integration.start_monitoring()
    assert integration.is_monitoring
    
    integration.stop_monitoring()
    assert not integration.is_monitoring


def test_integration_model_registration() -> None:
    # Test model registration through integration
    integration = ModelMonitoringIntegration()
    
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1)
    )
    
    integration.register_model(model, "integration_model")
    
    assert "integration_model" in integration.model_monitor.active_models


def test_integration_inference_tracking() -> None:
    # Test inference tracking through integration
    integration = ModelMonitoringIntegration()
    
    input_tensor = torch.randn(3, 5)
    output_tensor = torch.randn(3, 2)
    inference_time = 0.02
    
    # Register the model first
    model = nn.Linear(5, 2)
    integration.register_model(model, "test_model")
    
    integration.track_inference_result("test_model", input_tensor, output_tensor, inference_time)
    
    if integration.config.enable_model_monitoring:
        assert len(integration.model_monitor.model_metrics["test_model"]) > 0


def test_integration_training_tracking() -> None:
    # Test tracking training through integration
    integration = ModelMonitoringIntegration()
    
    integration.track_training_step(
        epoch=2,
        step=50,
        loss=0.3,
        learning_rate=0.005,
        validation_loss=0.35
    )
    
    if integration.config.enable_model_monitoring:
        assert len(integration.model_monitor.training_metrics) > 0


def test_gradient_norm_calculation() -> None:
    # Test gradient norm calculation
    integration = ModelMonitoringIntegration()
    
    # Create model and perform backward pass
    model = nn.Linear(5, 1)
    input_tensor = torch.randn(1, 5)
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()
    
    grad_norm = integration.calculate_gradient_norm(model)
    
    if integration.config.gradient_monitoring:
        assert isinstance(grad_norm, (float, type(None)))
        if grad_norm is not None:
            assert grad_norm >= 0.0


def test_monitoring_summary_generation() -> None:
    # Test generating monitoring summary
    integration = ModelMonitoringIntegration()
    integration.start_monitoring()
    
    model = nn.Linear(3, 1)
    integration.register_model(model, "summary_model")
    
    summary = integration.get_monitoring_summary()
    
    assert "monitoring_active" in summary
    assert "config" in summary
    assert summary["monitoring_active"] == True
    
    if integration.config.enable_model_monitoring:
        assert "model_summaries" in summary
    
    if integration.config.enable_system_monitoring:
        assert "system_summary" in summary
    
    integration.stop_monitoring()


def test_metrics_export_functionality() -> None:
    # Test exporting metrics to file
    import tempfile
    import json
    import os
    
    monitor = ModelPerformanceMonitor()
    
    # Add some training metrics
    for i in range(3):
        monitor.track_training_step(i, i, 0.5 - i * 0.1, 0.01)
    
    # Use pytest's temporary directory
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp:
        export_path = temp.name
    
    try:
        # Export and validate
        success = monitor.export_metrics(export_path, format="json")
        
        assert success
        assert os.path.exists(export_path)
        
        with open(export_path, 'r') as f:
            data = json.load(f)
            assert "training_metrics" in data
    finally:
        # Clean up
        if os.path.exists(export_path):
            os.unlink(export_path)


def test_system_metrics_export() -> None:
    # Test exporting system metrics
    import tempfile
    import json
    import os
    
    monitor = SystemResourceMonitor()
    
    # Add some system metrics
    for i in range(3):
        metrics = SystemMetrics(cpu_percent=50.0 + i * 5)
        monitor.metrics_history.append(metrics)
    
    # Use pytest's temporary directory
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp:
        export_path = temp.name
    
    try:
        # Export and validate
        success = monitor.export_metrics(export_path, format="json")
        
        assert success
        assert os.path.exists(export_path)
    finally:
        # Clean up
        if os.path.exists(export_path):
            os.unlink(export_path)


def test_configuration_integration() -> None:
    # Test configuration handling
    config = MonitoringConfig()
    
    assert config.enable_model_monitoring is not None
    assert config.enable_system_monitoring is not None
    assert config.model_history_size is not None
    assert config.system_update_interval is not None


def test_alert_callback_system() -> None:
    # Test alert callback system
    monitor = SystemResourceMonitor()
    
    # Track callback execution
    callback_executed = False
    
    def test_callback(message: str) -> None:
        nonlocal callback_executed
        callback_executed = True
    
    monitor.add_alert_callback(test_callback)
    
    # Create metrics that would trigger an alert
    high_memory_metrics = SystemMetrics(memory_percent=90.0)
    monitor._check_alerts(high_memory_metrics)
    
    assert callback_executed


def test_data_integrity_and_cleanup() -> None:
    # Test data integrity with history size limits
    monitor = ModelPerformanceMonitor(max_history=3)
    
    # Add more items than history size
    for i in range(5):
        monitor.track_training_step(0, i, 0.1, 0.01)
    
    # Should keep only most recent 3
    assert len(monitor.training_metrics) <= 3
    
    monitor.clear_metrics()
    assert len(monitor.training_metrics) == 0


def test_concurrent_monitoring_access() -> None:
    # Test accessing monitor while it's running
    monitor = SystemResourceMonitor(update_interval=0.1)
    monitor.start_monitoring()
    
    time.sleep(0.2)  # Allow collection of some metrics
    
    metrics1 = monitor.get_current_metrics()
    metrics2 = monitor.get_metrics_history(minutes=1)
    summary = monitor.get_summary_stats(minutes=1)
    
    assert metrics1 is not None
    assert isinstance(metrics2, list)
    assert isinstance(summary, dict)
    
    monitor.stop_monitoring()


def test_gpu_metrics_handling() -> None:
    # Test GPU metrics collection
    monitor = SystemResourceMonitor()
    metrics = monitor._collect_metrics()
    
    assert isinstance(metrics.gpu_utilization, list)
    assert isinstance(metrics.gpu_memory_used, list)
    assert isinstance(metrics.gpu_temperature, list)


def test_network_io_tracking() -> None:
    # Test network IO tracking
    monitor = SystemResourceMonitor()
    metrics = monitor._collect_metrics()
    
    assert 'bytes_sent' in metrics.network_io
    assert 'bytes_recv' in metrics.network_io
    assert 'packets_sent' in metrics.network_io
    assert 'packets_recv' in metrics.network_io


def test_threshold_customization() -> None:
    # Test customizing alert thresholds
    monitor = SystemResourceMonitor()
    
    original_thresholds = monitor.thresholds.copy()
    
    monitor.set_thresholds(
        cpu_percent=75.0,
        memory_percent=80.0,
        gpu_utilization=85.0
    )
    
    assert monitor.thresholds['cpu_percent'] == 75.0
    assert monitor.thresholds['memory_percent'] == 80.0
    assert monitor.thresholds['gpu_utilization'] == 85.0


def test_integration_export_all_metrics() -> None:
    # Test exporting all metrics from integration
    import tempfile
    import os
    
    integration = ModelMonitoringIntegration()
    integration.start_monitoring()
    
    model = nn.Linear(2, 1)
    integration.register_model(model, "export_test")
    integration.track_training_step(0, 1, 0.4, 0.01)
    
    # Use pytest's temporary directory
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp:
        export_path = temp.name
    
    try:
        success = integration.export_all_metrics(export_path)
        integration.stop_monitoring()
    finally:
        # Clean up
        if os.path.exists(export_path):
            os.unlink(export_path)
        if os.path.exists(export_path.replace('.json', '_model.json')):
            os.unlink(export_path.replace('.json', '_model.json'))
        if os.path.exists(export_path.replace('.json', '_system.json')):
            os.unlink(export_path.replace('.json', '_system.json'))


def test_model_metrics_dataclass_functionality() -> None:
    # Test ModelMetrics dataclass functionality
    metrics = ModelMetrics(
        model_name="test",
        inference_time=0.05,
        memory_usage=128.0,
        input_shape=(1, 10),
        parameters_count=100
    )
    
    assert metrics.model_name == "test"
    assert metrics.inference_time == 0.05
    assert metrics.memory_usage == 128.0
    assert isinstance(metrics.timestamp, datetime)


def test_training_metrics_dataclass_functionality() -> None:
    # Test TrainingMetrics dataclass functionality
    metrics = TrainingMetrics(
        epoch=5,
        step=100,
        loss=0.25,
        learning_rate=0.001,
        gradient_norm=0.8
    )
    
    assert metrics.epoch == 5
    assert metrics.step == 100
    assert metrics.loss == 0.25
    assert metrics.gradient_norm == 0.8
    assert isinstance(metrics.timestamp, datetime)


def test_monitoring_decorator_integration() -> None:
    # Test the monitoring decorator functionality
    from monitoring.model_integration import monitor_model_inference
    
    integration = ModelMonitoringIntegration()
    
    @monitor_model_inference(integration, "decorated_model")
    def dummy_inference(input_tensor: torch.Tensor) -> torch.Tensor:
        return torch.randn(1, 2)
    
    input_data = torch.randn(1, 5)
    result = dummy_inference(input_data)
    
    assert result is not None


def test_error_handling_in_monitoring() -> None:
    # Test handling errors in monitoring
    import pytest
    from unittest.mock import patch
    
    monitor = SystemResourceMonitor()
    
    # Simulate an error in cpu_percent
    with patch('psutil.cpu_percent', side_effect=Exception("Test error")):
        # Should handle the exception gracefully
        try:
            monitor.start_monitoring()
            time.sleep(0.1)
            monitor.stop_monitoring()
            # If we get here, the error was handled properly
            assert True
        except Exception:
            pytest.fail("Monitoring should handle internal errors gracefully")


def test_memory_management_in_long_running_monitor() -> None:
    # Test memory management with limited history size
    monitor = ModelPerformanceMonitor(max_history=5)
    
    # Add more items than the history limit
    for i in range(10):
        monitor.track_training_step(0, i, 0.1, 0.01)
    
    # Should maintain only max_history items
    assert len(monitor.training_metrics) == 5