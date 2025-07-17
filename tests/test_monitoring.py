#FILE: tests/test_monitoring.py
# ehsanasgharzde - TESTING MONITORING SYSTEM
# hosseinsolymanzadeh - PROPER COMMENTING

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
        # Initialize configuration and logger for the test cases
        self.config = MonitoringConfig()
        self.logger = logging.getLogger("test_monitor")
        self.logger.setLevel(logging.INFO)

    def test_comprehensive_monitor_initialization(self):
        # Test if ComprehensiveMonitor initializes all subcomponents correctly
        monitor = ComprehensiveMonitor(self.config)
        
        self.assertIsNotNone(monitor.data_monitor)      # Check data monitor initialized
        self.assertIsNotNone(monitor.system_monitor)    # Check system monitor initialized
        self.assertIsNotNone(monitor.training_monitor)  # Check training monitor initialized
        self.assertIsNotNone(monitor.profiler)          # Check performance profiler initialized
        self.assertIsNotNone(monitor.visualizer)        # Check visualizer initialized
        self.assertIsNotNone(monitor.hooks_manager)     # Check hooks manager initialized
        self.assertIsNotNone(monitor.integration)       # Check external integration initialized
        self.assertFalse(monitor.state.is_active)       # Monitoring should not be active initially

    def test_monitoring_start_stop_cycle(self):
        # Test lifecycle of monitoring (start → stop)
        monitor = ComprehensiveMonitor(self.config)
        
        self.assertFalse(monitor.state.is_active)  # Ensure inactive before starting
        
        monitor.start_monitoring()
        self.assertTrue(monitor.state.is_active)   # Should be active after start
        self.assertIsNotNone(monitor.state.start_time)  # Start time should be recorded
        
        report = monitor.stop_monitoring()
        self.assertFalse(monitor.state.is_active)       # Should be inactive after stop
        self.assertIsNotNone(monitor.state.end_time)    # End time should be recorded
        self.assertIsInstance(report, dict)             # Report should be a dictionary
        self.assertIn("monitoring_session", report)     # Should include session info

    def test_system_monitor_functionality(self):
        # Test system metrics collection and alert logic
        system_monitor = SystemMonitor(monitoring_interval=0.1, history_size=10)
        
        metrics = system_monitor.collect_system_metrics()  # Collect metrics snapshot
        self.assertIsNotNone(metrics.cpu_percent)          # CPU usage must be present
        self.assertIsNotNone(metrics.memory_percent)       # Memory usage must be present
        self.assertIsNotNone(metrics.timestamp)            # Timestamp must be present
        
        alerts = system_monitor.check_resource_alerts(metrics)  # Check if any alerts are triggered
        self.assertIsInstance(alerts, list)                     # Alerts should be returned as list
        
        summary = system_monitor.get_performance_summary()      # Get performance summary
        self.assertIsInstance(summary, dict)                    # Should be a dictionary
        self.assertIn("avg_cpu_percent", summary)               # Key should exist in summary

    def test_data_monitor_tensor_tracking(self):
        # Test registration and tracking of tensors in DataMonitor
        data_monitor = DataMonitor(
            track_gradients=True,
            track_activations=True,
            sample_rate=1.0
        )
        
        test_tensor = torch.randn(10, 10, requires_grad=True)   # Create test tensor
        tensor_id = data_monitor.register_tensor(test_tensor, "test_tensor", "activation")
        
        self.assertIsNotNone(tensor_id)                         # Ensure tensor is registered
        self.assertIn(tensor_id, data_monitor.tensor_registry)  # Tensor ID should exist in registry
        
        metrics = data_monitor.track_tensor_stats(test_tensor, "test_tensor")  # Track stats
        self.assertIsNotNone(metrics.mean_value)                             # Mean should be tracked
        self.assertIsNotNone(metrics.std_value)                              # Std deviation should be tracked
        self.assertEqual(metrics.tensor_id, tensor_id)                       # IDs should match
        
        data_monitor.cleanup_hooks()  # Clean up any hooks after tracking

    def test_training_monitor_step_logging(self):
        # Test logging of training steps and anomaly detection
        training_monitor = TrainingMonitor(
            log_interval=1,
            checkpoint_interval=5,
            logger=self.logger
        )
        
        training_monitor.start_training_session()  # Begin new training session
        
        for step in range(10):
            loss = 1.0 / (step + 1)                      # Simulate decreasing loss
            lr = 0.01 * (0.9 ** step)                    # Simulate decaying learning rate
            metrics = {"accuracy": 0.8 + step * 0.02}    # Simulate accuracy metric
            
            training_monitor.log_training_step(step, loss, metrics, lr)  # Log each step
        
        self.assertEqual(len(training_monitor.training_history), 10)  # Ensure 10 steps logged
        self.assertEqual(training_monitor.current_step, 9)            # Last step index
        
        convergence_analysis = training_monitor.analyze_loss_convergence()  # Analyze convergence
        self.assertIsInstance(convergence_analysis, dict)                  # Should return dict
        self.assertIn("status", convergence_analysis)                      # Key must exist
        
        anomalies = training_monitor.detect_training_anomalies()  # Detect anomalies
        self.assertIsInstance(anomalies, list)                    # Must return list

    def test_profiler_context_manager(self):
        # Test performance profiling using context manager
        profiler = PerformanceProfiler(
            profile_memory=True,
            profile_cuda=torch.cuda.is_available(),  # Enable CUDA profiling if available
            output_dir="test_profiler_output"
        )
        
        with profiler.profile_context("test_section"):
            result = torch.randn(100, 100).sum()  # Dummy computation for profiling
        
        self.assertIn("test_section", profiler.profiler_results)                  # Profile key should exist
        self.assertIsNotNone(profiler.profiler_results["test_section"]["duration"])  # Duration should be recorded

    def test_hooks_manager_model_integration(self):
        # Test hook registration and summary collection in a model
        data_monitor = DataMonitor()
        system_monitor = SystemMonitor()
        hooks_manager = MonitoringHooks(data_monitor, system_monitor, self.logger)
        
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
        )
        
        # Register hooks for all Linear layers
        for i, module in enumerate(model):
            if isinstance(module, torch.nn.Linear):
                hooks_manager.register_forward_hook(module, f"linear_{i}")
        
        self.assertGreater(len(hooks_manager.forward_hooks), 0)  # Hooks should be registered
        
        input_tensor = torch.randn(1, 10)        # Create test input
        output = model(input_tensor)             # Forward pass to trigger hooks
        
        summary = hooks_manager.get_hook_summary()  # Retrieve hook activity summary
        self.assertIn("forward_hooks", summary)     # Summary should include hook info
        
        hooks_manager.remove_all_hooks()            # Remove all registered hooks
        self.assertEqual(len(hooks_manager.forward_hooks), 0)  # No hooks should remain

    def test_monitoring_logger_functionality(self):
        # Initialize the monitoring logger with structured logging enabled
        logger = MonitoringLogger(
            log_dir="test_logs",
            log_level=logging.INFO,
            structured_logging=True
        )

        # Simulate logging of system metrics
        test_metrics = {"cpu": 50.0, "memory": 60.0}
        logger.log_system_metrics(test_metrics)

        # Simulate logging of an error event
        test_error = Exception("Test error")
        logger.log_error_event(test_error, {"context": "test"})

        # Simulate logging of an alert
        logger.log_alert("cpu_usage", "High CPU usage detected", "warning")

        # Retrieve and validate recent error logs
        recent_errors = logger.get_recent_errors()
        self.assertIsInstance(recent_errors, list)

        # Retrieve and validate recent alert logs
        recent_alerts = logger.get_recent_alerts()
        self.assertIsInstance(recent_alerts, list)

    def test_integration_with_model(self):
        # Initialize monitoring integration using configuration
        integration = MonitoringIntegration(self.config)

        # Define a simple model architecture
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
        )

        # Integrate monitoring hooks into the model
        integration.integrate_with_model(model)

        # Ensure hooks were successfully attached
        self.assertGreater(len(integration.hooks_manager.forward_hooks), 0)

        # Perform a forward pass to trigger hooks
        input_tensor = torch.randn(1, 10)
        output = model(input_tensor)

        # Retrieve and check integration status
        status = integration.get_current_status()
        self.assertIn("is_active", status)
        self.assertIn("metrics_collected", status)

    def test_data_validation_and_anomaly_detection(self):
        # Create a DataMonitor instance
        data_monitor = DataMonitor()

        # Validate a clean tensor and expect it to pass
        valid_tensor = torch.randn(10, 10)
        validation_result = data_monitor.validate_data_integrity(valid_tensor)
        self.assertTrue(validation_result["valid"])

        # Validate a tensor containing NaN values and expect failure
        invalid_tensor = torch.tensor([1.0, 2.0, float('nan'), 4.0])
        validation_result = data_monitor.validate_data_integrity(invalid_tensor)
        self.assertFalse(validation_result["valid"])
        self.assertGreater(validation_result["nan_count"], 0)

        # Run anomaly detection on valid data
        anomalies = data_monitor.detect_data_anomalies(valid_tensor)
        self.assertIsInstance(anomalies, list)

    def test_memory_usage_monitoring(self):
        # Create a performance profiler instance
        profiler = PerformanceProfiler()

        # Profile system and CUDA memory usage
        memory_stats = profiler.profile_memory_usage()
        self.assertIsInstance(memory_stats, dict)
        self.assertIn("system", memory_stats)

        # Validate presence of CUDA stats if GPU is available
        if torch.cuda.is_available():
            self.assertIn("cuda_0", memory_stats)

    def test_gradient_flow_monitoring(self):
        # Initialize DataMonitor with gradient tracking enabled
        data_monitor = DataMonitor(track_gradients=True)

        # Define a small neural network
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
        )

        # Monitor gradient flow in the model
        gradient_info = data_monitor.monitor_gradient_flow(model)
        self.assertIn("params_monitored", gradient_info)

        # Perform a forward and backward pass to activate hooks
        input_tensor = torch.randn(1, 10)
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        # Clean up gradient hooks
        data_monitor.cleanup_hooks()

    def test_configuration_validation(self):
        # Load and validate monitoring configuration
        config = MonitoringConfig()

        # Ensure all expected threshold/config sections exist
        self.assertIsNotNone(config.system_thresholds)
        self.assertIsNotNone(config.training_thresholds)
        self.assertIsNotNone(config.data_thresholds)
        self.assertIsNotNone(config.logging_config)
        self.assertIsNotNone(config.profiling_config)

        # Fetch and validate alert thresholds
        alert_thresholds = config.get_alert_thresholds()
        self.assertIsInstance(alert_thresholds, dict)
        self.assertIn("cpu_usage", alert_thresholds)

    def test_visualization_integration(self):
        # Initialize visualizer with configuration and logger
        visualizer = MonitoringVisualizer(self.config, MonitoringLogger())
        
        # Collect system metrics 5 times for visualization
        system_metrics = [
            SystemMonitor().collect_system_metrics()
            for _ in range(5)
        ]
        
        # Initialize and start a training session
        training_monitor = TrainingMonitor()
        training_monitor.start_training_session()
        
        # Simulate logging 10 training steps
        for i in range(10):
            training_monitor.log_training_step(i, 1.0/(i+1), {}, 0.01)
        
        # Generate and verify the existence of the plot
        plot_path = visualizer.plot_system_metrics(system_metrics)
        self.assertTrue(os.path.exists(plot_path) or plot_path == "")

    def test_comprehensive_workflow(self):
        # Create a comprehensive monitor instance
        monitor = ComprehensiveMonitor(self.config)
        
        # Define a simple neural network model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
        )
        
        # Start monitoring the model
        monitor.start_monitoring(model=model)
        
        # Simulate a training loop with 5 steps
        for step in range(5):
            input_tensor = torch.randn(1, 10)
            output = model(input_tensor)
            loss = output.sum()
            loss.backward()
            
            # Log training step with loss and dummy accuracy
            monitor.training_monitor.log_training_step(
                step, loss.item(), {"accuracy": 0.9}, 0.01
            )
        
        # Check real-time monitoring status
        status = monitor.get_real_time_status()
        self.assertTrue(status["is_active"])
        self.assertGreater(status["duration"], 0)
        
        # Stop monitoring and verify the report structure
        report = monitor.stop_monitoring()
        self.assertIsInstance(report, dict)
        self.assertIn("monitoring_session", report)

    def test_error_handling_and_recovery(self):
        # Initialize the monitor
        monitor = ComprehensiveMonitor(self.config)
        
        try:
            # Start monitoring session
            monitor.start_monitoring()
            
            # Create invalid tensor with NaN to simulate data error
            invalid_tensor = torch.tensor([float('nan')])
            monitor.data_monitor.validate_data_integrity(invalid_tensor)
            
            # Attempt to stop monitoring gracefully
            monitor.stop_monitoring()
            
        except Exception as e:
            # On exception, log error and clean up
            self.logger.error(f"Error in monitoring: {e}")
            monitor.cleanup()

    def test_multi_gpu_support(self):
        # Skip test if CUDA is not available
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        # Collect system metrics
        system_monitor = SystemMonitor()
        metrics = system_monitor.collect_system_metrics()
        
        # Validate GPU-related metrics if present
        if metrics.gpu_utilization:
            self.assertIsInstance(metrics.gpu_utilization, list)
            self.assertIsInstance(metrics.gpu_temperature, list)
            self.assertIsInstance(metrics.gpu_memory_used, list)

    def tearDown(self):
        import shutil
        
        # Directories to clean up after tests
        cleanup_dirs = [
            "test_logs",
            "test_profiler_output",
            "monitoring_output",
            "logs"
        ]
        
        # Remove each directory if it exists
        for dir_name in cleanup_dirs:
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name, ignore_errors=True)

if __name__ == "__main__":
    unittest.main()
