# FILE: tests/test_exports.py
# ehsanasgharzde - TESTS FOR EXPORT FUNCTIONALITY
# hosseinsolymanzadeh - PROPER COMMENTING

import torch
import torch.nn as nn
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
from utils.export import (
    ExportConfig, ExportFormat, OptimizationLevel, QuantizationType, Precision,
    ModelOptimizer, QuantizationHandler, ONNXExporter, TorchScriptExporter,
    ExportManager, DeploymentTester, export_onnx, export_torchscript,
    print_deployment_plan, ExportError, OptimizationError, QuantizationError,
    ValidationError, DeploymentError
)
from models.edepth import edepth

class TestExportConfig:
    # Test initialization of ExportConfig with specific parameters
    def test_config_initialization(self):
        config = ExportConfig(
            export_format=ExportFormat.ONNX,
            batch_size=2,
            input_shape=(2, 3, 224, 224)
        )
        # Check if export_format is set correctly
        assert config.export_format == ExportFormat.ONNX
        # Check if batch_size is set correctly
        assert config.batch_size == 2
        # Check if input_shape is set correctly
        assert config.input_shape == (2, 3, 224, 224)
        # Check default optimization_level
        assert config.optimization_level == OptimizationLevel.NONE
        # Check default quantization_type
        assert config.quantization_type == QuantizationType.NONE
        # Check default precision
        assert config.precision == Precision.FP32

    # Test the post-initialization behavior of ExportConfig
    def test_config_post_init(self):
        # Create a temporary directory for output_dir
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ExportConfig(
                export_format=ExportFormat.ONNX,
                output_dir=temp_dir
            )
            # Verify the temporary directory exists
            assert os.path.exists(temp_dir)


class TestModelOptimizer:
    # Test optimization when no optimization is requested
    def test_no_optimization(self):
        config = ExportConfig(
            export_format=ExportFormat.ONNX,
            optimization_level=OptimizationLevel.NONE
        )
        optimizer = ModelOptimizer(config)
        model = nn.Linear(10, 5)
        # Optimize model with no optimization level
        optimized = optimizer.optimize_model(model)
        # The optimized model should be the same as the input model
        assert optimized is model

    # Test basic level optimization on a simple CNN model
    def test_basic_optimization(self):
        config = ExportConfig(
            export_format=ExportFormat.ONNX,
            optimization_level=OptimizationLevel.BASIC
        )
        optimizer = ModelOptimizer(config)
        # Define a simple CNN model
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # Apply optimization
        optimized = optimizer.optimize_model(model)
        # The optimized model should not be None
        assert optimized is not None

    # Test advanced level optimization on a linear model
    def test_advanced_optimization(self):
        config = ExportConfig(
            export_format=ExportFormat.ONNX,
            optimization_level=OptimizationLevel.ADVANCED
        )
        optimizer = ModelOptimizer(config)
        model = nn.Linear(10, 5)
        # Apply advanced optimization
        optimized = optimizer.optimize_model(model)
        # The optimized model should not be None
        assert optimized is not None

    # Test validation method for optimized models
    def test_validate_optimized_model(self):
        config = ExportConfig(
            export_format=ExportFormat.ONNX,
            input_shape=(1, 10, 10, 10)
        )
        optimizer = ModelOptimizer(config)
        model = nn.Linear(10, 5)
        # Create a separate optimized model and load weights from original model
        optimized_model = nn.Linear(10, 5)
        optimized_model.load_state_dict(model.state_dict())
        
        # Validate if optimized model is equivalent to original
        is_valid = optimizer.validate_optimized_model(model, optimized_model)
        # Expect validation to return True
        assert is_valid is True


class TestQuantizationHandler:
    # Test dynamic quantization preparation returns the original model
    def test_dynamic_quantization(self):
        handler = QuantizationHandler(QuantizationType.DYNAMIC)
        model = nn.Linear(10, 5)
        prepared = handler.prepare_model_for_quantization(model)
        assert prepared is model

    # Test static quantization preparation adds qconfig attribute to the model
    def test_static_quantization_preparation(self):
        handler = QuantizationHandler(QuantizationType.STATIC)
        model = nn.Linear(10, 5)
        prepared = handler.prepare_model_for_quantization(model)
        assert hasattr(prepared, 'qconfig')

    # Test quantization aware training preparation adds qconfig attribute to the model
    def test_qat_preparation(self):
        handler = QuantizationHandler(QuantizationType.QAT)
        model = nn.Linear(10, 5)
        prepared = handler.prepare_model_for_quantization(model)
        assert hasattr(prepared, 'qconfig')

    # Test conversion of model to quantized version returns a non-None object
    def test_convert_to_quantized(self):
        handler = QuantizationHandler(QuantizationType.DYNAMIC)
        model = nn.Linear(10, 5)
        quantized = handler.convert_to_quantized(model)
        assert quantized is not None

class TestONNXExporter:
    # Test ONNX export creates a file at the specified path
    def test_onnx_export(self):
        config = ExportConfig(
            export_format=ExportFormat.ONNX,
            input_shape=(1, 3, 224, 224)
        )
        exporter = ONNXExporter(config)
        model = nn.Conv2d(3, 64, 3)
        
        # Create a temporary file path for the exported ONNX model
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            export_path = f.name
            
        try:
            # Export the model and check if the file exists
            exporter.export(model, export_path)
            assert os.path.exists(export_path)
        finally:
            # Clean up the temporary file if it still exists
            if os.path.exists(export_path):
                os.unlink(export_path)

    # Test running inference on the exported ONNX model produces output with expected batch size
    def test_onnx_inference(self):
        config = ExportConfig(
            export_format=ExportFormat.ONNX,
            input_shape=(1, 3, 224, 224)
        )
        exporter = ONNXExporter(config)
        model = nn.Conv2d(3, 64, 3)
        
        # Create a temporary file path for the exported ONNX model
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            export_path = f.name
            
        try:
            # Export the model to ONNX
            exporter.export(model, export_path)
            # Prepare dummy input for inference
            dummy_input = torch.randn(1, 3, 224, 224)
            # Run ONNX inference and check output validity
            output = exporter.test_onnx_inference(export_path, dummy_input)
            assert output is not None
            assert output.shape[0] == 1
        finally:
            # Clean up the temporary ONNX file
            if os.path.exists(export_path):
                os.unlink(export_path)

class TestTorchScriptExporter:
    # Test export of model using TorchScript tracing creates a file
    def test_torchscript_trace_export(self):
        config = ExportConfig(
            export_format=ExportFormat.TORCHSCRIPT_TRACE,
            input_shape=(1, 3, 224, 224)
        )
        exporter = TorchScriptExporter(config)
        model = nn.Conv2d(3, 64, 3)
        
        # Temporary file for the TorchScript traced model
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            export_path = f.name
            
        try:
            # Export the model and verify file creation
            exporter.export(model, export_path)
            assert os.path.exists(export_path)
        finally:
            # Remove temporary file
            if os.path.exists(export_path):
                os.unlink(export_path)

    # Test export of model using TorchScript scripting creates a file
    def test_torchscript_script_export(self):
        config = ExportConfig(
            export_format=ExportFormat.TORCHSCRIPT_SCRIPT,
            input_shape=(1, 3, 224, 224)
        )
        exporter = TorchScriptExporter(config)
        model = nn.Conv2d(3, 64, 3)
        
        # Temporary file for the TorchScript scripted model
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            export_path = f.name
            
        try:
            # Export the scripted model and verify file creation
            exporter.export(model, export_path)
            assert os.path.exists(export_path)
        finally:
            # Remove temporary file
            if os.path.exists(export_path):
                os.unlink(export_path)

    # Test inference using TorchScript model produces output with expected batch size
    def test_torchscript_inference(self):
        config = ExportConfig(
            export_format=ExportFormat.TORCHSCRIPT_TRACE,
            input_shape=(1, 3, 224, 224)
        )
        exporter = TorchScriptExporter(config)
        model = nn.Conv2d(3, 64, 3)
        
        # Temporary file for the TorchScript model
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            export_path = f.name
            
        try:
            # Export the model
            exporter.export(model, export_path)
            # Prepare dummy input tensor
            dummy_input = torch.randn(1, 3, 224, 224)
            # Run inference and check the output tensor
            output = exporter.test_torchscript_inference(export_path, dummy_input)
            assert output is not None
            assert output.shape[0] == 1
        finally:
            # Clean up the temporary file
            if os.path.exists(export_path):
                os.unlink(export_path)


class TestDeploymentTester:
    def test_model_accuracy_test(self):
        # Create DeploymentTester instance with empty config
        tester = DeploymentTester({})
        # Create two linear models with same dimensions
        model1 = nn.Linear(10, 5)
        model2 = nn.Linear(10, 5)
        # Copy state dict from model1 to model2 to make them identical
        model2.load_state_dict(model1.state_dict())
        
        # Generate dummy input tensor
        dummy_input = torch.randn(1, 10)
        # Test accuracy between the two models using dummy input
        accuracy = tester.test_model_accuracy(model1, model2, dummy_input)
        # Assert that accuracy test returns True (models match)
        assert accuracy is True

    def test_model_performance_test(self):
        # Create DeploymentTester instance
        tester = DeploymentTester({})
        # Create a linear model
        model = nn.Linear(10, 5)
        # Generate dummy input tensor
        dummy_input = torch.randn(1, 10)
        
        # Test model performance on dummy input for 10 runs
        performance = tester.test_model_performance(model, dummy_input, num_runs=10)
        # Assert performance metric is positive
        assert performance > 0

    def test_memory_usage_test(self):
        # Create DeploymentTester instance
        tester = DeploymentTester({})
        # Create a linear model
        model = nn.Linear(10, 5)
        # Generate dummy input tensor
        dummy_input = torch.randn(1, 10)
        
        # Test model memory usage with dummy input
        memory = tester.test_memory_usage(model, dummy_input)
        # Assert memory usage is non-negative
        assert memory >= 0

    def test_generate_report(self):
        # Create DeploymentTester instance
        tester = DeploymentTester({})
        # Example results dictionary with metrics
        results = {'accuracy': True, 'performance': 0.001, 'memory': 50.0}
        
        # Generate textual report from results
        report = tester.generate_test_report(results)
        # Check that report contains each metric's string representation
        assert "accuracy: True" in report
        assert "performance: 0.001" in report
        assert "memory: 50.0" in report

class TestExportManager:
    def test_export_manager_initialization(self):
        # Create export configuration with ONNX format
        config = ExportConfig(export_format=ExportFormat.ONNX)
        # Initialize ExportManager with config
        manager = ExportManager(config)
        # Assert manager's config matches input config
        assert manager.config == config
        # Assert optimizer is initialized
        assert manager.optimizer is not None
        # Assert quantization handler is initialized
        assert manager.quantization_handler is not None
        # Assert tester instance is initialized
        assert manager.tester is not None

    def test_validate_export_config(self):
        # Create export config with ONNX format
        config = ExportConfig(export_format=ExportFormat.ONNX)
        # Initialize ExportManager
        manager = ExportManager(config)
        # Validate config does not raise exceptions
        manager.validate_export_config(config)

    def test_validate_invalid_config(self):
        # Create export config with ONNX format
        config = ExportConfig(export_format=ExportFormat.ONNX)
        # Set invalid input shape (4D shape likely invalid for validation)
        config.input_shape = (1, 10, 10, 10)
        # Initialize ExportManager
        manager = ExportManager(config)
        
        # Expect ValueError raised during validation due to invalid config
        with pytest.raises(ValueError):
            manager.validate_export_config(config)

    def test_prepare_model_for_export(self):
        # Create export config with ONNX format
        config = ExportConfig(export_format=ExportFormat.ONNX)
        # Initialize ExportManager
        manager = ExportManager(config)
        # Create a linear model
        model = nn.Linear(10, 5)
        
        # Prepare model for export (e.g., convert or optimize)
        prepared = manager.prepare_model_for_export(model)
        # Assert preparation returns a non-None object
        assert prepared is not None

    def test_export_model_onnx(self):
        # Use temporary directory to save export
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create export config for ONNX with output dir and input shape
            config = ExportConfig(
                export_format=ExportFormat.ONNX,
                output_dir=temp_dir,
                input_shape=(1, 3, 224, 224)
            )
            # Initialize ExportManager
            manager = ExportManager(config)
            # Create a Conv2d model
            model = nn.Conv2d(3, 64, 3)
            
            # Export model to ONNX format
            export_path = manager.export_model(model)
            # Assert the export path exists in filesystem
            assert os.path.exists(export_path)
            # Assert the file extension is .onnx
            assert export_path.endswith('.onnx')

    def test_export_model_torchscript(self):
        # Use temporary directory for saving export
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create export config for TorchScript tracing with output dir and input shape
            config = ExportConfig(
                export_format=ExportFormat.TORCHSCRIPT_TRACE,
                output_dir=temp_dir,
                input_shape=(1, 3, 224, 224)
            )
            # Initialize ExportManager
            manager = ExportManager(config)
            # Create a Conv2d model
            model = nn.Conv2d(3, 64, 3)
            
            # Export model to TorchScript format
            export_path = manager.export_model(model)
            # Assert export file exists
            assert os.path.exists(export_path)
            # Assert file has .pt extension
            assert export_path.endswith('.pt')

    def test_batch_export(self):
        # Use temporary directory for saving exports
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create export config for ONNX with output dir and input shape
            config = ExportConfig(
                export_format=ExportFormat.ONNX,
                output_dir=temp_dir,
                input_shape=(1, 3, 224, 224)
            )
            # Initialize ExportManager
            manager = ExportManager(config)
            # Create a Conv2d model
            model = nn.Conv2d(3, 64, 3)
            
            # List of export formats to batch export
            formats = [ExportFormat.ONNX, ExportFormat.TORCHSCRIPT_TRACE]
            # Perform batch export returning results dict
            results = manager.batch_export(model, formats)
            
            # Assert results contain two entries (one per format)
            assert len(results) == 2
            # Assert ONNX export key is in results
            assert ExportFormat.ONNX.value in results
            # Assert TorchScript export key is in results
            assert ExportFormat.TORCHSCRIPT_TRACE.value in results

    def test_unsupported_export_format(self):
        # Create export config with unsupported format TensorRT
        config = ExportConfig(export_format=ExportFormat.TENSORRT)
        # Initialize ExportManager
        manager = ExportManager(config)
        # Create a linear model
        model = nn.Linear(10, 5)
        
        # Expect NotImplementedError when exporting with unsupported format
        with pytest.raises(NotImplementedError):
            manager.export_model(model)


class TestUtilityFunctions:
    def test_export_onnx_function(self):
        model = nn.Linear(10, 5)
        
        # Create a temporary file with .onnx suffix, not deleted automatically
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            export_path = f.name
            
        try:
            # Export the model to ONNX format and check result path matches export path
            result_path = export_onnx(model, export_path, input_shape=(1, 10))
            assert result_path == export_path
            # Verify the exported file exists
            assert os.path.exists(export_path)
        finally:
            # Clean up the temporary file if it still exists
            if os.path.exists(export_path):
                os.unlink(export_path)

    def test_export_torchscript_function(self):
        model = nn.Linear(10, 5)
        
        # Create a temporary file with .pt suffix, not deleted automatically
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            export_path = f.name
            
        try:
            # Export the model to TorchScript format and check result path
            result_path = export_torchscript(model, export_path, input_shape=(1, 10))
            assert result_path == export_path
            # Verify the exported file exists
            assert os.path.exists(export_path)
        finally:
            # Clean up the temporary file if it still exists
            if os.path.exists(export_path):
                os.unlink(export_path)

    def test_print_deployment_plan(self):
        model = nn.Linear(10, 5)
        
        # Patch built-in print function to monitor its calls
        with patch('builtins.print') as mock_print:
            # Call the deployment plan print function for "edge" deployment
            print_deployment_plan(model, "edge")
            # Assert that print was called at least once
            mock_print.assert_called()

class TestEdgeIntegration:
    def test_edepth_model_export(self):
        # Use a temporary directory to store export output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup export configuration for ONNX with specified input shape
            config = ExportConfig(
                export_format=ExportFormat.ONNX,
                output_dir=temp_dir,
                input_shape=(1, 3, 224, 224)
            )
            manager = ExportManager(config)
            
            # Initialize the edepth model with specified backbone, no pretrained weights
            model = edepth(
                backbone_name='vit_base_patch16_224',
                pretrained=False
            )
            
            # Export the model and get the path
            export_path = manager.export_model(model)
            # Verify the exported file exists
            assert os.path.exists(export_path)

    def test_edepth_model_testing(self):
        # Use a temporary directory to store export output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup export configuration for TorchScript tracing
            config = ExportConfig(
                export_format=ExportFormat.TORCHSCRIPT_TRACE,
                output_dir=temp_dir,
                input_shape=(1, 3, 224, 224)
            )
            manager = ExportManager(config)
            
            # Initialize the edepth model
            model = edepth(
                backbone_name='vit_base_patch16_224',
                pretrained=False
            )
            
            # Export the model
            export_path = manager.export_model(model)
            # Test the exported model and capture results
            test_results = manager.test_exported_model(model, export_path)
            
            # Check that the test passed and relevant keys are present in results
            assert test_results['test_passed'] is True
            assert 'accuracy_test' in test_results
            assert 'avg_inference_time' in test_results
            assert 'memory_usage_mb' in test_results

class TestErrorHandling:
    def test_invalid_model_type(self):
        # Setup exporter with ONNX config
        config = ExportConfig(export_format=ExportFormat.ONNX)
        exporter = ONNXExporter(config)
        
        # Validate model with invalid input type, expecting ValueError
        with pytest.raises(ValueError):
            exporter.validate_model("not a model") #type: ignore 

    def test_missing_export_file(self):
        # Setup exporter with ONNX config
        config = ExportConfig(export_format=ExportFormat.ONNX)
        exporter = ONNXExporter(config)
        
        # Validate a non-existent ONNX file, expecting an Exception
        with pytest.raises(Exception):
            exporter.validate_onnx_model("nonexistent.onnx")

    def test_export_error_handling(self):
        # Setup export manager with ONNX config
        config = ExportConfig(export_format=ExportFormat.ONNX)
        manager = ExportManager(config)
        
        # Create a mock model that raises exception on eval()
        invalid_model = MagicMock()
        invalid_model.eval = MagicMock(side_effect=Exception("Model error"))
        
        # Attempt to export invalid model, expecting an Exception
        with pytest.raises(Exception):
            manager.export_model(invalid_model)

class TestPerformanceOptimization:
    def test_optimization_validation(self):
        # Setup optimizer config with BASIC optimization level for ONNX export
        config = ExportConfig(
            export_format=ExportFormat.ONNX,
            optimization_level=OptimizationLevel.BASIC
        )
        optimizer = ModelOptimizer(config)
        
        # Create original and optimized models
        original_model = nn.Linear(10, 5)
        optimized_model = optimizer.optimize_model(original_model)
        
        # Validate optimized model against original model
        is_valid = optimizer.validate_optimized_model(original_model, optimized_model)
        assert is_valid is True

    def test_quantization_validation(self):
        # Create quantization handler with dynamic quantization type
        handler = QuantizationHandler(QuantizationType.DYNAMIC)
        
        # Create original model and quantized version
        original_model = nn.Linear(10, 5)
        quantized_model = handler.convert_to_quantized(original_model)
        
        # Validate quantized model
        is_valid = handler.validate_quantized_model(original_model, quantized_model)
        # Check that validation result is boolean
        assert isinstance(is_valid, bool)

class TestCompatibility:
    def test_different_input_shapes(self):
        # Test various input shapes for dummy input creation
        shapes = [(1, 3, 224, 224), (1, 3, 256, 256), (2, 3, 224, 224)]
        
        for shape in shapes:
            # Setup exporter config for each input shape
            config = ExportConfig(
                export_format=ExportFormat.ONNX,
                input_shape=shape
            )
            exporter = ONNXExporter(config)
            # Create dummy input tensor and verify shape matches
            dummy_input = exporter.create_dummy_input(shape)
            assert dummy_input.shape == shape

    def test_different_batch_sizes(self):
        # Test multiple batch sizes for dummy input creation
        batch_sizes = [1, 2, 4, 8]
        
        for batch_size in batch_sizes:
            # Setup exporter config for each batch size
            config = ExportConfig(
                export_format=ExportFormat.ONNX,
                batch_size=batch_size,
                input_shape=(batch_size, 3, 224, 224)
            )
            exporter = ONNXExporter(config)
            # Create dummy input with specified batch size and verify batch dimension
            dummy_input = exporter.create_dummy_input((1, 3, 224, 224), batch_size)
            assert dummy_input.shape[0] == batch_size

class TestRegressionPrevention:
    def test_consistent_export_results(self):
        model = nn.Linear(10, 5)
        
        # Use temporary directory for exports
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup export config for ONNX with 4D input shape
            config = ExportConfig(
                export_format=ExportFormat.ONNX,
                output_dir=temp_dir,
                input_shape=(1, 10, 10, 10)
            )
            manager = ExportManager(config)
            
            # Export the model twice and get paths
            path1 = manager.export_model(model)
            path2 = manager.export_model(model)
            
            # Verify both export files exist
            assert os.path.exists(path1)
            assert os.path.exists(path2)

    def test_export_determinism(self):
        model = nn.Linear(10, 5)
        # Fix random seed for reproducibility
        torch.manual_seed(42)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup export config for TorchScript tracing
            config = ExportConfig(
                export_format=ExportFormat.TORCHSCRIPT_TRACE,
                output_dir=temp_dir,
                input_shape=(1, 10, 10, 10 )
            )
            manager = ExportManager(config)
            
            # Export the model and load it back
            export_path = manager.export_model(model)
            loaded_model = torch.jit.load(export_path) #type: ignore 
            
            # Create random dummy input
            dummy_input = torch.randn(1, 10)
            with torch.no_grad():
                # Run inference twice and compare outputs for determinism
                output1 = loaded_model(dummy_input)
                output2 = loaded_model(dummy_input)
            
            assert torch.allclose(output1, output2)
