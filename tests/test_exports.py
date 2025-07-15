# FILE: tests/test_exports.py
# ehsanasgharzde - TESTS FOR EXPORT FUNCTIONALITY

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
from models.model_fixed import edepth

class TestExportConfig:
    def test_config_initialization(self):
        config = ExportConfig(
            export_format=ExportFormat.ONNX,
            batch_size=2,
            input_shape=(2, 3, 224, 224)
        )
        assert config.export_format == ExportFormat.ONNX
        assert config.batch_size == 2
        assert config.input_shape == (2, 3, 224, 224)
        assert config.optimization_level == OptimizationLevel.NONE
        assert config.quantization_type == QuantizationType.NONE
        assert config.precision == Precision.FP32

    def test_config_post_init(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ExportConfig(
                export_format=ExportFormat.ONNX,
                output_dir=temp_dir
            )
            assert os.path.exists(temp_dir)

class TestModelOptimizer:
    def test_no_optimization(self):
        config = ExportConfig(
            export_format=ExportFormat.ONNX,
            optimization_level=OptimizationLevel.NONE
        )
        optimizer = ModelOptimizer(config)
        model = nn.Linear(10, 5)
        optimized = optimizer.optimize_model(model)
        assert optimized is model

    def test_basic_optimization(self):
        config = ExportConfig(
            export_format=ExportFormat.ONNX,
            optimization_level=OptimizationLevel.BASIC
        )
        optimizer = ModelOptimizer(config)
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        optimized = optimizer.optimize_model(model)
        assert optimized is not None

    def test_advanced_optimization(self):
        config = ExportConfig(
            export_format=ExportFormat.ONNX,
            optimization_level=OptimizationLevel.ADVANCED
        )
        optimizer = ModelOptimizer(config)
        model = nn.Linear(10, 5)
        optimized = optimizer.optimize_model(model)
        assert optimized is not None

    def test_validate_optimized_model(self):
        config = ExportConfig(
            export_format=ExportFormat.ONNX,
            input_shape=(1, 10, 10, 10)
        )
        optimizer = ModelOptimizer(config)
        model = nn.Linear(10, 5)
        optimized_model = nn.Linear(10, 5)
        optimized_model.load_state_dict(model.state_dict())
        
        is_valid = optimizer.validate_optimized_model(model, optimized_model)
        assert is_valid is True

class TestQuantizationHandler:
    def test_dynamic_quantization(self):
        handler = QuantizationHandler(QuantizationType.DYNAMIC)
        model = nn.Linear(10, 5)
        prepared = handler.prepare_model_for_quantization(model)
        assert prepared is model

    def test_static_quantization_preparation(self):
        handler = QuantizationHandler(QuantizationType.STATIC)
        model = nn.Linear(10, 5)
        prepared = handler.prepare_model_for_quantization(model)
        assert hasattr(prepared, 'qconfig')

    def test_qat_preparation(self):
        handler = QuantizationHandler(QuantizationType.QAT)
        model = nn.Linear(10, 5)
        prepared = handler.prepare_model_for_quantization(model)
        assert hasattr(prepared, 'qconfig')

    def test_convert_to_quantized(self):
        handler = QuantizationHandler(QuantizationType.DYNAMIC)
        model = nn.Linear(10, 5)
        quantized = handler.convert_to_quantized(model)
        assert quantized is not None

class TestONNXExporter:
    def test_onnx_export(self):
        config = ExportConfig(
            export_format=ExportFormat.ONNX,
            input_shape=(1, 3, 224, 224)
        )
        exporter = ONNXExporter(config)
        model = nn.Conv2d(3, 64, 3)
        
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            export_path = f.name
            
        try:
            exporter.export(model, export_path)
            assert os.path.exists(export_path)
        finally:
            if os.path.exists(export_path):
                os.unlink(export_path)

    def test_onnx_inference(self):
        config = ExportConfig(
            export_format=ExportFormat.ONNX,
            input_shape=(1, 3, 224, 224)
        )
        exporter = ONNXExporter(config)
        model = nn.Conv2d(3, 64, 3)
        
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            export_path = f.name
            
        try:
            exporter.export(model, export_path)
            dummy_input = torch.randn(1, 3, 224, 224)
            output = exporter.test_onnx_inference(export_path, dummy_input)
            assert output is not None
            assert output.shape[0] == 1
        finally:
            if os.path.exists(export_path):
                os.unlink(export_path)

class TestTorchScriptExporter:
    def test_torchscript_trace_export(self):
        config = ExportConfig(
            export_format=ExportFormat.TORCHSCRIPT_TRACE,
            input_shape=(1, 3, 224, 224)
        )
        exporter = TorchScriptExporter(config)
        model = nn.Conv2d(3, 64, 3)
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            export_path = f.name
            
        try:
            exporter.export(model, export_path)
            assert os.path.exists(export_path)
        finally:
            if os.path.exists(export_path):
                os.unlink(export_path)

    def test_torchscript_script_export(self):
        config = ExportConfig(
            export_format=ExportFormat.TORCHSCRIPT_SCRIPT,
            input_shape=(1, 3, 224, 224)
        )
        exporter = TorchScriptExporter(config)
        model = nn.Conv2d(3, 64, 3)
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            export_path = f.name
            
        try:
            exporter.export(model, export_path)
            assert os.path.exists(export_path)
        finally:
            if os.path.exists(export_path):
                os.unlink(export_path)

    def test_torchscript_inference(self):
        config = ExportConfig(
            export_format=ExportFormat.TORCHSCRIPT_TRACE,
            input_shape=(1, 3, 224, 224)
        )
        exporter = TorchScriptExporter(config)
        model = nn.Conv2d(3, 64, 3)
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            export_path = f.name
            
        try:
            exporter.export(model, export_path)
            dummy_input = torch.randn(1, 3, 224, 224)
            output = exporter.test_torchscript_inference(export_path, dummy_input)
            assert output is not None
            assert output.shape[0] == 1
        finally:
            if os.path.exists(export_path):
                os.unlink(export_path)

class TestDeploymentTester:
    def test_model_accuracy_test(self):
        tester = DeploymentTester({})
        model1 = nn.Linear(10, 5)
        model2 = nn.Linear(10, 5)
        model2.load_state_dict(model1.state_dict())
        
        dummy_input = torch.randn(1, 10)
        accuracy = tester.test_model_accuracy(model1, model2, dummy_input)
        assert accuracy is True

    def test_model_performance_test(self):
        tester = DeploymentTester({})
        model = nn.Linear(10, 5)
        dummy_input = torch.randn(1, 10)
        
        performance = tester.test_model_performance(model, dummy_input, num_runs=10)
        assert performance > 0

    def test_memory_usage_test(self):
        tester = DeploymentTester({})
        model = nn.Linear(10, 5)
        dummy_input = torch.randn(1, 10)
        
        memory = tester.test_memory_usage(model, dummy_input)
        assert memory >= 0

    def test_generate_report(self):
        tester = DeploymentTester({})
        results = {'accuracy': True, 'performance': 0.001, 'memory': 50.0}
        
        report = tester.generate_test_report(results)
        assert "accuracy: True" in report
        assert "performance: 0.001" in report
        assert "memory: 50.0" in report

class TestExportManager:
    def test_export_manager_initialization(self):
        config = ExportConfig(export_format=ExportFormat.ONNX)
        manager = ExportManager(config)
        assert manager.config == config
        assert manager.optimizer is not None
        assert manager.quantization_handler is not None
        assert manager.tester is not None

    def test_validate_export_config(self):
        config = ExportConfig(export_format=ExportFormat.ONNX)
        manager = ExportManager(config)
        
        manager.validate_export_config(config)

    def test_validate_invalid_config(self):
        config = ExportConfig(export_format=ExportFormat.ONNX)
        config.input_shape = (1, 10, 10, 10)
        manager = ExportManager(config)
        
        with pytest.raises(ValueError):
            manager.validate_export_config(config)

    def test_prepare_model_for_export(self):
        config = ExportConfig(export_format=ExportFormat.ONNX)
        manager = ExportManager(config)
        model = nn.Linear(10, 5)
        
        prepared = manager.prepare_model_for_export(model)
        assert prepared is not None

    def test_export_model_onnx(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ExportConfig(
                export_format=ExportFormat.ONNX,
                output_dir=temp_dir,
                input_shape=(1, 3, 224, 224)
            )
            manager = ExportManager(config)
            model = nn.Conv2d(3, 64, 3)
            
            export_path = manager.export_model(model)
            assert os.path.exists(export_path)
            assert export_path.endswith('.onnx')

    def test_export_model_torchscript(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ExportConfig(
                export_format=ExportFormat.TORCHSCRIPT_TRACE,
                output_dir=temp_dir,
                input_shape=(1, 3, 224, 224)
            )
            manager = ExportManager(config)
            model = nn.Conv2d(3, 64, 3)
            
            export_path = manager.export_model(model)
            assert os.path.exists(export_path)
            assert export_path.endswith('.pt')

    def test_batch_export(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ExportConfig(
                export_format=ExportFormat.ONNX,
                output_dir=temp_dir,
                input_shape=(1, 3, 224, 224)
            )
            manager = ExportManager(config)
            model = nn.Conv2d(3, 64, 3)
            
            formats = [ExportFormat.ONNX, ExportFormat.TORCHSCRIPT_TRACE]
            results = manager.batch_export(model, formats)
            
            assert len(results) == 2
            assert ExportFormat.ONNX.value in results
            assert ExportFormat.TORCHSCRIPT_TRACE.value in results

    def test_unsupported_export_format(self):
        config = ExportConfig(export_format=ExportFormat.TENSORRT)
        manager = ExportManager(config)
        model = nn.Linear(10, 5)
        
        with pytest.raises(NotImplementedError):
            manager.export_model(model)

class TestUtilityFunctions:
    def test_export_onnx_function(self):
        model = nn.Linear(10, 5)
        
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            export_path = f.name
            
        try:
            result_path = export_onnx(model, export_path, input_shape=(1, 10))
            assert result_path == export_path
            assert os.path.exists(export_path)
        finally:
            if os.path.exists(export_path):
                os.unlink(export_path)

    def test_export_torchscript_function(self):
        model = nn.Linear(10, 5)
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            export_path = f.name
            
        try:
            result_path = export_torchscript(model, export_path, input_shape=(1, 10))
            assert result_path == export_path
            assert os.path.exists(export_path)
        finally:
            if os.path.exists(export_path):
                os.unlink(export_path)

    def test_print_deployment_plan(self):
        model = nn.Linear(10, 5)
        
        with patch('builtins.print') as mock_print:
            print_deployment_plan(model, "edge")
            mock_print.assert_called()

class TestEdgeIntegration:
    def test_edepth_model_export(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ExportConfig(
                export_format=ExportFormat.ONNX,
                output_dir=temp_dir,
                input_shape=(1, 3, 224, 224)
            )
            manager = ExportManager(config)
            
            model = edepth(
                backbone_name='vit_base_patch16_224',
                pretrained=False
            )
            
            export_path = manager.export_model(model)
            assert os.path.exists(export_path)

    def test_edepth_model_testing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ExportConfig(
                export_format=ExportFormat.TORCHSCRIPT_TRACE,
                output_dir=temp_dir,
                input_shape=(1, 3, 224, 224)
            )
            manager = ExportManager(config)
            
            model = edepth(
                backbone_name='vit_base_patch16_224',
                pretrained=False
            )
            
            export_path = manager.export_model(model)
            test_results = manager.test_exported_model(model, export_path)
            
            assert test_results['test_passed'] is True
            assert 'accuracy_test' in test_results
            assert 'avg_inference_time' in test_results
            assert 'memory_usage_mb' in test_results

class TestErrorHandling:
    def test_invalid_model_type(self):
        config = ExportConfig(export_format=ExportFormat.ONNX)
        exporter = ONNXExporter(config)
        
        with pytest.raises(ValueError):
            exporter.validate_model("not a model") #type: ignore 

    def test_missing_export_file(self):
        config = ExportConfig(export_format=ExportFormat.ONNX)
        exporter = ONNXExporter(config)
        
        with pytest.raises(Exception):
            exporter.validate_onnx_model("nonexistent.onnx")

    def test_export_error_handling(self):
        config = ExportConfig(export_format=ExportFormat.ONNX)
        manager = ExportManager(config)
        
        invalid_model = MagicMock()
        invalid_model.eval = MagicMock(side_effect=Exception("Model error"))
        
        with pytest.raises(Exception):
            manager.export_model(invalid_model)

class TestPerformanceOptimization:
    def test_optimization_validation(self):
        config = ExportConfig(
            export_format=ExportFormat.ONNX,
            optimization_level=OptimizationLevel.BASIC
        )
        optimizer = ModelOptimizer(config)
        
        original_model = nn.Linear(10, 5)
        optimized_model = optimizer.optimize_model(original_model)
        
        is_valid = optimizer.validate_optimized_model(original_model, optimized_model)
        assert is_valid is True

    def test_quantization_validation(self):
        handler = QuantizationHandler(QuantizationType.DYNAMIC)
        
        original_model = nn.Linear(10, 5)
        quantized_model = handler.convert_to_quantized(original_model)
        
        is_valid = handler.validate_quantized_model(original_model, quantized_model)
        assert isinstance(is_valid, bool)

class TestCompatibility:
    def test_different_input_shapes(self):
        shapes = [(1, 3, 224, 224), (1, 3, 256, 256), (2, 3, 224, 224)]
        
        for shape in shapes:
            config = ExportConfig(
                export_format=ExportFormat.ONNX,
                input_shape=shape
            )
            exporter = ONNXExporter(config)
            dummy_input = exporter.create_dummy_input(shape)
            assert dummy_input.shape == shape

    def test_different_batch_sizes(self):
        batch_sizes = [1, 2, 4, 8]
        
        for batch_size in batch_sizes:
            config = ExportConfig(
                export_format=ExportFormat.ONNX,
                batch_size=batch_size,
                input_shape=(batch_size, 3, 224, 224)
            )
            exporter = ONNXExporter(config)
            dummy_input = exporter.create_dummy_input((1, 3, 224, 224), batch_size)
            assert dummy_input.shape[0] == batch_size

class TestRegressionPrevention:
    def test_consistent_export_results(self):
        model = nn.Linear(10, 5)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ExportConfig(
                export_format=ExportFormat.ONNX,
                output_dir=temp_dir,
                input_shape=(1, 10, 10, 10)
            )
            manager = ExportManager(config)
            
            path1 = manager.export_model(model)
            path2 = manager.export_model(model)
            
            assert os.path.exists(path1)
            assert os.path.exists(path2)

    def test_export_determinism(self):
        model = nn.Linear(10, 5)
        torch.manual_seed(42)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ExportConfig(
                export_format=ExportFormat.TORCHSCRIPT_TRACE,
                output_dir=temp_dir,
                input_shape=(1, 10, 10, 10 )
            )
            manager = ExportManager(config)
            
            export_path = manager.export_model(model)
            loaded_model = torch.jit.load(export_path) #type: ignore 
            
            dummy_input = torch.randn(1, 10)
            with torch.no_grad():
                output1 = loaded_model(dummy_input)
                output2 = loaded_model(dummy_input)
            
            assert torch.allclose(output1, output2)