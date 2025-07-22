# FILE: tests/test_exports.py
# ehsanasgharzde - TESTS FOR EXPORT FUNCTIONALITY
# hosseinsolymanzadeh - PROPER COMMENTING
# hosseinsolymanzadeh - FIXED REDUNDANT CODE BY EXTRACTING PURE FUNCTIONS AND BASECLASS LEVEL METHODS

import os
import torch
import pytest
import tempfile
import torch.nn as nn
from unittest.mock import MagicMock, patch

# Import all classes and functions from updated export module
from export import (
    ExportFormat, OptimizationLevel, QuantizationType,
    ExportConfig, ExportResult, ModelExporter,
    export_model_multiple_formats, quick_export_onnx, quick_export_torchscript,
    ONNX_AVAILABLE
)

# Test ExportFormat enum functionality
def test_export_format_enum() -> None:
    # Verify all expected export formats are available
    assert ExportFormat.ONNX.value == "onnx"
    assert ExportFormat.TORCHSCRIPT_TRACE.value == "torchscript_trace"
    assert ExportFormat.TORCHSCRIPT_SCRIPT.value == "torchscript_script"
    assert ExportFormat.TENSORRT.value == "tensorrt"
    assert ExportFormat.OPENVINO.value == "openvino"
    assert ExportFormat.COREML.value == "coreml"


# Test OptimizationLevel enum functionality
def test_optimization_level_enum() -> None:
    # Verify all optimization levels are correctly defined
    assert OptimizationLevel.NONE.value == "none"
    assert OptimizationLevel.BASIC.value == "basic"
    assert OptimizationLevel.ADVANCED.value == "advanced"


# Test QuantizationType enum functionality
def test_quantization_type_enum() -> None:
    # Verify all quantization types are correctly defined
    assert QuantizationType.NONE.value == "none"
    assert QuantizationType.DYNAMIC.value == "dynamic"
    assert QuantizationType.STATIC.value == "static"
    assert QuantizationType.QAT.value == "qat"


# Test ExportConfig dataclass initialization and defaults
def test_export_config_initialization() -> None:
    # Test basic config creation with required parameters
    config = ExportConfig(
        format=ExportFormat.ONNX,
        output_path="/tmp/model.onnx"
    )
    
    # Verify required fields are set correctly
    assert config.format == ExportFormat.ONNX
    assert config.output_path == "/tmp/model.onnx"
    
    # Verify default values
    assert config.input_shape == (1, 3, 224, 224)
    assert config.optimization_level == OptimizationLevel.BASIC
    assert config.quantization == QuantizationType.NONE
    assert config.opset_version == 11
    assert config.batch_size == 1
    assert config.precision == "fp32"


# Test ExportConfig with custom parameters
def test_export_config_custom_parameters() -> None:
    # Test config creation with custom parameters
    config = ExportConfig(
        format=ExportFormat.TORCHSCRIPT_TRACE,
        output_path="/tmp/model.pt",
        input_shape=(2, 3, 256, 256),
        optimization_level=OptimizationLevel.ADVANCED,
        quantization=QuantizationType.DYNAMIC,
        batch_size=4,
        precision="fp16"
    )
    
    # Verify custom parameters are set correctly
    assert config.format == ExportFormat.TORCHSCRIPT_TRACE
    assert config.input_shape == (2, 3, 256, 256)
    assert config.optimization_level == OptimizationLevel.ADVANCED
    assert config.quantization == QuantizationType.DYNAMIC
    assert config.batch_size == 4
    assert config.precision == "fp16"


# Test ExportResult dataclass initialization
def test_export_result_initialization() -> None:
    # Test successful export result
    result = ExportResult(
        success=True,
        export_path="/tmp/model.onnx",
        model_size_mb=45.2,
        export_time_seconds=12.5
    )
    
    # Verify result fields are set correctly
    assert result.success is True
    assert result.export_path == "/tmp/model.onnx"
    assert result.model_size_mb == 45.2
    assert result.export_time_seconds == 12.5
    assert result.optimization_applied is False
    assert result.quantization_applied is False


# Test ModelExporter initialization with model instance
def test_model_exporter_with_model_instance() -> None:
    # Create a simple model and initialize exporter
    model = nn.Linear(10, 5)
    exporter = ModelExporter(model=model)
    
    # Verify exporter is initialized correctly
    assert exporter.model is model
    assert isinstance(exporter.model, nn.Module)


# Test ModelExporter initialization with model name
def test_model_exporter_with_model_name() -> None:
    # Mock the create_model function to avoid dependencies
    with patch('export.create_model') as mock_create:
        mock_model = nn.Linear(10, 5)
        mock_create.return_value = mock_model
        
        # Initialize exporter with model name
        exporter = ModelExporter(model_name="test_model")
        
        # Verify create_model was called and exporter is initialized
        mock_create.assert_called_once_with("test_model", None)
        assert exporter.model is mock_model


# Test ModelExporter raises error when neither model nor model_name provided
def test_model_exporter_validation_error() -> None:
    # Test that ValueError is raised when required parameters are missing
    with pytest.raises(ValueError, match="Either model instance or model_name must be provided"):
        ModelExporter()


# Test ModelExporter from_checkpoint class method
def test_model_exporter_from_checkpoint() -> None:
    # Mock the create_model_from_checkpoint function
    with patch('export.create_model_from_checkpoint') as mock_create:
        mock_model = nn.Linear(10, 5)
        mock_create.return_value = mock_model
        
        # Create exporter from checkpoint
        exporter = ModelExporter.from_checkpoint(
            checkpoint_path="/tmp/checkpoint.pth",
            model_name="test_model"
        )
        
        # Verify create_model_from_checkpoint was called correctly
        mock_create.assert_called_once_with(
            checkpoint_path="/tmp/checkpoint.pth",
            model_name="test_model",
            config=None
        )
        assert exporter.model is mock_model


# Test ModelExporter ONNX export functionality
@pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
def test_model_exporter_onnx_export() -> None:
    # Create a simple model and exporter
    model = nn.Conv2d(3, 16, 3)
    exporter = ModelExporter(model=model)
    
    # Create temporary file for export
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        export_path = f.name
    
    try:
        # Configure ONNX export
        config = ExportConfig(
            format=ExportFormat.ONNX,
            output_path=export_path,
            input_shape=(1, 3, 224, 224)
        )
        
        # Perform export
        result = exporter.export(config)
        
        # Verify export succeeded
        assert result.success is True
        assert result.export_path == export_path
        assert os.path.exists(export_path)
        assert result.model_size_mb is not None
        assert result.model_size_mb > 0
        
    finally:
        # Clean up temporary file
        if os.path.exists(export_path):
            os.unlink(export_path)


# Test ModelExporter TorchScript trace export functionality
def test_model_exporter_torchscript_trace_export() -> None:
    # Create a simple model and exporter
    model = nn.Conv2d(3, 16, 3)
    exporter = ModelExporter(model=model)
    
    # Create temporary file for export
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        export_path = f.name
    
    try:
        # Configure TorchScript trace export
        config = ExportConfig(
            format=ExportFormat.TORCHSCRIPT_TRACE,
            output_path=export_path,
            input_shape=(1, 3, 224, 224)
        )
        
        # Perform export
        result = exporter.export(config)
        
        # Verify export succeeded
        assert result.success is True
        assert result.export_path == export_path
        assert os.path.exists(export_path)
        assert result.model_size_mb is not None
        
    finally:
        # Clean up temporary file
        if os.path.exists(export_path):
            os.unlink(export_path)


# Test ModelExporter TorchScript script export functionality
def test_model_exporter_torchscript_script_export() -> None:
    # Create a simple model and exporter
    model = nn.Linear(10, 5)  # Linear models script better than conv layers
    exporter = ModelExporter(model=model)
    
    # Create temporary file for export
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        export_path = f.name
    
    try:
        # Configure TorchScript script export
        config = ExportConfig(
            format=ExportFormat.TORCHSCRIPT_SCRIPT,
            output_path=export_path,
            input_shape=(1, 10)
        )
        
        # Perform export
        result = exporter.export(config)
        
        # Verify export succeeded (may fallback to trace)
        assert result.success is True
        assert result.export_path == export_path
        assert os.path.exists(export_path)
        
    finally:
        # Clean up temporary file
        if os.path.exists(export_path):
            os.unlink(export_path)


# Test ModelExporter with optimization applied
def test_model_exporter_with_optimization() -> None:
    # Create a model with layers that can be optimized
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3),
        nn.BatchNorm2d(64),
        nn.ReLU()
    )
    exporter = ModelExporter(model=model)
    
    # Create temporary file for export
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        export_path = f.name
    
    try:
        # Configure export with basic optimization
        config = ExportConfig(
            format=ExportFormat.TORCHSCRIPT_TRACE,
            output_path=export_path,
            optimization_level=OptimizationLevel.BASIC
        )
        
        # Perform export
        result = exporter.export(config)
        
        # Verify export succeeded and optimization was applied
        assert result.success is True
        assert result.optimization_applied is True
        
    finally:
        # Clean up temporary file
        if os.path.exists(export_path):
            os.unlink(export_path)


# Test ModelExporter with quantization applied
def test_model_exporter_with_quantization() -> None:
    # Create a model suitable for quantization
    model = nn.Sequential(
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 10)
    )
    exporter = ModelExporter(model=model)
    
    # Create temporary file for export
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        export_path = f.name
    
    try:
        # Configure export with dynamic quantization
        config = ExportConfig(
            format=ExportFormat.TORCHSCRIPT_TRACE,
            output_path=export_path,
            quantization=QuantizationType.DYNAMIC
        )
        
        # Perform export
        result = exporter.export(config)
        
        # Verify export succeeded and quantization was applied
        assert result.success is True
        assert result.quantization_applied is True
        
    finally:
        # Clean up temporary file
        if os.path.exists(export_path):
            os.unlink(export_path)


# Test ModelExporter export failure handling
def test_model_exporter_export_failure() -> None:
    # Create a mock model that will cause export failure
    model = MagicMock()
    model.eval = MagicMock(side_effect=RuntimeError("Model evaluation failed"))
    
    exporter = ModelExporter(model=model)
    
    # Configure export
    config = ExportConfig(
        format=ExportFormat.TORCHSCRIPT_TRACE,
        output_path="/tmp/will_fail.pt"
    )
    
    # Perform export and expect failure
    result = exporter.export(config)
    
    # Verify export failed with error message
    assert result.success is False
    assert result.error_message is not None
    assert "Model evaluation failed" in result.error_message


# Test ModelExporter unsupported format handling
def test_model_exporter_unsupported_format() -> None:
    model = nn.Linear(10, 5)
    exporter = ModelExporter(model=model)
    
    # Create a mock unsupported format
    with patch.object(ExportFormat, 'TENSORRT') as mock_format:
        mock_format.value = "unsupported_format"
        
        config = ExportConfig(
            format=mock_format,
            output_path="/tmp/model.unsupported"
        )
        
        # Perform export and expect failure
        result = exporter.export(config)
        
        # Verify export failed
        assert result.success is False
        assert result.error_message is not None


# Test ModelExporter benchmark functionality
def test_model_exporter_benchmark() -> None:
    # Create a simple model and exporter
    model = nn.Linear(100, 50)
    exporter = ModelExporter(model=model)
    
    # Run benchmark with small number of runs for testing
    results = exporter.benchmark_model(
        input_shape=(1, 100),
        num_runs=5,
        warmup_runs=2
    )
    
    # Verify benchmark results contain expected metrics
    assert 'total_time_seconds' in results
    assert 'average_time_seconds' in results
    assert 'throughput_fps' in results
    assert 'average_time_ms' in results
    
    # Verify metrics are reasonable values
    assert results['total_time_seconds'] > 0
    assert results['average_time_seconds'] > 0
    assert results['throughput_fps'] > 0
    assert results['average_time_ms'] > 0


# Test export_model_multiple_formats utility function
def test_export_model_multiple_formats() -> None:
    # Create a simple model for export
    model = nn.Conv2d(3, 16, 3)
    
    # Use temporary directory for exports
    with tempfile.TemporaryDirectory() as temp_dir:
        # Export to multiple formats
        formats = [ExportFormat.TORCHSCRIPT_TRACE]  # Use only available format
        if ONNX_AVAILABLE:
            formats.append(ExportFormat.ONNX)
        
        results = export_model_multiple_formats(
            model=model,
            output_dir=temp_dir,
            model_name="test_model",
            formats=formats,
            input_shape=(1, 3, 224, 224)
        )
        
        # Verify results contain expected formats
        assert len(results) == len(formats)
        
        for fmt in formats:
            assert fmt in results
            result = results[fmt]
            assert isinstance(result, ExportResult)
            
            # Check if export succeeded or failed with proper error
            if result.success:
                assert result.export_path is not None
                assert os.path.exists(result.export_path)


# Test quick_export_onnx utility function
@pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
def test_quick_export_onnx() -> None:
    # Create a simple model for export
    model = nn.Conv2d(3, 16, 3)
    
    # Create temporary file for export
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        export_path = f.name
    
    try:
        # Perform quick ONNX export
        result = quick_export_onnx(
            model=model,
            output_path=export_path,
            input_shape=(1, 3, 224, 224)
        )
        
        # Verify export succeeded
        assert result.success is True
        assert result.export_path == export_path
        assert os.path.exists(export_path)
        
    finally:
        # Clean up temporary file
        if os.path.exists(export_path):
            os.unlink(export_path)


# Test quick_export_torchscript utility function
def test_quick_export_torchscript() -> None:
    # Create a simple model for export
    model = nn.Conv2d(3, 16, 3)
    
    # Create temporary file for export
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        export_path = f.name
    
    try:
        # Perform quick TorchScript export
        result = quick_export_torchscript(
            model=model,
            output_path=export_path,
            input_shape=(1, 3, 224, 224)
        )
        
        # Verify export succeeded
        assert result.success is True
        assert result.export_path == export_path
        assert os.path.exists(export_path)
        
    finally:
        # Clean up temporary file
        if os.path.exists(export_path):
            os.unlink(export_path)


# Test export with different input shapes
def test_export_different_input_shapes() -> None:
    # Test various input shapes
    input_shapes = [
        (1, 3, 224, 224),
        (2, 3, 256, 256),
        (1, 1, 128, 128),
        (4, 3, 320, 320)
    ]
    
    for input_shape in input_shapes:
        # Create appropriate model for input shape
        model = nn.Conv2d(input_shape[1], 16, 3)
        exporter = ModelExporter(model=model)
        
        # Create temporary file for export
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            export_path = f.name
        
        try:
            # Configure export with specific input shape
            config = ExportConfig(
                format=ExportFormat.TORCHSCRIPT_TRACE,
                output_path=export_path,
                input_shape=input_shape
            )
            
            # Perform export
            result = exporter.export(config)
            
            # Verify export succeeded for this input shape
            assert result.success is True, f"Export failed for input shape {input_shape}"
            
        finally:
            # Clean up temporary file
            if os.path.exists(export_path):
                os.unlink(export_path)


# Test export with different precision settings
def test_export_different_precisions() -> None:
    # Test different precision settings
    precisions = ["fp32", "fp16"]
    
    for precision in precisions:
        # Create a simple model
        model = nn.Linear(64, 32)
        exporter = ModelExporter(model=model)
        
        # Create temporary file for export
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            export_path = f.name
        
        try:
            # Configure export with specific precision
            config = ExportConfig(
                format=ExportFormat.TORCHSCRIPT_TRACE,
                output_path=export_path,
                precision=precision,
                input_shape=(1, 64)
            )
            
            # Perform export
            result = exporter.export(config)
            
            # Verify export succeeded for this precision
            assert result.success is True, f"Export failed for precision {precision}"
            
        finally:
            # Clean up temporary file
            if os.path.exists(export_path):
                os.unlink(export_path)


# Test export directory creation
def test_export_directory_creation() -> None:
    # Create model and exporter
    model = nn.Linear(10, 5)
    exporter = ModelExporter(model=model)
    
    # Use temporary directory as base
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create nested directory path that doesn't exist
        nested_dir = os.path.join(temp_dir, "models", "exports")
        export_path = os.path.join(nested_dir, "model.pt")
        
        # Configure export to non-existent directory
        config = ExportConfig(
            format=ExportFormat.TORCHSCRIPT_TRACE,
            output_path=export_path,
            input_shape=(1, 10)
        )
        
        # Perform export
        result = exporter.export(config)
        
        # Verify export succeeded and directory was created
        assert result.success is True
        assert os.path.exists(export_path)
        assert os.path.exists(nested_dir)


# Test export with metadata
def test_export_with_metadata() -> None:
    # Create model and exporter
    model = nn.Linear(10, 5)
    exporter = ModelExporter(model=model)
    
    # Create temporary file for export
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        export_path = f.name
    
    try:
        # Configure export with metadata
        metadata = {
            "model_name": "test_linear_model",
            "version": "1.0",
            "author": "test_suite"
        }
        
        config = ExportConfig(
            format=ExportFormat.TORCHSCRIPT_TRACE,
            output_path=export_path,
            input_shape=(1, 10),
            metadata=metadata
        )
        
        # Perform export
        result = exporter.export(config)
        
        # Verify export succeeded
        assert result.success is True
        assert result.export_path == export_path
        
    finally:
        # Clean up temporary file
        if os.path.exists(export_path):
            os.unlink(export_path)


# Test export consistency - same model should produce similar results
def test_export_consistency() -> None:
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create model and exporter
    model = nn.Linear(10, 5)
    exporter = ModelExporter(model=model)
    
    export_paths = []
    results = []
    
    # Perform multiple exports of same model
    for i in range(3):
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            export_path = f.name
            export_paths.append(export_path)
        
        config = ExportConfig(
            format=ExportFormat.TORCHSCRIPT_TRACE,
            output_path=export_path,
            input_shape=(1, 10)
        )
        
        result = exporter.export(config)
        results.append(result)
    
    try:
        # Verify all exports succeeded
        for i, result in enumerate(results):
            assert result.success is True, f"Export {i} failed"
            assert os.path.exists(result.export_path), f"Export {i} file not found"
        
        # Verify model sizes are similar (within reasonable tolerance)
        sizes = [result.model_size_mb for result in results if result.model_size_mb]
        if len(sizes) > 1:
            avg_size = sum(sizes) / len(sizes)
            for size in sizes:
                # Allow 5% variation in file size
                assert abs(size - avg_size) / avg_size < 0.05, "Export sizes vary too much"
        
    finally:
        # Clean up temporary files
        for export_path in export_paths:
            if os.path.exists(export_path):
                os.unlink(export_path)


# Test error handling for invalid output path
def test_export_invalid_output_path() -> None:
    # Create model and exporter
    model = nn.Linear(10, 5)
    exporter = ModelExporter(model=model)
    
    # Configure export with invalid output path (read-only directory)
    invalid_path = "/proc/invalid_export.pt"  # Should fail on most systems
    
    config = ExportConfig(
        format=ExportFormat.TORCHSCRIPT_TRACE,
        output_path=invalid_path,
        input_shape=(1, 10)
    )
    
    # Perform export and expect it to handle the error gracefully
    result = exporter.export(config)
    
    # Verify export failed with error message
    assert result.success is False
    assert result.error_message is not None


# Test ModelExporter with complex model architecture
def test_model_exporter_complex_model() -> None:
    # Create a more complex model architecture
    class ComplexModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(128)
            self.relu2 = nn.ReLU(inplace=True)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(128, 10)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.pool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x
    
    # Create complex model and exporter
    model = ComplexModel()
    exporter = ModelExporter(model=model)
    
    # Create temporary file for export
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        export_path = f.name
    
    try:
        # Configure export
        config = ExportConfig(
            format=ExportFormat.TORCHSCRIPT_TRACE,
            output_path=export_path,
            input_shape=(1, 3, 224, 224),
            optimization_level=OptimizationLevel.BASIC
        )
        
        # Perform export
        result = exporter.export(config)
        
        # Verify export succeeded
        assert result.success is True
        assert result.export_path == export_path
        assert os.path.exists(export_path)
        assert result.optimization_applied is True
        
        # Verify model file is reasonable size (not empty)
        file_size = os.path.getsize(export_path)
        assert file_size > 1000  # At least 1KB
        
    finally:
        # Clean up temporary file
        if os.path.exists(export_path):
            os.unlink(export_path)


# Test integration with factory module (if available)
def test_model_exporter_factory_integration() -> None:
    # Mock the create_model function for testing
    with patch('export.create_model') as mock_create_model:
        # Create a mock model to return
        mock_model = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 10)
        )
        mock_create_model.return_value = mock_model
        
        # Initialize exporter with model name (should use factory)
        exporter = ModelExporter(model_name="test_model", config={"pretrained": False})
        
        # Verify create_model was called with correct parameters
        mock_create_model.assert_called_once_with("test_model", {"pretrained": False})
        
        # Create temporary file for export
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            export_path = f.name
        
        try:
            # Configure and perform export
            config = ExportConfig(
                format=ExportFormat.TORCHSCRIPT_TRACE,
                output_path=export_path,
                input_shape=(1, 3, 224, 224)
            )
            
            result = exporter.export(config)
            
            # Verify export succeeded
            assert result.success is True
            assert os.path.exists(export_path)
            
        finally:
            # Clean up temporary file
            if os.path.exists(export_path):
                os.unlink(export_path)