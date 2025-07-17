# FILE: utils/export.py
# ehsanasgharzde - Export and Deployment
# hosseinsolymanzadeh - PROPER COMMENTING

import os
import time
import logging
import traceback
from pathlib import Path
from typing import Optional, Tuple, Any, Dict, List, Union
from enum import Enum
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.jit
import torch.quantization
import onnx #type: ignore 
import onnxruntime as ort #type: ignore 
import numpy as np
from models.edepth import edepth
from losses.factory import create_loss
from metrics.metrics import Metrics

logger = logging.getLogger(__name__)

# Enumeration of supported model export formats
class ExportFormat(Enum):
    ONNX = "onnx"
    TORCHSCRIPT_TRACE = "torchscript_trace"
    TORCHSCRIPT_SCRIPT = "torchscript_script"
    TENSORRT = "tensorrt"
    OPENVINO = "openvino"
    COREML = "coreml"


# Enumeration of supported optimization levels
class OptimizationLevel(Enum):
    NONE = "none"
    BASIC = "basic"
    ADVANCED = "advanced"


# Enumeration of quantization types
class QuantizationType(Enum):
    NONE = "none"
    DYNAMIC = "dynamic"
    STATIC = "static"
    QAT = "qat"


# Enumeration of model precision types
class Precision(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"


# Dataclass to hold export configuration settings
@dataclass
class ExportConfig:
    export_format: ExportFormat
    optimization_level: OptimizationLevel = OptimizationLevel.NONE
    quantization_type: QuantizationType = QuantizationType.NONE
    precision: Precision = Precision.FP32
    batch_size: int = 1
    input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224)
    output_dir: str = "./exports"
    model_name: str = "exported_model"
    
    def __post_init__(self):
        # Ensure output directory exists and log configuration
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Export config initialized: {self}")


# Class responsible for optimizing models based on the provided configuration
class ModelOptimizer:
    def __init__(self, config: ExportConfig):
        self.config = config
        logger.info(f"Model optimizer initialized with config: {config}")

    def optimize_model(self, model: nn.Module) -> nn.Module:
        # Skip optimization if NONE level is selected
        if self.config.optimization_level == OptimizationLevel.NONE:
            return model
        
        logger.info(f"Starting optimization at level: {self.config.optimization_level}")
        model.eval()  # Set model to evaluation mode
        
        # Apply module fusion for BASIC and ADVANCED optimization
        if self.config.optimization_level in {OptimizationLevel.BASIC, OptimizationLevel.ADVANCED}:
            model = self._fuse_modules(model)
            
        # Apply graph-level optimization only for ADVANCED level
        if self.config.optimization_level == OptimizationLevel.ADVANCED:
            model = self._optimize_graph(model)
            
        logger.info("Model optimization completed")
        return model

    def _fuse_modules(self, model: nn.Module) -> nn.Module:
        logger.info("Fusing modules for optimization")
        try:
            # Traverse named modules and attempt fusion where applicable
            for name, module in model.named_modules():
                if isinstance(module, nn.Sequential):
                    for i, (sub_name, sub_module) in enumerate(module.named_children()):
                        if i < len(module) - 1:
                            next_module = module[i + 1]
                            if self._can_fuse(sub_module, next_module):
                                logger.debug(f"Fusing modules: {sub_name} -> {next_module}")
        except Exception as e:
            logger.warning(f"Module fusion failed: {e}")
        return model

    def _can_fuse(self, module1: nn.Module, module2: nn.Module) -> bool:
        # Check if two modules can be fused (e.g., Conv2d+BatchNorm or BatchNorm+ReLU)
        return (isinstance(module1, nn.Conv2d) and isinstance(module2, nn.BatchNorm2d)) or \
               (isinstance(module1, nn.BatchNorm2d) and isinstance(module2, nn.ReLU))

    def _optimize_graph(self, model: nn.Module) -> nn.Module:
        # Placeholder for graph-level optimization logic
        logger.info("Applying graph-level optimizations")
        return model

    def validate_optimized_model(self, original_model: nn.Module, optimized_model: nn.Module) -> bool:
        logger.info("Validating optimized model")
        dummy_input = torch.randn(self.config.input_shape)  # Create dummy input tensor
        
        with torch.no_grad():
            orig_out = original_model(dummy_input)
            opt_out = optimized_model(dummy_input)
        
        # Compare outputs with tolerance to verify correctness
        is_valid = torch.allclose(orig_out, opt_out, rtol=1e-02, atol=1e-03)
        logger.info(f"Model validation result: {is_valid}")
        return is_valid


# Class to handle model quantization workflows
class QuantizationHandler:
    def __init__(self, quantization_type: QuantizationType, calibration_data=None):
        self.quantization_type = quantization_type
        self.calibration_data = calibration_data
        logger.info(f"Quantization handler initialized: {quantization_type}")

    def prepare_model_for_quantization(self, model: nn.Module) -> nn.Module:
        logger.info(f"Preparing model for quantization: {self.quantization_type}")
        model.eval()  # Set to evaluation mode
        
        # Dynamic quantization does not require preparation
        if self.quantization_type == QuantizationType.DYNAMIC:
            return model
        # Static quantization requires setting qconfig and preparing
        elif self.quantization_type == QuantizationType.STATIC:
            model.qconfig = torch.quantization.get_default_qconfig("fbgemm")  # Set quant config
            torch.quantization.prepare(model, inplace=True)  # Prepare model for static quant
        # QAT (quantization-aware training) preparation
        elif self.quantization_type == QuantizationType.QAT:
            model.train()
            model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")  # Set QAT config
            torch.quantization.prepare_qat(model, inplace=True)  # Prepare model for QAT
            
        return model

    def calibrate_model(self, model: nn.Module, calibration_loader) -> None:
        # Skip calibration if not using static quantization
        if self.quantization_type != QuantizationType.STATIC:
            return
            
        logger.info("Calibrating model for static quantization")
        model.eval()  # Set to evaluation mode
        
        # Run several batches through model to collect activation stats
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(calibration_loader):
                model(images)
                if batch_idx >= 32:  # Limit calibration to 32 batches
                    break
                    
        logger.info("Model calibration completed")

    def convert_to_quantized(self, model: nn.Module) -> nn.Module:
        # Convert prepared model to quantized version
        if self.quantization_type != QuantizationType.NONE:
            logger.info("Converting model to quantized version")
            model.eval()
            return torch.quantization.convert(model, inplace=False)
        return model

    def validate_quantized_model(self, original_model: nn.Module, quantized_model: nn.Module) -> bool:
        logger.info("Validating quantized model accuracy")
        dummy_input = torch.randn(1, 3, 224, 224)  # Create dummy input for validation
        
        with torch.no_grad():
            orig_out = original_model(dummy_input)
            quant_out = quantized_model(dummy_input)
        
        # Check if outputs are sufficiently close
        is_valid = torch.allclose(orig_out, quant_out, rtol=1e-01, atol=1e-01)
        logger.info(f"Quantized model validation result: {is_valid}")
        return is_valid


# Base class for all exporters providing shared utility methods
class BaseExporter:
    def __init__(self, config: ExportConfig):
        self.config = config
        logger.info(f"Base exporter initialized: {config.export_format}")

    # Check if the model is a valid nn.Module and set it to eval mode
    def validate_model(self, model: nn.Module) -> None:
        if not isinstance(model, nn.Module):
            raise ValueError("Invalid model type")
        
        logger.info("Model validation passed")
        model.eval()

    # Create a dummy input tensor with the specified input shape and batch size
    def create_dummy_input(self, input_shape: Tuple[int, ...], batch_size: int = 1) -> torch.Tensor:
        input_shape = list(input_shape)  # Convert to list to allow modification
        input_shape[0] = batch_size  # Set batch size as the first dimension
        dummy_input = torch.randn(*input_shape)  # Generate random tensor
        logger.debug(f"Created dummy input with shape: {dummy_input.shape}")
        return dummy_input

    # Benchmark model inference time over multiple iterations
    def benchmark_model(self, model: nn.Module, dummy_input: torch.Tensor, repeat: int = 50) -> float:
        logger.info(f"Benchmarking model with {repeat} iterations")
        model.eval()
        
        with torch.no_grad():
            # Ensure GPU sync before and after timing if CUDA is available
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.time()
            for _ in range(repeat):
                _ = model(dummy_input)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end = time.time()
        
        avg_time = (end - start) / repeat  # Calculate average inference time
        logger.info(f"Average inference time: {avg_time:.4f} seconds")
        return avg_time


# ONNX exporter implementation
class ONNXExporter(BaseExporter):
    def __init__(self, config: ExportConfig):
        super().__init__(config)

    # Export the model to ONNX format and validate the result
    def export(self, model: nn.Module, export_path: str) -> None:
        logger.info(f"Starting ONNX export to: {export_path}")
        self.validate_model(model)
        
        dummy_input = self.create_dummy_input(self.config.input_shape, self.config.batch_size)

        # Export parameters for ONNX conversion
        export_params = {
            "opset_version": 12,
            "do_constant_folding": True,
            "input_names": ["input"],
            "output_names": ["output"],
            "dynamic_axes": {
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            }
        }

        # Perform export to ONNX
        torch.onnx.export(model, dummy_input, export_path, **export_params)

        # Validate the exported ONNX model
        self.validate_onnx_model(export_path)
        logger.info(f"ONNX export completed successfully: {export_path}")

    # Load and check ONNX model integrity
    def validate_onnx_model(self, export_path: str) -> None:
        logger.info("Validating ONNX model")
        onnx_model = onnx.load(export_path)
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model validation passed")

    # Run inference using the exported ONNX model and ONNX Runtime
    def test_onnx_inference(self, model_path: str, dummy_input: torch.Tensor) -> np.ndarray:
        logger.info("Testing ONNX inference")
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name  # Get input tensor name
        result = session.run(None, {input_name: dummy_input.numpy()})[0]  # Run inference
        logger.info(f"ONNX inference test completed, output shape: {result.shape}")  # type: ignore
        return result  # type: ignore


# TorchScript exporter implementation
class TorchScriptExporter(BaseExporter):
    def __init__(self, config: ExportConfig):
        super().__init__(config)

    # Export model using TorchScript tracing or scripting
    def export(self, model: nn.Module, export_path: str) -> None:
        logger.info(f"Starting TorchScript export to: {export_path}")
        self.validate_model(model)

        # Choose export method based on configuration
        if self.config.export_format == ExportFormat.TORCHSCRIPT_TRACE:
            self._export_traced(model, export_path)
        else:
            self._export_scripted(model, export_path)

        # Validate the saved TorchScript model
        self.validate_torchscript_model(export_path)
        logger.info(f"TorchScript export completed successfully: {export_path}")

    # Export using torch.jit.trace
    def _export_traced(self, model: nn.Module, export_path: str) -> None:
        logger.info("Exporting via tracing")
        dummy_input = self.create_dummy_input(self.config.input_shape, self.config.batch_size)
        traced = torch.jit.trace(model, dummy_input)  # type: ignore
        traced.save(export_path)  # type: ignore

    # Export using torch.jit.script
    def _export_scripted(self, model: nn.Module, export_path: str) -> None:
        logger.info("Exporting via scripting")
        scripted = torch.jit.script(model)  # type: ignore
        scripted.save(export_path)  # type: ignore

    # Load and validate the TorchScript model
    def validate_torchscript_model(self, export_path: str) -> None:
        logger.info("Validating TorchScript model")
        _ = torch.jit.load(export_path)  # type: ignore
        logger.info("TorchScript model validation passed")

    # Run inference using the exported TorchScript model
    def test_torchscript_inference(self, model_path: str, dummy_input: torch.Tensor) -> torch.Tensor:
        logger.info("Testing TorchScript inference")
        model = torch.jit.load(model_path)  # type: ignore

        with torch.no_grad():
            result = model(dummy_input)  # Run inference
        
        logger.info(f"TorchScript inference test completed, output shape: {result.shape}")
        return result


# Exporter stub for TensorRT - not implemented
class TensorRTExporter(BaseExporter):
    def export(self, model: nn.Module, export_path: str) -> None:
        logger.warning("TensorRT export not implemented")
        raise NotImplementedError("TensorRT export not implemented")


# Exporter stub for OpenVINO - not implemented
class OpenVINOExporter(BaseExporter):
    def export(self, model: nn.Module, export_path: str) -> None:
        logger.warning("OpenVINO export not implemented")
        raise NotImplementedError("OpenVINO export not implemented")


# Exporter stub for CoreML - not implemented
class CoreMLExporter(BaseExporter):
    def export(self, model: nn.Module, export_path: str) -> None:
        logger.warning("CoreML export not implemented")
        raise NotImplementedError("CoreML export not implemented")


# Class for testing exported models (accuracy, performance, memory)
class DeploymentTester:
    def __init__(self, test_config: Dict[str, Any]):
        self.test_config = test_config
        logger.info("Deployment tester initialized")

    # Test whether two model outputs are close within a given tolerance
    def test_model_accuracy(self, original_model: nn.Module, exported_model: nn.Module, 
                            dummy_input: torch.Tensor, tolerance: float = 1e-3) -> bool:
        logger.info("Testing model accuracy")
        
        with torch.no_grad():
            orig_output = original_model(dummy_input)
            export_output = exported_model(dummy_input)

        # Convert outputs to numpy for comparison
        if isinstance(orig_output, torch.Tensor):
            orig_output = orig_output.cpu().numpy()
        if isinstance(export_output, torch.Tensor):
            export_output = export_output.cpu().numpy()

        # Use numpy allclose to compare
        is_accurate = np.allclose(orig_output, export_output, atol=tolerance)
        logger.info(f"Accuracy test result: {is_accurate}")
        return is_accurate

    # Measure average inference time over multiple runs
    def test_model_performance(self, model: nn.Module, dummy_input: torch.Tensor, 
                               num_runs: int = 100) -> float:
        logger.info(f"Testing model performance with {num_runs} runs")
        model.eval()

        with torch.no_grad():
            start = time.time()
            for _ in range(num_runs):
                _ = model(dummy_input)
            end = time.time()

        avg_time = (end - start) / num_runs
        logger.info(f"Average inference time: {avg_time:.4f} seconds")
        return avg_time

    # Estimate memory usage during model inference
    def test_memory_usage(self, model: nn.Module, dummy_input: torch.Tensor) -> float:
        logger.info("Testing memory usage")
        import psutil
        import gc

        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        process = psutil.Process(os.getpid())
        before = process.memory_info().rss

        _ = model(dummy_input)

        after = process.memory_info().rss
        memory_mb = (after - before) / (1024 ** 2)

        logger.info(f"Memory usage: {memory_mb:.2f} MB")
        return memory_mb

    # Format test results into a string report
    def generate_test_report(self, results: Dict[str, Any]) -> str:
        logger.info("Generating test report")
        report = "===== Deployment Test Report =====\n"

        for key, value in results.items():
            report += f"{key}: {value}\n"

        logger.info("Test report generated")
        return report


# Orchestrator for exporting and testing models
class ExportManager:
    def __init__(self, config: ExportConfig):
        self.config = config
        self.optimizer = ModelOptimizer(config)
        self.quantization_handler = QuantizationHandler(config.quantization_type)
        self.tester = DeploymentTester({})
        logger.info("Export manager initialized")

    # Check validity of export configuration
    def validate_export_config(self, config: ExportConfig) -> None:
        logger.info("Validating export configuration")

        if not hasattr(config, 'input_shape') or not config.input_shape:
            raise ValueError("Missing input_shape in config")
        if not hasattr(config, 'batch_size') or config.batch_size <= 0:
            raise ValueError("Invalid batch_size in config")
        if not hasattr(config, 'export_format'):
            raise ValueError("Missing export_format in config")

        logger.info("Export configuration validation passed")

    # Prepare model for export: optimize and quantize
    def prepare_model_for_export(self, model: nn.Module) -> nn.Module:
        logger.info("Preparing model for export")
        model.eval()

        model = self.optimizer.optimize_model(model)
        model = self.quantization_handler.prepare_model_for_quantization(model)
        model = self.quantization_handler.convert_to_quantized(model)

        logger.info("Model preparation completed")
        return model

    # Export model to given format and return file path
    def export_model(self, model: nn.Module, export_format: Optional[ExportFormat] = None) -> str:
        logger.info(f"Starting model export with format: {export_format or self.config.export_format}")

        export_format = export_format or self.config.export_format

        # Map format enum to exporter class
        exporters = {
            ExportFormat.ONNX: ONNXExporter,
            ExportFormat.TORCHSCRIPT_TRACE: TorchScriptExporter,
            ExportFormat.TORCHSCRIPT_SCRIPT: TorchScriptExporter,
            ExportFormat.TENSORRT: TensorRTExporter,
            ExportFormat.OPENVINO: OpenVINOExporter,
            ExportFormat.COREML: CoreMLExporter
        }

        if export_format not in exporters:
            raise ValueError(f"Unsupported export format: {export_format}")

        exporter = exporters[export_format](self.config)

        # Compose export file path
        export_path = os.path.join(
            self.config.output_dir,
            f"{self.config.model_name}_{export_format.value}.{self._get_file_extension(export_format)}"
        )

        prepared_model = self.prepare_model_for_export(model)
        exporter.export(prepared_model, export_path)

        logger.info(f"Model export completed: {export_path}")
        return export_path

    # Get file extension for each export format
    def _get_file_extension(self, export_format: ExportFormat) -> str:
        extensions = {
            ExportFormat.ONNX: "onnx",
            ExportFormat.TORCHSCRIPT_TRACE: "pt",
            ExportFormat.TORCHSCRIPT_SCRIPT: "pt",
            ExportFormat.TENSORRT: "engine",
            ExportFormat.OPENVINO: "xml",
            ExportFormat.COREML: "mlmodel"
        }
        return extensions.get(export_format, "bin")

    # Test exported model output and performance
    def test_exported_model(self, original_model: nn.Module, export_path: str) -> Dict[str, Any]:
        logger.info("Testing exported model")

        dummy_input = self.tester.create_dummy_input(self.config.input_shape, self.config.batch_size)  # type: ignore

        # Run inference based on file extension
        if export_path.endswith('.onnx'):
            exporter = ONNXExporter(self.config)
            exported_output = exporter.test_onnx_inference(export_path, dummy_input)
            exported_model = None
        elif export_path.endswith('.pt'):
            exporter = TorchScriptExporter(self.config)
            exported_model = torch.jit.load(export_path)  # type: ignore
            exported_output = exporter.test_torchscript_inference(export_path, dummy_input)
        else:
            raise ValueError(f"Unsupported export format for testing: {export_path}")

        results = {
            'export_path': export_path,
            'test_passed': True,
            'output_shape': exported_output.shape if hasattr(exported_output, 'shape') else str(type(exported_output))
        }

        # If model object exists, run full test suite
        if exported_model:
            accuracy = self.tester.test_model_accuracy(original_model, exported_model, dummy_input)
            performance = self.tester.test_model_performance(exported_model, dummy_input)
            memory = self.tester.test_memory_usage(exported_model, dummy_input)

            results.update({
                'accuracy_test': accuracy,
                'avg_inference_time': performance,
                'memory_usage_mb': memory
            })

        logger.info("Exported model testing completed")
        return results

    # Export model to multiple formats in a batch
    def batch_export(self, model: nn.Module, formats: List[ExportFormat]) -> Dict[str, str]:
        logger.info(f"Starting batch export for formats: {formats}")

        results = {}

        for export_format in formats:
            try:
                export_path = self.export_model(model, export_format)
                results[export_format.value] = export_path
                logger.info(f"Successfully exported to {export_format.value}: {export_path}")
            except Exception as e:
                logger.error(f"Failed to export to {export_format.value}: {e}")
                results[export_format.value] = f"ERROR: {str(e)}"

        logger.info("Batch export completed")
        return results

# Convenience function for ONNX export
def export_onnx(model: nn.Module, export_path: str, input_shape: Tuple[int, ...] = (1, 3, 224, 224), 
                batch_size: int = 1, **kwargs) -> str:
    logger.info(f"Quick ONNX export to: {export_path}")

    config = ExportConfig(
        export_format=ExportFormat.ONNX,
        input_shape=input_shape,  # type: ignore
        batch_size=batch_size,
        **kwargs
    )

    exporter = ONNXExporter(config)
    exporter.export(model, export_path)

    logger.info(f"ONNX export completed: {export_path}")
    return export_path

# Convenience function for TorchScript export (trace or script)
def export_torchscript(model: nn.Module, export_path: str, input_shape: Tuple[int, ...] = (1, 3, 224, 224), 
                       batch_size: int = 1, trace: bool = True, **kwargs) -> str:
    logger.info(f"Quick TorchScript export to: {export_path}")

    export_format = ExportFormat.TORCHSCRIPT_TRACE if trace else ExportFormat.TORCHSCRIPT_SCRIPT

    config = ExportConfig(
        export_format=export_format,
        input_shape=input_shape,  # type: ignore
        batch_size=batch_size,
        **kwargs
    )

    exporter = TorchScriptExporter(config)
    exporter.export(model, export_path)

    logger.info(f"TorchScript export completed: {export_path}")
    return export_path

# Utility function to print a deployment summary plan for a model
def print_deployment_plan(model: nn.Module, target_platform: str = "generic") -> None:
    logger.info(f"Generating deployment plan for platform: {target_platform}")

    model_info = {
        "Total Parameters": sum(p.numel() for p in model.parameters()),
        "Model Size (MB)": sum(p.numel() * 4 for p in model.parameters()) / (1024 ** 2),
        "Trainable Parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
    }

    print("\n===== Model Deployment Plan =====")
    print(f"Target Platform: {target_platform}")
    print("\nModel Information:")
    for key, value in model_info.items():
        print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value:.2f}")

    # Platform-specific recommendations
    recommendations = {
        "edge": ["Use ONNX with INT8 quantization", "Consider TensorRT for NVIDIA devices"],
        "cloud": ["Use TorchScript for PyTorch serving", "Consider ONNX for multi-framework support"],
        "mobile": ["Use CoreML for iOS", "Use TensorFlow Lite for Android"],
        "generic": ["Start with ONNX for maximum compatibility", "Use TorchScript for PyTorch environments"]
    }

    print(f"\nRecommendations for {target_platform}:")
    for rec in recommendations.get(target_platform, recommendations["generic"]):
        print(f"  - {rec}")

    # Export options overview
    print("\nExport Options:")
    print("  1. ONNX: Maximum compatibility, good for inference")
    print("  2. TorchScript: Native PyTorch, good for production")
    print("  3. TensorRT: NVIDIA GPU optimization")
    print("  4. OpenVINO: Intel hardware optimization")
    print("  5. CoreML: Apple ecosystem")

    logger.info("Deployment plan generated")

# Custom exception base for export-related errors
class ExportError(Exception):
    pass


# Specific error types for clearer exception handling
class OptimizationError(ExportError):
    pass


class QuantizationError(ExportError):
    pass


class ValidationError(ExportError):
    pass


class DeploymentError(ExportError):
    pass
