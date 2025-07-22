# FILE: export.py
# ehsanasgharzde - EXPORT AND DEPLOYMENT
# hosseinsolymanzadeh - PROPER COMMENTING
# ehsanasgharzde - UPDATED IMPORTS AND FACTORY INTEGRATION

import os
import time
import logging
import traceback
from typing import Optional, Tuple, Any, Dict, List
from enum import Enum
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.jit
import torch.quantization

from models.factory import create_model, create_model_from_checkpoint

logger = logging.getLogger(__name__)

# Optional imports for different export formats
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX not available. Install with: pip install onnx onnxruntime")

try:
    import tensorrt as trt  # type: ignore
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    logger.warning("TensorRT not available.")

try:
    import openvino as ov   # type: ignore
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False
    logger.warning("OpenVINO not available.")

try:
    import coremltools as ct  # type: ignore
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False
    logger.warning("CoreML not available.")

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

# Enumeration of supported quantization types
class QuantizationType(Enum):
    NONE = "none"
    DYNAMIC = "dynamic"
    STATIC = "static"
    QAT = "qat"  # Quantization Aware Training

@dataclass
class ExportConfig:
    format: ExportFormat
    output_path: str
    input_shape: Tuple[int, ...] = (1, 3, 224, 224)
    optimization_level: OptimizationLevel = OptimizationLevel.BASIC
    quantization: QuantizationType = QuantizationType.NONE
    opset_version: int = 11  # For ONNX export
    batch_size: int = 1
    precision: str = "fp32"  # fp32, fp16, int8
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExportResult:
    success: bool
    export_path: Optional[str] = None
    model_size_mb: Optional[float] = None
    export_time_seconds: Optional[float] = None
    optimization_applied: bool = False
    quantization_applied: bool = False
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class ModelExporter:
    def __init__(self, model: Optional[nn.Module] = None, model_name: Optional[str] = None, 
                 config: Optional[Dict[str, Any]] = None):
        if model is not None:
            self.model = model
        elif model_name is not None:
            # Use factory to create model
            self.model = create_model(model_name, config)
        else:
            raise ValueError("Either model instance or model_name must be provided")
            
        self.model.eval()
        logger.info(f"Initialized ModelExporter with model: {type(self.model).__name__}")
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, model_name: Optional[str] = None, 
                       config: Optional[Dict[str, Any]] = None) -> 'ModelExporter':
        model = create_model_from_checkpoint(
            checkpoint_path=checkpoint_path,
            model_name=model_name,
            config=config
        )
        return cls(model=model)
    
    def export(self, export_config: ExportConfig) -> ExportResult:
        start_time = time.time()
        result = ExportResult(success=False)
        
        try:
            # Create output directory if needed
            os.makedirs(os.path.dirname(export_config.output_path), exist_ok=True)
            
            # Apply optimizations
            if export_config.optimization_level != OptimizationLevel.NONE:
                self._apply_optimizations(export_config.optimization_level)
                result.optimization_applied = True
            
            # Apply quantization
            if export_config.quantization != QuantizationType.NONE:
                self._apply_quantization(export_config.quantization)
                result.quantization_applied = True
            
            # Export based on format
            if export_config.format == ExportFormat.ONNX:
                self._export_onnx(export_config)
            elif export_config.format == ExportFormat.TORCHSCRIPT_TRACE:
                self._export_torchscript_trace(export_config)
            elif export_config.format == ExportFormat.TORCHSCRIPT_SCRIPT:
                self._export_torchscript_script(export_config)
            elif export_config.format == ExportFormat.TENSORRT:
                self._export_tensorrt(export_config)
            elif export_config.format == ExportFormat.OPENVINO:
                self._export_openvino(export_config)
            elif export_config.format == ExportFormat.COREML:
                self._export_coreml(export_config)
            else:
                raise ValueError(f"Unsupported export format: {export_config.format}")
            
            # Calculate file size
            if os.path.exists(export_config.output_path):
                file_size = os.path.getsize(export_config.output_path)
                result.model_size_mb = file_size / (1024 * 1024)
            
            result.success = True
            result.export_path = export_config.output_path
            result.export_time_seconds = time.time() - start_time
            
            logger.info(f"Successfully exported model to {export_config.output_path}")
            
        except Exception as e:
            result.error_message = str(e)
            logger.error(f"Export failed: {e}")
            logger.error(traceback.format_exc())
        
        return result
    
    def _export_onnx(self, config: ExportConfig) -> None:
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX not available. Install with: pip install onnx onnxruntime")
        
        dummy_input = torch.randn(*config.input_shape)
        
        # Prepare dynamic axes if specified
        dynamic_axes = config.dynamic_axes or {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
        
        torch.onnx.export(
            self.model,
            dummy_input,
            config.output_path,
            export_params=True,
            opset_version=config.opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(config.output_path)
        onnx.checker.check_model(onnx_model)
        logger.info(f"ONNX model exported and verified: {config.output_path}")
    
    def _export_torchscript_trace(self, config: ExportConfig) -> None:
        dummy_input = torch.randn(*config.input_shape)
        
        with torch.no_grad():
            traced_model = torch.jit.trace(self.model, dummy_input) # type: ignore
            traced_model.save(config.output_path) # type: ignore
        
        logger.info(f"TorchScript traced model exported: {config.output_path}")
    
    def _export_torchscript_script(self, config: ExportConfig) -> None:
        try:
            scripted_model = torch.jit.script(self.model) # type: ignore
            scripted_model.save(config.output_path) # type: ignore
            logger.info(f"TorchScript scripted model exported: {config.output_path}")
        except Exception as e:
            logger.warning(f"TorchScript scripting failed: {e}")
            logger.info("Falling back to tracing...")
            self._export_torchscript_trace(config)
    
    def _export_tensorrt(self, config: ExportConfig) -> None:
        if not TENSORRT_AVAILABLE:
            raise RuntimeError("TensorRT not available")
        
        # First export to ONNX, then convert to TensorRT
        onnx_path = config.output_path.replace('.trt', '.onnx')
        onnx_config = ExportConfig(
            format=ExportFormat.ONNX,
            output_path=onnx_path,
            input_shape=config.input_shape,
            opset_version=config.opset_version
        )
        self._export_onnx(onnx_config)
        
        # Convert ONNX to TensorRT
        self._convert_onnx_to_tensorrt(onnx_path, config.output_path, config)
        
        # Clean up temporary ONNX file
        if os.path.exists(onnx_path):
            os.remove(onnx_path)
    
    def _export_openvino(self, config: ExportConfig) -> None:
        if not OPENVINO_AVAILABLE:
            raise RuntimeError("OpenVINO not available")
        
        dummy_input = torch.randn(*config.input_shape)
        
        # Convert PyTorch model to OpenVINO
        ov_model = ov.convert_model(
            self.model,
            example_input=dummy_input,
            input=config.input_shape
        )
        
        # Save OpenVINO model
        ov.save_model(ov_model, config.output_path)
        logger.info(f"OpenVINO model exported: {config.output_path}")
    
    def _export_coreml(self, config: ExportConfig) -> None:
        if not COREML_AVAILABLE:
            raise RuntimeError("CoreML not available")
        
        dummy_input = torch.randn(*config.input_shape)
        
        # Trace the model
        traced_model = torch.jit.trace(self.model, dummy_input) # type: ignore
        
        # Convert to CoreML
        coreml_model = ct.convert(
            traced_model,
            inputs=[ct.TensorType(shape=config.input_shape)],
            convert_to="neuralnetwork" if config.precision == "fp32" else "mlprogram"
        )
        
        coreml_model.save(config.output_path)
        logger.info(f"CoreML model exported: {config.output_path}")
    
    def _apply_optimizations(self, level: OptimizationLevel) -> None:
        if level == OptimizationLevel.BASIC:
            # Basic optimizations
            if hasattr(self.model, 'fuse_model'):
                self.model.fuse_model() # type: ignore
        elif level == OptimizationLevel.ADVANCED:
            # Advanced optimizations
            self.model = torch.jit.optimize_for_inference(torch.jit.script(self.model)) # type: ignore
        
        logger.info(f"Applied {level.value} optimizations")
    
    def _apply_quantization(self, quant_type: QuantizationType) -> None:
        if quant_type == QuantizationType.DYNAMIC:
            self.model = torch.quantization.quantize_dynamic(
                self.model,
                {torch.nn.Linear, torch.nn.Conv2d},
                dtype=torch.qint8
            )
            logger.info("Applied dynamic quantization")
        elif quant_type == QuantizationType.STATIC:
            # Static quantization requires calibration dataset
            logger.warning("Static quantization requires calibration data - skipping")
        elif quant_type == QuantizationType.QAT:
            logger.warning("QAT quantization should be applied during training - skipping")
    
    def _convert_onnx_to_tensorrt(self, onnx_path: str, trt_path: str, config: ExportConfig) -> None:
        logger_trt = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger_trt)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger_trt)
        
        # Parse ONNX model
        with open(onnx_path, 'rb') as model_file:
            if not parser.parse(model_file.read()):
                for error in range(parser.num_errors):
                    logger.error(parser.get_error(error))
                raise RuntimeError("Failed to parse ONNX model")
        
        # Build TensorRT engine
        config_trt = builder.create_builder_config()
        if config.precision == "fp16":
            config_trt.set_flag(trt.BuilderFlag.FP16)
        elif config.precision == "int8":
            config_trt.set_flag(trt.BuilderFlag.INT8)
        
        engine = builder.build_engine(network, config_trt)
        
        # Save TensorRT engine
        with open(trt_path, 'wb') as f:
            f.write(engine.serialize())
        
        logger.info(f"TensorRT engine saved: {trt_path}")
    
    def benchmark_model(self, input_shape: Tuple[int, ...] = (1, 3, 224, 224), 
                       num_runs: int = 100, warmup_runs: int = 10) -> Dict[str, float]:
        self.model.eval()
        device = next(self.model.parameters()).device
        dummy_input = torch.randn(*input_shape, device=device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = self.model(dummy_input)
        
        # Benchmark
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = self.model(dummy_input)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_runs
        throughput = num_runs / total_time
        
        results = {
            'total_time_seconds': total_time,
            'average_time_seconds': avg_time,
            'throughput_fps': throughput,
            'average_time_ms': avg_time * 1000
        }
        
        logger.info(f"Benchmark results: {results}")
        return results

def export_model_multiple_formats(
    model: nn.Module,
    output_dir: str,
    model_name: str = "model",
    formats: Optional[List[ExportFormat]] = None,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224)
) -> Dict[ExportFormat, ExportResult]:
    if formats is None:
        formats = [ExportFormat.ONNX, ExportFormat.TORCHSCRIPT_TRACE]
    
    exporter = ModelExporter(model)
    results = {}
    
    for fmt in formats:
        try:
            # Determine file extension
            ext_map = {
                ExportFormat.ONNX: '.onnx',
                ExportFormat.TORCHSCRIPT_TRACE: '.pt',
                ExportFormat.TORCHSCRIPT_SCRIPT: '.pt',
                ExportFormat.TENSORRT: '.trt',
                ExportFormat.OPENVINO: '.xml',
                ExportFormat.COREML: '.mlmodel'
            }
            
            output_path = os.path.join(output_dir, f"{model_name}{ext_map[fmt]}")
            
            config = ExportConfig(
                format=fmt,
                output_path=output_path,
                input_shape=input_shape
            )
            
            results[fmt] = exporter.export(config)
            
        except Exception as e:
            results[fmt] = ExportResult(success=False, error_message=str(e))
            logger.error(f"Failed to export to {fmt.value}: {e}")
    
    return results

# Utility functions for easy export
def quick_export_onnx(model: nn.Module, output_path: str, 
                     input_shape: Tuple[int, ...] = (1, 3, 224, 224)) -> ExportResult:
    exporter = ModelExporter(model)
    config = ExportConfig(
        format=ExportFormat.ONNX,
        output_path=output_path,
        input_shape=input_shape
    )
    return exporter.export(config)

def quick_export_torchscript(model: nn.Module, output_path: str,
                           input_shape: Tuple[int, ...] = (1, 3, 224, 224)) -> ExportResult:
    exporter = ModelExporter(model)
    config = ExportConfig(
        format=ExportFormat.TORCHSCRIPT_TRACE,
        output_path=output_path,
        input_shape=input_shape
    )
    return exporter.export(config)