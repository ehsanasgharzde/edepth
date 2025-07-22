# FILE: tests/test_model.py
# ehsanasgharzde - COMPLETE MODEL TEST SUITE
# hosseinsolymanzadeh - PROPER COMMENTING
# hosseinsolymanzadeh - FIXED REDUNDANT CODE BY EXTRACTING PURE FUNCTIONS AND BASECLASS LEVEL METHODS

import time
import torch
import pytest
import logging
import torch.nn as nn
from typing import Dict, Any, List, Tuple

# Updated imports based on new module structure
from ..models.edepth import edepth
from ..models.factory import (
    create_model, get_available_models, ModelBuilder,
    get_model_info_summary, estimate_model_parameters, validate_model_config,
)

# Import centralized utilities
from ..utils.model_utils import (
    get_model_info, ModelInfo, interpolate_features
)
from ..utils.model_validation import (
    validate_tensor_input, TensorValidationError, ConfigValidationError
)
from ..configs.model_config import (
    get_model_config, get_backbone_config, list_available_models,
    list_available_backbones
)

logger = logging.getLogger(__name__)

def test_model_creation_basic() -> None:
    # Test basic model creation with default parameters
    model: nn.Module = create_model('vit_base_patch16_224')
    assert isinstance(model, edepth), f"Expected edepth model, got {type(model)}"
    
    # Verify model has required attributes
    assert hasattr(model, 'backbone'), "Model missing backbone attribute"
    assert hasattr(model, 'decoder'), "Model missing decoder attribute"
    assert hasattr(model, 'forward'), "Model missing forward method"

def test_model_creation_with_config() -> None:
    # Test model creation with custom configuration
    config: Dict[str, Any] = {
        'backbone_name': 'vit_base_patch16_224',
        'extract_layers': [6, 8, 10, 11],
        'decoder_channels': [128, 256, 512, 768],
        'use_attention': True,
        'final_activation': 'sigmoid'
    }
    
    model: nn.Module = create_model('vit_base_patch16_224', config)
    assert isinstance(model, edepth), f"Expected edepth model, got {type(model)}"

def test_model_forward_pass_basic() -> None:
    # Test basic forward pass functionality
    model: nn.Module = create_model('vit_base_patch16_224')
    model.eval()
    
    input_tensor: torch.Tensor = torch.randn(2, 3, 224, 224)
    
    with torch.no_grad():
        output: torch.Tensor = model(input_tensor)
    
    # Validate output shape and properties
    expected_shape: Tuple[int, int, int, int] = (2, 1, 224, 224)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains infinite values"

def test_model_forward_pass_different_sizes() -> None:
    # Test forward pass with different input sizes
    model: nn.Module = create_model('vit_base_patch16_224')
    model.eval()
    
    test_sizes: List[Tuple[int, int, int, int]] = [
        (1, 3, 224, 224),
        (4, 3, 224, 224),
        (1, 3, 384, 384),
        (2, 3, 512, 512)
    ]
    
    for size in test_sizes:
        input_tensor: torch.Tensor = torch.randn(*size)
        
        with torch.no_grad():
            output: torch.Tensor = model(input_tensor)
        
        expected_batch: int = size[0]
        expected_height: int = size[2]
        expected_width: int = size[3]
        
        assert output.shape[0] == expected_batch, f"Batch size mismatch: expected {expected_batch}, got {output.shape[0]}"
        assert output.shape[1] == 1, f"Channel mismatch: expected 1, got {output.shape[1]}"

def test_model_gradient_flow() -> None:
    # Test gradient flow through the model
    model: nn.Module = create_model('vit_base_patch16_224')
    model.train()
    
    input_tensor: torch.Tensor = torch.randn(1, 3, 224, 224, requires_grad=True)
    target: torch.Tensor = torch.rand(1, 1, 224, 224)
    
    # Forward pass
    output: torch.Tensor = model(input_tensor)
    loss: torch.Tensor = torch.nn.functional.mse_loss(output, target)
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Check gradient presence
    param_count: int = 0
    grad_count: int = 0
    
    for param in model.parameters():
        param_count += 1
        if param.grad is not None:
            grad_count += 1
            assert not torch.isnan(param.grad).any(), "Gradient contains NaN values"
    
    gradient_coverage: float = grad_count / param_count if param_count > 0 else 0
    assert gradient_coverage > 0.8, f"Low gradient coverage: {gradient_coverage:.2f}"

def test_model_parameter_count() -> None:
    # Test parameter counting functionality
    model: nn.Module = create_model('vit_base_patch16_224')
    
    model_info: ModelInfo = get_model_info(model)
    total_params: int = model_info.total_parameters
    trainable_params: int = model_info.trainable_parameters
    
    assert total_params > 0, "Total parameters should be greater than 0"
    assert trainable_params > 0, "Trainable parameters should be greater than 0"
    assert trainable_params <= total_params, "Trainable parameters cannot exceed total parameters"

def test_backbone_functionality() -> None:
    # Test backbone-specific functionality
    model: edepth = edepth('vit_base_patch16_224')
    
    # Test backbone freeze/unfreeze
    model.freeze_backbone()
    frozen_params: int = sum(1 for p in model.backbone.parameters() if not p.requires_grad)
    total_backbone_params: int = sum(1 for _ in model.backbone.parameters())
    
    assert frozen_params == total_backbone_params, "Not all backbone parameters were frozen"
    
    model.unfreeze_backbone()
    unfrozen_params: int = sum(1 for p in model.backbone.parameters() if p.requires_grad)
    
    assert unfrozen_params == total_backbone_params, "Not all backbone parameters were unfrozen"

def test_factory_functions() -> None:
    # Test factory function implementations
    available_models: List[str] = get_available_models()
    assert len(available_models) > 0, "No available models found"
    assert 'vit_base_patch16_224' in available_models, "Expected model not in available models"
    
    # Test model info summary
    model: nn.Module = create_model('vit_base_patch16_224')
    info_summary: Dict[str, Any] = get_model_info_summary(model)
    
    required_keys: List[str] = ['total_parameters', 'trainable_parameters', 'model_type']
    for key in required_keys:
        assert key in info_summary, f"Missing key '{key}' in model info summary"

def test_model_builder() -> None:
    # Test ModelBuilder functionality
    builder: ModelBuilder = ModelBuilder()
    
    model: nn.Module = (builder
                       .backbone('vit_base_patch16_224')
                       .decoder([256, 512, 768, 768])
                       .training_config(use_checkpointing=False)
                       .build())
    
    assert isinstance(model, nn.Module), "ModelBuilder did not produce a valid model"
    assert hasattr(model, 'forward'), "Built model missing forward method"

def test_configuration_validation() -> None:
    # Test configuration validation functions
    valid_config: Dict[str, Any] = {
        'backbone_name': 'vit_base_patch16_224',
        'decoder_channels': [256, 512, 768, 768],
        'use_attention': False,
        'final_activation': 'sigmoid'
    }
    
    # This should not raise an exception
    validate_model_config(valid_config)
    
    # Test invalid configuration
    invalid_config: Dict[str, Any] = {
        'backbone_name': 'nonexistent_model',
        'decoder_channels': [-1, 0]  # Invalid negative/zero channels
    }
    
    with pytest.raises((ConfigValidationError, ValueError)):
        validate_model_config(invalid_config)

def test_tensor_validation_functions() -> None:
    # Test tensor validation utilities
    valid_tensor: torch.Tensor = torch.randn(2, 3, 224, 224)
    validate_tensor_input(valid_tensor, "test_tensor", expected_dims=4)
    
    # Test invalid tensor validation
    invalid_tensor: torch.Tensor = torch.randn(2, 3)  # Wrong dimensions
    
    with pytest.raises(TensorValidationError):
        validate_tensor_input(invalid_tensor, "invalid_tensor", expected_dims=4)

def test_feature_extraction() -> None:
    # Test feature extraction from backbone
    model: edepth = edepth('vit_base_patch16_224')
    model.eval()
    
    input_tensor: torch.Tensor = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        # Get backbone features
        backbone_features: List[torch.Tensor] = model.backbone(input_tensor)
        
        assert isinstance(backbone_features, list), "Backbone features should be a list"
        assert len(backbone_features) > 0, "No features extracted from backbone"
        
        # Validate each feature tensor
        for i, feature in enumerate(backbone_features):
            assert isinstance(feature, torch.Tensor), f"Feature {i} is not a tensor"
            assert feature.dim() == 4, f"Feature {i} should have 4 dimensions, got {feature.dim()}"

def test_model_checkpointing() -> None:
    # Test model checkpoint saving and loading
    model: nn.Module = create_model('vit_base_patch16_224')
    checkpoint_path: str = "/tmp/test_checkpoint.pth"
    
    # Test saving checkpoint
    if hasattr(model, 'save_checkpoint'):
        model.save_checkpoint(checkpoint_path)
        
        # Test loading checkpoint
        loaded_metadata: Dict[str, Any] = model.load_checkpoint(checkpoint_path)
        assert isinstance(loaded_metadata, dict), "Checkpoint metadata should be a dictionary"

def test_interpolation_functionality() -> None:
    # Test feature interpolation utility
    feature_tensor: torch.Tensor = torch.randn(1, 256, 14, 14)
    target_size: Tuple[int, int] = (28, 28)
    
    interpolated: torch.Tensor = interpolate_features(feature_tensor, target_size)
    
    assert interpolated.shape[-2:] == target_size, f"Interpolation failed: expected {target_size}, got {interpolated.shape[-2:]}"
    assert interpolated.shape[:2] == feature_tensor.shape[:2], "Batch and channel dimensions should remain unchanged"

def test_config_loading() -> None:
    # Test configuration loading functions
    available_backbones: List[str] = list_available_backbones()
    available_models: List[str] = list_available_models()
    
    assert len(available_backbones) > 0, "No available backbones found"
    assert len(available_models) > 0, "No available models found"
    
    # Test getting specific configurations
    for model_name in available_models[:3]:  # Test first 3 models
        config: Dict[str, Any] = get_model_config(model_name)
        backbone_config: Dict[str, Any] = get_backbone_config(model_name)
        
        assert 'backbone_name' in config, f"Config missing backbone_name for {model_name}"
        assert 'patch_size' in backbone_config, f"Backbone config missing patch_size for {model_name}"

def test_edge_cases() -> None:
    # Test various edge cases
    
    # Test with minimum batch size
    model: nn.Module = create_model('vit_base_patch16_224')
    model.eval()
    
    min_input: torch.Tensor = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        min_output: torch.Tensor = model(min_input)
    
    assert min_output.shape[0] == 1, "Minimum batch size test failed"
    
    # Test with larger batch size
    large_input: torch.Tensor = torch.randn(8, 3, 224, 224)
    with torch.no_grad():
        large_output: torch.Tensor = model(large_input)
    
    assert large_output.shape[0] == 8, "Large batch size test failed"

def test_device_compatibility() -> None:
    # Test device compatibility
    model: nn.Module = create_model('vit_base_patch16_224')
    
    # Test CPU operation
    cpu_input: torch.Tensor = torch.randn(1, 3, 224, 224)
    model.eval()
    
    with torch.no_grad():
        cpu_output: torch.Tensor = model(cpu_input)
    
    assert cpu_output.device == cpu_input.device, "Device mismatch between input and output"
    
    # Test GPU operation if available
    if torch.cuda.is_available():
        model_cuda: nn.Module = model.cuda()
        cuda_input: torch.Tensor = torch.randn(1, 3, 224, 224).cuda()
        
        with torch.no_grad():
            cuda_output: torch.Tensor = model_cuda(cuda_input)
        
        assert cuda_output.device == cuda_input.device, "CUDA device mismatch"

def test_model_modes() -> None:
    # Test training and evaluation modes
    model: nn.Module = create_model('vit_base_patch16_224')
    
    # Test training mode
    model.train()
    assert model.training, "Model should be in training mode"
    
    # Test evaluation mode
    model.eval()
    assert not model.training, "Model should be in evaluation mode"

def test_activation_functions() -> None:
    # Test different activation functions
    activations: List[str] = ['sigmoid', 'tanh', 'relu', 'none']
    
    for activation in activations:
        config: Dict[str, Any] = {
            'backbone_name': 'vit_base_patch16_224',
            'final_activation': activation
        }
        
        model: nn.Module = create_model('vit_base_patch16_224', config)
        input_tensor: torch.Tensor = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            output: torch.Tensor = model(input_tensor)
        
        # Validate output range based on activation
        if activation == 'sigmoid':
            assert output.min() >= 0.0 and output.max() <= 1.0, f"Sigmoid output out of range [0,1]"
        elif activation == 'tanh':
            assert output.min() >= -1.0 and output.max() <= 1.0, f"Tanh output out of range [-1,1]"
        elif activation == 'relu':
            assert output.min() >= 0.0, f"ReLU output contains negative values"

def test_attention_mechanism() -> None:
    # Test attention mechanism in decoder
    config_with_attention: Dict[str, Any] = {
        'backbone_name': 'vit_base_patch16_224',
        'use_attention': True
    }
    
    config_without_attention: Dict[str, Any] = {
        'backbone_name': 'vit_base_patch16_224',
        'use_attention': False
    }
    
    model_with_attn: nn.Module = create_model('vit_base_patch16_224', config_with_attention)
    model_without_attn: nn.Module = create_model('vit_base_patch16_224', config_without_attention)
    
    input_tensor: torch.Tensor = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        output_with_attn: torch.Tensor = model_with_attn(input_tensor)
        output_without_attn: torch.Tensor = model_without_attn(input_tensor)
    
    # Outputs should have same shape but potentially different values
    assert output_with_attn.shape == output_without_attn.shape, "Attention mechanism changed output shape"

def test_model_memory_usage() -> None:
    # Test model memory usage tracking
    model: nn.Module = create_model('vit_base_patch16_224')
    model_info: ModelInfo = get_model_info(model)
    
    memory_mb: float = model_info.model_size_mb
    assert memory_mb > 0, "Model memory usage should be positive"
    assert memory_mb < 10000, "Model memory usage seems unreasonably high"  # Sanity check

def test_parameter_estimation() -> None:
    # Test parameter estimation functionality
    estimation: Dict[str, Any] = estimate_model_parameters('vit_base_patch16_224')
    
    if 'error' not in estimation:
        assert 'total_parameters' in estimation, "Parameter estimation missing total_parameters"
        assert estimation['total_parameters'] > 0, "Estimated parameters should be positive"

def test_model_summary_functionality() -> None:
    # Test model summary generation
    model: edepth = edepth('vit_base_patch16_224')
    
    if hasattr(model, 'get_model_summary'):
        summary: Dict[str, Any] = model.get_model_summary()
        
        expected_keys: List[str] = ['backbone_name', 'extract_layers', 'decoder_channels']
        for key in expected_keys:
            assert key in summary, f"Model summary missing key: {key}"

def run_comprehensive_test_suite() -> Dict[str, Any]:
    # Comprehensive test suite runner
    test_results: Dict[str, Any] = {
        'timestamp': time.time(),
        'tests_passed': 0,
        'tests_failed': 0,
        'failures': []
    }
    
    test_functions: List[callable] = [
        test_model_creation_basic,
        test_model_creation_with_config,
        test_model_forward_pass_basic,
        test_model_forward_pass_different_sizes,
        test_model_gradient_flow,
        test_model_parameter_count,
        test_backbone_functionality,
        test_factory_functions,
        test_model_builder,
        test_configuration_validation,
        test_tensor_validation_functions,
        test_feature_extraction,
        test_interpolation_functionality,
        test_config_loading,
        test_edge_cases,
        test_device_compatibility,
        test_model_modes,
        test_activation_functions,
        test_attention_mechanism,
        test_model_memory_usage,
        test_parameter_estimation,
        test_model_summary_functionality
    ]
    
    for test_func in test_functions:
        try:
            test_func()
            test_results['tests_passed'] += 1
            logger.info(f"✓ {test_func.__name__} passed")
        except Exception as e:
            test_results['tests_failed'] += 1
            test_results['failures'].append({
                'test': test_func.__name__,
                'error': str(e)
            })
            logger.error(f"✗ {test_func.__name__} failed: {e}")
    
    return test_results