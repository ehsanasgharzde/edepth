# FILE: tests/test_decoder.py
# ehsanasgharzde - COMPLETE DECODER TEST SUITE
# hosseinsolymanzadeh - PROPER COMMENTING
# hosseinsolymanzadeh - FIXED REDUNDANT CODE BY EXTRACTING PURE FUNCTIONS AND BASECLASS LEVEL METHODS

import gc
import torch
import pytest
import psutil
import logging
from typing import List, Tuple

# Updated imports to match the new module structure
from ..models.decoders.decoder import (
    FusionBlock, DPT, validate_features_compatibility, interpolate_to_target_size
)
from ..utils.model_validation import (
    validate_dpt_features, validate_tensor_input, validate_interpolation_target,
    TensorValidationError
)
from ..utils.model_utils import (
    interpolate_features, get_model_info, ModelInfo, 
    count_parameters
)

logger = logging.getLogger(__name__)

# Global test device configuration
def get_test_device() -> torch.device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    return device

# FusionBlock Tests
def test_fusion_block_basic() -> None:
    device = get_test_device()
    
    # Initialize block parameters
    in_ch: int = 256
    out_ch: int = 128
    block = FusionBlock(in_ch, out_ch).to(device)
    
    # Create dummy input tensor
    x = torch.randn(2, in_ch, 32, 32).to(device)
    output = block(x)
    
    # Verify output dimensions (upsampled by default scale_factor=2)
    assert output.shape == (2, out_ch, 64, 64)
    assert output.requires_grad
    logger.info("FusionBlock basic test passed")

def test_fusion_block_with_attention() -> None:
    device = get_test_device()
    
    # Initialize block with attention
    in_ch: int = 256
    out_ch: int = 128
    block = FusionBlock(in_ch, out_ch, use_attention=True).to(device)
    
    # Create input tensor
    x = torch.randn(2, in_ch, 32, 32).to(device)
    output = block(x)
    
    # Verify output shape and attention module existence
    assert output.shape == (2, out_ch, 64, 64)
    assert hasattr(block, 'attention')
    logger.info("FusionBlock attention test passed")

def test_fusion_block_custom_scale_factor() -> None:
    device = get_test_device()
    
    # Initialize block with custom scale factor
    in_ch: int = 256
    out_ch: int = 128
    scale_factor: int = 4
    block = FusionBlock(in_ch, out_ch, scale_factor=scale_factor).to(device)
    
    # Create input tensor
    x = torch.randn(2, in_ch, 16, 16).to(device)
    output = block(x)
    
    # Verify 4x spatial upsampling
    assert output.shape == (2, out_ch, 64, 64)
    logger.info("FusionBlock scale factor test passed")

def test_fusion_block_no_upsampling() -> None:
    device = get_test_device()
    
    # Initialize block without upsampling
    in_ch: int = 256
    out_ch: int = 128
    block = FusionBlock(in_ch, out_ch, scale_factor=1).to(device)
    
    # Create input tensor
    x = torch.randn(2, in_ch, 32, 32).to(device)
    output = block(x)
    
    # Verify same spatial dimensions
    assert output.shape == (2, out_ch, 32, 32)
    logger.info("FusionBlock no upsampling test passed")

# DPT Decoder Tests
def test_dpt_initialization() -> None:
    device = get_test_device()
    
    # Define channel configurations
    backbone_channels: List[int] = [96, 192, 384, 768]
    decoder_channels: List[int] = [256, 512, 1024, 1024]
    
    # Initialize decoder
    decoder = DPT(backbone_channels, decoder_channels).to(device)
    
    # Verify all components are properly initialized
    assert len(decoder.projections) == 4
    assert len(decoder.fusion_blocks) == 3
    assert hasattr(decoder, 'output_head')
    logger.info("DPT initialization test passed")

def test_dpt_forward_pass() -> None:
    device = get_test_device()
    
    # Initialize decoder
    backbone_channels: List[int] = [96, 192, 384, 768]
    decoder_channels: List[int] = [256, 512, 1024, 1024]
    decoder = DPT(backbone_channels, decoder_channels).to(device)
    
    # Create simulated multi-scale features
    features: List[torch.Tensor] = [
        torch.randn(1, 96, 64, 64).to(device),
        torch.randn(1, 192, 32, 32).to(device),
        torch.randn(1, 384, 16, 16).to(device),
        torch.randn(1, 768, 8, 8).to(device)
    ]
    
    # Execute forward pass
    output = decoder(features)
    
    # Verify output shape matches highest resolution input
    assert output.shape == (1, 1, 64, 64)
    assert output.requires_grad
    logger.info("DPT forward pass test passed")

def test_dpt_activation_functions() -> None:
    device = get_test_device()
    
    # Define test configurations
    backbone_channels: List[int] = [96, 192, 384, 768]
    decoder_channels: List[int] = [256, 512, 1024, 1024]
    activations: List[str] = ['sigmoid', 'tanh', 'relu', 'none']
    
    # Test each activation function
    for activation in activations:
        decoder = DPT(
            backbone_channels, 
            decoder_channels, 
            final_activation=activation
        ).to(device)
        
        # Create test features
        features: List[torch.Tensor] = [
            torch.randn(1, 96, 64, 64).to(device),
            torch.randn(1, 192, 32, 32).to(device),
            torch.randn(1, 384, 16, 16).to(device),
            torch.randn(1, 768, 8, 8).to(device)
        ]
        
        # Execute forward pass
        output = decoder(features)
        
        # Validate output range based on activation function
        if activation == 'sigmoid':
            assert torch.all(output >= 0) and torch.all(output <= 1)
        elif activation == 'tanh':
            assert torch.all(output >= -1) and torch.all(output <= 1)
        elif activation == 'relu':
            assert torch.all(output >= 0)
        
        logger.info(f"DPT {activation} activation test passed")

def test_dpt_with_attention_enabled() -> None:
    device = get_test_device()
    
    # Initialize decoder with attention
    backbone_channels: List[int] = [96, 192, 384, 768]
    decoder_channels: List[int] = [256, 512, 1024, 1024]
    decoder = DPT(
        backbone_channels, 
        decoder_channels, 
        use_attention=True
    ).to(device)
    
    # Create test features
    features: List[torch.Tensor] = [
        torch.randn(1, 96, 64, 64).to(device),
        torch.randn(1, 192, 32, 32).to(device),
        torch.randn(1, 384, 16, 16).to(device),
        torch.randn(1, 768, 8, 8).to(device)
    ]
    
    # Execute forward pass
    output = decoder(features)
    
    # Verify output shape
    assert output.shape == (1, 1, 64, 64)
    logger.info("DPT with attention test passed")

def test_dpt_model_info() -> None:
    device = get_test_device()
    
    # Initialize decoder
    backbone_channels: List[int] = [96, 192, 384, 768]
    decoder_channels: List[int] = [256, 512, 1024, 1024]
    decoder = DPT(backbone_channels, decoder_channels).to(device)
    
    # Get model information
    model_info: ModelInfo = decoder.get_model_info()
    assert isinstance(model_info, ModelInfo)
    
    # Verify summary contains expected fields
    summary = model_info.get_summary()
    assert 'total_parameters' in summary
    assert 'trainable_parameters' in summary
    assert 'model_size_mb' in summary
    logger.info("DPT get_model_info test passed")

# Validation Function Tests
def test_validate_dpt_features_valid() -> None:
    # Create valid feature tensors
    features: List[torch.Tensor] = [
        torch.randn(1, 96, 64, 64),
        torch.randn(1, 192, 32, 32),
        torch.randn(1, 384, 16, 16),
        torch.randn(1, 768, 8, 8)
    ]
    
    # Should not raise any exceptions
    validate_dpt_features(features)
    logger.info("DPT feature validation test passed")

def test_validate_features_compatibility() -> None:
    # Create compatible features and expected channels
    features: List[torch.Tensor] = [
        torch.randn(1, 96, 64, 64),
        torch.randn(1, 192, 32, 32),
        torch.randn(1, 384, 16, 16),
        torch.randn(1, 768, 8, 8)
    ]
    expected_channels: List[int] = [96, 192, 384, 768]
    
    # Should not raise any exceptions
    validate_features_compatibility(features, expected_channels)
    logger.info("Feature compatibility validation test passed")

def test_validate_features_count_mismatch() -> None:
    # Create fewer features than expected
    features: List[torch.Tensor] = [
        torch.randn(1, 96, 64, 64),
        torch.randn(1, 192, 32, 32),
        torch.randn(1, 384, 16, 16)
    ]
    expected_channels: List[int] = [96, 192, 384, 768]
    
    # Should raise TensorValidationError
    with pytest.raises(TensorValidationError):
        validate_features_compatibility(features, expected_channels)
    logger.info("Feature count mismatch validation test passed")

def test_validate_features_channel_mismatch() -> None:
    # Create features with channel mismatch
    features: List[torch.Tensor] = [
        torch.randn(1, 96, 64, 64),
        torch.randn(1, 128, 32, 32),  # Wrong channels: 128 instead of 192
        torch.randn(1, 384, 16, 16),
        torch.randn(1, 768, 8, 8)
    ]
    expected_channels: List[int] = [96, 192, 384, 768]
    
    # Should raise TensorValidationError
    with pytest.raises(TensorValidationError):
        validate_features_compatibility(features, expected_channels)
    logger.info("Feature channel mismatch validation test passed")

def test_tensor_validation() -> None:
    # Create valid tensor
    valid_tensor = torch.randn(2, 128, 32, 32)
    
    # Should not raise exceptions
    validate_tensor_input(valid_tensor, "test_tensor", expected_dims=4)
    
    # Test with invalid dimensions
    invalid_tensor = torch.randn(128, 32, 32)
    with pytest.raises(TensorValidationError):
        validate_tensor_input(invalid_tensor, "test_tensor", expected_dims=4)
    
    logger.info("Tensor validation test passed")

# Utility Function Tests
def test_interpolate_features_utility() -> None:
    device = get_test_device()
    
    # Create test tensor
    features = torch.randn(2, 128, 32, 32).to(device)
    target_size: Tuple[int, int] = (64, 64)
    
    # Execute interpolation
    output = interpolate_features(features, target_size)
    
    # Verify output shape
    assert output.shape == (2, 128, 64, 64)
    logger.info("Interpolate features utility test passed")

def test_interpolate_to_target_size() -> None:
    device = get_test_device()
    
    # Create test tensor
    features = torch.randn(1, 256, 16, 16).to(device)
    target_size: Tuple[int, int] = (32, 32)
    
    # Execute interpolation
    output = interpolate_to_target_size(features, target_size)
    
    # Verify output shape
    assert output.shape == (1, 256, 32, 32)
    logger.info("Interpolate to target size test passed")

def test_count_parameters_utility() -> None:
    device = get_test_device()
    
    # Create test decoder
    decoder = DPT([96, 192, 384, 768], [256, 256, 256, 256]).to(device)
    
    # Count parameters
    total_params: int = count_parameters(decoder, trainable_only=False)
    trainable_params: int = count_parameters(decoder, trainable_only=True)
    
    # Verify parameter counts
    assert total_params >= trainable_params
    assert total_params > 0
    assert trainable_params > 0
    logger.info("Parameter counting utility test passed")

def test_get_model_info_utility() -> None:
    device = get_test_device()
    
    # Create test decoder
    decoder = DPT([96, 192, 384, 768], [256, 256, 256, 256]).to(device)
    
    # Get model information
    model_info: ModelInfo = get_model_info(decoder)
    assert isinstance(model_info, ModelInfo)
    
    # Verify summary information
    summary = model_info.get_summary()
    assert summary['total_parameters'] > 0
    assert summary['trainable_parameters'] > 0
    assert summary['model_size_mb'] > 0
    assert summary['model_type'] == 'DPT'
    logger.info("Get model info utility test passed")

# Memory and Performance Tests
def test_memory_usage_large_input() -> None:
    device = get_test_device()
    
    # Clean up memory before test
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Create decoder
    decoder = DPT([96, 192, 384, 768], [256, 256, 256, 256]).to(device)
    
    # Create large input features
    features: List[torch.Tensor] = [
        torch.randn(1, 96, 256, 256).to(device),
        torch.randn(1, 192, 128, 128).to(device),
        torch.randn(1, 384, 64, 64).to(device),
        torch.randn(1, 768, 32, 32).to(device)
    ]
    
    # Monitor memory usage
    if torch.cuda.is_available():
        mem_before = torch.cuda.memory_allocated()
        output = decoder(features)
        mem_after = torch.cuda.memory_allocated()
        mem_used = (mem_after - mem_before) / (1024 ** 2)
        logger.info(f"GPU memory used: {mem_used:.2f} MB")
    else:
        process = psutil.Process()
        mem_before = process.memory_info().rss
        output = decoder(features)
        mem_after = process.memory_info().rss
        mem_used = (mem_after - mem_before) / (1024 ** 2)
        logger.info(f"CPU memory used: {mem_used:.2f} MB")
    
    # Verify output shape
    assert output.shape == (1, 1, 256, 256)
    logger.info("Memory usage test passed")

def test_gradient_computation() -> None:
    device = get_test_device()
    
    # Create decoder
    decoder = DPT([96, 192, 384, 768], [256, 256, 256, 256]).to(device)
    
    # Create features with gradient tracking
    features: List[torch.Tensor] = [
        torch.randn(1, 96, 64, 64, requires_grad=True).to(device),
        torch.randn(1, 192, 32, 32, requires_grad=True).to(device),
        torch.randn(1, 384, 16, 16, requires_grad=True).to(device),
        torch.randn(1, 768, 8, 8, requires_grad=True).to(device)
    ]
    
    # Forward pass and backward pass
    output = decoder(features)
    loss = output.sum()
    loss.backward()
    
    # Verify gradients are computed
    for i, feat in enumerate(features):
        assert feat.grad is not None
        logger.info(f"Gradient computed for feature {i}")
    
    logger.info("Gradient computation test passed")

# Integration Tests
def test_full_pipeline_integration() -> None:
    device = get_test_device()
    
    # Create decoder with attention and sigmoid activation
    decoder = DPT(
        [96, 192, 384, 768], 
        [256, 256, 256, 256], 
        use_attention=True, 
        final_activation='sigmoid'
    ).to(device)
    
    # Create batch of features
    batch_size: int = 2
    features: List[torch.Tensor] = [
        torch.randn(batch_size, 96, 64, 64).to(device),
        torch.randn(batch_size, 192, 32, 32).to(device),
        torch.randn(batch_size, 384, 16, 16).to(device),
        torch.randn(batch_size, 768, 8, 8).to(device)
    ]
    
    # Validate features before processing
    validate_dpt_features(features)
    validate_features_compatibility(features, [96, 192, 384, 768])
    
    # Execute inference
    with torch.no_grad():
        output = decoder(features)
    
    # Verify output properties
    assert output.shape == (batch_size, 1, 64, 64)
    assert torch.all(output >= 0) and torch.all(output <= 1)
    logger.info("Full pipeline integration test passed")

def test_error_handling() -> None:
    device = get_test_device()
    
    # Create decoder
    decoder = DPT([96, 192, 384, 768], [256, 256, 256, 256]).to(device)
    
    # Create incorrect number of input features
    wrong_features: List[torch.Tensor] = [
        torch.randn(1, 96, 64, 64).to(device),
        torch.randn(1, 192, 32, 32).to(device)
    ]
    
    # Should raise TensorValidationError
    with pytest.raises(TensorValidationError):
        decoder(wrong_features)
    
    logger.info("Error handling test passed")

def test_channel_mismatch_error_handling() -> None:
    device = get_test_device()
    
    # Create decoder expecting specific channels
    decoder = DPT([96, 192, 384, 768], [256, 256, 256, 256]).to(device)
    
    # Create features with wrong channel dimensions
    wrong_channel_features: List[torch.Tensor] = [
        torch.randn(1, 64, 64, 64).to(device),  # Wrong: 64 instead of 96
        torch.randn(1, 192, 32, 32).to(device),
        torch.randn(1, 384, 16, 16).to(device),
        torch.randn(1, 768, 8, 8).to(device)
    ]
    
    # Should raise validation error
    with pytest.raises(TensorValidationError):
        decoder(wrong_channel_features)
    
    logger.info("Channel mismatch error handling test passed")

def test_initialization_validation() -> None:
    # Test mismatched backbone and decoder channel lengths
    with pytest.raises(TensorValidationError):
        DPT([96, 192, 384], [256, 512, 1024, 1024])  # Mismatched lengths
    
    logger.info("Initialization validation test passed")

def test_interpolation_validation() -> None:
    device = get_test_device()
    
    # Create test tensor
    features = torch.randn(1, 256, 32, 32).to(device)
    
    # Test valid interpolation
    target_size: Tuple[int, int] = (64, 64)
    validate_interpolation_target(features, target_size)
    
    # Test invalid target size (should raise error)
    with pytest.raises(TensorValidationError):
        validate_interpolation_target(features, (-1, 64))  # Negative dimension
    
    logger.info("Interpolation validation test passed")
    
    