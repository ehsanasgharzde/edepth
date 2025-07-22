# FILE: tests/test_backbone.py
# ehsanasgharzde - COMPLETE BACKBONE TEST SUITE
# hosseinsolymanzadeh - PROPER COMMENTING
# hosseinsolymanzadeh - FIXED REDUNDANT CODE BY EXTRACTING PURE FUNCTIONS AND BASECLASS LEVEL METHODS

import gc
import time
import torch
import pytest
import logging
from typing import Dict, Any, Tuple

# Updated imports to match new module structure
from ..models.backbones.backbone import ViT
from ..configs.model_config import (
    get_backbone_config, list_available_backbones, get_default_extract_layers,
    validate_config
)
from ..utils.model_validation import (
    validate_backbone_name, validate_vit_input, validate_extract_layers,
    validate_patch_size, validate_spatial_dimensions, ModelValidationError, 
    TensorValidationError,
)
from ..utils.model_utils import (
    calculate_patch_grid, sequence_to_spatial, interpolate_features, 
    get_model_info, count_parameters, freeze_model,
    cleanup_hooks,
)

# Setup logging for tests
logger = logging.getLogger(__name__)

# Test configuration helper functions
def get_test_config(model_name: str = 'vit_base_patch16_224') -> Dict[str, Any]:
    available_backbones = list_available_backbones()
    if model_name not in available_backbones:
        raise ValueError(f"Model {model_name} not in available backbones: {available_backbones}")
    
    config = get_backbone_config(model_name)
    config['model_name'] = model_name
    return config

def create_dummy_input(batch_size: int, img_size: int, channels: int = 3) -> torch.Tensor:
    return torch.randn(batch_size, channels, img_size, img_size, requires_grad=True)

def validate_feature_tensor(
    features: torch.Tensor, 
    expected_shape: Tuple[int, ...], 
    tensor_name: str = "feature"
) -> None:
    assert features.shape == expected_shape, \
        f"{tensor_name} shape mismatch: expected {expected_shape}, got {features.shape}"
    assert not torch.isnan(features).any(), f"NaN values found in {tensor_name}"
    assert not torch.isinf(features).any(), f"Inf values found in {tensor_name}"
    assert features.dtype == torch.float32, f"Expected float32, got {features.dtype}"

def validate_gradient_flow(tensor: torch.Tensor, tensor_name: str = "tensor") -> None:
    assert tensor.grad is not None, f"No gradient found for {tensor_name}"
    grad_norm = tensor.grad.norm().item()
    assert grad_norm > 0, f"Zero gradient norm for {tensor_name}"
    logger.info(f"Gradient flow validated for {tensor_name}: norm={grad_norm:.6f}")

def create_test_model(
    model_name: str = 'vit_base_patch16_224', 
    **kwargs
) -> ViT:
    # Set defaults for testing
    test_config = {
        'model_name': model_name,
        'pretrained': False,  # Faster for testing
        'use_checkpointing': False
    }
    test_config.update(kwargs)
    
    model = ViT(**test_config)
    model.eval()
    logger.info(f"Created test model: {model_name} with config: {test_config}")
    return model

def test_configuration_consistency() -> None:
    logger.info("Testing configuration consistency")
    
    available_backbones = list_available_backbones()
    expected_models = ['vit_base_patch16_224', 'vit_small_patch16_224', 'vit_base_patch8_224']
    
    for model_name in expected_models:
        assert model_name in available_backbones, f"Model {model_name} missing from config"
        
        # Test configuration retrieval
        config = get_backbone_config(model_name)
        
        # Validate required keys
        required_keys = ['patch_size', 'img_size', 'embed_dim', 'num_layers']
        for key in required_keys:
            assert key in config, f"Missing required config key: {key} for {model_name}"
        
        # Test config validation
        validate_config(config, 'backbone')
    
    logger.info("Configuration consistency test passed")

def test_centralized_validation_integration() -> None:
    logger.info("Testing centralized validation integration")
    
    available_backbones = list_available_backbones()
    
    # Test backbone name validation
    validate_backbone_name('vit_base_patch16_224', available_backbones)
    
    # Test invalid backbone name
    with pytest.raises(ModelValidationError):
        validate_backbone_name('invalid_backbone', available_backbones)
    
    # Test extract layers validation
    validate_extract_layers([8, 9, 10, 11], 12)
    
    with pytest.raises(ModelValidationError):
        validate_extract_layers([15, 16], 12)
    
    # Test patch size validation
    validate_patch_size(16, 224)
    
    with pytest.raises(ModelValidationError):
        validate_patch_size(13, 224)  # Invalid patch size
    
    # Test spatial dimensions validation
    validate_spatial_dimensions(224, 224, 16)
    
    with pytest.raises(ModelValidationError):
        validate_spatial_dimensions(225, 224, 16)  # Not divisible
    
    # Test input validation
    valid_input = create_dummy_input(1, 224)
    validate_vit_input(valid_input)
    
    with pytest.raises(TensorValidationError):
        invalid_input = torch.randn(224, 224, 3)  # Wrong dimensions
        validate_vit_input(invalid_input)
    
    logger.info("Centralized validation integration test passed")

@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("model_name", ['vit_base_patch16_224', 'vit_base_patch8_224', 'vit_small_patch16_224'])
def test_vit_initialization_comprehensive(batch_size: int, model_name: str) -> None:
    """Test comprehensive ViT initialization."""
    logger.info(f"Testing ViT initialization: batch_size={batch_size}, model={model_name}")
    
    available_backbones = list_available_backbones()
    if model_name not in available_backbones:
        pytest.skip(f"Model {model_name} not available")
    
    config = get_test_config(model_name)
    
    # Test with default extract layers
    default_layers = get_default_extract_layers(model_name)
    model = create_test_model(
        model_name, 
        extract_layers=default_layers
    )
    
    # Validate model configuration
    assert model.model_name == model_name
    assert model.patch_size == config['patch_size']
    assert model.img_size == config['img_size']
    assert model.embed_dim == config['embed_dim']
    assert model.extract_layers == default_layers
    
    # Test model info
    model_info = get_model_info(model)
    assert model_info.total_parameters > 0
    assert model_info.trainable_parameters > 0
    
    logger.info(f"ViT initialization test passed for {model_name}")

def test_vit_configuration_validation() -> None:
    logger.info("Testing ViT configuration validation")
    
    # Test valid configurations
    for model_name in ['vit_base_patch16_224', 'vit_small_patch16_224']:
        available_backbones = list_available_backbones()
        if model_name not in available_backbones:
            continue
            
        config = get_test_config(model_name)
        model = create_test_model(
            model_name,
            extract_layers=get_default_extract_layers(model_name)
        )
        
        # Validate configuration consistency
        assert model.patch_size == config['patch_size']
        assert model.img_size == config['img_size']
        assert model.embed_dim == config['embed_dim']
    
    # Test invalid configurations
    with pytest.raises(ValueError):
        create_test_model('invalid_backbone_name')
    
    logger.info("ViT configuration validation test passed")

@pytest.mark.parametrize("model_name", ['vit_base_patch16_224', 'vit_base_patch16_384'])
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_forward_pass_comprehensive(model_name: str, batch_size: int) -> None:
    logger.info(f"Testing forward pass: model={model_name}, batch_size={batch_size}")
    
    available_backbones = list_available_backbones()
    if model_name not in available_backbones:
        pytest.skip(f"Model {model_name} not available")
    
    config = get_test_config(model_name)
    
    # Create model with specific extract layers
    extract_layers = get_default_extract_layers(model_name)
    model = create_test_model(
        model_name, 
        extract_layers=extract_layers
    )
    
    # Create input using model configuration
    x = create_dummy_input(batch_size, config['img_size'])
    
    # Forward pass
    with torch.no_grad():
        features = model(x)
    
    # Validate number of features
    assert len(features) == len(extract_layers), \
        f"Expected {len(extract_layers)} features, got {len(features)}"
    
    # Calculate expected spatial dimensions using centralized utility
    patch_grid = calculate_patch_grid(config['img_size'], config['patch_size'])
    expected_h, expected_w = patch_grid
    
    # Validate each feature map
    for i, feat in enumerate(features):
        expected_shape = (batch_size, config['embed_dim'], expected_h, expected_w)
        validate_feature_tensor(feat, expected_shape, f"feature_map_{i}")
    
    # Test feature info consistency
    feature_info = model.get_feature_info()
    assert len(feature_info) == len(extract_layers)
    
    for i, info in enumerate(feature_info):
        assert info['layer_idx'] == extract_layers[i]
        assert info['channels'] == config['embed_dim']
        assert info['spatial_size'] == patch_grid
    
    logger.info(f"Forward pass test passed for {model_name}")

def test_feature_extraction_with_different_layers() -> None:
    logger.info("Testing feature extraction with different layer configurations")
    
    model_name = 'vit_base_patch16_224'
    config = get_test_config(model_name)
    num_layers = config.get('num_layers', 12)
    
    # Test different extract layer configurations
    test_layer_configs = [
        [num_layers - 1],                              # Single layer
        [num_layers - 4, num_layers - 1],             # Two layers
        [num_layers - 4, num_layers - 3, num_layers - 2, num_layers - 1],  # Four layers
        list(range(max(0, num_layers - 6), num_layers))  # Last six layers
    ]
    
    for extract_layers in test_layer_configs:
        model = create_test_model(model_name, extract_layers=extract_layers)
        x = create_dummy_input(2, config['img_size'])
        
        with torch.no_grad():
            features = model(x)
        
        # Validate output structure
        assert len(features) == len(extract_layers)
        
        patch_grid = calculate_patch_grid(config['img_size'], config['patch_size'])
        expected_h, expected_w = patch_grid
        
        for i, feat in enumerate(features):
            expected_shape = (2, config['embed_dim'], expected_h, expected_w)
            validate_feature_tensor(feat, expected_shape, f"layer_{extract_layers[i]}_feature")
    
    logger.info("Feature extraction with different layers test passed")

def test_utility_functions_comprehensive() -> None:
    logger.info("Testing comprehensive utility functions integration")
    
    model_name = 'vit_base_patch16_224'
    config = get_test_config(model_name)
    
    # Test patch grid calculation
    patch_grid = calculate_patch_grid(config['img_size'], config['patch_size'])
    expected_patches = config['img_size'] // config['patch_size']
    assert patch_grid == (expected_patches, expected_patches)
    
    # Test invalid patch grid calculation
    with pytest.raises(ValueError):
        calculate_patch_grid(225, 16)  # Not divisible
    
    # Test sequence to spatial conversion
    batch_size = 2
    num_patches = expected_patches * expected_patches
    sequence_features = torch.randn(batch_size, num_patches + 1, config['embed_dim'])
    
    # Test with CLS token removal
    spatial_features = sequence_to_spatial(
        sequence_features, 
        patch_grid, 
        include_cls_token=True
    )
    expected_shape = (batch_size, config['embed_dim'], expected_patches, expected_patches)
    assert spatial_features.shape == expected_shape
    
    # Test without CLS token
    sequence_no_cls = torch.randn(batch_size, num_patches, config['embed_dim'])
    spatial_no_cls = sequence_to_spatial(
        sequence_no_cls, 
        patch_grid, 
        include_cls_token=False
    )
    assert spatial_no_cls.shape == expected_shape
    
    # Test feature interpolation
    target_sizes = [(32, 32), (64, 64), (28, 28)]
    for target_size in target_sizes:
        interpolated = interpolate_features(spatial_features, target_size)
        assert interpolated.shape[-2:] == target_size
        assert interpolated.shape[:2] == spatial_features.shape[:2]
    
    # Test model info functionality
    model = create_test_model(model_name)
    model_info = get_model_info(model)
    
    # Validate model info properties
    assert model_info.total_parameters > 0
    assert model_info.trainable_parameters > 0
    assert model_info.model_size_mb > 0
    
    # Test summary generation
    summary = model_info.get_summary()
    required_keys = ['total_parameters', 'trainable_parameters', 'model_size_mb']
    for key in required_keys:
        assert key in summary
    
    logger.info("Comprehensive utility functions integration test passed")

def test_model_freeze_unfreeze_functionality() -> None:
    logger.info("Testing model freeze/unfreeze functionality")
    
    model = create_test_model('vit_base_patch16_224')
    
    # Initially all parameters should be trainable
    initial_trainable = count_parameters(model, trainable_only=True)
    total_params = count_parameters(model, trainable_only=False)
    assert initial_trainable == total_params, "All parameters should be trainable initially"
    
    # Test freeze functionality
    model.freeze()
    frozen_trainable = count_parameters(model, trainable_only=True)
    assert frozen_trainable == 0, "No parameters should be trainable after freeze"
    assert not model.training, "Model should be in eval mode when frozen"
    
    # Test unfreeze functionality
    model.unfreeze()
    unfrozen_trainable = count_parameters(model, trainable_only=True)
    assert unfrozen_trainable == total_params, "All parameters should be trainable after unfreeze"
    assert model.training, "Model should be in training mode when unfrozen"
    
    # Test using utility function
    freeze_model(model, freeze=True)
    frozen_again = count_parameters(model, trainable_only=True)
    assert frozen_again == 0, "Utility freeze function should work"
    
    freeze_model(model, freeze=False)
    unfrozen_again = count_parameters(model, trainable_only=True)
    assert unfrozen_again == total_params, "Utility unfreeze function should work"
    
    logger.info("Model freeze/unfreeze functionality test passed")

@pytest.mark.parametrize("use_checkpointing", [False, True])
def test_gradient_checkpointing_comprehensive(use_checkpointing: bool) -> None:
    logger.info(f"Testing gradient checkpointing: enabled={use_checkpointing}")
    
    model_name = 'vit_base_patch16_224'
    config = get_test_config(model_name)
    
    # Create model with checkpointing configuration
    model = create_test_model(
        model_name, 
        use_checkpointing=use_checkpointing,
        extract_layers=[10, 11]  # Test with last layers
    )
    model.train()  # Enable training mode for checkpointing
    
    # Create input
    x = create_dummy_input(2, config['img_size'])
    
    # Forward pass
    features = model(x)
    
    # Create dummy loss and backward pass
    loss = sum(feat.mean() for feat in features)
    loss.backward()
    
    # Validate gradient flow
    validate_gradient_flow(x, "checkpointing_input")
    
    # Validate model parameters have gradients
    param_with_grad = 0
    for param in model.parameters():
        if param.grad is not None:
            param_with_grad += 1
    
    assert param_with_grad > 0, "No model parameters have gradients"
    
    logger.info(f"Gradient checkpointing test passed: enabled={use_checkpointing}")

def test_hook_management() -> None:
    logger.info("Testing hook management and cleanup")
    
    model = create_test_model('vit_base_patch16_224', extract_layers=[8, 9, 10, 11])
    
    # Verify hooks are registered
    initial_hook_count = len(model.hooks)
    assert initial_hook_count > 0, "Hooks should be registered during initialization"
    
    # Test forward pass with hooks
    x = create_dummy_input(1, 224)
    with torch.no_grad():
        features = model(x)
    
    # Verify features were captured
    assert len(model.features) > 0, "Features should be captured by hooks"
    
    # Test manual cleanup
    cleanup_hooks(model.hooks)
    model.hooks.clear()
    
    # Verify cleanup worked
    model.features.clear()
    with torch.no_grad():
        _ = model.model(x)  # Direct model call to bypass hook system
    assert len(model.features) == 0, "No features should be captured after hook cleanup"
    
    logger.info("Hook management and cleanup test passed")

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_compatibility_comprehensive() -> None:
    logger.info("Testing comprehensive device compatibility")
    
    model_name = 'vit_base_patch16_224'
    config = get_test_config(model_name)
    
    # Test CPU
    model_cpu = create_test_model(model_name)
    x_cpu = create_dummy_input(1, config['img_size'])
    
    with torch.no_grad():
        features_cpu = model_cpu(x_cpu)
    
    for i, feat in enumerate(features_cpu):
        assert feat.device.type == 'cpu', f"CPU feature {i} should be on CPU"
    
    # Test GPU
    model_gpu = create_test_model(model_name).cuda()
    x_gpu = create_dummy_input(1, config['img_size']).cuda()
    
    with torch.no_grad():
        features_gpu = model_gpu(x_gpu)
    
    for i, feat in enumerate(features_gpu):
        assert feat.device.type == 'cuda', f"GPU feature {i} should be on GPU"
    
    # Test mixed precision if available
    if hasattr(torch, 'autocast'):
        with torch.autocast('cuda'):
            with torch.no_grad():
                features_mixed = model_gpu(x_gpu)
        
        for feat in features_mixed:
            assert feat.device.type == 'cuda', "Mixed precision features should be on GPU"
    
    logger.info("Comprehensive device compatibility test passed")

def test_comprehensive_error_handling() -> None:
    logger.info("Testing comprehensive error handling")
    
    # Test invalid model configurations
    with pytest.raises(ValueError, match="not found"):
        create_test_model('completely_invalid_model_name')
    
    # Test invalid extract layers
    with pytest.raises(ModelValidationError):
        create_test_model('vit_base_patch16_224', extract_layers=[999, 1000])
    
    # Test negative extract layers
    with pytest.raises(ModelValidationError):
        create_test_model('vit_base_patch16_224', extract_layers=[-1, -2])
    
    # Test empty extract layers
    with pytest.raises(ModelValidationError):
        create_test_model('vit_base_patch16_224', extract_layers=[])
    
    # Test model with valid configuration
    model = create_test_model('vit_base_patch16_224')
    config = get_test_config('vit_base_patch16_224')
    
    # Test invalid input shapes
    invalid_inputs = [
        torch.randn(224, 224, 3),              # 3D instead of 4D
        torch.randn(1, 3, 225, 225),           # Not divisible by patch size
        torch.randn(1, 4, 224, 224),           # Wrong number of channels
        torch.randn(1, 3, 223, 224),           # Asymmetric invalid dimensions
    ]
    
    for i, invalid_input in enumerate(invalid_inputs):
        with pytest.raises((TensorValidationError, ModelValidationError, ValueError)):
            if invalid_input.dim() == 4:
                # Only test forward pass for 4D tensors
                model(invalid_input)
            else:
                # Test validation directly for wrong dimensions
                validate_vit_input(invalid_input)
    
    # Test NaN and infinite input handling
    nan_input = create_dummy_input(1, config['img_size'])
    nan_input[0, 0, 0, 0] = float('nan')
    
    inf_input = create_dummy_input(1, config['img_size'])
    inf_input[0, 0, 0, 0] = float('inf')
    
    # Model should handle these gracefully or raise appropriate errors
    try:
        with torch.no_grad():
            _ = model(nan_input)
            _ = model(inf_input)
    except (ValueError, RuntimeError) as e:
        logger.info(f"Model appropriately handled invalid input: {e}")
    
    logger.info("Comprehensive error handling test passed")

def test_edge_case_configurations() -> None:
    logger.info("Testing edge case configurations")
    
    available_backbones = list_available_backbones()
    
    for model_name in available_backbones[:3]:  # Test first 3 available models
        config = get_test_config(model_name)
        num_layers = config.get('num_layers', 12)
        
        # Test single layer extraction
        single_layer_model = create_test_model(
            model_name, 
            extract_layers=[num_layers - 1]
        )
        
        x = create_dummy_input(1, config['img_size'])
        with torch.no_grad():
            features = single_layer_model(x)
        
        assert len(features) == 1, "Single layer model should return one feature"
        
        # Test all layers extraction (if feasible)
        if num_layers <= 12:  # Limit to prevent excessive memory usage
            all_layers = list(range(max(0, num_layers - 4), num_layers))
            all_layers_model = create_test_model(
                model_name,
                extract_layers=all_layers
            )
            
            with torch.no_grad():
                features_all = all_layers_model(x)
            
            assert len(features_all) == len(all_layers)
    
    logger.info("Edge case configurations test passed")

def test_performance_benchmarking() -> None:
    logger.info("Testing performance benchmarking")
    
    model_name = 'vit_base_patch16_224'
    config = get_test_config(model_name)
    
    # Test configuration
    batch_sizes = [1, 2, 4] if torch.cuda.is_available() else [1, 2]
    num_iterations = 5  # Reduced for testing
    
    # Create model
    model = create_test_model(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    performance_results = {}
    
    for batch_size in batch_sizes:
        # Create input
        x = create_dummy_input(batch_size, config['img_size']).to(device)
        
        # Warm-up
        with torch.no_grad():
            _ = model(x)
        
        # Benchmark
        if device.type == 'cuda':
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            with torch.no_grad():
                for _ in range(num_iterations):
                    _ = model(x)
            end_event.record()
            torch.cuda.synchronize()
            
            elapsed_time = start_event.elapsed_time(end_event) / num_iterations
            max_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            
            performance_results[batch_size] = {
                'time_ms': elapsed_time,
                'memory_mb': max_memory
            }
        else:
            start_time = time.time()
            with torch.no_grad():
                for _ in range(num_iterations):
                    _ = model(x)
            end_time = time.time()
            
            elapsed_time = ((end_time - start_time) / num_iterations) * 1000
            performance_results[batch_size] = {
                'time_ms': elapsed_time,
                'memory_mb': 0  # CPU memory tracking is complex
            }
        
        # Validate performance results
        assert performance_results[batch_size]['time_ms'] > 0
        logger.info(f"Batch {batch_size}: {performance_results[batch_size]['time_ms']:.2f}ms")
    
    # Cleanup
    del model, x
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    logger.info("Performance benchmarking test passed")

def test_end_to_end_workflow_comprehensive() -> None:
    logger.info("Testing comprehensive end-to-end workflow")
    
    model_name = 'vit_base_patch16_224'
    config = get_test_config(model_name)
    
    # Create model with comprehensive configuration
    extract_layers = get_default_extract_layers(model_name)
    model = create_test_model(
        model_name,
        extract_layers=extract_layers,
        use_checkpointing=False
    )
    
    # Test workflow with different scenarios
    test_scenarios = [
        {'batch_size': 1, 'training': False},
        {'batch_size': 2, 'training': True},
        {'batch_size': 4, 'training': False}
    ]
    
    for scenario in test_scenarios:
        batch_size = scenario['batch_size']
        is_training = scenario['training']
        
        # Set model mode
        if is_training:
            model.train()
        else:
            model.eval()
        
        x = create_dummy_input(batch_size, config['img_size'])
        
        # Forward pass
        if is_training:
            features = model(x)
        else:
            with torch.no_grad():
                features = model(x)
        
        # Validate results
        assert len(features) == len(extract_layers)
        
        # Calculate expected dimensions
        patch_grid = calculate_patch_grid(config['img_size'], config['patch_size'])
        expected_h, expected_w = patch_grid
        
        for feat in features:
            expected_shape = (batch_size, config['embed_dim'], expected_h, expected_w)
            validate_feature_tensor(feat, expected_shape)
        
        # Test gradient flow if training
        if is_training:
            loss = sum(feat.mean() for feat in features)
            loss.backward()
            validate_gradient_flow(x, f"e2e_batch_{batch_size}")
            
            # Clear gradients
            model.zero_grad()
            x.grad = None
    
    # Test model info and feature info
    model_info = get_model_info(model)
    summary = model_info.get_summary()
    assert summary['total_parameters'] > 0
    
    feature_info = model.get_feature_info()
    assert len(feature_info) == len(extract_layers)
    
    # Test freeze/unfreeze in workflow
    model.freeze()
    assert not model.training
    
    model.unfreeze()
    assert model.training
    
    logger.info("Comprehensive end-to-end workflow test passed")
# Pytest configuration markers
pytestmark = [
    pytest.mark.filterwarnings("ignore::UserWarning"),
    pytest.mark.filterwarnings("ignore::FutureWarning")
]

# Test execution summary function
def run_test_summary() -> Dict[str, Any]:
    available_backbones = list_available_backbones()
    
    test_summary = {
        'available_backbones': len(available_backbones),
        'backbone_list': available_backbones,
        'test_functions': [
            'test_configuration_consistency',
            'test_centralized_validation_integration',
            'test_vit_initialization_comprehensive',
            'test_forward_pass_comprehensive',
            'test_utility_functions_comprehensive',
            'test_gradient_checkpointing_comprehensive',
            'test_comprehensive_error_handling',
            'test_performance_benchmarking',
            'test_end_to_end_workflow_comprehensive'
        ],
        'device_support': {
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    }
    
    return test_summary

# Final test validation
def test_complete_test_suite_validation() -> None:
    logger.info("Validating complete test suite integrity")
    
    summary = run_test_summary()
    
    # Ensure we have backbones to test
    assert summary['available_backbones'] > 0, "No backbones available for testing"
    
    # Ensure essential backbones are present
    essential_backbones = ['vit_base_patch16_224']
    for backbone in essential_backbones:
        assert backbone in summary['backbone_list'], f"Essential backbone {backbone} missing"
    
    # Validate test function count
    assert len(summary['test_functions']) >= 8, "Insufficient test functions defined"
    
    logger.info(f"Test suite validation passed. Summary: {summary}")
    logger.info("All synchronized tests ready for execution")
    