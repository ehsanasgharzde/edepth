# FILE: tests/test_losses.py
# ehsanasgharzde - COMPLETE LOSS FUNCTION TEST SUITE
# hosseinsolymanzadeh - PROPER COMMENTING
# hosseinsolymanzadeh - FIXED REDUNDANT CODE BY EXTRACTING PURE FUNCTIONS AND BASECLASS LEVEL METHODS

import json
import torch
import pytest
import logging
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, List, Tuple, Optional, Any

# Import loss factory functions
from ..losses.factory import (
    create_loss, create_combined_loss, validate_loss_config,
    register_loss, get_registered_losses, get_loss_weights_schedule,
    export_loss_configuration, integrate_with_trainer, visualize_loss_components,
    LOSS_REGISTRY, LOSS_CONFIG_SCHEMAS
)

# Import concrete loss implementations
from ..losses.losses import (
    SiLogLoss, EdgeAwareSmoothnessLoss, GradientConsistencyLoss,
    MultiScaleLoss, BerHuLoss, RMSELoss,
    MAELoss, MultiLoss,
)

# Import base loss classes and utility functions
from ..utils.loss import (
    compute_loss_statistics, compute_spatial_gradients, compute_image_gradient_magnitude,
    compute_edge_weights, ImageGuidedLoss, BaseLoss,
    DepthLoss, GradientBasedLoss,
)

# Import core utility functions
from ..utils.core import (
    create_default_mask, apply_mask_safely, validate_tensor_inputs,
    validate_numerical_stability, resize_tensors_to_scale, validate_depth_image_compatibility
)

# Configure logging for tests
logger = logging.getLogger(__name__)

# Test constants
TEST_SHAPES: List[Tuple[int, int, int, int]] = [(1, 1, 32, 32), (2, 1, 64, 64), (1, 1, 128, 128)]
TEST_DEVICES: List[str] = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
ALL_LOSS_CLASSES: List[type] = [
    SiLogLoss, EdgeAwareSmoothnessLoss, GradientConsistencyLoss,
    MultiScaleLoss, BerHuLoss, RMSELoss, MAELoss, MultiLoss
]

# Test fixtures
@pytest.fixture(scope='function')
def dummy_data() -> Dict[str, Any]:
    batch_size, channels, height, width = 2, 1, 32, 32
    
    # Create valid depth data (positive values)
    pred = torch.rand(batch_size, channels, height, width) * 10.0 + 0.1
    target = torch.rand(batch_size, channels, height, width) * 10.0 + 0.1
    image = torch.rand(batch_size, 3, height, width)
    
    # Create valid mask (some pixels masked out)
    mask = torch.rand(batch_size, channels, height, width) > 0.2
    
    # Enable gradient computation
    pred.requires_grad_(True)
    
    return {
        'pred': pred,
        'target': target,
        'image': image,
        'mask': mask,
        'shape': (batch_size, channels, height, width)
    }

@pytest.fixture(scope='function')
def edge_case_data() -> Dict[str, Any]:
    batch_size, channels, height, width = 1, 1, 16, 16
    
    return {
        'zero_pred': torch.zeros(batch_size, channels, height, width, requires_grad=True),
        'zero_target': torch.zeros(batch_size, channels, height, width),
        'nan_pred': torch.full((batch_size, channels, height, width), float('nan'), requires_grad=True),
        'inf_pred': torch.full((batch_size, channels, height, width), float('inf'), requires_grad=True),
        'empty_mask': torch.zeros(batch_size, channels, height, width, dtype=torch.bool),
        'full_mask': torch.ones(batch_size, channels, height, width, dtype=torch.bool),
        'shape': (batch_size, channels, height, width)
    }

# Utility functions for testing
def assert_loss_properties(loss_value: torch.Tensor, should_be_finite: bool = True, should_be_non_negative: bool = True) -> None:
    if should_be_finite:
        assert torch.isfinite(loss_value).all(), f"Loss contains non-finite values: {loss_value}"
    
    if should_be_non_negative:
        assert loss_value.item() >= 0, f"Loss should be non-negative: {loss_value.item()}"

def assert_gradient_flow(pred_tensor: torch.Tensor, loss_value: torch.Tensor) -> None:
    if pred_tensor.requires_grad:
        loss_value.backward(retain_graph=True)
        assert pred_tensor.grad is not None, "Gradients should flow to prediction tensor"
        assert not torch.isnan(pred_tensor.grad).any(), "Gradients should not contain NaN"

# Core function tests
def test_create_default_mask(dummy_data: Dict[str, Any]) -> None:
    target = dummy_data['target']
    mask = create_default_mask(target)
    
    assert mask.dtype == torch.bool
    assert mask.shape == target.shape
    assert mask.sum() > 0  # Should have some valid pixels

def test_apply_mask_safely(dummy_data: Dict[str, Any]) -> None:
    pred = dummy_data['pred']
    mask = dummy_data['mask']
    
    masked_tensor, count = apply_mask_safely(pred, mask)
    
    assert count >= 0
    assert masked_tensor.numel() == count
    
    # Test shape mismatch
    wrong_mask = torch.ones(1, 1, 16, 16, dtype=torch.bool)
    with pytest.raises(ValueError, match="doesn't match mask shape"):
        apply_mask_safely(pred, wrong_mask)

def test_validate_tensor_inputs(dummy_data: Dict[str, Any]) -> None:
    pred = dummy_data['pred']
    target = dummy_data['target']
    mask = dummy_data['mask']
    
    # Valid inputs should pass
    info = validate_tensor_inputs(pred, target, mask)
    assert 'shape' in info
    assert 'device' in info
    assert 'has_mask' in info
    
    # Shape mismatch should fail
    wrong_target = torch.rand(1, 1, 16, 16)
    with pytest.raises(ValueError, match="shapes must match"):
        validate_tensor_inputs(pred, wrong_target)

def test_validate_numerical_stability() -> None:
    # Test with NaN values
    nan_tensor = torch.tensor([1.0, float('nan'), 3.0])
    cleaned = validate_numerical_stability(nan_tensor, "test")
    assert not torch.isnan(cleaned).any()
    
    # Test with Inf values
    inf_tensor = torch.tensor([1.0, float('inf'), 3.0])
    cleaned = validate_numerical_stability(inf_tensor, "test")
    assert not torch.isinf(cleaned).any()

def test_resize_tensors_to_scale(dummy_data: Dict[str, Any]) -> None:
    pred = dummy_data['pred']
    target = dummy_data['target']
    mask = dummy_data['mask']
    
    # Test scale = 1.0 (no change)
    pred_1, target_1, mask_1 = resize_tensors_to_scale(pred, target, mask, 1.0)
    assert pred_1.shape == pred.shape
    
    # Test scale = 0.5 (half size)
    pred_half, target_half, mask_half = resize_tensors_to_scale(pred, target, mask, 0.5)
    expected_h, expected_w = int(pred.shape[2] * 0.5), int(pred.shape[3] * 0.5)
    assert pred_half.shape[2] == expected_h
    assert pred_half.shape[3] == expected_w

def test_validate_depth_image_compatibility(dummy_data: Dict[str, Any]) -> None:
    pred = dummy_data['pred']
    image = dummy_data['image']
    
    # Valid compatibility should pass
    validate_depth_image_compatibility(pred, image)
    
    # Test with wrong shapes
    wrong_image = torch.rand(1, 3, 16, 16)
    with pytest.raises(ValueError, match="Shape mismatch"):
        validate_depth_image_compatibility(pred, wrong_image)

# Loss utility function tests
def test_compute_spatial_gradients(dummy_data: Dict[str, Any]) -> None:
    pred = dummy_data['pred']
    
    grad_x, grad_y = compute_spatial_gradients(pred)
    
    # Check dimensions are reduced by 1 in respective directions
    assert grad_x.shape[-1] == pred.shape[-1] - 1  # Width reduced
    assert grad_y.shape[-2] == pred.shape[-2] - 1  # Height reduced

def test_compute_image_gradient_magnitude(dummy_data: Dict[str, Any]) -> None:
    image = dummy_data['image']  # Should be 3-channel RGB
    
    grad_mag_x, grad_mag_y = compute_image_gradient_magnitude(image)
    
    # Should return single channel magnitude
    assert grad_mag_x.shape[1] == 1
    assert grad_mag_y.shape[1] == 1
    
    # Test with wrong input shape
    wrong_image = torch.rand(2, 1, 32, 32)  # Single channel instead of 3
    with pytest.raises(ValueError, match="Expected 4D RGB image"):
        compute_image_gradient_magnitude(wrong_image)

def test_compute_edge_weights(dummy_data: Dict[str, Any]) -> None:
    image = dummy_data['image']
    
    weight_x, weight_y = compute_edge_weights(image, alpha=1.0)
    
    assert weight_x.shape == (image.shape[0], 1, image.shape[2], image.shape[3])
    assert weight_y.shape == (image.shape[0], 1, image.shape[2], image.shape[3])
    
    # Weights should be in [0, 1] range due to exponential
    assert (weight_x >= 0).all() and (weight_x <= 1).all()
    assert (weight_y >= 0).all() and (weight_y <= 1).all()

def test_compute_loss_statistics() -> None:
    loss_values = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    stats = compute_loss_statistics(loss_values)
    
    assert 'mean' in stats
    assert 'std' in stats
    assert 'min' in stats
    assert 'max' in stats
    assert 'median' in stats
    assert 'count' in stats
    
    assert stats['count'] == 5
    assert stats['min'] == 1.0
    assert stats['max'] == 5.0

# Base loss class tests
def test_base_loss_abstract() -> None:
    with pytest.raises(TypeError):
        BaseLoss()  # Should fail - abstract class

def test_depth_loss_validation(dummy_data: Dict[str, Any]) -> None:
    # Create a concrete implementation for testing
    class TestDepthLoss(DepthLoss):
        def compute_depth_loss(self, pred: torch.Tensor, target: torch.Tensor, 
                             mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
            return torch.mean((pred - target) ** 2)
    
    pred = dummy_data['pred']
    target = dummy_data['target']
    mask = dummy_data['mask']
    
    loss_fn = TestDepthLoss()
    loss_value = loss_fn(pred, target, mask)
    
    assert_loss_properties(loss_value)
    assert len(loss_fn.loss_history) == 1

def test_gradient_based_loss(dummy_data: Dict[str, Any]) -> None:
    # Create a concrete implementation
    class TestGradientLoss(GradientBasedLoss):
        def compute_gradient_loss(self, pred_grad_x: torch.Tensor, pred_grad_y: torch.Tensor,
                                target_grad_x: torch.Tensor, target_grad_y: torch.Tensor,
                                mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
            return torch.mean((pred_grad_x - target_grad_x) ** 2 + (pred_grad_y - target_grad_y) ** 2)
    
    pred = dummy_data['pred']
    target = dummy_data['target']
    
    loss_fn = TestGradientLoss()
    loss_value = loss_fn(pred, target)
    
    assert_loss_properties(loss_value)

def test_image_guided_loss(dummy_data: Dict[str, Any]) -> None:
    # Create a concrete implementation
    class TestImageGuidedLoss(ImageGuidedLoss):
        def compute_image_guided_loss(self, pred: torch.Tensor, image: torch.Tensor, 
                                    mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
            return torch.mean(pred ** 2)
    
    pred = dummy_data['pred']
    image = dummy_data['image']
    
    loss_fn = TestImageGuidedLoss()
    loss_value = loss_fn(pred, image, image=image)
    
    assert_loss_properties(loss_value)

# Individual loss implementation tests
@pytest.mark.parametrize('loss_class', ALL_LOSS_CLASSES)
@pytest.mark.parametrize('shape', TEST_SHAPES)
def test_loss_forward_backward(loss_class: type, shape: Tuple[int, int, int, int], dummy_data: Dict[str, Any]) -> None:
    batch_size, channels, height, width = shape
    
    # Create appropriately sized tensors
    pred = torch.rand(batch_size, channels, height, width, requires_grad=True) + 0.1
    target = torch.rand(batch_size, channels, height, width) + 0.1
    image = torch.rand(batch_size, 3, height, width)
    mask = torch.rand(batch_size, channels, height, width) > 0.2
    
    # Handle different loss function signatures
    try:
        if loss_class == EdgeAwareSmoothnessLoss:
            loss_fn = loss_class()
            loss_value = loss_fn(pred, image, image=image)
        elif loss_class == MultiScaleLoss:
            base_loss = SiLogLoss()
            loss_fn = loss_class(base_loss=base_loss)
            loss_value = loss_fn(pred, target, mask)
        elif loss_class == MultiLoss:
            # MultiLoss requires loss configurations
            loss_configs = [
                {'type': 'silog', 'weight': 0.7, 'params': {}},
                {'type': 'rmse', 'weight': 0.3, 'params': {}}
            ]
            loss_fn = loss_class(loss_configs=loss_configs)
            loss_value = loss_fn(pred, target, mask, image=image)
        else:
            loss_fn = loss_class()
            loss_value = loss_fn(pred, target, mask)
        
        # Common assertions
        assert_loss_properties(loss_value)
        assert_gradient_flow(pred, loss_value)
        
    except Exception as e:
        pytest.fail(f"Loss {loss_class.__name__} failed with shape {shape}: {e}")

@pytest.mark.parametrize('device', TEST_DEVICES)
def test_loss_device_consistency(device: str, dummy_data: Dict[str, Any]) -> None:
    pred = dummy_data['pred'].to(device)
    target = dummy_data['target'].to(device)
    mask = dummy_data['mask'].to(device)
    
    loss_fn = SiLogLoss()
    loss_value = loss_fn(pred, target, mask)
    
    assert loss_value.device.type == device
    assert_loss_properties(loss_value)

def test_silog_loss_parameters(dummy_data: Dict[str, Any]) -> None:
    pred = dummy_data['pred']
    target = dummy_data['target']
    mask = dummy_data['mask']
    
    # Test different parameter combinations
    params = [
        {'lambda_var': 0.85, 'eps': 1e-7},
        {'lambda_var': 0.5, 'eps': 1e-6},
        {'lambda_var': 1.0, 'eps': 1e-8}
    ]
    
    for param_set in params:
        loss_fn = SiLogLoss(**param_set)
        loss_value = loss_fn(pred, target, mask)
        assert_loss_properties(loss_value)

# Factory function tests
def test_loss_registration() -> None:
    initial_count = len(get_registered_losses())
    
    # Register a new loss
    register_loss('test_loss', SiLogLoss)
    
    # Check registration
    assert 'test_loss' in get_registered_losses()
    assert len(get_registered_losses()) == initial_count + 1

@pytest.mark.parametrize('loss_name', ['SiLogLoss', 'BerHuLoss', 'RMSELoss', 'MAELoss'])
def test_create_loss_by_name(loss_name: str, dummy_data: Dict[str, Any]) -> None:
    loss_fn = create_loss(loss_name)
    
    pred = dummy_data['pred']
    target = dummy_data['target']
    mask = dummy_data['mask']
    
    loss_value = loss_fn(pred, target, mask)
    assert_loss_properties(loss_value)

def test_create_loss_with_config(dummy_data: Dict[str, Any]) -> None:
    config = {'lambda_var': 0.9, 'eps': 1e-6}
    loss_fn = create_loss('SiLogLoss', config)
    
    pred = dummy_data['pred']
    target = dummy_data['target']
    mask = dummy_data['mask']
    
    loss_value = loss_fn(pred, target, mask)
    assert_loss_properties(loss_value)

def test_create_loss_invalid_name() -> None:
    with pytest.raises(ValueError, match="not found in registry"):
        create_loss('NonExistentLoss')

def test_validate_loss_config() -> None:
    # Test valid config
    config = {'lambda_var': 0.85, 'eps': 1e-7}
    validated = validate_loss_config('SiLogLoss', config)
    assert validated['lambda_var'] == 0.85
    assert validated['eps'] == 1e-7
    
    # Test with defaults
    config = {}
    validated = validate_loss_config('SiLogLoss', config)
    assert 'lambda_var' in validated
    assert 'eps' in validated

def test_create_combined_loss(dummy_data: Dict[str, Any]) -> None:
    loss_configs = [
        {'name': 'SiLogLoss', 'weight': 0.7, 'params': {'lambda_var': 0.85}},
        {'name': 'RMSELoss', 'weight': 0.3, 'params': {}}
    ]
    
    combined_loss = create_combined_loss(loss_configs)
    assert isinstance(combined_loss, MultiLoss)
    
    # Test combined loss functionality
    pred = dummy_data['pred']
    target = dummy_data['target']
    mask = dummy_data['mask']
    image = dummy_data['image']
    
    loss_value = combined_loss(pred, target, mask, image=image)
    assert_loss_properties(loss_value)

# Utility function tests
def test_get_loss_weights_schedule() -> None:
    base_weights = {'loss1': 1.0, 'loss2': 0.5}
    
    # Test constant schedule
    weights = get_loss_weights_schedule(5, 10, base_weights, 'constant')
    assert weights == base_weights
    
    # Test linear schedule
    weights = get_loss_weights_schedule(5, 10, base_weights, 'linear')
    assert weights['loss1'] < base_weights['loss1']  # Should decay
    
    # Test exponential schedule
    weights = get_loss_weights_schedule(5, 10, base_weights, 'exponential')
    assert weights['loss1'] < base_weights['loss1']  # Should decay
    
    # Test cosine schedule
    weights = get_loss_weights_schedule(5, 10, base_weights, 'cosine')
    assert weights['loss1'] < base_weights['loss1']  # Should decay

def test_export_loss_configuration() -> None:
    loss_fn = create_loss('SiLogLoss', {'lambda_var': 0.9})
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        export_path = f.name
    
    try:
        export_loss_configuration(loss_fn, export_path)
        
        # Verify file was created and contains expected data
        with open(export_path, 'r') as f:
            config = json.load(f)
        
        assert 'class_name' in config
        assert 'parameters' in config
        assert 'metadata' in config
        assert config['class_name'] == 'SiLogLoss'
    finally:
        Path(export_path).unlink(missing_ok=True)

def test_integrate_with_trainer() -> None:
    # Create mock trainer
    trainer = Mock()
    trainer.add_hook = Mock()
    
    loss_config = {
        'name': 'SiLogLoss',
        'config': {'lambda_var': 0.85}
    }
    
    integrate_with_trainer(trainer, loss_config)
    
    # Verify trainer was configured
    assert hasattr(trainer, 'loss_fn')
    assert trainer.loss_fn is not None

def test_visualize_loss_components() -> None:
    loss_history = {
        'silog': [1.0, 0.8, 0.6, 0.4],
        'rmse': [2.0, 1.8, 1.6, 1.4]
    }
    
    # Test without saving (just ensure no errors)
    with patch('matplotlib.pyplot.show'):
        visualize_loss_components(loss_history)
    
    # Test with empty history
    with patch('matplotlib.pyplot.show'):
        visualize_loss_components({})

# Edge case tests
def test_empty_mask_handling(edge_case_data: Dict[str, Any]) -> None:
    pred = torch.rand(1, 1, 16, 16, requires_grad=True) + 0.1
    target = torch.rand(1, 1, 16, 16) + 0.1
    empty_mask = edge_case_data['empty_mask']
    
    loss_fn = SiLogLoss()
    
    # Should handle empty mask gracefully
    loss_value = loss_fn(pred, target, empty_mask)
    
    # Should return a valid loss value
    assert torch.isfinite(loss_value)

def test_shape_mismatch_errors(dummy_data: Dict[str, Any]) -> None:
    pred = dummy_data['pred']
    target = torch.rand(1, 1, 16, 16)  # Different shape
    mask = dummy_data['mask']
    
    loss_fn = SiLogLoss()
    
    with pytest.raises((ValueError, RuntimeError)):
        loss_fn(pred, target, mask)

def test_zero_tensors(edge_case_data: Dict[str, Any]) -> None:
    zero_pred = edge_case_data['zero_pred']
    zero_target = edge_case_data['zero_target']
    full_mask = edge_case_data['full_mask']
    
    # Test with losses that can handle zero values
    loss_fn = MAELoss()
    loss_value = loss_fn(zero_pred, zero_target, full_mask)
    
    assert torch.isclose(loss_value, torch.tensor(0.0))

# Performance tests
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_memory_usage(dummy_data: Dict[str, Any]) -> None:
    device = 'cuda'
    
    pred = dummy_data['pred'].to(device)
    target = dummy_data['target'].to(device)
    mask = dummy_data['mask'].to(device)
    
    # Clear cache and measure initial memory
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated(device)
    
    # Run loss computation
    loss_fn = SiLogLoss()
    loss_value = loss_fn(pred, target, mask)
    loss_value.backward()
    
    # Check memory was allocated
    final_memory = torch.cuda.memory_allocated(device)
    assert final_memory > initial_memory
    
    # Cleanup
    torch.cuda.empty_cache()

@pytest.mark.parametrize('batch_size', [1, 4, 8])
def test_batch_size_scaling(batch_size: int) -> None:
    pred = torch.rand(batch_size, 1, 32, 32, requires_grad=True) + 0.1
    target = torch.rand(batch_size, 1, 32, 32) + 0.1
    mask = torch.ones(batch_size, 1, 32, 32, dtype=torch.bool)
    
    loss_fn = SiLogLoss()
    loss_value = loss_fn(pred, target, mask)
    
    assert_loss_properties(loss_value)
    assert loss_value.shape == torch.Size([])  # Should be scalar

# Integration tests
def test_complete_training_workflow(dummy_data: Dict[str, Any]) -> None:
    # Setup
    loss_configs = [
        {'name': 'SiLogLoss', 'weight': 0.7, 'params': {'lambda_var': 0.85}},
        {'name': 'RMSELoss', 'weight': 0.3, 'params': {}}
    ]
    
    combined_loss = create_combined_loss(loss_configs)
    
    # Simulate training loop
    pred = dummy_data['pred']
    target = dummy_data['target']
    mask = dummy_data['mask']
    image = dummy_data['image']
    
    loss_history = []
    
    for epoch in range(3):
        # Forward pass
        loss_value = combined_loss(pred, target, mask, image=image)
        loss_history.append(loss_value.item())
        
        # Backward pass
        loss_value.backward(retain_graph=True)
        
        # Check gradients
        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()
        
        # Reset gradients
        pred.grad.zero_()
    
    # Compute statistics
    stats = compute_loss_statistics(loss_history)
    assert stats['count'] == 3
    assert all(v >= 0 for v in loss_history)
    