# FILE: tests/test_losses.py
# ehsanasgharzde - COMPLETE LOSS FUNCTION TEST SUITE
# hosseinsolymanzadeh - PROPER COMMENTING
# hosseinsolymanzadeh - FIXED REDUNDANT CODE BY EXTRACTING PURE FUNCTIONS AND BASECLASS LEVEL METHODS

import pytest
import torch
import numpy as np
import logging
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

# Updated imports to match current module structure
from losses.factory import (
    create_loss,
    create_combined_loss,
    validate_loss_config,
    register_loss,
    get_registered_losses,
    get_loss_weights_schedule,
    export_loss_configuration,
    integrate_with_trainer,
    visualize_loss_components,
    LOSS_REGISTRY,
    LOSS_CONFIG_SCHEMAS
)

from losses.losses import (
    SiLogLoss,
    EdgeAwareSmoothnessLoss,
    GradientConsistencyLoss,
    MultiScaleLoss,
    BerHuLoss,
    RMSELoss,
    MAELoss,
    MultiLoss,
)

# Import updated utility functions
from utils.loss import (
    BaseLoss,
    DepthLoss,
    GradientBasedLoss,
    ImageGuidedLoss,
    compute_loss_statistics,
    compute_spatial_gradients,
    compute_image_gradient_magnitude,
    compute_edge_weights
)

from utils.core import (
    create_default_mask,
    apply_mask_safely,
    validate_tensor_inputs,
    validate_numerical_stability,
    validate_depth_values,
    resize_tensors_to_scale,
    validate_depth_image_compatibility
)

from metrics.metrics import (
    rmse,
    mae,
    delta1,
    delta2,
    delta3,
    silog,
    compute_all_metrics,
    compute_batch_metrics,
    validate_metric_sanity,
    create_metric_report,
    compute_bootstrap_ci
)

# Configure logging for tests
logger = logging.getLogger(__name__)

# Test constants
TEST_SHAPES = [(1, 1, 32, 32), (2, 1, 64, 64), (1, 1, 128, 128)]
TEST_DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
ALL_LOSS_CLASSES = [
    SiLogLoss, EdgeAwareSmoothnessLoss, GradientConsistencyLoss,
    MultiScaleLoss, BerHuLoss, RMSELoss, MAELoss, MultiLoss
]


@pytest.fixture(scope='function')
def dummy_data() -> dict:
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
def edge_case_data() -> dict:
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


class TestCoreFunctions:
    
    def test_create_default_mask(self, dummy_data: dict) -> None:
        target = dummy_data['target']
        mask = create_default_mask(target)
        
        assert mask.dtype == torch.bool
        assert mask.shape == target.shape
        assert mask.sum() > 0  # Should have some valid pixels
    
    def test_apply_mask_safely(self, dummy_data: dict) -> None:
        pred = dummy_data['pred']
        mask = dummy_data['mask']
        
        masked_tensor, count = apply_mask_safely(pred, mask)
        
        assert count >= 0
        assert masked_tensor.numel() == count
        
        # Test shape mismatch
        wrong_mask = torch.ones(1, 1, 16, 16, dtype=torch.bool)
        with pytest.raises(ValueError, match="doesn't match mask shape"):
            apply_mask_safely(pred, wrong_mask)
    
    def test_validate_tensor_inputs(self, dummy_data: dict) -> None:
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
    
    def test_validate_numerical_stability(self) -> None:
        # Test with NaN values
        nan_tensor = torch.tensor([1.0, float('nan'), 3.0])
        cleaned = validate_numerical_stability(nan_tensor, "test")
        assert not torch.isnan(cleaned).any()
        
        # Test with Inf values
        inf_tensor = torch.tensor([1.0, float('inf'), 3.0])
        cleaned = validate_numerical_stability(inf_tensor, "test")
        assert not torch.isinf(cleaned).any()
    
    def test_resize_tensors_to_scale(self, dummy_data: dict) -> None:
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


class TestLossUtilities:
    
    def test_compute_spatial_gradients(self, dummy_data: dict) -> None:
        pred = dummy_data['pred']
        
        grad_x, grad_y = compute_spatial_gradients(pred)
        
        # Check dimensions are reduced by 1 in respective directions
        assert grad_x.shape[-1] == pred.shape[-1] - 1  # Width reduced
        assert grad_y.shape[-2] == pred.shape[-2] - 1  # Height reduced
    
    def test_compute_image_gradient_magnitude(self, dummy_data: dict) -> None:
        image = dummy_data['image']  # Should be 3-channel RGB
        
        grad_mag_x, grad_mag_y = compute_image_gradient_magnitude(image)
        
        # Should return single channel magnitude
        assert grad_mag_x.shape[1] == 1
        assert grad_mag_y.shape[1] == 1
        
        # Test with wrong input shape
        wrong_image = torch.rand(2, 1, 32, 32)  # Single channel instead of 3
        with pytest.raises(ValueError, match="Expected 4D RGB image"):
            compute_image_gradient_magnitude(wrong_image)
    
    def test_compute_edge_weights(self, dummy_data: dict) -> None:
        image = dummy_data['image']
        
        weight_x, weight_y = compute_edge_weights(image, alpha=1.0)
        
        assert weight_x.shape == (image.shape[0], 1, image.shape[2], image.shape[3])
        assert weight_y.shape == (image.shape[0], 1, image.shape[2], image.shape[3])
        
        # Weights should be in [0, 1] range due to exponential
        assert (weight_x >= 0).all() and (weight_x <= 1).all()
        assert (weight_y >= 0).all() and (weight_y <= 1).all()
    
    def test_compute_loss_statistics(self) -> None:
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


class TestBaseLossClasses:
    
    def test_base_loss_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseLoss()  # Should fail - abstract class
    
    def test_depth_loss_validation(self, dummy_data: dict) -> None:
        # Create a concrete implementation for testing
        class TestDepthLoss(DepthLoss):
            def compute_depth_loss(self, pred, target, mask=None, **kwargs):
                return torch.mean((pred - target) ** 2)
        
        pred = dummy_data['pred']
        target = dummy_data['target']
        mask = dummy_data['mask']
        
        loss_fn = TestDepthLoss()
        loss_value = loss_fn(pred, target, mask)
        
        assert_loss_properties(loss_value)
        assert len(loss_fn.loss_history) == 1
    
    def test_gradient_based_loss(self, dummy_data: dict) -> None:
        # Create a concrete implementation
        class TestGradientLoss(GradientBasedLoss):
            def compute_gradient_loss(self, pred_grad_x, pred_grad_y, target_grad_x, target_grad_y, mask=None, **kwargs):
                return torch.mean((pred_grad_x - target_grad_x) ** 2 + (pred_grad_y - target_grad_y) ** 2)
        
        pred = dummy_data['pred']
        target = dummy_data['target']
        
        loss_fn = TestGradientLoss()
        loss_value = loss_fn(pred, target)
        
        assert_loss_properties(loss_value)
    
    def test_image_guided_loss(self, dummy_data: dict) -> None:
        # Create a concrete implementation
        class TestImageGuidedLoss(ImageGuidedLoss):
            def compute_image_guided_loss(self, pred, image, mask=None, **kwargs):
                return torch.mean(pred ** 2)
        
        pred = dummy_data['pred']
        image = dummy_data['image']
        
        loss_fn = TestImageGuidedLoss()
        loss_value = loss_fn(pred, image, image=image)
        
        assert_loss_properties(loss_value)


class TestLossImplementations:
    
    @pytest.mark.parametrize('loss_class', ALL_LOSS_CLASSES)
    @pytest.mark.parametrize('shape', TEST_SHAPES)
    def test_loss_forward_backward(self, loss_class: type, shape: tuple, dummy_data: dict) -> None:
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
    def test_loss_device_consistency(self, device: str, dummy_data: dict) -> None:
        pred = dummy_data['pred'].to(device)
        target = dummy_data['target'].to(device)
        mask = dummy_data['mask'].to(device)
        
        loss_fn = SiLogLoss()
        loss_value = loss_fn(pred, target, mask)
        
        assert loss_value.device.type == device
        assert_loss_properties(loss_value)
    
    def test_silog_loss_parameters(self, dummy_data: dict) -> None:
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


class TestMetrics:
    
    def test_rmse_metric(self, dummy_data: dict) -> None:
        pred = dummy_data['pred']
        target = dummy_data['target']
        mask = dummy_data['mask']
        
        rmse_value = rmse(pred, target, mask)
        
        assert isinstance(rmse_value, float)
        assert rmse_value >= 0
        assert not np.isnan(rmse_value)
    
    def test_mae_metric(self, dummy_data: dict) -> None:
        pred = dummy_data['pred']
        target = dummy_data['target']
        mask = dummy_data['mask']
        
        mae_value = mae(pred, target, mask)
        
        assert isinstance(mae_value, float)
        assert mae_value >= 0
        assert not np.isnan(mae_value)
    
    @pytest.mark.parametrize('delta_func', [delta1, delta2, delta3])
    def test_delta_metrics(self, delta_func: callable, dummy_data: dict) -> None:
        pred = dummy_data['pred']
        target = dummy_data['target']
        mask = dummy_data['mask']
        
        delta_value = delta_func(pred, target, mask)
        
        assert isinstance(delta_value, float)
        assert 0.0 <= delta_value <= 1.0
    
    def test_silog_metric(self, dummy_data: dict) -> None:
        pred = dummy_data['pred']
        target = dummy_data['target']
        mask = dummy_data['mask']
        
        silog_value = silog(pred, target, mask)
        
        assert isinstance(silog_value, float)
        assert silog_value >= 0
        assert not np.isnan(silog_value)
    
    def test_compute_bootstrap_ci(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        lower, upper = compute_bootstrap_ci(values, confidence_level=0.95)
        
        assert not np.isnan(lower)
        assert not np.isnan(upper)
        assert lower <= upper
        
        # Test edge case with insufficient data
        small_values = np.array([1.0])
        lower, upper = compute_bootstrap_ci(small_values)
        assert np.isnan(lower) and np.isnan(upper)
    
    def test_validate_metric_sanity(self) -> None:
        # Valid metrics
        valid_metrics = {
            'rmse': 1.0,
            'mae': 0.8,
            'delta1': 0.95,
            'silog': 0.5
        }
        
        is_valid, warnings = validate_metric_sanity(valid_metrics)
        assert is_valid
        assert len(warnings) == 0
        
        # Invalid metrics
        invalid_metrics = {
            'rmse': 0.5,  # Less than MAE
            'mae': 0.8,
            'delta1': 1.5,  # Out of range
            'silog': -0.1  # Negative
        }
        
        is_valid, warnings = validate_metric_sanity(invalid_metrics)
        assert not is_valid
        assert len(warnings) > 0
    
    def test_create_metric_report(self, dummy_data: dict) -> None:
        pred = dummy_data['pred']
        target = dummy_data['target']
        mask = dummy_data['mask']
        
        report = create_metric_report(pred, target, mask, "test_dataset")
        
        assert 'dataset' in report
        assert 'valid_pixel_count' in report
        assert 'metrics' in report
        assert 'sanity_check' in report
        assert 'warnings' in report
        assert report['dataset'] == "test_dataset"


class TestLossFactory:
    
    def test_loss_registration(self) -> None:
        initial_count = len(get_registered_losses())
        
        # Register a new loss
        register_loss('test_loss', SiLogLoss)
        
        # Check registration
        assert 'test_loss' in get_registered_losses()
        assert len(get_registered_losses()) == initial_count + 1
        
        # Test overwrite warning
        with pytest.warns(UserWarning, match="already registered"):
            register_loss('test_loss', RMSELoss)
    
    @pytest.mark.parametrize('loss_name', ['SiLogLoss', 'BerHuLoss', 'RMSELoss', 'MAELoss'])
    def test_create_loss_by_name(self, loss_name: str, dummy_data: dict) -> None:
        loss_fn = create_loss(loss_name)
        
        pred = dummy_data['pred']
        target = dummy_data['target']
        mask = dummy_data['mask']
        
        loss_value = loss_fn(pred, target, mask)
        assert_loss_properties(loss_value)
    
    def test_create_loss_with_config(self, dummy_data: dict) -> None:
        config = {'lambda_var': 0.9, 'eps': 1e-6}
        loss_fn = create_loss('SiLogLoss', config)
        
        pred = dummy_data['pred']
        target = dummy_data['target']
        mask = dummy_data['mask']
        
        loss_value = loss_fn(pred, target, mask)
        assert_loss_properties(loss_value)
    
    def test_create_loss_invalid_name(self) -> None:
        with pytest.raises(ValueError, match="not found in registry"):
            create_loss('NonExistentLoss')
    
    def test_validate_loss_config(self) -> None:
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
    
    def test_create_combined_loss(self, dummy_data: dict) -> None:
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


class TestUtilityFunctions:
    
    def test_get_loss_weights_schedule(self) -> None:
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
    
    def test_export_loss_configuration(self) -> None:
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
    
    def test_integrate_with_trainer(self) -> None:
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
    
    def test_visualize_loss_components(self) -> None:
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


class TestEdgeCases:
    
    def test_empty_mask_handling(self, edge_case_data: dict) -> None:
        pred = torch.rand(1, 1, 16, 16, requires_grad=True) + 0.1
        target = torch.rand(1, 1, 16, 16) + 0.1
        empty_mask = edge_case_data['empty_mask']
        
        loss_fn = SiLogLoss()
        
        with pytest.warns(UserWarning, match="No valid pixels"):
            loss_value = loss_fn(pred, target, empty_mask)
        
        # Should return zero loss for empty mask
        assert torch.isclose(loss_value, torch.tensor(0.0))
    
    def test_shape_mismatch_errors(self, dummy_data: dict) -> None:
        pred = dummy_data['pred']
        target = torch.rand(1, 1, 16, 16)  # Different shape
        mask = dummy_data['mask']
        
        loss_fn = SiLogLoss()
        
        with pytest.raises((ValueError, RuntimeError)):
            loss_fn(pred, target, mask)
    
    def test_zero_tensors(self, edge_case_data: dict) -> None:
        zero_pred = edge_case_data['zero_pred']
        zero_target = edge_case_data['zero_target']
        full_mask = edge_case_data['full_mask']
        
        # Test with losses that can handle zero values
        loss_fn = MAELoss()
        loss_value = loss_fn(zero_pred, zero_target, full_mask)
        
        assert torch.isclose(loss_value, torch.tensor(0.0))


class TestPerformanceAndMemory:
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_memory_usage(self, dummy_data: dict) -> None:
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
    def test_batch_size_scaling(self, batch_size: int) -> None:
        pred = torch.rand(batch_size, 1, 32, 32, requires_grad=True) + 0.1
        target = torch.rand(batch_size, 1, 32, 32) + 0.1
        mask = torch.ones(batch_size, 1, 32, 32, dtype=torch.bool)
        
        loss_fn = SiLogLoss()
        loss_value = loss_fn(pred, target, mask)
        
        assert_loss_properties(loss_value)
        assert loss_value.shape == torch.Size([])  # Should be scalar


class TestIntegration:
    
    def test_complete_training_workflow(self, dummy_data: dict) -> None:
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
