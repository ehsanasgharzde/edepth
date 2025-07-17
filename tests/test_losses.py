# FILE: tests/test_losses.py
# ehsanasgharzde - COMPLETE LOSS FUNCTION TEST SUITE
# hosseinsolymanzadeh - PROPER COMMENTING

import pytest
import torch
import numpy as np
import logging
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from losses.factory import (
    create_loss,
    create_combined_loss,
    validate_loss_config,
    register_loss,
    get_registered_losses,
    get_loss_weights_schedule,
    export_loss_configuration,
    integrate_with_trainer,
    compute_loss_statistics,
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
def dummy_data():
    # Create dummy tensors for testing.
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
def edge_case_data():
    # Create edge case tensors for testing.
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

def assert_loss_properties(loss_value, should_be_finite=True, should_be_non_negative=True):
  
    if should_be_finite:
        assert torch.isfinite(loss_value).all(), f"Loss contains non-finite values: {loss_value}"
    
    if should_be_non_negative:
        assert loss_value.item() >= 0, f"Loss should be non-negative: {loss_value.item()}"

def assert_gradient_flow(pred_tensor, loss_value):
    
    # Check that gradients flow through the computation.
    if pred_tensor.requires_grad:
        loss_value.backward(retain_graph=True)
        assert pred_tensor.grad is not None, "Gradients should flow to prediction tensor"
        assert not torch.isnan(pred_tensor.grad).any(), "Gradients should not contain NaN"


class TestLoss:
    # Test individual loss function implementations.
    
    @pytest.mark.parametrize('loss_class', ALL_LOSS_CLASSES)
    @pytest.mark.parametrize('shape', TEST_SHAPES)
    def test_loss_forward_backward(self, loss_class, shape, dummy_data):
        # Test forward and backward pass for all loss functions.
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
                loss_value = loss_fn(pred, image)
            elif loss_class == MultiLoss:
                # MultiLoss requires a list of losses
                component_losses = [SiLogLoss(), RMSELoss()]
                loss_fn = loss_class(losses=component_losses, weights=[0.5, 0.5])
                loss_value = loss_fn(pred, target, mask, image)
            else:
                loss_fn = loss_class()
                loss_value = loss_fn(pred, target, mask)
            
            # Common assertions
            assert_loss_properties(loss_value)
            assert_gradient_flow(pred, loss_value)
            
        except Exception as e:
            pytest.fail(f"Loss {loss_class.__name__} failed with shape {shape}: {e}")
    
    @pytest.mark.parametrize('loss_class', [SiLogLoss, BerHuLoss, RMSELoss, MAELoss, GradientConsistencyLoss])
    def test_loss_without_mask(self, loss_class, dummy_data):
        # Test loss functions without explicit mask.
        pred = dummy_data['pred']
        target = dummy_data['target']
        
        loss_fn = loss_class()
        loss_value = loss_fn(pred, target)
        
        assert_loss_properties(loss_value)
        assert_gradient_flow(pred, loss_value)
    
    @pytest.mark.parametrize('device', TEST_DEVICES)
    def test_loss_device_consistency(self, device, dummy_data):
        # Test that losses work on different devices.
        pred = dummy_data['pred'].to(device)
        target = dummy_data['target'].to(device)
        mask = dummy_data['mask'].to(device)
        
        loss_fn = SiLogLoss()
        loss_value = loss_fn(pred, target, mask)
        
        assert loss_value.device.type == device
        assert_loss_properties(loss_value)
    
    def test_silog_loss_parameters(self, dummy_data):
        # Test SiLogLoss with various parameters.
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
    
    def test_berhu_loss_thresholds(self, dummy_data):
        # Test BerHuLoss with different threshold values.
        pred = dummy_data['pred']
        target = dummy_data['target']
        mask = dummy_data['mask']
        
        thresholds = [0.1, 0.2, 0.5, 1.0]
        
        for threshold in thresholds:
            loss_fn = BerHuLoss(threshold=threshold)
            loss_value = loss_fn(pred, target, mask)
            assert_loss_properties(loss_value)
    
    def test_multiscale_loss_configurations(self, dummy_data):
        # Test MultiScaleLoss with different scale configurations.
        pred = dummy_data['pred']
        target = dummy_data['target']
        mask = dummy_data['mask']
        
        configurations = [
            {'scales': [1.0], 'weights': [1.0]},
            {'scales': [1.0, 0.5], 'weights': [0.8, 0.2]},
            {'scales': [1.0, 0.5, 0.25], 'weights': [0.6, 0.3, 0.1]}
        ]
        
        for config in configurations:
            loss_fn = MultiScaleLoss(
                base_loss_fn=SiLogLoss(), #type: ignore
                scales=config['scales'], 
                weights=config['weights']
            )
            loss_value = loss_fn(pred, target, mask)
            assert_loss_properties(loss_value)


class TestEdgeCases:
    # Test edge cases and error conditions.
    
    @pytest.mark.parametrize('loss_class', [SiLogLoss, BerHuLoss, RMSELoss, MAELoss])
    def test_empty_mask_handling(self, loss_class, edge_case_data):
        # Test behavior with empty masks.
        pred = torch.rand(1, 1, 16, 16, requires_grad=True) + 0.1
        target = torch.rand(1, 1, 16, 16) + 0.1
        empty_mask = edge_case_data['empty_mask']
        
        loss_fn = loss_class()
        
        with pytest.warns(UserWarning, match="No valid pixels"):
            loss_value = loss_fn(pred, target, empty_mask)
        
        # Should return zero loss for empty mask
        assert torch.isclose(loss_value, torch.tensor(0.0))
    
    def test_nan_input_handling(self, dummy_data):
        # Test handling of NaN inputs.
        pred = dummy_data['pred'].clone()
        target = dummy_data['target'].clone()
        mask = dummy_data['mask']
        
        # Introduce NaN values
        pred[0, 0, 0, 0] = float('nan')
        
        loss_fn = SiLogLoss()
        
        # Should handle NaN gracefully (either warning or error)
        try:
            loss_value = loss_fn(pred, target, mask)
            # If it doesn't raise, loss should be finite
            assert torch.isfinite(loss_value) or torch.isclose(loss_value, torch.tensor(0.0))
        except (ValueError, RuntimeError):
            # It's acceptable to raise an error for NaN inputs
            pass
    
    def test_shape_mismatch_errors(self, dummy_data):
        # Test that shape mismatches raise appropriate errors.
        pred = dummy_data['pred']
        target = torch.rand(1, 1, 16, 16)  # Different shape
        mask = dummy_data['mask']
        
        loss_fn = SiLogLoss()
        
        with pytest.raises((ValueError, RuntimeError)):
            loss_fn(pred, target, mask)
    
    def test_zero_tensors(self, edge_case_data):
        # Test behavior with zero tensors.
        zero_pred = edge_case_data['zero_pred']
        zero_target = edge_case_data['zero_target']
        full_mask = edge_case_data['full_mask']
        
        # Test with losses that can handle zero values
        loss_fn = MAELoss()
        loss_value = loss_fn(zero_pred, zero_target, full_mask)
        
        assert torch.isclose(loss_value, torch.tensor(0.0))


class TestLossFactory:
    # Test loss factory functionality.
    
    def test_loss_registration(self):
        # Test loss registration and retrieval.
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
    def test_create_loss_by_name(self, loss_name, dummy_data):
        # Test creating losses by name.
        loss_fn = create_loss(loss_name)
        
        pred = dummy_data['pred']
        target = dummy_data['target']
        mask = dummy_data['mask']
        
        # Test that created loss works
        if loss_name == 'MultiLoss':
            # MultiLoss requires special handling
            pytest.skip("MultiLoss requires additional configuration")
        else:
            loss_value = loss_fn(pred, target, mask)
            assert_loss_properties(loss_value)
    
    def test_create_loss_with_config(self, dummy_data):
        # Test creating losses with configuration.
        config = {'lambda_var': 0.9, 'eps': 1e-6}
        loss_fn = create_loss('SiLogLoss', config)
        
        pred = dummy_data['pred']
        target = dummy_data['target']
        mask = dummy_data['mask']
        
        loss_value = loss_fn(pred, target, mask)
        assert_loss_properties(loss_value)
    
    def test_create_loss_invalid_name(self):
        # Test error handling for invalid loss names.
        with pytest.raises(ValueError, match="not found in registry"):
            create_loss('NonExistentLoss')
    
    def test_validate_loss_config(self):
        # Test loss configuration validation.
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
        
        # Test unknown loss (should pass through)
        config = {'custom_param': 123}
        validated = validate_loss_config('UnknownLoss', config)
        assert validated['custom_param'] == 123
    
    def test_create_combined_loss(self, dummy_data):
        # Test creating combined losses.
        loss_configs = [
            {'name': 'SiLogLoss', 'weight': 0.7, 'config': {'lambda_var': 0.85}},
            {'name': 'RMSELoss', 'weight': 0.3, 'config': {}}
        ]
        
        combined_loss = create_combined_loss(loss_configs)
        assert isinstance(combined_loss, MultiLoss)
        
        # Test combined loss functionality
        pred = dummy_data['pred']
        target = dummy_data['target']
        mask = dummy_data['mask']
        image = dummy_data['image']
        
        loss_value = combined_loss(pred, target, mask, image)
        assert_loss_properties(loss_value)
    
    def test_create_combined_loss_validation(self):
        # Test combined loss validation.
        # Test empty config
        with pytest.raises(ValueError, match="At least one loss"):
            create_combined_loss([])
        
        # Test missing name
        with pytest.raises(ValueError, match="missing 'name' field"):
            create_combined_loss([{'weight': 1.0}])


class TestUtilityFunctions:
    # Test utility functions.
    
    def test_compute_loss_statistics(self):
        # Test loss statistics computation.
        loss_values = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0]  # Include a spike
        
        stats = compute_loss_statistics(loss_values)
        
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'median' in stats
        assert 'count' in stats
        assert 'spike_count' in stats
        
        assert stats['count'] == len(loss_values)
        assert stats['min'] == 1.0
        assert stats['max'] == 10.0
        assert stats['spike_count'] >= 1  # Should detect the spike at 10.0
    
    def test_get_loss_weights_schedule(self):
        # Test loss weight scheduling.
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
    
    def test_export_loss_configuration(self, dummy_data):
        # Test loss configuration export.
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
    
    def test_integrate_with_trainer(self, dummy_data):
        # Test trainer integration.
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
        
        # Test combined loss integration
        combined_config = {
            'combined': [
                {'name': 'SiLogLoss', 'weight': 0.7},
                {'name': 'RMSELoss', 'weight': 0.3}
            ]
        }
        
        integrate_with_trainer(trainer, combined_config)
        assert isinstance(trainer.loss_fn, MultiLoss)


class TestPerformanceAndMemory:
    # Test performance and memory usage.
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_memory_usage(self, dummy_data):
        # Test CUDA memory usage.
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
    
    @pytest.mark.benchmark
    def test_loss_computation_speed(self, benchmark, dummy_data):
        # Benchmark loss computation speed.
        pred = dummy_data['pred']
        target = dummy_data['target']
        mask = dummy_data['mask']
        
        loss_fn = SiLogLoss()
        
        def loss_computation():
            return loss_fn(pred, target, mask)
        
        result = benchmark(loss_computation)
        assert_loss_properties(result)
    
    @pytest.mark.parametrize('batch_size', [1, 4, 8])
    def test_batch_size_scaling(self, batch_size):
        # Test that losses scale properly with batch size.
        pred = torch.rand(batch_size, 1, 32, 32, requires_grad=True) + 0.1
        target = torch.rand(batch_size, 1, 32, 32) + 0.1
        mask = torch.ones(batch_size, 1, 32, 32, dtype=torch.bool)
        
        loss_fn = SiLogLoss()
        loss_value = loss_fn(pred, target, mask)
        
        assert_loss_properties(loss_value)
        assert loss_value.shape == torch.Size([])  # Should be scalar

class TestIntegration:
    # Integration tests for complete workflows.
    
    def test_complete_training_workflow(self, dummy_data):
        # Test complete training workflow simulation.
        # Setup
        loss_configs = [
            {'name': 'SiLogLoss', 'weight': 0.7, 'config': {'lambda_var': 0.85}},
            {'name': 'EdgeAwareSmoothnessLoss', 'weight': 0.3, 'config': {}}
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
            loss_value = combined_loss(pred, target, mask, image)
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
    
    def test_factory_to_training_pipeline(self, dummy_data):
        # Test factory integration with training pipeline.
        # Create loss through factory
        loss_fn = create_loss('SiLogLoss', {'lambda_var': 0.9})
        
        # Simulate training
        pred = dummy_data['pred']
        target = dummy_data['target']
        mask = dummy_data['mask']
        
        # Multiple iterations
        for i in range(5):
            loss_value = loss_fn(pred, target, mask)
            assert_loss_properties(loss_value)
            
            loss_value.backward(retain_graph=True)
            assert pred.grad is not None
            
            pred.grad.zero_()

if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
