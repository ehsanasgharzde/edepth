# FILE: tests/test_metrics.py
# ehsanasgharzde - COMPLETE METRICS TEST SUITE
# hosseinsolymanzadeh - PROPER COMMENTING
# hosseinsolymanzadeh - FIXED REDUNDANT CODE BY EXTRACTING PURE FUNCTIONS AND BASECLASS LEVEL METHODS

import torch
import pytest
import numpy as np
from typing import Dict, List, Optional, Callable, Union, Any

from metrics.metrics import (
    rmse, mae, delta1, delta2, delta3, silog, delta_metric,
    compute_bootstrap_ci, validate_metric_sanity, create_metric_report,
    compute_all_metrics, compute_batch_metrics
)
from metrics.factory import (
    get_metric, get_all_metrics, get_core_metrics,
    list_metrics, create_evaluator
)
from utils.core_utils import (
    create_default_mask, apply_mask_safely, resize_tensors_to_scale,
    validate_tensor_inputs, validate_depth_image_compatibility,
    validate_depth_values, validate_numerical_stability
)

def test_create_default_mask() -> None:
    # Test with valid depth values
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    valid_target: torch.Tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device)
    
    mask: torch.Tensor = create_default_mask(valid_target)
    expected_mask: torch.Tensor = torch.tensor([True, True, True, True, True], device=device)
    assert torch.equal(mask, expected_mask), "Should create correct mask for valid values"
    
    # Test with invalid depth values (zeros, negatives, inf, nan)
    invalid_target: torch.Tensor = torch.tensor([0.0, -1.0, float('inf'), float('nan')], device=device)
    mask = create_default_mask(invalid_target)
    expected_mask = torch.tensor([False, False, False, False], device=device)
    assert torch.equal(mask, expected_mask), "Should create correct mask for invalid values"
    
    # Test mixed valid and invalid values
    mixed_target: torch.Tensor = torch.tensor([1.0, 0.0, 3.0, -1.0, 5.0], device=device)
    mask = create_default_mask(mixed_target)
    expected_mask = torch.tensor([True, False, True, False, True], device=device)
    assert torch.equal(mask, expected_mask), "Should handle mixed valid/invalid values"

def test_apply_mask_safely() -> None:
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tensor: torch.Tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device)
    mask: torch.Tensor = torch.tensor([True, False, True, True, False], device=device)
    
    # Test normal masking operation
    masked_tensor: torch.Tensor
    count: int
    masked_tensor, count = apply_mask_safely(tensor, mask)
    
    expected_tensor: torch.Tensor = torch.tensor([1.0, 3.0, 4.0], device=device)
    assert torch.equal(masked_tensor, expected_tensor), "Should apply mask correctly"
    assert count == 3, "Should count valid elements correctly"
    
    # Test with empty mask (all False)
    empty_mask: torch.Tensor = torch.tensor([False, False, False, False, False], device=device)
    masked_tensor, count = apply_mask_safely(tensor, empty_mask)
    
    assert count == 0, "Should handle empty mask"
    assert masked_tensor.numel() == 0, "Should return empty tensor for empty mask"
    
    # Test shape mismatch error
    wrong_mask: torch.Tensor = torch.tensor([True, False], device=device)
    with pytest.raises(ValueError):
        apply_mask_safely(tensor, wrong_mask)

def test_validate_tensor_inputs() -> None:
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pred: torch.Tensor = torch.rand(32, 32, device=device)
    target: torch.Tensor = torch.rand(32, 32, device=device)
    mask: torch.Tensor = torch.ones(32, 32, dtype=torch.bool, device=device)
    
    # Test valid inputs with mask
    info: Dict[str, Any] = validate_tensor_inputs(pred, target, mask)
    
    assert info['shape'] == pred.shape, "Should return correct shape"
    assert info['device'] == pred.device, "Should return correct device"
    assert info['dtype'] == pred.dtype, "Should return correct dtype"
    assert info['has_mask'] == True, "Should detect mask presence"
    
    # Test valid inputs without mask
    info_no_mask: Dict[str, Any] = validate_tensor_inputs(pred, target)
    assert info_no_mask['has_mask'] == False, "Should detect no mask"
    
    # Test shape mismatch
    pred_wrong_shape: torch.Tensor = torch.rand(16, 16, device=device)
    with pytest.raises(ValueError):
        validate_tensor_inputs(pred_wrong_shape, target)
    
    # Test device mismatch
    if torch.cuda.is_available():
        pred_cpu: torch.Tensor = torch.rand(32, 32)
        with pytest.raises(ValueError):
            validate_tensor_inputs(pred_cpu, target)
    
    # Test dtype validation
    pred_int: torch.Tensor = torch.randint(0, 10, (32, 32), device=device)
    with pytest.raises(TypeError):
        validate_tensor_inputs(pred_int, target)

def test_resize_tensors_to_scale() -> None:
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pred: torch.Tensor = torch.rand(1, 1, 64, 64, device=device)
    target: torch.Tensor = torch.rand(1, 1, 64, 64, device=device)
    mask: torch.Tensor = torch.ones(1, 1, 64, 64, dtype=torch.bool, device=device)
    
    # Test no scaling (scale = 1.0)
    pred_resized: torch.Tensor
    target_resized: torch.Tensor
    mask_resized: Optional[torch.Tensor]
    pred_resized, target_resized, mask_resized = resize_tensors_to_scale(pred, target, mask, 1.0)
    
    assert torch.equal(pred_resized, pred), "No scaling should return original tensor"
    assert torch.equal(target_resized, target), "No scaling should return original tensor"
    assert mask_resized is not None and torch.equal(mask_resized, mask), "No scaling should return original mask"
    
    # Test downscaling
    pred_resized, target_resized, mask_resized = resize_tensors_to_scale(pred, target, mask, 0.5)
    
    assert pred_resized.shape == (1, 1, 32, 32), "Should resize to half dimensions"
    assert target_resized.shape == (1, 1, 32, 32), "Should resize to half dimensions"
    assert mask_resized is not None and mask_resized.shape == (1, 1, 32, 32), "Should resize mask to half dimensions"
    
    # Test upscaling
    pred_resized, target_resized, mask_resized = resize_tensors_to_scale(pred, target, mask, 2.0)
    
    assert pred_resized.shape == (1, 1, 128, 128), "Should resize to double dimensions"
    assert target_resized.shape == (1, 1, 128, 128), "Should resize to double dimensions"
    assert mask_resized is not None and mask_resized.shape == (1, 1, 128, 128), "Should resize mask to double dimensions"

def test_validate_depth_image_compatibility() -> None:
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create valid depth and image tensors
    depth: torch.Tensor = torch.rand(2, 1, 256, 256, device=device)  # batch=2, depth_channels=1
    image: torch.Tensor = torch.rand(2, 3, 256, 256, device=device)  # batch=2, rgb_channels=3
    
    # Should pass validation
    validate_depth_image_compatibility(depth, image)
    
    # Test batch size mismatch
    image_wrong_batch: torch.Tensor = torch.rand(3, 3, 256, 256, device=device)
    with pytest.raises(ValueError):
        validate_depth_image_compatibility(depth, image_wrong_batch)
    
    # Test spatial dimension mismatch
    image_wrong_size: torch.Tensor = torch.rand(2, 3, 128, 128, device=device)
    with pytest.raises(ValueError):
        validate_depth_image_compatibility(depth, image_wrong_size)
    
    # Test wrong number of depth channels
    depth_wrong_channels: torch.Tensor = torch.rand(2, 3, 256, 256, device=device)
    with pytest.raises(ValueError):
        validate_depth_image_compatibility(depth_wrong_channels, image)
    
    # Test wrong number of image channels
    image_wrong_channels: torch.Tensor = torch.rand(2, 1, 256, 256, device=device)
    with pytest.raises(ValueError):
        validate_depth_image_compatibility(depth, image_wrong_channels)

def test_validate_depth_values() -> None:
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test with valid depth values
    valid_depth: torch.Tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device)
    result: torch.Tensor = validate_depth_values(valid_depth)
    assert torch.equal(result, valid_depth), "Should return original tensor for valid values"
    
    # Test with negative values (should log warning but return tensor)
    negative_depth: torch.Tensor = torch.tensor([1.0, -2.0, 3.0], device=device)
    result = validate_depth_values(negative_depth)
    assert torch.equal(result, negative_depth), "Should return original tensor even with negatives"

def test_validate_numerical_stability() -> None:
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test with valid tensor
    valid_tensor: torch.Tensor = torch.tensor([1.0, 2.0, 3.0], device=device)
    result: torch.Tensor = validate_numerical_stability(valid_tensor, "test_tensor")
    assert torch.equal(result, valid_tensor), "Should return original tensor if valid"
    
    # Test with NaN values
    nan_tensor: torch.Tensor = torch.tensor([1.0, float('nan'), 3.0], device=device)
    result = validate_numerical_stability(nan_tensor, "nan_tensor")
    expected: torch.Tensor = torch.tensor([1.0, 0.0, 3.0], device=device)
    assert torch.equal(result, expected), "Should replace NaN with 0.0"
    
    # Test with Inf values
    inf_tensor: torch.Tensor = torch.tensor([1.0, float('inf'), -float('inf')], device=device)
    result = validate_numerical_stability(inf_tensor, "inf_tensor")
    expected = torch.tensor([1.0, 1e6, -1e6], device=device)
    assert torch.equal(result, expected), "Should replace Inf values with finite limits"

def test_rmse() -> None:
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test perfect predictions (should give RMSE = 0)
    pred_perfect: torch.Tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device)
    target_perfect: torch.Tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device)
    
    rmse_val: float = rmse(pred_perfect, target_perfect)
    assert abs(rmse_val) < 1e-6, f"RMSE should be ~0 for perfect predictions, got {rmse_val}"
    
    # Test known error pattern
    pred_offset: torch.Tensor = torch.tensor([1.1, 2.1, 3.1, 4.1, 5.1], device=device)
    target_base: torch.Tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device)
    
    rmse_val = rmse(pred_offset, target_base)
    expected_rmse: float = 0.1  # Constant offset
    assert abs(rmse_val - expected_rmse) < 1e-6, f"RMSE should be {expected_rmse}, got {rmse_val}"
    
    # Test with mask
    mask: torch.Tensor = torch.tensor([True, True, False, True, True], device=device)
    rmse_masked: float = rmse(pred_offset, target_base, mask)
    assert not np.isnan(rmse_masked), "RMSE with mask should not be NaN"
    
    # Test empty mask
    empty_mask: torch.Tensor = torch.tensor([False, False, False, False, False], device=device)
    rmse_empty: float = rmse(pred_offset, target_base, empty_mask)
    assert np.isnan(rmse_empty), "RMSE with empty mask should be NaN"

def test_mae() -> None:
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test perfect predictions
    pred_perfect: torch.Tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device)
    target_perfect: torch.Tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device)
    
    mae_val: float = mae(pred_perfect, target_perfect)
    assert abs(mae_val) < 1e-6, f"MAE should be ~0 for perfect predictions, got {mae_val}"
    
    # Test known error pattern
    pred_offset: torch.Tensor = torch.tensor([1.1, 2.1, 3.1, 4.1, 5.1], device=device)
    target_base: torch.Tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device)
    
    mae_val = mae(pred_offset, target_base)
    expected_mae: float = 0.1  # Constant absolute error
    assert abs(mae_val - expected_mae) < 1e-6, f"MAE should be {expected_mae}, got {mae_val}"

def test_delta_metrics() -> None:
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test perfect predictions (should give delta accuracy = 1.0)
    pred_perfect: torch.Tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device)
    target_perfect: torch.Tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device)
    
    delta1_val: float = delta1(pred_perfect, target_perfect)
    delta2_val: float = delta2(pred_perfect, target_perfect)
    delta3_val: float = delta3(pred_perfect, target_perfect)
    
    assert abs(delta1_val - 1.0) < 1e-6, f"Delta1 should be 1.0 for perfect predictions, got {delta1_val}"
    assert abs(delta2_val - 1.0) < 1e-6, f"Delta2 should be 1.0 for perfect predictions, got {delta2_val}"
    assert abs(delta3_val - 1.0) < 1e-6, f"Delta3 should be 1.0 for perfect predictions, got {delta3_val}"
    
    # Test small offset (should still have high delta accuracy)
    pred_small_offset: torch.Tensor = torch.tensor([1.05, 2.05, 3.05, 4.05, 5.05], device=device)
    target_base: torch.Tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device)
    
    delta1_small: float = delta1(pred_small_offset, target_base)
    assert delta1_small == 1.0, f"Delta1 should be 1.0 for small offset, got {delta1_small}"
    
    # Test direct delta_metric function
    delta_val: float = delta_metric(pred_perfect, target_perfect, 1.25)
    assert abs(delta_val - 1.0) < 1e-6, "delta_metric should return 1.0 for perfect predictions"

def test_silog() -> None:
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test perfect predictions
    pred_perfect: torch.Tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device)
    target_perfect: torch.Tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device)
    
    silog_val: float = silog(pred_perfect, target_perfect)
    assert abs(silog_val) < 1e-6, f"SiLog should be ~0 for perfect predictions, got {silog_val}"
    
    # Test with realistic depth values
    pred_realistic: torch.Tensor = torch.rand(100, device=device) * 10 + 0.1
    target_realistic: torch.Tensor = torch.rand(100, device=device) * 10 + 0.1
    
    silog_val = silog(pred_realistic, target_realistic)
    assert not np.isnan(silog_val), "SiLog should not be NaN for realistic values"
    assert silog_val >= 0, "SiLog should be non-negative"

def test_compute_all_metrics() -> None:
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pred: torch.Tensor = torch.rand(256, 256, device=device) * 10 + 0.1
    target: torch.Tensor = torch.rand(256, 256, device=device) * 10 + 0.1
    
    # Test without confidence intervals
    metrics: Dict[str, Union[float, Dict]] = compute_all_metrics(pred, target)
    
    expected_metrics: List[str] = ['rmse', 'mae', 'delta1', 'delta2', 'delta3', 'silog']
    for metric_name in expected_metrics:
        assert metric_name in metrics, f"Should contain {metric_name}"
        assert isinstance(metrics[metric_name], float), f"{metric_name} should be float"
        assert not np.isnan(metrics[metric_name]), f"{metric_name} should not be NaN"

def test_compute_batch_metrics() -> None:
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size: int = 4
    
    pred_batch: torch.Tensor = torch.rand(batch_size, 128, 128, device=device) * 10 + 0.1
    target_batch: torch.Tensor = torch.rand(batch_size, 128, 128, device=device) * 10 + 0.1
    
    # Test default metrics
    results: Dict[str, Dict[str, Union[float, List]]] = compute_batch_metrics(pred_batch, target_batch)
    
    expected_metrics: List[str] = ['rmse', 'mae', 'delta1', 'delta2', 'delta3', 'silog']
    for metric_name in expected_metrics:
        assert metric_name in results, f"Should contain {metric_name}"
        
        metric_stats: Dict[str, Union[float, List]] = results[metric_name]
        assert 'mean' in metric_stats, f"{metric_name} should have mean"
        assert 'std' in metric_stats, f"{metric_name} should have std"
        assert 'min' in metric_stats, f"{metric_name} should have min"
        assert 'max' in metric_stats, f"{metric_name} should have max"
        assert 'per_sample' in metric_stats, f"{metric_name} should have per_sample"
        
        per_sample: List = metric_stats['per_sample']
        assert len(per_sample) == batch_size, f"Should have {batch_size} per-sample values"

def test_validate_metric_sanity() -> None:
    # Test valid metrics
    valid_metrics: Dict[str, float] = {
        'rmse': 1.5,
        'mae': 1.0,
        'delta1': 0.8,
        'delta2': 0.9,
        'delta3': 0.95,
        'silog': 0.2
    }
    
    is_valid: bool
    warnings_list: List[str]
    is_valid, warnings_list = validate_metric_sanity(valid_metrics)
    
    assert is_valid, "Valid metrics should pass sanity check"
    assert len(warnings_list) == 0, "Should not have warnings for valid metrics"
    
    # Test invalid metrics (RMSE < MAE)
    invalid_metrics: Dict[str, float] = {
        'rmse': 0.5,
        'mae': 1.0,
        'delta1': 0.8,
        'silog': 0.2
    }
    
    is_valid, warnings_list = validate_metric_sanity(invalid_metrics)
    assert not is_valid, "Invalid metrics should fail sanity check"
    assert len(warnings_list) > 0, "Should have warnings for invalid metrics"
    
    # Test out-of-range delta values
    invalid_delta_metrics: Dict[str, float] = {
        'rmse': 1.5,
        'mae': 1.0,
        'delta1': 1.5,  # Invalid: > 1.0
        'silog': 0.2
    }
    
    is_valid, warnings_list = validate_metric_sanity(invalid_delta_metrics)
    assert not is_valid, "Out-of-range delta should fail sanity check"
    assert any('delta1' in warning for warning in warnings_list), "Should warn about delta1"

def test_create_metric_report() -> None:
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pred: torch.Tensor = torch.rand(128, 128, device=device) * 10 + 0.1
    target: torch.Tensor = torch.rand(128, 128, device=device) * 10 + 0.1
    dataset_name: str = "TestDataset"
    
    report: Dict = create_metric_report(pred, target, dataset_name=dataset_name)
    
    # Check report structure
    assert 'dataset' in report, "Report should contain dataset name"
    assert report['dataset'] == dataset_name, "Should have correct dataset name"
    assert 'valid_pixel_count' in report, "Report should contain valid pixel count"
    assert 'depth_statistics' in report, "Report should contain depth statistics"
    assert 'metrics' in report, "Report should contain metrics"
    assert 'sanity_check' in report, "Report should contain sanity check result"
    assert 'warnings' in report, "Report should contain warnings"
    
    # Check depth statistics structure
    depth_stats: Dict = report['depth_statistics']
    stat_keys: List[str] = ['min', 'max', 'mean', 'std']
    for key in stat_keys:
        assert key in depth_stats, f"Should contain {key} depth statistic"
    
    # Check metrics are computed
    metrics: Dict = report['metrics']
    expected_metrics: List[str] = ['rmse', 'mae', 'delta1', 'delta2', 'delta3', 'silog']
    for metric in expected_metrics:
        assert metric in metrics, f"Metric {metric} should be in report"

def test_compute_bootstrap_ci() -> None:
    # Test with sufficient data
    values: np.ndarray = np.random.normal(5.0, 1.0, 1000)
    
    ci_lower: float
    ci_upper: float
    ci_lower, ci_upper = compute_bootstrap_ci(values, confidence_level=0.95)
    
    assert not np.isnan(ci_lower), "CI lower should not be NaN"
    assert not np.isnan(ci_upper), "CI upper should not be NaN"
    assert ci_lower < ci_upper, "CI lower should be less than upper"
    assert abs(ci_lower - 5.0) < 1.0, "CI should be near true mean"
    assert abs(ci_upper - 5.0) < 1.0, "CI should be near true mean"
    
    # Test with insufficient data
    small_values: np.ndarray = np.array([1.0])
    ci_lower, ci_upper = compute_bootstrap_ci(small_values)
    
    assert np.isnan(ci_lower), "CI lower should be NaN for insufficient data"
    assert np.isnan(ci_upper), "CI upper should be NaN for insufficient data"
    
    # Test different confidence levels
    ci_lower_90: float
    ci_upper_90: float
    ci_lower_90, ci_upper_90 = compute_bootstrap_ci(values, confidence_level=0.90)
    
    # 90% CI should be narrower than 95% CI
    assert (ci_upper_90 - ci_lower_90) < (ci_upper - ci_lower), "90% CI should be narrower than 95% CI"

def test_get_metric() -> None:
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pred: torch.Tensor = torch.rand(100, device=device) * 10 + 0.1
    target: torch.Tensor = torch.rand(100, device=device) * 10 + 0.1
    
    # Test valid metric names
    rmse_func: Callable = get_metric('rmse')
    mae_func: Callable = get_metric('mae')
    delta1_func: Callable = get_metric('delta1')
    silog_func: Callable = get_metric('silog')
    
    # Test that functions work
    rmse_val: float = rmse_func(pred, target)
    mae_val: float = mae_func(pred, target)
    delta1_val: float = delta1_func(pred, target)
    silog_val: float = silog_func(pred, target)
    
    assert not np.isnan(rmse_val), "RMSE function should work"
    assert not np.isnan(mae_val), "MAE function should work"
    assert not np.isnan(delta1_val), "Delta1 function should work"
    assert not np.isnan(silog_val), "SiLog function should work"
    
    # Test invalid metric name
    with pytest.raises(ValueError, match="Unknown metric"):
        get_metric('invalid_metric')

def test_get_all_metrics() -> None:
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pred: torch.Tensor = torch.rand(100, device=device) * 10 + 0.1
    target: torch.Tensor = torch.rand(100, device=device) * 10 + 0.1
    
    all_metrics: Dict[str, Callable] = get_all_metrics()
    
    expected_metrics: List[str] = ['rmse', 'mae', 'delta1', 'delta2', 'delta3', 'silog']
    for metric_name in expected_metrics:
        assert metric_name in all_metrics, f"Should contain {metric_name}"
        
        # Test that function works
        metric_func: Callable = all_metrics[metric_name]
        result: float = metric_func(pred, target)
        assert not np.isnan(result), f"{metric_name} should produce valid result"

def test_get_core_metrics() -> None:
    core_metrics: Dict[str, Callable] = get_core_metrics()
    all_metrics: Dict[str, Callable] = get_all_metrics()
    
    # Core metrics should be same as all metrics in current implementation
    assert set(core_metrics.keys()) == set(all_metrics.keys()), "Core metrics should match all metrics"

def test_list_metrics() -> None:
    metric_names: List[str] = list_metrics()
    
    expected_metrics: List[str] = ['rmse', 'mae', 'delta1', 'delta2', 'delta3', 'silog']
    assert set(metric_names) == set(expected_metrics), "Should return expected metric names"
    assert len(metric_names) == len(expected_metrics), "Should have correct count"

def test_create_evaluator() -> None:
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pred: torch.Tensor = torch.rand(100, device=device) * 10 + 0.1
    target: torch.Tensor = torch.rand(100, device=device) * 10 + 0.1
    
    # Test default evaluator (all metrics)
    evaluator: Callable = create_evaluator()
    results: Dict[str, float] = evaluator(pred, target)
    
    expected_metrics: List[str] = ['rmse', 'mae', 'delta1', 'delta2', 'delta3', 'silog']
    for metric_name in expected_metrics:
        assert metric_name in results, f"Should contain {metric_name}"
        assert not np.isnan(results[metric_name]), f"{metric_name} should be valid"
    
    # Test custom evaluator with subset of metrics
    custom_metrics: List[str] = ['rmse', 'mae', 'delta1']
    custom_evaluator: Callable = create_evaluator(custom_metrics)
    custom_results: Dict[str, float] = custom_evaluator(pred, target)
    
    assert len(custom_results) == 3, "Should only contain requested metrics"
    for metric_name in custom_metrics:
        assert metric_name in custom_results, f"Should contain {metric_name}"
        assert not np.isnan(custom_results[metric_name]), f"{metric_name} should be valid"
    
    # Test with mask
    mask: torch.Tensor = torch.ones(100, dtype=torch.bool, device=device)
    mask_results: Dict[str, float] = evaluator(pred, target, mask)
    
    for metric_name in expected_metrics:
        assert metric_name in mask_results, f"Should handle mask for {metric_name}"
    
    # Test invalid metric in evaluator
    with pytest.raises(ValueError, match="Unknown metric"):
        create_evaluator(['invalid_metric'])

def test_integration_pipeline() -> None:
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Simulate realistic model predictions and ground truth
    batch_size: int = 4
    height: int = 256
    width: int = 256
    
    pred: torch.Tensor = torch.rand(batch_size, height, width, device=device) * 10 + 0.1
    target: torch.Tensor = torch.rand(batch_size, height, width, device=device) * 10 + 0.1
    
    # Create evaluator with all metrics
    evaluator: Callable = create_evaluator()
    
    # Process each sample in batch
    batch_results: List[Dict[str, float]] = []
    for i in range(batch_size):
        results: Dict[str, float] = evaluator(pred[i], target[i])
        batch_results.append(results)
    
    # Validate all results
    for i, results in enumerate(batch_results):
        is_valid: bool
        warnings_list: List[str]
        is_valid, warnings_list = validate_metric_sanity(results)
        
        # Create individual reports
        report: Dict = create_metric_report(pred[i], target[i], dataset_name=f"Sample_{i}")
        
        assert 'metrics' in report, f"Sample {i} should have metrics in report"
        assert len(results) == 6, f"Sample {i} should have all 6 metrics"

def test_numerical_edge_cases() -> None:
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test very small values near epsilon
    tiny_pred: torch.Tensor = torch.full((100,), 1e-8, device=device)
    tiny_target: torch.Tensor = torch.full((100,), 1e-8, device=device)
    
    rmse_tiny: float = rmse(tiny_pred, tiny_target)
    mae_tiny: float = mae(tiny_pred, tiny_target)
    silog_tiny: float = silog(tiny_pred, tiny_target)
    
    assert not np.isnan(rmse_tiny), "RMSE should handle tiny values"
    assert not np.isnan(mae_tiny), "MAE should handle tiny values"
    assert not np.isnan(silog_tiny), "SiLog should handle tiny values"
    
    # Test large values
    large_pred: torch.Tensor = torch.full((100,), 1e6, device=device)
    large_target: torch.Tensor = torch.full((100,), 1e6, device=device)
    
    rmse_large: float = rmse(large_pred, large_target)
    mae_large: float = mae(large_pred, large_target)
    
    assert not np.isnan(rmse_large), "RMSE should handle large values"
    assert not np.isnan(mae_large), "MAE should handle large values"
    
    # Test mixed scales
    mixed_pred: torch.Tensor = torch.tensor([1e-6, 1.0, 1e6], device=device)
    mixed_target: torch.Tensor = torch.tensor([1e-6, 1.0, 1e6], device=device)
    
    rmse_mixed: float = rmse(mixed_pred, mixed_target)
    assert not np.isnan(rmse_mixed), "Should handle mixed scales"

def test_memory_efficiency() -> None:
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test with reasonably large tensors
    if torch.cuda.is_available():
        try:
            large_pred: torch.Tensor = torch.rand(1024, 1024, device=device) * 10 + 0.1
            large_target: torch.Tensor = torch.rand(1024, 1024, device=device) * 10 + 0.1
            
            # Should not run out of memory
            evaluator: Callable = create_evaluator(['rmse', 'mae'])
            results: Dict[str, float] = evaluator(large_pred, large_target)
            
            assert not np.isnan(results['rmse']), "RMSE should work with large tensors"
            assert not np.isnan(results['mae']), "MAE should work with large tensors"
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                pytest.skip("Skipping large tensor test due to memory constraints")
            else:
                raise

def test_device_compatibility() -> None:
    # Test CPU tensors
    cpu_pred: torch.Tensor = torch.rand(50, 50) * 10 + 0.1
    cpu_target: torch.Tensor = torch.rand(50, 50) * 10 + 0.1
    
    evaluator: Callable = create_evaluator(['rmse', 'mae'])
    cpu_results: Dict[str, float] = evaluator(cpu_pred, cpu_target)
    
    assert not np.isnan(cpu_results['rmse']), "Should handle CPU tensors"
    assert not np.isnan(cpu_results['mae']), "Should handle CPU tensors"
    
    # Test GPU tensors (if available)
    if torch.cuda.is_available():
        gpu_pred: torch.Tensor = cpu_pred.cuda()
        gpu_target: torch.Tensor = cpu_target.cuda()
        
        gpu_results: Dict[str, float] = evaluator(gpu_pred, gpu_target)
        
        assert not np.isnan(gpu_results['rmse']), "Should handle GPU tensors"
        assert not np.isnan(gpu_results['mae']), "Should handle GPU tensors"


def test_comprehensive_validation() -> None:
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test comprehensive workflow
    pred: torch.Tensor = torch.rand(128, 128, device=device) * 5 + 1.0
    target: torch.Tensor = torch.rand(128, 128, device=device) * 5 + 1.0
    
    # Step 1: Validate inputs
    validation_info: Dict[str, Any] = validate_tensor_inputs(pred, target)
    assert validation_info['has_mask'] == False, "Should detect no mask"
    
    # Step 2: Create and validate mask
    mask: torch.Tensor = create_default_mask(target)
    masked_pred: torch.Tensor
    count: int
    masked_pred, count = apply_mask_safely(pred, mask)
    assert count > 0, "Should have valid pixels"
    
    # Step 3: Compute all metrics
    all_results: Dict[str, Union[float, Dict]] = compute_all_metrics(pred, target, mask)
    
    # Step 4: Validate metric sanity
    is_valid: bool
    warnings_list: List[str]
    is_valid, warnings_list = validate_metric_sanity(all_results)  # type: ignore
    
    # Step 5: Create comprehensive report
    report: Dict = create_metric_report(pred, target, mask, "ComprehensiveTest")
    
    # Verify complete pipeline
    assert 'metrics' in report, "Report should contain metrics"
    assert 'sanity_check' in report, "Report should contain sanity check"
    assert report['valid_pixel_count'] > 0, "Should have valid pixels"

def run_all_tests() -> bool:
    test_functions: List[Callable[[], None]] = [
        # Core utility tests
        test_create_default_mask,
        test_apply_mask_safely,
        test_validate_tensor_inputs,
        test_resize_tensors_to_scale,
        test_validate_depth_image_compatibility,
        test_validate_depth_values,
        test_validate_numerical_stability,
        
        # Metric implementation tests
        test_rmse,
        test_mae,
        test_delta_metrics,
        test_silog,
        test_compute_all_metrics,
        test_compute_batch_metrics,
        test_validate_metric_sanity,
        test_create_metric_report,
        test_compute_bootstrap_ci,
        
        # Factory function tests
        test_get_metric,
        test_get_all_metrics,
        test_get_core_metrics,
        test_list_metrics,
        test_create_evaluator,
        
        # Integration and edge case tests
        test_integration_pipeline,
        test_numerical_edge_cases,
        test_memory_efficiency,
        test_device_compatibility,
        test_comprehensive_validation,
    ]
    
    passed: int = 0
    failed: int = 0
    
    print("Running synchronized functional test suite...")
    print(f"Total tests to execute: {len(test_functions)}")
    print("-" * 60)
    
    for test_func in test_functions:
        try:
            test_func()
            print(f"{test_func.__name__}")
            passed += 1
        except Exception as e:
            print(f"{test_func.__name__}: {str(e)}")
            failed += 1
    
    print("-" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("All tests passed! The module is ready for production.")
    else:
        print(f"{failed} test(s) failed. Please review the implementation.")
    
    return failed == 0
