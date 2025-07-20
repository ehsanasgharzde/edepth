# FILE: tests/test_metrics.py
# ehsanasgharzde - COMPLETE METRICS TEST SUITE
# hosseinsolymanzadeh - PROPER COMMENTING
#hosseinsolymanzadeh - FIXED REDUNDANT CODE BY EXTRACTING PURE FUNCTIONS AND BASECLASS LEVEL METHODS

import torch
import numpy as np
import pytest
import sys

from metrics.metrics import (
    rmse, mae, delta1, delta2, delta3, silog, 
    compute_bootstrap_ci, validate_metric_sanity, create_metric_report
)
from metrics.factory import get_metric, get_all_metrics, create_evaluator
from utils.core import create_default_mask, apply_mask_safely, validate_tensor_inputs


class TestMetricFunctions:
    
    def setup_method(self) -> None:
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create synthetic test data with known properties
        self.pred_perfect = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=self.device)
        self.target_perfect = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=self.device)
        
        # Test data with known error patterns
        self.pred_offset = torch.tensor([1.1, 2.1, 3.1, 4.1, 5.1], device=self.device)
        self.target_base = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=self.device)
        
        # 2D test data for spatial metrics
        self.pred_2d = torch.rand(32, 32, device=self.device) * 10 + 0.1
        self.target_2d = torch.rand(32, 32, device=self.device) * 10 + 0.1
        
        # Batch test data
        self.pred_batch = torch.rand(4, 32, 32, device=self.device) * 10 + 0.1
        self.target_batch = torch.rand(4, 32, 32, device=self.device) * 10 + 0.1
    
    def test_input_validation(self) -> None: 
        # Test non-tensor inputs
        with pytest.raises(TypeError):
            validate_tensor_inputs([1, 2, 3], self.target_perfect) 
        
        with pytest.raises(TypeError):
            validate_tensor_inputs(self.pred_perfect, [1, 2, 3])  
        
        # Test shape mismatch
        pred_wrong_shape = torch.tensor([1.0, 2.0], device=self.device)
        with pytest.raises(ValueError):
            validate_tensor_inputs(pred_wrong_shape, self.target_perfect)
        
        # Test device mismatch
        pred_cpu = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        target_gpu = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=self.device)
        if self.device.type == 'cuda':
            with pytest.raises(ValueError):
                validate_tensor_inputs(pred_cpu, target_gpu)
        
        # Test invalid mask shape
        invalid_mask = torch.ones(3, device=self.device)
        with pytest.raises(ValueError):
            validate_tensor_inputs(self.pred_perfect, self.target_perfect, invalid_mask)
    
    def test_perfect_predictions(self) -> None:
        # Perfect predictions should yield specific metric values
        rmse_val = rmse(self.pred_perfect, self.target_perfect)
        mae_val = mae(self.pred_perfect, self.target_perfect)
        delta1_val = delta1(self.pred_perfect, self.target_perfect)
        delta2_val = delta2(self.pred_perfect, self.target_perfect)
        delta3_val = delta3(self.pred_perfect, self.target_perfect)
        silog_val = silog(self.pred_perfect, self.target_perfect)
        
        # Perfect predictions should have zero error
        assert abs(rmse_val) < 1e-6, f"RMSE should be ~0 for perfect predictions, got {rmse_val}"
        assert abs(mae_val) < 1e-6, f"MAE should be ~0 for perfect predictions, got {mae_val}"
        assert abs(silog_val) < 1e-6, f"SiLog should be ~0 for perfect predictions, got {silog_val}"
        
        # Perfect predictions should have 100% delta accuracy
        assert abs(delta1_val - 1.0) < 1e-6, f"Delta1 should be 1.0 for perfect predictions, got {delta1_val}"
        assert abs(delta2_val - 1.0) < 1e-6, f"Delta2 should be 1.0 for perfect predictions, got {delta2_val}"
        assert abs(delta3_val - 1.0) < 1e-6, f"Delta3 should be 1.0 for perfect predictions, got {delta3_val}"
    
    def test_known_error_patterns(self) -> None:
        # Test with constant offset (should give known MAE)
        mae_val = mae(self.pred_offset, self.target_base)
        expected_mae = 0.1  # All predictions are off by 0.1
        assert abs(mae_val - expected_mae) < 1e-6, f"MAE should be {expected_mae}, got {mae_val}"
        
        # Test RMSE with known pattern
        rmse_val = rmse(self.pred_offset, self.target_base)
        expected_rmse = 0.1  # Constant offset gives RMSE equal to offset
        assert abs(rmse_val - expected_rmse) < 1e-6, f"RMSE should be {expected_rmse}, got {rmse_val}"
        
        # Test delta metrics with known ratios
        # For offset of 0.1 on values 1-5, max ratio should be 5.1/5 = 1.02 < 1.25
        delta1_val = delta1(self.pred_offset, self.target_base)
        assert abs(delta1_val - 1.0) < 1e-6, f"Delta1 should be 1.0 for small offset, got {delta1_val}"
    
    def test_edge_cases(self) -> None: 
        # Test with very small values near epsilon
        small_pred = torch.tensor([1e-8, 1e-7, 1e-6], device=self.device)
        small_target = torch.tensor([1e-8, 1e-7, 1e-6], device=self.device)
        
        # Should not crash and should handle small values gracefully
        rmse_val = rmse(small_pred, small_target)
        assert not np.isnan(rmse_val), "RMSE should not be NaN for small values"
        
        # Test with large values
        large_pred = torch.tensor([1e3, 1e4, 1e5], device=self.device)
        large_target = torch.tensor([1e3, 1e4, 1e5], device=self.device)
        
        rmse_val = rmse(large_pred, large_target)
        assert not np.isnan(rmse_val), "RMSE should not be NaN for large values"
        
        # Test with single value
        single_pred = torch.tensor([5.0], device=self.device)
        single_target = torch.tensor([4.0], device=self.device)
        
        mae_val = mae(single_pred, single_target)
        assert abs(mae_val - 1.0) < 1e-6, f"MAE should be 1.0 for single value test, got {mae_val}"
    
    def test_masking_functionality(self) -> None:
        # Create test data with some invalid values
        pred_with_invalid = torch.tensor([1.0, 2.0, 0.0, 4.0, -1.0], device=self.device)
        target_with_invalid = torch.tensor([1.0, 2.0, 0.0, 4.0, -1.0], device=self.device)
        
        # Create mask to exclude invalid values
        valid_mask = torch.tensor([True, True, False, True, False], device=self.device)
        
        # Test with explicit mask
        rmse_val = rmse(pred_with_invalid, target_with_invalid, mask=valid_mask)
        assert not np.isnan(rmse_val), "RMSE should not be NaN with valid mask"
        
        # Test automatic mask creation
        pred_auto_mask = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=self.device)
        target_auto_mask = torch.tensor([1.0, 2.0, 0.0, 4.0, float('inf')], device=self.device)
        
        # Should automatically mask out invalid values
        rmse_val = rmse(pred_auto_mask, target_auto_mask)
        assert not np.isnan(rmse_val), "RMSE should handle automatic masking"
    
    def test_bootstrap_confidence_intervals(self) -> None:
        # Test with sufficient data points
        values = np.random.normal(5.0, 1.0, 1000)
        
        ci_lower, ci_upper = compute_bootstrap_ci(values)
        assert not np.isnan(ci_lower), "CI lower should not be NaN"
        assert not np.isnan(ci_upper), "CI upper should not be NaN"
        assert ci_lower < ci_upper, "CI lower should be less than upper"
        
        # Test with insufficient data
        small_values = np.array([1.0])
        ci_lower, ci_upper = compute_bootstrap_ci(small_values)
        assert np.isnan(ci_lower), "CI lower should be NaN for insufficient data"
        assert np.isnan(ci_upper), "CI upper should be NaN for insufficient data"
    
    def test_metric_sanity_validation(self) -> None:
        # Test with valid metrics
        valid_metrics = {
            'rmse': 1.5,
            'mae': 1.0,
            'delta1': 0.8,
            'delta2': 0.9,
            'delta3': 0.95,
            'silog': 0.2
        }
        
        is_valid, warnings_list = validate_metric_sanity(valid_metrics)
        assert is_valid, "Valid metrics should pass sanity check"
        assert len(warnings_list) == 0, "Should not have warnings for valid metrics"
        
        # Test with invalid metrics (RMSE < MAE)
        invalid_metrics = {
            'rmse': 0.5,
            'mae': 1.0,
            'delta1': 0.8,
            'silog': 0.2
        }
        
        is_valid, warnings_list = validate_metric_sanity(invalid_metrics)
        assert not is_valid, "Invalid metrics should fail sanity check"
        assert len(warnings_list) > 0, "Should have warnings for invalid metrics"
        
        # Test with out-of-range delta values
        invalid_delta_metrics = {
            'rmse': 1.5,
            'mae': 1.0,
            'delta1': 1.5,  # Invalid: > 1.0
            'silog': 0.2
        }
        
        is_valid, warnings_list = validate_metric_sanity(invalid_delta_metrics)
        assert not is_valid, "Out-of-range delta should fail sanity check"
        assert any('delta1' in warning for warning in warnings_list), "Should warn about delta1"
    
    def test_comprehensive_metric_report(self) -> None:
        report = create_metric_report(
            self.pred_2d, self.target_2d, dataset_name="TestDataset"
        )
        
        # Check report structure
        assert 'dataset' in report, "Report should contain dataset name"
        assert 'valid_pixel_count' in report, "Report should contain valid pixel count"
        assert 'depth_statistics' in report, "Report should contain depth statistics"
        assert 'metrics' in report, "Report should contain metrics"
        assert 'sanity_check' in report, "Report should contain sanity check result"
        assert 'warnings' in report, "Report should contain warnings"
        
        # Check depth statistics
        depth_stats = report['depth_statistics']
        assert 'min' in depth_stats, "Should contain min depth"
        assert 'max' in depth_stats, "Should contain max depth"
        assert 'mean' in depth_stats, "Should contain mean depth"
        assert 'std' in depth_stats, "Should contain std depth"
        
        # Check metrics are computed
        metrics = report['metrics']
        expected_metrics = ['rmse', 'mae', 'delta1', 'delta2', 'delta3', 'silog']
        for metric in expected_metrics:
            assert metric in metrics, f"Metric {metric} should be in report"
    
    def test_numerical_stability(self) -> None:
        # Test with very small values
        tiny_pred = torch.full((100,), 1e-10, device=self.device)
        tiny_target = torch.full((100,), 1e-10, device=self.device)
        
        # Should not produce NaN or infinite values
        rmse_val = rmse(tiny_pred, tiny_target)
        mae_val = mae(tiny_pred, tiny_target)
        silog_val = silog(tiny_pred, tiny_target)
        
        assert not np.isnan(rmse_val), "RMSE should not be NaN for tiny values"
        assert not np.isnan(mae_val), "MAE should not be NaN for tiny values"
        assert not np.isnan(silog_val), "SiLog should not be NaN for tiny values"
        
        # Test with large values
        huge_pred = torch.full((100,), 1e8, device=self.device)
        huge_target = torch.full((100,), 1e8, device=self.device)
        
        rmse_val = rmse(huge_pred, huge_target)
        mae_val = mae(huge_pred, huge_target)
        
        assert not np.isnan(rmse_val), "RMSE should not be NaN for huge values"
        assert not np.isnan(mae_val), "MAE should not be NaN for huge values"
    
    def test_memory_efficiency(self) -> None:
        # Create large tensors to test memory handling
        if torch.cuda.is_available():
            # Only test on GPU if available, as CPU memory is more limited
            large_pred = torch.rand(2048, 2048, device=self.device) * 10 + 0.1
            large_target = torch.rand(2048, 2048, device=self.device) * 10 + 0.1
            
            # Should not run out of memory
            try:
                rmse_val = rmse(large_pred, large_target)
                mae_val = mae(large_pred, large_target)
                assert not np.isnan(rmse_val), "RMSE should be computed for large tensors"
                assert not np.isnan(mae_val), "MAE should be computed for large tensors"
            except RuntimeError as e:
                if "out of memory" in str(e):
                    pytest.skip("Skipping large tensor test due to memory constraints")
                else:
                    raise
    
    def test_device_consistency(self) -> None:
        # Test CPU tensors
        cpu_pred = torch.rand(100) * 10 + 0.1
        cpu_target = torch.rand(100) * 10 + 0.1
        
        rmse_val = rmse(cpu_pred, cpu_target)
        assert not np.isnan(rmse_val), "Should handle CPU tensors"
        
        # Test GPU tensors (if available)
        if torch.cuda.is_available():
            gpu_pred = cpu_pred.cuda()
            gpu_target = cpu_target.cuda()
            
            rmse_val = rmse(gpu_pred, gpu_target)
            assert not np.isnan(rmse_val), "Should handle GPU tensors"


class TestMetricFactory:
    
    def setup_method(self) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pred = torch.rand(100, device=self.device) * 10 + 0.1
        self.target = torch.rand(100, device=self.device) * 10 + 0.1
    
    def test_get_metric(self) -> None:
        # Test valid metric names
        rmse_func = get_metric('rmse')
        mae_func = get_metric('mae')
        delta1_func = get_metric('delta1')
        
        # Test that functions work
        rmse_val = rmse_func(self.pred, self.target)
        mae_val = mae_func(self.pred, self.target)
        delta1_val = delta1_func(self.pred, self.target)
        
        assert not np.isnan(rmse_val), "RMSE function should work"
        assert not np.isnan(mae_val), "MAE function should work"
        assert not np.isnan(delta1_val), "Delta1 function should work"
        
        # Test invalid metric name
        with pytest.raises(ValueError):
            get_metric('invalid_metric')
    
    def test_get_all_metrics(self) -> None:
        all_metrics = get_all_metrics()
        
        expected_metrics = ['rmse', 'mae', 'delta1', 'delta2', 'delta3', 'silog']
        for metric_name in expected_metrics:
            assert metric_name in all_metrics, f"Should contain {metric_name}"
            
            # Test that function works
            metric_func = all_metrics[metric_name]
            result = metric_func(self.pred, self.target)
            assert not np.isnan(result), f"{metric_name} should produce valid result"
    
    def test_create_evaluator(self) -> None:
        # Test default evaluator (all metrics)
        evaluator = create_evaluator()
        results = evaluator(self.pred, self.target)
        
        expected_metrics = ['rmse', 'mae', 'delta1', 'delta2', 'delta3', 'silog']
        for metric_name in expected_metrics:
            assert metric_name in results, f"Should contain {metric_name}"
            assert not np.isnan(results[metric_name]), f"{metric_name} should be valid"
        
        # Test custom evaluator
        custom_metrics = ['rmse', 'mae']
        custom_evaluator = create_evaluator(custom_metrics)
        custom_results = custom_evaluator(self.pred, self.target)
        
        assert len(custom_results) == 2, "Should only contain requested metrics"
        assert 'rmse' in custom_results, "Should contain RMSE"
        assert 'mae' in custom_results, "Should contain MAE"
        
        # Test invalid metric in evaluator
        with pytest.raises(ValueError):
            create_evaluator(['invalid_metric'])


class TestCoreUtilities:
    
    def setup_method(self) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_create_default_mask(self) -> None:
        # Test with valid values
        valid_target = torch.tensor([1.0, 2.0, 3.0], device=self.device)
        mask = create_default_mask(valid_target)
        
        expected_mask = torch.tensor([True, True, True], device=self.device)
        assert torch.equal(mask, expected_mask), "Should create correct mask for valid values"
        
        # Test with invalid values
        invalid_target = torch.tensor([0.0, -1.0, float('inf')], device=self.device)
        mask = create_default_mask(invalid_target)
        
        expected_mask = torch.tensor([False, False, False], device=self.device)
        assert torch.equal(mask, expected_mask), "Should create correct mask for invalid values"
    
    def test_apply_mask_safely(self) -> None: 
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], device=self.device)
        mask = torch.tensor([True, False, True, True], device=self.device)
        
        masked_tensor, count = apply_mask_safely(tensor, mask)
        
        expected_tensor = torch.tensor([1.0, 3.0, 4.0], device=self.device)
        assert torch.equal(masked_tensor, expected_tensor), "Should apply mask correctly"
        assert count == 3, "Should count valid elements correctly"
        
        # Test with empty mask
        empty_mask = torch.tensor([False, False, False, False], device=self.device)
        masked_tensor, count = apply_mask_safely(tensor, empty_mask)
        
        assert count == 0, "Should handle empty mask"
        assert masked_tensor.numel() == 0, "Should return empty tensor for empty mask"
    
    def test_validate_tensor_inputs(self) -> None: 
        pred = torch.rand(10, 10, device=self.device)
        target = torch.rand(10, 10, device=self.device)
        mask = torch.ones(10, 10, dtype=torch.bool, device=self.device)
        
        # Test valid inputs
        info = validate_tensor_inputs(pred, target, mask)
        
        assert info['shape'] == pred.shape, "Should return correct shape"
        assert info['device'] == pred.device, "Should return correct device"
        assert info['has_mask'] == True, "Should detect mask presence"
        
        # Test without mask
        info_no_mask = validate_tensor_inputs(pred, target)
        assert info_no_mask['has_mask'] == False, "Should detect no mask"


class TestIntegrationScenarios:
    
    def setup_method(self) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_evaluation_pipeline(self) -> None: 
        # Simulate model predictions and ground truth
        pred = torch.rand(256, 256, device=self.device) * 10 + 0.1
        target = torch.rand(256, 256, device=self.device) * 10 + 0.1
        
        # Create evaluator with all metrics
        evaluator = create_evaluator()
        results = evaluator(pred, target)
        
        # Validate results
        is_valid, warnings = validate_metric_sanity(results)
        
        # Create comprehensive report
        report = create_metric_report(pred, target, dataset_name="Integration_Test")
        
        # Verify pipeline works end-to-end
        assert 'metrics' in report, "Report should contain metrics"
        assert 'sanity_check' in report, "Report should contain sanity check"
        assert len(results) == 6, "Should compute all metrics"
    
    def test_batch_evaluation(self) -> None:
        batch_size = 8
        height, width = 128, 128
        
        # Simulate batch of predictions and targets
        pred_batch = torch.rand(batch_size, height, width, device=self.device) * 10 + 0.1
        target_batch = torch.rand(batch_size, height, width, device=self.device) * 10 + 0.1
        
        # Evaluate each sample in batch
        evaluator = create_evaluator(['rmse', 'mae', 'delta1'])
        batch_results = []
        
        for i in range(batch_size):
            results = evaluator(pred_batch[i], target_batch[i])
            batch_results.append(results)
        
        # Compute batch statistics
        rmse_values = [r['rmse'] for r in batch_results]
        mae_values = [r['mae'] for r in batch_results]
        
        assert len(batch_results) == batch_size, "Should process all samples"
        assert all(not np.isnan(v) for v in rmse_values), "All RMSE values should be valid"
        assert all(not np.isnan(v) for v in mae_values), "All MAE values should be valid"
    
    def test_model_comparison(self) -> None: 
        # Create base target
        target = torch.rand(128, 128, device=self.device) * 5 + 2
        
        # Simulate different model performances
        models = {
            'perfect': target.clone(),
            'good': target + torch.randn_like(target) * 0.1,
            'poor': target + torch.randn_like(target) * 0.5
        }
        
        # Evaluate all models
        evaluator = create_evaluator(['rmse', 'mae'])
        model_results = {}
        
        for model_name, pred in models.items():
            results = evaluator(pred, target)
            model_results[model_name] = results
        
        # Verify ranking
        assert model_results['perfect']['rmse'] < model_results['good']['rmse']
        assert model_results['good']['rmse'] < model_results['poor']['rmse']
        
        assert model_results['perfect']['mae'] < model_results['good']['mae']
        assert model_results['good']['mae'] < model_results['poor']['mae']


def run_comprehensive_tests() -> bool:
    print("Running comprehensive depth metrics test suite...")
    
    # Initialize test classes
    metric_tests = TestMetricFunctions()
    factory_tests = TestMetricFactory()
    core_tests = TestCoreUtilities()
    integration_tests = TestIntegrationScenarios()
    
    test_methods = [
        (metric_tests, 'test_input_validation'),
        (metric_tests, 'test_perfect_predictions'),
        (metric_tests, 'test_known_error_patterns'),
        (metric_tests, 'test_edge_cases'),
        (metric_tests, 'test_masking_functionality'),
        (metric_tests, 'test_bootstrap_confidence_intervals'),
        (metric_tests, 'test_metric_sanity_validation'),
        (metric_tests, 'test_comprehensive_metric_report'),
        (metric_tests, 'test_numerical_stability'),
        (metric_tests, 'test_memory_efficiency'),
        (metric_tests, 'test_device_consistency'),
        (factory_tests, 'test_get_metric'),
        (factory_tests, 'test_get_all_metrics'),
        (factory_tests, 'test_create_evaluator'),
        (core_tests, 'test_create_default_mask'),
        (core_tests, 'test_apply_mask_safely'),
        (core_tests, 'test_validate_tensor_inputs'),
        (integration_tests, 'test_evaluation_pipeline'),
        (integration_tests, 'test_batch_evaluation'),
        (integration_tests, 'test_model_comparison'),
    ]
    
    passed = 0
    failed = 0
    
    for test_instance, test_method in test_methods:
        try:
            # Setup the test
            test_instance.setup_method()
            
            # Run the test
            getattr(test_instance, test_method)()
            
            print(f"{test_method}")
            passed += 1
        except Exception as e:
            print(f"{test_method}: {str(e)}")
            failed += 1
    
    print(f"\nTest Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("All tests passed!")
    else:
        print(f"Some tests failed. Please check the implementation.")
    
    return failed == 0