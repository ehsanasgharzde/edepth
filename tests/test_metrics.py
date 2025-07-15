# FILE: tests/test_metrics.py
# ehsanasgharzde - COMPLETE METRICS TEST SUITE

import torch
import numpy as np
import pytest
import sys
from metrics.metrics_fixed import Metrics, rmse, mae, delta1, delta2, delta3, silog, compute_all_metrics


class TestMetrics:
    """Test class for DepthMetrics functionality."""
    
    def setup_method(self):
        """Setup test fixtures before each test method."""
        self.metrics = Metrics()
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
    
    def test_initialization(self):
        """Test DepthMetrics initialization with different parameters."""
        # Test default initialization
        default_metrics = Metrics()
        assert default_metrics.min_depth == 0.001
        assert default_metrics.max_depth == 80.0
        assert default_metrics.eps == 1e-6
        
        # Test custom initialization
        custom_metrics = Metrics(min_depth=0.1, max_depth=10.0, eps=1e-7)
        assert custom_metrics.min_depth == 0.1
        assert custom_metrics.max_depth == 10.0
        assert custom_metrics.eps == 1e-7
    
    def test_input_validation(self):
        """Test input validation for various error conditions."""
        # Test non-tensor inputs
        with pytest.raises(TypeError):
            self.metrics._validate_inputs([1, 2, 3], self.target_perfect) #type: ignore
        
        with pytest.raises(TypeError):
            self.metrics._validate_inputs(self.pred_perfect, [1, 2, 3]) #type: ignore
        
        # Test shape mismatch
        pred_wrong_shape = torch.tensor([1.0, 2.0], device=self.device)
        with pytest.raises(ValueError):
            self.metrics._validate_inputs(pred_wrong_shape, self.target_perfect)
        
        # Test device mismatch
        pred_cpu = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        target_gpu = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=self.device)
        if self.device.type == 'cuda':
            with pytest.raises(ValueError):
                self.metrics._validate_inputs(pred_cpu, target_gpu)
        
        # Test invalid mask shape
        invalid_mask = torch.ones(3, device=self.device)
        with pytest.raises(ValueError):
            self.metrics._validate_inputs(self.pred_perfect, self.target_perfect, invalid_mask)
    
    def test_perfect_predictions(self):
        """Test metrics with perfect predictions (should yield optimal values)."""
        # Perfect predictions should yield specific metric values
        rmse_val = self.metrics.rmse(self.pred_perfect, self.target_perfect)
        mae_val = self.metrics.mae(self.pred_perfect, self.target_perfect)
        delta1_val = self.metrics.delta1(self.pred_perfect, self.target_perfect)
        delta2_val = self.metrics.delta2(self.pred_perfect, self.target_perfect)
        delta3_val = self.metrics.delta3(self.pred_perfect, self.target_perfect)
        silog_val = self.metrics.silog(self.pred_perfect, self.target_perfect)
        
        # Perfect predictions should have zero error
        assert abs(rmse_val) < 1e-6, f"RMSE should be ~0 for perfect predictions, got {rmse_val}" #type: ignore
        assert abs(mae_val) < 1e-6, f"MAE should be ~0 for perfect predictions, got {mae_val}" #type: ignore
        assert abs(silog_val) < 1e-6, f"SiLog should be ~0 for perfect predictions, got {silog_val}" #type: ignore
        
        # Perfect predictions should have 100% delta accuracy
        assert abs(delta1_val - 1.0) < 1e-6, f"Delta1 should be 1.0 for perfect predictions, got {delta1_val}" #type: ignore
        assert abs(delta2_val - 1.0) < 1e-6, f"Delta2 should be 1.0 for perfect predictions, got {delta2_val}" #type: ignore
        assert abs(delta3_val - 1.0) < 1e-6, f"Delta3 should be 1.0 for perfect predictions, got {delta3_val}" #type: ignore
    
    def test_known_error_patterns(self):
        """Test metrics with known error patterns to validate correctness."""
        # Test with constant offset (should give known MAE)
        mae_val = self.metrics.mae(self.pred_offset, self.target_base)
        expected_mae = 0.1  # All predictions are off by 0.1
        assert abs(mae_val - expected_mae) < 1e-6, f"MAE should be {expected_mae}, got {mae_val}" #type: ignore
        
        # Test RMSE with known pattern
        rmse_val = self.metrics.rmse(self.pred_offset, self.target_base)
        expected_rmse = 0.1  # Constant offset gives RMSE equal to offset
        assert abs(rmse_val - expected_rmse) < 1e-6, f"RMSE should be {expected_rmse}, got {rmse_val}" #type: ignore
        
        # Test delta metrics with known ratios
        # For offset of 0.1 on values 1-5, max ratio should be 5.1/5 = 1.02 < 1.25
        delta1_val = self.metrics.delta1(self.pred_offset, self.target_base)
        assert abs(delta1_val - 1.0) < 1e-6, f"Delta1 should be 1.0 for small offset, got {delta1_val}" #type: ignore
    
    def test_edge_cases(self):
        """Test handling of edge cases and boundary conditions."""
        # Test with very small values near epsilon
        small_pred = torch.tensor([1e-8, 1e-7, 1e-6], device=self.device)
        small_target = torch.tensor([1e-8, 1e-7, 1e-6], device=self.device)
        
        # Should not crash and should handle small values gracefully
        rmse_val = self.metrics.rmse(small_pred, small_target)
        assert not np.isnan(rmse_val), "RMSE should not be NaN for small values"
        
        # Test with large values
        large_pred = torch.tensor([1e3, 1e4, 1e5], device=self.device)
        large_target = torch.tensor([1e3, 1e4, 1e5], device=self.device)
        
        rmse_val = self.metrics.rmse(large_pred, large_target)
        assert not np.isnan(rmse_val), "RMSE should not be NaN for large values"
        
        # Test with single value
        single_pred = torch.tensor([5.0], device=self.device)
        single_target = torch.tensor([4.0], device=self.device)
        
        mae_val = self.metrics.mae(single_pred, single_target)
        assert abs(mae_val - 1.0) < 1e-6, f"MAE should be 1.0 for single value test, got {mae_val}" #type: ignore
    
    def test_masking_functionality(self):
        """Test proper mask handling and application."""
        # Create test data with some invalid values
        pred_with_invalid = torch.tensor([1.0, 2.0, 0.0, 4.0, -1.0], device=self.device)
        target_with_invalid = torch.tensor([1.0, 2.0, 0.0, 4.0, -1.0], device=self.device)
        
        # Create mask to exclude invalid values
        valid_mask = torch.tensor([True, True, False, True, False], device=self.device)
        
        # Test with explicit mask
        rmse_val = self.metrics.rmse(pred_with_invalid, target_with_invalid, mask=valid_mask)
        assert not np.isnan(rmse_val), "RMSE should not be NaN with valid mask"
        
        # Test automatic mask creation
        pred_auto_mask = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=self.device)
        target_auto_mask = torch.tensor([1.0, 2.0, 0.0, 4.0, float('inf')], device=self.device)
        
        # Should automatically mask out invalid values
        rmse_val = self.metrics.rmse(pred_auto_mask, target_auto_mask)
        assert not np.isnan(rmse_val), "RMSE should handle automatic masking"
    
    def test_return_count_functionality(self):
        """Test return_count parameter functionality."""
        # Test with return_count=True
        rmse_val, count = self.metrics.rmse(self.pred_perfect, self.target_perfect, return_count=True) #type: ignore
        assert count == len(self.pred_perfect), f"Count should be {len(self.pred_perfect)}, got {count}" #type: ignore
        
        # Test with masked data
        mask = torch.tensor([True, True, False, True, True], device=self.device)
        rmse_val, count = self.metrics.rmse(self.pred_perfect, self.target_perfect, mask=mask, return_count=True) #type: ignore
        expected_count = mask.sum().item()
        assert count == expected_count, f"Count should be {expected_count}, got {count}" #type: ignore
    
    def test_confidence_intervals(self):
        """Test confidence interval computation."""
        # Test with sufficient data points
        large_pred = torch.randn(1000, device=self.device) * 2 + 5
        large_target = torch.randn(1000, device=self.device) * 2 + 5
        
        # Test RMSE with confidence intervals
        rmse_val, ci_info = self.metrics.rmse(large_pred, large_target, with_confidence=True) #type: ignore
        assert isinstance(ci_info, dict), "Confidence info should be a dictionary"
        assert 'ci_lower' in ci_info, "Should contain ci_lower"
        assert 'ci_upper' in ci_info, "Should contain ci_upper"
        assert ci_info['ci_lower'] <= rmse_val <= ci_info['ci_upper'], "RMSE should be within confidence interval" #type: ignore
        
        # Test MAE with confidence intervals
        mae_val, ci_info = self.metrics.mae(large_pred, large_target, with_confidence=True) #type: ignore   
        assert isinstance(ci_info, dict), "Confidence info should be a dictionary"
        assert 'ci_lower' in ci_info, "Should contain ci_lower"
        assert 'ci_upper' in ci_info, "Should contain ci_upper"
    
    def test_batch_processing(self):
        """Test batch metric computation functionality."""
        # Test with default metrics
        batch_results = self.metrics.compute_batch_metrics(self.pred_batch, self.target_batch) #type: ignore
        
        # Check that all default metrics are computed
        expected_metrics = ['rmse', 'mae', 'delta1', 'delta2', 'delta3', 'silog']
        for metric in expected_metrics:
            assert metric in batch_results, f"Metric {metric} should be in batch results"
            assert 'mean' in batch_results[metric], f"Should contain mean for {metric}"
            assert 'std' in batch_results[metric], f"Should contain std for {metric}"
            assert 'min' in batch_results[metric], f"Should contain min for {metric}"
            assert 'max' in batch_results[metric], f"Should contain max for {metric}"
            assert 'per_sample' in batch_results[metric], f"Should contain per_sample for {metric}"
            assert len(batch_results[metric]['per_sample']) == self.pred_batch.shape[0], \
                f"Should have results for all batch samples"
        
        # Test with custom metrics list
        custom_metrics = ['rmse', 'mae']
        batch_results = self.metrics.compute_batch_metrics(
            self.pred_batch, self.target_batch, metrics_list=custom_metrics
        ) #type: ignore
        
        assert len(batch_results) == len(custom_metrics), "Should only compute requested metrics"
        for metric in custom_metrics:
            assert metric in batch_results, f"Metric {metric} should be in results"
    
    def test_metric_sanity_validation(self):
        """Test metric sanity checking functionality."""
        # Test with valid metrics
        valid_metrics = {
            'rmse': 1.5,
            'mae': 1.0,
            'delta1': 0.8,
            'delta2': 0.9,
            'delta3': 0.95,
            'silog': 0.2
        }
        
        is_valid, warnings_list = self.metrics.validate_metric_sanity(valid_metrics) #type: ignore
        assert is_valid, "Valid metrics should pass sanity check"
        assert len(warnings_list) == 0, "Should not have warnings for valid metrics"
        
        # Test with invalid metrics (RMSE < MAE)
        invalid_metrics = {
            'rmse': 0.5,
            'mae': 1.0,
            'delta1': 0.8,
            'silog': 0.2
        }
        
        is_valid, warnings_list = self.metrics.validate_metric_sanity(invalid_metrics) #type: ignore
        assert not is_valid, "Invalid metrics should fail sanity check"
        assert len(warnings_list) > 0, "Should have warnings for invalid metrics"
        
        # Test with out-of-range delta values
        invalid_delta_metrics = {
            'rmse': 1.5,
            'mae': 1.0,
            'delta1': 1.5,  # Invalid: > 1.0
            'silog': 0.2
        }
        
        is_valid, warnings_list = self.metrics.validate_metric_sanity(invalid_delta_metrics) #type: ignore
        assert not is_valid, "Out-of-range delta should fail sanity check"
        assert any('delta1' in warning for warning in warnings_list), "Should warn about delta1"
    
    def test_comprehensive_metric_report(self):
        """Test comprehensive metric report generation."""
        report = self.metrics.create_metric_report(
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
    
    def test_standalone_functions(self):
        """Test standalone function interfaces for backward compatibility."""
        # Test standalone functions
        rmse_val = rmse(self.pred_perfect, self.target_perfect)
        mae_val = mae(self.pred_perfect, self.target_perfect)
        delta1_val = delta1(self.pred_perfect, self.target_perfect)
        delta2_val = delta2(self.pred_perfect, self.target_perfect)
        delta3_val = delta3(self.pred_perfect, self.target_perfect)
        silog_val = silog(self.pred_perfect, self.target_perfect)
        
        # Test compute_all_metrics function
        all_metrics = compute_all_metrics(self.pred_perfect, self.target_perfect)
        
        # Check that standalone functions match class methods
        class_rmse = self.metrics.rmse(self.pred_perfect, self.target_perfect)
        class_mae = self.metrics.mae(self.pred_perfect, self.target_perfect)
        
        assert abs(rmse_val - class_rmse) < 1e-6, "Standalone RMSE should match class method" #type: ignore
        assert abs(mae_val - class_mae) < 1e-6, "Standalone MAE should match class method" #type: ignore
        
        # Check all_metrics dictionary structure
        expected_keys = ['rmse', 'mae', 'delta1', 'delta2', 'delta3', 'silog']
        for key in expected_keys:
            assert key in all_metrics, f"Key {key} should be in all_metrics"
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Test with very small values
        tiny_pred = torch.full((100,), 1e-10, device=self.device)
        tiny_target = torch.full((100,), 1e-10, device=self.device)
        
        # Should not produce NaN or infinite values
        rmse_val = self.metrics.rmse(tiny_pred, tiny_target)
        mae_val = self.metrics.mae(tiny_pred, tiny_target)
        silog_val = self.metrics.silog(tiny_pred, tiny_target)
        
        assert not np.isnan(rmse_val), "RMSE should not be NaN for tiny values"
        assert not np.isnan(mae_val), "MAE should not be NaN for tiny values"
        assert not np.isnan(silog_val), "SiLog should not be NaN for tiny values"
        
        # Test with large values
        huge_pred = torch.full((100,), 1e8, device=self.device)
        huge_target = torch.full((100,), 1e8, device=self.device)
        
        rmse_val = self.metrics.rmse(huge_pred, huge_target)
        mae_val = self.metrics.mae(huge_pred, huge_target)
        
        assert not np.isnan(rmse_val), "RMSE should not be NaN for huge values"
        assert not np.isnan(mae_val), "MAE should not be NaN for huge values"
    
    def test_memory_efficiency(self):
        """Test memory efficiency with large tensors."""
        # Create large tensors to test memory handling
        if torch.cuda.is_available():
            # Only test on GPU if available, as CPU memory is more limited
            large_pred = torch.rand(2048, 2048, device=self.device) * 10 + 0.1
            large_target = torch.rand(2048, 2048, device=self.device) * 10 + 0.1
            
            # Should not run out of memory
            try:
                rmse_val = self.metrics.rmse(large_pred, large_target)
                mae_val = self.metrics.mae(large_pred, large_target)
                assert not np.isnan(rmse_val), "RMSE should be computed for large tensors"
                assert not np.isnan(mae_val), "MAE should be computed for large tensors"
            except RuntimeError as e:
                if "out of memory" in str(e):
                    pytest.skip("Skipping large tensor test due to memory constraints")
                else:
                    raise
    
    def test_device_consistency(self):
        """Test handling of different device configurations."""
        # Test CPU tensors
        cpu_pred = torch.rand(100) * 10 + 0.1
        cpu_target = torch.rand(100) * 10 + 0.1
        
        cpu_metrics = Metrics()
        rmse_val = cpu_metrics.rmse(cpu_pred, cpu_target)
        assert not np.isnan(rmse_val), "Should handle CPU tensors"
        
        # Test GPU tensors (if available)
        if torch.cuda.is_available():
            gpu_pred = cpu_pred.cuda()
            gpu_target = cpu_target.cuda()
            
            gpu_metrics = Metrics()
            rmse_val = gpu_metrics.rmse(gpu_pred, gpu_target)
            assert not np.isnan(rmse_val), "Should handle GPU tensors"


class TestIntegrationScenarios:
    """Integration tests for realistic usage scenarios."""
    
    def setup_method(self):
        """Setup for integration tests."""
        self.metrics = Metrics()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_nyu_dataset_simulation(self):
        """Test with NYU dataset-like parameters."""
        # NYU dataset typically has depth range 0.1-10.0 meters
        nyu_metrics = Metrics(min_depth=0.1, max_depth=10.0)
        
        # Simulate NYU-like data
        pred_nyu = torch.rand(480, 640, device=self.device) * 9.9 + 0.1
        target_nyu = torch.rand(480, 640, device=self.device) * 9.9 + 0.1
        
        # Should work without issues
        report = nyu_metrics.create_metric_report(pred_nyu, target_nyu, dataset_name="NYU")
        
        assert report['dataset'] == "NYU"
        assert report['valid_pixel_count'] > 0
        assert 'metrics' in report
    
    def test_kitti_dataset_simulation(self):
        """Test with KITTI dataset-like parameters."""
        # KITTI dataset typically has depth range 0.001-80.0 meters
        kitti_metrics = Metrics(min_depth=0.001, max_depth=80.0)
        
        # Simulate KITTI-like data with sparse depth (some zeros)
        pred_kitti = torch.rand(352, 1216, device=self.device) * 79.999 + 0.001
        target_kitti = torch.rand(352, 1216, device=self.device) * 79.999 + 0.001
        
        # Add some sparse regions (common in KITTI)
        sparse_mask = torch.rand(352, 1216, device=self.device) > 0.3
        target_kitti[~sparse_mask] = 0.0
        
        # Should handle sparse data appropriately
        report = kitti_metrics.create_metric_report(pred_kitti, target_kitti, dataset_name="KITTI")
        
        assert report['dataset'] == "KITTI"
        assert report['valid_pixel_count'] > 0
        assert 'metrics' in report
    
    def test_training_loop_integration(self):
        """Test integration with training loop simulation."""
        # Simulate training batch processing
        batch_size = 8
        height, width = 256, 256
        
        pred_batch = torch.rand(batch_size, height, width, device=self.device) * 10 + 0.1
        target_batch = torch.rand(batch_size, height, width, device=self.device) * 10 + 0.1
        
        # Simulate training loop
        epoch_metrics = []
        for epoch in range(3):
            # Add some noise to simulate different batches
            noise_pred = pred_batch + torch.randn_like(pred_batch) * 0.1
            noise_target = target_batch + torch.randn_like(target_batch) * 0.1
            
            batch_results = self.metrics.compute_batch_metrics(noise_pred, noise_target)
            epoch_metrics.append(batch_results)
        
        # Should have consistent results across epochs
        assert len(epoch_metrics) == 3
        for epoch_result in epoch_metrics:
            assert 'rmse' in epoch_result
            assert 'mae' in epoch_result
            assert not np.isnan(epoch_result['rmse']['mean'])
            assert not np.isnan(epoch_result['mae']['mean'])
    
    def test_model_comparison_scenario(self):
        """Test model comparison scenario."""
        # Simulate results from different models
        model_results = {}
        
        # Model A: Good performance
        pred_a = torch.rand(256, 256, device=self.device) * 5 + 2
        target_base = torch.rand(256, 256, device=self.device) * 5 + 2
        model_results['ModelA'] = self.metrics.create_metric_report(pred_a, target_base, dataset_name="Test")
        
        # Model B: Worse performance
        pred_b = pred_a + torch.randn_like(pred_a) * 0.5
        model_results['ModelB'] = self.metrics.create_metric_report(pred_b, target_base, dataset_name="Test")
        
        # Model C: Best performance
        pred_c = target_base + torch.randn_like(target_base) * 0.1
        model_results['ModelC'] = self.metrics.create_metric_report(pred_c, target_base, dataset_name="Test")
        
        # Check that we can distinguish between models
        rmse_a = model_results['ModelA']['metrics']['rmse']
        rmse_b = model_results['ModelB']['metrics']['rmse']
        rmse_c = model_results['ModelC']['metrics']['rmse']
        
        # Model C should have best (lowest) RMSE
        assert rmse_c < rmse_a, "Model C should have better RMSE than Model A"
        assert rmse_c < rmse_b, "Model C should have better RMSE than Model B"


def run_comprehensive_tests():
    """Run comprehensive test suite with detailed output."""
    print("Running comprehensive depth metrics test suite...")
    
    # Initialize test classes
    basic_tests = TestMetrics()
    integration_tests = TestIntegrationScenarios()
    
    test_methods = [
        (basic_tests, 'test_initialization'),
        (basic_tests, 'test_input_validation'),
        (basic_tests, 'test_perfect_predictions'),
        (basic_tests, 'test_known_error_patterns'),
        (basic_tests, 'test_edge_cases'),
        (basic_tests, 'test_masking_functionality'),
        (basic_tests, 'test_return_count_functionality'),
        (basic_tests, 'test_confidence_intervals'),
        (basic_tests, 'test_batch_processing'),
        (basic_tests, 'test_metric_sanity_validation'),
        (basic_tests, 'test_comprehensive_metric_report'),
        (basic_tests, 'test_standalone_functions'),
        (basic_tests, 'test_numerical_stability'),
        (basic_tests, 'test_memory_efficiency'),
        (basic_tests, 'test_device_consistency'),
        (integration_tests, 'test_nyu_dataset_simulation'),
        (integration_tests, 'test_kitti_dataset_simulation'),
        (integration_tests, 'test_training_loop_integration'),
        (integration_tests, 'test_model_comparison_scenario'),
    ]
    
    passed = 0
    failed = 0
    
    for test_instance, test_method in test_methods:
        try:
            # Setup the test
            test_instance.setup_method()
            
            # Run the test
            getattr(test_instance, test_method)()
            
            print(f"✓ {test_method}")
            passed += 1
        except Exception as e:
            print(f"✗ {test_method}: {str(e)}")
            failed += 1
    
    print(f"\nTest Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("All tests passed! ✓")
    else:
        print(f"Some tests failed. Please check the implementation.")
    
    return failed == 0


if __name__ == "__main__":
    # Run tests when script is executed directly
    success = run_comprehensive_tests()
    
    if not success:
        sys.exit(1)
    else:
        print("\nMetrics implementation is working correctly!")
