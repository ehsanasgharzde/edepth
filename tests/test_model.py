# FILE: tests/test_model.py
# ehsanasgharzde - COMPLETE MODEL TEST SUITE
# hosseinsolymanzadeh - PROPER COMMENTING

import torch
import torch.nn as nn
import logging
import time
from typing import Dict, Any, List, Tuple, Optional, Union
from models.edepth import edepth
from models.factory import create_model, get_model_info, create_model_from_checkpoint
from metrics.metrics import Metrics
from losses.factory import create_loss

logger = logging.getLogger(__name__)

def analyze_model_complexity(model: nn.Module, input_size: tuple = (1, 3, 224, 224)) -> Dict[str, Any]:
    device = next(model.parameters()).device  # Get the device (CPU/GPU) on which the model is located
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():  # Disable gradient calculation for efficiency
        dummy_input = torch.randn(*input_size).to(device)  # Create a dummy input tensor with given shape

        complexity_info = {}  # Dictionary to store complexity-related information

        # Count total number of parameters
        total_params = sum(p.numel() for p in model.parameters())
        # Count only trainable parameters (those requiring gradients)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Store parameter statistics in the dictionary
        complexity_info['total_parameters'] = total_params
        complexity_info['trainable_parameters'] = trainable_params
        complexity_info['non_trainable_parameters'] = total_params - trainable_params

        # Estimate memory used by parameters (in megabytes, assuming float32 = 4 bytes)
        param_memory_mb = total_params * 4 / (1024 ** 2)
        complexity_info['parameter_memory_mb'] = param_memory_mb

        # Attempt to get detailed model summary if supported
        try:
            model_info = model.get_model_summary(input_size)  # type: ignore
            complexity_info['model_summary'] = model_info
        except Exception as e:
            logger.warning(f"Could not get detailed model summary: {e}")

        # Attempt to get intermediate feature map shapes if supported
        try:
            feature_shapes = model.get_feature_shapes(input_size)  # type: ignore
            complexity_info['feature_shapes'] = feature_shapes
        except Exception as e:
            logger.warning(f"Could not get feature shapes: {e}")

        activation_memory = 0  # Initialize estimated activation memory usage

        # Loop over all modules without children (leaf modules)
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:
                try:
                    output = module(dummy_input)  # Forward dummy input through module
                    if isinstance(output, torch.Tensor):
                        # Estimate memory used by activations (in bytes)
                        activation_memory += output.numel() * 4
                except:
                    pass  # Silently ignore modules that can't process the dummy input

        # Convert activation memory to megabytes and store it
        complexity_info['estimated_activation_memory_mb'] = activation_memory / (1024 ** 2)

        # Log completion of complexity analysis
        logger.info(f"Model complexity analysis completed for {model.__class__.__name__}")

    return complexity_info  # Return the collected complexity information

def compare_models(model_names: List[str], input_size: tuple = (1, 3, 224, 224)) -> Dict[str, Any]:
    # Dictionary to store detailed results for each model
    comparison_results = {}
    
    for model_name in model_names:
        try:
            # Retrieve general info about the model
            model_info = get_model_info(model_name)
            
            # Create model instance
            model = create_model(model_name)
            
            # Analyze model complexity (e.g., params, memory)
            complexity_info = analyze_model_complexity(model, input_size)
            
            # Store successful result with metadata
            comparison_results[model_name] = {
                'model_info': model_info,
                'complexity': complexity_info,
                'creation_status': 'success'
            }
            
            # Log success
            logger.info(f"Successfully analyzed model: {model_name}")
            
        except Exception as e:
            # In case of any failure, record the error details
            comparison_results[model_name] = {
                'model_info': None,
                'complexity': None,
                'creation_status': 'failed',
                'error': str(e)
            }
            # Log the error
            logger.error(f"Failed to analyze model {model_name}: {e}")
    
    # Prepare a concise comparison table across models
    comparison_table = {}
    for model_name, results in comparison_results.items():
        if results['creation_status'] == 'success':
            # Extract complexity and model info for successful cases
            complexity = results['complexity']
            model_info = results['model_info']
            
            # Store summary metrics for comparison
            comparison_table[model_name] = {
                'params': complexity['total_parameters'],
                'trainable_params': complexity['trainable_parameters'],
                'memory_mb': complexity['parameter_memory_mb'],
                'patch_size': model_info.get('patch_size', 'N/A'),
                'embed_dim': model_info.get('embed_dim', 'N/A'),
                'estimated_params': model_info.get('estimated_params', 'N/A')
            }
        else:
            # If model creation failed, mark all fields as error
            comparison_table[model_name] = {
                'params': 'Error',
                'trainable_params': 'Error',
                'memory_mb': 'Error',
                'patch_size': 'Error',
                'embed_dim': 'Error',
                'estimated_params': 'Error'
            }
    
    # Log completion of the comparison process
    logger.info(f"Model comparison completed for {len(model_names)} models")
    
    # Return both detailed and summarized comparison results
    return {
        'detailed_results': comparison_results,
        'comparison_table': comparison_table
    }

def benchmark_model(model: nn.Module, input_size: tuple = (1, 3, 224, 224), num_runs: int = 100) -> Dict[str, Any]:
    # Get the device of the model's parameters
    device = next(model.parameters()).device
    model.eval()  # Set model to evaluation mode
    
    # Create a dummy input tensor with specified size
    dummy_input = torch.randn(*input_size).to(device)
    
    forward_times = []  # List to store forward pass durations
    backward_times = []  # List to store backward pass durations
    memory_usage = []  # List to store memory usage (for CUDA)
    
    for i in range(num_runs):
        # Clear CUDA cache if available
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Measure forward pass time
        start_time = time.perf_counter()
        with torch.no_grad():
            output = model(dummy_input)
        forward_time = time.perf_counter() - start_time
        forward_times.append(forward_time)
        
        # Record memory usage if on CUDA
        if torch.cuda.is_available():
            memory_usage.append(torch.cuda.max_memory_allocated() / 1024 / 1024)  # Convert to MB
        
        # Clone input with gradients enabled for backward pass
        dummy_input_grad = dummy_input.clone().requires_grad_(True)
        
        # Measure backward pass time
        start_time = time.perf_counter()
        output = model(dummy_input_grad)
        loss = output.sum()  # Dummy loss
        loss.backward()
        backward_time = time.perf_counter() - start_time
        backward_times.append(backward_time)
        
        # Log progress every 10 runs
        if i % 10 == 0:
            logger.debug(f"Benchmark progress: {i+1}/{num_runs}")
    
    # Convert timing lists to tensors for easier stats
    forward_times = torch.tensor(forward_times)
    backward_times = torch.tensor(backward_times)
    
    # Compile benchmark results with various statistics
    benchmark_results = {
        'forward_pass': {
            'mean_time_ms': float(forward_times.mean() * 1000),
            'std_time_ms': float(forward_times.std() * 1000),
            'min_time_ms': float(forward_times.min() * 1000),
            'max_time_ms': float(forward_times.max() * 1000),
            'throughput_fps': float(1.0 / forward_times.mean())
        },
        'backward_pass': {
            'mean_time_ms': float(backward_times.mean() * 1000),
            'std_time_ms': float(backward_times.std() * 1000),
            'min_time_ms': float(backward_times.min() * 1000),
            'max_time_ms': float(backward_times.max() * 1000)
        },
        'total_time': {
            'mean_time_ms': float((forward_times + backward_times).mean() * 1000),
            'throughput_fps': float(1.0 / (forward_times + backward_times).mean())
        },
        'memory_usage': {
            'max_memory_mb': float(max(memory_usage)) if memory_usage else 0,
            'avg_memory_mb': float(sum(memory_usage) / len(memory_usage)) if memory_usage else 0
        },
        'benchmark_config': {
            'num_runs': num_runs,
            'input_size': input_size,
            'device': str(device)
        }
    }
    
    # Log benchmark summary
    logger.info(f"Benchmark completed: {benchmark_results['forward_pass']['throughput_fps']:.2f} FPS")
    
    return benchmark_results


def test_model_forward_pass(model: nn.Module, input_size: tuple = (1, 3, 224, 224)) -> Dict[str, Any]:
    # Get model device
    device = next(model.parameters()).device
    model.eval()  # Set model to evaluation mode
    
    test_results = {}  # Dictionary to store test results
    
    with torch.no_grad():
        # Create dummy input
        dummy_input = torch.randn(*input_size).to(device)
        
        try:
            # Run forward pass
            output = model(dummy_input)
            
            # Save output info
            test_results['forward_pass'] = 'success'
            test_results['output_shape'] = tuple(output.shape)
            test_results['output_dtype'] = str(output.dtype)
            test_results['output_device'] = str(output.device)
            
            # Collect statistics on the output tensor
            test_results['output_stats'] = {
                'mean': float(output.mean()),
                'std': float(output.std()),
                'min': float(output.min()),
                'max': float(output.max()),
                'has_nan': bool(torch.isnan(output).any()),
                'has_inf': bool(torch.isinf(output).any())
            }
            
            # Log success
            logger.info(f"Forward pass test successful: {test_results['output_shape']}")
        
        except Exception as e:
            # Catch and log errors in forward pass
            test_results['forward_pass'] = 'failed'
            test_results['error'] = str(e)
            logger.error(f"Forward pass test failed: {e}")
    return test_results

def test_model_metrics(model: nn.Module, input_size: tuple = (1, 3, 224, 224)) -> Dict[str, Any]:
    # Get the device of the model parameters (CPU or GPU)
    device = next(model.parameters()).device
    model.eval()  # Set the model to evaluation mode
    
    metrics_handler = Metrics()  # Initialize the metrics handler
    
    with torch.no_grad():  # Disable gradient computation for inference
        dummy_input = torch.randn(*input_size).to(device)  # Create dummy input tensor
        dummy_target = torch.rand(*input_size).to(device)  # Create dummy target tensor
        
        try:
            pred = model(dummy_input)  # Forward pass through the model
            
            # Compute all metrics between prediction and target
            all_metrics = metrics_handler.compute_all_metrics(pred, dummy_target)
            
            metrics_results = {
                'metrics_computation': 'success',
                'metrics_values': all_metrics,
                'pred_shape': tuple(pred.shape),
                'target_shape': tuple(dummy_target.shape)
            }
            
            logger.info(f"Metrics test successful: {list(all_metrics.keys())}")  # Log successful metrics
            
        except Exception as e:
            # If any error occurs during metrics computation
            metrics_results = {
                'metrics_computation': 'failed',
                'error': str(e)
            }
            logger.error(f"Metrics test failed: {e}")  # Log failure message
    
    return metrics_results  # Return results dictionary

def test_model_loss_computation(model: nn.Module, input_size: tuple = (1, 3, 224, 224)) -> Dict[str, Any]:
    # Get the device of the model parameters
    device = next(model.parameters()).device
    model.train()  # Set model to training mode (for loss computation)
    
    loss_results = {}
    
    try:
        # Create loss functions
        silog_loss = create_loss('silog')
        berhu_loss = create_loss('berhu')
        
        # Generate dummy input and target
        dummy_input = torch.randn(*input_size).to(device)
        dummy_target = torch.rand(*input_size).to(device)
        
        pred = model(dummy_input)  # Forward pass
        
        # Compute losses
        silog_value = silog_loss(pred, dummy_target)
        berhu_value = berhu_loss(pred, dummy_target)
        
        loss_results = {
            'loss_computation': 'success',
            'silog_loss': float(silog_value),
            'berhu_loss': float(berhu_value),
            'losses_require_grad': {
                'silog': silog_value.requires_grad,
                'berhu': berhu_value.requires_grad
            }
        }
        
        logger.info(f"Loss computation test successful")  # Log success
        
    except Exception as e:
        # Handle errors in loss computation
        loss_results = {
            'loss_computation': 'failed',
            'error': str(e)
        }
        logger.error(f"Loss computation test failed: {e}")  # Log error
    
    return loss_results  # Return results dictionary

def test_model_gradient_flow(model: nn.Module, input_size: tuple = (1, 3, 224, 224)) -> Dict[str, Any]:
    # Get the device of the model parameters
    device = next(model.parameters()).device
    model.train()  # Set model to training mode to enable gradient flow
    
    gradient_results = {}
    
    try:
        # Generate dummy input and target
        dummy_input = torch.randn(*input_size).to(device)
        dummy_target = torch.rand(*input_size).to(device)
        
        silog_loss = create_loss('silog')  # Create loss function
        
        pred = model(dummy_input)  # Forward pass
        loss = silog_loss(pred, dummy_target)  # Compute loss
        
        model.zero_grad()  # Clear previous gradients
        loss.backward()  # Backpropagation to compute gradients
        
        grad_norms = {}  # Store gradient norms for parameters
        param_count = 0
        grad_count = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()  # Compute norm of gradient
                grad_norms[name] = grad_norm
                grad_count += 1  # Count parameters with gradients
            param_count += 1  # Count total parameters
        
        gradient_results = {
            'gradient_flow': 'success',
            'total_parameters': param_count,
            'parameters_with_gradients': grad_count,
            'gradient_coverage': grad_count / param_count if param_count > 0 else 0,
            'gradient_norms': grad_norms,
            'mean_gradient_norm': float(torch.tensor(list(grad_norms.values())).mean()) if grad_norms else 0
        }
        
        logger.info(f"Gradient flow test successful: {grad_count}/{param_count} parameters have gradients")
        
    except Exception as e:
        # Handle errors during gradient computation
        gradient_results = {
            'gradient_flow': 'failed',
            'error': str(e)
        }
        logger.error(f"Gradient flow test failed: {e}")  # Log failure
    
    return gradient_results  # Return results dictionary

def run_comprehensive_model_test(model_name: str, input_size: tuple = (1, 3, 224, 224)) -> Dict[str, Any]:
    # Log the start of the comprehensive model test
    logger.info(f"Starting comprehensive test for model: {model_name}")
    
    # Initialize the results dictionary with basic metadata
    comprehensive_results = {
        'model_name': model_name,
        'input_size': input_size,
        'timestamp': time.time()
    }
    
    try:
        # Attempt to create the model instance
        model = create_model(model_name)
        comprehensive_results['model_creation'] = 'success'
        
        # Perform complexity analysis on the model
        comprehensive_results['complexity_analysis'] = analyze_model_complexity(model, input_size)
        
        # Run a forward pass to ensure model operates correctly
        comprehensive_results['forward_pass_test'] = test_model_forward_pass(model, input_size)
        
        # Run evaluation metrics to assess model performance
        comprehensive_results['metrics_test'] = test_model_metrics(model, input_size)
        
        # Test loss computation on a sample input
        comprehensive_results['loss_computation_test'] = test_model_loss_computation(model, input_size)
        
        # Check if gradients can properly flow through the model
        comprehensive_results['gradient_flow_test'] = test_model_gradient_flow(model, input_size)
        
        # Benchmark the model over several runs for performance profiling
        comprehensive_results['benchmark_results'] = benchmark_model(model, input_size, num_runs=10)
        
        # Mark the overall test as successful
        comprehensive_results['overall_status'] = 'success'
        
        # Log success
        logger.info(f"Comprehensive test completed successfully for {model_name}")
        
    except Exception as e:
        # If any error occurs, log and record the failure
        comprehensive_results['model_creation'] = 'failed'
        comprehensive_results['error'] = str(e)
        comprehensive_results['overall_status'] = 'failed'
        logger.error(f"Comprehensive test failed for {model_name}: {e}")
    
    # Return the complete test results
    return comprehensive_results

def validate_model_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    # Dictionary to store validation results
    validation_results = {}
    
    try:
        # Attempt to load model from the given checkpoint path
        model = create_model_from_checkpoint(checkpoint_path)
        validation_results['checkpoint_loading'] = 'success'
        
        # Run comprehensive model test using the loaded model
        validation_results['model_tests'] = run_comprehensive_model_test(
            model.__class__.__name__.lower(), 
            input_size=(1, 3, 224, 224)
        )
        
        # Log success
        logger.info(f"Checkpoint validation completed for {checkpoint_path}")
        
    except Exception as e:
        # On failure, log the error and update the result
        validation_results['checkpoint_loading'] = 'failed'
        validation_results['error'] = str(e)
        logger.error(f"Checkpoint validation failed: {e}")
    
    # Return checkpoint validation results
    return validation_results

def generate_model_report(model_names: List[str], output_path: Optional[str] = None) -> Dict[str, Any]:
    # Log the beginning of report generation
    logger.info(f"Generating model report for {len(model_names)} models")
    
    # Initialize the report dictionary
    report = {
        'report_timestamp': time.time(),
        'tested_models': model_names,
        'model_results': {}
    }
    
    # Iterate over each model and test it
    for model_name in model_names:
        logger.info(f"Testing model: {model_name}")
        report['model_results'][model_name] = run_comprehensive_model_test(model_name)
    
    # If an output path is provided, save the report to disk
    if output_path:
        import json
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Model report saved to {output_path}")
    
    # Return the generated report
    return report
