import torch
import torch.nn as nn
import logging
import time
from typing import Dict, Any, List, Tuple, Optional, Union
from models.model_fixed import edepth
from models.factory import create_model, get_model_info, create_model_from_checkpoint
from metrics.metrics_fixed import Metrics
from losses.factory import create_loss

logger = logging.getLogger(__name__)

def analyze_model_complexity(model: nn.Module, input_size: tuple = (1, 3, 224, 224)) -> Dict[str, Any]:
    device = next(model.parameters()).device
    model.eval()
    
    with torch.no_grad():
        dummy_input = torch.randn(*input_size).to(device)
        
        complexity_info = {}
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        complexity_info['total_parameters'] = total_params
        complexity_info['trainable_parameters'] = trainable_params
        complexity_info['non_trainable_parameters'] = total_params - trainable_params
        
        param_memory_mb = total_params * 4 / (1024 ** 2)
        complexity_info['parameter_memory_mb'] = param_memory_mb
        
        try:
            model_info = model.get_model_summary(input_size) #type: ignore
            complexity_info['model_summary'] = model_info
        except Exception as e:
            logger.warning(f"Could not get detailed model summary: {e}")
        
        try:
            feature_shapes = model.get_feature_shapes(input_size) #type: ignore 
            complexity_info['feature_shapes'] = feature_shapes
        except Exception as e:
            logger.warning(f"Could not get feature shapes: {e}")
        
        activation_memory = 0
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:
                try:
                    output = module(dummy_input)
                    if isinstance(output, torch.Tensor):
                        activation_memory += output.numel() * 4
                except:
                    pass
        
        complexity_info['estimated_activation_memory_mb'] = activation_memory / (1024 ** 2)
        
        logger.info(f"Model complexity analysis completed for {model.__class__.__name__}")
        
    return complexity_info

def compare_models(model_names: List[str], input_size: tuple = (1, 3, 224, 224)) -> Dict[str, Any]:
    comparison_results = {}
    
    for model_name in model_names:
        try:
            model_info = get_model_info(model_name)
            model = create_model(model_name)
            
            complexity_info = analyze_model_complexity(model, input_size)
            
            comparison_results[model_name] = {
                'model_info': model_info,
                'complexity': complexity_info,
                'creation_status': 'success'
            }
            
            logger.info(f"Successfully analyzed model: {model_name}")
            
        except Exception as e:
            comparison_results[model_name] = {
                'model_info': None,
                'complexity': None,
                'creation_status': 'failed',
                'error': str(e)
            }
            logger.error(f"Failed to analyze model {model_name}: {e}")
    
    comparison_table = {}
    for model_name, results in comparison_results.items():
        if results['creation_status'] == 'success':
            complexity = results['complexity']
            model_info = results['model_info']
            
            comparison_table[model_name] = {
                'params': complexity['total_parameters'],
                'trainable_params': complexity['trainable_parameters'],
                'memory_mb': complexity['parameter_memory_mb'],
                'patch_size': model_info.get('patch_size', 'N/A'),
                'embed_dim': model_info.get('embed_dim', 'N/A'),
                'estimated_params': model_info.get('estimated_params', 'N/A')
            }
        else:
            comparison_table[model_name] = {
                'params': 'Error',
                'trainable_params': 'Error',
                'memory_mb': 'Error',
                'patch_size': 'Error',
                'embed_dim': 'Error',
                'estimated_params': 'Error'
            }
    
    logger.info(f"Model comparison completed for {len(model_names)} models")
    
    return {
        'detailed_results': comparison_results,
        'comparison_table': comparison_table
    }

def benchmark_model(model: nn.Module, input_size: tuple = (1, 3, 224, 224), num_runs: int = 100) -> Dict[str, Any]:
    device = next(model.parameters()).device
    model.eval()
    
    dummy_input = torch.randn(*input_size).to(device)
    
    forward_times = []
    backward_times = []
    memory_usage = []
    
    for i in range(num_runs):
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        start_time = time.perf_counter()
        
        with torch.no_grad():
            output = model(dummy_input)
        
        forward_time = time.perf_counter() - start_time
        forward_times.append(forward_time)
        
        if torch.cuda.is_available():
            memory_usage.append(torch.cuda.max_memory_allocated() / 1024 / 1024)
        
        dummy_input_grad = dummy_input.clone().requires_grad_(True)
        
        start_time = time.perf_counter()
        
        output = model(dummy_input_grad)
        loss = output.sum()
        loss.backward()
        
        backward_time = time.perf_counter() - start_time
        backward_times.append(backward_time)
        
        if i % 10 == 0:
            logger.debug(f"Benchmark progress: {i+1}/{num_runs}")
    
    forward_times = torch.tensor(forward_times)
    backward_times = torch.tensor(backward_times)
    
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
    
    logger.info(f"Benchmark completed: {benchmark_results['forward_pass']['throughput_fps']:.2f} FPS")
    
    return benchmark_results

def test_model_forward_pass(model: nn.Module, input_size: tuple = (1, 3, 224, 224)) -> Dict[str, Any]:
    device = next(model.parameters()).device
    model.eval()
    
    test_results = {}
    
    with torch.no_grad():
        dummy_input = torch.randn(*input_size).to(device)
        
        try:
            output = model(dummy_input)
            
            test_results['forward_pass'] = 'success'
            test_results['output_shape'] = tuple(output.shape)
            test_results['output_dtype'] = str(output.dtype)
            test_results['output_device'] = str(output.device)
            
            test_results['output_stats'] = {
                'mean': float(output.mean()),
                'std': float(output.std()),
                'min': float(output.min()),
                'max': float(output.max()),
                'has_nan': bool(torch.isnan(output).any()),
                'has_inf': bool(torch.isinf(output).any())
            }
            
            logger.info(f"Forward pass test successful: {test_results['output_shape']}")
            
        except Exception as e:
            test_results['forward_pass'] = 'failed'
            test_results['error'] = str(e)
            logger.error(f"Forward pass test failed: {e}")
    
    return test_results

def test_model_metrics(model: nn.Module, input_size: tuple = (1, 3, 224, 224)) -> Dict[str, Any]:
    device = next(model.parameters()).device
    model.eval()
    
    metrics_handler = Metrics()
    
    with torch.no_grad():
        dummy_input = torch.randn(*input_size).to(device)
        dummy_target = torch.rand(*input_size).to(device)
        
        try:
            pred = model(dummy_input)
            
            all_metrics = metrics_handler.compute_all_metrics(pred, dummy_target)
            
            metrics_results = {
                'metrics_computation': 'success',
                'metrics_values': all_metrics,
                'pred_shape': tuple(pred.shape),
                'target_shape': tuple(dummy_target.shape)
            }
            
            logger.info(f"Metrics test successful: {list(all_metrics.keys())}")
            
        except Exception as e:
            metrics_results = {
                'metrics_computation': 'failed',
                'error': str(e)
            }
            logger.error(f"Metrics test failed: {e}")
    
    return metrics_results

def test_model_loss_computation(model: nn.Module, input_size: tuple = (1, 3, 224, 224)) -> Dict[str, Any]:
    device = next(model.parameters()).device
    model.train()
    
    loss_results = {}
    
    try:
        silog_loss = create_loss('silog')
        berhu_loss = create_loss('berhu')
        
        dummy_input = torch.randn(*input_size).to(device)
        dummy_target = torch.rand(*input_size).to(device)
        
        pred = model(dummy_input)
        
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
        
        logger.info(f"Loss computation test successful")
        
    except Exception as e:
        loss_results = {
            'loss_computation': 'failed',
            'error': str(e)
        }
        logger.error(f"Loss computation test failed: {e}")
    
    return loss_results

def test_model_gradient_flow(model: nn.Module, input_size: tuple = (1, 3, 224, 224)) -> Dict[str, Any]:
    device = next(model.parameters()).device
    model.train()
    
    gradient_results = {}
    
    try:
        dummy_input = torch.randn(*input_size).to(device)
        dummy_target = torch.rand(*input_size).to(device)
        
        silog_loss = create_loss('silog')
        
        pred = model(dummy_input)
        loss = silog_loss(pred, dummy_target)
        
        model.zero_grad()
        loss.backward()
        
        grad_norms = {}
        param_count = 0
        grad_count = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms[name] = grad_norm
                grad_count += 1
            param_count += 1
        
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
        gradient_results = {
            'gradient_flow': 'failed',
            'error': str(e)
        }
        logger.error(f"Gradient flow test failed: {e}")
    
    return gradient_results

def run_comprehensive_model_test(model_name: str, input_size: tuple = (1, 3, 224, 224)) -> Dict[str, Any]:
    logger.info(f"Starting comprehensive test for model: {model_name}")
    
    comprehensive_results = {
        'model_name': model_name,
        'input_size': input_size,
        'timestamp': time.time()
    }
    
    try:
        model = create_model(model_name)
        comprehensive_results['model_creation'] = 'success'
        
        comprehensive_results['complexity_analysis'] = analyze_model_complexity(model, input_size)
        
        comprehensive_results['forward_pass_test'] = test_model_forward_pass(model, input_size)
        
        comprehensive_results['metrics_test'] = test_model_metrics(model, input_size)
        
        comprehensive_results['loss_computation_test'] = test_model_loss_computation(model, input_size)
        
        comprehensive_results['gradient_flow_test'] = test_model_gradient_flow(model, input_size)
        
        comprehensive_results['benchmark_results'] = benchmark_model(model, input_size, num_runs=10)
        
        comprehensive_results['overall_status'] = 'success'
        
        logger.info(f"Comprehensive test completed successfully for {model_name}")
        
    except Exception as e:
        comprehensive_results['model_creation'] = 'failed'
        comprehensive_results['error'] = str(e)
        comprehensive_results['overall_status'] = 'failed'
        logger.error(f"Comprehensive test failed for {model_name}: {e}")
    
    return comprehensive_results

def validate_model_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    validation_results = {}
    
    try:
        model = create_model_from_checkpoint(checkpoint_path)
        validation_results['checkpoint_loading'] = 'success'
        
        validation_results['model_tests'] = run_comprehensive_model_test(
            model.__class__.__name__.lower(), 
            input_size=(1, 3, 224, 224)
        )
        
        logger.info(f"Checkpoint validation completed for {checkpoint_path}")
        
    except Exception as e:
        validation_results['checkpoint_loading'] = 'failed'
        validation_results['error'] = str(e)
        logger.error(f"Checkpoint validation failed: {e}")
    
    return validation_results

def generate_model_report(model_names: List[str], output_path: Optional[str] = None) -> Dict[str, Any]:
    logger.info(f"Generating model report for {len(model_names)} models")
    
    report = {
        'report_timestamp': time.time(),
        'tested_models': model_names,
        'model_results': {}
    }
    
    for model_name in model_names:
        logger.info(f"Testing model: {model_name}")
        report['model_results'][model_name] = run_comprehensive_model_test(model_name)
    
    if output_path:
        import json
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Model report saved to {output_path}")
    
    return report