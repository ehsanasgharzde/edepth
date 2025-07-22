# FILE: metrics/metrics.py
# ehsanasgharzde - COMPLETE METRICS IMPLEMENTATION
# hosseinsolymanzadeh - FIXED REDUNDANT CODE BY EXTRACTING PURE FUNCTIONS AND BASECLASS LEVEL METHODS

import torch
import logging
import numpy as np
from typing import Optional, Tuple, Union, List, Dict

from ..utils.core import create_default_mask, apply_mask_safely, validate_tensor_inputs

logger = logging.getLogger(__name__)

# Configuration constants
EPS = 1e-6

def compute_bootstrap_ci(values: np.ndarray, confidence_level: float = 0.95, n_bootstrap: int = 1000) -> Tuple[float, float]:
    # If there are fewer than 2 values, return NaN for confidence interval
    if len(values) < 2:
        return float('nan'), float('nan')
    
    # Initialize random number generator with a fixed seed for reproducibility
    rng = np.random.default_rng(42)
    bootstrap_values = []
    
    # Perform bootstrap resampling to compute the mean distribution
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        bootstrap_values.append(np.mean(sample))
    
    # Calculate alpha for the given confidence level
    alpha = 1 - confidence_level
    
    # Compute the lower and upper percentiles for the confidence interval
    lower = np.percentile(bootstrap_values, 100 * (alpha / 2))
    upper = np.percentile(bootstrap_values, 100 * (1 - alpha / 2))
    
    # Return the confidence interval bounds
    return lower, upper #type: ignore

def rmse(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    # Validate input tensors and optional mask for consistency
    validate_tensor_inputs(pred, target, mask)
    
    # Create default mask if none provided (usually all True)
    if mask is None:
        mask = create_default_mask(target)
    
    # Apply the mask to prediction and get count of valid elements
    pred_masked, count = apply_mask_safely(pred, mask)
    # Apply the mask to target as well (count not needed again)
    target_masked, _ = apply_mask_safely(target, mask)
    
    # If no valid elements, set result to NaN
    if count == 0:
        return float('nan')
    
    # Compute mean squared error and then RMSE
    mse = torch.mean((pred_masked - target_masked) ** 2)
    return torch.sqrt(mse).item()
    

def mae(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    # Validate input tensors and optional mask for consistency
    validate_tensor_inputs(pred, target, mask)
    
    # Create default mask if none provided (typically all True)
    if mask is None:
        mask = create_default_mask(target)
    
    # Apply mask to prediction tensor and get count of valid elements
    pred_masked, count = apply_mask_safely(pred, mask)
    # Apply mask to target tensor as well (count not needed again)
    target_masked, _ = apply_mask_safely(target, mask)
    
    # If no valid elements, set result to NaN
    if count == 0:
        return float('nan')
    
    # Compute mean absolute error (MAE) over masked elements
    return torch.mean(torch.abs(pred_masked - target_masked)).item()
    
def delta_metric(pred: torch.Tensor, target: torch.Tensor, threshold: float, 
                 mask: Optional[torch.Tensor] = None) -> float:
    # Validate input tensors and optional mask for consistency
    validate_tensor_inputs(pred, target, mask)
    
    # Create default mask if none provided (usually all True)
    if mask is None:
        mask = create_default_mask(target)
    
    # Create a combined valid mask where:
    # - Original mask is True
    # - Both pred and target values are greater than epsilon to avoid division errors
    valid_mask = mask & (pred > EPS) & (target > EPS)
    
    # Apply valid mask to predictions and get count of valid elements
    pred_masked, count = apply_mask_safely(pred, valid_mask)
    # Apply valid mask to target (count not needed again)
    target_masked, _ = apply_mask_safely(target, valid_mask)
    
    # If no valid elements, result is zero (no valid comparisons)
    if count == 0:
        return 0.0
    
    # Compute the ratio of pred to target and target to pred element-wise
    ratio = torch.max(pred_masked / target_masked, target_masked / pred_masked)
    # Calculate the fraction of elements where the ratio is less than threshold
    return (ratio < threshold).float().mean().item()
    
def delta1(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    # Calculate delta1 accuracy metric with threshold 1.25
    return delta_metric(pred, target, 1.25, mask)

def delta2(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    # Calculate delta2 accuracy metric with threshold 1.25 squared
    return delta_metric(pred, target, 1.25 ** 2, mask)

def delta3(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    # Calculate delta3 accuracy metric with threshold 1.25 cubed
    return delta_metric(pred, target, 1.25 ** 3, mask)

def silog(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    # Validate input tensors and optional mask
    validate_tensor_inputs(pred, target, mask)
    
    # Create default mask if none provided
    if mask is None:
        mask = create_default_mask(target)
    
    # Combine mask with valid value checks (pred and target > epsilon)
    valid_mask = mask & (pred > EPS) & (target > EPS)
    
    # Apply mask safely to pred and get valid count
    pred_masked, count = apply_mask_safely(pred, valid_mask)
    # Apply mask safely to target (count not needed here)
    target_masked, _ = apply_mask_safely(target, valid_mask)
    
    # If no valid elements, return NaN
    if count == 0:
        return float('nan')
    
    # Calculate the difference of log predictions and log targets
    log_diff = torch.log(pred_masked + EPS) - torch.log(target_masked + EPS)
    # Mean of squared log differences
    mean_sq = torch.mean(log_diff ** 2)
    # Mean of log differences
    mean_val = torch.mean(log_diff)

    # SILog error calculation with scale correction factor 0.85
    return torch.sqrt(mean_sq - 0.85 * (mean_val ** 2)).item()

 
def compute_all_metrics(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None, 
                       include_confidence: bool = False) -> Dict[str, Union[float, Dict]]:
    # Initialize dictionary to store metric results
    metrics = {}
    
    try:
        # Compute RMSE and MAE with optional confidence intervals
        if include_confidence:
            metrics['rmse'] = rmse(pred, target, mask)
            metrics['mae'] = mae(pred, target, mask)
        else:
            metrics['rmse'] = rmse(pred, target, mask)
            metrics['mae'] = mae(pred, target, mask)
        
        # Compute delta accuracy metrics at thresholds 1.25, 1.25^2, 1.25^3
        metrics['delta1'] = delta1(pred, target, mask)
        metrics['delta2'] = delta2(pred, target, mask)
        metrics['delta3'] = delta3(pred, target, mask)
        
        # Compute scale-invariant log error metric
        metrics['silog'] = silog(pred, target, mask)
    
    except Exception as e:
        # Log any errors encountered during metric computation
        logger.error(f"Error computing metrics: {e}")
        raise
    
    # Return dictionary containing all computed metrics
    return metrics
    
# Core evaluation metrics
METRICS = {
    'rmse': rmse,
    'mae': mae,
    'delta1': delta1,
    'delta2': delta2,
    'delta3': delta3,
    'silog': silog
}

def compute_batch_metrics(pred_batch: torch.Tensor, target_batch: torch.Tensor, 
                         mask_batch: Optional[torch.Tensor] = None, 
                         metrics_list: Optional[List[str]] = None) -> Dict[str, Dict[str, Union[float, List]]]:
    # Use default list of metrics if none provided
    if metrics_list is None:
        metrics_list = ['rmse', 'mae', 'delta1', 'delta2', 'delta3', 'silog']
    
    # Determine the batch size from prediction tensor
    batch_size = pred_batch.shape[0]
    # Initialize results dictionary to hold lists of metric values for each metric
    results = {metric: [] for metric in metrics_list}
    
    # Loop over each sample in the batch
    for i in range(batch_size):
        pred = pred_batch[i]                   # Extract i-th prediction
        target = target_batch[i]               # Extract i-th target
        mask = mask_batch[i] if mask_batch is not None else None  # Extract i-th mask if available
        
        try:
            # Compute each requested metric if the method exists
            for metric in metrics_list:
                if metric in METRICS:  #Check if metric exists in registry
                    # Call the metric method dynamically and store the result
                    value = METRICS(metric)(pred, target, mask)
                    results[metric].append(value)
                else:
                    # Log a warning for unknown metrics and append NaN
                    logger.warning(f"Unknown metric: {metric}")
                    results[metric].append(float('nan'))
        except Exception as e:
            # On error, log the exception and append NaN for all metrics of this sample
            logger.error(f"Error computing metrics for batch {i}: {e}")
            for metric in metrics_list:
                results[metric].append(float('nan'))
    
    # Prepare summary dictionary to return aggregate statistics per metric
    summary = {}
    for metric, values in results.items():
        values_array = np.array(values, dtype=np.float64)
        # Compute mean, std, min, max ignoring NaNs and include all per-sample values
        summary[metric] = {
            'mean': np.nanmean(values_array),
            'std': np.nanstd(values_array),
            'min': np.nanmin(values_array),
            'max': np.nanmax(values_array),
            'per_sample': values
        }
    
    # Return the summary dictionary with statistics and per-sample metric values
    return summary

def validate_metric_sanity(metrics: Dict[str, float]) -> Tuple[bool, List[str]]:
    # Initialize list to collect warning messages
    warnings_list = []
    # Flag to indicate if metrics pass sanity checks
    is_valid = True
    
    # Retrieve RMSE and MAE values from metrics dictionary
    rmse_val = metrics.get('rmse')
    mae_val = metrics.get('mae')
    
    # Check that RMSE is not less than MAE (which is generally unexpected)
    if rmse_val is not None and mae_val is not None:
        if not np.isnan(rmse_val) and not np.isnan(mae_val):
            if rmse_val < mae_val:
                is_valid = False
                warnings_list.append(f"RMSE ({rmse_val:.4f}) < MAE ({mae_val:.4f})")
    
    # Validate that delta accuracy metrics are within the valid range [0, 1]
    for delta_name in ['delta1', 'delta2', 'delta3']:
        delta_val = metrics.get(delta_name)
        if delta_val is not None and not np.isnan(delta_val):
            if not (0.0 <= delta_val <= 1.0):
                is_valid = False
                warnings_list.append(f"{delta_name} ({delta_val:.4f}) out of [0,1] range")
    
    # Check that the scale-invariant logarithmic error (silog) is not negative
    silog_val = metrics.get('silog')
    if silog_val is not None and not np.isnan(silog_val):
        if silog_val < 0.0:
            is_valid = False
            warnings_list.append(f"SiLog ({silog_val:.4f}) is negative")
    
    # Return overall validity flag and list of warning messages
    return is_valid, warnings_list

def create_metric_report(pred: torch.Tensor, target: torch.Tensor, 
                        mask: Optional[torch.Tensor] = None, 
                        dataset_name: str = "Unknown") -> Dict:
    # If no mask is provided, create a default mask based on target validity
    if mask is None:
        mask = create_default_mask(target)
    
    # Count the number of valid pixels indicated by the mask
    valid_pixels = mask.sum().item()
    
    # If no valid pixels found, log warning and return empty report with a warning
    if valid_pixels == 0:
        logger.warning("No valid pixels for metric computation")
        return {
            'dataset': dataset_name,
            'valid_pixel_count': 0,
            'metrics': {},
            'warnings': ['No valid pixels found']
        }
    
    # Compute all metrics with confidence intervals using masked prediction and target
    metrics = compute_all_metrics(pred, target, mask, include_confidence=True)
    # Check metrics for sanity and collect any warnings
    is_valid, warnings_list = validate_metric_sanity(metrics)  # type: ignore
    
    # Extract statistics of the valid target depth pixels
    target_masked = target[mask]
    depth_stats = {
        'min': target_masked.min().item(),
        'max': target_masked.max().item(),
        'mean': target_masked.mean().item(),
        'std': target_masked.std().item()
    }
    
    # Return a dictionary summarizing dataset name, valid pixels, depth stats, metrics, sanity flag, and warnings
    return {
        'dataset': dataset_name,
        'valid_pixel_count': valid_pixels,
        'depth_statistics': depth_stats,
        'metrics': metrics,
        'sanity_check': is_valid,
        'warnings': warnings_list
    }
