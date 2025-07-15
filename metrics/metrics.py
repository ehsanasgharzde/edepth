# FILE: metrics/metrics_fixed.py
# ehsanasgharzde - COMPLETE METRICS IMPLEMENTATION

import torch
import numpy as np
import logging
from typing import Optional, Tuple, Union, List, Dict

logger = logging.getLogger(__name__)

class Metrics:
    def __init__(self, min_depth: float = 0.001, max_depth: float = 80.0, eps: float = 1e-6):
        self.min_depth = min_depth  # Minimum valid depth value, used to avoid extremely small or invalid depths
        self.max_depth = max_depth  # Maximum valid depth value, often used to clip predictions or targets
        self.eps = eps              # Small value to avoid division by zero or log(0)
        
    def _validate_inputs(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Check if 'pred' is a torch.Tensor
        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"Expected 'pred' to be torch.Tensor, got {type(pred)}")
        
        # Check if 'target' is a torch.Tensor
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"Expected 'target' to be torch.Tensor, got {type(target)}")
        
        # Ensure 'pred' and 'target' have the same shape
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
        
        # Ensure 'pred' and 'target' are on the same device
        if pred.device != target.device:
            raise ValueError(f"Device mismatch: pred {pred.device} vs target {target.device}")
        
        # If a mask is provided, validate its type, shape, and device
        if mask is not None:
            # Check if 'mask' is a torch.Tensor
            if not isinstance(mask, torch.Tensor):
                raise TypeError(f"Expected 'mask' to be torch.Tensor, got {type(mask)}")
            # Ensure 'mask' has the same shape as 'pred'
            if mask.shape != pred.shape:
                raise ValueError(f"Mask shape {mask.shape} doesn't match pred shape {pred.shape}")
            # Ensure 'mask' is on the same device as 'pred'
            if mask.device != pred.device:
                raise ValueError(f"Mask device {mask.device} doesn't match pred device {pred.device}")
    
    def _create_default_mask(self, target: torch.Tensor) -> torch.Tensor:
        # Create a boolean mask where:
        # - target values are finite
        # - target values are greater than 0
        # - target values are within the specified min and max depth range
        mask = (
            torch.isfinite(target) &
            (target > 0) &
            (target >= self.min_depth) &
            (target <= self.max_depth)
        )
        
        # If no valid pixels are found, log a warning and return a mask of all False
        if mask.sum() == 0:
            logger.warning("No valid pixels found after applying depth range mask")
            return torch.zeros_like(target, dtype=torch.bool)
        
        # Return the computed mask
        return mask
    
    def _apply_mask_safely(self, tensor: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, int]:
        # Ensure the tensor and mask have the same shape
        if tensor.shape != mask.shape:
            raise ValueError(f"Tensor shape {tensor.shape} doesn't match mask shape {mask.shape}")
        
        # Apply the mask to the tensor
        masked_tensor = tensor[mask]
        # Count the number of valid (unmasked) elements
        valid_count = masked_tensor.numel()
        
        # If no valid pixels remain after masking, log a warning and return an empty tensor
        if valid_count == 0:
            logger.warning("No valid pixels after applying mask")
            return torch.tensor([], device=tensor.device, dtype=tensor.dtype), 0
        
        # If very few valid pixels remain, log a warning
        if valid_count < 100:
            logger.warning(f"Very few valid pixels: {valid_count}")
        
        # Return the masked tensor and the count of valid elements
        return masked_tensor, valid_count
    
    def _compute_bootstrap_ci(self, values: np.ndarray, confidence_level: float = 0.95, n_bootstrap: int = 1000) -> Tuple[float, float]:
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
    
    def rmse(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None, 
             return_count: bool = False, with_confidence: bool = False) -> Union[float, Tuple]:
        # Validate input tensors and optional mask for consistency
        self._validate_inputs(pred, target, mask)
        
        # Create default mask if none provided (usually all True)
        if mask is None:
            mask = self._create_default_mask(target)
        
        # Apply the mask to prediction and get count of valid elements
        pred_masked, count = self._apply_mask_safely(pred, mask)
        # Apply the mask to target as well (count not needed again)
        target_masked, _ = self._apply_mask_safely(target, mask)
        
        # If no valid elements, set result to NaN
        if count == 0:
            result = float('nan')
        else:
            # Compute mean squared error and then RMSE
            mse = torch.mean((pred_masked - target_masked) ** 2)
            result = torch.sqrt(mse).item()
        
        # If confidence intervals requested and enough data points
        if with_confidence and count > 1:
            # Compute squared differences and convert to numpy array
            diff_squared = ((pred_masked - target_masked) ** 2).cpu().numpy()
            # Compute bootstrap confidence interval for squared differences
            ci_lower, ci_upper = self._compute_bootstrap_ci(diff_squared)
            confidence_info = {'ci_lower': ci_lower, 'ci_upper': ci_upper}
        else:
            confidence_info = None
        
        # Return results based on requested outputs
        if return_count and with_confidence:
            return result, count, confidence_info
        elif return_count:
            return result, count
        elif with_confidence:
            return result, confidence_info
        else:
            return result
    
    def mae(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None, 
            return_count: bool = False, with_confidence: bool = False) -> Union[float, Tuple]:
        # Validate input tensors and optional mask for consistency
        self._validate_inputs(pred, target, mask)
        
        # Create default mask if none provided (typically all True)
        if mask is None:
            mask = self._create_default_mask(target)
        
        # Apply mask to prediction tensor and get count of valid elements
        pred_masked, count = self._apply_mask_safely(pred, mask)
        # Apply mask to target tensor as well (count not needed again)
        target_masked, _ = self._apply_mask_safely(target, mask)
        
        # If no valid elements, set result to NaN
        if count == 0:
            result = float('nan')
        else:
            # Compute mean absolute error (MAE) over masked elements
            result = torch.mean(torch.abs(pred_masked - target_masked)).item()
        
        # Compute confidence intervals if requested and enough data points available
        if with_confidence and count > 1:
            # Calculate absolute differences and convert to numpy array
            abs_diff = torch.abs(pred_masked - target_masked).cpu().numpy()
            # Compute bootstrap confidence interval on absolute differences
            ci_lower, ci_upper = self._compute_bootstrap_ci(abs_diff)
            confidence_info = {'ci_lower': ci_lower, 'ci_upper': ci_upper}
        else:
            confidence_info = None
        
        # Return results depending on flags for count and confidence intervals
        if return_count and with_confidence:
            return result, count, confidence_info
        elif return_count:
            return result, count
        elif with_confidence:
            return result, confidence_info
        else:
            return result
    
    def _delta_metric(self, pred: torch.Tensor, target: torch.Tensor, threshold: float, 
                     mask: Optional[torch.Tensor] = None, return_count: bool = False) -> Union[float, Tuple]:
        # Validate input tensors and optional mask for consistency
        self._validate_inputs(pred, target, mask)
        
        # Create default mask if none provided (usually all True)
        if mask is None:
            mask = self._create_default_mask(target)
        
        # Create a combined valid mask where:
        # - Original mask is True
        # - Both pred and target values are greater than epsilon to avoid division errors
        valid_mask = mask & (pred > self.eps) & (target > self.eps)
        
        # Apply valid mask to predictions and get count of valid elements
        pred_masked, count = self._apply_mask_safely(pred, valid_mask)
        # Apply valid mask to target (count not needed again)
        target_masked, _ = self._apply_mask_safely(target, valid_mask)
        
        # If no valid elements, result is zero (no valid comparisons)
        if count == 0:
            result = 0.0
        else:
            # Compute the ratio of pred to target and target to pred element-wise
            ratio = torch.max(pred_masked / target_masked, target_masked / pred_masked)
            # Calculate the fraction of elements where the ratio is less than threshold
            result = (ratio < threshold).float().mean().item()
        
        # Return result and optionally the count of valid elements
        if return_count:
            return result, count
        else:
            return result
    
    def delta1(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None, 
               return_count: bool = False) -> Union[float, Tuple]:
        # Calculate delta1 accuracy metric with threshold 1.25
        return self._delta_metric(pred, target, 1.25, mask, return_count)
    
    def delta2(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None, 
               return_count: bool = False) -> Union[float, Tuple]:
        # Calculate delta2 accuracy metric with threshold 1.25 squared
        return self._delta_metric(pred, target, 1.25 ** 2, mask, return_count)
    
    def delta3(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None, 
               return_count: bool = False) -> Union[float, Tuple]:
        # Calculate delta3 accuracy metric with threshold 1.25 cubed
        return self._delta_metric(pred, target, 1.25 ** 3, mask, return_count)
    
    def silog(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None, 
              return_count: bool = False) -> Union[float, Tuple]:
        # Validate input tensors and optional mask
        self._validate_inputs(pred, target, mask)
        
        # Create default mask if none provided
        if mask is None:
            mask = self._create_default_mask(target)
        
        # Combine mask with valid value checks (pred and target > epsilon)
        valid_mask = mask & (pred > self.eps) & (target > self.eps)
        
        # Apply mask safely to pred and get valid count
        pred_masked, count = self._apply_mask_safely(pred, valid_mask)
        # Apply mask safely to target (count not needed here)
        target_masked, _ = self._apply_mask_safely(target, valid_mask)
        
        # If no valid elements, return NaN
        if count == 0:
            result = float('nan')
        else:
            # Calculate the difference of log predictions and log targets
            log_diff = torch.log(pred_masked + self.eps) - torch.log(target_masked + self.eps)
            # Mean of squared log differences
            mean_sq = torch.mean(log_diff ** 2)
            # Mean of log differences
            mean_val = torch.mean(log_diff)
            # SILog error calculation with scale correction factor 0.85
            result = torch.sqrt(mean_sq - 0.85 * (mean_val ** 2)).item()
        
        # Return result with optional count
        if return_count:
            return result, count
        else:
            return result
    
    def compute_all_metrics(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None, 
                           include_confidence: bool = False) -> Dict[str, Union[float, Dict]]:
        # Initialize dictionary to store metric results
        metrics = {}
        
        try:
            # Compute RMSE and MAE with optional confidence intervals
            if include_confidence:
                metrics['rmse'] = self.rmse(pred, target, mask, with_confidence=True)
                metrics['mae'] = self.mae(pred, target, mask, with_confidence=True)
            else:
                metrics['rmse'] = self.rmse(pred, target, mask)
                metrics['mae'] = self.mae(pred, target, mask)
            
            # Compute delta accuracy metrics at thresholds 1.25, 1.25^2, 1.25^3
            metrics['delta1'] = self.delta1(pred, target, mask)
            metrics['delta2'] = self.delta2(pred, target, mask)
            metrics['delta3'] = self.delta3(pred, target, mask)
            
            # Compute scale-invariant log error metric
            metrics['silog'] = self.silog(pred, target, mask)
        
        except Exception as e:
            # Log any errors encountered during metric computation
            logger.error(f"Error computing metrics: {e}")
            raise
        
        # Return dictionary containing all computed metrics
        return metrics
    
    def compute_batch_metrics(self, pred_batch: torch.Tensor, target_batch: torch.Tensor, 
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
                    if hasattr(self, metric):
                        # Call the metric method dynamically and store the result
                        value = getattr(self, metric)(pred, target, mask)
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
    
    def validate_metric_sanity(self, metrics: Dict[str, float]) -> Tuple[bool, List[str]]:
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
    
    def create_metric_report(self, pred: torch.Tensor, target: torch.Tensor, 
                            mask: Optional[torch.Tensor] = None, 
                            dataset_name: str = "Unknown") -> Dict:
        # If no mask is provided, create a default mask based on target validity
        if mask is None:
            mask = self._create_default_mask(target)
        
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
        metrics = self.compute_all_metrics(pred, target, mask, include_confidence=True)
        # Check metrics for sanity and collect any warnings
        is_valid, warnings_list = self.validate_metric_sanity(metrics)  # type: ignore
        
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

def rmse(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    # Compute Root Mean Squared Error using Metrics class
    return Metrics().rmse(pred, target, mask)  # type: ignore

def mae(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    # Compute Mean Absolute Error using Metrics class
    return Metrics().mae(pred, target, mask)  # type: ignore

def delta1(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    # Compute delta1 accuracy metric using Metrics class
    return Metrics().delta1(pred, target, mask)  # type: ignore

def delta2(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    # Compute delta2 accuracy metric using Metrics class
    return Metrics().delta2(pred, target, mask)  # type: ignore

def delta3(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    # Compute delta3 accuracy metric using Metrics class
    return Metrics().delta3(pred, target, mask)  # type: ignore

def silog(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    # Compute scale-invariant logarithmic error (SiLog) using Metrics class
    return Metrics().silog(pred, target, mask)  # type: ignore

def compute_all_metrics(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
    # Compute all defined metrics and return them in a dictionary using Metrics class
    return Metrics().compute_all_metrics(pred, target, mask)  # type: ignore