# FILE: utils/tensor_validation.py
# ehsanasgharzde - FIXED REDUNDANT CODE BY EXTRACTING PURE FUNCTIONS AND BASECLASS LEVEL METHODS

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from logger.logger import setup_logging 

# Setup logger for factory operations
logger = setup_logging(__file__)

# Configuration constants
EPS = 1e-6

def create_default_mask(target: torch.Tensor) -> torch.Tensor:
    mask = (
        torch.isfinite(target) &
        (target > 0)
    ).to(torch.bool)
    
    if mask.sum() == 0:
        logger.warning("No valid pixels found after applying depth range mask")
        return torch.zeros_like(target, dtype=torch.bool)
    
    return mask

def apply_mask_safely(tensor: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, int]:
    if tensor.shape != mask.shape:
        raise ValueError(f"Tensor shape {tensor.shape} doesn't match mask shape {mask.shape}")
    
    masked_tensor = tensor[mask]
    valid_count = masked_tensor.numel()
    
    if valid_count == 0:
        logger.warning("No valid pixels after applying mask")
        return torch.tensor([], device=tensor.device, dtype=tensor.dtype), 0
    
    if valid_count < 100:
        logger.warning(f"Very few valid pixels: {valid_count}")
    
    return masked_tensor, valid_count

def resize_tensors_to_scale(pred: torch.Tensor, target: torch.Tensor, 
                           mask: Optional[torch.Tensor], scale: float) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if scale == 1.0:
        return pred, target, mask
    
    # Calculate new size
    *batch_dims, h, w = pred.shape
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Resize prediction and target
    pred_resized = F.interpolate(pred, size=(new_h, new_w), mode='bilinear', align_corners=False)
    target_resized = F.interpolate(target, size=(new_h, new_w), mode='bilinear', align_corners=False)
    
    # Resize mask if present
    mask_resized = None
    if mask is not None:
        mask_resized = F.interpolate(mask.float(), size=(new_h, new_w), mode='nearest')
        mask_resized = mask_resized.bool()
    
    return pred_resized, target_resized, mask_resized

def validate_tensor_inputs(pred: torch.Tensor, target: torch.Tensor, 
                          mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
    # Shape validation
    if pred.shape != target.shape:
        raise ValueError(f"Prediction and target shapes must match. Got {pred.shape} vs {target.shape}")
    
    # Device validation
    if pred.device != target.device:
        raise ValueError(f"Prediction and target must be on same device. Got {pred.device} vs {target.device}")
    
    # Dtype validation
    if not pred.dtype.is_floating_point or not target.dtype.is_floating_point:
        raise TypeError("Prediction and target must be floating point tensors")
    
    # Mask validation
    if mask is not None:
        if not isinstance(mask, torch.Tensor):
            raise TypeError("mask must be torch.Tensor")
        if mask.shape != pred.shape:
            raise ValueError(f"Mask shape {mask.shape} doesn't match prediction shape {pred.shape}")
        if mask.device != pred.device:
            raise ValueError("Mask must be on same device as prediction")
    
    return {
        'shape': pred.shape,
        'device': pred.device,
        'dtype': pred.dtype,
        'has_mask': mask is not None
    }

def validate_depth_image_compatibility(pred: torch.Tensor, image: torch.Tensor):
    # Check batch size and spatial dimensions match
    if pred.shape[0] != image.shape[0] or pred.shape[2:] != image.shape[2:]:
        raise ValueError(f"Shape mismatch: pred {pred.shape}, image {image.shape}")
    
    # Validate expected number of channels (1 for depth, 3 for RGB image)
    if pred.shape[1] != 1:
        raise ValueError(f"Expected pred with 1 channel, got {pred.shape[1]}")
    
    if image.shape[1] != 3:
        raise ValueError(f"Expected image with 3 channels, got {image.shape[1]}")
    
    # Ensure both tensors share the same dtype and device
    if pred.dtype != image.dtype:
        raise ValueError(f"pred and image must have same dtype: {pred.dtype} vs {image.dtype}")
    
    if pred.device != image.device:
        raise ValueError(f"pred and image must be on same device: {pred.device} vs {image.device}")

def validate_depth_values(depth: torch.Tensor) -> torch.Tensor:
    if torch.any(depth < 0):
        logger.warning("Negative depth values detected")
    
    return depth

def validate_numerical_stability(tensor: torch.Tensor, name: str = "tensor") -> torch.Tensor:
    if torch.isnan(tensor).any():
        logger.warning(f"{name} contains NaN values")
        tensor = torch.nan_to_num(tensor, nan=0.0)
    
    if torch.isinf(tensor).any():
        logger.warning(f"{name} contains Inf values")
        tensor = torch.nan_to_num(tensor, posinf=1e6, neginf=-1e6)
    
    return tensor