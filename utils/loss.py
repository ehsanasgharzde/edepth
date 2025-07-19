# FILE: utils/loss.py
# ehsanasgharzde - FIXED REDUNDANT CODE BY EXTRACTING PURE FUNCTINOS AND BASECLASS LEVEL METHODS

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
import logging
from typing import Optional, Tuple, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

# Numerical stability epsilon
EPS = 1e-8


class BaseLoss(nn.Module, ABC):
    def __init__(self, name: str = None): # type: ignore
        super().__init__()
        self.name = name or self.__class__.__name__
        self.loss_history = []
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        # Validate inputs
        validation_info = validate_tensor_inputs(pred, target, mask)
        
        # Check numerical stability
        pred = validate_numerical_stability(pred, "prediction")
        target = validate_numerical_stability(target, "target")
        
        # Compute the actual loss
        loss_value = self.compute_loss(pred, target, mask, **kwargs)
        
        # Ensure loss is scalar
        if loss_value.dim() > 0:
            loss_value = loss_value.mean()
        
        # Log loss value
        self.loss_history.append(float(loss_value.item()))
        logger.debug(f"{self.name} loss: {loss_value.item():.6f}")
        
        return loss_value
    
    @abstractmethod
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor, 
                    mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        pass
    
    def get_statistics(self) -> Dict[str, float]:
        return compute_loss_statistics(self.loss_history)
    
    def reset_history(self):
        self.loss_history = []


class DepthLoss(BaseLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor, 
                    mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        # Validate depth values
        pred = validate_depth_values(pred)
        target = validate_depth_values(target)
        
        return self.compute_depth_loss(pred, target, mask, **kwargs)
    
    @abstractmethod
    def compute_depth_loss(self, pred: torch.Tensor, target: torch.Tensor, 
                          mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        pass

class GradientBasedLoss(BaseLoss):
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor, 
                    mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        # Compute gradients
        pred_grad_x, pred_grad_y = compute_spatial_gradients(pred)
        target_grad_x, target_grad_y = compute_spatial_gradients(target)
        
        return self.compute_gradient_loss(
            pred_grad_x, pred_grad_y, target_grad_x, target_grad_y, mask, **kwargs
        )
    
    @abstractmethod
    def compute_gradient_loss(self, pred_grad_x: torch.Tensor, pred_grad_y: torch.Tensor,
                             target_grad_x: torch.Tensor, target_grad_y: torch.Tensor,
                             mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        pass


class ImageGuidedLoss(BaseLoss):
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor, 
                    mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        # For image-guided losses, target can be the RGB image
        # or image can be passed via kwargs
        image = kwargs.get('image', target)
        
        if image is None:
            raise ValueError(f"{self.name} requires 'image' parameter")
        
        return self.compute_image_guided_loss(pred, image, mask, **kwargs)
    
    @abstractmethod
    def compute_image_guided_loss(self, pred: torch.Tensor, image: torch.Tensor, 
                                 mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        pass


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

def compute_image_gradient_magnitude(image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if image.dim() != 4 or image.shape[1] != 3:
        raise ValueError(f"Expected 4D RGB image with 3 channels, got shape {image.shape}")
    
    # Compute gradients for each channel
    grad_x, grad_y = compute_spatial_gradients(image)
    
    # Compute magnitude by averaging across channels
    grad_mag_x = torch.mean(torch.abs(grad_x), dim=1, keepdim=True)
    grad_mag_y = torch.mean(torch.abs(grad_y), dim=1, keepdim=True)
    
    return grad_mag_x, grad_mag_y

def compute_edge_weights(image: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    grad_mag_x, grad_mag_y = compute_image_gradient_magnitude(image)
    
    # Apply exponential decay to create edge-aware weights
    weight_x = torch.exp(-alpha * grad_mag_x)
    weight_y = torch.exp(-alpha * grad_mag_y)
    
    return weight_x, weight_y

def apply_mask_and_validate(tensor: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mask is not None:
        # Apply mask
        masked_tensor = tensor[mask]
        
        # Check if mask eliminates all pixels
        if masked_tensor.numel() == 0:
            logger.warning("Mask eliminates all pixels, returning zero loss")
            return torch.tensor(0.0, device=tensor.device, requires_grad=True)
        
        return masked_tensor
    
    return tensor.view(-1)  # Flatten for loss computation

def compute_spatial_gradients(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Validate input
    if tensor.dim() < 2:
        raise ValueError(f"Input tensor must have at least 2 dimensions, got {tensor.dim()}")
    
    # Compute gradients
    grad_x = torch.abs(tensor[..., :, 1:] - tensor[..., :, :-1])  # Horizontal gradient
    grad_y = torch.abs(tensor[..., 1:, :] - tensor[..., :-1, :])  # Vertical gradient
    
    return grad_x, grad_y

def compute_loss_statistics(loss_values: list) -> Dict[str, float]:
    if not loss_values:
        return {}
    
    values = np.array(loss_values)
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'median': float(np.median(values)),
        'count': len(values)
    }

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