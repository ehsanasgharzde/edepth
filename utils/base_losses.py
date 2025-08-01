# FILE: utils/base_losses.py
# ehsanasgharzde - FIXED REDUNDANT CODE BY EXTRACTING PURE FUNCTIONS AND BASECLASS LEVEL METHODS

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict
import numpy as np

from utils.tensor_validation import *
from logger.logger import setup_logging 

# Setup logger for factory operations
logger = setup_logging(__file__)

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