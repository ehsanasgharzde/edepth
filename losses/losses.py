# FILE: losses/losses_fixed.py
# ehsanasgharzde - COMPLETE LOSS FUNCTION IMPLEMENTATIONS
# hosseinsolymanzadeh - PROPER COMMENTING
# ehsanasgharzde - FIXED REDUNDANT CODE BY EXTRACTING PURE FUNCTIONS AND BASECLASS LEVEL METHODS

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

from utils.loss import *
from utils.core import *

logger = logging.getLogger(__name__)

class SiLogLoss(DepthLoss):
    def __init__(self, lambda_var: float = 0.85, eps: float = 1e-7, **kwargs):
        super().__init__(**kwargs)
        
        if not 0 <= lambda_var <= 1:
            raise ValueError(f"lambda_var must be in [0,1], got {lambda_var}")
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}")
        
        self.lambda_var = lambda_var
        self.eps = eps
        logger.info(f"Initialized {self.name} with lambda_var={lambda_var}, eps={eps}")
    
    def compute_depth_loss(self, pred: torch.Tensor, target: torch.Tensor, 
                          mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        # Convert to log space (add eps for numerical stability)
        log_pred = torch.log(pred + self.eps)
        log_target = torch.log(target + self.eps)
        
        # Compute log difference
        log_diff = log_pred - log_target

        if mask is None:
            mask = create_default_mask(log_diff)
        
        # Apply mask
        log_diff_masked = apply_mask_safely(log_diff, mask)
        
        # Compute scale-invariant loss
        term1 = torch.mean(log_diff_masked ** 2) # type: ignore
        term2 = self.lambda_var * (torch.mean(log_diff_masked) ** 2) # type: ignore
        
        return term1 - term2


class MAELoss(BaseLoss):
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor, 
                    mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        abs_diff = torch.abs(pred - target)

        if mask is None:
            mask = create_default_mask(abs_diff)

        abs_diff_masked = apply_mask_safely(abs_diff, mask)
        return torch.mean(abs_diff_masked) # type: ignore


class RMSELoss(BaseLoss):
    def __init__(self, eps: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}")
        
        self.eps = eps
    
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor, 
                    mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        squared_diff = (pred - target) ** 2

        if mask is None:
            mask = create_default_mask(squared_diff)

        squared_diff_masked = apply_mask_safely(squared_diff, mask)
        mse = torch.mean(squared_diff_masked) # type: ignore
        return torch.sqrt(mse + self.eps)


class BerHuLoss(BaseLoss):
    def __init__(self, threshold: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        
        if threshold <= 0:
            raise ValueError(f"threshold must be > 0, got {threshold}")
        
        self.threshold = threshold
    
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor, 
                    mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        abs_diff = torch.abs(pred - target)

        if mask is None:
            mask = create_default_mask(abs_diff)

        abs_diff_masked = apply_mask_safely(abs_diff, mask)
        
        # Compute adaptive threshold
        c = self.threshold * torch.max(abs_diff_masked) # type: ignore
        
        # Apply BerHu formula
        l1_mask = abs_diff_masked <= c
        l2_mask = ~l1_mask
        
        berhu_loss = torch.zeros_like(abs_diff_masked) # type: ignore
        berhu_loss[l1_mask] = abs_diff_masked[l1_mask]
        berhu_loss[l2_mask] = (abs_diff_masked[l2_mask] ** 2 + c ** 2) / (2 * c)
        
        return torch.mean(berhu_loss)


class GradientConsistencyLoss(GradientBasedLoss):
    def __init__(self, weight_x: float = 1.0, weight_y: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        
        if weight_x < 0 or weight_y < 0:
            raise ValueError("Gradient weights must be non-negative")
        
        self.weight_x = weight_x
        self.weight_y = weight_y
    
    def compute_gradient_loss(self, pred_grad_x: torch.Tensor, pred_grad_y: torch.Tensor,
                            target_grad_x: torch.Tensor, target_grad_y: torch.Tensor,
                            mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        # Compute gradient differences
        grad_diff_x = torch.abs(pred_grad_x - target_grad_x)
        grad_diff_y = torch.abs(pred_grad_y - target_grad_y)

        # Create or adjust mask to match gradient shapes
        if mask is None:
            mask_x = create_default_mask(grad_diff_x)
            mask_y = create_default_mask(grad_diff_y)
        else:
            # Assume input is (B, 1, H, W) or (B, H, W)
            if mask.ndim == 4:
                mask_x = mask[:, :, :, 1:] if mask.shape[-1] > 1 else None
                mask_y = mask[:, :, 1:, :] if mask.shape[-2] > 1 else None
            elif mask.ndim == 3:
                mask_x = mask[:, :, 1:] if mask.shape[-1] > 1 else None
                mask_y = mask[:, 1:, :] if mask.shape[-2] > 1 else None
            else:
                raise ValueError(f"Unsupported mask shape: {mask.shape}")

        # Compute masked gradient loss
        loss_x = torch.mean(apply_mask_safely(grad_diff_x, mask_x)) # type: ignore
        loss_y = torch.mean(apply_mask_safely(grad_diff_y, mask_y)) # type: ignore

        return self.weight_x * loss_x + self.weight_y * loss_y



class EdgeAwareSmoothnessLoss(ImageGuidedLoss):  
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        
        # Validate input parameters to be strictly positive
        if alpha <= 0 or beta <= 0:
            raise ValueError(f"Alpha and beta must be positive values, got alpha={alpha}, beta={beta}")
        
        # Store parameters controlling edge sensitivity and scaling
        self.alpha = alpha  # Controls edge sensitivity (higher = more sensitive to edges)
        self.beta = beta    # Overall loss scaling factor
        
        logger.info(f"Initialized {self.name} with alpha={alpha}, beta={beta}")
    
    def compute_image_guided_loss(self, pred: torch.Tensor, image: torch.Tensor, 
                                 mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        # Validate compatibility between depth prediction and RGB image
        validate_depth_image_compatibility(pred, image)
        
        # Compute spatial gradients of the predicted depth map
        pred_grad_x, pred_grad_y = compute_spatial_gradients(pred)
        
        # Compute edge-aware weights based on image gradients
        weight_x, weight_y = compute_edge_weights(image, self.alpha)
        
        # Apply edge-aware weights to depth gradients
        smoothness_x = weight_x * torch.abs(pred_grad_x)
        smoothness_y = weight_y * torch.abs(pred_grad_y)
        
        # Apply mask if provided (adjust mask dimensions for gradients)
        if mask is not None:
            # Mask needs to be adjusted for gradient dimensions
            mask_x = mask[..., :, 1:] if mask.shape[-1] > 1 else None
            mask_y = mask[..., 1:, :] if mask.shape[-2] > 1 else None
            
            if mask_x is not None:
                smoothness_x = apply_mask_safely(smoothness_x, mask_x)
            if mask_y is not None:
                smoothness_y = apply_mask_safely(smoothness_y, mask_y)
        
        # Compute final loss as average weighted smoothness in both directions
        loss_x = torch.mean(smoothness_x) if smoothness_x.numel() > 0 else torch.tensor(0.0) # type: ignore
        loss_y = torch.mean(smoothness_y) if smoothness_y.numel() > 0 else torch.tensor(0.0) # type: ignore
        
        total_loss = self.beta * (loss_x + loss_y)
        
        # Log internal metrics for debugging
        logger.debug(f"EdgeAware smoothness breakdown - "
                    f"Loss X: {loss_x.item():.6f}, Loss Y: {loss_y.item():.6f}, "
                    f"Mean depth grad X: {pred_grad_x.abs().mean():.6f}, "
                    f"Mean depth grad Y: {pred_grad_y.abs().mean():.6f}, "
                    f"Weight X mean: {weight_x.mean():.4f}, Weight Y mean: {weight_y.mean():.4f}")
        
        return total_loss


class MultiScaleLoss(BaseLoss):
    def __init__(self, base_loss: BaseLoss, scales: list = [1.0, 0.5, 0.25], 
                 weights: list = [1.0, 0.5, 0.25], **kwargs):
        super().__init__(**kwargs)
        
        if len(scales) != len(weights):
            raise ValueError("Number of scales must match number of weights")
        if any(s <= 0 for s in scales):
            raise ValueError("All scales must be positive")
        if any(w < 0 for w in weights):
            raise ValueError("All weights must be non-negative")
        
        self.base_loss = base_loss
        self.scales = scales
        self.weights = weights
    
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor, 
                    mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        
        for scale, weight in zip(self.scales, self.weights):
            if weight == 0:
                continue
            
            # Resize tensors
            pred_scaled, target_scaled, mask_scaled = resize_tensors_to_scale(
                pred, target, mask, scale
            )
            
            # Compute loss at this scale
            scale_loss = self.base_loss(pred_scaled, target_scaled, mask_scaled, **kwargs)
            total_loss += weight * scale_loss
        
        return total_loss


class MultiLoss(BaseLoss):
    def __init__(self, loss_configs: list, **kwargs):
        super().__init__(**kwargs)
        
        if not loss_configs:
            raise ValueError("At least one loss configuration is required.")
        
        self.losses = []
        self.weights = []
        self.requires_image = []  # Track which losses need image input
        
        for config in loss_configs:
            loss_type = config.get('type')
            weight = config.get('weight', 1.0)
            loss_params = config.get('params', {})
            
            if weight < 0:
                raise ValueError(f"Weight must be non-negative, got {weight}")
            
            # Create loss instance based on type
            if loss_type == 'silog':
                loss_fn = SiLogLoss(**loss_params)
                requires_image = False
            elif loss_type == 'mae':
                loss_fn = MAELoss(**loss_params)
                requires_image = False
            elif loss_type == 'rmse':
                loss_fn = RMSELoss(**loss_params)
                requires_image = False
            elif loss_type == 'berhu':
                loss_fn = BerHuLoss(**loss_params)
                requires_image = False
            elif loss_type == 'gradient':
                loss_fn = GradientConsistencyLoss(**loss_params)
                requires_image = False
            elif loss_type == 'edge_smooth' or loss_type == 'smoothness':
                loss_fn = EdgeAwareSmoothnessLoss(**loss_params)
                requires_image = True  # This loss requires RGB image input
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")
            
            self.losses.append(loss_fn)
            self.weights.append(weight)
            self.requires_image.append(requires_image)
        
        logger.info(f"Initialized MultiLoss with {len(self.losses)} component losses")
    
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor, 
                    mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        # Extract image from kwargs if provided
        image = kwargs.get('image', None)
        
        # Check if any loss requires image input
        if any(self.requires_image) and image is None:
            raise ValueError("Some losses require RGB image input via 'image' parameter")
        
        total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        loss_breakdown = {}
        
        for loss_fn, weight, needs_image in zip(self.losses, self.weights, self.requires_image):
            if weight == 0:
                continue
            
            try:
                # Call appropriate loss based on type
                if needs_image:
                    # Image-guided losses (like EdgeAwareSmoothnessLoss)
                    # For these losses, 'image' is the guidance and 'pred' is the depth
                    component_loss = loss_fn.compute_loss(pred, image, mask, **kwargs)
                else:
                    # Standard losses comparing pred vs target
                    component_loss = loss_fn.compute_loss(pred, target, mask, **kwargs)
                
                weighted_loss = weight * component_loss
                total_loss += weighted_loss
                
                # Store breakdown for debugging
                loss_breakdown[loss_fn.name] = float(component_loss.item())
                
            except Exception as e:
                logger.error(f"Error computing {loss_fn.name}: {str(e)}")
                raise RuntimeError(f"Failed to compute loss component '{loss_fn.name}': {str(e)}")
        
        logger.debug(f"MultiLoss breakdown: {loss_breakdown}")
        return total_loss
    
    def get_loss_breakdown(self, pred: torch.Tensor, target: torch.Tensor, 
                          mask: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, float]:
        image = kwargs.get('image', None)
        loss_breakdown = {}
        
        for loss_fn, weight, needs_image in zip(self.losses, self.weights, self.requires_image):
            if weight == 0:
                loss_breakdown[loss_fn.name] = 0.0
                continue
            
            try:
                if needs_image:
                    component_loss = loss_fn.compute_loss(pred, image, mask, **kwargs)
                else:
                    component_loss = loss_fn.compute_loss(pred, target, mask, **kwargs)
                
                loss_breakdown[loss_fn.name] = float(component_loss.item())
                
            except Exception as e:
                logger.warning(f"Could not compute {loss_fn.name} for breakdown: {str(e)}")
                loss_breakdown[loss_fn.name] = float('nan')
        
        return loss_breakdown