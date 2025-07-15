# FILE: losses/losses_fixed.py
# ehsanasgharzde - COMPLETE LOSS FUNCTION IMPLEMENTATIONS

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Callable, Dict, Optional, List

logger = logging.getLogger(__name__)

# Numerical stability epsilon
EPS = 1e-8

# Default loss weights (can be tuned)
DEFAULT_LOSS_WEIGHTS = {
    'l1': 1.0,
    'l2': 1.0,
    'ssim': 0.85,
    'bce': 1.0,
    'dice': 1.0,
}

# Supported loss types
SUPPORTED_LOSSES = ['l1', 'l2', 'bce', 'dice', 'focal', 'ssim', 'combo', 'custom']

class SiLogLoss(nn.Module):
    """
    Scale-Invariant Logarithmic Loss for depth estimation
    Key characteristics:
    - Scale invariant: handles different depth ranges
    - Logarithmic: better for relative depth errors
    - Variance term: reduces overfitting to mean depth
    """
    
    def __init__(self, eps: float = 1e-7, lambda_var: float = 0.85):
        super().__init__()
        
        # Validate epsilon value to avoid division/log(0)
        if eps <= 0:
            raise ValueError(f"eps must be > 0 for numerical stability. Got: {eps}")
        
        # Validate lambda_var is within [0, 1] for weighting the variance term
        if not (0 <= lambda_var <= 1):
            raise ValueError(f"lambda_var must be in [0, 1]. Got: {lambda_var}")

        # Save parameters
        self.eps = eps
        self.lambda_var = lambda_var

        # Log initialization parameters
        logger.info(f"Initialized SiLogLoss with eps={self.eps}, lambda_var={self.lambda_var}")
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Check for shape mismatch between prediction and ground truth
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape}, target {target.shape}")
        
        # Ensure inputs are floating point tensors
        if not pred.dtype.is_floating_point or not target.dtype.is_floating_point:
            raise TypeError("pred and target must be float tensors")

        # Ensure tensors are on the same device
        if pred.device != target.device:
            raise ValueError("pred and target must be on the same device")

        # Warn if target contains invalid depth values (non-positive)
        if (target <= 0).any():
            logger.warning("Target contains non-positive depth values")
        
        # Create or refine validity mask to exclude invalid/masked pixels
        if mask is None:
            mask = (target > self.eps) & torch.isfinite(target) & torch.isfinite(pred)
        else:
            mask = mask & (target > self.eps) & torch.isfinite(target) & torch.isfinite(pred)
        
        # Count valid pixels and warn if none are valid
        valid_pixels = mask.sum().item()
        total_pixels = mask.numel()
        if valid_pixels == 0:
            logger.warning("Empty mask: no valid pixels for loss computation")
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype, requires_grad=True)
        
        # Log the ratio of valid pixels
        logger.debug(f"Valid pixels: {valid_pixels}/{total_pixels} ({100.0 * valid_pixels / total_pixels:.2f}%)")

        # Add epsilon for numerical stability before taking logarithm
        pred = pred[mask] + self.eps
        target = target[mask] + self.eps

        # Compute log difference
        log_diff = torch.log(pred) - torch.log(target)

        # Check for numerical instability in log difference
        if torch.isnan(log_diff).any() or torch.isinf(log_diff).any():
            raise ValueError("Log difference contains NaN or Inf values")

        # Compute mean squared error of log differences
        mse_log = torch.mean(log_diff ** 2)

        # Compute mean of log differences for scale-invariant variance term
        mean_log = torch.mean(log_diff)
        var_term = self.lambda_var * (mean_log ** 2)

        # Final loss = MSE(log) - lambda * variance term
        loss = mse_log - var_term

        # Validate that the final loss is finite
        if not torch.isfinite(loss):
            raise ValueError(f"SiLogLoss computed an invalid value: {loss.item()}")
        
        # Log detailed loss components
        logger.debug(f"SiLogLoss | MSE(log): {mse_log.item():.6f}, Var term: {var_term.item():.6f}, Final Loss: {loss.item():.6f}")
        
        return loss


class EdgeAwareSmoothnessLoss(nn.Module):
    """
    Edge-Aware Smoothness Loss for depth estimation.
    Encourages smooth depth predictions while preserving edges
    using image gradients to weight the smoothness cost.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__()
        # Validate input parameters to be strictly positive
        if alpha <= 0 or beta <= 0:
            raise ValueError("Alpha and beta must be positive values.")
        
        # Store parameters controlling edge sensitivity and scaling
        self.alpha = alpha
        self.beta = beta

        # Log configuration of the loss function
        logger.debug(f"EdgeAwareSmoothnessLoss initialized with alpha={alpha}, beta={beta}")

    def _compute_gradient_x(self, img: torch.Tensor) -> torch.Tensor:
        """Compute horizontal (x-direction) image gradients"""
        return img[:, :, :, :-1] - img[:, :, :, 1:]

    def _compute_gradient_y(self, img: torch.Tensor) -> torch.Tensor:
        """Compute vertical (y-direction) image gradients"""
        return img[:, :, :-1, :] - img[:, :, 1:, :]

    def forward(self, pred: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        # Ensure batch size and spatial dimensions match
        if pred.shape[0] != image.shape[0] or pred.shape[2:] != image.shape[2:]:
            raise ValueError(f"Shape mismatch: pred {pred.shape}, image {image.shape}")
        
        # Validate expected number of channels (1 for depth, 3 for RGB image)
        if pred.shape[1] != 1 or image.shape[1] != 3:
            raise ValueError("Expected pred with 1 channel and image with 3 channels")
        
        # Ensure both tensors share the same dtype and device
        if pred.dtype != image.dtype or pred.device != image.device:
            raise ValueError("pred and image must have same dtype and device")

        # Compute spatial gradients of the predicted depth map
        pred_dx = self._compute_gradient_x(pred)
        pred_dy = self._compute_gradient_y(pred)

        # Compute spatial gradients of the RGB image (used to guide smoothness)
        image_dx = self._compute_gradient_x(image)
        image_dy = self._compute_gradient_y(image)

        # Compute average image gradient magnitude per pixel for each direction
        grad_mag_x = torch.mean(torch.abs(image_dx), dim=1, keepdim=True)
        grad_mag_y = torch.mean(torch.abs(image_dy), dim=1, keepdim=True)

        # Compute edge-aware weights by applying exponential decay to gradients
        weight_x = torch.exp(-self.alpha * grad_mag_x)
        weight_y = torch.exp(-self.alpha * grad_mag_y)

        # Compute edge-aware smoothness loss using depth gradients
        smoothness_x = weight_x * torch.abs(pred_dx)
        smoothness_y = weight_y * torch.abs(pred_dy)

        # Final loss is the average weighted smoothness in both directions, scaled by beta
        loss = self.beta * (smoothness_x.mean() + smoothness_y.mean())

        # Log internal metrics for debugging and diagnostics
        logger.debug(f"Smoothness loss: {loss.item():.6f}, "
                     f"Mean depth grad X: {pred_dx.abs().mean():.6f}, "
                     f"Mean depth grad Y: {pred_dy.abs().mean():.6f}, "
                     f"Weight X mean: {weight_x.mean():.4f}, Weight Y mean: {weight_y.mean():.4f}")

        return loss


class GradientConsistencyLoss(nn.Module):
    """
    Gradient Consistency Loss for depth estimation.
    Penalizes inconsistencies in depth gradients between predicted and ground truth maps.
    """

    def __init__(self, weight_x: float = 1.0, weight_y: float = 1.0):
        super().__init__()
        # Validate that weights are non-negative
        if weight_x < 0 or weight_y < 0:
            raise ValueError("weight_x and weight_y must be non-negative")

        # Store weights for horizontal and vertical gradient differences
        self.weight_x = weight_x
        self.weight_y = weight_y

        # Log configuration
        logger.debug(f"Initialized GradientConsistencyLoss with weight_x={weight_x}, weight_y={weight_y}")

    def _compute_gradient_x(self, img: torch.Tensor) -> torch.Tensor:
        """Compute horizontal (x-direction) gradients using finite difference."""
        return img[..., :, :-1] - img[..., :, 1:]

    def _compute_gradient_y(self, img: torch.Tensor) -> torch.Tensor:
        """Compute vertical (y-direction) gradients using finite difference."""
        return img[..., :-1, :] - img[..., 1:, :]

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Validate shape match
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape}, target {target.shape}")
        # Validate input tensor types
        if not pred.dtype.is_floating_point or not target.dtype.is_floating_point:
            raise TypeError("Both pred and target must be float tensors")
        # Ensure tensors are on the same device
        if pred.device != target.device:
            raise ValueError("pred and target must be on the same device")

        # Compute gradients for prediction and target in x and y directions
        pred_dx = self._compute_gradient_x(pred)
        pred_dy = self._compute_gradient_y(pred)
        target_dx = self._compute_gradient_x(target)
        target_dy = self._compute_gradient_y(target)

        # Prepare validity masks for each gradient direction
        if mask is not None:
            # Check mask shape
            if mask.shape != pred.shape:
                raise ValueError(f"Mask shape must match pred shape. Got {mask.shape}")
            # Compute gradient validity masks
            mask_dx = mask[..., :, :-1] & mask[..., :, 1:]
            mask_dy = mask[..., :-1, :] & mask[..., 1:, :]
        else:
            # Default to all valid pixels
            mask_dx = torch.ones_like(pred_dx, dtype=torch.bool)
            mask_dy = torch.ones_like(pred_dy, dtype=torch.bool)

        # Compute absolute differences in gradients
        diff_dx = torch.abs(pred_dx - target_dx)
        diff_dy = torch.abs(pred_dy - target_dy)

        # Apply masks to keep only valid differences
        diff_dx = diff_dx[mask_dx]
        diff_dy = diff_dy[mask_dy]

        # Warn and return zero loss if no valid pixels remain
        if diff_dx.numel() == 0 or diff_dy.numel() == 0:
            logger.warning("GradientConsistencyLoss: no valid pixels after masking.")
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype, requires_grad=True)

        # Compute mean absolute gradient differences
        loss_dx = diff_dx.mean()
        loss_dy = diff_dy.mean()

        # Total weighted loss
        total_loss = self.weight_x * loss_dx + self.weight_y * loss_dy

        # Log loss components for diagnostics
        logger.debug(f"GradientConsistencyLoss | loss_dx: {loss_dx.item():.6f}, "
                     f"loss_dy: {loss_dy.item():.6f}, total: {total_loss.item():.6f}")

        return total_loss


class MultiScaleLoss(nn.Module):
    """
    Multi-Scale Loss for depth estimation.
    Computes losses at multiple resolutions to better supervise both global structure and fine details.
    """

    def __init__(self, scales: List[float] = [1.0, 0.5, 0.25], weights: List[float] = [1.0, 0.5, 0.25], mode: str = 'bilinear'):
        super().__init__()
        # Ensure scales and weights lists match in length
        if len(scales) != len(weights):
            raise ValueError("Length of scales and weights must match")
        # Ensure all scales are positive
        if not all(s > 0 for s in scales):
            raise ValueError("All scales must be positive")
        # Ensure all weights are non-negative
        if not all(w >= 0 for w in weights):
            raise ValueError("All weights must be non-negative")

        # Store parameters
        self.scales = scales
        self.weights = weights
        self.mode = mode

        # Log configuration
        logger.debug(f"Initialized MultiScaleLoss with scales={scales}, weights={weights}, interpolation={mode}")

    def forward(self, pred: torch.Tensor, target: torch.Tensor, base_loss_fn: nn.Module, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Ensure prediction and target shapes match
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape}, target {target.shape}")
        # If mask is provided, check shape compatibility
        if mask is not None and mask.shape != pred.shape:
            raise ValueError(f"Mask shape must match prediction shape: mask {mask.shape}, pred {pred.shape}")
        # Ensure the provided loss function is callable
        if not callable(base_loss_fn):
            raise TypeError("base_loss_fn must be a callable loss function")

        # Initialize total loss tensor
        total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        scale_losses = []

        # Loop over each scale and corresponding weight
        for i, (scale, weight) in enumerate(zip(self.scales, self.weights)):
            if scale == 1.0:
                # No resizing needed for full scale
                scaled_pred = pred
                scaled_target = target
                scaled_mask = mask
            else:
                # Compute new resolution based on scale
                new_size = [int(pred.shape[2] * scale), int(pred.shape[3] * scale)]
                # Downsample prediction and target
                scaled_pred = F.interpolate(pred, size=new_size, mode=self.mode, align_corners=False)
                scaled_target = F.interpolate(target, size=new_size, mode=self.mode, align_corners=False)
                # Downsample mask using nearest interpolation, if provided
                scaled_mask = (
                    F.interpolate(mask.float(), size=new_size, mode='nearest') > 0.5
                    if mask is not None else None
                )

            # Compute loss at current scale
            loss = base_loss_fn(scaled_pred, scaled_target, scaled_mask)
            # Apply corresponding weight
            weighted_loss = weight * loss
            # Accumulate total loss
            total_loss += weighted_loss
            # Track individual scale loss for logging
            scale_losses.append(weighted_loss.item())

            # Log each scale's individual contribution
            logger.debug(f"Scale {scale:.2f}: Loss = {loss.item():.6f}, Weighted = {weighted_loss.item():.6f}")

        # Log total loss and per-scale breakdown
        logger.debug(f"MultiScaleLoss total: {total_loss.item():.6f}, per-scale: {scale_losses}")
        return total_loss


class BerHuLoss(nn.Module):
    """
    Reverse Huber (BerHu) Loss for depth estimation.
    Combines L1 loss for small residuals and L2 loss for large residuals,
    improving robustness to outliers.
    """

    def __init__(self, threshold: float = 0.2):
        super().__init__()
        # Validate threshold
        if threshold <= 0:
            raise ValueError("Threshold must be a positive float")
        self.threshold = threshold

        # Log initialization
        logger.debug(f"Initialized BerHuLoss with threshold = {threshold}")

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Check shape compatibility
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape}, target {target.shape}")
        # Ensure tensors are on the same device
        if pred.device != target.device:
            raise ValueError("pred and target must be on the same device")
        # Ensure input tensors are of float type
        if not pred.dtype.is_floating_point or not target.dtype.is_floating_point:
            raise TypeError("Inputs must be floating-point tensors")

        # Compute absolute difference between prediction and target
        diff = torch.abs(pred - target)

        # Apply optional mask
        if mask is not None:
            # Validate mask shape
            if mask.shape != pred.shape:
                raise ValueError(f"Mask shape must match prediction shape: mask {mask.shape}")
            # Convert mask to boolean if necessary
            if not mask.dtype == torch.bool:
                mask = mask > 0.5
            # Apply mask to filter valid pixels
            diff = diff[mask]

        # If no valid pixels remain, return zero loss
        if diff.numel() == 0:
            logger.warning("BerHuLoss: no valid pixels after masking.")
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype, requires_grad=True)

        # Compute BerHu threshold c (proportional to maximum residual)
        c = self.threshold * diff.max().item()

        # Identify elements for L1 and L2 regions
        l1_mask = diff <= c
        l2_mask = diff > c

        # Extract L1 and L2 components
        l1_loss = diff[l1_mask]
        l2_loss = diff[l2_mask]

        # Compute L1 loss (mean of small residuals)
        loss_l1 = l1_loss.mean() if l1_loss.numel() > 0 else torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        # Compute L2 loss (scaled for large residuals)
        loss_l2 = ((l2_loss ** 2 + c ** 2) / (2 * c)).mean() if l2_loss.numel() > 0 else torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # Total loss is sum of L1 and L2 parts
        total_loss = loss_l1 + loss_l2

        # Log detailed loss components
        logger.debug(f"BerHuLoss | c: {c:.6f}, L1 mean: {loss_l1.item():.6f}, L2 mean: {loss_l2.item():.6f}, total: {total_loss.item():.6f}")

        return total_loss


class RMSELoss(nn.Module):
    """
    Root Mean Square Error Loss for depth estimation.
    Computes sqrt(mean((pred - target)^2)).
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        # Validate epsilon to avoid sqrt of zero
        if eps <= 0:
            raise ValueError("eps must be a positive float")
        self.eps = eps
        # Log initialization details
        logger.debug(f"Initialized RMSELoss with epsilon = {eps}")

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Check for shape mismatch
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape}, target {target.shape}")
        # Ensure tensors are on the same device
        if pred.device != target.device:
            raise ValueError("pred and target must be on the same device")
        # Ensure input tensors are of float type
        if not pred.dtype.is_floating_point or not target.dtype.is_floating_point:
            raise TypeError("Inputs must be floating-point tensors")

        # Compute squared difference
        sq_diff = (pred - target) ** 2

        # Apply mask if provided
        if mask is not None:
            # Check mask shape
            if mask.shape != pred.shape:
                raise ValueError(f"Mask shape must match input shape: mask {mask.shape}")
            # Convert to boolean mask if needed
            if not mask.dtype == torch.bool:
                mask = mask > 0.5
            # Apply mask
            sq_diff = sq_diff[mask]

        # If no valid pixels remain, return zero loss
        if sq_diff.numel() == 0:
            logger.warning("RMSELoss: no valid pixels after masking.")
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype, requires_grad=True)

        # Compute mean of squared errors
        mean_sq_error = sq_diff.mean()
        # Compute RMSE with epsilon for numerical stability
        rmse = torch.sqrt(mean_sq_error + self.eps)

        # Log computation details
        logger.debug(f"RMSELoss | mean squared error: {mean_sq_error:.6f}, RMSE: {rmse:.6f}")
        return rmse


class MAELoss(nn.Module):
    """
    Mean Absolute Error Loss for depth estimation.
    Computes the average of the absolute differences between predicted and true values.
    Less sensitive to outliers than MSE.
    """

    def __init__(self):
        super().__init__()
        # Log initialization of MAELoss
        logger.debug("Initialized MAELoss.")

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Ensure prediction and target have the same shape
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape}, target {target.shape}")
        # Ensure prediction and target are on the same device
        if pred.device != target.device:
            raise ValueError("pred and target must be on the same device")
        # Check that inputs are of floating-point type
        if not pred.dtype.is_floating_point or not target.dtype.is_floating_point:
            raise TypeError("Inputs must be floating-point tensors")

        # Compute absolute differences
        abs_diff = torch.abs(pred - target)

        # Apply optional mask
        if mask is not None:
            # Validate mask shape
            if mask.shape != pred.shape:
                raise ValueError(f"Mask shape must match input shape: mask {mask.shape}")
            # Ensure mask is boolean
            if not mask.dtype == torch.bool:
                mask = mask > 0.5
            # Apply mask to absolute differences
            abs_diff = abs_diff[mask]

        # Handle case of no valid pixels
        if abs_diff.numel() == 0:
            logger.warning("MAELoss: no valid pixels after masking.")
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype, requires_grad=True)

        # Compute mean absolute error
        mae = abs_diff.mean()

        # Log the computed MAE
        logger.debug(f"MAELoss | Mean Absolute Error: {mae:.6f}")
        return mae


class MultiLoss(nn.Module):
    """
    Combined Loss for comprehensive depth estimation.
    Combines multiple loss functions with configurable weights.
    """

    def __init__(self, silog_weight: float = 1.0, smoothness_weight: float = 0.1, gradient_weight: float = 0.1, berhu_weight: float = 0.0, loss_config: Optional[Dict] = None):
        super().__init__()

        # Validate that all weights are non-negative
        for name, w in zip(["SiLog", "Smoothness", "Gradient", "BerHu"], [silog_weight, smoothness_weight, gradient_weight, berhu_weight]):
            if w < 0:
                raise ValueError(f"{name} weight must be non-negative.")

        # Store weights in a dictionary for easier access
        self.weights = {
            "silog": silog_weight,
            "smoothness": smoothness_weight,
            "gradient": gradient_weight,
            "berhu": berhu_weight,
        }

        # Initialize individual loss functions
        self.silog_loss = SiLogLoss()
        self.smoothness_loss = EdgeAwareSmoothnessLoss()
        self.gradient_loss = GradientConsistencyLoss()
        self.berhu_loss = BerHuLoss() if berhu_weight > 0 else None  # Conditionally initialize BerHu

        # Log initialization details
        logger.debug(f"Initialized MultiLoss with weights: {self.weights}")
        self.loss_history = []  # Track loss values across epochs or batches

    def forward(self, pred: torch.Tensor, target: torch.Tensor, image: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Check for shape mismatch between prediction and target
        if pred.shape != target.shape:
            raise ValueError("Prediction and target shape mismatch.")
        # Check for batch size mismatch between image and prediction
        if image.shape[0] != pred.shape[0]:
            raise ValueError("Batch size mismatch between image and depth tensors.")

        # Initialize total loss
        total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        loss_breakdown = {}  # Dictionary to store individual losses for logging

        # Compute SiLog loss if its weight is positive
        if self.weights["silog"] > 0:
            silog = self.silog_loss(pred, target, mask)
            total_loss += self.weights["silog"] * silog
            loss_breakdown["silog"] = silog.item()

        # Compute smoothness loss if its weight is positive
        if self.weights["smoothness"] > 0:
            smooth = self.smoothness_loss(pred, image)
            total_loss += self.weights["smoothness"] * smooth
            loss_breakdown["smoothness"] = smooth.item()

        # Compute gradient consistency loss if its weight is positive
        if self.weights["gradient"] > 0:
            grad = self.gradient_loss(pred, target, mask)
            total_loss += self.weights["gradient"] * grad
            loss_breakdown["gradient"] = grad.item()

        # Compute BerHu loss if its weight is positive and it's initialized
        if self.weights["berhu"] > 0 and self.berhu_loss is not None:
            berhu = self.berhu_loss(pred, target, mask)
            total_loss += self.weights["berhu"] * berhu
            loss_breakdown["berhu"] = berhu.item()

        # Check for numerical issues (NaN or Inf)
        if not torch.isfinite(total_loss):
            logger.error("MultiLoss returned a non-finite value.")
            raise RuntimeError("Loss computation resulted in NaN or Inf.")

        # Add total loss to breakdown for logging
        loss_breakdown["total"] = total_loss.item()
        logger.debug(f"MultiLoss Breakdown: {loss_breakdown}")

        # Save breakdown to history for analysis/debugging
        self.loss_history.append(loss_breakdown)

        return total_loss