import torch
import torch.nn as nn
import torch.nn.functional as F

class SiLogLoss(nn.Module):
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps
    def forward(self, pred, target, mask=None):
        if mask is None:
            mask = (target > self.eps) & torch.isfinite(target)
        pred = pred[mask]
        target = target[mask]
        log_diff = torch.log(pred + self.eps) - torch.log(target + self.eps)
        silog = torch.sqrt(torch.mean(log_diff ** 2) - 0.85 * torch.mean(log_diff) ** 2)
        return silog

class EdgeAwareSmoothnessLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, image):
        pred_dx = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        pred_dy = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
        img_dx = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]), 1, keepdim=True)
        img_dy = torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]), 1, keepdim=True)
        weight_x = torch.exp(-img_dx)
        weight_y = torch.exp(-img_dy)
        smoothness_x = pred_dx * weight_x
        smoothness_y = pred_dy * weight_y
        return (smoothness_x.mean() + smoothness_y.mean())

class GradientConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, target, mask=None):
        if mask is None:
            mask = (target > 0) & torch.isfinite(target)
        pred_dx = pred[:, :, :, :-1] - pred[:, :, :, 1:]
        pred_dy = pred[:, :, :-1, :] - pred[:, :, 1:, :]
        target_dx = target[:, :, :, :-1] - target[:, :, :, 1:]
        target_dy = target[:, :, :-1, :] - target[:, :, 1:, :]
        loss_x = torch.abs(pred_dx - target_dx)[mask[:, :, :, :-1]]
        loss_y = torch.abs(pred_dy - target_dy)[mask[:, :, :-1, :]]
        return (loss_x.mean() + loss_y.mean())

class MultiScaleLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, *args, **kwargs):
        # Placeholder for future multi-scale loss
        raise NotImplementedError('Multi-scale loss not implemented yet.') 