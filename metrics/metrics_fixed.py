import torch

def rmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> float:
    if mask is None:
        mask = (target > 0) & torch.isfinite(target)
    pred = pred[mask]
    target = target[mask]
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()

def mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> float:
    if mask is None:
        mask = (target > 0) & torch.isfinite(target)
    pred = pred[mask]
    target = target[mask]
    return torch.mean(torch.abs(pred - target)).item()

def delta_metric(pred: torch.Tensor, target: torch.Tensor, threshold: float, mask: torch.Tensor = None) -> float:
    eps = 1e-7
    if mask is None:
        mask = (target > eps) & (pred > eps) & torch.isfinite(target) & torch.isfinite(pred)
    pred = pred[mask]
    target = target[mask]
    ratio = torch.max(pred / (target + eps), target / (pred + eps))
    valid = (ratio < threshold)
    return valid.float().mean().item()

def delta1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> float:
    return delta_metric(pred, target, 1.25, mask)

def delta2(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> float:
    return delta_metric(pred, target, 1.25 ** 2, mask)

def delta3(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> float:
    return delta_metric(pred, target, 1.25 ** 3, mask)

def silog(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> float:
    eps = 1e-7
    if mask is None:
        mask = (target > eps) & torch.isfinite(target)
    pred = pred[mask]
    target = target[mask]
    log_diff = torch.log(pred + eps) - torch.log(target + eps)
    return torch.sqrt(torch.mean(log_diff ** 2) - 0.85 * torch.mean(log_diff) ** 2).item() 