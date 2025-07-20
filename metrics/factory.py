# FILE: metrics/factory.py
#hosseinsolymanzadeh - FIXED REDUNDANT CODE BY EXTRACTING PURE FUNCTINOS AND BASECLASS LEVEL METHODS

from typing import Dict, Callable, List, Optional
import logging

from .metrics import (
    rmse, mae, delta1, delta2, delta3, silog)

logger = logging.getLogger(__name__)

# Core evaluation metrics
METRICS = {
    'rmse': rmse,
    'mae': mae,
    'delta1': delta1,
    'delta2': delta2,
    'delta3': delta3,
    'silog': silog
}

def get_metric(name: str) -> Callable:
    # Get a metric function by name.
    if name not in METRICS:
        available = ', '.join(METRICS.keys())
        raise ValueError(f"Unknown metric '{name}'. Available: {available}")
    return METRICS[name]

def get_all_metrics() -> Dict[str, Callable]:
    # Get all available metrics and utilities.
    return METRICS.copy()

def get_core_metrics() -> Dict[str, Callable]:
    # Get only the core evaluation metrics.
    return METRICS.copy()

def list_metrics() -> List[str]:
    # List core metric names only.
    return list(METRICS.keys())

def create_evaluator(metric_names: Optional[List[str]] = None) -> Callable:
    if metric_names is None:
        metric_names = list(METRICS.keys())
    
    # Validate metric names
    for name in metric_names:
        if name not in METRICS:
            available = ', '.join(METRICS.keys())
            raise ValueError(f"Unknown metric '{name}'. Available: {available}")
    
    def evaluator(pred, target, mask=None):
        return {name: METRICS[name](pred, target, mask) for name in metric_names}
    
    return evaluator
