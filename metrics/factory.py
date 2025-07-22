# FILE: metrics/factory.py
# hosseinsolymanzadeh - FIXED REDUNDANT CODE BY EXTRACTING PURE FUNCTIONS AND BASECLASS LEVEL METHODS

import logging
from typing import Dict, Callable, List, Optional

from .metrics import METRICS

logger = logging.getLogger(__name__)

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

