# FILE: losses/factory.py
# ehsanasgharzde - Complete Loss Factory Implementation
# hosseinsolymanzadeh - PROPER COMMENTING
# ehsanasgharzde - FIXED REDUNDANT CODE BY EXTRACTING PURE FUNCTIONS AND BASECLASS LEVEL METHODS

import logging
from typing import Type, Dict, Optional, Any, List, Callable
import numpy as np
import matplotlib.pyplot as plt
import json
import torch

# Import loss classes
from losses.losses import (
    SiLogLoss,
    EdgeAwareSmoothnessLoss,
    GradientConsistencyLoss,
    MultiScaleLoss,
    BerHuLoss,
    RMSELoss,
    MAELoss,
    MultiLoss,
)

from utils.base_losses import compute_loss_statistics

# Setup logger for factory operations
logger = logging.getLogger(__name__)

# Global loss registry for dynamic loss creation
LOSS_REGISTRY: Dict[str, Type[Callable]] = {}


def register_loss(name: str, loss_class: Type[Callable]) -> None:
    if name in LOSS_REGISTRY:
        logger.warning(f"Loss function '{name}' already registered. Overwriting.")
    
    LOSS_REGISTRY[name] = loss_class
    logger.info(f"Registered loss function: {name}")


def get_registered_losses() -> List[str]:
    return list(LOSS_REGISTRY.keys())


def create_loss(loss_name: str, loss_config: Optional[Dict] = None, **kwargs) -> Callable:
    if loss_name not in LOSS_REGISTRY:
        available_losses = get_registered_losses()
        raise ValueError(f"Loss '{loss_name}' not found in registry. Available losses: {available_losses}")
    
    # Merge the provided config dict with additional kwargs (kwargs override config)
    config = loss_config or {}
    final_config = {**config, **kwargs}
    
    # Validate the final configuration according to the schema
    final_config = validate_loss_config(loss_name, final_config)
    
    # Instantiate the loss class with validated config
    try:
        loss_class = LOSS_REGISTRY[loss_name]
        loss_instance = loss_class(**final_config) # type: ignore
        
        logger.info(f"Created loss '{loss_name}' with config: {final_config}")
        return loss_instance
        
    except Exception as e:
        logger.error(f"Failed to create loss '{loss_name}': {str(e)}")
        raise


def create_combined_loss(loss_configs: List[Dict]) -> MultiLoss:
    if not loss_configs:
        raise ValueError("At least one loss configuration is required.")
    
    # Validate each loss config and convert to MultiLoss format
    multiloss_configs = []
    
    for i, cfg in enumerate(loss_configs):
        if 'name' not in cfg and 'type' not in cfg:
            raise ValueError(f"Loss config at index {i} missing 'name' or 'type' key.")
        
        # Convert 'name' to 'type' for consistency
        loss_type = cfg.get('type', cfg.get('name', '')).lower()
        weight = cfg.get('weight', 1.0)
        params = cfg.get('params', cfg.get('config', {}))
        
        if weight < 0:
            raise ValueError(f"Weight for loss '{loss_type}' must be non-negative.")
        
        multiloss_config = {
            'type': loss_type,
            'weight': weight,
            'params': params
        }
        multiloss_configs.append(multiloss_config)
    
    # Create and return MultiLoss instance
    combined_loss = MultiLoss(loss_configs=multiloss_configs)
    
    logger.info(f"Created MultiLoss with {len(multiloss_configs)} component losses")
    return combined_loss


def get_loss_weights_schedule(epoch: int, total_epochs: int, base_weights: Dict[str, float], 
                            schedule_type: str = 'constant') -> Dict[str, float]:
    """Generate weight schedule for different loss components during training"""
    if schedule_type == 'constant':
        return base_weights.copy()
    
    progress = epoch / total_epochs
    adjusted_weights = {}
    
    for loss_name, base_weight in base_weights.items():
        if schedule_type == 'linear':
            # Linear decay
            factor = 1.0 - 0.5 * progress
        elif schedule_type == 'exponential':
            # Exponential decay
            factor = np.exp(-2 * progress)
        elif schedule_type == 'cosine':
            # Cosine annealing
            factor = 0.5 * (1 + np.cos(np.pi * progress))
        else:
            factor = 1.0
        
        adjusted_weights[loss_name] = base_weight * factor
    
    logger.debug(f"Epoch {epoch}: adjusted weights = {adjusted_weights}")
    return adjusted_weights


def visualize_loss_components(loss_history: Dict[str, List[float]], 
                            save_path: Optional[str] = None) -> None:
    """Visualize loss component history over training"""
    if not loss_history:
        logger.warning("No loss history to visualize")
        return
    
    plt.figure(figsize=(12, 8))
    
    for loss_name, values in loss_history.items():
        if values:
            epochs = range(len(values))
            plt.plot(epochs, values, label=loss_name, marker='o', markersize=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('Loss Components Over Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Loss visualization saved to {save_path}")
    
    plt.show()


def export_loss_configuration(loss_instance: Callable, export_path: str) -> None:
    """Export loss configuration to JSON file"""
    config = {
        'class_name': loss_instance.__class__.__name__,
        'parameters': {},
        'metadata': {
            'created_at': str(np.datetime64('now')),
            'registry_available': get_registered_losses()
        }
    }
    
    # Extract parameters from loss instance
    for param_name, param_value in loss_instance.__dict__.items():
        if not param_name.startswith('_'):
            try:
                # Convert to JSON-serializable format
                if isinstance(param_value, (int, float, str, bool, list, dict)):
                    config['parameters'][param_name] = param_value
                elif isinstance(param_value, torch.Tensor):
                    config['parameters'][param_name] = param_value.tolist()
                else:
                    config['parameters'][param_name] = str(param_value)
            except Exception as e:
                logger.warning(f"Could not serialize parameter '{param_name}': {e}")
    
    with open(export_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Loss configuration exported to {export_path}")


def integrate_with_trainer(trainer: Any, loss_config: Dict) -> None:
    """Integrate loss function with trainer instance"""
    # Create loss function
    if 'combined' in loss_config:
        loss_fn = create_combined_loss(loss_config['combined'])
    else:
        loss_name = loss_config['name']
        config = loss_config.get('config', {})
        loss_fn = create_loss(loss_name, config)
    
    # Set loss function in trainer
    trainer.loss_fn = loss_fn
    
    # Setup logging hooks if trainer supports it
    if hasattr(trainer, 'add_hook'):
        def loss_logging_hook(epoch, metrics):
            loss_stats = compute_loss_statistics(metrics.get('loss_values', []))
            logger.info(f"Epoch {epoch} loss stats: {loss_stats}")
        
        trainer.add_hook('epoch_end', loss_logging_hook)
    
    # Export configuration if requested
    if loss_config.get('export_config'):
        export_path = loss_config.get('export_path', 'loss_config.json')
        export_loss_configuration(loss_fn, export_path)
    
    logger.info("Loss integration with trainer completed")


# Register all implemented loss functions
register_loss('SiLogLoss', SiLogLoss)
register_loss('EdgeAwareSmoothnessLoss', EdgeAwareSmoothnessLoss)
register_loss('GradientConsistencyLoss', GradientConsistencyLoss)
register_loss('MultiScaleLoss', MultiScaleLoss)
register_loss('BerHuLoss', BerHuLoss)
register_loss('RMSELoss', RMSELoss)
register_loss('MAELoss', MAELoss)
register_loss('MultiLoss', MultiLoss)

# Register short aliases for convenience
register_loss('silog', SiLogLoss)
register_loss('smoothness', EdgeAwareSmoothnessLoss)
register_loss('gradient', GradientConsistencyLoss)
register_loss('multiscale', MultiScaleLoss)
register_loss('berhu', BerHuLoss)
register_loss('rmse', RMSELoss)
register_loss('mae', MAELoss)
register_loss('multi', MultiLoss)

logger.info("All loss functions registered successfully")
logger.info(f"Available losses: {get_registered_losses()}")