# FILE: losses/factory.py
# ehsanasgharzde - Complete Loss Factory Implementation
# hosseinsolymanzadeh - PROPER COMMENTING

import logging
from typing import Type, Dict, Optional, Any, List, Callable, Union
import numpy as np
import matplotlib.pyplot as plt
import json
import torch
import torch.nn as nn

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

# Setup logger for factory operations
logger = logging.getLogger(__name__)

# Global loss registry for dynamic loss creation
LOSS_REGISTRY: Dict[str, Type[Callable]] = {}

# Loss configuration validation schemas
LOSS_CONFIG_SCHEMAS = {
    # Configuration schema for SiLogLoss with optional parameters and their defaults
    'SiLogLoss': {
        'required': [],
        'optional': {'lambda_var': float, 'epsilon': float},
        'defaults': {'lambda_var': 0.85, 'epsilon': 1e-8}
    },
    # Configuration schema for BerHuLoss
    'BerHuLoss': {
        'required': [],
        'optional': {'threshold': float, 'epsilon': float},
        'defaults': {'threshold': 0.2, 'epsilon': 1e-8}
    },
    # Configuration schema for EdgeAwareSmoothnessLoss
    'EdgeAwareSmoothnessLoss': {
        'required': [],
        'optional': {'alpha': float, 'beta': float},
        'defaults': {'alpha': 1.0, 'beta': 1.0}
    },
    # Configuration schema for GradientConsistencyLoss
    'GradientConsistencyLoss': {
        'required': [],
        'optional': {'weight_x': float, 'weight_y': float},
        'defaults': {'weight_x': 1.0, 'weight_y': 1.0}
    },
    # Configuration schema for MultiScaleLoss with scales and weights lists
    'MultiScaleLoss': {
        'required': [],
        'optional': {'scales': list, 'weights': list},
        'defaults': {'scales': [1.0, 0.5, 0.25], 'weights': [1.0, 0.5, 0.25]}
    },
    # Configuration schema for RMSELoss
    'RMSELoss': {
        'required': [],
        'optional': {'epsilon': float},
        'defaults': {'epsilon': 1e-8}
    },
    # Configuration schema for MAELoss (no optional parameters)
    'MAELoss': {
        'required': [],
        'optional': {},
        'defaults': {}
    },
    # Configuration schema for MultiLoss requiring 'losses' parameter and optional weights
    'MultiLoss': {
        'required': ['losses'],
        'optional': {'weights': list},
        'defaults': {}
    }
}

def register_loss(name: str, loss_class: Type[Callable]) -> None:
    # Register a loss function class with a name in the global registry
    if name in LOSS_REGISTRY:
        logger.warning(f"Loss function '{name}' already registered. Overwriting.")
    
    LOSS_REGISTRY[name] = loss_class
    logger.info(f"Registered loss function: {name}")

def get_registered_losses() -> List[str]:
    # Return a list of all registered loss function names
    return list(LOSS_REGISTRY.keys())

def create_loss(loss_name: str, loss_config: Optional[Dict] = None, **kwargs) -> Callable:
    # Factory function to create an instance of a registered loss function
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
        loss_instance = loss_class(**final_config) #type: ignore
        
        logger.info(f"Created loss '{loss_name}' with config: {final_config}")
        return loss_instance
        
    except Exception as e:
        logger.error(f"Failed to create loss '{loss_name}': {str(e)}")
        raise

def validate_loss_config(loss_name: str, config: Dict) -> Dict:
    # Validate the provided configuration dict against the schema for the loss_name
    if loss_name not in LOSS_CONFIG_SCHEMAS:
        logger.warning(f"No validation schema for loss '{loss_name}'. Using config as-is.")
        return config
    
    schema = LOSS_CONFIG_SCHEMAS[loss_name]
    validated_config = {}
    
    # Ensure all required parameters are present in the config
    for param in schema['required']:
        if param not in config:
            raise ValueError(f"Required parameter '{param}' missing for loss '{loss_name}'")
        validated_config[param] = config[param]
    
    # Validate and convert optional parameters, applying defaults if not provided
    for param, param_type in schema['optional'].items():
        if param in config:
            value = config[param]
            # Check and enforce type, except for lists which are not strictly type-checked here
            if not isinstance(value, param_type) and param_type != list:
                try:
                    value = param_type(value)
                except (ValueError, TypeError):
                    raise ValueError(f"Parameter '{param}' must be of type {param_type.__name__}")
            validated_config[param] = value
        elif param in schema['defaults']:
            # Apply default value if param missing
            validated_config[param] = schema['defaults'][param]
    
    # Include any extra parameters that are not specified in the schema as-is
    for param, value in config.items():
        if param not in validated_config:
            validated_config[param] = value
    
    logger.debug(f"Validated config for '{loss_name}': {validated_config}")
    return validated_config

def create_combined_loss(loss_configs: List[Dict]) -> MultiLoss:

    if not loss_configs:
        raise ValueError("At least one loss configuration is required.")

    # Initialize weights with default zeros
    weights = {
        'silog': 0.0,
        'smoothness': 0.0,
        'gradient': 0.0,
        'berhu': 0.0,
    }

    # Parse loss_configs and update weights
    for i, cfg in enumerate(loss_configs):
        if 'name' not in cfg:
            raise ValueError(f"Loss config at index {i} missing 'name' key.")
        name = cfg['name'].lower()
        if name not in weights:
            raise ValueError(f"Unsupported loss name: '{name}'. Supported: {list(weights.keys())}")
        weight = cfg.get('weight', 1.0)
        if weight < 0:
            raise ValueError(f"Weight for loss '{name}' must be non-negative.")
        weights[name] = weight  # Overwrite or set

    # Create and return MultiLoss instance with parsed weights
    combined_loss = MultiLoss(
        silog_weight=weights['silog'],
        smoothness_weight=weights['smoothness'],
        gradient_weight=weights['gradient'],
        berhu_weight=weights['berhu']
    )

    logger.info(f"Created MultiLoss with weights: {weights}")
    return combined_loss

def compute_loss_statistics(loss_values: List[float]) -> Dict[str, float]:

    if not loss_values:
        return {}
    
    loss_array = np.array(loss_values)
    
    stats = {
        'mean': float(np.mean(loss_array)),
        'std': float(np.std(loss_array)),
        'min': float(np.min(loss_array)),
        'max': float(np.max(loss_array)),
        'median': float(np.median(loss_array)),
        'count': len(loss_values)
    }
    
    # Moving average (last 10 values)
    if len(loss_values) >= 10:
        stats['moving_avg_10'] = float(np.mean(loss_array[-10:]))
    
    # Spike detection (values > 2 std from mean)
    if len(loss_values) > 1:
        mean_val = stats['mean']
        std_val = stats['std']
        spikes = np.abs(loss_array - mean_val) > 2 * std_val
        stats['spike_count'] = int(np.sum(spikes))
        stats['spike_ratio'] = float(stats['spike_count'] / len(loss_values))
    
    return stats

def get_loss_weights_schedule(epoch: int, total_epochs: int, base_weights: Dict[str, float], 
                            schedule_type: str = 'constant') -> Dict[str, float]:

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
