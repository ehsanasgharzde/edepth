# FILE: losses/factory.py
# ehsanasgharzde - Complete Loss Factory Implementation
# hosseinsolymanzadeh - PROPER COMMENTING
# ehsanasgharzde - FIXED REDUNDANT CODE BY EXTRACTING PURE FUNCTIONS AND BASECLASS LEVEL METHODS

from typing import Type, Dict, Optional, Any, List, Union
import numpy as np
import matplotlib.pyplot as plt
import json
import torch
from pathlib import Path

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

from utils.base_losses import BaseLoss, compute_loss_statistics
from configs.config import LossConfig, LossType
from logger.logger import setup_logging 

# Setup logger for factory operations
logger = setup_logging(__file__)

# Global loss registry for dynamic loss creation
LOSS_REGISTRY: Dict[str, Type[BaseLoss]] = {}

def register_loss(name: str, loss_class: Type[BaseLoss]) -> None:
    if name in LOSS_REGISTRY:
        logger.warning(f"Loss function '{name}' already registered. Overwriting.")
    
    LOSS_REGISTRY[name] = loss_class
    logger.info(f"Registered loss function: {name}")

def get_registered_losses() -> List[str]:
    return list(LOSS_REGISTRY.keys())

def validate_loss_config(loss_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    validated_config = config.copy()
    
    # Remove any None values
    validated_config = {k: v for k, v in validated_config.items() if v is not None}
    
    # Specific validation based on loss type
    if loss_name.lower() in ['silog', 'silog_loss']:
        if 'lambda_var' in validated_config:
            lambda_var = validated_config['lambda_var']
            if not 0 <= lambda_var <= 1:
                raise ValueError(f"SiLog lambda_var must be in [0,1], got {lambda_var}")
                
    elif loss_name.lower() in ['berhu', 'berhu_loss']:
        if 'threshold' in validated_config:
            threshold = validated_config['threshold']
            if threshold <= 0:
                raise ValueError(f"BerHu threshold must be positive, got {threshold}")
                
    elif loss_name.lower() in ['multi_scale', 'multiscale_loss']:
        if 'weights' in validated_config:
            weights = validated_config['weights']
            if not isinstance(weights, list) or any(w <= 0 for w in weights):
                raise ValueError("MultiScale weights must be a list of positive values")
    
    return validated_config

def create_loss_from_config(loss_config: LossConfig) -> BaseLoss:
    loss_type = loss_config.loss_type.lower()
    
    if loss_type == LossType.SILOG.value:
        return SiLogLoss(
            lambda_var=loss_config.silog_lambda,
            name="SiLogLoss"
        )
    elif loss_type == LossType.BERHU.value:
        return BerHuLoss(
            threshold=loss_config.berhu_threshold,
            name="BerHuLoss"
        )
    elif loss_type == LossType.RMSE.value:
        return RMSELoss(name="RMSELoss")
    elif loss_type == LossType.MAE.value:
        return MAELoss(name="MAELoss")
    elif loss_type == LossType.EDGE_AWARE_SMOOTHNESS.value:
        return EdgeAwareSmoothnessLoss(
            alpha=loss_config.edge_aware_weight,
            name="EdgeAwareSmoothnessLoss"
        )
    elif loss_type == LossType.GRADIENT_CONSISTENCY.value:
        return GradientConsistencyLoss(
            weight_x=loss_config.gradient_consistency_weight,
            weight_y=loss_config.gradient_consistency_weight,
            name="GradientConsistencyLoss"
        )
    elif loss_type == LossType.MULTI_SCALE.value:
        base_loss = SiLogLoss(lambda_var=loss_config.silog_lambda)
        return MultiScaleLoss(
            base_loss=base_loss,
            weights=loss_config.multi_scale_weights,
            name="MultiScaleLoss"
        )
    elif loss_type == LossType.MULTI_LOSS.value:
        # Create combined loss based on loss_weights
        loss_configs = []
        for loss_name, weight in loss_config.loss_weights.items():
            if weight > 0:
                loss_configs.append({
                    'type': loss_name,
                    'weight': weight,
                    'params': _get_loss_params_from_config(loss_name, loss_config)
                })
        return MultiLoss(loss_configs=loss_configs, name="MultiLoss")
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

def _get_loss_params_from_config(loss_name: str, loss_config: LossConfig) -> Dict[str, Any]:
    params = {}
    
    if loss_name.lower() == 'silog':
        params['lambda_var'] = loss_config.silog_lambda
    elif loss_name.lower() == 'berhu':
        params['threshold'] = loss_config.berhu_threshold
    elif loss_name.lower() == 'edge_aware_smoothness':
        params['alpha'] = loss_config.edge_aware_weight
    elif loss_name.lower() == 'gradient_consistency':
        params['weight_x'] = loss_config.gradient_consistency_weight
        params['weight_y'] = loss_config.gradient_consistency_weight
    
    return params

def create_loss(loss_name: str, loss_config: Optional[Dict] = None, **kwargs) -> BaseLoss:
    if loss_name not in LOSS_REGISTRY:
        available_losses = get_registered_losses()
        raise ValueError(f"Loss '{loss_name}' not found in registry. Available losses: {available_losses}")
    
    # Merge the provided config dict with additional kwargs (kwargs override config)
    config = loss_config or {}
    final_config = {**config, **kwargs}
    
    # Validate the final configuration
    final_config = validate_loss_config(loss_name, final_config)
    
    # Instantiate the loss class with validated config
    try:
        loss_class = LOSS_REGISTRY[loss_name]
        loss_instance = loss_class(**final_config)
        
        logger.info(f"Created loss '{loss_name}' with config: {final_config}")
        return loss_instance
        
    except Exception as e:
        logger.error(f"Failed to create loss '{loss_name}': {str(e)}")
        raise

def create_combined_loss(loss_configs: List[Dict[str, Any]]) -> MultiLoss:
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
                            save_path: Optional[Union[str, Path]] = None) -> None:
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

def export_loss_configuration(loss_instance: BaseLoss, export_path: Union[str, Path]) -> None:
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
    
    export_path = Path(export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(export_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Loss configuration exported to {export_path}")

def integrate_with_trainer(trainer: Any, loss_config: Union[LossConfig, Dict[str, Any]]) -> None:
    # Handle different input types
    if isinstance(loss_config, LossConfig):
        loss_fn = create_loss_from_config(loss_config)
    elif isinstance(loss_config, dict):
        if 'combined' in loss_config:
            loss_fn = create_combined_loss(loss_config['combined'])
        else:
            loss_name = loss_config['name']
            config = loss_config.get('config', {})
            loss_fn = create_loss(loss_name, config)
    else:
        raise ValueError(f"Unsupported loss_config type: {type(loss_config)}")
    
    # Set loss function in trainer
    trainer.loss_fn = loss_fn
    
    # Setup logging hooks if trainer supports it
    if hasattr(trainer, 'add_hook'):
        def loss_logging_hook(epoch, metrics):
            loss_stats = compute_loss_statistics(metrics.get('loss_values', []))
            logger.info(f"Epoch {epoch} loss stats: {loss_stats}")
        
        trainer.add_hook('epoch_end', loss_logging_hook)
    
    # Export configuration if requested
    if isinstance(loss_config, dict) and loss_config.get('export_config'):
        export_path = loss_config.get('export_path', 'loss_config.json')
        export_loss_configuration(loss_fn, export_path)
    
    logger.info("Loss integration with trainer completed")

def create_loss_factory() -> 'LossFactory':
    return LossFactory()

class LossFactory:
    
    def __init__(self):
        self.registry = LOSS_REGISTRY.copy()
    
    def create(self, loss_config: Union[str, LossConfig, Dict[str, Any]], **kwargs) -> BaseLoss:
        if isinstance(loss_config, str):
            # Simple string name
            return create_loss(loss_config, **kwargs)
        elif isinstance(loss_config, LossConfig):
            # LossConfig dataclass
            return create_loss_from_config(loss_config)
        elif isinstance(loss_config, dict):
            # Dictionary configuration
            if 'type' in loss_config or 'name' in loss_config:
                loss_name = loss_config.get('type', loss_config.get('name'))
                config = loss_config.get('params', loss_config.get('config', {}))
                return create_loss(loss_name, config, **kwargs)
            else:
                raise ValueError("Dictionary config must contain 'type' or 'name' key")
        else:
            raise ValueError(f"Unsupported loss_config type: {type(loss_config)}")
    
    def create_multi(self, loss_configs: List[Dict[str, Any]]) -> MultiLoss:
        return create_combined_loss(loss_configs)
    
    def get_available_losses(self) -> List[str]:
        return list(self.registry.keys())
    
    def register(self, name: str, loss_class: Type[BaseLoss]) -> None:
        register_loss(name, loss_class)
        self.registry[name] = loss_class

# Register all implemented loss functions
register_loss('silog', SiLogLoss)
register_loss('silog_loss', SiLogLoss)
register_loss('edge_aware_smoothness', EdgeAwareSmoothnessLoss)
register_loss('smoothness', EdgeAwareSmoothnessLoss)
register_loss('gradient_consistency', GradientConsistencyLoss)
register_loss('gradient', GradientConsistencyLoss)
register_loss('multi_scale', MultiScaleLoss)
register_loss('multiscale', MultiScaleLoss)
register_loss('berhu', BerHuLoss)
register_loss('berhu_loss', BerHuLoss)
register_loss('rmse', RMSELoss)
register_loss('rmse_loss', RMSELoss)
register_loss('mae', MAELoss)
register_loss('mae_loss', MAELoss)
register_loss('multi_loss', MultiLoss)
register_loss('multi', MultiLoss)

logger.info("All loss functions registered successfully")
logger.info(f"Available losses: {get_registered_losses()}")