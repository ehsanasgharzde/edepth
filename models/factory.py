# FILE: models/factory.py
# ehsanasgharzde - DYNAMIC MODEL CREATION AND CONFIGURATION VALIDATION
# ehsanasgharzde - FIXED REDUNDANT CODE BY EXTRACTING PURE FUNCTINOS AND BASECLASS LEVEL METHODS

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any, Type
import os
import yaml
import logging
import traceback

# Import centralized utilities
from .edepth import edepth
from ..configs.model_config import (
    get_model_config, list_available_models, validate_config,
    get_backbone_config, list_available_backbones
)
from ..utils.model_validation import ( 
    validate_backbone_name, ConfigValidationError,
    validate_decoder_channels
)
from ..utils.model_utils import ( # type: ignore
    get_model_info, ModelInfo, load_model_checkpoint,
    save_model_checkpoint
)

logger = logging.getLogger(__name__)

# Registry for custom models
MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {}

def get_available_models() -> List[str]:
    built_in_models = list_available_models()
    registered_models = list(MODEL_REGISTRY.keys())
    return built_in_models + registered_models

def create_model(
    model_name: str,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> nn.Module:
    try:
        # Validate model name using centralized validation
        available_models = get_available_models()
        validate_backbone_name(model_name, available_models, "model_name")

        # Get base configuration using centralized config
        if config is None:
            if model_name in MODEL_CONFIGS:
                # Use registered model config
                base_config = MODEL_CONFIGS[model_name].copy()
            else:
                # Use built-in model config
                base_config = get_model_config(model_name)
        else:
            base_config = config.copy()

        # Override with kwargs
        base_config.update(kwargs)

        # Validate configuration using centralized validation
        validate_model_config(base_config)

        # Create model instance
        if model_name in MODEL_REGISTRY:
            # Create registered custom model
            model_class = MODEL_REGISTRY[model_name]
            model = model_class(**base_config)
        else:
            # Create built-in edepth model
            model = edepth.from_config(base_config)

        logger.info(f"Successfully created model '{model_name}'")
        return model

    except Exception as e:
        logger.error(f"Failed to create model '{model_name}': {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise RuntimeError(f"Model creation failed: {e}")

def validate_model_config(config: Dict[str, Any]) -> None:
    try:
        # Validate backbone configuration if present
        if 'backbone_name' in config:
            available_backbones = list_available_backbones()
            validate_backbone_name(
                config['backbone_name'], 
                available_backbones, 
                "backbone_name"
            )

        # Validate decoder channels if present
        if 'decoder_channels' in config:
            validate_decoder_channels(
                config['decoder_channels'],
                name="decoder_channels"
            )

        # Validate using centralized config validation
        validate_config(config, config_type='backbone')
        validate_config(config, config_type='decoder')

    except Exception as e:
        raise ConfigValidationError(f"Model configuration validation failed: {e}")

def load_config_from_file(config_path: str) -> Dict[str, Any]:
    if not os.path.isfile(config_path):
        raise ConfigValidationError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r') as f:
            if config_path.endswith(('.yaml', '.yml')):
                config = yaml.safe_load(f)
            else:
                raise ConfigValidationError(
                    "Unsupported config file format. Use YAML (.yaml/.yml)"
                )

        logger.info(f"Loaded configuration from {config_path}")
        return config

    except Exception as e:
        raise ConfigValidationError(f"Failed to load config file: {e}")

def register_model(
    name: str, 
    model_class: Type[nn.Module], 
    default_config: Optional[Dict[str, Any]] = None
) -> None:
    # Validate model class
    if not issubclass(model_class, nn.Module):
        raise ValueError(f"Model class must be a subclass of nn.Module")

    if not hasattr(model_class, 'forward') or not callable(getattr(model_class, 'forward')):
        raise ValueError(f"Model class must have a callable 'forward' method")

    # Register model
    MODEL_REGISTRY[name] = model_class

    if default_config is not None:
        MODEL_CONFIGS[name] = default_config

    logger.info(f"Registered model '{name}' with class {model_class.__name__}")

def create_model_from_checkpoint(
    checkpoint_path: str,
    model_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    map_location: Optional[str] = None,
    strict: bool = True
) -> nn.Module:
    if not os.path.isfile(checkpoint_path):
        raise RuntimeError(f"Checkpoint file not found: {checkpoint_path}")

    try:
        # Load checkpoint metadata first
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Extract model information from checkpoint
        checkpoint_model_name = checkpoint.get('metadata', {}).get('backbone_name')
        checkpoint_config = checkpoint.get('metadata', {}).get('model_config')

        # Determine model name and config
        final_model_name = model_name or checkpoint_model_name
        final_config = config or checkpoint_config

        if final_model_name is None:
            raise RuntimeError("Model name not provided and not found in checkpoint")

        # Create model
        model = create_model(final_model_name, final_config)

        # Load checkpoint using centralized utility
        metadata = load_model_checkpoint(
            model=model,
            checkpoint_path=checkpoint_path,
            map_location=map_location,
            strict=strict
        )

        logger.info(f"Model loaded from checkpoint: {checkpoint_path}")
        return model

    except Exception as e:
        logger.error(f"Failed to load model from checkpoint: {e}")
        raise RuntimeError(f"Checkpoint loading failed: {e}")

class ModelBuilder:
    def __init__(self):
        self._backbone_name: Optional[str] = None
        self._config: Dict[str, Any] = {}
        self._built: bool = False

    def backbone(self, backbone_name: str) -> 'ModelBuilder':
        available_backbones = list_available_backbones()
        validate_backbone_name(backbone_name, available_backbones, "backbone_name")

        self._backbone_name = backbone_name

        # Load default config for backbone
        try:
            default_config = get_model_config(backbone_name)
            self._config.update(default_config)
        except:
            # Fallback to backbone config only
            backbone_config = get_backbone_config(backbone_name)
            self._config.update(backbone_config)

        return self

    def decoder(
        self, 
        channels: List[int], 
        use_attention: bool = False,
        final_activation: str = 'sigmoid'
    ) -> 'ModelBuilder':
        validate_decoder_channels(channels, name="decoder_channels")

        self._config.update({
            'decoder_channels': channels,
            'use_attention': use_attention,
            'final_activation': final_activation
        })
        return self

    def training_config(
        self,
        use_checkpointing: bool = False,
        pretrained: bool = True
    ) -> 'ModelBuilder':
        self._config.update({
            'use_checkpointing': use_checkpointing,
            'pretrained': pretrained
        })
        return self

    def build(self) -> nn.Module:
        if self._built:
            raise RuntimeError("Model has already been built")

        if not self._backbone_name:
            raise RuntimeError("Backbone must be specified")

        # Validate complete configuration
        validate_model_config(self._config)

        # Create model using centralized factory
        model = create_model(self._backbone_name, self._config)
        self._built = True

        logger.info(f"Model built with backbone '{self._backbone_name}'")
        return model

# Convenience functions using centralized utilities
def get_model_info_summary(model: nn.Module) -> Dict[str, Any]:
    model_info = get_model_info(model)
    return model_info.get_summary()

def estimate_model_parameters(model_name: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
    try:
        # Get configuration
        if config is None:
            config = get_model_config(model_name)

        # Estimate based on configuration
        embed_dim = config.get('embed_dim', 768) # type: ignore
        num_layers = config.get('num_layers', 12) # type: ignore
        decoder_channels = config.get('decoder_channels', [256, 512, 768, 768]) # type: ignore

        # Rough estimation formulas
        backbone_params = embed_dim * embed_dim * num_layers * 4  # Simplified ViT estimation
        decoder_params = sum(ch * ch for ch in decoder_channels) * 2  # Simplified DPT estimation

        return {
            'backbone_parameters': backbone_params,
            'decoder_parameters': decoder_params,
            'total_parameters': backbone_params + decoder_params
        }

    except Exception as e:
        logger.warning(f"Parameter estimation failed: {e}")
        return {'error': str(e)} # type: ignore

# Register the default edepth model
register_model('edepth', edepth, get_model_config('vit_base_patch16_224'))

# Export main functions
__all__ = [
    'create_model',
    'validate_model_config', 
    'load_config_from_file',
    'register_model',
    'create_model_from_checkpoint',
    'ModelBuilder',
    'get_model_info_summary',
    'estimate_model_parameters',
    'get_available_models'
]