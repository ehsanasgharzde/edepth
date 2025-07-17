# FILE: models/factory.py
# ehsanasgharzde - DYNAMIC MODEL CREATION AND CONFIGURATION VALIDATION

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any, Type
import os
import json
import yaml
import logging
import traceback
from models.edepth import edepth

logger = logging.getLogger(__name__)

MODEL_CONFIGS = {
    "vit_small_patch16_224": {
        "backbone_name": "vit_small_patch16_224",
        "patch_size": 16,
        "img_size": 224,
        "embed_dim": 384,
        "num_heads": 6,
        "num_layers": 12,
        "extract_layers": [8, 9, 10, 11],
        "decoder_channels": [128, 256, 384, 384],
        "pretrained": True,
        "use_attention": False,
        "final_activation": "sigmoid",
        "interpolation_mode": "bilinear",
        "num_classes": 1,
        "dropout": 0.0,
        "drop_path": 0.1,
        "memory_efficient": False,
        "gradient_checkpointing": False
    },
    "vit_base_patch16_224": {
        "backbone_name": "vit_base_patch16_224",
        "patch_size": 16,
        "img_size": 224,
        "embed_dim": 768,
        "num_heads": 12,
        "num_layers": 12,
        "extract_layers": [8, 9, 10, 11],
        "decoder_channels": [256, 512, 768, 768],
        "pretrained": True,
        "use_attention": False,
        "final_activation": "sigmoid",
        "interpolation_mode": "bilinear",
        "num_classes": 1,
        "dropout": 0.0,
        "drop_path": 0.1,
        "memory_efficient": False,
        "gradient_checkpointing": False
    },
    "vit_base_patch16_384": {
        "backbone_name": "vit_base_patch16_384",
        "patch_size": 16,
        "img_size": 384,
        "embed_dim": 768,
        "num_heads": 12,
        "num_layers": 12,
        "extract_layers": [8, 9, 10, 11],
        "decoder_channels": [256, 512, 768, 768],
        "pretrained": True,
        "use_attention": False,
        "final_activation": "sigmoid",
        "interpolation_mode": "bilinear",
        "num_classes": 1,
        "dropout": 0.0,
        "drop_path": 0.1,
        "memory_efficient": True,
        "gradient_checkpointing": True
    },
    "vit_base_patch8_224": {
        "backbone_name": "vit_base_patch8_224",
        "patch_size": 8,
        "img_size": 224,
        "embed_dim": 768,
        "num_heads": 12,
        "num_layers": 12,
        "extract_layers": [8, 9, 10, 11],
        "decoder_channels": [256, 512, 768, 768],
        "pretrained": True,
        "use_attention": True,
        "final_activation": "sigmoid",
        "interpolation_mode": "bilinear",
        "num_classes": 1,
        "dropout": 0.0,
        "drop_path": 0.1,
        "memory_efficient": True,
        "gradient_checkpointing": True
    },
    "vit_large_patch16_224": {
        "backbone_name": "vit_large_patch16_224",
        "patch_size": 16,
        "img_size": 224,
        "embed_dim": 1024,
        "num_heads": 16,
        "num_layers": 24,
        "extract_layers": [20, 21, 22, 23],
        "decoder_channels": [512, 768, 1024, 1024],
        "pretrained": True,
        "use_attention": True,
        "final_activation": "sigmoid",
        "interpolation_mode": "bilinear",
        "num_classes": 1,
        "dropout": 0.0,
        "drop_path": 0.2,
        "memory_efficient": True,
        "gradient_checkpointing": True
    },
    "deit_small_patch16_224": {
        "backbone_name": "deit_small_patch16_224",
        "patch_size": 16,
        "img_size": 224,
        "embed_dim": 384,
        "num_heads": 6,
        "num_layers": 12,
        "extract_layers": [8, 9, 10, 11],
        "decoder_channels": [128, 256, 384, 384],
        "pretrained": True,
        "use_attention": False,
        "final_activation": "sigmoid",
        "interpolation_mode": "bilinear",
        "num_classes": 1,
        "dropout": 0.0,
        "drop_path": 0.1,
        "memory_efficient": False,
        "gradient_checkpointing": False
    },
    "deit_base_patch16_224": {
        "backbone_name": "deit_base_patch16_224",
        "patch_size": 16,
        "img_size": 224,
        "embed_dim": 768,
        "num_heads": 12,
        "num_layers": 12,
        "extract_layers": [8, 9, 10, 11],
        "decoder_channels": [256, 512, 768, 768],
        "pretrained": True,
        "use_attention": False,
        "final_activation": "sigmoid",
        "interpolation_mode": "bilinear",
        "num_classes": 1,
        "dropout": 0.0,
        "drop_path": 0.1,
        "memory_efficient": False,
        "gradient_checkpointing": False
    }
}

MODEL_REGISTRY = {}

CONFIG_SCHEMAS = {
    'backbone': {
        'required': ['backbone_name'],
        'optional': ['extract_layers', 'pretrained', 'patch_size', 'img_size', 'use_checkpointing'],
        'validation': {
            'backbone_name': lambda x: x in MODEL_CONFIGS,
            'patch_size': lambda x: x in [8, 16, 32],
            'img_size': lambda x: x > 0 and x % 32 == 0,
            'extract_layers': lambda x: isinstance(x, list) and len(x) > 0,
        }
    },
    'decoder': {
        'required': ['decoder_channels'],
        'optional': ['use_attention', 'final_activation', 'interpolation_mode'],
        'validation': {
            'decoder_channels': lambda x: isinstance(x, list) and all(isinstance(c, int) and c > 0 for c in x),
            'final_activation': lambda x: x in ['sigmoid', 'tanh', 'relu', 'none'],
            'interpolation_mode': lambda x: x in ['bilinear', 'nearest', 'bicubic']
        }
    }
}

def get_model_config(backbone_name: str, **kwargs) -> Dict[str, Any]:
    # Check if the requested backbone name exists in predefined model configurations
    if backbone_name not in MODEL_CONFIGS:
        # List all available backbone names for user reference
        available_backbones = list(MODEL_CONFIGS.keys())
        # Raise error if backbone_name is invalid
        raise ValueError(f"Backbone '{backbone_name}' not found. Available backbones: {available_backbones}")
    
    # Copy the base config to avoid mutating the original
    config = MODEL_CONFIGS[backbone_name].copy()
    # Update the copied config with any provided overrides in kwargs
    config.update(kwargs)
    
    # Log info about the config retrieval and applied overrides
    logger.info(f"Retrieved model config for '{backbone_name}' with overrides: {kwargs}")
    # Return the final configuration dictionary
    return config

def create_model(model_name: str = None, config: Optional[Dict[str, Any]] = None, **kwargs) -> nn.Module: #type: ignore
    # Use provided config or initialize empty dictionary
    config = config or {}
    # Update config with any additional keyword arguments passed
    config.update(kwargs)
    
    # Determine backbone name either from config or directly from model_name
    backbone_name = config.get('backbone_name', model_name)
    # Raise error if no backbone name is specified
    if not backbone_name:
        raise ValueError("backbone_name must be specified in config or as model_name")
    
    # Retrieve full model configuration using backbone name and config overrides
    final_config = get_model_config(backbone_name, **config)
    
    # Validate the final model configuration before model creation
    validate_model_config(final_config)
    
    # Log the start of model creation process
    logger.info(f"Creating model with backbone '{backbone_name}'")
    
    try:
        # Instantiate the model using the edepth class and final configuration
        model = edepth(**final_config)
        # Log successful creation with the parameter count
        logger.info(f"Model created successfully with {model.count_parameters():,} parameters")
        # Return the constructed model instance
        return model
    except Exception as e:
        # Log error message and traceback if model creation fails
        logger.error(f"Failed to create model: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Raise runtime error with failure details
        raise RuntimeError(f"Model creation failed: {str(e)}")

def validate_model_config(config: Dict[str, Any]) -> bool:
    # Initialize list to collect validation errors
    errors = []
    
    # Retrieve schema definitions for backbone and decoder components
    backbone_schema = CONFIG_SCHEMAS.get('backbone', {})
    decoder_schema = CONFIG_SCHEMAS.get('decoder', {})
    
    # Check required fields in backbone schema are present in config
    for field in backbone_schema.get('required', []):
        if field not in config:
            errors.append(f"Missing required field: '{field}'")
    
    # Check required fields in decoder schema are present in config
    for field in decoder_schema.get('required', []):
        if field not in config:
            errors.append(f"Missing required field: '{field}'")
    
    # Merge validation rules from both backbone and decoder schemas
    all_validations = {**backbone_schema.get('validation', {}), **decoder_schema.get('validation', {})}
    
    # For each validation rule, apply the validator if key exists in config
    for key, validator in all_validations.items():
        if key in config:
            try:
                # If validation fails, record an error message
                if not validator(config[key]):
                    errors.append(f"Invalid value for '{key}': {config[key]}")
            except Exception as e:
                # Record any exceptions raised during validation
                errors.append(f"Validation error for '{key}': {str(e)}")
    
    # If any errors were collected, log them and raise a ValueError
    if errors:
        error_msg = "\n".join(errors)
        logger.error(f"Model config validation failed:\n{error_msg}")
        raise ValueError(f"Model config validation failed:\n{error_msg}")
    
    # Return True if all validations pass successfully
    return True

def load_config_from_file(config_path: str) -> Dict[str, Any]:
    # Check if the config file exists
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        # Open the config file for reading
        with open(config_path, 'r') as f:
            # Parse YAML files
            if config_path.lower().endswith(('.yaml', '.yml')):
                config = yaml.safe_load(f)
            # Parse JSON files
            elif config_path.lower().endswith('.json'):
                config = json.load(f)
            # Raise error for unsupported file formats
            else:
                raise ValueError("Unsupported config file format. Use YAML (.yaml/.yml) or JSON (.json).")
    except Exception as e:
        # Log and raise error if parsing fails
        logger.error(f"Failed to parse config file '{config_path}': {str(e)}")
        raise RuntimeError(f"Failed to parse config file '{config_path}': {str(e)}")
    
    # Validate the loaded configuration
    validate_model_config(config)
    
    # Log successful load and validation
    logger.info(f"Loaded and validated config from '{config_path}'")
    
    # Return the validated configuration dictionary
    return config

def get_model_info(backbone_name: str) -> Dict[str, Any]:
    # Check if the backbone_name exists in the model configurations
    if backbone_name not in MODEL_CONFIGS:
        raise KeyError(f"Backbone '{backbone_name}' not found in configurations.")
    
    # Retrieve the configuration dictionary for the given backbone
    config = MODEL_CONFIGS[backbone_name]
    
    # Prepare the info dictionary with relevant architecture and config details
    info = {
        "backbone_name": backbone_name,
        "architecture": "DPT with Vision Transformer",
        "patch_size": config.get('patch_size'),               # Size of input patch for ViT
        "img_size": config.get('img_size'),                   # Expected input image size
        "embed_dim": config.get('embed_dim'),                 # Embedding dimension of backbone
        "num_layers": config.get('num_layers'),               # Number of transformer layers
        "extract_layers": config.get('extract_layers'),       # Backbone layers for feature extraction
        "decoder_channels": config.get('decoder_channels'),   # Channels used in decoder
        "memory_efficient": config.get('memory_efficient', False),       # Memory optimization flag
        "gradient_checkpointing": config.get('gradient_checkpointing', False), # Gradient checkpoint flag
        "estimated_params": _estimate_parameters(config)      # Estimated parameter count helper
    }
    
    # Return the detailed model information dictionary
    return info

def _estimate_parameters(config: Dict[str, Any]) -> str:
    # Retrieve embedding dimension, defaulting to 768 if not provided
    embed_dim = config.get('embed_dim', 768)
    # Retrieve number of transformer layers, default 12
    num_layers = config.get('num_layers', 12)
    # Retrieve decoder channel sizes, default list if not provided
    decoder_channels = config.get('decoder_channels', [256, 512, 768, 768])
    
    # Estimate backbone parameters as embed_dim^2 * num_layers * 4 (approximation)
    backbone_params = embed_dim * embed_dim * num_layers * 4
    
    # Estimate decoder parameters as sum of decoder channels times 1000 (approximate)
    decoder_params = sum(decoder_channels) * 1000
    
    # Sum total parameters
    total_params = backbone_params + decoder_params
    
    # Format parameter count into human-readable string
    if total_params > 1e9:
        return f"~{total_params/1e9:.1f}B"
    elif total_params > 1e6:
        return f"~{total_params/1e6:.1f}M"
    else:
        return f"~{total_params/1e3:.1f}K"

def list_available_models() -> List[str]:
    # Initialize list to hold model descriptions
    models = []
    
    # Iterate over all backbone names in the configuration registry
    for backbone_name in MODEL_CONFIGS.keys():
        config = MODEL_CONFIGS[backbone_name]
        
        # Extract relevant configuration details with defaults
        patch_size = config.get('patch_size', 'N/A')
        img_size = config.get('img_size', 'N/A')
        embed_dim = config.get('embed_dim', 'N/A')
        
        # Estimate parameter count using helper function
        params = _estimate_parameters(config)
        
        # Format model description string and add to list
        models.append(f"{backbone_name} (patch:{patch_size}, img:{img_size}, dim:{embed_dim}, params:{params})")
    
    # Return list of model descriptions
    return models

def register_model(name: str, model_class: Type[nn.Module], default_config: Dict[str, Any]):
    # Ensure the model class implements a callable 'forward' method
    if not hasattr(model_class, 'forward') or not callable(getattr(model_class, 'forward')):
        raise ValueError(f"Model class '{name}' must implement a callable 'forward' method.")
    
    # Add the model class and its default configuration to the registry
    MODEL_REGISTRY[name] = {
        'model_class': model_class,
        'default_config': default_config,
    }
    
    # Log successful registration of the model
    logger.info(f"Registered model '{name}' in registry")

def create_model_from_checkpoint(checkpoint_path: str, strict: bool = True) -> nn.Module:
    # Check if checkpoint file exists
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    try:
        # Load checkpoint data from file
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract model configuration from checkpoint
        config = checkpoint.get('config')
        if config is None:
            raise RuntimeError("Checkpoint does not contain model configuration")
        
        # Extract backbone name from configuration
        backbone_name = config.get('backbone_name')
        if backbone_name is None:
            raise RuntimeError("Backbone name missing in checkpoint config")
        
        # Create model instance using configuration
        model = create_model(backbone_name, config)
        
        # Retrieve state dictionary from checkpoint
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        # Load weights into model with optional strict key checking
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
        
        # Log any missing or unexpected keys during loading
        if missing_keys:
            logger.warning(f"Missing keys when loading checkpoint: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")
        
        # Set model to evaluation mode
        model.eval()
        logger.info(f"Model loaded from checkpoint '{checkpoint_path}' successfully")
        return model
        
    except Exception as e:
        # Log and raise error if checkpoint loading fails
        logger.error(f"Failed to load model from checkpoint: {str(e)}")
        raise RuntimeError(f"Failed to load model from checkpoint: {str(e)}")

class ModelBuilder:
    def __init__(self):
        # Initialize builder with empty config, no backbone, and build state
        self._config = {}
        self._backbone_name = None
        self._built = False
    
    def backbone(self, backbone_name: str, **kwargs):
        # Validate backbone name against known configs
        if backbone_name not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Set backbone name and update config with additional parameters
        self._backbone_name = backbone_name
        self._config.update(kwargs)
        return self  # Enable method chaining
    
    def decoder(self, decoder_channels: List[int], **kwargs):
        # Set decoder channels and update config with additional parameters
        self._config['decoder_channels'] = decoder_channels
        self._config.update(kwargs)
        return self  # Enable method chaining
    
    def training_config(self, **kwargs):
        # Update training-related configuration parameters
        self._config.update(kwargs)
        return self  # Enable method chaining
    
    def build(self) -> nn.Module:
        # Prevent multiple builds from the same builder instance
        if self._built:
            raise RuntimeError("Model has already been built")
        
        # Ensure backbone was specified before building model
        if not self._backbone_name:
            raise RuntimeError("Backbone must be specified")
        
        # Create the model using the stored backbone and config
        model = create_model(self._backbone_name, self._config)
        self._built = True
        
        # Log successful model build
        logger.info(f"Model built with backbone '{self._backbone_name}'")
        return model

register_model('edepth', edepth, MODEL_CONFIGS['vit_base_patch16_224'])