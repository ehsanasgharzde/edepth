# FILE: configs/model_config.py
# ehsanasgharzde - FIXED REDUNDANT CODE BY EXTRACTING PURE FUNCTINOS AND BASECLASS LEVEL METHODS

from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

# Centralized backbone configurations - Single source of truth
BACKBONE_CONFIGS = {
    "vit_small_patch16_224": {
        "patch_size": 16, 
        "img_size": 224, 
        "embed_dim": 384,
        "num_heads": 6,
        "num_layers": 12,
        "pretrained": True
    },
    "vit_base_patch16_224": {
        "patch_size": 16, 
        "img_size": 224, 
        "embed_dim": 768,
        "num_heads": 12,
        "num_layers": 12,
        "pretrained": True
    },
    "vit_base_patch16_384": {
        "patch_size": 16, 
        "img_size": 384, 
        "embed_dim": 768,
        "num_heads": 12,
        "num_layers": 12,
        "pretrained": True
    },
    "vit_base_patch8_224": {
        "patch_size": 8, 
        "img_size": 224, 
        "embed_dim": 768,
        "num_heads": 12,
        "num_layers": 12,
        "pretrained": True
    },
    "vit_large_patch16_224": {
        "patch_size": 16, 
        "img_size": 224, 
        "embed_dim": 1024,
        "num_heads": 16,
        "num_layers": 24,
        "pretrained": True
    },
    "deit_small_patch16_224": {
        "patch_size": 16, 
        "img_size": 224, 
        "embed_dim": 384,
        "num_heads": 6,
        "num_layers": 12,
        "pretrained": True
    },
    "deit_base_patch16_224": {
        "patch_size": 16, 
        "img_size": 224, 
        "embed_dim": 768,
        "num_heads": 12,
        "num_layers": 12,
        "pretrained": True
    },
}

# Default model configurations with decoder settings
DEFAULT_MODEL_CONFIGS = {
    "vit_small_patch16_224": {
        "backbone_name": "vit_small_patch16_224",
        "extract_layers": [8, 9, 10, 11],
        "decoder_channels": [128, 256, 384, 384],
        "use_attention": False,
        "final_activation": "sigmoid",
        "interpolation_mode": "bilinear",
        "num_classes": 1,
        "use_checkpointing": False
    },
    "vit_base_patch16_224": {
        "backbone_name": "vit_base_patch16_224",
        "extract_layers": [8, 9, 10, 11],
        "decoder_channels": [256, 512, 768, 768],
        "use_attention": False,
        "final_activation": "sigmoid",
        "interpolation_mode": "bilinear",
        "num_classes": 1,
        "use_checkpointing": False
    },
    "vit_base_patch16_384": {
        "backbone_name": "vit_base_patch16_384",
        "extract_layers": [8, 9, 10, 11],
        "decoder_channels": [256, 512, 768, 768],
        "use_attention": True,
        "final_activation": "sigmoid",
        "interpolation_mode": "bilinear",
        "num_classes": 1,
        "use_checkpointing": True
    },
    "vit_base_patch8_224": {
        "backbone_name": "vit_base_patch8_224",
        "extract_layers": [8, 9, 10, 11],
        "decoder_channels": [256, 512, 768, 768],
        "use_attention": True,
        "final_activation": "sigmoid",
        "interpolation_mode": "bilinear",
        "num_classes": 1,
        "use_checkpointing": True
    },
    "vit_large_patch16_224": {
        "backbone_name": "vit_large_patch16_224",
        "extract_layers": [20, 21, 22, 23],
        "decoder_channels": [256, 512, 1024, 1024],
        "use_attention": True,
        "final_activation": "sigmoid",
        "interpolation_mode": "bilinear",
        "num_classes": 1,
        "use_checkpointing": True
    },
    "deit_small_patch16_224": {
        "backbone_name": "deit_small_patch16_224",
        "extract_layers": [8, 9, 10, 11],
        "decoder_channels": [128, 256, 384, 384],
        "use_attention": False,
        "final_activation": "sigmoid",
        "interpolation_mode": "bilinear",
        "num_classes": 1,
        "use_checkpointing": False
    },
    "deit_base_patch16_224": {
        "backbone_name": "deit_base_patch16_224",
        "extract_layers": [8, 9, 10, 11],
        "decoder_channels": [256, 512, 768, 768],
        "use_attention": False,
        "final_activation": "sigmoid",
        "interpolation_mode": "bilinear",
        "num_classes": 1,
        "use_checkpointing": False
    },
}

# Configuration validation rules
CONFIG_VALIDATION_RULES = {
    'backbone': {
        'required': ['patch_size', 'img_size', 'embed_dim'],
        'optional': ['num_heads', 'num_layers', 'pretrained'],
        'validation': {
            'patch_size': lambda x: x in [8, 16, 32],
            'img_size': lambda x: x > 0 and x % 32 == 0,
            'embed_dim': lambda x: x > 0 and x % 64 == 0,
        }
    },
    'decoder': {
        'required': ['decoder_channels'],
        'optional': ['use_attention', 'final_activation', 'interpolation_mode'],
        'validation': {
            'decoder_channels': lambda x: isinstance(x, list) and all(isinstance(c, int) and c > 0 for c in x),
            'use_attention': lambda x: isinstance(x, bool),
            'final_activation': lambda x: x in ['sigmoid', 'tanh', 'relu', 'none'],
            'interpolation_mode': lambda x: x in ['bilinear', 'nearest', 'bicubic'],
        }
    }
}

def get_backbone_config(backbone_name: str) -> Dict[str, Any]:
    if backbone_name not in BACKBONE_CONFIGS:
        available_backbones = list(BACKBONE_CONFIGS.keys())
        raise ValueError(f"Backbone '{backbone_name}' not found. Available backbones: {available_backbones}")

    return BACKBONE_CONFIGS[backbone_name].copy()

def get_model_config(model_name: str) -> Dict[str, Any]:
    if model_name not in DEFAULT_MODEL_CONFIGS:
        available_models = list(DEFAULT_MODEL_CONFIGS.keys())
        raise ValueError(f"Model '{model_name}' not found. Available models: {available_models}")

    # Merge backbone and model configs
    backbone_config = get_backbone_config(model_name)
    model_config = DEFAULT_MODEL_CONFIGS[model_name].copy()

    # Merge configurations
    merged_config = {**backbone_config, **model_config}

    return merged_config

def validate_config(config: Dict[str, Any], config_type: str = 'backbone') -> None:
    if config_type not in CONFIG_VALIDATION_RULES:
        raise ValueError(f"Unknown config type: {config_type}")

    rules = CONFIG_VALIDATION_RULES[config_type]

    # Check required fields
    for field in rules['required']:
        if field not in config:
            raise ValueError(f"Required field '{field}' missing from {config_type} config")

    # Validate fields
    for field, validator in rules['validation'].items():
        if field in config:
            if not validator(config[field]):
                raise ValueError(f"Invalid value for '{field}' in {config_type} config: {config[field]}")

def list_available_backbones() -> List[str]:
    return list(BACKBONE_CONFIGS.keys())

def list_available_models() -> List[str]:
    return list(DEFAULT_MODEL_CONFIGS.keys())

def get_default_extract_layers(backbone_name: str) -> List[int]:
    backbone_config = get_backbone_config(backbone_name)
    num_layers = backbone_config.get('num_layers', 12)

    # Default to last 4 layers
    return [num_layers - 4, num_layers - 3, num_layers - 2, num_layers - 1]

def get_default_decoder_channels(backbone_name: str) -> List[int]:
    backbone_config = get_backbone_config(backbone_name)
    embed_dim = backbone_config['embed_dim']

    # Scale decoder channels based on embedding dimension
    if embed_dim <= 384:
        return [128, 256, 384, 384]
    elif embed_dim <= 768:
        return [256, 512, 768, 768]
    else:
        return [256, 512, 1024, 1024]