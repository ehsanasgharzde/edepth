# File: models/backbones/backbone_fixed.py
# ehsanasgharzde - FULL BACKBONE VIT IMPLEMENTATION
# ehsanasgharzde - FIXED REDUNDANT CODE BY EXTRACTING PURE FUNCTINOS AND BASECLASS LEVEL METHODS

import torch
import torch.nn as nn
import timm
from typing import List, Optional, Dict, Any
import logging
import torch.utils.checkpoint as checkpoint

# Import centralized utilities
from ...configs.model_config import get_backbone_config, list_available_backbones
from ...utils.model_validation import (
    validate_backbone_name, validate_patch_size, validate_extract_layers, 
    validate_spatial_dimensions, validate_vit_input
)
from ...utils.model_utils import ( 
    calculate_patch_grid, sequence_to_spatial, interpolate_features,
    cleanup_hooks, apply_gradient_checkpointing
) 

logger = logging.getLogger(__name__)

class ViT(nn.Module):
    def __init__(
        self, 
        model_name: str = 'vit_base_patch16_224',
        extract_layers: Optional[List[int]] = None,
        pretrained: bool = True,
        use_checkpointing: bool = False,
        **kwargs
    ):
        super().__init__()

        # Validate backbone name using centralized validation
        available_backbones = list_available_backbones()
        validate_backbone_name(model_name, available_backbones)

        # Get configuration using centralized config
        self.config = get_backbone_config(model_name)
        self.model_name = model_name
        self.use_checkpointing = use_checkpointing

        # Extract configuration parameters
        self.patch_size = self.config['patch_size']
        self.img_size = self.config['img_size']
        self.embed_dim = self.config['embed_dim']
        self.num_layers = self.config.get('num_layers', 12)

        # Validate configuration using centralized validation
        validate_patch_size(self.patch_size, self.img_size)

        # Setup extract layers with validation
        if extract_layers is None:
            extract_layers = [self.num_layers - 4, self.num_layers - 3, 
                            self.num_layers - 2, self.num_layers - 1]

        validate_extract_layers(extract_layers, self.num_layers)
        self.extract_layers = extract_layers

        # Load backbone model
        self.load_backbone_model(pretrained)

        # Setup feature extraction
        self.setup_feature_extraction()

        # Apply gradient checkpointing if requested
        if use_checkpointing:
            apply_gradient_checkpointing(self.model, enable=True)

        logger.info(f"ViT backbone '{model_name}' initialized successfully")

    def load_backbone_model(self, pretrained: bool) -> None:
        try:
            self.model = timm.create_model(
                self.model_name,
                pretrained=pretrained,
                features_only=False,
                out_indices=self.extract_layers
            )

            # Verify model configuration
            if hasattr(self.model, 'embed_dim'):
                actual_embed_dim = self.model.embed_dim
                if actual_embed_dim != self.embed_dim:
                    logger.warning(
                        f"Embed dim mismatch: expected {self.embed_dim}, "
                        f"got {actual_embed_dim}"
                    )

            logger.info(f"Loaded {self.model_name} with pretrained={pretrained}")

        except Exception as e:
            logger.error(f"Failed to load backbone model: {e}")
            raise RuntimeError(f"Backbone loading failed: {e}")

    def setup_feature_extraction(self) -> None:
        self.features = {}
        self.hooks = []

        # Create hook function
        def create_hook(layer_name: str):
            def hook_fn(module, input, output):
                self.features[layer_name] = output
            return hook_fn

        # Register hooks for extract layers
        for layer_idx in self.extract_layers:
            layer_name = f'blocks.{layer_idx}'
            try:
                layer_module = dict(self.model.named_modules())[layer_name]
                hook = layer_module.register_forward_hook(create_hook(layer_name))
                self.hooks.append(hook)
            except KeyError:
                logger.warning(f"Layer {layer_name} not found in model")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Validate input using centralized validation
        validate_vit_input(x)

        # Validate spatial dimensions
        batch_size, channels, height, width = x.shape
        validate_spatial_dimensions(height, width, self.patch_size)

        # Clear previous features
        self.features.clear()

        # Forward pass through model
        if self.use_checkpointing and self.training:
            # Use gradient checkpointing
            def create_custom_forward():
                def custom_forward(*inputs):
                    return self.model(*inputs)
                return custom_forward

            _ = checkpoint.checkpoint(create_custom_forward(), x)
        else:
            _ = self.model(x)

        # Extract features and convert to spatial format
        extracted_features = []
        patch_grid = calculate_patch_grid(self.img_size, self.patch_size)

        for layer_idx in self.extract_layers:
            layer_name = f'blocks.{layer_idx}'
            if layer_name in self.features:
                feature = self.features[layer_name]

                # Convert sequence to spatial format
                spatial_feature = sequence_to_spatial(
                    feature, 
                    patch_grid, 
                    include_cls_token=True
                )

                # Interpolate to match input resolution if needed
                target_size = (height // self.patch_size, width // self.patch_size)
                if spatial_feature.shape[-2:] != target_size:
                    spatial_feature = interpolate_features(
                        spatial_feature, 
                        target_size,
                        mode='bilinear'
                    )

                extracted_features.append(spatial_feature)

        if len(extracted_features) != len(self.extract_layers):
            logger.warning(
                f"Expected {len(self.extract_layers)} features, "
                f"got {len(extracted_features)}"
            )

        return extracted_features

    def get_feature_info(self) -> List[Dict[str, Any]]:
        feature_info = []
        patch_grid = calculate_patch_grid(self.img_size, self.patch_size)

        for layer_idx in self.extract_layers:
            info = {
                'layer_idx': layer_idx,
                'channels': self.embed_dim,
                'spatial_size': patch_grid,
                'layer_name': f'blocks.{layer_idx}'
            }
            feature_info.append(info)

        return feature_info

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        logger.info("ViT backbone frozen")

    def unfreeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = True
        self.train()
        logger.info("ViT backbone unfrozen")

    def __del__(self):
        if hasattr(self, 'hooks'):
            cleanup_hooks(self.hooks)