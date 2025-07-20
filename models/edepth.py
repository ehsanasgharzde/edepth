# FILE: models/model_fixed.py
# ehsanasgharzde - COMPLETE DPT MODEL INTEGRATION WITH BACKBONE AND DECODER
# ehsanasgharzde - FIXED REDUNDANT CODE BY EXTRACTING PURE FUNCTINOS AND BASECLASS LEVEL METHODS

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any
import logging
import warnings

# Import centralized utilities
from .backbones.backbone import ViT
from .decoders.decoder import DPT
from ..configs.model_config import get_model_config, list_available_models, validate_config 
from ..utils.model_validation import (
    validate_backbone_name, validate_decoder_channels, validate_model_output, 
    validate_vit_input, validate_dpt_features, ConfigValidationError
)
from ..utils.model_utils import ( # type: ignore
    get_model_info, ModelInfo, save_model_checkpoint, load_model_checkpoint, 
    calculate_feature_shapes, validate_feature_compatibility
)
from ..utils.model_utils import print_model_summary # type: ignore

logger = logging.getLogger(__name__)

class edepth(nn.Module):
    def __init__(
        self,
        backbone_name: str = 'vit_base_patch16_224',
        extract_layers: Optional[List[int]] = None,
        decoder_channels: Optional[List[int]] = None,
        use_attention: bool = False,
        final_activation: str = 'sigmoid',
        use_checkpointing: bool = False,
        **kwargs
    ):
        super().__init__()

        # Validate backbone name using centralized validation
        available_models = list_available_models()
        validate_backbone_name(backbone_name, available_models, "backbone_name")

        # Get complete model configuration using centralized config
        self.config = get_model_config(backbone_name)

        # Override config with provided parameters
        if extract_layers is not None:
            self.config['extract_layers'] = extract_layers
        if decoder_channels is not None:
            self.config['decoder_channels'] = decoder_channels
        if use_attention is not None:
            self.config['use_attention'] = use_attention
        if final_activation is not None:
            self.config['final_activation'] = final_activation
        if use_checkpointing is not None:
            self.config['use_checkpointing'] = use_checkpointing

        # Validate complete configuration
        self.validate_complete_config()

        # Store key parameters
        self.backbone_name = backbone_name
        self.extract_layers = self.config['extract_layers']
        self.decoder_channels = self.config['decoder_channels']

        # Initialize backbone
        self.backbone = ViT(
            model_name=backbone_name,
            extract_layers=self.extract_layers,
            pretrained=self.config.get('pretrained', True),
            use_checkpointing=use_checkpointing
        )

        # Get backbone feature dimensions
        backbone_channels = [self.config['embed_dim']] * len(self.extract_layers)

        # Initialize decoder
        self.decoder = DPT(
            backbone_channels=backbone_channels,
            decoder_channels=self.decoder_channels,
            use_attention=self.config['use_attention'],
            final_activation=self.config['final_activation']
        )

        # Validate feature compatibility
        validate_feature_compatibility(
            [torch.zeros(1, ch, 14, 14) for ch in backbone_channels],
            backbone_channels
        )

        logger.info(f"EDepth model initialized with backbone '{backbone_name}'")

    def validate_complete_config(self) -> None:
        try:
            # Validate backbone configuration
            validate_config(self.config, config_type='backbone')

            # Validate decoder configuration  
            validate_config(self.config, config_type='decoder')

            # Additional model-specific validations
            validate_decoder_channels(
                self.config['decoder_channels'],
                name="decoder_channels"
            )

        except Exception as e:
            raise ConfigValidationError(f"Configuration validation failed: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Validate input using centralized validation
        validate_vit_input(x)

        batch_size, channels, height, width = x.shape

        # Extract features from backbone
        backbone_features = self.backbone(x)

        # Validate backbone features
        validate_dpt_features(backbone_features)

        # Decode features to depth
        depth = self.decoder(backbone_features)

        # Validate output using centralized validation
        validate_model_output(
            depth,
            expected_shape=(batch_size, 1, None, None),  # None for flexible spatial dims # type: ignore
            expected_range=(0.0, 1.0) if self.config['final_activation'] == 'sigmoid' else None,
            name="depth_output"
        )

        return depth

    def get_model_info(self) -> ModelInfo:
        return get_model_info(self)

    def get_model_summary(self) -> Dict[str, Any]:
        model_info = self.get_model_info()
        summary = model_info.get_summary()

        # Add model-specific information
        summary.update({
            'backbone_name': self.backbone_name,
            'extract_layers': self.extract_layers,
            'decoder_channels': self.decoder_channels,
            'use_attention': self.config['use_attention'],
            'final_activation': self.config['final_activation'],
            'backbone_parameters': get_model_info(self.backbone).total_parameters,
            'decoder_parameters': get_model_info(self.decoder).total_parameters
        })

        return summary

    def freeze_backbone(self) -> None:
        self.backbone.freeze()

    def unfreeze_backbone(self) -> None:
        self.backbone.unfreeze()

    def get_feature_shapes(self, input_size: tuple = (224, 224)) -> List[tuple]:
        return calculate_feature_shapes(
            backbone_name=self.backbone_name,
            img_size=input_size[0],  # Assume square images
            extract_layers=self.extract_layers,
            embed_dim=self.config['embed_dim'],
            patch_size=self.config['patch_size']
        )

    def save_checkpoint(
        self,
        checkpoint_path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        **metadata
    ) -> None:
        # Add model-specific metadata
        model_metadata = {
            'backbone_name': self.backbone_name,
            'model_config': self.config,
            **metadata
        }

        save_model_checkpoint(
            model=self,
            checkpoint_path=checkpoint_path,
            optimizer=optimizer,
            epoch=epoch,
            step=step,
            metadata=model_metadata
        )

    def load_checkpoint(
        self,
        checkpoint_path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        strict: bool = True,
        map_location: Optional[str] = None
    ) -> Dict[str, Any]:
        return load_model_checkpoint(
            model=self,
            checkpoint_path=checkpoint_path,
            optimizer=optimizer,
            strict=strict,
            map_location=map_location
        )

    def print_summary(self, input_size: tuple = (3, 224, 224)) -> None:
        print_model_summary(self, input_size)

        # Print additional model-specific information
        print("\nModel Configuration:")
        print("-" * 40)
        for key, value in self.config.items():
            print(f"{key:20}: {value}")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'edepth':
        # Extract parameters from config
        backbone_name = config.get('backbone_name', 'vit_base_patch16_224')
        extract_layers = config.get('extract_layers')
        decoder_channels = config.get('decoder_channels')
        use_attention = config.get('use_attention', False)
        final_activation = config.get('final_activation', 'sigmoid')
        use_checkpointing = config.get('use_checkpointing', False)

        return cls(
            backbone_name=backbone_name,
            extract_layers=extract_layers,
            decoder_channels=decoder_channels,
            use_attention=use_attention,
            final_activation=final_activation,
            use_checkpointing=use_checkpointing
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def to_device(self, device: torch.device) -> 'edepth':
        return self.to(device)

    def train_mode(self) -> 'edepth':
        return self.train()

    def eval_mode(self) -> 'edepth':
        return self.eval()