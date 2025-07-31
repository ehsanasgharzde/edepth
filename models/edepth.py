# FILE: models/edepth.py
# ehsanasgharzde - COMPLETE DPT MODEL INTEGRATION WITH BACKBONE AND DECODER
# ehsanasgharzde - FIXED REDUNDANT CODE BY EXTRACTING PURE FUNCTIONS AND BASECLASS LEVEL METHODS

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any

# Import centralized utilities
from backbones.backbone import ViT
from decoders.decoder import DPT
from configs.config import ModelConfig, BackboneType, ConfigValidationError
from utils.model_validation import (
    validate_backbone_name, validate_decoder_channels, validate_model_output, 
    validate_vit_input, validate_dpt_features
)
from utils.model_operation import (
    get_model_info, ModelInfo, save_model_checkpoint, load_model_checkpoint, 
    calculate_feature_shapes, validate_feature_compatibility, print_model_summary
)
from logger.logger import setup_logging 

# Setup logger for factory operations
logger = setup_logging(__file__)

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
        available_models = [b.value for b in BackboneType]
        validate_backbone_name(backbone_name, available_models, "backbone_name")

        # Create model configuration using the current config system
        self.config = ModelConfig(
            backbone=backbone_name,
            extract_layers=extract_layers if extract_layers is not None else [2, 5, 8, 11],
            decoder_channels=decoder_channels if decoder_channels is not None else [256, 512, 1024, 1024]
        ).to_dict()

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
            use_attention=self.config.get('use_attention', False),
            final_activation=self.config.get('final_activation', 'sigmoid')
        )

        # Validate feature compatibility
        validate_feature_compatibility(
            [torch.zeros(1, ch, 14, 14) for ch in backbone_channels],
            backbone_channels
        )

        logger.info(f"EDepth model initialized with backbone '{backbone_name}'")

    def validate_complete_config(self) -> None:
        try:
            # Create ModelConfig instance for validation
            model_config = ModelConfig()
            
            # Set values from self.config
            model_config.backbone = self.config.get('backbone', 'vit_base_patch16_224')
            model_config.extract_layers = self.config.get('extract_layers', [2, 5, 8, 11])
            model_config.decoder_channels = self.config.get('decoder_channels', [256, 512, 1024, 1024])
            
            # Validate through ModelConfig's validate method
            model_config.validate()
            
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
            'use_attention': self.config.get('use_attention', False),
            'final_activation': self.config.get('final_activation', 'sigmoid'),
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
            embed_dim=self.config.get('embed_dim', 768),
            patch_size=self.config.get('patch_size', 16)
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
        backbone_name = config.get('backbone', 'vit_base_patch16_224')
        extract_layers = config.get('extract_layers')
        decoder_channels = config.get('decoder_channels')
        use_attention = config.get('use_attention', False)
        final_activation = config.get('final_activation', 'sigmoid')
        use_checkpointing = config.get('use_gradient_checkpointing', False)

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