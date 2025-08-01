# File: models/decoders/decoder_fixed.py
# ehsanasgharzde - COMPLETE DPT DECODER WITH FUSION BLOCK IMPLEMENTATION
# hosseinsolymanzadeh - PROPER COMMENTING
# ehsanasgharzde - FIXED REDUNDANT CODE BY EXTRACTING PURE FUNCTIONS AND BASECLASS LEVEL METHODS

import torch
import torch.nn as nn
from typing import List, Tuple

# Import centralized utilities
from utils.model_validation import (
    validate_feature_tensors, validate_tensor_input, validate_interpolation_target,
    validate_dpt_features, TensorValidationError
)
from utils.model_operation import (
    interpolate_features, initialize_weights, ModelInfo, get_model_info 
)
from logger.logger import setup_logging 

# Setup logger for factory operations
logger = setup_logging(__file__)

class FusionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2, use_attention: bool = False):
        super().__init__()
        self.scale_factor = scale_factor
        self.use_attention = use_attention

        # Projection layer to match the required output channels
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Optional attention module based on squeeze-and-excitation
        if use_attention:
            self.attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_channels, max(1, out_channels // 16), 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(max(1, out_channels // 16), out_channels, 1),
                nn.Sigmoid()
            )

        # Initialize weights using centralized utility
        initialize_weights(self, init_type='xavier_uniform')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Validate input using centralized validation
        validate_tensor_input(x, "FusionBlock input", expected_dims=4)

        # Apply projection
        x = self.proj(x)

        # Apply attention if enabled
        if self.use_attention:
            attention_weights = self.attention(x)
            x = x * attention_weights

        # Upsample using centralized interpolation utility
        if self.scale_factor > 1:
            target_size = (x.size(2) * self.scale_factor, x.size(3) * self.scale_factor)
            x = interpolate_features(x, target_size, mode='bilinear', align_corners=False)

        return x

class DPT(nn.Module):
    def __init__(
        self,
        backbone_channels: List[int],
        decoder_channels: List[int] = [256, 512, 1024, 1024],
        use_attention: bool = False,
        final_activation: str = 'sigmoid'
    ):
        super().__init__()

        # Validate inputs using centralized validation
        if len(backbone_channels) != len(decoder_channels):
            raise TensorValidationError(
                f"Backbone channels ({len(backbone_channels)}) must match "
                f"decoder channels ({len(decoder_channels)})"
            )

        self.backbone_channels = backbone_channels
        self.decoder_channels = decoder_channels
        self.num_stages = len(decoder_channels)
        self.use_attention = use_attention
        self.final_activation = final_activation

        # Create projection layers for backbone features
        self.projections = nn.ModuleList()
        for i, (in_ch, out_ch) in enumerate(zip(backbone_channels, decoder_channels)):
            proj = self.make_projection(in_ch, out_ch)
            self.projections.append(proj)

        # Create fusion blocks for upsampling and feature fusion
        self.fusion_blocks = nn.ModuleList()
        for i in range(self.num_stages - 1):
            fusion = FusionBlock(
                decoder_channels[i + 1], 
                decoder_channels[i], 
                scale_factor=2,
                use_attention=use_attention
            )
            self.fusion_blocks.append(fusion)

        # Final output head
        self.output_head = nn.Sequential(
            nn.Conv2d(decoder_channels[0], decoder_channels[0] // 2, 3, padding=1),
            nn.BatchNorm2d(decoder_channels[0] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels[0] // 2, 1, 1)
        )

        # Initialize weights using centralized utility
        initialize_weights(self, init_type='xavier_uniform')

        logger.info(f"DPT decoder initialized with {self.num_stages} stages")

    def make_projection(self, in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, backbone_features: List[torch.Tensor]) -> torch.Tensor:
        # Validate backbone features using centralized validation
        validate_dpt_features(backbone_features)
        validate_feature_tensors(
            backbone_features, 
            expected_channels=self.backbone_channels,
            expected_count=self.num_stages,
            name="backbone_features"
        )

        # Project backbone features to decoder channels
        projected_features = []
        for i, (feat, proj) in enumerate(zip(backbone_features, self.projections)):
            projected = proj(feat)
            projected_features.append(projected)
            logger.debug(f"Projected feature {i}: {feat.shape} -> {projected.shape}")

        # Start from the deepest feature (highest resolution in feature hierarchy)
        current_feature = projected_features[-1]

        # Progressive fusion from deep to shallow features
        for i in range(self.num_stages - 2, -1, -1):
            # Get the target feature to fuse with
            target_feature = projected_features[i]

            # Apply fusion block to upsample and prepare current feature
            fusion_block = self.fusion_blocks[i]
            upsampled_feature = fusion_block(current_feature)

            # Ensure spatial compatibility using centralized interpolation
            if upsampled_feature.shape[-2:] != target_feature.shape[-2:]:
                target_size = target_feature.shape[-2:]
                upsampled_feature = interpolate_features(
                    upsampled_feature, 
                    target_size, 
                    mode='bilinear', 
                    align_corners=False
                )

            # Fuse features by addition
            current_feature = upsampled_feature + target_feature
            logger.debug(f"Fused feature at stage {i}: {current_feature.shape}")

        # Generate final depth prediction
        depth = self.output_head(current_feature)

        # Apply final activation
        if self.final_activation == 'sigmoid':
            depth = torch.sigmoid(depth)
        elif self.final_activation == 'tanh':
            depth = torch.tanh(depth)
        elif self.final_activation == 'relu':
            depth = torch.relu(depth)
        # 'none' or any other value: no activation

        # Validate output using centralized validation
        validate_tensor_input(depth, "DPT output", expected_dims=4)

        logger.debug(f"Final depth shape: {depth.shape}")
        return depth

    def get_model_info(self) -> ModelInfo:
        return get_model_info(self)

# Utility functions using centralized validation (replacing duplicated functions)
def validate_features_compatibility(
    features: List[torch.Tensor], 
    expected_channels: List[int]
) -> None:
    validate_feature_tensors(
        features, 
        expected_channels=expected_channels,
        name="feature_compatibility"
    )

def interpolate_to_target_size(
    features: torch.Tensor, 
    target_size: Tuple[int, int]
) -> torch.Tensor:
    validate_interpolation_target(features, target_size)
    return interpolate_features(features, target_size, mode='bilinear', align_corners=False)