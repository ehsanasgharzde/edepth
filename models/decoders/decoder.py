# File: models/decoders/decoder_fixed.py
# ehsanasgharzde - COMPLETE DPT DECODER WITH FUSION BLOCK IMPLEMENTATION

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

class FusionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2, use_attention: bool = False):
        super().__init__()
        self.scale_factor = scale_factor  # Upsampling scale factor
        self.use_attention = use_attention  # Flag to enable/disable attention

        # Projection layer: adjusts channel dimensions
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),  # 1x1 convolution
            nn.BatchNorm2d(out_channels),             # Batch normalization
            nn.ReLU(inplace=True)                     # ReLU activation
        )

        # Optional attention mechanism
        if use_attention:
            self.attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),                                  # Global context pooling
                nn.Conv2d(out_channels, max(1, out_channels // 16), 1),  # Bottleneck
                nn.ReLU(inplace=True),                                   
                nn.Conv2d(max(1, out_channels // 16), out_channels, 1),  # Expand back to out_channels
                nn.Sigmoid()                                             # Attention weights
            )

        # Refinement layer after fusion
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),  # 3x3 convolution
            nn.BatchNorm2d(out_channels),                         # Batch normalization
            nn.ReLU(inplace=True)                                 # ReLU activation
        )

        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # Custom weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # He initialization
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Project input features to desired output channels
        x = self.proj(x)

        # Upsample if scale factor is not 1
        if self.scale_factor != 1:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)

        # Add skip connection if provided
        if skip is not None:
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
            x = x + skip  # Fuse features

        # Apply attention if enabled
        if self.use_attention:
            att = self.attention(x)
            x = x * att  # Channel-wise reweighting

        # Final refinement
        return self.refine(x)


class DPT(nn.Module):
    def __init__(self, 
                 backbone_channels: List[int], 
                 decoder_channels: List[int], 
                 num_stages: int = 4,
                 use_attention: bool = False,
                 final_activation: str = 'sigmoid'):
        super().__init__()

        # Validate that backbone and decoder channel lists match the number of stages
        if len(backbone_channels) != num_stages:
            raise ValueError(f"backbone_channels length {len(backbone_channels)} != num_stages {num_stages}")
        if len(decoder_channels) != num_stages:
            raise ValueError(f"decoder_channels length {len(decoder_channels)} != num_stages {num_stages}")
        
        self.num_stages = num_stages  # Number of decoder stages
        self.final_activation = final_activation  # Final activation function to apply

        logger.info(f"Initializing DPT with {num_stages} stages, attention: {use_attention}")
        
        # Projection layers to align feature channels from backbone to decoder
        self.projections = nn.ModuleList([
            self._make_projection(in_ch, out_ch) 
            for in_ch, out_ch in zip(backbone_channels, decoder_channels)
        ])
        
        # Fusion blocks for combining features across stages
        self.fusions = nn.ModuleList([
            FusionBlock(decoder_channels[i], decoder_channels[i], 
                       scale_factor=2 if i > 0 else 1,  # Upsample after the first stage
                       use_attention=use_attention)
            for i in range(num_stages)
        ])
        
        # Final output layer to produce single-channel depth map
        self.output_conv = nn.Conv2d(decoder_channels[-1], 1, 1)

        # Initialize model weights
        self._init_weights()
    
    def _make_projection(self, in_channels: int, out_channels: int) -> nn.Sequential:
        # Creates a projection block with Conv-BN-ReLU to align channels
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _init_weights(self):
        # Custom weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # He initialization
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, backbone_features: List[torch.Tensor]) -> torch.Tensor:
        # Forward pass through DPT

        # Validate input features from backbone
        if len(backbone_features) != self.num_stages:
            raise ValueError(f"Expected {self.num_stages} features, got {len(backbone_features)}")
        
        logger.debug(f"Processing {len(backbone_features)} backbone features")
        
        # Project backbone features to decoder channel dimensions
        projected = [proj(feat) for proj, feat in zip(self.projections, backbone_features)]
        
        # Process first stage without skip connection
        x = self.fusions[0](projected[0])
        logger.debug(f"Stage 0 output shape: {x.shape}")
        
        # Process subsequent stages with skip connections
        for i in range(1, self.num_stages):
            x = self.fusions[i](x, projected[i])
            logger.debug(f"Stage {i} output shape: {x.shape}")
        
        # Upsample final feature map to match the highest resolution (e.g., 4x the smallest feature map)
        target_size = [s * 4 for s in backbone_features[-1].shape[-2:]]
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        
        # Generate single-channel depth output
        depth = self.output_conv(x)
        
        # Apply the selected final activation function
        if self.final_activation == 'sigmoid':
            depth = torch.sigmoid(depth)
        elif self.final_activation == 'tanh':
            depth = torch.tanh(depth)
        elif self.final_activation == 'relu':
            depth = F.relu(depth)
        
        logger.debug(f"Final depth shape: {depth.shape}")
        return depth

def validate_features(features: List[torch.Tensor], expected_channels: List[int]) -> None:
    # Check that the number of input feature maps matches the expected count
    if len(features) != len(expected_channels):
        raise ValueError(f"Feature count mismatch: expected {len(expected_channels)}, got {len(features)}")
    
    # Iterate through each feature and its expected channel count
    for idx, (feat, expected_c) in enumerate(zip(features, expected_channels)):
        # Raise an error if the feature's channel dimension doesn't match the expected one
        if feat.size(1) != expected_c:
            raise ValueError(f"Channel mismatch at index {idx}: expected {expected_c}, got {feat.size(1)}")


def interpolate_to_size(features: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
    # Resize the input feature map to the target spatial size using bilinear interpolation
    # align_corners=False ensures a smooth and consistent interpolation
    return F.interpolate(features, size=target_size, mode='bilinear', align_corners=False)