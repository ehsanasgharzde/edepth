# File: models/decoders/decoder_fixed.py
# ehsanasgharzde - COMPLETE DPT DECODER WITH FUSION BLOCK IMPLEMENTATION
# hosseinsolymanzadeh - PROPER COMMENTING

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
        self.use_attention = use_attention  # Flag to enable or disable attention mechanism

        # Projection layer to match the required output channels
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),  # 1x1 convolution for channel reduction or expansion
            nn.BatchNorm2d(out_channels),             # Normalize output to stabilize training
            nn.ReLU(inplace=True)                     # Apply ReLU activation
        )

        # Optional attention module based on squeeze-and-excitation
        if use_attention:
            self.attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),                                  # Global average pooling to capture channel context
                nn.Conv2d(out_channels, max(1, out_channels // 16), 1),  # Reduce channels (bottleneck)
                nn.ReLU(inplace=True),                                   # Non-linear activation
                nn.Conv2d(max(1, out_channels // 16), out_channels, 1),  # Expand back to original channel size
                nn.Sigmoid()                                             # Output attention weights between 0 and 1
            )

        # Refinement layer to process the fused feature map
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),  # 3x3 convolution with padding to preserve size
            nn.BatchNorm2d(out_channels),                         # Normalize features
            nn.ReLU(inplace=True)                                 # Activation function
        )

        # Initialize the weights of all layers
        self._init_weights()
    
    def _init_weights(self):
        # Apply initialization to each module
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming initialization suited for ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # Zero-initialize biases
            elif isinstance(m, nn.BatchNorm2d):
                # Initialize BatchNorm weights to 1 and biases to 0
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Project the input feature map to the desired number of output channels
        x = self.proj(x)

        # Perform upsampling if required
        if self.scale_factor != 1:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)

        # Add skip connection if available
        if skip is not None:
            if skip.shape[-2:] != x.shape[-2:]:
                # Resize skip feature to match current feature size
                skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
            x = x + skip  # Element-wise addition of skip and upsampled features

        # Apply attention mechanism if enabled
        if self.use_attention:
            att = self.attention(x)  # Compute attention weights
            x = x * att  # Multiply features by attention map (channel-wise scaling)

        # Apply refinement to the fused features
        return self.refine(x)


class DPT(nn.Module):
    def __init__(self, 
                 backbone_channels: List[int], 
                 decoder_channels: List[int], 
                 num_stages: int = 4,
                 use_attention: bool = False,
                 final_activation: str = 'sigmoid'):
        super().__init__()

        # Ensure the number of input and output channels matches the number of stages
        if len(backbone_channels) != num_stages:
            raise ValueError(f"backbone_channels length {len(backbone_channels)} != num_stages {num_stages}")
        if len(decoder_channels) != num_stages:
            raise ValueError(f"decoder_channels length {len(decoder_channels)} != num_stages {num_stages}")
        
        self.num_stages = num_stages  # Total number of decoder stages
        self.final_activation = final_activation  # Type of final activation to apply on output

        logger.info(f"Initializing DPT with {num_stages} stages, attention: {use_attention}")
        
        # Create projection layers to align backbone feature dimensions to decoder dimensions
        self.projections = nn.ModuleList([
            self._make_projection(in_ch, out_ch) 
            for in_ch, out_ch in zip(backbone_channels, decoder_channels)
        ])
        
        # Create fusion blocks for each stage, including optional attention and upsampling
        self.fusions = nn.ModuleList([
            FusionBlock(decoder_channels[i], decoder_channels[i], 
                       scale_factor=2 if i > 0 else 1,  # No upsampling in first stage
                       use_attention=use_attention)
            for i in range(num_stages)
        ])
        
        # Final convolution to reduce feature map to single-channel depth map
        self.output_conv = nn.Conv2d(decoder_channels[-1], 1, 1)

        # Initialize weights for all layers
        self._init_weights()
    
    def _make_projection(self, in_channels: int, out_channels: int) -> nn.Sequential:
        # Build a 1x1 Conv + BN + ReLU block for projecting backbone features to decoder space
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _init_weights(self):
        # Initialize convolutional and batch norm layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # He initialization
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # Zero-initialize bias if present
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)  # Set BN scale (gamma) to 1
                nn.init.zeros_(m.bias)   # Set BN bias (beta) to 0
    
    def forward(self, backbone_features: List[torch.Tensor]) -> torch.Tensor:
        # Forward pass through the DPT decoder

        # Ensure correct number of features is provided from backbone
        if len(backbone_features) != self.num_stages:
            raise ValueError(f"Expected {self.num_stages} features, got {len(backbone_features)}")
        
        logger.debug(f"Processing {len(backbone_features)} backbone features")
        
        # Apply projection layers to each backbone feature map
        projected = [proj(feat) for proj, feat in zip(self.projections, backbone_features)]
        
        # First fusion stage uses only the projected feature (no skip)
        x = self.fusions[0](projected[0])
        logger.debug(f"Stage 0 output shape: {x.shape}")
        
        # Process remaining stages with skip connections
        for i in range(1, self.num_stages):
            x = self.fusions[i](x, projected[i])
            logger.debug(f"Stage {i} output shape: {x.shape}")
        
        # Upsample final feature map to match input resolution (typically 4x upscale)
        target_size = [s * 4 for s in backbone_features[-1].shape[-2:]]
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        
        # Produce final depth prediction
        depth = self.output_conv(x)
        
        # Apply selected activation function to output
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
    
    # Iterate through each feature tensor and its corresponding expected channel count
    for idx, (feat, expected_c) in enumerate(zip(features, expected_channels)):
        # Validate that the current feature's channel size matches the expected value
        if feat.size(1) != expected_c:
            raise ValueError(f"Channel mismatch at index {idx}: expected {expected_c}, got {feat.size(1)}")


def interpolate_to_size(features: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
    # Resize the input feature map to the given spatial dimensions (height, width)
    # Use bilinear interpolation for smooth resizing
    # align_corners=False avoids artifacts when interpolating feature maps
    return F.interpolate(features, size=target_size, mode='bilinear', align_corners=False)