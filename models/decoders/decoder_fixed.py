import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import torch.utils.checkpoint as checkpoint

class ConvProjection(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.proj(x)

class FusionBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class DPTDecoderFixed(nn.Module):
    def __init__(self, feature_dims: List[int], decoder_channels: List[int], patch_size: int = 16, num_classes: int = 1, use_checkpointing: bool = False):
        super().__init__()
        assert len(feature_dims) == len(decoder_channels)
        self.use_checkpointing = use_checkpointing
        self.projections = nn.ModuleList([
            ConvProjection(f, c) for f, c in zip(feature_dims, decoder_channels)
        ])
        self.fusions = nn.ModuleList([
            FusionBlock(decoder_channels[i+1], decoder_channels[i], decoder_channels[i])
            for i in range(len(decoder_channels) - 1)
        ])
        self.final_conv = nn.Sequential(
            nn.Conv2d(decoder_channels[0], decoder_channels[0] // 2, 3, padding=1),
            nn.BatchNorm2d(decoder_channels[0] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels[0] // 2, num_classes, 1)
        )
        self.patch_size = patch_size
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # features: [low, mid, high, highest]
        feats = [proj(f) for proj, f in zip(self.projections, features)]
        x = feats[-1]
        for i in reversed(range(len(self.fusions))):
            if self.use_checkpointing:
                x = checkpoint.checkpoint(self.fusions[i], x, feats[i])
            else:
                x = self.fusions[i](x, feats[i])
        x = self.final_conv(x)
        # Final upsampling to patch_size scale
        scale = self.patch_size
        x = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
        return x 