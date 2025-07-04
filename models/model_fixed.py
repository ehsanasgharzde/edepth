import torch
import torch.nn as nn
from typing import List, Optional
from .backbones.backbone_fixed import FixedViTBackbone
from .decoders.decoder_fixed import DPTDecoderFixed
import logging
import traceback

class FixedDPTModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = 'vit_base_patch16_224',
        extract_layers: Optional[List[int]] = None,
        decoder_channels: Optional[List[int]] = None,
        patch_size: int = 16,
        num_classes: int = 1,
        pretrained: bool = True
    ):
        super().__init__()
        self.backbone = FixedViTBackbone(
            model_name=backbone_name,
            extract_layers=extract_layers,
            pretrained=pretrained
        )
        feature_dims = [f['num_chs'] for f in self.backbone.backbone.feature_info.get_dicts()]
        if decoder_channels is None:
            decoder_channels = [256, 512, 1024, 1024][:len(feature_dims)]
        self.decoder = DPTDecoderFixed(
            feature_dims=feature_dims,
            decoder_channels=decoder_channels,
            patch_size=patch_size,
            num_classes=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            if x.dim() != 4:
                raise ValueError(f"Input must be 4D (B, C, H, W), got {x.shape}")
            features = self.backbone(x)
            depth = self.decoder(features)
            # Output shape: (B, 1, H, W)
            if depth.shape[-2:] != x.shape[-2:]:
                depth = torch.nn.functional.interpolate(depth, size=x.shape[-2:], mode='bilinear', align_corners=False)
            return depth
        except Exception as e:
            logging.error(f"Error in FixedDPTModel forward: {e}\n{traceback.format_exc()}")
            raise 