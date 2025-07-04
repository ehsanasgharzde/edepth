import torch
import torch.nn as nn
import timm
from typing import List, Optional
import logging
import traceback
import torch.utils.checkpoint as checkpoint

logger = logging.getLogger(__name__)

class FixedViTBackbone(nn.Module):
    def __init__(self, model_name: str = 'vit_base_patch16_224', extract_layers: Optional[List[int]] = None, pretrained: bool = True, use_checkpointing: bool = False):
        super().__init__()
        self.model_name = model_name
        self.extract_layers = extract_layers or [3, 6, 9, 12]
        self.use_checkpointing = use_checkpointing
        try:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=self.extract_layers
            )
        except Exception as e:
            logging.error(f"Failed to load ViT backbone: {e}\n{traceback.format_exc()}")
            raise

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        try:
            if x.dim() != 4:
                raise ValueError(f"Input must be 4D (B, C, H, W), got {x.shape}")
            if self.use_checkpointing and hasattr(self.backbone, 'blocks') and hasattr(self.backbone, 'forward_features'):
                blocks = self.backbone.blocks
                if (hasattr(blocks, '__iter__') and not isinstance(blocks, torch.Tensor)):
                    for block in blocks:
                        x = checkpoint.checkpoint(block, x)
                    features = self.backbone.forward_features(x)
                else:
                    features = self.backbone(x)
            else:
                features = self.backbone(x)
            # Ensure all features are (B, C, H, W)
            for i, f in enumerate(features):
                if f.dim() != 4:
                    raise ValueError(f"Feature at layer {self.extract_layers[i]} is not 4D: {f.shape}")
            return features
        except Exception as e:
            logging.error(f"Error in forward_features: {e}\n{traceback.format_exc()}")
            raise

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.forward_features(x) 