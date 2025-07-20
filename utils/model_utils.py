# FILE: utils/model_utils.py
# ehsanasgharzde - FIXED REDUNDANT CODE BY EXTRACTING PURE FUNCTINOS AND BASECLASS LEVEL METHODS

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
import logging

logger = logging.getLogger(__name__)

class ModelInfo:
    def __init__(self, model: nn.Module):
        self.model = model
        self._param_count = None
        self._trainable_params = None
        self._model_size_mb = None
        self._feature_shapes = None

    @property
    def total_parameters(self) -> int:
        if self._param_count is None:
            self._param_count = sum(p.numel() for p in self.model.parameters())
        return self._param_count

    @property
    def trainable_parameters(self) -> int:
        if self._trainable_params is None:
            self._trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return self._trainable_params

    @property
    def model_size_mb(self) -> float:
        if self._model_size_mb is None:
            param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
            self._model_size_mb = (param_size + buffer_size) / (1024 * 1024)
        return self._model_size_mb

    def get_summary(self) -> Dict[str, Any]:
        return {
            'total_parameters': self.total_parameters,
            'trainable_parameters': self.trainable_parameters,
            'frozen_parameters': self.total_parameters - self.trainable_parameters,
            'model_size_mb': round(self.model_size_mb, 2),
            'model_type': type(self.model).__name__
        }

def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def get_model_info(model: nn.Module) -> ModelInfo:
    return ModelInfo(model)

def calculate_patch_grid(img_size: int, patch_size: int) -> Tuple[int, int]:
    if img_size % patch_size != 0:
        raise ValueError(f"Image size {img_size} must be divisible by patch size {patch_size}")

    num_patches = img_size // patch_size
    return num_patches, num_patches

def calculate_feature_shapes(
    backbone_name: str,
    img_size: int,
    extract_layers: List[int],
    embed_dim: int,
    patch_size: int
) -> List[Tuple[int, int, int]]:
    num_patches_h, num_patches_w = calculate_patch_grid(img_size, patch_size)

    # All ViT layers have the same spatial dimensions
    feature_shapes = []
    for layer_idx in extract_layers:
        feature_shapes.append((embed_dim, num_patches_h, num_patches_w))

    return feature_shapes

def sequence_to_spatial(
    sequence: torch.Tensor,
    patch_grid: Tuple[int, int],
    include_cls_token: bool = True
) -> torch.Tensor:
    B, N, C = sequence.shape
    H, W = patch_grid

    if include_cls_token:
        # Remove CLS token
        sequence = sequence[:, 1:, :]
        N = N - 1

    if N != H * W:
        raise ValueError(f"Sequence length {N} doesn't match patch grid {H}x{W} = {H*W}")

    # Reshape to spatial format
    spatial = sequence.transpose(1, 2).reshape(B, C, H, W)
    return spatial

def interpolate_features(
    features: torch.Tensor,
    target_size: Tuple[int, int],
    mode: str = 'bilinear',
    align_corners: bool = False
) -> torch.Tensor:
    if features.shape[-2:] == target_size:
        return features

    return F.interpolate(
        features,
        size=target_size,
        mode=mode,
        align_corners=align_corners
    )

def freeze_model(model: nn.Module, freeze: bool = True) -> None:
    for param in model.parameters():
        param.requires_grad = not freeze

    if freeze:
        model.eval()
        logger.info(f"Frozen {count_parameters(model)} parameters in {type(model).__name__}")
    else:
        model.train()
        logger.info(f"Unfrozen {count_parameters(model)} parameters in {type(model).__name__}")

def setup_feature_hooks(
    model: nn.Module,
    layer_names: List[str],
    hook_fn: Optional[Callable] = None
) -> List[torch.utils.hooks.RemovableHandle]: # type: ignore
    if hook_fn is None:
        features = {}

        def default_hook_fn(name):
            def hook(module, input, output):
                features[name] = output
            return hook

        hook_fn = default_hook_fn

    handles = []
    for name in layer_names:
        module = dict(model.named_modules())[name]
        handle = module.register_forward_hook(hook_fn(name))
        handles.append(handle)

    return handles

def cleanup_hooks(handles: List[torch.utils.hooks.RemovableHandle]) -> None: # type: ignore
    for handle in handles:
        handle.remove()

def apply_gradient_checkpointing(
    model: nn.Module,
    enable: bool = True,
    modules_to_checkpoint: Optional[List[str]] = None
) -> None:
    if not enable:
        return

    if modules_to_checkpoint is None:
        # Default: checkpoint transformer blocks
        modules_to_checkpoint = ['blocks', 'layers', 'encoder']

    for name, module in model.named_modules():
        if any(checkpoint_name in name for checkpoint_name in modules_to_checkpoint):
            if hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = True # type: ignore
            logger.debug(f"Applied gradient checkpointing to {name}")

def save_model_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_info': get_model_info(model).get_summary(),
        'model_class': type(model).__name__
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if epoch is not None:
        checkpoint['epoch'] = epoch

    if step is not None:
        checkpoint['step'] = step

    if metadata is not None:
        checkpoint['metadata'] = metadata

    try:
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        raise RuntimeError(f"Checkpoint save failed: {e}")

def load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    strict: bool = True,
    map_location: Optional[Union[str, torch.device]] = None
) -> Dict[str, Any]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)

        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Extract metadata
        metadata = {
            'epoch': checkpoint.get('epoch'),
            'step': checkpoint.get('step'),
            'model_info': checkpoint.get('model_info'),
            'model_class': checkpoint.get('model_class'),
            'metadata': checkpoint.get('metadata')
        }

        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        return metadata

    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise RuntimeError(f"Checkpoint load failed: {e}")

def initialize_weights(model: nn.Module, init_type: str = 'xavier_uniform') -> None:
    init_functions = {
        'xavier_uniform': nn.init.xavier_uniform_,
        'xavier_normal': nn.init.xavier_normal_,
        'kaiming_uniform': nn.init.kaiming_uniform_,
        'kaiming_normal': nn.init.kaiming_normal_
    }

    if init_type not in init_functions:
        raise ValueError(f"Unknown init_type: {init_type}. Available: {list(init_functions.keys())}")

    init_fn = init_functions[init_type]

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            init_fn(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    logger.info(f"Initialized model weights with {init_type}")

def get_layer_by_name(model: nn.Module, layer_name: str) -> nn.Module:
    try:
        return dict(model.named_modules())[layer_name]
    except KeyError:
        available_layers = list(dict(model.named_modules()).keys())
        raise ValueError(f"Layer '{layer_name}' not found. Available layers: {available_layers[:10]}...")

def print_model_summary(model: nn.Module, input_size: Optional[Tuple[int, ...]] = None) -> None:
    info = get_model_info(model)
    summary = info.get_summary()

    print("=" * 60)
    print(f"MODEL SUMMARY: {summary['model_type']}")
    print("=" * 60)
    print(f"Total Parameters:     {summary['total_parameters']:,}")
    print(f"Trainable Parameters: {summary['trainable_parameters']:,}")
    print(f"Frozen Parameters:    {summary['frozen_parameters']:,}")
    print(f"Model Size:           {summary['model_size_mb']} MB")

    if input_size is not None:
        print(f"Input Size:           {input_size}")

    print("=" * 60)

def validate_feature_compatibility(
    backbone_features: List[torch.Tensor],
    decoder_expected_channels: List[int]
) -> None:
    if len(backbone_features) != len(decoder_expected_channels):
        raise ValueError(
            f"Feature count mismatch: backbone provides {len(backbone_features)} features, "
            f"decoder expects {len(decoder_expected_channels)}"
        )

    for i, (feat, expected_c) in enumerate(zip(backbone_features, decoder_expected_channels)):
        actual_c = feat.size(1)
        if actual_c != expected_c:
            logger.warning(
                f"Channel mismatch at feature {i}: backbone={actual_c}, decoder expects={expected_c}"
            )

def create_feature_pyramid(
    features: List[torch.Tensor],
    target_channels: int = 256
) -> List[torch.Tensor]:
    pyramid_features = []

    for feat in features:
        if feat.size(1) != target_channels:
            # Add 1x1 conv to match target channels
            conv = nn.Conv2d(feat.size(1), target_channels, 1).to(feat.device)
            feat = conv(feat)

        pyramid_features.append(feat)

    return pyramid_features