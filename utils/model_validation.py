# FILE: utils/model_validation.py
# ehsanasgharzde - FIXED REDUNDANT CODE BY EXTRACTING PURE FUNCTIONS AND BASECLASS LEVEL METHODS

import torch
from typing import List, Optional, Tuple

from logger.logger import setup_logging 

# Setup logger for factory operations
logger = setup_logging(__file__)

class ModelValidationError(ValueError):
    pass

class TensorValidationError(ModelValidationError):
    pass

class ConfigValidationError(ModelValidationError):
    pass

def validate_tensor_input(
    tensor: torch.Tensor, 
    name: str = "input",
    expected_dims: Optional[int] = None,
    expected_shape: Optional[Tuple[int, ...]] = None,
    min_dims: Optional[int] = None,
    max_dims: Optional[int] = None,
    allow_nan: bool = False,
    allow_inf: bool = False
) -> None:
    # Check if input is a tensor
    if not isinstance(tensor, torch.Tensor):
        raise TensorValidationError(f"{name} must be a torch.Tensor, got {type(tensor)}")

    # Check dimensions
    actual_dims = tensor.dim()

    if expected_dims is not None and actual_dims != expected_dims:
        raise TensorValidationError(
            f"{name} must have {expected_dims} dimensions, got {actual_dims} (shape: {tensor.shape})"
        )

    if min_dims is not None and actual_dims < min_dims:
        raise TensorValidationError(
            f"{name} must have at least {min_dims} dimensions, got {actual_dims}"
        )

    if max_dims is not None and actual_dims > max_dims:
        raise TensorValidationError(
            f"{name} must have at most {max_dims} dimensions, got {actual_dims}"
        )

    # Check exact shape if specified
    if expected_shape is not None:
        if tensor.shape != expected_shape:
            raise TensorValidationError(
                f"{name} must have shape {expected_shape}, got {tensor.shape}"
            )

    # Check for NaN values
    if not allow_nan and torch.isnan(tensor).any():
        raise TensorValidationError(f"{name} contains NaN values")

    # Check for infinite values
    if not allow_inf and torch.isinf(tensor).any():
        raise TensorValidationError(f"{name} contains infinite values")

def validate_feature_tensors(
    features: List[torch.Tensor],
    expected_channels: Optional[List[int]] = None,
    expected_count: Optional[int] = None,
    name: str = "features"
) -> None:
    if not isinstance(features, list):
        raise TensorValidationError(f"{name} must be a list, got {type(features)}")

    if expected_count is not None and len(features) != expected_count:
        raise TensorValidationError(
            f"{name} must contain {expected_count} tensors, got {len(features)}"
        )

    if expected_channels is not None:
        if len(features) != len(expected_channels):
            raise TensorValidationError(
                f"Feature count mismatch: expected {len(expected_channels)}, got {len(features)}"
            )

        for idx, (feat, expected_c) in enumerate(zip(features, expected_channels)):
            validate_tensor_input(feat, f"{name}[{idx}]", expected_dims=4)

            if feat.size(1) != expected_c:
                raise TensorValidationError(
                    f"Channel mismatch at {name}[{idx}]: expected {expected_c}, got {feat.size(1)}"
                )

def validate_backbone_name(
    backbone_name: str,
    available_backbones: List[str],
    name: str = "backbone_name"
) -> None:
    if not isinstance(backbone_name, str):
        raise ConfigValidationError(f"{name} must be a string, got {type(backbone_name)}")

    if not backbone_name:
        raise ConfigValidationError(f"{name} cannot be empty")

    if backbone_name not in available_backbones:
        raise ConfigValidationError(
            f"Backbone '{backbone_name}' not found. Available backbones: {available_backbones}"
        )

def validate_decoder_channels(
    decoder_channels: List[int],
    min_channels: int = 1,
    max_channels: int = 2048,
    name: str = "decoder_channels"
) -> None:
    if not isinstance(decoder_channels, list):
        raise ConfigValidationError(f"{name} must be a list, got {type(decoder_channels)}")

    if len(decoder_channels) == 0:
        raise ConfigValidationError(f"{name} cannot be empty")

    for idx, ch in enumerate(decoder_channels):
        if not isinstance(ch, int):
            raise ConfigValidationError(
                f"{name}[{idx}] must be an integer, got {type(ch)}"
            )

        if ch < min_channels or ch > max_channels:
            raise ConfigValidationError(
                f"{name}[{idx}] must be between {min_channels} and {max_channels}, got {ch}"
            )

def validate_patch_size(
    patch_size: int,
    img_size: int,
    allowed_patch_sizes: List[int] = [8, 16, 32],
    name: str = "patch_size"
) -> None:
    if not isinstance(patch_size, int):
        raise ConfigValidationError(f"{name} must be an integer, got {type(patch_size)}")

    if patch_size not in allowed_patch_sizes:
        raise ConfigValidationError(
            f"{name} must be one of {allowed_patch_sizes}, got {patch_size}"
        )

    if img_size % patch_size != 0:
        raise ConfigValidationError(
            f"Image size {img_size} must be divisible by {name} {patch_size}"
        )

def validate_extract_layers(
    extract_layers: List[int],
    max_layers: int,
    name: str = "extract_layers"
) -> None:
    if not isinstance(extract_layers, list):
        raise ConfigValidationError(f"{name} must be a list, got {type(extract_layers)}")

    if len(extract_layers) == 0:
        raise ConfigValidationError(f"{name} cannot be empty")

    for idx, layer in enumerate(extract_layers):
        if not isinstance(layer, int):
            raise ConfigValidationError(
                f"{name}[{idx}] must be an integer, got {type(layer)}"
            )

        if layer < 0 or layer >= max_layers:
            raise ConfigValidationError(
                f"{name}[{idx}] must be between 0 and {max_layers-1}, got {layer}"
            )

def validate_model_output(
    output: torch.Tensor,
    expected_shape: Optional[Tuple[int, ...]] = None,
    expected_range: Optional[Tuple[float, float]] = None,
    name: str = "model_output"
) -> None:
    validate_tensor_input(output, name, allow_nan=False, allow_inf=False)

    if expected_shape is not None:
        actual_shape = output.shape
        for i, (actual, expected) in enumerate(zip(actual_shape, expected_shape)):
            if expected is not None and actual != expected:
                raise TensorValidationError(
                    f"{name} shape mismatch at dimension {i}: expected {expected}, got {actual}"
                )

    if expected_range is not None:
        min_val, max_val = expected_range
        actual_min = output.min().item()
        actual_max = output.max().item()

        if actual_min < min_val or actual_max > max_val:
            raise TensorValidationError(
                f"{name} values out of range [{min_val}, {max_val}]: got [{actual_min:.4f}, {actual_max:.4f}]"
            )

def validate_spatial_dimensions(
    height: int,
    width: int,
    patch_size: int,
    name: str = "spatial_dimensions"
) -> None:
    if not isinstance(height, int) or height <= 0:
        raise ConfigValidationError(f"Height must be a positive integer, got {height}")

    if not isinstance(width, int) or width <= 0:
        raise ConfigValidationError(f"Width must be a positive integer, got {width}")

    if height % patch_size != 0:
        raise ConfigValidationError(
            f"Height {height} not divisible by patch size {patch_size}"
        )

    if width % patch_size != 0:
        raise ConfigValidationError(
            f"Width {width} not divisible by patch size {patch_size}"
        )

def validate_interpolation_target(
    features: torch.Tensor,
    target_size: Tuple[int, int],
    name: str = "interpolation_target"
) -> None:
    validate_tensor_input(features, f"{name}_features", min_dims=3)

    if not isinstance(target_size, (tuple, list)) or len(target_size) != 2:
        raise TensorValidationError(f"{name} must be a tuple/list of 2 integers")

    target_h, target_w = target_size

    if not isinstance(target_h, int) or target_h <= 0:
        raise TensorValidationError(f"{name} height must be a positive integer, got {target_h}")

    if not isinstance(target_w, int) or target_w <= 0:
        raise TensorValidationError(f"{name} width must be a positive integer, got {target_w}")

# Convenience validation functions for common patterns
def validate_vit_input(x: torch.Tensor) -> None:
    validate_tensor_input(x, "ViT input", expected_dims=4)

def validate_dpt_features(features: List[torch.Tensor]) -> None:
    validate_feature_tensors(features, name="DPT features")

    # Ensure all features have 4 dimensions (B, C, H, W)
    for i, feat in enumerate(features):
        validate_tensor_input(feat, f"DPT feature {i}", expected_dims=4)

def validate_depth_output(depth: torch.Tensor, batch_size: int, height: int, width: int) -> None:
    validate_tensor_input(depth, "depth output", expected_dims=4)
    validate_model_output(
        depth, 
        expected_shape=(batch_size, 1, height, width),
        expected_range=(0.0, 1.0),
        name="depth output"
    )