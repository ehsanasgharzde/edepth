# FILE: models/model_fixed.py
# ehsanasgharzde - COMPLETE DPT MODEL INTEGRATION WITH BACKBONE AND DECODER

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any
import logging
import warnings
import traceback
from .backbones.backbone_fixed import ViT
from .decoders.decoder_fixed import DPT
import torch.nn.functional as F

logger = logging.getLogger(__name__)

SUPPORTED_BACKBONES = {
    "vit_small_patch16_224": {"patch_size": 16, "img_size": 224, "embed_dim": 384},
    "vit_base_patch16_224": {"patch_size": 16, "img_size": 224, "embed_dim": 768},
    "vit_base_patch16_384": {"patch_size": 16, "img_size": 384, "embed_dim": 768},
    "vit_base_patch8_224": {"patch_size": 8, "img_size": 224, "embed_dim": 768},
    "vit_large_patch16_224": {"patch_size": 16, "img_size": 224, "embed_dim": 1024},
    "deit_small_patch16_224": {"patch_size": 16, "img_size": 224, "embed_dim": 384},
    "deit_base_patch16_224": {"patch_size": 16, "img_size": 224, "embed_dim": 768},
}

DEFAULT_MODEL_CONFIG = {
    'backbone_name': 'vit_base_patch16_224',
    'extract_layers': None,
    'decoder_channels': [256, 512, 1024, 1024],
    'patch_size': 16,
    'num_classes': 1,
    'pretrained': True,
    'use_attention': False,
    'final_activation': 'sigmoid',
    'interpolation_mode': 'bilinear'
}

MODEL_CONFIGS = {
    "vit_small_patch16_224": {
        "backbone_name": "vit_small_patch16_224",
        "patch_size": 16,
        "img_size": 224,
        "embed_dim": 384,
        "num_heads": 6,
        "num_layers": 12,
        "extract_layers": [8, 9, 10, 11],
        "decoder_channels": [128, 256, 384, 384],
        "pretrained": True,
        "use_attention": False,
        "final_activation": "sigmoid",
        "interpolation_mode": "bilinear",
        "num_classes": 1,
        "dropout": 0.0,
        "drop_path": 0.1,
        "memory_efficient": False,
        "gradient_checkpointing": False
    },
    "vit_base_patch16_224": {
        "backbone_name": "vit_base_patch16_224",
        "patch_size": 16,
        "img_size": 224,
        "embed_dim": 768,
        "num_heads": 12,
        "num_layers": 12,
        "extract_layers": [8, 9, 10, 11],
        "decoder_channels": [256, 512, 768, 768],
        "pretrained": True,
        "use_attention": False,
        "final_activation": "sigmoid",
        "interpolation_mode": "bilinear",
        "num_classes": 1,
        "dropout": 0.0,
        "drop_path": 0.1,
        "memory_efficient": False,
        "gradient_checkpointing": False
    },
    "vit_base_patch16_384": {
        "backbone_name": "vit_base_patch16_384",
        "patch_size": 16,
        "img_size": 384,
        "embed_dim": 768,
        "num_heads": 12,
        "num_layers": 12,
        "extract_layers": [8, 9, 10, 11],
        "decoder_channels": [256, 512, 768, 768],
        "pretrained": True,
        "use_attention": False,
        "final_activation": "sigmoid",
        "interpolation_mode": "bilinear",
        "num_classes": 1,
        "dropout": 0.0,
        "drop_path": 0.1,
        "memory_efficient": True,
        "gradient_checkpointing": True
    },
    "vit_base_patch8_224": {
        "backbone_name": "vit_base_patch8_224",
        "patch_size": 8,
        "img_size": 224,
        "embed_dim": 768,
        "num_heads": 12,
        "num_layers": 12,
        "extract_layers": [8, 9, 10, 11],
        "decoder_channels": [256, 512, 768, 768],
        "pretrained": True,
        "use_attention": True,
        "final_activation": "sigmoid",
        "interpolation_mode": "bilinear",
        "num_classes": 1,
        "dropout": 0.0,
        "drop_path": 0.1,
        "memory_efficient": True,
        "gradient_checkpointing": True
    },
    "vit_large_patch16_224": {
        "backbone_name": "vit_large_patch16_224",
        "patch_size": 16,
        "img_size": 224,
        "embed_dim": 1024,
        "num_heads": 16,
        "num_layers": 24,
        "extract_layers": [20, 21, 22, 23],
        "decoder_channels": [512, 768, 1024, 1024],
        "pretrained": True,
        "use_attention": True,
        "final_activation": "sigmoid",
        "interpolation_mode": "bilinear",
        "num_classes": 1,
        "dropout": 0.0,
        "drop_path": 0.2,
        "memory_efficient": True,
        "gradient_checkpointing": True
    },
    "deit_small_patch16_224": {
        "backbone_name": "deit_small_patch16_224",
        "patch_size": 16,
        "img_size": 224,
        "embed_dim": 384,
        "num_heads": 6,
        "num_layers": 12,
        "extract_layers": [8, 9, 10, 11],
        "decoder_channels": [128, 256, 384, 384],
        "pretrained": True,
        "use_attention": False,
        "final_activation": "sigmoid",
        "interpolation_mode": "bilinear",
        "num_classes": 1,
        "dropout": 0.0,
        "drop_path": 0.1,
        "memory_efficient": False,
        "gradient_checkpointing": False
    },
    "deit_base_patch16_224": {
        "backbone_name": "deit_base_patch16_224",
        "patch_size": 16,
        "img_size": 224,
        "embed_dim": 768,
        "num_heads": 12,
        "num_layers": 12,
        "extract_layers": [8, 9, 10, 11],
        "decoder_channels": [256, 512, 768, 768],
        "pretrained": True,
        "use_attention": False,
        "final_activation": "sigmoid",
        "interpolation_mode": "bilinear",
        "num_classes": 1,
        "dropout": 0.0,
        "drop_path": 0.1,
        "memory_efficient": False,
        "gradient_checkpointing": False
    }
}


class edepth(nn.Module):
    def __init__(self,
                 backbone_name: str = 'vit_base_patch16_224',  # Name of the vision transformer backbone to use
                 extract_layers: Optional[List[int]] = None,    # Indices of layers to extract features from
                 decoder_channels: Optional[List[int]] = None,  # Decoder channel configuration
                 patch_size: int = 16,                          # Patch size used in the ViT
                 num_classes: int = 1,                          # Number of output classes (usually 1 for depth)
                 pretrained: bool = True,                       # Whether to use pretrained backbone weights
                 use_attention: bool = False,                   # Whether to use attention in decoder
                 final_activation: str = 'sigmoid',             # Final activation function for output
                 interpolation_mode: str = 'bilinear',          # Interpolation mode for resizing output
                 **kwargs):                                     # Additional arguments for flexibility
                 
        super().__init__()  # Initialize the parent nn.Module
        
        # Store configuration in a dictionary for easy access and logging
        self.config = {
            'backbone_name': backbone_name,
            'extract_layers': extract_layers,
            'decoder_channels': decoder_channels,
            'patch_size': patch_size,
            'num_classes': num_classes,
            'pretrained': pretrained,
            'use_attention': use_attention,
            'final_activation': final_activation,
            'interpolation_mode': interpolation_mode,
            **kwargs
        }

        # Log the backbone being used
        logger.info(f"Initializing edepth model with backbone: {backbone_name}")
        
        # Validate the model configuration
        self._validate_config()
        
        # Initialize the vision transformer backbone
        self.backbone = ViT(
            model_name=backbone_name,
            extract_layers=extract_layers,
            pretrained=pretrained,
            **kwargs
        )
        
        # Extract feature dimensions from the backbone
        self.feature_dims = self._get_backbone_feature_dims()
        logger.info(f"Backbone feature dimensions: {self.feature_dims}")
        
        # Use default decoder channels if none are provided
        if decoder_channels is None:
            decoder_channels = self._get_default_decoder_channels()
        
        # Validate the format and correctness of decoder channels
        self._validate_decoder_channels(decoder_channels)
        
        # Initialize the decoder with extracted features and configuration
        self.decoder = DPT(
            backbone_channels=self.feature_dims,
            decoder_channels=decoder_channels,
            num_stages=len(self.feature_dims),
            use_attention=use_attention,
            final_activation=final_activation
        )
        
        # Collect model meta-information (architecture, stats, etc.)
        self.model_info = self._collect_model_info()
        
        # Log successful initialization and parameter count
        logger.info(f"Successfully initialized edepth model")
        logger.info(f"Total parameters: {self.count_parameters():,}")

    def _get_model_config(self, backbone_name: str, **kwargs) -> Dict[str, Any]:
        # Check if the specified backbone exists in predefined configurations
        if backbone_name not in MODEL_CONFIGS:
            available_backbones = list(MODEL_CONFIGS.keys())  # Get list of supported backbones
            raise ValueError(f"Backbone '{backbone_name}' not found. Available backbones: {available_backbones}")
        
        # Retrieve and copy the default config for the specified backbone
        config = MODEL_CONFIGS[backbone_name].copy()
        
        # Update the config with any user-provided overrides
        config.update(kwargs)
        
        # Log the final model config used
        logger.info(f"Retrieved model config for '{backbone_name}' with overrides: {kwargs}")
        
        return config  # Return the complete model configuration

    def _validate_config(self):
        # Define list of supported backbones
        supported_backbones = list(SUPPORTED_BACKBONES.keys())
        
        # Validate that the specified backbone is supported
        if self.config['backbone_name'] not in supported_backbones:
            raise ValueError(f"Unsupported backbone_name: '{self.config['backbone_name']}'. "
                             f"Supported options: {supported_backbones}")
        
        # Validate the patch size
        if self.config['patch_size'] not in [8, 14, 16, 32]:
            raise ValueError(f"Invalid patch_size: {self.config['patch_size']}. "
                             f"Supported values: [8, 14, 16, 32]")
        
        # Validate the final activation function
        supported_activations = ['sigmoid', 'softmax', 'tanh', 'none']
        if self.config['final_activation'] not in supported_activations:
            raise ValueError(f"Unsupported final_activation: '{self.config['final_activation']}'. "
                             f"Choose from: {supported_activations}")
        
        # Validate the interpolation mode
        supported_interp_modes = ['bilinear', 'nearest', 'bicubic']
        if self.config['interpolation_mode'] not in supported_interp_modes:
            raise ValueError(f"Unsupported interpolation_mode: '{self.config['interpolation_mode']}'. "
                             f"Supported modes: {supported_interp_modes}")
        
        # Validate extract_layers if provided
        if self.config['extract_layers'] is not None:
            # Ensure it's a list of integers
            if not isinstance(self.config['extract_layers'], list) or not all(isinstance(i, int) for i in self.config['extract_layers']):
                raise TypeError("extract_layers must be a list of integers.")
            # Check for duplicate values
            if len(set(self.config['extract_layers'])) != len(self.config['extract_layers']):
                raise ValueError("extract_layers contains duplicate values.")
            # Check for non-negative values
            if not all(i >= 0 for i in self.config['extract_layers']):
                raise ValueError("extract_layers must contain non-negative integers.")

        # Validate decoder_channels if provided
        if self.config['decoder_channels'] is not None:
            # Ensure it's a list of positive integers
            if not isinstance(self.config['decoder_channels'], list) or not all(isinstance(c, int) and c > 0 for c in self.config['decoder_channels']):
                raise ValueError("decoder_channels must be a list of positive integers.")

    def _get_backbone_feature_dims(self) -> List[int]:
        # Retrieve feature information from the backbone
        feature_info = self.backbone.get_feature_info()
        
        # Use extract_layers from config if provided; otherwise, use default from backbone
        extract_layers = self.config['extract_layers']
        if extract_layers is None:
            extract_layers = self.backbone.extract_layers
        
        feature_dims = []
        # Iterate through each specified extract layer index
        for i in extract_layers:
            # Ensure the index is within the bounds of the available features
            if i >= len(feature_info['feature_shapes']):
                raise IndexError(f"extract_layers index {i} is out of range for backbone features.")
            # Append the embedding dimension for each selected feature
            feature_dims.append(feature_info['embed_dim'])
        
        # Return the list of feature dimensions corresponding to extract_layers
        return feature_dims

    def _get_default_decoder_channels(self) -> List[int]:
        # Reverse the order of feature dimensions for top-down decoding
        base_channels = self.feature_dims[::-1]
        
        decoder_channels = []
        # Reduce each feature dimension by half, but not below 32
        for ch in base_channels:
            ch = max(ch // 2, 32)
            decoder_channels.append(ch)
        
        # Return the computed default decoder channel configuration
        return decoder_channels

    def _validate_decoder_channels(self, decoder_channels: List[int]):
        # Ensure decoder_channels is a list
        if not isinstance(decoder_channels, list):
            raise TypeError(f"Expected decoder_channels to be a list, got {type(decoder_channels)}")

        # Validate that the number of decoder channels matches the number of features from the backbone
        if len(decoder_channels) != len(self.feature_dims):
            raise ValueError(
                f"decoder_channels length ({len(decoder_channels)}) must match "
                f"feature_dims length ({len(self.feature_dims)})"
            )

        # Ensure each channel value is a positive integer
        for idx, ch in enumerate(decoder_channels):
            if not isinstance(ch, int) or ch <= 0:
                raise ValueError(f"decoder_channels[{idx}] must be a positive integer, got {ch}")

        # Warn about sudden jumps between decoder stages that may be unintended
        for i in range(1, len(decoder_channels)):
            prev, curr = decoder_channels[i - 1], decoder_channels[i]
            ratio = curr / prev
            if ratio > 2 or ratio < 0.25:
                warnings.warn(
                    f"Unusual decoder channel jump: {prev} -> {curr} (ratio {ratio:.2f}). "
                    f"Check if this is intentional.",
                    RuntimeWarning
                )

        # Warn if total number of decoder channels is too large (may impact memory usage)
        total_channels = sum(decoder_channels)
        if total_channels > 2048:
            warnings.warn(
                f"Total decoder channels ({total_channels}) may be excessive and memory-intensive.",
                RuntimeWarning
            )

    def _collect_model_info(self) -> Dict[str, Any]:
        # Initialize an empty dictionary to store model metadata
        model_info = {}
        
        # Store the name of the backbone class
        model_info["backbone_name"] = self.backbone.__class__.__name__
        # Count the number of parameters in the backbone
        model_info["backbone_params"] = sum(p.numel() for p in self.backbone.parameters())
        
        # Store the list of feature dimensions extracted from the backbone
        model_info["feature_dims"] = self.feature_dims
        # Count how many feature maps are being extracted
        model_info["num_features"] = len(self.feature_dims)
        
        # Store the decoder class name
        model_info["decoder_class"] = self.decoder.__class__.__name__
        # Count the number of parameters in the decoder
        model_info["decoder_params"] = sum(p.numel() for p in self.decoder.parameters())
        
        # Store number of output classes as defined in the config
        model_info["num_classes"] = self.config.get("num_classes", 1)
        # Store the final activation function used in the decoder output
        model_info["final_activation"] = self.config.get("final_activation", "sigmoid")
        
        # Count the total number of parameters in the model
        model_info["total_params"] = self.count_parameters()
        
        # Store the patch size used in the input projection of the backbone
        model_info["input_patch_size"] = self.config.get("patch_size", 16)
        # Store the interpolation method used for upsampling in the decoder
        model_info["interpolation_mode"] = self.config.get("interpolation_mode", "bilinear")
        
        # Return the compiled model metadata
        return model_info

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Check if input is a PyTorch tensor
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")
        # Ensure the tensor has 4 dimensions [Batch, Channels, Height, Width]
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor [B,C,H,W], got {x.dim()}D tensor")
        # Verify that the input has 3 channels (e.g., RGB image)
        if x.size(1) != 3:
            raise ValueError(f"Expected 3 input channels (RGB), got {x.size(1)}")

        # Log the shape of the input tensor
        logger.debug(f"Forward pass input shape: {x.shape}")

        try:
            # Extract features using the vision transformer backbone
            features = self.backbone(x)
            logger.debug(f"Extracted {len(features)} feature maps")
            
            # Log the shape of each extracted feature map
            for i, feat in enumerate(features):
                logger.debug(f"Feature {i} shape: {feat.shape}")

            # Pass the extracted features through the decoder to get depth output
            depth = self.decoder(features)
            logger.debug(f"Decoder output shape: {depth.shape}")

            # If the output size does not match the input, resize it using interpolation
            if depth.shape[-2:] != x.shape[-2:]:
                depth = F.interpolate(
                    depth, 
                    size=x.shape[-2:], 
                    mode=self.config['interpolation_mode'], 
                    align_corners=False
                )
                logger.debug(f"Resized output to: {depth.shape}")

            # Validate the shape of the final output
            self._validate_output(depth, x)
            return depth

        # Catch and log any exceptions that occur during the forward pass
        except Exception as e:
            logger.error(f"Forward pass failed: {str(e)}")
            logger.error(f"Input shape: {x.shape}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Model forward pass failed: {str(e)}")

    def _validate_output(self, output: torch.Tensor, input_tensor: torch.Tensor):
        # Ensure the output is a 4D tensor [B, C, H, W]
        if output.dim() != 4:
            raise ValueError(f"Output tensor must be 4D [B,C,H,W], got {output.dim()}D")

        # Batch size must match the input
        if output.size(0) != input_tensor.size(0):
            raise ValueError(f"Batch size mismatch: output {output.size(0)} vs input {input_tensor.size(0)}")

        # Spatial dimensions must match the input
        if output.size(-2) != input_tensor.size(-2) or output.size(-1) != input_tensor.size(-1):
            raise ValueError(
                f"Spatial size mismatch: output {output.shape[-2:]} vs input {input_tensor.shape[-2:]}"
            )

        # Channel size must match the expected number of classes
        expected_channels = self.config.get('num_classes', 1)
        if output.size(1) != expected_channels:
            raise ValueError(f"Output channel size mismatch: expected {expected_channels}, got {output.size(1)}")

        # Check for numerical stability
        if torch.isnan(output).any():
            raise ValueError("Output contains NaN values")
        if torch.isinf(output).any():
            raise ValueError("Output contains Inf values")

        # During training, ensure the output supports gradient computation
        if self.training:
            if output.requires_grad:
                logger.debug("Output requires gradient - OK for training")
            else:
                logger.warning("Output does NOT require gradient during training")

        logger.debug("Output validation passed")

    def count_parameters(self, trainable_only: bool = True) -> int:
        """
        Count the number of parameters in the model.

        Args:
            trainable_only (bool): If True, count only parameters that require gradients.
                                If False, count all parameters.

        Returns:
            int: Total number of parameters.
        """
        def _count_params(module):
            return sum(
                p.numel()
                for p in module.parameters()
                if (p.requires_grad or not trainable_only)
            )

        # Count parameters in the entire model
        total_params = _count_params(self)

        # Optionally log backbone and decoder parameter counts
        backbone_params = _count_params(self.backbone) if hasattr(self, 'backbone') else 0
        decoder_params = _count_params(self.decoder) if hasattr(self, 'decoder') else 0

        # Log the parameter counts in a human-readable format
        logger.debug(f"Total parameters: {total_params:,}")
        logger.debug(f"Backbone parameters: {backbone_params:,}")
        logger.debug(f"Decoder parameters: {decoder_params:,}")

        return total_params

    def get_model_summary(self, input_size: tuple = (1, 3, 224, 224)) -> Dict[str, Any]:
        """
        Generate a summary of the model architecture including parameter counts and memory usage.

        Args:
            input_size (tuple): Shape of the dummy input tensor (default: (1, 3, 224, 224)).

        Returns:
            Dict[str, Any]: Summary dictionary containing per-module details and memory estimates.
        """
        summary = {}  # Dictionary to store summary info per module
        hooks = []    # List to store forward hooks
        device = next(self.parameters()).device  # Get the device where model parameters are located

        # Hook function to record output shape and parameter info for each leaf module
        def hook_fn(module, input, output):
            module_name = module.__class__.__name__
            summary[module_name] = {
                "output_shape": tuple(output.shape) if isinstance(output, torch.Tensor) else "Multiple/Unknown",
                "num_params": sum(p.numel() for p in module.parameters()),
                "trainable_params": sum(p.numel() for p in module.parameters() if p.requires_grad)
            }

        # Register hooks on all leaf modules (modules without children)
        for name, module in self.named_modules():
            if len(list(module.children())) == 0:
                hooks.append(module.register_forward_hook(hook_fn))

        # Create a dummy input tensor on the model's device
        dummy_input = torch.randn(*input_size).to(device)

        # Run a forward pass with dummy input to trigger hooks
        self.eval()
        with torch.no_grad():
            try:
                self(dummy_input)
            except Exception:
                # Forward pass may fail due to input assumptions, ignore exceptions here
                pass

        # Remove all hooks to clean up
        for hook in hooks:
            hook.remove()

        # Calculate total and trainable parameters from summary
        total_params = sum(info["num_params"] for info in summary.values())
        trainable_params = sum(info["trainable_params"] for info in summary.values())

        # Calculate memory footprint for parameters (4 bytes per param) in MB
        params_mem = total_params * 4 / (1024 ** 2)

        # Calculate approximate activation memory in MB
        activations_mem = 0
        for info in summary.values():
            shape = info["output_shape"]
            if isinstance(shape, tuple):
                # Product of output tensor dimensions * 4 bytes per element
                activations_mem += (torch.prod(torch.tensor(shape)) * 4).item()
        activations_mem /= (1024 ** 2)

        # Collect all info in a dictionary and return
        model_info = {
            "modules": summary,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "params_memory_MB": round(params_mem, 2),
            "activations_memory_MB": round(activations_mem, 2),
        }

        return model_info

    def get_feature_shapes(self, input_size: tuple = (1, 3, 224, 224)) -> Dict[str, tuple]:
        # Set the model to evaluation mode
        self.eval()
        # Get the device where the model's parameters are located
        device = next(self.parameters()).device
        # Initialize a dictionary to store shapes
        shapes = {}

        # Disable gradient calculation for inference
        with torch.no_grad():
            # Create a dummy input tensor with the specified input size on the correct device
            x = torch.randn(*input_size).to(device)
            # Store the input tensor shape
            shapes['input'] = x.shape

            # Check if the backbone exists, raise error if not
            if hasattr(self, 'backbone'):
                # Pass the input through the backbone to extract features
                features = self.backbone(x)
                # If features are returned as a dict, convert to list
                if isinstance(features, dict):
                    features = list(features.values())
                # Store the shape of each feature map extracted from the backbone
                for i, feat in enumerate(features):
                    shapes[f'backbone_feature_{i}'] = feat.shape
            else:
                # Raise an error if backbone is not defined
                raise RuntimeError("Backbone not defined.")

            # Check if the decoder exists, raise error if not
            if hasattr(self, 'decoder'):
                # Pass the extracted features through the decoder to get output depth map
                depth = self.decoder(features)
                # Store the decoder output shape
                shapes['decoder_output'] = depth.shape
            else:
                # Raise an error if decoder is not defined
                raise RuntimeError("Decoder not defined.")

        # Return the dictionary containing shapes of inputs, backbone features, and decoder output
        return shapes

    def freeze_backbone(self):
        # Check if the model has a 'backbone' attribute; raise error if missing
        if not hasattr(self, 'backbone'):
            raise AttributeError("Model has no attribute 'backbone' to freeze.")

        frozen_count = 0  # Counter for the total number of frozen parameters
        # Iterate over all parameters of the backbone
        for param in self.backbone.parameters():
            # Disable gradient computation for the parameter to freeze it
            param.requires_grad = False
            # Increment the counter by the number of elements in the parameter tensor
            frozen_count += param.numel()

        # Log the total number of frozen parameters
        logger.info(f"Froze backbone parameters: {frozen_count} parameters set to requires_grad=False")

    def unfreeze_backbone(self):
        # Check if the model has a 'backbone' attribute; raise error if missing
        if not hasattr(self, 'backbone'):
            raise AttributeError("Model has no attribute 'backbone' to unfreeze.")

        unfrozen_count = 0  # Counter for the total number of unfrozen parameters
        # Iterate over all parameters of the backbone
        for param in self.backbone.parameters():
            # Enable gradient computation for the parameter to unfreeze it
            param.requires_grad = True
            # Increment the counter by the number of elements in the parameter tensor
            unfrozen_count += param.numel()

        # Log the total number of unfrozen parameters
        logger.info(f"Unfroze backbone parameters: {unfrozen_count} parameters set to requires_grad=True")

    def load_pretrained_weights(self, checkpoint_path: str, strict: bool = True):
        import os
        import torch

        # Check if the checkpoint file exists; raise error if not found
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        try:
            # Load the checkpoint from the file, mapping to CPU by default
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # If checkpoint is a dict and contains 'state_dict', use that for loading weights
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                # Otherwise, assume checkpoint itself is the state dict
                state_dict = checkpoint

            # Load the state dictionary into the model; return missing and unexpected keys
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=strict)

            # Log success message
            logger.info(f"Loaded pretrained weights from {checkpoint_path}")
            # Log warnings if there are missing keys in the loaded state dict
            if missing_keys:
                logger.warning(f"Missing keys when loading weights: {missing_keys}")
            # Log warnings if there are unexpected keys in the checkpoint
            if unexpected_keys:
                logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")

        except Exception as e:
            # Log error details and raise a runtime error if loading fails
            logger.error(f"Failed to load pretrained weights: {e}")
            raise RuntimeError(f"Loading pretrained weights failed: {e}")

    def save_checkpoint(self, checkpoint_path: str, include_config: bool = True, optimizer: Optional[torch.optim.Optimizer] = None, epoch: Optional[int] = None, step: Optional[int] = None):
        import torch

        # Prepare the checkpoint dictionary with the model's state dictionary
        checkpoint = {
            'model_state_dict': self.state_dict()
        }

        # Optionally include the model configuration in the checkpoint
        if include_config and hasattr(self, 'config'):
            checkpoint['config'] = self.config

        # If an optimizer is provided, save its state dictionary as well
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        # Prepare metadata dictionary with optional epoch and step information
        metadata = {}
        if epoch is not None:
            metadata['epoch'] = epoch
        if step is not None:
            metadata['step'] = step
        # Add metadata to checkpoint if any metadata is provided
        if metadata:
            checkpoint['metadata'] = metadata

        try:
            # Save the checkpoint dictionary to the specified path
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Checkpoint saved at {checkpoint_path}")
        except Exception as e:
            # Log error and raise exception if saving fails
            logger.error(f"Failed to save checkpoint: {e}")
            raise RuntimeError(f"Saving checkpoint failed: {e}")