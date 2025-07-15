# File: models/backbones/backbone_fixed.py
# ehsanasgharzde - Full Reimplementation

import torch
import torch.nn as nn
import timm
from typing import List, Optional, Tuple, Dict, Callable, Any
import logging
import torch.utils.checkpoint as checkpoint

logger = logging.getLogger(__name__)

# Recommended ViT models dictionary with comprehensive configurations
RECOMMENDED_VIT_MODELS = {
    "vit_small_patch16_224": {"patch_size": 16, "img_size": 224, "embed_dim": 384},
    "vit_base_patch16_224": {"patch_size": 16, "img_size": 224, "embed_dim": 768},
    "vit_base_patch16_384": {"patch_size": 16, "img_size": 384, "embed_dim": 768},
    "vit_base_patch8_224": {"patch_size": 8, "img_size": 224, "embed_dim": 768},
    "vit_large_patch16_224": {"patch_size": 16, "img_size": 224, "embed_dim": 1024},
    "deit_small_patch16_224": {"patch_size": 16, "img_size": 224, "embed_dim": 384}, #type: ignore
    "deit_base_patch16_224": {"patch_size": 16, "img_size": 224, "embed_dim": 768}, #type: ignore
}

class ViT(nn.Module):
    """
    Complete ViT backbone implementation for depth estimation with proper
    sequence-to-spatial conversion and multi-scale feature extraction.
    
    Key Features:
    - Proper ViT sequence format handling
    - Multi-scale feature extraction via hooks
    - Gradient checkpointing support
    - Comprehensive validation and logging
    - Dynamic patch grid calculation
    - Resource cleanup management
    """
    
    def __init__(
        self, 
        model_name: str = 'vit_base_patch16_224',
        extract_layers: Optional[List[int]] = None,
        pretrained: bool = True,
        use_checkpointing: bool = False,
        img_size: Optional[int] = None,
        patch_size: Optional[int] = None,
    ):
        super().__init__()
        
        logger.info(f"Initializing FixedViTBackbone with model: {model_name}")
        
        # Validate and set model configuration
        self._validate_and_set_config(model_name, img_size, patch_size)
        
        # Store configuration
        self.model_name = model_name
        self.pretrained = pretrained
        self.use_checkpointing = use_checkpointing
        
        # Load backbone model
        self._load_backbone_model()
        
        # Set up feature extraction layers
        self._setup_extract_layers(extract_layers)
        
        # Initialize feature storage and hooks
        self._features: Dict[int, torch.Tensor] = {}
        self._hooks: List[torch.utils.hooks.RemovableHandle] = [] #type: ignore
        self._setup_feature_hooks()
        
        logger.info(f"FixedViTBackbone initialized successfully. "
                   f"Extract layers: {self.extract_layers}, "
                   f"Patch size: {self.patch_size}, "
                   f"Embed dim: {self.embed_dim}")
    
    def _validate_and_set_config(self, model_name: str, img_size: Optional[int], patch_size: Optional[int]) -> None:
        """Validate model configuration and set parameters."""
        if model_name not in RECOMMENDED_VIT_MODELS:
            available_models = list(RECOMMENDED_VIT_MODELS.keys())
            logger.error(f"Model '{model_name}' not supported. Available: {available_models}")
            raise ValueError(f"Model '{model_name}' not in recommended models list.")
        
        # Get recommended configuration
        config = RECOMMENDED_VIT_MODELS[model_name]
        self.patch_size = patch_size or config["patch_size"]
        self.img_size = img_size or config["img_size"]
        self.expected_embed_dim = config["embed_dim"]
        
        # Validate divisibility
        if self.img_size % self.patch_size != 0:
            logger.error(f"Image size {self.img_size} not divisible by patch size {self.patch_size}")
            raise ValueError(f"Image size {self.img_size} must be divisible by patch size {self.patch_size}")
        
        logger.debug(f"Configuration validated: img_size={self.img_size}, patch_size={self.patch_size}")
    
    def _load_backbone_model(self) -> None:
        """Load the ViT backbone model with proper error handling."""
        try:
            logger.info(f"Loading TIMM model: {self.model_name} (pretrained={self.pretrained})")
            self.backbone = timm.create_model(
                self.model_name, 
                pretrained=self.pretrained, 
                features_only=False  # Critical: Don't use features_only for ViT
            )
            
            # Extract model properties
            self.embed_dim = self.backbone.embed_dim
            self.num_blocks = len(self.backbone.blocks) #type: ignore
            
            # Validate embed_dim matches expected
            if self.embed_dim != self.expected_embed_dim:
                logger.warning(f"Embed dim mismatch: expected {self.expected_embed_dim}, got {self.embed_dim}")
            
            logger.info(f"Model loaded successfully: {self.num_blocks} blocks, embed_dim={self.embed_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load ViT model '{self.model_name}': {e}")
            raise RuntimeError(f"Failed to load ViT backbone: {e}")
    
    def _setup_extract_layers(self, extract_layers: Optional[List[int]]) -> None:
        """Set up feature extraction layers with validation."""
        if extract_layers is None:
            # Default: extract from last 4 layers for multi-scale features
            self.extract_layers = [
                max(0, self.num_blocks - 4),
                max(0, self.num_blocks - 3), 
                max(0, self.num_blocks - 2),
                max(0, self.num_blocks - 1)
            ]
            logger.debug(f"Using default extract layers: {self.extract_layers}")
        else:
            self.extract_layers = extract_layers
            
        # Validate extract layers
        for layer_idx in self.extract_layers:
            if layer_idx < 0 or layer_idx >= self.num_blocks:
                logger.error(f"Invalid extract layer {layer_idx}. Must be in [0, {self.num_blocks-1}]")
                raise ValueError(f"Extract layer {layer_idx} out of range [0, {self.num_blocks-1}]")
        
        logger.info(f"Feature extraction layers configured: {self.extract_layers}")
    
    def _setup_feature_hooks(self) -> None:
        """Set up forward hooks for intermediate feature extraction."""
        logger.debug("Setting up feature extraction hooks")
        
        # Clean up existing hooks first
        self._cleanup_hooks()
        
        # Register hooks for each extract layer
        for layer_idx in self.extract_layers:
            try:
                block = self.backbone.blocks[layer_idx] #type: ignore
                hook = block.register_forward_hook(self._create_feature_hook(layer_idx)) #type: ignore
                self._hooks.append(hook)
                logger.debug(f"Registered hook for layer {layer_idx}")
            except Exception as e:
                logger.error(f"Failed to register hook for layer {layer_idx}: {e}")
                raise
        
        logger.info(f"Successfully registered {len(self._hooks)} feature hooks")
    
    def _create_feature_hook(self, layer_idx: int) -> Callable:
        """Create a forward hook function for a specific layer."""
        def hook_fn(module, input, output):
            """Hook function to capture intermediate features."""
            try:
                # Store the output tensor (sequence format: [B, N_patches+1, embed_dim])
                self._features[layer_idx] = output.clone()
                logger.debug(f"Captured features from layer {layer_idx}: {output.shape}")
            except Exception as e:
                logger.error(f"Error in hook for layer {layer_idx}: {e}")
                raise
        
        return hook_fn
    
    def _validate_input(self, x: torch.Tensor) -> None:
        """Comprehensive input validation with detailed logging."""
        logger.debug(f"Validating input tensor: {x.shape}")
        
        # Type validation
        if not isinstance(x, torch.Tensor):
            logger.error(f"Input must be torch.Tensor, got {type(x)}")
            raise TypeError(f"Input must be torch.Tensor, got {type(x)}")
        
        # Dimension validation
        if x.dim() != 4:
            logger.error(f"Input must be 4D [B,C,H,W], got {x.shape}")
            raise ValueError(f"Input must be 4D [B,C,H,W], got {x.shape}")
        
        batch_size, channels, height, width = x.shape
        
        # Channel validation
        if channels != 3:
            logger.error(f"Expected 3 channels (RGB), got {channels}")
            raise ValueError(f"Expected 3 channels (RGB), got {channels}")
        
        # Spatial dimension validation
        if height <= 0 or width <= 0:
            logger.error(f"Invalid spatial dimensions: {height}x{width}")
            raise ValueError(f"Spatial dimensions must be positive, got {height}x{width}")
        
        # Patch grid validation
        try:
            self._calculate_patch_grid(height, width)
        except ValueError as e:
            logger.error(f"Patch grid validation failed: {e}")
            raise
        
        # Device validation
        model_device = next(self.parameters()).device
        if x.device != model_device:
            logger.error(f"Device mismatch: input on {x.device}, model on {model_device}")
            raise ValueError(f"Device mismatch: input on {x.device}, model on {model_device}")
        
        # Dtype validation
        if x.dtype != torch.float32:
            logger.warning(f"Input dtype {x.dtype} != float32, may cause issues")
        
        # Value validation
        if torch.isnan(x).any():
            logger.error("Input contains NaN values")
            raise ValueError("Input contains NaN values")
        
        if torch.isinf(x).any():
            logger.error("Input contains infinite values")
            raise ValueError("Input contains infinite values")
        
        logger.debug(f"Input validation passed: {batch_size}x{channels}x{height}x{width}")
    
    def _calculate_patch_grid(self, height: int, width: int) -> Tuple[int, int]:
        """Calculate patch grid dimensions with validation."""
        if height % self.patch_size != 0:
            raise ValueError(f"Height {height} not divisible by patch size {self.patch_size}")
        
        if width % self.patch_size != 0:
            raise ValueError(f"Width {width} not divisible by patch size {self.patch_size}")
        
        h_patches = height // self.patch_size
        w_patches = width // self.patch_size
        
        if h_patches <= 0 or w_patches <= 0:
            raise ValueError(f"Invalid patch grid: {h_patches}x{w_patches}")
        
        return h_patches, w_patches
    
    def _sequence_to_spatial(self, features: torch.Tensor, h_patches: int, w_patches: int) -> torch.Tensor:
        """Convert ViT sequence format to spatial format with validation."""
        logger.debug(f"Converting sequence to spatial: {features.shape} -> patches {h_patches}x{w_patches}")
        
        B, N, C = features.shape
        
        # Remove CLS token (first token)
        if N != h_patches * w_patches + 1:
            logger.error(f"Token count mismatch: expected {h_patches * w_patches + 1}, got {N}")
            raise ValueError(f"Token count mismatch: expected {h_patches * w_patches + 1}, got {N}")
        
        # Extract patch tokens (skip CLS token)
        patch_tokens = features[:, 1:, :]  # [B, N_patches, C]
        
        # Reshape to spatial grid
        spatial_features = patch_tokens.view(B, h_patches, w_patches, C)
        
        # Permute to CNN format: [B, C, H, W]
        spatial_features = spatial_features.permute(0, 3, 1, 2).contiguous()
        
        logger.debug(f"Sequence to spatial conversion completed: {spatial_features.shape}")
        return spatial_features
    
    def _apply_gradient_checkpointing(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gradient checkpointing for memory efficiency."""
        if not self.training or not self.use_checkpointing:
            logger.debug("Skipping gradient checkpointing")
            return self.backbone(x)
        
        logger.debug("Applying gradient checkpointing")
        
        try:
            # Manual forward pass with checkpointing
            # Step 1: Patch embedding
            x = self.backbone.patch_embed(x) #type: ignore
            
            # Step 2: Add CLS token
            cls_tokens = self.backbone.cls_token.expand(x.shape[0], -1, -1) #type: ignore
            x = torch.cat((cls_tokens, x), dim=1)
            
            # Step 3: Add positional embeddings
            x = x + self.backbone.pos_embed #type: ignore
            x = self.backbone.pos_drop(x) #type: ignore
            
            # Step 4: Apply transformer blocks with checkpointing
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            
            for i, block in enumerate(self.backbone.blocks): #type: ignore
                x = checkpoint.checkpoint(create_custom_forward(block), x) #type: ignore
                # Note: Hooks still fire during checkpointed forward pass
            
            # Step 5: Final normalization
            x = self.backbone.norm(x) #type: ignore
            
            logger.debug("Gradient checkpointing completed successfully")
            return x
            
        except Exception as e:
            logger.error(f"Gradient checkpointing failed: {e}")
            raise
    
    def _interpolate_features(self, features: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        """Interpolate features to target size with logging."""
        current_h, current_w = features.shape[2], features.shape[3]
        target_h, target_w = target_size
        
        if (current_h, current_w) == (target_h, target_w):
            return features
        
        logger.debug(f"Interpolating features from {current_h}x{current_w} to {target_h}x{target_w}")
        
        import torch.nn.functional as F
        interpolated = F.interpolate(
            features,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )
        
        return interpolated
    
    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Main forward pass returning multi-scale spatial features.
        
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            List of spatial feature tensors [B, C, H_patches, W_patches]
        """
        logger.debug(f"Forward pass started with input shape: {x.shape}")
        
        # Step 1: Input validation
        self._validate_input(x)
        batch_size, _, height, width = x.shape
        
        # Step 2: Calculate patch grid
        h_patches, w_patches = self._calculate_patch_grid(height, width)
        logger.debug(f"Patch grid: {h_patches}x{w_patches}")
        
        # Step 3: Clear feature storage
        self._features.clear()
        
        # Step 4: Forward pass (hooks capture intermediate features)
        try:
            if self.use_checkpointing:
                final_features = self._apply_gradient_checkpointing(x)
            else:
                final_features = self.backbone(x)
            
            logger.debug(f"Backbone forward completed, captured {len(self._features)} feature maps")
            
        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            raise
        
        # Step 5: Process captured features
        processed_features = []
        for layer_idx in self.extract_layers:
            if layer_idx not in self._features:
                logger.error(f"Missing features for layer {layer_idx}")
                raise RuntimeError(f"Missing features for layer {layer_idx}")
            
            # Get sequence features
            feat_seq = self._features[layer_idx]
            
            # Convert to spatial format
            feat_spatial = self._sequence_to_spatial(feat_seq, h_patches, w_patches)
            
            # Optional: Interpolate to consistent size
            # Here we keep original patch grid size
            processed_features.append(feat_spatial)
            
            logger.debug(f"Processed layer {layer_idx}: {feat_spatial.shape}")
        
        # Step 6: Final validation
        for i, features in enumerate(processed_features):
            if features.ndim != 4:
                logger.error(f"Invalid feature dimensions at index {i}: {features.shape}")
                raise ValueError(f"Feature must be 4D, got {features.shape}")
        
        logger.debug(f"Forward pass completed successfully, returning {len(processed_features)} feature maps")
        return processed_features
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Main forward method - wrapper around forward_features."""
        return self.forward_features(x)
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Get detailed information about extracted features."""
        h_patches, w_patches = self._calculate_patch_grid(self.img_size, self.img_size)
        
        info = {
            'model_name': self.model_name,
            'extract_layers': self.extract_layers,
            'patch_size': self.patch_size,
            'img_size': self.img_size,
            'embed_dim': self.embed_dim,
            'num_blocks': self.num_blocks,
            'patch_grid': (h_patches, w_patches),
            'use_checkpointing': self.use_checkpointing,
            'pretrained': self.pretrained,
            'feature_shapes': {},
            'memory_estimate_mb': {},
            'notes': []
        }
        
        # Calculate feature info for each layer
        for layer_idx in self.extract_layers:
            feature_shape = (self.embed_dim, h_patches, w_patches)
            info['feature_shapes'][layer_idx] = feature_shape
            
            # Memory estimate (float32 = 4 bytes)
            num_elements = self.embed_dim * h_patches * w_patches #type: ignore 
            memory_mb = (num_elements * 4) / (1024 ** 2)
            info['memory_estimate_mb'][layer_idx] = memory_mb
        
        # Add notes
        if self.use_checkpointing:
            info['notes'].append("Gradient checkpointing enabled for memory efficiency")
        
        if len(self.extract_layers) > 4:
            info['notes'].append("Warning: Many extract layers may increase memory usage")
        
        return info
    
    def _cleanup_hooks(self) -> None:
        """Clean up registered hooks to prevent memory leaks."""
        logger.debug(f"Cleaning up {len(self._hooks)} hooks")
        
        for hook in self._hooks:
            try:
                hook.remove()
            except Exception as e:
                logger.warning(f"Error removing hook: {e}")
        
        self._hooks.clear()
        self._features.clear()
        
        logger.debug("Hook cleanup completed")
    
    def __del__(self):
        """Destructor to ensure proper cleanup."""
        try:
            self._cleanup_hooks()
        except Exception as e:
            logger.warning(f"Error during cleanup in destructor: {e}")