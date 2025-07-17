# FILE: tests/test_backbone.py
# ehsanasgharzde - COMPLETE BACKBONE TEST SUITE
# hosseinsolymanzadeh - PROPER COMMENTING

import torch
import pytest
import logging
from typing import Tuple, Dict, Any
from unittest.mock import patch, MagicMock
import time
import gc

from models.backbones.backbone import ViT, RECOMMENDED_VIT_MODELS

# Setup logging for tests
logger = logging.getLogger(__name__)

class TestViT:
    # Test suite for ViT backbone with improved organization and reduced duplication.
    
    @staticmethod
    def get_test_config(model_name: str = 'vit_base_patch16_224') -> Dict[str, Any]:
        # Get test configuration using the recommended models dictionary.
        if model_name not in RECOMMENDED_VIT_MODELS:
            raise ValueError(f"Model {model_name} not in recommended models")
        
        config = RECOMMENDED_VIT_MODELS[model_name].copy()
        config['model_name'] = model_name #type: ignore 
        return config
    
    @staticmethod
    def create_dummy_input(batch_size: int, img_size: int, channels: int = 3) -> torch.Tensor:
        # Create standardized dummy input tensor.
        return torch.randn(batch_size, channels, img_size, img_size, requires_grad=True)
    
    @staticmethod
    def create_dummy_sequence_features(batch_size: int, patch_size: int, 
                                     img_size: int, embed_dim: int) -> torch.Tensor:
        # Create dummy sequence features with CLS token.
        h_patches, w_patches = img_size // patch_size, img_size // patch_size
        num_patches = h_patches * w_patches
        return torch.randn(batch_size, num_patches + 1, embed_dim, requires_grad=True)
    
    @staticmethod
    def validate_feature_tensor(features: torch.Tensor, expected_shape: Tuple[int, ...], 
                              layer_name: str = "feature") -> None:
        # Standardized feature validation.
        assert features.shape == expected_shape, \
            f"{layer_name} shape mismatch: expected {expected_shape}, got {features.shape}"
        assert not torch.isnan(features).any(), f"NaN values found in {layer_name}"
        assert not torch.isinf(features).any(), f"Inf values found in {layer_name}"
        assert features.dtype == torch.float32, f"Expected float32, got {features.dtype}"
    
    @staticmethod
    def validate_gradient_flow(tensor: torch.Tensor, tensor_name: str = "tensor") -> None:
        # Validate gradient flow through tensor.
        assert tensor.grad is not None, f"No gradient found for {tensor_name}"
        grad_norm = tensor.grad.norm().item()
        assert grad_norm > 0, f"Zero gradient norm for {tensor_name}"
        logger.info(f"Gradient flow validated for {tensor_name}: norm={grad_norm:.6f}")
    
    def create_test_model(self, model_name: str = 'vit_base_patch16_224', 
                         **kwargs) -> ViT:
        # Create standardized test model.
        config = self.get_test_config(model_name)
        config.update(kwargs)
        config['pretrained'] = False  # Faster for testing
        
        model = ViT(**config)
        model.eval()
        logger.info(f"Created test model: {model_name} with config: {config}")
        return model

@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("model_name", ['vit_base_patch16_224', 'vit_base_patch8_224'])
def test_sequence_to_spatial_conversion(batch_size: int, model_name: str):
    # Test sequence-to-spatial conversion using model configuration.
    logger.info(f"Testing sequence-to-spatial conversion: batch_size={batch_size}, model={model_name}")
    
    test_helper = TestViT()
    config = test_helper.get_test_config(model_name)
    
    # Create model
    model = test_helper.create_test_model(model_name)
    
    # Create dummy features using model's calculated patch grid
    h_patches, w_patches = model._calculate_patch_grid(config['img_size'], config['img_size'])
    dummy_features = test_helper.create_dummy_sequence_features(
        batch_size, config['patch_size'], config['img_size'], config['embed_dim']
    )
    
    # Perform conversion using model method
    spatial = model._sequence_to_spatial(dummy_features, h_patches, w_patches)
    
    # Validate using helper method
    expected_shape = (batch_size, config['embed_dim'], h_patches, w_patches)
    test_helper.validate_feature_tensor(spatial, expected_shape, "spatial_features")
    
    # Test gradient flow
    spatial.sum().backward()
    test_helper.validate_gradient_flow(dummy_features, "sequence_features")
    
    logger.info(f"Sequence-to-spatial conversion test passed for {model_name}")

@pytest.mark.parametrize("model_name", ['vit_base_patch16_224', 'vit_base_patch16_384'])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_multi_scale_feature_extraction(model_name: str, batch_size: int):
    # Test multi-scale feature extraction using model configuration.
    logger.info(f"Testing multi-scale features: model={model_name}, batch_size={batch_size}")
    
    test_helper = TestViT()
    config = test_helper.get_test_config(model_name)
    
    # Create model with default extract layers
    model = test_helper.create_test_model(model_name, extract_layers=[6, 9, 11])
    
    # Create input using model configuration
    x = test_helper.create_dummy_input(batch_size, config['img_size'])
    
    # Forward pass
    with torch.no_grad():
        features = model(x)
    
    # Validate number of features matches extract_layers
    assert len(features) == len(model.extract_layers), \
        f"Expected {len(model.extract_layers)} features, got {len(features)}"
    
    # Validate each feature map
    h_patches, w_patches = model._calculate_patch_grid(config['img_size'], config['img_size'])
    for i, feat in enumerate(features):
        expected_shape = (batch_size, config['embed_dim'], h_patches, w_patches)
        test_helper.validate_feature_tensor(feat, expected_shape, f"feature_map_{i}")
    
    logger.info(f"Multi-scale feature extraction test passed for {model_name}")

@pytest.mark.parametrize("use_checkpointing", [False, True])
@pytest.mark.parametrize("model_name", ['vit_base_patch16_224'])
def test_gradient_checkpointing(use_checkpointing: bool, model_name: str):
    # Test gradient checkpointing functionality.
    logger.info(f"Testing gradient checkpointing: enabled={use_checkpointing}")
    
    test_helper = TestViT()
    config = test_helper.get_test_config(model_name)
    
    # Create model with checkpointing configuration
    model = test_helper.create_test_model(
        model_name, 
        use_checkpointing=use_checkpointing,
        extract_layers=[11]  # Just test final layer
    )
    model.train()  # Enable training mode for checkpointing
    
    # Create input
    x = test_helper.create_dummy_input(2, config['img_size'])
    
    # Forward pass
    features = model(x)
    
    # Create dummy loss and backward pass
    loss = sum(feat.mean() for feat in features)
    loss.backward() #type: ignore
    
    # Validate gradient flow
    test_helper.validate_gradient_flow(x, "input_tensor")
    
    logger.info(f"Gradient checkpointing test passed: enabled={use_checkpointing}")

@pytest.mark.parametrize("model_name", ['vit_base_patch16_224'])
def test_backbone_output_shape_and_consistency(model_name: str):
    #Test backbone output shapes and consistency.
    logger.info(f"Testing backbone output shapes for {model_name}")
    
    test_helper = TestViT()
    config = test_helper.get_test_config(model_name)
    
    # Create model with specific extract layers
    extract_layers = [3, 6, 9, 11]
    model = test_helper.create_test_model(model_name, extract_layers=extract_layers)
    
    # Create input
    x = test_helper.create_dummy_input(1, config['img_size'])
    
    # Forward pass
    features = model(x)
    
    # Test number of features
    assert len(features) == len(extract_layers), \
        f"Expected {len(extract_layers)} features, got {len(features)}"
    
    # Test each feature map
    h_patches, w_patches = model._calculate_patch_grid(config['img_size'], config['img_size'])
    for i, feat in enumerate(features):
        # Use model's embed_dim instead of hardcoded value
        expected_shape = (1, model.embed_dim, h_patches, w_patches)
        test_helper.validate_feature_tensor(feat, expected_shape, f"backbone_feature_{i}") #type: ignore
    
    # Test gradient flow
    loss = sum(feat.sum() for feat in features)
    loss.backward() #type: ignore
    test_helper.validate_gradient_flow(x, "backbone_input")
    
    logger.info(f"Backbone output shape test passed for {model_name}")

def test_model_configuration_validation():
    #Test model configuration validation.
    logger.info("Testing model configuration validation")
    
    test_helper = TestViT()
    
    # Test invalid model name
    with pytest.raises(ValueError, match="not in recommended models"):
        test_helper.create_test_model('invalid_model_name')
    
    # Test invalid extract layers
    with pytest.raises(ValueError, match="out of range"):
        test_helper.create_test_model('vit_base_patch16_224', extract_layers=[999])
    
    # Test incompatible img_size and patch_size
    with pytest.raises(ValueError, match="not divisible"):
        test_helper.create_test_model('vit_base_patch16_224', img_size=225, patch_size=16)
    
    logger.info("Model configuration validation test passed")

def test_input_validation():
    # Test input validation functionality.
    logger.info("Testing input validation")
    
    test_helper = TestViT()
    model = test_helper.create_test_model('vit_base_patch16_224')
    
    # Test invalid input dimensions
    with pytest.raises(ValueError, match="must be 4D"):
        invalid_input = torch.randn(224, 224, 3)  # 3D instead of 4D
        model._validate_input(invalid_input)
    
    # Test invalid channel count
    with pytest.raises(ValueError, match="Expected 3 channels"):
        invalid_input = torch.randn(1, 4, 224, 224)  # 4 channels instead of 3
        model._validate_input(invalid_input)
    
    # Test NaN input
    with pytest.raises(ValueError, match="NaN values"):
        invalid_input = torch.randn(1, 3, 224, 224)
        invalid_input[0, 0, 0, 0] = float('nan')
        model._validate_input(invalid_input)
    
    logger.info("Input validation test passed")

def test_feature_info_utility():
    #Test the feature info utility method.
    logger.info("Testing feature info utility")
    
    test_helper = TestViT()
    model = test_helper.create_test_model('vit_base_patch16_224', extract_layers=[6, 9, 11])
    
    # Get feature info
    info = model.get_feature_info()
    
    # Validate info structure
    required_keys = ['model_name', 'extract_layers', 'patch_size', 'img_size', 
                    'embed_dim', 'num_blocks', 'patch_grid', 'feature_shapes']
    for key in required_keys:
        assert key in info, f"Missing key {key} in feature info"
    
    # Validate feature shapes
    for layer_idx in model.extract_layers:
        assert layer_idx in info['feature_shapes'], f"Missing shape info for layer {layer_idx}"
    
    logger.info(f"Feature info utility test passed: {info['model_name']}")

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_compatibility():
    #Test device compatibility (GPU/CPU).
    logger.info("Testing device compatibility")
    
    test_helper = TestViT()
    
    # Test CPU
    model_cpu = test_helper.create_test_model('vit_base_patch16_224')
    x_cpu = test_helper.create_dummy_input(1, 224)
    features_cpu = model_cpu(x_cpu)
    assert features_cpu[0].device.type == 'cpu', "CPU model should produce CPU features"
    
    # Test GPU
    model_gpu = test_helper.create_test_model('vit_base_patch16_224').cuda()
    x_gpu = test_helper.create_dummy_input(1, 224).cuda()
    features_gpu = model_gpu(x_gpu)
    assert features_gpu[0].device.type == 'cuda', "GPU model should produce GPU features"
    
    logger.info("Device compatibility test passed")

def benchmark_performance():
    #Benchmark performance with proper cleanup.
    logger.info("Starting performance benchmark")
    
    test_helper = TestViT()
    
    # Test configuration
    img_size = 224
    batch_size = 8
    num_warmup = 5
    num_iterations = 20
    
    # Create model
    model = test_helper.create_test_model('vit_base_patch16_224')
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create input
    x = test_helper.create_dummy_input(batch_size, img_size).to(device)
    
    # Warm-up
    logger.info("Running warm-up iterations...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x)
    
    # Benchmark
    logger.info(f"Running {num_iterations} benchmark iterations...")
    if device.type == 'cuda':
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record(stream=torch.cuda.current_stream())
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(x)
        end_event.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        
        elapsed_time_ms = start_event.elapsed_time(end_event) / num_iterations
        max_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        
        logger.info(f"GPU Benchmark Results:")
        logger.info(f"  Average inference time: {elapsed_time_ms:.2f} ms")
        logger.info(f"  Peak GPU memory: {max_memory_mb:.2f} MB")
    else:
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(x)
        end_time = time.time()
        
        elapsed_time_ms = ((end_time - start_time) / num_iterations) * 1000
        logger.info(f"CPU Benchmark Results:")
        logger.info(f"  Average inference time: {elapsed_time_ms:.2f} ms")
    
    # Cleanup
    del model, x
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    logger.info("✓ Performance benchmark completed")

# Integration test
def test_end_to_end_workflow():
    #Test complete end-to-end workflow.
    logger.info("Testing end-to-end workflow")
    
    test_helper = TestViT()
    
    # Create model with multiple configurations
    model = test_helper.create_test_model(
        'vit_base_patch16_224',
        extract_layers=[3, 6, 9, 11],
        use_checkpointing=False
    )
    
    # Test with different input sizes (using model's configuration)
    config = test_helper.get_test_config('vit_base_patch16_224')
    
    for batch_size in [1, 4]:
        x = test_helper.create_dummy_input(batch_size, config['img_size'])
        
        # Forward pass
        features = model(x)
        
        # Validate results
        assert len(features) == 4, f"Expected 4 features, got {len(features)}"
        
        # Test gradient flow
        loss = sum(feat.mean() for feat in features)
        loss.backward() #type: ignore   
        test_helper.validate_gradient_flow(x, f"e2e_input_batch_{batch_size}")
        
        # Clear gradients
        model.zero_grad()
        x.grad = None #type: ignore 
    
    logger.info("End-to-end workflow test passed")

if __name__ == "__main__":
    # Run basic tests
    test_helper = TestViT()
    
    # Test configuration
    logger.info("Testing model configuration...")
    config = test_helper.get_test_config('vit_base_patch16_224')
    print(f"Test config: {config}")
    
    # Test model creation
    logger.info("Testing model creation...")
    model = test_helper.create_test_model('vit_base_patch16_224')
    print(f"Model info: {model.get_feature_info()}")
    
    logger.info("All manual tests passed!")
