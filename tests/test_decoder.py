# FILE: tests/test_decoder.py
# ehsanasgharzde - COMPLETE DECODER TEST SUITE
# hosseinsolymanzadeh - PROPER COMMENTING

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import pytest
import psutil
import gc
from models.decoders.decoder import FusionBlock, DPT, validate_features, interpolate_to_size

logger = logging.getLogger(__name__)

class TestFusionBlock:
    
    def setup_method(self):
        # Set device to GPU if available, otherwise fallback to CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
    def test_fusion_block_basic(self):
        # Basic test for FusionBlock without skip connection or attention
        in_ch, out_ch = 256, 128
        block = FusionBlock(in_ch, out_ch).to(self.device)
        
        # Create dummy input tensor
        x = torch.randn(2, in_ch, 32, 32).to(self.device)
        output = block(x)
        
        # Output should be upsampled (spatially doubled)
        assert output.shape == (2, out_ch, 64, 64)
        assert output.requires_grad
        logger.info("FusionBlock basic test passed")
        
    def test_fusion_block_with_skip(self):
        # Test FusionBlock when a skip connection is provided
        in_ch, out_ch = 256, 128
        block = FusionBlock(in_ch, out_ch).to(self.device)
        
        # Input and skip feature maps
        x = torch.randn(2, in_ch, 32, 32).to(self.device)
        skip = torch.randn(2, out_ch, 64, 64).to(self.device)
        output = block(x, skip)
        
        # Output should match the spatial and channel shape of skip
        assert output.shape == skip.shape
        assert output.requires_grad
        logger.info("FusionBlock with skip connection test passed")
        
    def test_fusion_block_attention(self):
        # Test FusionBlock with attention mechanism enabled
        in_ch, out_ch = 256, 128
        block = FusionBlock(in_ch, out_ch, use_attention=True).to(self.device)
        
        x = torch.randn(2, in_ch, 32, 32).to(self.device)
        output = block(x)
        
        # Output shape check and attention module existence
        assert output.shape == (2, out_ch, 64, 64)
        assert hasattr(block, 'attention')
        logger.info("FusionBlock attention test passed")
        
    def test_fusion_block_different_scale(self):
        # Test FusionBlock with a custom upsampling scale factor
        in_ch, out_ch = 256, 128
        block = FusionBlock(in_ch, out_ch, scale_factor=4).to(self.device)
        
        x = torch.randn(2, in_ch, 16, 16).to(self.device)
        output = block(x)
        
        # Expecting 4x spatial upsampling
        assert output.shape == (2, out_ch, 64, 64)
        logger.info("FusionBlock scale factor test passed")


class TestDPT:
    
    def setup_method(self):
        # Set device and define channel sizes for backbone and decoder
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.backbone_channels = [96, 192, 384, 768]
        self.decoder_channels = [256, 256, 256, 256]
        
    def test_dpt_initialization(self):
        # Test DPT decoder construction
        decoder = DPT(self.backbone_channels, self.decoder_channels).to(self.device)
        
        # Ensure all components are initialized correctly
        assert len(decoder.projections) == 4
        assert len(decoder.fusions) == 4
        assert isinstance(decoder.output_conv, nn.Conv2d)
        logger.info("DPT initialization test passed")
        
    def test_dpt_forward_pass(self):
        # Test forward pass through the decoder with simulated input features
        decoder = DPT(self.backbone_channels, self.decoder_channels).to(self.device)
        
        # Create dummy multi-scale features
        features = [
            torch.randn(1, 96, 64, 64).to(self.device),
            torch.randn(1, 192, 32, 32).to(self.device),
            torch.randn(1, 384, 16, 16).to(self.device),
            torch.randn(1, 768, 8, 8).to(self.device)
        ]
        
        output = decoder(features)
        
        # Check output shape and gradient tracking
        assert output.shape == (1, 1, 32, 32)
        assert output.requires_grad
        logger.info("DPT forward pass test passed")
        
    def test_dpt_different_activations(self):
        # Test various activation functions at the decoder output
        activations = ['sigmoid', 'tanh', 'relu', None]
        
        for activation in activations:
            decoder = DPT(self.backbone_channels, self.decoder_channels, 
                         final_activation=activation).to(self.device)
            
            features = [
                torch.randn(1, 96, 64, 64).to(self.device),
                torch.randn(1, 192, 32, 32).to(self.device),
                torch.randn(1, 384, 16, 16).to(self.device),
                torch.randn(1, 768, 8, 8).to(self.device)
            ]
            
            output = decoder(features)
            
            # Validate output range according to the activation function
            if activation == 'sigmoid':
                assert torch.all(output >= 0) and torch.all(output <= 1)
            elif activation == 'tanh':
                assert torch.all(output >= -1) and torch.all(output <= 1)
            elif activation == 'relu':
                assert torch.all(output >= 0)
                
            logger.info(f"DPT {activation} activation test passed")
            
    def test_dpt_with_attention(self):
        # Test DPT decoder with attention enabled in FusionBlocks
        decoder = DPT(self.backbone_channels, self.decoder_channels, 
                     use_attention=True).to(self.device)
        
        features = [
            torch.randn(1, 96, 64, 64).to(self.device),
            torch.randn(1, 192, 32, 32).to(self.device),
            torch.randn(1, 384, 16, 16).to(self.device),
            torch.randn(1, 768, 8, 8).to(self.device)
        ]
        
        output = decoder(features)
        
        # Check that output shape is as expected
        assert output.shape == (1, 1, 32, 32)
        logger.info("DPT with attention test passed")


class TestValidation:
    
    def test_validate_features_correct(self):
        # Create a list of feature tensors with the expected shapes and channels
        features = [
            torch.randn(1, 96, 64, 64),
            torch.randn(1, 192, 32, 32),
            torch.randn(1, 384, 16, 16),
            torch.randn(1, 768, 8, 8)
        ]
        # Define the expected number of channels for each feature tensor
        expected_channels = [96, 192, 384, 768]
        
        # Call the validation function to verify the features match expectations
        validate_features(features, expected_channels)
        # Log success message after validation passes
        logger.info("Feature validation test passed")
        
    def test_validate_features_mismatch(self):
        # Create feature tensors fewer than expected channels count
        features = [
            torch.randn(1, 96, 64, 64),
            torch.randn(1, 192, 32, 32),
            torch.randn(1, 384, 16, 16)
        ]
        # Define expected channels which include one extra channel size
        expected_channels = [96, 192, 384, 768]
        
        # Expect ValueError because number of features and expected channels differ
        with pytest.raises(ValueError):
            validate_features(features, expected_channels)
            
    def test_validate_features_channel_mismatch(self):
        # Create feature tensors with a channel mismatch in the second tensor
        features = [
            torch.randn(1, 96, 64, 64),
            torch.randn(1, 128, 32, 32),  # channel 128 instead of expected 192
            torch.randn(1, 384, 16, 16),
            torch.randn(1, 768, 8, 8)
        ]
        # Define expected channels with 192 in second position
        expected_channels = [96, 192, 384, 768]
        
        # Expect ValueError due to channel size mismatch in one of the features
        with pytest.raises(ValueError):
            validate_features(features, expected_channels)

class TestMemoryEfficiency:
    
    def test_memory_usage_large_input(self):
        # Set device to GPU if available, otherwise CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Run garbage collection to free memory
        gc.collect()
        # Clear CUDA cache if GPU is available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Initialize decoder model and move to device
        decoder = DPT([96, 192, 384, 768], [256, 256, 256, 256]).to(device)
        
        # Create input feature tensors with specified sizes on the device
        features = [
            torch.randn(1, 96, 256, 256).to(device),
            torch.randn(1, 192, 128, 128).to(device),
            torch.randn(1, 384, 64, 64).to(device),
            torch.randn(1, 768, 32, 32).to(device)
        ]
        
        if torch.cuda.is_available():
            # Measure GPU memory usage before inference
            mem_before = torch.cuda.memory_allocated()
            output = decoder(features)
            # Measure GPU memory usage after inference
            mem_after = torch.cuda.memory_allocated()
            # Calculate memory used in megabytes
            mem_used = (mem_after - mem_before) / (1024 ** 2)
            logger.info(f"GPU memory used: {mem_used:.2f} MB")
        else:
            # Use psutil to measure CPU memory usage before inference
            process = psutil.Process()
            mem_before = process.memory_info().rss
            output = decoder(features)
            # Measure CPU memory usage after inference
            mem_after = process.memory_info().rss
            # Calculate memory used in megabytes
            mem_used = (mem_after - mem_before) / (1024 ** 2)
            logger.info(f"CPU memory used: {mem_used:.2f} MB")
            
        # Verify output shape is as expected
        assert output.shape == (1, 1, 128, 128)
        
    def test_gradient_computation(self):
        # Set device to GPU if available, otherwise CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize decoder model and move to device
        decoder = DPT([96, 192, 384, 768], [256, 256, 256, 256]).to(device)
        
        # Create input feature tensors with requires_grad=True for gradient tracking
        features = [
            torch.randn(1, 96, 64, 64, requires_grad=True).to(device),
            torch.randn(1, 192, 32, 32, requires_grad=True).to(device),
            torch.randn(1, 384, 16, 16, requires_grad=True).to(device),
            torch.randn(1, 768, 8, 8, requires_grad=True).to(device)
        ]
        
        # Forward pass through decoder
        output = decoder(features)
        # Compute sum of outputs as loss
        loss = output.sum()
        # Backpropagate to compute gradients
        loss.backward()
        
        # Check that gradients are computed for all input features
        for i, feat in enumerate(features):
            assert feat.grad is not None
            logger.info(f"Gradient computed for feature {i}")


class TestInterpolation:
    
    def test_interpolate_to_size(self):
        # Select device: GPU if available, otherwise CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create a random tensor simulating features with shape (batch=2, channels=128, height=32, width=32)
        features = torch.randn(2, 128, 32, 32).to(device)
        # Define the target output size for interpolation
        target_size = (64, 64)
        
        # Perform interpolation to the target size
        output = interpolate_to_size(features, target_size)
        
        # Assert that the output shape matches expected (batch=2, channels=128, height=64, width=64)
        assert output.shape == (2, 128, 64, 64)
        # Log success message
        logger.info("Interpolation test passed")
        
    def test_interpolate_different_sizes(self):
        # Select device: GPU if available, otherwise CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create a random tensor simulating features with shape (batch=1, channels=256, height=16, width=16)
        features = torch.randn(1, 256, 16, 16).to(device)
        # Define multiple target sizes for interpolation testing
        target_sizes = [(32, 32), (64, 64), (128, 128), (256, 256)]
        
        # Loop over each target size and test interpolation
        for target_size in target_sizes:
            output = interpolate_to_size(features, target_size)
            # Assert the spatial dimensions of output match the target size
            assert output.shape[-2:] == target_size
            # Log success message for each size
            logger.info(f"Interpolation to {target_size} passed")


class TestIntegration:
    
    def test_full_pipeline(self):
        # Select device: GPU if available, otherwise CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize the DPT decoder model with specified feature and decoder channel sizes,
        # enabling attention and sigmoid as final activation
        decoder = DPT([96, 192, 384, 768], [256, 256, 256, 256], 
                     use_attention=True, final_activation='sigmoid').to(device)
        
        batch_size = 2
        # Create a list of random feature tensors with varying spatial resolutions
        features = [
            torch.randn(batch_size, 96, 64, 64).to(device),
            torch.randn(batch_size, 192, 32, 32).to(device),
            torch.randn(batch_size, 384, 16, 16).to(device),
            torch.randn(batch_size, 768, 8, 8).to(device)
        ]
        
        # Validate that the features have the expected channel dimensions
        validate_features(features, [96, 192, 384, 768])
        
        # Perform forward pass with no gradient computation (inference mode)
        with torch.no_grad():
            output = decoder(features)
            
        # Assert output shape matches expected (batch_size, 1, 32, 32)
        assert output.shape == (batch_size, 1, 32, 32)
        # Assert output values are within the range [0, 1] (due to sigmoid activation)
        assert torch.all(output >= 0) and torch.all(output <= 1)
        # Log success message
        logger.info("Full pipeline integration test passed")
        
    def test_error_handling(self):
        # Select device: GPU if available, otherwise CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize the DPT decoder model with default parameters
        decoder = DPT([96, 192, 384, 768], [256, 256, 256, 256]).to(device)
        
        # Create features with incorrect number of inputs (only 2 instead of 4)
        wrong_features = [
            torch.randn(1, 96, 64, 64).to(device),
            torch.randn(1, 192, 32, 32).to(device)
        ]
        
        # Expect a ValueError to be raised when passing incorrect features to decoder
        with pytest.raises(ValueError):
            decoder(wrong_features)
            
        # Log success message when error handling is verified
        logger.info("Error handling test passed")

# If this script is run as main, execute all pytest tests with verbose output
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
