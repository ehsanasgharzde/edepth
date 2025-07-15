# FILE: tests/test_decoder.py
# ehsanasgharzde - COMPLETE DECODER TEST SUITE

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import pytest
import psutil
import gc
from models.decoders.decoder_fixed import FusionBlock, DPT, validate_features, interpolate_to_size

logger = logging.getLogger(__name__)

class TestFusionBlock:
    
    def setup_method(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
    def test_fusion_block_basic(self):
        in_ch, out_ch = 256, 128
        block = FusionBlock(in_ch, out_ch).to(self.device)
        
        x = torch.randn(2, in_ch, 32, 32).to(self.device)
        output = block(x)
        
        assert output.shape == (2, out_ch, 64, 64)
        assert output.requires_grad
        logger.info("FusionBlock basic test passed")
        
    def test_fusion_block_with_skip(self):
        in_ch, out_ch = 256, 128
        block = FusionBlock(in_ch, out_ch).to(self.device)
        
        x = torch.randn(2, in_ch, 32, 32).to(self.device)
        skip = torch.randn(2, out_ch, 64, 64).to(self.device)
        output = block(x, skip)
        
        assert output.shape == skip.shape
        assert output.requires_grad
        logger.info("FusionBlock with skip connection test passed")
        
    def test_fusion_block_attention(self):
        in_ch, out_ch = 256, 128
        block = FusionBlock(in_ch, out_ch, use_attention=True).to(self.device)
        
        x = torch.randn(2, in_ch, 32, 32).to(self.device)
        output = block(x)
        
        assert output.shape == (2, out_ch, 64, 64)
        assert hasattr(block, 'attention')
        logger.info("FusionBlock attention test passed")
        
    def test_fusion_block_different_scale(self):
        in_ch, out_ch = 256, 128
        block = FusionBlock(in_ch, out_ch, scale_factor=4).to(self.device)
        
        x = torch.randn(2, in_ch, 16, 16).to(self.device)
        output = block(x)
        
        assert output.shape == (2, out_ch, 64, 64)
        logger.info("FusionBlock scale factor test passed")

class TestDPT:
    
    def setup_method(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.backbone_channels = [96, 192, 384, 768]
        self.decoder_channels = [256, 256, 256, 256]
        
    def test_dpt_initialization(self):
        decoder = DPT(self.backbone_channels, self.decoder_channels).to(self.device)
        
        assert len(decoder.projections) == 4
        assert len(decoder.fusions) == 4
        assert isinstance(decoder.output_conv, nn.Conv2d)
        logger.info("DPT initialization test passed")
        
    def test_dpt_forward_pass(self):
        decoder = DPT(self.backbone_channels, self.decoder_channels).to(self.device)
        
        features = [
            torch.randn(1, 96, 64, 64).to(self.device),
            torch.randn(1, 192, 32, 32).to(self.device),
            torch.randn(1, 384, 16, 16).to(self.device),
            torch.randn(1, 768, 8, 8).to(self.device)
        ]
        
        output = decoder(features)
        
        assert output.shape == (1, 1, 32, 32)
        assert output.requires_grad
        logger.info("DPT forward pass test passed")
        
    def test_dpt_different_activations(self):
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
            
            if activation == 'sigmoid':
                assert torch.all(output >= 0) and torch.all(output <= 1)
            elif activation == 'tanh':
                assert torch.all(output >= -1) and torch.all(output <= 1)
            elif activation == 'relu':
                assert torch.all(output >= 0)
                
            logger.info(f"DPT {activation} activation test passed")
            
    def test_dpt_with_attention(self):
        decoder = DPT(self.backbone_channels, self.decoder_channels, 
                     use_attention=True).to(self.device)
        
        features = [
            torch.randn(1, 96, 64, 64).to(self.device),
            torch.randn(1, 192, 32, 32).to(self.device),
            torch.randn(1, 384, 16, 16).to(self.device),
            torch.randn(1, 768, 8, 8).to(self.device)
        ]
        
        output = decoder(features)
        
        assert output.shape == (1, 1, 32, 32)
        logger.info("DPT with attention test passed")

class TestValidation:
    
    def test_validate_features_correct(self):
        features = [
            torch.randn(1, 96, 64, 64),
            torch.randn(1, 192, 32, 32),
            torch.randn(1, 384, 16, 16),
            torch.randn(1, 768, 8, 8)
        ]
        expected_channels = [96, 192, 384, 768]
        
        validate_features(features, expected_channels)
        logger.info("Feature validation test passed")
        
    def test_validate_features_mismatch(self):
        features = [
            torch.randn(1, 96, 64, 64),
            torch.randn(1, 192, 32, 32),
            torch.randn(1, 384, 16, 16)
        ]
        expected_channels = [96, 192, 384, 768]
        
        with pytest.raises(ValueError):
            validate_features(features, expected_channels)
            
    def test_validate_features_channel_mismatch(self):
        features = [
            torch.randn(1, 96, 64, 64),
            torch.randn(1, 128, 32, 32),
            torch.randn(1, 384, 16, 16),
            torch.randn(1, 768, 8, 8)
        ]
        expected_channels = [96, 192, 384, 768]
        
        with pytest.raises(ValueError):
            validate_features(features, expected_channels)

class TestMemoryEfficiency:
    
    def test_memory_usage_large_input(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        decoder = DPT([96, 192, 384, 768], [256, 256, 256, 256]).to(device)
        
        features = [
            torch.randn(1, 96, 256, 256).to(device),
            torch.randn(1, 192, 128, 128).to(device),
            torch.randn(1, 384, 64, 64).to(device),
            torch.randn(1, 768, 32, 32).to(device)
        ]
        
        if torch.cuda.is_available():
            mem_before = torch.cuda.memory_allocated()
            output = decoder(features)
            mem_after = torch.cuda.memory_allocated()
            mem_used = (mem_after - mem_before) / (1024 ** 2)
            logger.info(f"GPU memory used: {mem_used:.2f} MB")
        else:
            process = psutil.Process()
            mem_before = process.memory_info().rss
            output = decoder(features)
            mem_after = process.memory_info().rss
            mem_used = (mem_after - mem_before) / (1024 ** 2)
            logger.info(f"CPU memory used: {mem_used:.2f} MB")
            
        assert output.shape == (1, 1, 128, 128)
        
    def test_gradient_computation(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        decoder = DPT([96, 192, 384, 768], [256, 256, 256, 256]).to(device)
        
        features = [
            torch.randn(1, 96, 64, 64, requires_grad=True).to(device),
            torch.randn(1, 192, 32, 32, requires_grad=True).to(device),
            torch.randn(1, 384, 16, 16, requires_grad=True).to(device),
            torch.randn(1, 768, 8, 8, requires_grad=True).to(device)
        ]
        
        output = decoder(features)
        loss = output.sum()
        loss.backward()
        
        for i, feat in enumerate(features):
            assert feat.grad is not None
            logger.info(f"Gradient computed for feature {i}")

class TestInterpolation:
    
    def test_interpolate_to_size(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        features = torch.randn(2, 128, 32, 32).to(device)
        target_size = (64, 64)
        
        output = interpolate_to_size(features, target_size)
        
        assert output.shape == (2, 128, 64, 64)
        logger.info("Interpolation test passed")
        
    def test_interpolate_different_sizes(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        features = torch.randn(1, 256, 16, 16).to(device)
        target_sizes = [(32, 32), (64, 64), (128, 128), (256, 256)]
        
        for target_size in target_sizes:
            output = interpolate_to_size(features, target_size)
            assert output.shape[-2:] == target_size
            logger.info(f"Interpolation to {target_size} passed")

class TestIntegration:
    
    def test_full_pipeline(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        decoder = DPT([96, 192, 384, 768], [256, 256, 256, 256], 
                     use_attention=True, final_activation='sigmoid').to(device)
        
        batch_size = 2
        features = [
            torch.randn(batch_size, 96, 64, 64).to(device),
            torch.randn(batch_size, 192, 32, 32).to(device),
            torch.randn(batch_size, 384, 16, 16).to(device),
            torch.randn(batch_size, 768, 8, 8).to(device)
        ]
        
        validate_features(features, [96, 192, 384, 768])
        
        with torch.no_grad():
            output = decoder(features)
            
        assert output.shape == (batch_size, 1, 32, 32)
        assert torch.all(output >= 0) and torch.all(output <= 1)
        logger.info("Full pipeline integration test passed")
        
    def test_error_handling(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        decoder = DPT([96, 192, 384, 768], [256, 256, 256, 256]).to(device)
        
        wrong_features = [
            torch.randn(1, 96, 64, 64).to(device),
            torch.randn(1, 192, 32, 32).to(device)
        ]
        
        with pytest.raises(ValueError):
            decoder(wrong_features)
            
        logger.info("Error handling test passed")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])