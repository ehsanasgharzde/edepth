# FILE: tests/test_components.py
# ehsanasgharzde - COMPLETE COMPONENT TEST SUITE
# hosseinsolymanzadeh - PROPER COMMENTING   

import torch
import pytest
from models.backbones.backbone import ViT
from models.decoders.decoder import DPT
from losses.losses import SiLogLoss, EdgeAwareSmoothnessLoss, GradientConsistencyLoss

# Test to check the output shape of the ViT backbone for a specific image and patch size
@pytest.mark.parametrize('img_size,patch_size', [((224, 224), 16)])
def test_backbone_output_shape(img_size, patch_size):
    # Instantiate ViT model with selected layers for feature extraction
    model = ViT(model_name='vit_base_patch16_224', extract_layers=[3, 6, 9, 12], pretrained=False)
    # Create a dummy input tensor with specified image size
    x = torch.randn(1, 3, *img_size)
    # Forward pass through the model
    feats = model(x)
    
    # Assert that the number of extracted features matches the number of layers specified
    assert len(feats) == 4
    # Ensure each feature map has valid shape (non-zero spatial dimensions and correct batch size)
    for f in feats:
        assert f.shape[0] == 1 and f.shape[1] > 0 and f.shape[2] > 0 and f.shape[3] > 0

# Test to verify decoder's fusion output shape and validity
@pytest.mark.parametrize('C', [256])
def test_decoder_fusion_correctness(C):
    # Create DPT decoder with uniform feature dimensions and decoder channels
    decoder = DPT(feature_dims=[C, C, C, C], decoder_channels=[C, C, C, C], patch_size=16, num_classes=1)
    # Create dummy feature maps with same shape for fusion
    feats = [torch.randn(1, C, 14, 14), torch.randn(1, C, 14, 14), torch.randn(1, C, 14, 14), torch.randn(1, C, 14, 14)]
    # Forward pass through the decoder
    out = decoder(feats)
    
    # Assert output has expected batch size and single output channel
    assert out.shape[0] == 1 and out.shape[1] == 1
    # Assert spatial dimensions are valid (non-zero)
    assert out.shape[2] > 0 and out.shape[3] > 0

# Test to check multiple loss functions return finite, non-negative values
@pytest.mark.parametrize('shape', [(1, 1, 32, 32)])
def test_losses_with_dummy_tensors(shape):
    # Create random prediction and target tensors, ensuring they're not zero
    pred = torch.rand(*shape) + 0.1
    target = torch.rand(*shape) + 0.1
    # Dummy input image for edge-aware loss
    image = torch.rand(1, 3, 32, 32)
    # Binary mask where the target is greater than zero
    mask = (target > 0)
    # Compute SiLog loss
    silog = SiLogLoss()(pred, target, mask)
    # Compute edge-aware smoothness loss
    edge = EdgeAwareSmoothnessLoss()(pred, image)
    # Compute gradient consistency loss
    grad = GradientConsistencyLoss()(pred, target, mask)
    
    # Ensure all computed losses are finite and non-negative
    assert torch.isfinite(silog)
    assert silog >= 0
    assert torch.isfinite(edge)
    assert edge >= 0
    assert torch.isfinite(grad)
    assert grad >= 0 
