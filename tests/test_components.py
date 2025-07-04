import torch
import pytest
from edepth_rewrite.models.backbones.backbone_fixed import FixedViTBackbone
from edepth_rewrite.models.decoders.decoder_fixed import DPTDecoderFixed
from edepth_rewrite.losses.losses_fixed import SiLogLoss, EdgeAwareSmoothnessLoss, GradientConsistencyLoss

@pytest.mark.parametrize('img_size,patch_size', [((224, 224), 16)])
def test_backbone_output_shape(img_size, patch_size):
    model = FixedViTBackbone(model_name='vit_base_patch16_224', extract_layers=[3, 6, 9, 12], pretrained=False)
    x = torch.randn(1, 3, *img_size)
    feats = model(x)
    assert len(feats) == 4
    for f in feats:
        assert f.shape[0] == 1 and f.shape[1] > 0 and f.shape[2] > 0 and f.shape[3] > 0

@pytest.mark.parametrize('C', [256])
def test_decoder_fusion_correctness(C):
    decoder = DPTDecoderFixed(feature_dims=[C, C, C, C], decoder_channels=[C, C, C, C], patch_size=16, num_classes=1)
    feats = [torch.randn(1, C, 14, 14), torch.randn(1, C, 14, 14), torch.randn(1, C, 14, 14), torch.randn(1, C, 14, 14)]
    out = decoder(feats)
    assert out.shape[0] == 1 and out.shape[1] == 1
    assert out.shape[2] > 0 and out.shape[3] > 0

@pytest.mark.parametrize('shape', [(1, 1, 32, 32)])
def test_losses_with_dummy_tensors(shape):
    pred = torch.rand(*shape) + 0.1
    target = torch.rand(*shape) + 0.1
    image = torch.rand(1, 3, 32, 32)
    mask = (target > 0)
    silog = SiLogLoss()(pred, target, mask)
    edge = EdgeAwareSmoothnessLoss()(pred, image)
    grad = GradientConsistencyLoss()(pred, target, mask)
    assert torch.isfinite(silog)
    assert silog >= 0
    assert torch.isfinite(edge)
    assert edge >= 0
    assert torch.isfinite(grad)
    assert grad >= 0 