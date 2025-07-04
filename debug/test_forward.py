import torch
from ..models.model_fixed import FixedDPTModel
import logging
import traceback

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Running test_forward on device: {device}")
    try:
        model = FixedDPTModel(
            backbone_name='vit_base_patch16_224',
            extract_layers=[3, 6, 9, 12],
            decoder_channels=[256, 512, 1024, 1024],
            patch_size=16,
            num_classes=1,
            pretrained=False
        ).to(device)
        x = torch.randn(1, 3, 224, 224).to(device)
        y = model(x)
        print(y.shape)  # Should be [1, 1, 224, 224]
    except Exception as e:
        logging.error(f"Error in test_forward: {e}\n{traceback.format_exc()}") 