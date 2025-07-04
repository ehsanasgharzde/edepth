import torch
from PIL import Image
import numpy as np
from typing import Union, Tuple
from ..models.model_fixed import FixedDPTModel
import logging
import traceback
from torch.cuda.amp import autocast

def preprocess_image(img: Union[str, Image.Image], img_size: Tuple[int, int], mean, std) -> torch.Tensor:
    if isinstance(img, str):
        img = Image.open(img).convert('RGB')
    img = img.resize(img_size[::-1], Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - mean) / std
    arr = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return arr

def postprocess_depth(depth: torch.Tensor, min_depth: float, max_depth: float) -> np.ndarray:
    depth = depth.squeeze().cpu().numpy()
    depth = np.clip(depth, min_depth, max_depth)
    return depth

def run_inference(
    model_ckpt: str,
    image: Union[str, Image.Image],
    img_size: Tuple[int, int],
    mean, std,
    min_depth: float = 0.1,
    max_depth: float = 10.0,
    device: str = 'cuda',
    model_kwargs=None
):
    model_kwargs = model_kwargs or {}
    logging.info(f"Running inference on device: {device}")
    try:
        model = FixedDPTModel(**model_kwargs).to(device)
        logging.info(f"Model loaded to device: {device}")
        state = torch.load(model_ckpt, map_location=device)
        model.load_state_dict(state['model_state_dict'], strict=False)
        model.eval()
        x = preprocess_image(image, img_size, mean, std).to(device)
        with torch.no_grad():
            with autocast():
                pred = model(x)
        depth = postprocess_depth(pred, min_depth, max_depth)
        return depth
    except Exception as e:
        logging.error(f"Error during inference: {e}\n{traceback.format_exc()}")
        raise 