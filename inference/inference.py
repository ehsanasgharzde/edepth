# FILE: inference/inference.py
# ehsanasgharzde - INFERENCE ENGINE
# hosseinsolymanzadeh - PROPER COMMENTING
# ehsanasgharzde, hosseinsolymanzadeh - FIXED REDUNDANT CODE BY EXTRACTING PURE FUNCTIONS AND BASECLASS LEVEL METHODS

import torch
import cv2
import numpy as np
from typing import Union, List, Tuple, Optional, Dict, Any
from pathlib import Path
import logging
from torch.cuda.amp import autocast # type: ignore
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from dataclasses import dataclass
import gc
import json
import yaml
from tqdm import tqdm

from models.factory import create_model, create_model_from_checkpoint
from config import Config, ConfigFactory, config_manager

logger = logging.getLogger(__name__)

@dataclass
class InferenceConfig:
    batch_size: int = 1
    use_mixed_precision: bool = True
    confidence_estimation: bool = False
    monte_carlo_samples: int = 10
    output_format: str = 'numpy'
    save_confidence_maps: bool = False
    preprocessing_workers: int = 4
    postprocessing_workers: int = 2
    memory_efficient: bool = True
    tta_enabled: bool = False
    tta_transforms: List[str] = None # type: ignore
    device: str = 'cuda'

    def __post_init__(self):
        if self.tta_transforms is None:
            self.tta_transforms = ['horizontal_flip']
        if self.output_format not in ['numpy', 'torch', 'opencv']:
            raise ValueError(f"Invalid output_format: {self.output_format}")
        if self.confidence_estimation and self.monte_carlo_samples <= 0:
            raise ValueError("monte_carlo_samples must be > 0 for confidence estimation")

# Image loading and preprocessing functions
def load_image_cv2(image_path: Union[str, Path]) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def resize_image_cv2(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_LANCZOS4)

def normalize_image(image: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    image_norm = image.astype(np.float32) / 255.0
    for i in range(3):
        image_norm[:, :, i] = (image_norm[:, :, i] - mean[i]) / std[i]
    return image_norm

def preprocess_single_image(
    image_input: Union[str, Path, np.ndarray],
    target_size: Tuple[int, int],
    mean: np.ndarray,
    std: np.ndarray
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    try:
        if isinstance(image_input, (str, Path)):
            image = load_image_cv2(image_input)
        else:
            image = image_input
        
        original_size = (image.shape[0], image.shape[1])  # H, W
        
        # Resize and normalize
        image_resized = resize_image_cv2(image, target_size)
        image_norm = normalize_image(image_resized, mean, std)
        
        # Convert to tensor (H, W, C) -> (C, H, W) -> (1, C, H, W)
        tensor = torch.from_numpy(image_norm).permute(2, 0, 1).unsqueeze(0)
        
        return tensor, original_size
        
    except Exception as e:
        logger.error(f"Preprocessing failed for {image_input}: {e}")
        return torch.zeros((1, 3, *target_size)), (target_size[0], target_size[1])

def preprocess_batch_images(
    images: List[Union[str, Path, np.ndarray]],
    target_size: Tuple[int, int],
    mean: np.ndarray,
    std: np.ndarray,
    max_workers: int = 4
) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(preprocess_single_image, img, target_size, mean, std)
            for img in images
        ]
        
        results = []
        original_sizes = []
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing"):
            tensor, orig_size = future.result()
            results.append(tensor)
            original_sizes.append(orig_size)
    
    return torch.cat(results, dim=0), original_sizes

# Postprocessing functions
def postprocess_depth_cv2(
    depth_tensor: torch.Tensor,
    original_size: Tuple[int, int],
    min_depth: float = 0.1,
    max_depth: float = 10.0
) -> np.ndarray:
    depth = depth_tensor.squeeze().cpu().numpy()
    
    # Resize to original size
    depth_resized = cv2.resize(depth, (original_size[1], original_size[0]), interpolation=cv2.INTER_LANCZOS4)
    
    # Scale to metric depth range
    depth_metric = depth_resized * (max_depth - min_depth) + min_depth
    
    return depth_metric

def create_depth_visualization_cv2(depth: np.ndarray, colormap: int = cv2.COLORMAP_PLASMA) -> np.ndarray:
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
    depth_8bit = (depth_norm * 255).astype(np.uint8)
    colored_depth = cv2.applyColorMap(depth_8bit, colormap)
    return cv2.cvtColor(colored_depth, cv2.COLOR_BGR2RGB)

def save_depth_outputs_cv2(
    depth: np.ndarray,
    output_path: Path,
    save_raw: bool = True,
    save_colored: bool = True,
    save_16bit: bool = True
) -> Dict[str, Path]:
    outputs = {}
    
    if save_raw:
        raw_path = output_path.with_suffix('.npy')
        np.save(raw_path, depth)
        outputs['raw'] = raw_path
    
    if save_16bit:
        depth_16bit = (depth / depth.max() * 65535).astype(np.uint16)
        png_path = output_path.with_suffix('.png')
        cv2.imwrite(str(png_path), depth_16bit)
        outputs['16bit'] = png_path
    
    if save_colored:
        colored_depth = create_depth_visualization_cv2(depth)
        colored_path = output_path.with_suffix('_colored.jpg')
        colored_bgr = cv2.cvtColor(colored_depth, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(colored_path), colored_bgr)
        outputs['colored'] = colored_path
    
    return outputs

# Test-time augmentation functions
def apply_tta_transforms(image_tensor: torch.Tensor, transforms: List[str]) -> List[torch.Tensor]:
    augmented = [image_tensor]
    
    for transform in transforms:
        if transform == 'horizontal_flip':
            augmented.append(torch.flip(image_tensor, dims=[-1]))
        elif transform == 'vertical_flip':
            augmented.append(torch.flip(image_tensor, dims=[-2]))
        elif transform == 'rotate90':
            augmented.append(torch.rot90(image_tensor, k=1, dims=[-2, -1]))
        elif transform == 'rotate180':
            augmented.append(torch.rot90(image_tensor, k=2, dims=[-2, -1]))
        elif transform == 'rotate270':
            augmented.append(torch.rot90(image_tensor, k=3, dims=[-2, -1]))
    
    return augmented

def reverse_tta_transforms(predictions: List[torch.Tensor], transforms: List[str]) -> torch.Tensor:
    reversed_preds = [predictions[0]]
    
    for i, transform in enumerate(transforms, 1):
        if i < len(predictions):
            pred = predictions[i]
            if transform == 'horizontal_flip':
                reversed_preds.append(torch.flip(pred, dims=[-1]))
            elif transform == 'vertical_flip':
                reversed_preds.append(torch.flip(pred, dims=[-2]))
            elif transform == 'rotate90':
                reversed_preds.append(torch.rot90(pred, k=3, dims=[-2, -1]))
            elif transform == 'rotate180':
                reversed_preds.append(torch.rot90(pred, k=2, dims=[-2, -1]))
            elif transform == 'rotate270':
                reversed_preds.append(torch.rot90(pred, k=1, dims=[-2, -1]))
            else:
                reversed_preds.append(pred)
    
    return torch.stack(reversed_preds).mean(dim=0)

# Model loading and inference functions
def load_model_for_inference(
    model_path: Optional[str] = None,
    model_name: str = 'vit_base_patch16_224',
    config_path: Optional[str] = None,
    device: str = 'auto'
) -> Tuple[torch.nn.Module, Config]:
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load configuration
    if config_path:
        config = config_manager.load_config_from_file(config_path)
    else:
        config = ConfigFactory.create_nyu_config()
    
    # Load model
    if model_path and Path(model_path).exists():
        logger.info(f"Loading model from checkpoint: {model_path}")
        model = create_model_from_checkpoint(
            checkpoint_path=model_path,
            map_location=device
        )
    else:
        logger.info(f"Creating new model: {model_name}")
        model = create_model(
            model_name=model_name,
            backbone_name=model_name,
            pretrained=True
        )
    
    model = model.to(device).eval()
    
    # Optimize model for inference
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True # type: ignore
        try:
            model = torch.jit.script(model)  # type: ignore
            logger.info("Model successfully scripted")
        except Exception as e:
            logger.warning(f"JIT scripting failed: {e}")
    
    logger.info(f"Model loaded on {device}")
    return model, config  # type: ignore

@torch.no_grad()
def run_model_inference(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: str,
    use_mixed_precision: bool = True
) -> torch.Tensor:
    input_tensor = input_tensor.to(device)
    
    if use_mixed_precision and device == 'cuda':
        with autocast():
            output = model(input_tensor)
    else:
        output = model(input_tensor)
    
    return output

def estimate_monte_carlo_confidence(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: str,
    num_samples: int = 10
) -> Tuple[torch.Tensor, torch.Tensor]:
    def enable_dropout(m):
        if isinstance(m, torch.nn.Dropout):
            m.train()
    
    model.apply(enable_dropout)
    predictions = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            pred = run_model_inference(model, input_tensor, device, use_mixed_precision=False)
            predictions.append(pred)
    
    predictions = torch.stack(predictions)
    mean_pred = predictions.mean(dim=0)
    var_pred = predictions.var(dim=0)
    confidence = 1.0 / (var_pred + 1e-6)
    
    return mean_pred, confidence

def clear_memory_cache(device: str):
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# Main inference functions
def predict_single_image(
    image_input: Union[str, Path, np.ndarray],
    model: torch.nn.Module,
    config: Config,
    inference_config: InferenceConfig,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    start_time = time.time()
    
    # Preprocess image
    mean = np.array(config.data.normalize_mean)
    std = np.array(config.data.normalize_std)
    
    image_tensor, original_size = preprocess_single_image(
        image_input, config.data.img_size, mean, std
    )
    
    # Apply TTA if enabled
    if inference_config.tta_enabled:
        augmented = apply_tta_transforms(image_tensor, inference_config.tta_transforms)
        predictions = []
        
        for aug_input in augmented:
            pred = run_model_inference(
                model, aug_input, inference_config.device, 
                inference_config.use_mixed_precision
            )
            predictions.append(pred)
        
        depth_pred = reverse_tta_transforms(predictions, inference_config.tta_transforms)
    else:
        depth_pred = run_model_inference(
            model, image_tensor, inference_config.device,
            inference_config.use_mixed_precision
        )
    
    # Estimate confidence if requested
    confidence = None
    if inference_config.confidence_estimation:
        _, confidence = estimate_monte_carlo_confidence(
            model, image_tensor, inference_config.device,
            inference_config.monte_carlo_samples
        )
    
    # Postprocess depth
    depth = postprocess_depth_cv2(
        depth_pred, original_size,
        config.data.min_depth, config.data.max_depth
    )
    
    results = {
        'depth_map': depth,
        'original_size': original_size,
        'processed_size': config.data.img_size,
        'inference_time': time.time() - start_time
    }
    
    if confidence is not None:
        conf_map = postprocess_depth_cv2(confidence, original_size, 0.0, 1.0)
        results['confidence'] = conf_map
    
    # Save outputs if output directory specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(image_input, (str, Path)):
            output_path = output_dir / Path(image_input).stem
        else:
            output_path = output_dir / f"image_{int(time.time())}"
        
        saved_outputs = save_depth_outputs_cv2(depth, output_path)
        results['saved_outputs'] = saved_outputs
        
        if confidence is not None and inference_config.save_confidence_maps:
            conf_outputs = save_depth_outputs_cv2(conf_map, output_path.with_suffix('_confidence'))
            results['saved_confidence'] = conf_outputs
    
    return results

def predict_batch_images(
    images: List[Union[str, Path, np.ndarray]],
    model: torch.nn.Module,
    config: Config,
    inference_config: InferenceConfig,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    total_start = time.time()
    results = []
    batch_size = inference_config.batch_size
    
    mean = np.array(config.data.normalize_mean)
    std = np.array(config.data.normalize_std)
    
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]
        batch_start = time.time()
        
        # Preprocess batch
        input_tensor, original_sizes = preprocess_batch_images(
            batch_images, config.data.img_size, mean, std,
            inference_config.preprocessing_workers
        )
        
        # Run inference
        predictions = run_model_inference(
            model, input_tensor, inference_config.device,
            inference_config.use_mixed_precision
        )
        
        # Process each prediction in batch
        for j, (pred, orig_size) in enumerate(zip(predictions, original_sizes)):
            depth = postprocess_depth_cv2(
                pred.unsqueeze(0), orig_size,
                config.data.min_depth, config.data.max_depth
            )
            
            result = {
                'depth_map': depth,
                'original_size': orig_size,
                'processed_size': config.data.img_size
            }
            
            # Save outputs if requested
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                image_input = batch_images[j]
                if isinstance(image_input, (str, Path)):
                    output_path = output_dir / Path(image_input).stem
                else:
                    output_path = output_dir / f"batch_{i}_image_{j}"
                
                saved_outputs = save_depth_outputs_cv2(depth, output_path)
                result['saved_outputs'] = saved_outputs
            
            results.append(result)
        
        batch_time = time.time() - batch_start
        logger.info(f"Processed batch {i//batch_size + 1}: {batch_time:.2f}s")
        
        # Clear cache after each batch
        clear_memory_cache(inference_config.device)
    
    total_time = time.time() - total_start
    
    return {
        'results': results,
        'total_time': total_time,
        'throughput': len(images) / total_time if total_time > 0 else 0,
        'processed_images': len(results)
    }

def predict_directory(
    input_dir: Union[str, Path],
    model: torch.nn.Module,
    config: Config,
    inference_config: InferenceConfig,
    output_dir: Optional[Path] = None,
    file_extensions: List[str] = None  # type: ignore
) -> Dict[str, Any]:
    if file_extensions is None:
        file_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']  # type: ignore
    
    input_path = Path(input_dir)
    image_files = [
        p for p in input_path.rglob('*') 
        if p.suffix.lower() in file_extensions
    ]
    
    logger.info(f"Found {len(image_files)} images in {input_dir}")
    
    failed_files = []
    valid_images = []
    
    for file_path in image_files:
        try:
            valid_images.append(str(file_path))
        except Exception as e:
            failed_files.append({'file': str(file_path), 'error': str(e)})
            logger.error(f"Failed to process {file_path}: {e}")
    
    # Process valid images
    results = predict_batch_images(
        valid_images, model, config, inference_config, output_dir
    )
    
    return {
        'processed_files': len(valid_images),
        'failed_files': len(failed_files),
        'results': results,
        'errors': failed_files
    }

def benchmark_inference(
    model_path: str,
    test_images: List[str],
    batch_sizes: List[int] = None,  # type: ignore
    config_path: Optional[str] = None
) -> Dict[str, Any]:
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8]
    
    results = {}
    
    for batch_size in batch_sizes:
        try:
            # Load model and config
            model, config = load_model_for_inference(model_path, config_path=config_path)
            
            # Create inference config
            inference_config = InferenceConfig(
                batch_size=batch_size,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            start_time = time.time()
            output = predict_batch_images(test_images, model, config, inference_config)  # type: ignore
            elapsed = time.time() - start_time
            
            results[batch_size] = {
                'time': elapsed,
                'throughput': len(test_images) / elapsed,
                'memory_used': torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            }
            
            # Reset memory stats
            if torch.cuda.is_available():
                torch.cuda.reset_max_memory_allocated()
            
        except Exception as e:
            results[batch_size] = {'error': str(e)}
            logger.error(f"Benchmark failed for batch_size {batch_size}: {e}")
    
    # Find optimal batch size
    optimal_batch_size = max(
        (k for k, v in results.items() if 'throughput' in v),
        key=lambda k: results[k]['throughput'],
        default=1
    )
    
    results['optimal_batch_size'] = optimal_batch_size
    return results

# Configuration and utility functions
def load_inference_config(config_path: str) -> InferenceConfig:
    with open(config_path, 'r') as f:
        if config_path.endswith(('.yaml', '.yml')):
            config_data = yaml.safe_load(f)
        else:
            config_data = json.load(f)
    
    return InferenceConfig(**config_data)

def create_default_inference_config(device: str = 'auto') -> InferenceConfig:
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return InferenceConfig(
        batch_size=4 if device == 'cuda' else 1,
        device=device,
        use_mixed_precision=device == 'cuda',
        memory_efficient=True
    )

def setup_inference_logging(level: str = 'INFO'):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('inference.log')
        ]
    )