# FILE: inference/inference.py
# ehsanasgharzde - INFERENCE ENGINE
# hosseinsolymanzadeh - PROPER COMMENTING

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from typing import Union, List, Tuple, Optional, Dict, Any
from pathlib import Path
import logging
import traceback
from torch.cuda.amp import autocast #type: ignore 
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from dataclasses import dataclass, field
from torchvision import transforms
from tqdm import tqdm
import gc
import os
import json
import yaml

from ..models.edepth import edepth
from ..config import ModelConfig, DataConfig

logger = logging.getLogger(__name__)

@dataclass
class InferenceConfig:
    # Batch size for inference
    batch_size: int = 1
    # Use mixed precision (e.g., float16) to speed up inference and reduce memory
    use_mixed_precision: bool = True
    # Whether to estimate confidence of predictions
    confidence_estimation: bool = False
    # Number of Monte Carlo samples to use if confidence estimation enabled
    monte_carlo_samples: int = 10
    # Desired output format of inference results
    output_format: str = 'numpy'
    # Whether to save confidence maps alongside predictions
    save_confidence_maps: bool = False
    # Number of worker threads for preprocessing
    preprocessing_workers: int = 4
    # Number of worker threads for postprocessing
    postprocessing_workers: int = 2
    # Flag for memory-efficient processing
    memory_efficient: bool = True
    # Enable test-time augmentation (TTA)
    tta_enabled: bool = False
    # List of TTA transforms to apply if enabled
    tta_transforms: Optional[List[str]] = field(default_factory=lambda: ['horizontal_flip', 'vertical_flip'])
    # Device to run inference on (e.g., 'cuda' or 'cpu')
    device: str = 'cuda'

    def __post_init__(self):
        # Validate output_format is one of allowed values
        if self.output_format not in ['numpy', 'torch', 'pil']:
            raise ValueError(f"Invalid output_format: {self.output_format}")
        # Ensure monte_carlo_samples is positive if confidence estimation enabled
        if self.confidence_estimation and self.monte_carlo_samples <= 0:
            raise ValueError("monte_carlo_samples must be > 0 for confidence estimation")

class ProcessingPipeline:
    def __init__(self, img_size: Tuple[int, int], mean: np.ndarray, std: np.ndarray,
                 min_depth: float = 0.1, max_depth: float = 10.0, num_workers: int = 4):
        # Target image size for resizing
        self.img_size = img_size
        # Mean used for normalization
        self.mean = mean
        # Standard deviation used for normalization
        self.std = std
        # Minimum depth value for normalization/scaling
        self.min_depth = min_depth
        # Maximum depth value for normalization/scaling
        self.max_depth = max_depth
        # Number of worker threads for parallel processing
        self.num_workers = num_workers
        # Thread pool executor for async task execution
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        # Cache to store preprocessed images keyed by filename or identifier
        self.cache = {}
        
        # Compose transformation pipeline for input images
        self.transform = transforms.Compose([
            # Resize image to target size with bilinear interpolation
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR),
            # Convert image to PyTorch tensor and scale pixel values to [0,1]
            transforms.ToTensor(),
            # Normalize tensor using mean and std
            transforms.Normalize(mean=self.mean.tolist(), std=self.std.tolist())
        ])
        
        # Dictionary mapping output format strings to conversion functions
        self.output_converters = {
            # Convert tensor to numpy array on CPU
            'numpy': lambda x: x.cpu().numpy(),
            # Keep tensor on CPU
            'torch': lambda x: x.cpu(),
            # Convert tensor to PIL Image with normalization
            'pil': self._to_pil
        }
    
    def _to_pil(self, depth: torch.Tensor) -> Image.Image:
        # Convert depth tensor to numpy array on CPU
        depth_np = depth.cpu().numpy()
        # Normalize depth values to 0-255 range for visualization
        depth_norm = 255 * (depth_np - self.min_depth) / (self.max_depth - self.min_depth)
        # Clip values to valid byte range and convert to uint8
        depth_norm = np.clip(depth_norm, 0, 255).astype(np.uint8)
        # Create PIL Image from normalized depth array
        return Image.fromarray(depth_norm)
    
    def _load_image(self, img: Union[str, Image.Image, np.ndarray]) -> Image.Image:
        # Load image from file path string
        if isinstance(img, str):
            return Image.open(img).convert('RGB')
        # Convert numpy array image to PIL Image
        elif isinstance(img, np.ndarray):
            return Image.fromarray(img).convert('RGB')
        # If already a PIL Image, convert mode to RGB
        elif isinstance(img, Image.Image):
            return img.convert('RGB')
        else:
            # Raise error for unsupported image types
            raise TypeError(f"Unsupported image type: {type(img)}")
    
    def preprocess(self, img: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
        try:
            # Use file path string as cache key if available
            cache_key = str(img) if isinstance(img, str) else None
            # Return cached tensor if available
            if cache_key and cache_key in self.cache:
                return self.cache[cache_key]
            
            # Load image as PIL Image
            pil_img = self._load_image(img)
            # Apply transformations and add batch dimension
            tensor_img = self.transform(pil_img).unsqueeze(0) #type: ignore 
            
            # Cache the result if caching key is available
            if cache_key:
                self.cache[cache_key] = tensor_img
            
            return tensor_img
        except Exception as e:
            # Log error and return zero tensor on failure
            logger.error(f"Preprocessing failed for {img}: {e}")
            return torch.zeros((1, 3, *self.img_size))
    
    def preprocess_batch(self, images: List[Union[str, Image.Image, np.ndarray]]) -> torch.Tensor:
        # Submit preprocessing tasks asynchronously for each image in the batch
        futures = [self.executor.submit(self.preprocess, img) for img in images]
        results = []
        
        # Collect results as they complete, showing progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing"):
            try:
                # Append the preprocessed tensor result
                results.append(future.result())
            except Exception as e:
                # Log error and append zero tensor if preprocessing fails
                logger.error(f"Batch preprocessing error: {e}")
                results.append(torch.zeros((1, 3, *self.img_size)))
        
        # Concatenate all preprocessed tensors along the batch dimension
        return torch.cat(results, dim=0)
    
    def postprocess(self, depth: torch.Tensor, output_format: str = 'numpy') -> Union[np.ndarray, torch.Tensor, Image.Image]:
        try:
            # Remove batch dimension if present
            depth = depth.squeeze()
            # Move tensor to CPU if not already there
            if depth.device.type != 'cpu':
                depth = depth.cpu()
            
            # Clamp depth values to configured min and max range
            depth = torch.clamp(depth, self.min_depth, self.max_depth)
            
            # Select converter function for desired output format
            converter = self.output_converters.get(output_format, self.output_converters['numpy'])
            # Convert and return processed depth
            return converter(depth)
        except Exception as e:
            # Log error and return zero tensor or array depending on output format
            logger.error(f"Postprocessing failed: {e}")
            return np.zeros((1, 1)) if output_format == 'numpy' else torch.zeros((1, 1))
    
    def postprocess_batch(self, depths: torch.Tensor, output_format: str = 'numpy') -> List[Any]:
        # Apply postprocessing on each depth map in the batch individually
        return [self.postprocess(depth, output_format) for depth in depths]
    
    def apply_tta(self, img: torch.Tensor, transforms: List[str]) -> List[torch.Tensor]:
        # Initialize list with original image tensor
        augmented = [img]
        
        # Apply each test-time augmentation transform to create augmented versions
        for transform in transforms:
            if transform == 'horizontal_flip':
                augmented.append(torch.flip(img, dims=[-1]))  # Flip horizontally
            elif transform == 'vertical_flip':
                augmented.append(torch.flip(img, dims=[-2]))  # Flip vertically
            elif transform == 'rotate90':
                augmented.append(torch.rot90(img, k=1, dims=[-2, -1]))  # Rotate 90 degrees
            elif transform == 'rotate180':
                augmented.append(torch.rot90(img, k=2, dims=[-2, -1]))  # Rotate 180 degrees
            elif transform == 'rotate270':
                augmented.append(torch.rot90(img, k=3, dims=[-2, -1]))  # Rotate 270 degrees
        
        # Return list of original and augmented images
        return augmented
    
    def reverse_tta(self, predictions: List[torch.Tensor], transforms: List[str]) -> torch.Tensor:
        # Initialize with prediction of original (non-augmented) image
        reversed_preds = [predictions[0]]
        
        # Reverse each augmentation transform applied, for all augmented predictions
        for i, transform in enumerate(transforms, 1):
            if i < len(predictions):
                pred = predictions[i]
                if transform == 'horizontal_flip':
                    reversed_preds.append(torch.flip(pred, dims=[-1]))  # Flip back horizontally
                elif transform == 'vertical_flip':
                    reversed_preds.append(torch.flip(pred, dims=[-2]))  # Flip back vertically
                elif transform == 'rotate90':
                    reversed_preds.append(torch.rot90(pred, k=3, dims=[-2, -1]))  # Rotate back -90 degrees
                elif transform == 'rotate180':
                    reversed_preds.append(torch.rot90(pred, k=2, dims=[-2, -1]))  # Rotate back 180 degrees (self-inverse)
                elif transform == 'rotate270':
                    reversed_preds.append(torch.rot90(pred, k=1, dims=[-2, -1]))  # Rotate back 90 degrees
                else:
                    # If unknown transform, keep prediction as is
                    reversed_preds.append(pred)
        
        # Average all reversed predictions for final output
        return torch.stack(reversed_preds).mean(dim=0)

class ModelManager:
    def __init__(self, model_config: ModelConfig, device: str = 'cuda'):
        # Store model configuration
        self.model_config = model_config
        # Set device to 'cuda' if available and requested, otherwise 'cpu'
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        # Cache to store loaded models keyed by checkpoint path
        self.model_cache = {}
        
        # Log initialization and selected device
        logger.info(f"ModelManager initialized on device: {self.device}")
    
    def load_model(self, model_ckpt: str, model_kwargs: Dict[str, Any] = None) -> torch.nn.Module: #type: ignore 
        # Return cached model if already loaded
        if model_ckpt in self.model_cache:
            logger.info(f"Loading model from cache: {model_ckpt}")
            return self.model_cache[model_ckpt]
        
        logger.info(f"Loading model checkpoint: {model_ckpt}")
        
        try:
            # Instantiate model with provided keyword arguments
            model = edepth(**model_kwargs or {})
            # Load checkpoint with correct device mapping
            checkpoint = torch.load(model_ckpt, map_location=self.device)
            
            # Extract state_dict from checkpoint, handling possible wrapper keys
            state_dict = checkpoint.get('state_dict', checkpoint)
            # Remove 'module.' prefix if present from keys (common in DataParallel models)
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            # Load weights into model
            model.load_state_dict(new_state_dict)
            # Move model to appropriate device
            model.to(self.device)
            # Set model to evaluation mode
            model.eval()
            
            # Optimize model (e.g., scripting, backend tuning)
            model = self._optimize_model(model)
            # Cache the loaded and optimized model
            self.model_cache[model_ckpt] = model
            
            logger.info("Model loaded and optimized successfully")
            return model
            
        except Exception as e:
            # Log and raise exception on failure
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _optimize_model(self, model: torch.nn.Module) -> torch.nn.Module:
        # Enable cudnn benchmark mode for potentially faster runtime on CUDA
        if self.device == 'cuda':
            torch.backends.cudnn.benchmark = True #type: ignore 
        
        try:
            # Try to convert model to TorchScript via scripting for optimization
            model = torch.jit.script(model) #type: ignore 
            logger.info("Model successfully scripted")
        except Exception as e:
            # Log warning if scripting fails, but continue
            logger.warning(f"JIT scripting failed: {e}")
        
        return model
    
    def infer(self, model: torch.nn.Module, input_tensor: torch.Tensor, 
              use_mixed_precision: bool = True) -> torch.Tensor:
        # Record inference start time for logging
        start_time = time.time()
        
        with torch.no_grad():
            # Use automatic mixed precision if enabled and device is CUDA
            if use_mixed_precision and self.device == 'cuda':
                with autocast(): #type: ignore 
                    output = model(input_tensor.to(self.device))
            else:
                # Regular inference without mixed precision
                output = model(input_tensor.to(self.device))
        
        # Log inference duration
        logger.info(f"Inference completed in {time.time() - start_time:.4f}s")
        return output

class ConfidenceEstimator:
    def __init__(self, method: str = 'monte_carlo', num_samples: int = 10):
        # Store confidence estimation method (e.g., 'monte_carlo', 'ensemble')
        self.method = method
        # Number of samples or models to use for confidence estimation
        self.num_samples = num_samples
    
    def estimate_confidence(self, model: torch.nn.Module, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: #type: ignore 
        # Dispatch to appropriate confidence estimation method
        if self.method == 'monte_carlo':
            return self._monte_carlo_confidence(model, input_tensor)
        elif self.method == 'ensemble':
            return self._ensemble_confidence(model, input_tensor) #type: ignore 
        else:
            # Raise error for unknown methods
            raise ValueError(f"Unknown confidence method: {self.method}")
    
    def _monte_carlo_confidence(self, model: torch.nn.Module, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: #type: ignore 
        # Enable dropout layers during inference by setting them to train mode
        def enable_dropout(m):
            if isinstance(m, torch.nn.Dropout):
                m.train()
        
        # Apply dropout enabling function recursively to all submodules
        model.apply(enable_dropout)
        predictions = []
        
        with torch.no_grad():
            # Perform multiple forward passes to collect stochastic predictions
            for _ in range(self.num_samples):
                pred = model(input_tensor)
                predictions.append(pred)
        
        # Stack predictions into one tensor
        predictions = torch.stack(predictions) #type: ignore 
        # Compute mean prediction across samples
        mean_pred = predictions.mean(dim=0) #type: ignore 
        # Compute variance of predictions as uncertainty measure
        var_pred = predictions.var(dim=0) #type: ignore 
        # Confidence inversely proportional to variance, small epsilon to avoid div by zero
        confidence = 1.0 / (var_pred + 1e-6)
        
        return mean_pred, confidence
    
    def _ensemble_confidence(self, predictions: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]: #type: ignore 
        # Stack ensemble predictions into one tensor
        predictions = torch.stack(predictions) #type: ignore 
        # Compute mean prediction across ensemble models
        mean_pred = predictions.mean(dim=0) #type: ignore 
        # Compute variance of predictions as uncertainty
        var_pred = predictions.var(dim=0) #type: ignore 
        # Confidence is inverse of variance with epsilon
        confidence = 1.0 / (var_pred + 1e-6)
        
        return mean_pred, confidence
class InferenceEngine:
    def __init__(self, model_ckpt: str, model_config: ModelConfig, 
                 data_config: DataConfig, inference_config: InferenceConfig): #type: ignore 
        # Path to the model checkpoint file
        self.model_ckpt = model_ckpt
        # Configuration for the model architecture and parameters
        self.model_config = model_config
        # Configuration for dataset-related parameters like image size, normalization
        self.data_config = data_config
        # Configuration for inference parameters like batch size, device, etc.
        self.inference_config = inference_config
        
        # Initialize image processing pipeline with dataset image size and normalization params
        self.processing = ProcessingPipeline(
            data_config.img_size, data_config.mean, data_config.std, #type: ignore 
            data_config.min_depth, data_config.max_depth
        )
        
        # Manager responsible for loading and running the model on device
        self.model_manager = ModelManager(model_config, inference_config.device)
        # Load model weights from checkpoint
        self.model = self.model_manager.load_model(model_ckpt)
        
        # Initialize confidence estimator if enabled in inference config
        self.confidence_estimator = None
        if inference_config.confidence_estimation:
            self.confidence_estimator = ConfidenceEstimator(
                num_samples=inference_config.monte_carlo_samples
            )
        
        # Clear GPU and Python memory caches at initialization
        self._clear_cache()
    
    def _clear_cache(self):
        # Empty CUDA cache if running on GPU
        if self.inference_config.device == 'cuda':
            torch.cuda.empty_cache()
        # Run garbage collector to free up Python memory
        gc.collect()
    
    def predict_single(self, image: Union[str, Image.Image, np.ndarray],
                      return_confidence: bool = False) -> Dict[str, Any]:
        # Record start time for measuring inference duration
        start_time = time.time()
        
        # Preprocess the input image into tensor ready for model inference
        input_tensor = self.processing.preprocess(image)
        
        if self.inference_config.tta_enabled:
            # Apply test-time augmentations to the input tensor
            augmented = self.processing.apply_tta(input_tensor, self.inference_config.tta_transforms) #type: ignore 
            predictions = []
            
            # Run inference on each augmented input and collect predictions
            for aug_input in augmented:
                pred = self.model_manager.infer(self.model, aug_input, self.inference_config.use_mixed_precision)
                predictions.append(pred)
            
            # Reverse the TTA augmentations on predictions to get final output
            prediction = self.processing.reverse_tta(predictions, self.inference_config.tta_transforms) #type: ignore 
        else:
            # Direct inference without TTA
            prediction = self.model_manager.infer(self.model, input_tensor, self.inference_config.use_mixed_precision)
        
        # Initialize confidence to None
        confidence = None
        # If requested and enabled, estimate confidence for prediction
        if return_confidence and self.confidence_estimator:
            _, confidence = self.confidence_estimator.estimate_confidence(self.model, input_tensor)
        
        # Postprocess the prediction into desired output format (numpy, torch, pil)
        depth_map = self.processing.postprocess(prediction, self.inference_config.output_format)
        
        # Prepare result dictionary with depth map and inference time
        result = {
            'depth_map': depth_map,
            'inference_time': time.time() - start_time,
        }
        
        # Include confidence map if available
        if confidence is not None:
            result['confidence'] = self.processing.postprocess(confidence, self.inference_config.output_format)
        
        return result
    
    def predict_batch(self, images: List[Union[str, Image.Image, np.ndarray]],
                      batch_size: int = None, return_confidence: bool = False) -> Dict[str, Any]: #type: ignore 
        # Use provided batch_size or fallback to config batch size
        batch_size = batch_size or self.inference_config.batch_size
        # List to collect results for all images
        results = []
        # Total time taken for batch processing
        total_time = 0

        # Process images in batches of size batch_size
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            # Start timer for this batch
            batch_start = time.time()

            # Preprocess batch of images to tensors
            input_tensor = self.processing.preprocess_batch(batch_images)
            # Run inference on batch tensor
            predictions = self.model_manager.infer(self.model, input_tensor, self.inference_config.use_mixed_precision)

            # Postprocess batch predictions into desired output format
            batch_results = self.processing.postprocess_batch(predictions, self.inference_config.output_format)

            # Initialize confidences to None
            confidences = None
            # Estimate confidence maps for batch if requested and enabled
            if return_confidence and self.confidence_estimator:
                _, conf_tensor = self.confidence_estimator.estimate_confidence(self.model, input_tensor)
                confidences = self.processing.postprocess_batch(conf_tensor, self.inference_config.output_format)

            # Combine depth maps and confidence maps (if any) into results list
            for j, depth_map in enumerate(batch_results):
                result = {'depth_map': depth_map}
                if confidences:
                    result['confidence'] = confidences[j]
                results.append(result)

            # Measure time taken for this batch and accumulate
            batch_time = time.time() - batch_start
            total_time += batch_time

            # Clear GPU and Python caches after each batch
            self._clear_cache()

        # Return dict with all results, total processing time and throughput (images/sec)
        return {
            'results': results,
            'total_time': total_time,
            'throughput': len(images) / total_time if total_time > 0 else 0
        }

    def predict_directory(self, input_dir: str, output_dir: str = None, #type: ignore 
                          file_extensions: List[str] = None) -> Dict[str, Any]: #type: ignore 
        # Default accepted image file extensions if none provided
        if file_extensions is None:
            file_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

        # Convert input directory to Path object
        input_path = Path(input_dir)
        # Recursively list all image files matching extensions
        image_files = [p for p in input_path.rglob('*') if p.suffix.lower() in file_extensions]

        images = []
        # List to keep track of files that fail to load
        failed_files = []

        # Try to collect file paths as strings for processing
        for file_path in image_files:
            try:
                images.append(str(file_path))
            except Exception as e:
                failed_files.append({'file': str(file_path), 'error': str(e)})
                logger.error(f"Failed to load {file_path}: {e}")

        # Run batch prediction on all images collected
        results = self.predict_batch(images, return_confidence=self.inference_config.confidence_estimation)

        # Save results to disk if output directory specified
        if output_dir:
            self._save_results(results['results'], image_files, output_dir)

        # Return summary dict including counts and any errors
        return {
            'processed_files': len(images),
            'failed_files': len(failed_files),
            'results': results,
            'errors': failed_files
        }

    def _save_results(self, results: List[Dict[str, Any]], image_files: List[Path], output_dir: str): #type: ignore 
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Iterate over results and corresponding input files
        for i, (result, file_path) in enumerate(zip(results, image_files)):
            depth_map = result['depth_map']
            # Save depth map as normalized PNG if it is a numpy array
            if isinstance(depth_map, np.ndarray):
                normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) #type: ignore 
                output_file = output_path / f"{file_path.stem}_depth.png"
                cv2.imwrite(str(output_file), normalized)

            # Save confidence map if present and a numpy array
            if 'confidence' in result:
                conf_map = result['confidence']
                if isinstance(conf_map, np.ndarray):
                    conf_normalized = cv2.normalize(conf_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) #type: ignore 
                    conf_file = output_path / f"{file_path.stem}_confidence.png"
                    cv2.imwrite(str(conf_file), conf_normalized)

def load_config(config_path: str) -> InferenceConfig:
    # Open the configuration file at the given path
    with open(config_path, 'r') as f:
        # Check if the file is a YAML file based on its extension
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            # Parse YAML configuration
            config_data = yaml.safe_load(f)
        else:
            # Parse JSON configuration
            config_data = json.load(f)
    
    # Create and return an InferenceConfig object initialized with parsed data
    return InferenceConfig(**config_data)

def create_inference_engine(model_ckpt: str, model_config: ModelConfig = None, #type: ignore 
                           data_config: DataConfig = None, #type: ignore 
                           inference_config: InferenceConfig = None) -> InferenceEngine: #type: ignore 
    # Initialize model configuration if not provided
    if model_config is None:
        model_config = ModelConfig()
    # Initialize data configuration if not provided
    if data_config is None:
        data_config = DataConfig()
    # Initialize inference configuration if not provided
    if inference_config is None:
        inference_config = InferenceConfig()
    
    # Create and return an InferenceEngine instance with given configurations
    return InferenceEngine(model_ckpt, model_config, data_config, inference_config)

def benchmark_inference(model_ckpt: str, test_images: List[str], 
                       batch_sizes: List[int] = [1, 4, 8, 16]) -> Dict[str, Any]: #type: ignore 
    # Dictionary to store benchmark results for each batch size
    results = {}
    
    # Iterate over each batch size to test performance
    for batch_size in batch_sizes:
        try:
            # Create an inference configuration for current batch size
            config = InferenceConfig(batch_size=batch_size)
            # Create the inference engine with this configuration
            engine = create_inference_engine(model_ckpt, inference_config=config)
            
            # Record start time before prediction
            start_time = time.time()
            # Run batch prediction on test images
            output = engine.predict_batch(test_images, batch_size=batch_size) #type: ignore 
            # Calculate elapsed time for inference
            elapsed = time.time() - start_time
            
            # Store timing, throughput, and memory usage results
            results[batch_size] = {
                'time': elapsed,
                'throughput': len(test_images) / elapsed,
                'memory_used': torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            }
            
        except Exception as e:
            # On error, store error message and log it
            results[batch_size] = {'error': str(e)}
            logger.error(f"Benchmark failed for batch_size {batch_size}: {e}")
    
    # Determine the batch size with the highest throughput among successful runs
    optimal_batch_size = max(
        (k for k, v in results.items() if 'throughput' in v),
        key=lambda k: results[k]['throughput'],
        default=1
    )
    
    # Store the optimal batch size in results
    results['optimal_batch_size'] = optimal_batch_size
    return results

def setup_logging(level: str = 'INFO'):
    # Configure logging with specified level, format, and output handlers
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),    # Log to console
            logging.FileHandler('inference.log')  # Log to file
        ]
    )

def main():
    # Initialize logging configuration
    setup_logging()
    
    # Define path to the model checkpoint file
    model_ckpt = "path/to/model.pth"
    # Define list of test image paths
    test_images = ["path/to/image1.jpg", "path/to/image2.jpg"]
    
    # Create inference engine with default configurations
    engine = create_inference_engine(model_ckpt)
    
    # Run inference on a single image with confidence output
    single_result = engine.predict_single(test_images[0], return_confidence=True)
    # Log inference time for single image
    logger.info(f"Single inference completed: {single_result['inference_time']:.4f}s")
    
    # Run inference on batch of images with confidence output
    batch_results = engine.predict_batch(test_images, return_confidence=True) #type: ignore 
    # Log throughput for batch inference
    logger.info(f"Batch inference completed: {batch_results['throughput']:.2f} images/s")
    
    # Perform benchmarking across different batch sizes
    benchmark_results = benchmark_inference(model_ckpt, test_images)
    # Log the optimal batch size found
    logger.info(f"Optimal batch size: {benchmark_results['optimal_batch_size']}")

# Entry point for script execution
if __name__ == "__main__":
    main()

