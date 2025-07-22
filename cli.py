# FILE: cli.py
# ehsanasgharzde - COMPREHENSIVE COMMAND LINE INTERFACE IMPLEMENTATION
# hosseinsolymanzadeh - PROPER COMMENTING

import argparse
import logging
import sys
import os
import json
import time
from typing import Dict, Any, Optional
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
import traceback
import psutil
import platform
from pathlib import Path

from configs.config_loader import load_config
from models.edepth import edepth
from training.trainer import Trainer
from utils.export import print_deployment_plan #type: ignore 
from datasets.nyu_dataset import NYUV2Dataset
from datasets.kitti_dataset import KITTIDataset
from datasets.enrich_dataset import ENRICHDataset
from datasets.unreal_dataset import UnrealStereo4KDataset
from metrics.metrics import Metrics  # type: ignore

logger = logging.getLogger(__name__)

def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None) -> None:
    # Determine the directory for the log file if provided
    log_dir = Path(log_file).parent if log_file else None
    
    # Create the directory if it doesn't exist
    if log_file and log_dir and not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)

    # Define the logging format
    log_format = "%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
    
    # Convert log level string to logging module level
    log_level_enum = getattr(logging, log_level.upper(), logging.INFO)

    # Always log to stdout
    handlers = [logging.StreamHandler(sys.stdout)]
    
    # Optionally log to a file
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    # Configure logging with specified handlers and format
    logging.basicConfig(
        level=log_level_enum,
        format=log_format,
        handlers=handlers,
        force=True
    )

def create_argument_parser() -> argparse.ArgumentParser:
    # Create an argument parser for the CLI
    parser = argparse.ArgumentParser(
        description="edepth SOTA CLI - Monocular Depth Estimation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=""" 
Examples:
  Training:       python cli.py --config configs/nyu.yaml --mode train
  Evaluation:     python cli.py --config configs/nyu.yaml --mode eval --ckpt model.pth
  Inference:      python cli.py --config configs/nyu.yaml --mode inference --ckpt model.pth --image test.jpg
  Batch inference: python cli.py --config configs/nyu.yaml --mode batch_inference --ckpt model.pth --input_dir images/
  Export ONNX:    python cli.py --config configs/nyu.yaml --mode export_onnx --ckpt model.pth
        """
    )
    
    # General configuration
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--mode', type=str, 
                       choices=['train', 'eval', 'inference', 'batch_inference', 'export_onnx', 'export_torchscript', 'deploy_plan', 'compare_models'],
                       required=True, help='Operation mode')
    
    # Hardware options
    parser.add_argument('--device', type=str, default='auto', help='Device: cuda, cpu, or auto')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of data loader workers')
    
    # Checkpointing and model loading
    parser.add_argument('--ckpt', type=str, default=None, help='Checkpoint path for resume/evaluation/inference')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained backbone weights')
    
    # Inference input/output
    parser.add_argument('--image', type=str, default=None, help='Single image path for inference')
    parser.add_argument('--input_dir', type=str, default=None, help='Input directory for batch inference')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory for results')
    parser.add_argument('--save_raw', action='store_true', help='Save raw depth values')
    parser.add_argument('--save_colored', action='store_true', help='Save colored depth maps')
    parser.add_argument('--save_comparison', action='store_true', help='Save input-output comparison')
    
    # Evaluation configuration
    parser.add_argument('--eval_split', type=str, default='val', choices=['val', 'test'], help='Evaluation split')
    parser.add_argument('--eval_metrics', type=str, nargs='+', default=['all'], help='Metrics to compute')
    parser.add_argument('--save_predictions', action='store_true', help='Save evaluation predictions')
    parser.add_argument('--eval_subset', type=int, default=None, help='Evaluate on subset of data')
    
    # Export options
    parser.add_argument('--onnx_path', type=str, default='model.onnx', help='ONNX export path')
    parser.add_argument('--ts_path', type=str, default='model.ts', help='TorchScript export path')
    parser.add_argument('--opset_version', type=int, default=11, help='ONNX opset version')
    parser.add_argument('--optimize', action='store_true', help='Optimize exported model')
    
    # Logging and verbosity
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--log_file', type=str, default=None, help='Log file path')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--quiet', action='store_true', help='Quiet mode')
    
    # Model comparison
    parser.add_argument('--model_paths', type=str, nargs='+', help='Model paths for comparison')
    parser.add_argument('--comparison_metrics', type=str, nargs='+', default=['rmse', 'mae', 'delta1'])
    
    return parser

def select_device(device_arg: str, gpu_id: int = 0) -> torch.device:
    # Auto-select device based on system capability
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{gpu_id}')
            logger.info(f"Auto-selected GPU: {torch.cuda.get_device_name(device)}")
        else:
            device = torch.device('cpu')
            logger.warning("No GPU found. Falling back to CPU.")
    
    # Explicit CUDA device
    elif device_arg.startswith('cuda'):
        if not torch.cuda.is_available():
            logger.error("CUDA requested but not available. Exiting.")
            sys.exit(1)
        device = torch.device(device_arg)
    
    # Use CPU explicitly
    elif device_arg == 'cpu':
        device = torch.device('cpu')
    
    # Invalid device argument
    else:
        logger.error(f"Invalid device argument: {device_arg}")
        sys.exit(1)

    # Log hardware details
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(device)
        mem = props.total_memory / (1024 ** 3)
        logger.info(f"Using GPU: {props.name} | {mem:.2f} GB VRAM | Compute: {props.major}.{props.minor}")
    else:
        logger.info(f"Using CPU with {psutil.cpu_count(logical=True)} logical cores")

    return device

def create_dataset(config: Dict[str, Any], split: str) -> torch.utils.data.Dataset:
    # Retrieve dataset configuration from config
    dataset_config = config.get("dataset", {})
    dataset_name = dataset_config.get("name", "").lower()
    
    # Validate split value
    if split not in ['train', 'val', 'test']:
        raise ValueError(f"Invalid split: {split}. Expected one of: train, val, test")

    # Add split info to dataset args
    dataset_args = dataset_config.copy()
    dataset_args['split'] = split
    
    # Initialize dataset based on name
    if dataset_name == 'nyu':
        dataset = NYUV2Dataset(**dataset_args)
    elif dataset_name == 'kitti':
        dataset = KITTIDataset(**dataset_args)
    elif dataset_name == 'enrich':
        dataset = ENRICHDataset(**dataset_args)
    elif dataset_name == 'unreal':
        dataset = UnrealStereo4KDataset(**dataset_args)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Log dataset information
    logger.info(f"Loaded {dataset_name.upper()} ({split}) with {len(dataset)} samples")
    
    return dataset

def create_model(config: Dict[str, Any], device: torch.device, checkpoint_path: Optional[str] = None) -> edepth:
    # Extract model-specific configuration from the overall config dictionary
    model_config = config.get('model', {})
    
    # Initialize the model with the given configuration
    model = edepth(**model_config)
    # Move model to the specified device (CPU or GPU)
    model.to(device)

    # If a checkpoint path is provided, load the saved weights
    if checkpoint_path:
        # Check if checkpoint file exists, raise error if not found
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint data mapped to the device
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract the model state dict from checkpoint if it exists
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Load the weights into the model, allowing for missing/unexpected keys
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded checkpoint from: {checkpoint_path}")
        # Log warnings if any keys are missing or unexpected
        if missing_keys:
            logger.warning(f"Missing keys during load: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys during load: {unexpected_keys}")

    # Calculate total and trainable parameter counts for logging
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total Parameters: {total_params:,} | Trainable: {trainable_params:,}")

    # Return the initialized and optionally loaded model
    return model

def train_mode(args: argparse.Namespace, config: Dict[str, Any], device: torch.device) -> None:
    # Set random seed for reproducibility
    torch.manual_seed(config.get('seed', 42))

    # Create training and validation datasets
    train_dataset = create_dataset(config, 'train')
    val_dataset = create_dataset(config, 'val')

    # Create DataLoader for training dataset with shuffling and batching
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=args.num_workers or config.get('data', {}).get('num_workers', 4),
        pin_memory=True,
        drop_last=True
    )

    # Create DataLoader for validation dataset without shuffling
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=args.num_workers or config.get('data', {}).get('num_workers', 4),
        pin_memory=True
    )

    # Initialize the model, loading checkpoint if resuming training
    model = create_model(config, device, args.ckpt if args.resume else None)
    
    # Initialize the Trainer with model, data loaders, config, and device
    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )

    # If resuming from checkpoint, load it into the trainer
    if args.resume and args.ckpt:
        trainer.load_checkpoint(args.ckpt)
        logger.info(f"Resumed training from checkpoint: {args.ckpt}")

    # Start the training process with specified number of epochs
    trainer.train(
        num_epochs=config['training']['epochs'],
        resume_from=args.ckpt if args.resume else None
    )

def eval_mode(args: argparse.Namespace, config: Dict[str, Any], device: torch.device) -> None:
    # Check if checkpoint path is provided and exists, else exit with error
    if args.ckpt is None or not Path(args.ckpt).exists():
        logger.error(f"Checkpoint path not provided or does not exist: {args.ckpt}")
        sys.exit(1)

    # Create the model and load weights from checkpoint
    model = create_model(config, device, args.ckpt)
    # Set model to evaluation mode
    model.eval()

    # Create evaluation dataset for the specified split
    eval_dataset = create_dataset(config, args.eval_split)
    # If evaluation subset size is specified, use a subset of the dataset
    if args.eval_subset:
        eval_dataset = torch.utils.data.Subset(eval_dataset, range(args.eval_subset))

    # Create DataLoader for evaluation dataset without shuffling
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=args.num_workers or config.get('data', {}).get('num_workers', 4),
        pin_memory=True
    )

    # Initialize metrics object with min and max depth thresholds from config
    metrics = Metrics(
        min_depth=config.get('data', {}).get('min_depth', 0.001),
        max_depth=config.get('data', {}).get('max_depth', 80.0)
    )
    
    # Lists to store metrics for all batches and predictions if saved
    all_metrics = []
    predictions = []

    # Disable gradient computation for evaluation
    with torch.no_grad():
        # Iterate over evaluation batches with progress bar
        for batch_idx, batch in enumerate(tqdm(eval_loader, desc="Evaluating")):
            # Handle batch either as dict or tuple
            if isinstance(batch, dict):
                images = batch['image'].to(device)
                gt_depths = batch['depth'].to(device)
                masks = batch.get('mask', None)
                if masks is not None:
                    masks = masks.to(device)
            else:
                images, gt_depths = batch[0].to(device), batch[1].to(device)
                masks = None

            # Run model forward pass to predict depths
            pred_depths = model(images)

            # Compute evaluation metrics for the batch
            batch_metrics = metrics.compute_all_metrics(pred_depths, gt_depths, masks)
            all_metrics.append(batch_metrics)

            # Optionally save predictions to list
            if args.save_predictions:
                predictions.extend(pred_depths.cpu().numpy())

    # Aggregate metrics by averaging across all batches
    final_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        final_metrics[key] = np.mean(values)

    # Print evaluation results summary
    print("\nEvaluation Results:")
    print("=" * 50)
    for k, v in final_metrics.items():
        print(f"{k}: {v:.4f}")

    # Create output directory if it does not exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save evaluation results and config to JSON file
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'config': config,
            'checkpoint': args.ckpt,
            'eval_split': args.eval_split,
            'metrics': final_metrics,
            'num_samples': len(eval_dataset) #type: ignore 
        }, f, indent=2)
    logger.info(f"Saved evaluation report to {results_path}")

    # Save raw predictions if requested
    if args.save_predictions:
        predictions_path = output_dir / "predictions.npy"
        np.save(predictions_path, np.array(predictions))
        logger.info(f"Saved predictions to {predictions_path}")

def inference_mode(args: argparse.Namespace, config: Dict[str, Any], device: torch.device) -> None:
    # Check if checkpoint and image path are provided; exit with error if missing
    if args.ckpt is None or args.image is None:
        logger.error("Both --ckpt and --image are required for inference mode")
        sys.exit(1)

    # Verify that the specified image file exists; exit if not found
    if not os.path.exists(args.image):
        logger.error(f"Image file not found: {args.image}")
        sys.exit(1)

    # Create the model using the given config, device, and checkpoint
    model = create_model(config, device, args.ckpt)
    model.eval()  # Set model to evaluation mode

    # Load and convert the input image to RGB
    raw_img = Image.open(args.image).convert("RGB")
    orig_size = raw_img.size  # Save original image size

    # Extract image processing config parameters
    img_config = config.get('data', {})
    img_size = img_config.get('img_size', (480, 640))
    normalize_mean = img_config.get('normalize_mean', [0.485, 0.456, 0.406])
    normalize_std = img_config.get('normalize_std', [0.229, 0.224, 0.225])

    # Compose preprocessing transforms: resize, convert to tensor, normalize
    preprocess = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])

    # Apply preprocessing and move tensor to device
    input_tensor = preprocess(raw_img).unsqueeze(0).to(device)  # type: ignore

    # Measure inference time and run model prediction without gradient calculation
    start_time = time.time()
    with torch.no_grad():
        pred_depth = model(input_tensor).squeeze(0).squeeze(0).cpu().numpy()
    inference_time = time.time() - start_time

    # Resize predicted depth map back to original image size using cubic interpolation
    pred_depth_resized = cv2.resize(pred_depth, orig_size, interpolation=cv2.INTER_CUBIC)

    # Prepare output directory and image name stem
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_name = Path(args.image).stem

    # Optionally save raw depth map as numpy file
    if args.save_raw:
        np.save(output_dir / f"{image_name}_depth_raw.npy", pred_depth_resized)

    # Optionally save colored depth map visualization using inferno colormap
    if args.save_colored:
        normalized = (pred_depth_resized - pred_depth_resized.min()) / \
                     (pred_depth_resized.max() - pred_depth_resized.min() + 1e-8)
        colored = (255 * plt.cm.inferno(normalized)[:, :, :3]).astype(np.uint8)  # type: ignore
        colored_path = output_dir / f"{image_name}_depth_colored.png"
        cv2.imwrite(str(colored_path), cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))

    # Optionally save side-by-side comparison of original and colored depth image
    if args.save_comparison:
        rgb_img = cv2.cvtColor(np.array(raw_img), cv2.COLOR_RGB2BGR)
        colored_resized = cv2.resize(colored, rgb_img.shape[:2][::-1])
        comparison = np.hstack((rgb_img, colored_resized))
        comparison_path = output_dir / f"{image_name}_comparison.png"
        cv2.imwrite(str(comparison_path), comparison)

    # Print summary of inference results
    print(f"\nInference Results:")
    print(f"Input image       : {args.image}")
    print(f"Inference time    : {inference_time:.3f}s")
    print(f"Depth map shape   : {pred_depth_resized.shape}")
    print(f"Depth value range : [{pred_depth_resized.min():.3f}, {pred_depth_resized.max():.3f}]")
    print(f"Output directory  : {output_dir}")


def batch_inference_mode(args: argparse.Namespace, config: Dict[str, Any], device: torch.device) -> None:
    # Check if checkpoint and input directory are provided; exit with error if missing
    if args.ckpt is None or args.input_dir is None:
        logger.error("Both --ckpt and --input_dir are required for batch inference mode")
        sys.exit(1)

    input_dir = Path(args.input_dir)
    # Verify that input directory exists; exit if not found
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    # Define allowed image file extensions and list image files in directory
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [f for f in input_dir.glob('*') if f.suffix.lower() in image_extensions]

    # Exit if no valid image files found
    if not image_files:
        logger.error(f"No image files found in {input_dir}")
        sys.exit(1)

    # Create model and set to evaluation mode
    model = create_model(config, device, args.ckpt)
    model.eval()

    # Prepare output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract image processing parameters from config
    img_config = config.get('data', {})
    img_size = img_config.get('img_size', (480, 640))
    normalize_mean = img_config.get('normalize_mean', [0.485, 0.456, 0.406])
    normalize_std = img_config.get('normalize_std', [0.229, 0.224, 0.225])

    # Compose preprocessing transforms
    preprocess = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])

    summary = []  # List to collect per-image inference info
    total_time = 0.0  # Accumulate total inference time

    # Process images with progress bar
    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            image_name = image_path.stem
            # Load and convert image to RGB
            raw_img = Image.open(image_path).convert("RGB")
            orig_size = raw_img.size

            # Preprocess and move to device
            input_tensor = preprocess(raw_img).unsqueeze(0).to(device)  # type: ignore

            # Run inference and measure time
            start_time = time.time()
            with torch.no_grad():
                pred_depth = model(input_tensor).squeeze(0).squeeze(0).cpu().numpy()
            inference_time = time.time() - start_time
            total_time += inference_time

            # Resize predicted depth to original image size
            pred_depth_resized = cv2.resize(pred_depth, orig_size, interpolation=cv2.INTER_CUBIC)

            # Save raw depth if requested
            if args.save_raw:
                np.save(output_dir / f"{image_name}_depth_raw.npy", pred_depth_resized)

            # Save colored depth map if requested
            if args.save_colored:
                norm = (pred_depth_resized - pred_depth_resized.min()) / \
                       (pred_depth_resized.max() - pred_depth_resized.min() + 1e-8)
                colored = (255 * plt.cm.inferno(norm)[:, :, :3]).astype(np.uint8)  # type: ignore
                cv2.imwrite(str(output_dir / f"{image_name}_depth_colored.png"),
                            cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))

            # Save side-by-side comparison image if requested
            if args.save_comparison:
                rgb_img = cv2.cvtColor(np.array(raw_img), cv2.COLOR_RGB2BGR)
                colored_resized = cv2.resize(colored, rgb_img.shape[:2][::-1])
                comparison = np.hstack((rgb_img, colored_resized))
                cv2.imwrite(str(output_dir / f"{image_name}_comparison.png"), comparison)

            # Append inference details for summary report
            summary.append({
                "image": str(image_path.name),
                "inference_time_sec": round(inference_time, 4),
                "depth_shape": list(pred_depth_resized.shape),
                "depth_min": round(float(pred_depth_resized.min()), 3),
                "depth_max": round(float(pred_depth_resized.max()), 3)
            })

        except Exception as e:
            # Log any errors encountered while processing an image and continue
            logger.error(f"Error processing {image_path.name}: {e}")
            continue

    # Prepare batch report with timing and image details
    report = {
        "total_images": len(summary),
        "total_time_sec": round(total_time, 3),
        "average_time_per_image_sec": round(total_time / len(summary), 3) if summary else 0,
        "images": summary
    }

    # Save batch report as JSON file in output directory
    with open(output_dir / "batch_report.json", "w") as f:
        json.dump(report, f, indent=4)

    # Print summary of batch inference
    print(f"\nBatch inference complete.")
    print(f"Processed {len(summary)} images.")
    print(f"Total time: {report['total_time_sec']}s | Avg: {report['average_time_per_image_sec']}s")
    print(f"Results saved to: {output_dir}")

def evaluate_model(model, dataloader, device, metrics):
    model.eval()  # Set model to evaluation mode
    all_metrics = []  # List to store metrics for all batches

    with torch.no_grad():  # Disable gradient calculation for inference
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            if isinstance(batch, dict):
                # If batch is a dict, extract images, ground truth depths, and optional masks
                images = batch['image'].to(device)
                gt_depths = batch['depth'].to(device)
                masks = batch.get('mask', None)
                if masks is not None:
                    masks = masks.to(device)
            else:
                # Otherwise assume batch is a tuple/list of (images, depths)
                images, gt_depths = batch[0].to(device), batch[1].to(device)
                masks = None

            preds = model(images)  # Get model predictions
            batch_metrics = metrics.compute_all_metrics(preds, gt_depths, masks)  # Compute metrics for batch
            all_metrics.append(batch_metrics)  # Append batch metrics

    final_metrics = {}
    # Aggregate metrics over all batches by computing mean for each metric key
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        final_metrics[key] = np.mean(values)

    return final_metrics  # Return aggregated metrics

def compare_models_mode(args: argparse.Namespace, config: Dict[str, Any], device: torch.device) -> None:
    # Check if at least two models are provided for comparison
    if not args.model_paths or len(args.model_paths) < 2:
        logger.error("At least 2 model paths are required for comparison")
        sys.exit(1)
    
    models = {}
    # Load each model from provided paths
    for i, model_path in enumerate(args.model_paths):
        model_name = Path(model_path).stem  # Use filename (without extension) as model name
        models[model_name] = create_model(config, device, model_path)

    # Create validation dataset and dataloader
    val_dataset = create_dataset(config, 'val')
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    # Initialize metrics calculator with dataset depth range
    metrics = Metrics(
        min_depth=config.get('data', {}).get('min_depth', 0.001),
        max_depth=config.get('data', {}).get('max_depth', 80.0)
    )

    comparison_results = {}
    # Evaluate each model on validation set and store results
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}")
        model_metrics = evaluate_model(model, val_loader, device, metrics)
        comparison_results[model_name] = model_metrics
        print(f"{model_name} - RMSE: {model_metrics.get('rmse', 'N/A'):.4f} | MAE: {model_metrics.get('mae', 'N/A'):.4f}")

    # Prepare output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save comparison results as JSON report
    report_path = output_dir / "model_comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(comparison_results, f, indent=4)

    print(f"\nComparison report saved to {report_path}")

    # Prepare data for bar plot visualization
    model_names = list(comparison_results.keys())
    rmse_values = [comparison_results[m].get('rmse', 0) for m in model_names]
    mae_values = [comparison_results[m].get('mae', 0) for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots()
    # Plot RMSE bars shifted left
    ax.bar(x - width/2, rmse_values, width, label='RMSE')
    # Plot MAE bars shifted right
    ax.bar(x + width/2, mae_values, width, label='MAE')

    ax.set_ylabel('Error')
    ax.set_title('Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15)
    ax.legend()

    plt.tight_layout()
    fig_path = output_dir / "model_comparison_plot.png"
    plt.savefig(fig_path)
    print(f"Comparison plot saved to {fig_path}")

def export_onnx_mode(args: argparse.Namespace, config: Dict[str, Any], device: torch.device) -> None:
    logger.info("Exporting model to ONNX format...")

    model = create_model(config, device, args.ckpt)
    model.eval()  # Set model to evaluation mode

    input_config = config.get('data', {})
    img_size = input_config.get('img_size', (480, 640))
    # Create dummy input tensor with batch size 1 and specified image size
    dummy_input = torch.randn(1, 3, img_size[0], img_size[1], device=device)

    output_path = Path(args.output_dir) / args.onnx_path

    try:
        # Export model to ONNX format
        torch.onnx.export(
            model, dummy_input, output_path,
            export_params=True,
            opset_version=args.opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        logger.info(f"Model exported to ONNX: {output_path}")
    except Exception as e:
        logger.error(f"Failed to export ONNX: {e}")
        if args.verbose:
            traceback.print_exc()

def export_torchscript_mode(args: argparse.Namespace, config: Dict[str, Any], device: torch.device) -> None:
    logger.info("Exporting model to TorchScript format...")

    model = create_model(config, device, args.ckpt)
    model.eval()  # Set model to evaluation mode

    input_config = config.get('data', {})
    img_size = input_config.get('img_size', (480, 640))
    # Create dummy input tensor for tracing
    dummy_input = torch.randn(1, 3, img_size[0], img_size[1], device=device)

    output_path = Path(args.output_dir) / args.ts_path

    try:
        # Trace model with dummy input to create TorchScript version
        traced = torch.jit.trace(model, dummy_input)  # type: ignore
        traced.save(str(output_path))  # type: ignore
        logger.info(f"Model exported to TorchScript: {output_path}")
    except Exception as e:
        logger.error(f"Failed to export TorchScript: {e}")
        if args.verbose:
            traceback.print_exc()

def print_system_info() -> None:
    print("System Information:")
    print(f"- Python version: {platform.python_version()}")
    print(f"- PyTorch version: {torch.__version__}")
    print(f"- CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"- CUDA version: {torch.version.cuda}")
        print(f"- GPU name: {torch.cuda.get_device_name(0)}")
        # GPU total memory in MB
        print(f"- GPU memory: {torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)} MB")
    mem = psutil.virtual_memory()
    # System RAM in MB
    print(f"- System RAM: {mem.total // (1024 ** 2)} MB")

def main():
    parser = create_argument_parser()  # Initialize argument parser
    args = parser.parse_args()  # Parse CLI arguments

    setup_logging(args.log_level, args.log_file)  # Setup logging configuration

    # Check mutually exclusive flags quiet and verbose
    if args.quiet and args.verbose:
        logger.error("Cannot use both --quiet and --verbose flags.")
        sys.exit(1)

    try:
        config = load_config(args.config)  # Load configuration file
        logger.info(f"Loaded configuration from: {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)

    try:
        device = select_device(args.device, args.gpu_id)  # Select compute device
        logger.info(f"Using device: {device}")
    except Exception as e:
        logger.error(f"Failed to select device: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)

    try:
        # Create output directory if it does not exist
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {args.output_dir}")
    except Exception as e:
        logger.error(f"Failed to create output directory: {e}")
        sys.exit(1)

    try:
        # Map modes to corresponding functions
        mode_dispatch = {
            'train': train_mode,
            'eval': eval_mode,
            'inference': inference_mode,
            'batch_inference': batch_inference_mode,
            'export_onnx': export_onnx_mode,
            'export_torchscript': export_torchscript_mode,
            'deploy_plan': lambda args, config, device: print_deployment_plan(),
            'compare_models': compare_models_mode
        }

        if args.mode not in mode_dispatch:
            logger.error(f"Unknown mode: {args.mode}")
            parser.print_help()
            sys.exit(1)

        logger.info(f"Running in '{args.mode}' mode")
        mode_func = mode_dispatch[args.mode]

        # Run the selected mode function
        if args.mode == 'deploy_plan':
            mode_func(args, config, device)
        else:
            mode_func(args, config, device)

    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred during execution: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
