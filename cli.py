# FILE: cli.py
# ehsanasgharzde - COMPREHENSIVE COMMAND LINE INTERFACE IMPLEMENTATION

import argparse
import logging
import sys
import os
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter #type: ignore 
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
import traceback
import psutil
import platform

from configs.config_loader import load_config
from models.model_fixed import edepth
from training.trainer_fixed import Trainer
from utils.export import export_onnx, export_torchscript, print_deployment_plan #type: ignore 
from datasets.nyu_dataset import NYUV2Dataset
from datasets.kitti_dataset import KITTIDataset
from datasets.enrich_dataset import ENRICHDataset
from datasets.unreal_dataset import UnrealStereo4KDataset
from losses.factory import create_loss
from metrics.metrics_fixed import Metrics
from inference.inference_fixed import run_inference, run_batch_inference #type: ignore  

logger = logging.getLogger(__name__)

def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None) -> None:
    log_dir = Path(log_file).parent if log_file else None
    if log_file and log_dir and not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)

    log_format = "%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
    log_level_enum = getattr(logging, log_level.upper(), logging.INFO)

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=log_level_enum,
        format=log_format,
        handlers=handlers,
        force=True
    )

def create_argument_parser() -> argparse.ArgumentParser:
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
    
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--mode', type=str, 
                       choices=['train', 'eval', 'inference', 'batch_inference', 'export_onnx', 'export_torchscript', 'deploy_plan', 'compare_models'],
                       required=True, help='Operation mode')
    
    parser.add_argument('--device', type=str, default='auto', help='Device: cuda, cpu, or auto')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of data loader workers')
    
    parser.add_argument('--ckpt', type=str, default=None, help='Checkpoint path for resume/evaluation/inference')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained backbone weights')
    
    parser.add_argument('--image', type=str, default=None, help='Single image path for inference')
    parser.add_argument('--input_dir', type=str, default=None, help='Input directory for batch inference')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory for results')
    parser.add_argument('--save_raw', action='store_true', help='Save raw depth values')
    parser.add_argument('--save_colored', action='store_true', help='Save colored depth maps')
    parser.add_argument('--save_comparison', action='store_true', help='Save input-output comparison')
    
    parser.add_argument('--eval_split', type=str, default='val', choices=['val', 'test'], help='Evaluation split')
    parser.add_argument('--eval_metrics', type=str, nargs='+', default=['all'], help='Metrics to compute')
    parser.add_argument('--save_predictions', action='store_true', help='Save evaluation predictions')
    parser.add_argument('--eval_subset', type=int, default=None, help='Evaluate on subset of data')
    
    parser.add_argument('--onnx_path', type=str, default='model.onnx', help='ONNX export path')
    parser.add_argument('--ts_path', type=str, default='model.ts', help='TorchScript export path')
    parser.add_argument('--opset_version', type=int, default=11, help='ONNX opset version')
    parser.add_argument('--optimize', action='store_true', help='Optimize exported model')
    
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--log_file', type=str, default=None, help='Log file path')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--quiet', action='store_true', help='Quiet mode')
    
    parser.add_argument('--model_paths', type=str, nargs='+', help='Model paths for comparison')
    parser.add_argument('--comparison_metrics', type=str, nargs='+', default=['rmse', 'mae', 'delta1'])
    
    return parser

def select_device(device_arg: str, gpu_id: int = 0) -> torch.device:
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{gpu_id}')
            logger.info(f"Auto-selected GPU: {torch.cuda.get_device_name(device)}")
        else:
            device = torch.device('cpu')
            logger.warning("No GPU found. Falling back to CPU.")
    elif device_arg.startswith('cuda'):
        if not torch.cuda.is_available():
            logger.error("CUDA requested but not available. Exiting.")
            sys.exit(1)
        device = torch.device(device_arg)
    elif device_arg == 'cpu':
        device = torch.device('cpu')
    else:
        logger.error(f"Invalid device argument: {device_arg}")
        sys.exit(1)

    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(device)
        mem = props.total_memory / (1024 ** 3)
        logger.info(f"Using GPU: {props.name} | {mem:.2f} GB VRAM | Compute: {props.major}.{props.minor}")
    else:
        logger.info(f"Using CPU with {psutil.cpu_count(logical=True)} logical cores")

    return device

def create_dataset(config: Dict[str, Any], split: str) -> torch.utils.data.Dataset:
    dataset_config = config.get("dataset", {})
    dataset_name = dataset_config.get("name", "").lower()
    
    if split not in ['train', 'val', 'test']:
        raise ValueError(f"Invalid split: {split}. Expected one of: train, val, test")

    dataset_args = dataset_config.copy()
    dataset_args['split'] = split
    
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

    logger.info(f"Loaded {dataset_name.upper()} ({split}) with {len(dataset)} samples")
    return dataset

def create_model(config: Dict[str, Any], device: torch.device, checkpoint_path: Optional[str] = None) -> edepth:
    model_config = config.get('model', {})
    
    model = edepth(**model_config)
    model.to(device)

    if checkpoint_path:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded checkpoint from: {checkpoint_path}")
        if missing_keys:
            logger.warning(f"Missing keys during load: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys during load: {unexpected_keys}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total Parameters: {total_params:,} | Trainable: {trainable_params:,}")

    return model

def train_mode(args: argparse.Namespace, config: Dict[str, Any], device: torch.device) -> None:
    torch.manual_seed(config.get('seed', 42))

    train_dataset = create_dataset(config, 'train')
    val_dataset = create_dataset(config, 'val')

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=args.num_workers or config.get('data', {}).get('num_workers', 4),
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=args.num_workers or config.get('data', {}).get('num_workers', 4),
        pin_memory=True
    )

    model = create_model(config, device, args.ckpt if args.resume else None)
    
    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )

    if args.resume and args.ckpt:
        trainer.load_checkpoint(args.ckpt)
        logger.info(f"Resumed training from checkpoint: {args.ckpt}")

    trainer.train(
        num_epochs=config['training']['epochs'],
        resume_from=args.ckpt if args.resume else None
    )

def eval_mode(args: argparse.Namespace, config: Dict[str, Any], device: torch.device) -> None:
    if args.ckpt is None or not Path(args.ckpt).exists():
        logger.error(f"Checkpoint path not provided or does not exist: {args.ckpt}")
        sys.exit(1)

    model = create_model(config, device, args.ckpt)
    model.eval()

    eval_dataset = create_dataset(config, args.eval_split)
    if args.eval_subset:
        eval_dataset = torch.utils.data.Subset(eval_dataset, range(args.eval_subset))

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=args.num_workers or config.get('data', {}).get('num_workers', 4),
        pin_memory=True
    )

    metrics = Metrics(
        min_depth=config.get('data', {}).get('min_depth', 0.001),
        max_depth=config.get('data', {}).get('max_depth', 80.0)
    )
    
    all_metrics = []
    predictions = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_loader, desc="Evaluating")):
            if isinstance(batch, dict):
                images = batch['image'].to(device)
                gt_depths = batch['depth'].to(device)
                masks = batch.get('mask', None)
                if masks is not None:
                    masks = masks.to(device)
            else:
                images, gt_depths = batch[0].to(device), batch[1].to(device)
                masks = None

            pred_depths = model(images)

            batch_metrics = metrics.compute_all_metrics(pred_depths, gt_depths, masks)
            all_metrics.append(batch_metrics)

            if args.save_predictions:
                predictions.extend(pred_depths.cpu().numpy())

    final_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        final_metrics[key] = np.mean(values)

    print("\nEvaluation Results:")
    print("=" * 50)
    for k, v in final_metrics.items():
        print(f"{k}: {v:.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    if args.save_predictions:
        predictions_path = output_dir / "predictions.npy"
        np.save(predictions_path, np.array(predictions))
        logger.info(f"Saved predictions to {predictions_path}")

def inference_mode(args: argparse.Namespace, config: Dict[str, Any], device: torch.device) -> None:
    if args.ckpt is None or args.image is None:
        logger.error("Both --ckpt and --image are required for inference mode")
        sys.exit(1)

    if not os.path.exists(args.image):
        logger.error(f"Image file not found: {args.image}")
        sys.exit(1)

    model = create_model(config, device, args.ckpt)
    model.eval()

    raw_img = Image.open(args.image).convert("RGB")
    orig_size = raw_img.size

    img_config = config.get('data', {})
    img_size = img_config.get('img_size', (480, 640))
    normalize_mean = img_config.get('normalize_mean', [0.485, 0.456, 0.406])
    normalize_std = img_config.get('normalize_std', [0.229, 0.224, 0.225])

    preprocess = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])

    input_tensor = preprocess(raw_img).unsqueeze(0).to(device) #type: ignore 

    start_time = time.time()
    with torch.no_grad():
        pred_depth = model(input_tensor).squeeze(0).squeeze(0).cpu().numpy()
    inference_time = time.time() - start_time

    pred_depth_resized = cv2.resize(pred_depth, orig_size, interpolation=cv2.INTER_CUBIC)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_name = Path(args.image).stem

    if args.save_raw:
        np.save(output_dir / f"{image_name}_depth_raw.npy", pred_depth_resized)

    if args.save_colored:
        normalized = (pred_depth_resized - pred_depth_resized.min()) / \
                     (pred_depth_resized.max() - pred_depth_resized.min() + 1e-8)
        colored = (255 * plt.cm.inferno(normalized)[:, :, :3]).astype(np.uint8) #type: ignore 
        colored_path = output_dir / f"{image_name}_depth_colored.png"
        cv2.imwrite(str(colored_path), cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))

    if args.save_comparison:
        rgb_img = cv2.cvtColor(np.array(raw_img), cv2.COLOR_RGB2BGR)
        colored_resized = cv2.resize(colored, rgb_img.shape[:2][::-1])
        comparison = np.hstack((rgb_img, colored_resized))
        comparison_path = output_dir / f"{image_name}_comparison.png"
        cv2.imwrite(str(comparison_path), comparison)

    print(f"\nInference Results:")
    print(f"Input image       : {args.image}")
    print(f"Inference time    : {inference_time:.3f}s")
    print(f"Depth map shape   : {pred_depth_resized.shape}")
    print(f"Depth value range : [{pred_depth_resized.min():.3f}, {pred_depth_resized.max():.3f}]")
    print(f"Output directory  : {output_dir}")

def batch_inference_mode(args: argparse.Namespace, config: Dict[str, Any], device: torch.device) -> None:
    if args.ckpt is None or args.input_dir is None:
        logger.error("Both --ckpt and --input_dir are required for batch inference mode")
        sys.exit(1)

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [f for f in input_dir.glob('*') if f.suffix.lower() in image_extensions]

    if not image_files:
        logger.error(f"No image files found in {input_dir}")
        sys.exit(1)

    model = create_model(config, device, args.ckpt)
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_config = config.get('data', {})
    img_size = img_config.get('img_size', (480, 640))
    normalize_mean = img_config.get('normalize_mean', [0.485, 0.456, 0.406])
    normalize_std = img_config.get('normalize_std', [0.229, 0.224, 0.225])

    preprocess = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])

    summary = []
    total_time = 0.0

    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            image_name = image_path.stem
            raw_img = Image.open(image_path).convert("RGB")
            orig_size = raw_img.size

            input_tensor = preprocess(raw_img).unsqueeze(0).to(device) #type: ignore 

            start_time = time.time()
            with torch.no_grad():
                pred_depth = model(input_tensor).squeeze(0).squeeze(0).cpu().numpy()
            inference_time = time.time() - start_time
            total_time += inference_time

            pred_depth_resized = cv2.resize(pred_depth, orig_size, interpolation=cv2.INTER_CUBIC)

            if args.save_raw:
                np.save(output_dir / f"{image_name}_depth_raw.npy", pred_depth_resized)

            if args.save_colored:
                norm = (pred_depth_resized - pred_depth_resized.min()) / \
                       (pred_depth_resized.max() - pred_depth_resized.min() + 1e-8)
                colored = (255 * plt.cm.inferno(norm)[:, :, :3]).astype(np.uint8) #type: ignore     
                cv2.imwrite(str(output_dir / f"{image_name}_depth_colored.png"),
                            cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))

            if args.save_comparison:
                rgb_img = cv2.cvtColor(np.array(raw_img), cv2.COLOR_RGB2BGR)
                colored_resized = cv2.resize(colored, rgb_img.shape[:2][::-1])
                comparison = np.hstack((rgb_img, colored_resized))
                cv2.imwrite(str(output_dir / f"{image_name}_comparison.png"), comparison)

            summary.append({
                "image": str(image_path.name),
                "inference_time_sec": round(inference_time, 4),
                "depth_shape": list(pred_depth_resized.shape),
                "depth_min": round(float(pred_depth_resized.min()), 3),
                "depth_max": round(float(pred_depth_resized.max()), 3)
            })

        except Exception as e:
            logger.error(f"Error processing {image_path.name}: {e}")
            continue

    report = {
        "total_images": len(summary),
        "total_time_sec": round(total_time, 3),
        "average_time_per_image_sec": round(total_time / len(summary), 3) if summary else 0,
        "images": summary
    }

    with open(output_dir / "batch_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print(f"\nBatch inference complete.")
    print(f"Processed {len(summary)} images.")
    print(f"Total time: {report['total_time_sec']}s | Avg: {report['average_time_per_image_sec']}s")
    print(f"Results saved to: {output_dir}")

def evaluate_model(model, dataloader, device, metrics):
    model.eval()
    all_metrics = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            if isinstance(batch, dict):
                images = batch['image'].to(device)
                gt_depths = batch['depth'].to(device)
                masks = batch.get('mask', None)
                if masks is not None:
                    masks = masks.to(device)
            else:
                images, gt_depths = batch[0].to(device), batch[1].to(device)
                masks = None

            preds = model(images)
            batch_metrics = metrics.compute_all_metrics(preds, gt_depths, masks)
            all_metrics.append(batch_metrics)

    final_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        final_metrics[key] = np.mean(values)

    return final_metrics

def compare_models_mode(args: argparse.Namespace, config: Dict[str, Any], device: torch.device) -> None:
    if not args.model_paths or len(args.model_paths) < 2:
        logger.error("At least 2 model paths are required for comparison")
        sys.exit(1)
    
    models = {}
    for i, model_path in enumerate(args.model_paths):
        model_name = Path(model_path).stem
        models[model_name] = create_model(config, device, model_path)

    val_dataset = create_dataset(config, 'val')
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    metrics = Metrics(
        min_depth=config.get('data', {}).get('min_depth', 0.001),
        max_depth=config.get('data', {}).get('max_depth', 80.0)
    )

    comparison_results = {}
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}")
        model_metrics = evaluate_model(model, val_loader, device, metrics)
        comparison_results[model_name] = model_metrics
        print(f"{model_name} - RMSE: {model_metrics.get('rmse', 'N/A'):.4f} | MAE: {model_metrics.get('mae', 'N/A'):.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / "model_comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(comparison_results, f, indent=4)

    print(f"\nComparison report saved to {report_path}")

    model_names = list(comparison_results.keys())
    rmse_values = [comparison_results[m].get('rmse', 0) for m in model_names]
    mae_values = [comparison_results[m].get('mae', 0) for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width/2, rmse_values, width, label='RMSE')
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
    model.eval()

    input_config = config.get('data', {})
    img_size = input_config.get('img_size', (480, 640))
    dummy_input = torch.randn(1, 3, img_size[0], img_size[1], device=device)

    output_path = Path(args.output_dir) / args.onnx_path

    try:
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
    model.eval()

    input_config = config.get('data', {})
    img_size = input_config.get('img_size', (480, 640))
    dummy_input = torch.randn(1, 3, img_size[0], img_size[1], device=device)

    output_path = Path(args.output_dir) / args.ts_path

    try:
        traced = torch.jit.trace(model, dummy_input) #type: ignore 
        traced.save(str(output_path)) #type: ignore     
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
        print(f"- GPU memory: {torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)} MB")
    mem = psutil.virtual_memory()
    print(f"- System RAM: {mem.total // (1024 ** 2)} MB")

def main():
    parser = create_argument_parser()
    args = parser.parse_args()

    setup_logging(args.log_level, args.log_file)

    if args.quiet and args.verbose:
        logger.error("Cannot use both --quiet and --verbose flags.")
        sys.exit(1)

    try:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from: {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)

    try:
        device = select_device(args.device, args.gpu_id)
        logger.info(f"Using device: {device}")
    except Exception as e:
        logger.error(f"Failed to select device: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)

    try:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {args.output_dir}")
    except Exception as e:
        logger.error(f"Failed to create output directory: {e}")
        sys.exit(1)

    try:
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