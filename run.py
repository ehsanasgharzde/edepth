# FILE: cli.py
# ehsanasgharzde - COMPREHENSIVE COMMAND LINE INTERFACE IMPLEMENTATION
# hosseinsolymanzadeh - PROPER COMMENTING

import os
import sys
import time
import json
import yaml
import argparse
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import psutil
import platform
import logging
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
import cv2
from tqdm import tqdm

from models.factory import create_model, create_model_from_checkpoint, get_available_models
from training.trainer import Trainer
from configs.config import TestConfig
from utils.export_model import ModelExporter, ExportFormat, DeploymentPlan # type: ignore
from utils.dataset_tools import BaseDataset
from datasets import create_dataset, DATASET_REGISTRY
from metrics.factory import create_evaluator
from benchmarks.benchmark import ModelArchitectureBenchmark, ExportFormatBenchmark, PrecisionBenchmark
from inference.inference import InferenceEngine, load_image_cv2 # type: ignore
from monitoring.model_integration import ModelMonitoringIntegration
from monitoring.streamlit_dashboard import run_dashboard # type: ignore
from tests.test import ComponentTestDiscovery, TestExecutor, signal_handler  # type: ignore
from logger.logger import setup_logging 

# Setup logger for factory operations
logger = setup_logging(__file__)

class DepthEstimationCLI:
    def __init__(self):
        self.device = None
        self.config = None
        self.model = None
        self.monitoring = None
        self.start_time = time.time()
        
    def setup_logging(self, log_level: str = 'INFO', log_file: Optional[str] = None, quiet: bool = False, verbose: bool = False):
        if quiet:
            level = logging.WARNING
        elif verbose:
            level = logging.DEBUG
        else:
            level = getattr(logging, log_level.upper(), logging.INFO)
            
        handlers = []
        if not quiet:
            handlers.append(logging.StreamHandler(sys.stdout))
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(log_file))
            
        logging.basicConfig(
            level=level,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            handlers=handlers,
            force=True
        )
        
    def setup_device(self, device_arg: str = 'auto', gpu_ids: List[int] = None) -> torch.device: # type: ignore
        if device_arg == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
                logger.info(f"Auto-selected GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = torch.device('cpu')
                logger.warning("No GPU available, using CPU")
        elif device_arg.startswith('cuda'):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            device = torch.device(device_arg)
        else:
            device = torch.device('cpu')
            
        self.device = device
        
        if device.type == 'cuda':
            props = torch.cuda.get_device_properties(device)
            memory_gb = props.total_memory / (1024**3)
            logger.info(f"Using GPU: {props.name}, Memory: {memory_gb:.2f}GB")
        else:
            logger.info(f"Using CPU with {psutil.cpu_count()} cores")
            
        return device
        
    def load_config(self, config_path: str, overrides: Dict[str, Any] = None) -> Dict[str, Any]: # type: ignore
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
                
        if overrides:
            config.update(overrides)
            
        self.config = config
        logger.info(f"Loaded configuration from {config_path}")
        return config
        
    def create_model(self, config: Dict[str, Any], checkpoint_path: Optional[str] = None) -> nn.Module:
        model_config = config.get('model', {})
        
        if checkpoint_path and Path(checkpoint_path).exists():
            model = create_model_from_checkpoint(checkpoint_path, device=self.device) # type: ignore
            logger.info(f"Loaded model from checkpoint: {checkpoint_path}")
        else:
            model = create_model(
                model_name=model_config.get('name', 'edepth'),
                **model_config
            )
            model.to(self.device)
            
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        self.model = model
        return model
        
    def create_dataset(self, config: Dict[str, Any], split: str = 'train') -> BaseDataset:
        dataset_config = config.get('dataset', {})
        dataset_name = dataset_config.get('name', '').lower()
        
        dataset_args = dataset_config.copy()
        dataset_args['split'] = split
        
        if dataset_name in DATASET_REGISTRY:
            dataset = DATASET_REGISTRY[dataset_name](**dataset_args)
        else:
            dataset = create_dataset(dataset_name, **dataset_args)
            
        logger.info(f"Created {dataset_name} dataset ({split}): {len(dataset)} samples")
        return dataset
        
    def setup_distributed(self, rank: int = 0, world_size: int = 1, backend: str = 'nccl'):
        if world_size > 1:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            dist.init_process_group(backend, rank=rank, world_size=world_size)
            torch.cuda.set_device(rank)
            logger.info(f"Initialized distributed training: rank {rank}/{world_size}")
            
    def train(self, args):
        config = self.load_config(args.config, args.config_overrides)
        device = self.setup_device(args.device, args.gpu_ids)
        
        if args.distributed:
            self.setup_distributed(args.rank, args.world_size)
            
        train_dataset = self.create_dataset(config, 'train')
        val_dataset = self.create_dataset(config, 'val')
        
        if args.distributed:
            train_sampler = DistributedSampler(train_dataset)
            val_sampler = DistributedSampler(val_dataset)
        else:
            train_sampler = None
            val_sampler = None
            
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        model = self.create_model(config, args.checkpoint if args.resume else None)
        
        if args.distributed:
            model = DDP(model, device_ids=[args.rank])
            
        if args.monitoring:
            self.monitoring = ModelMonitoringIntegration()
            model = self.monitoring.wrap_model(model) # type: ignore
            
        trainer = Trainer(
            config=config,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            use_amp=args.mixed_precision, # type: ignore
            output_dir=args.output_dir # type: ignore
        )
        
        if args.resume and args.checkpoint:
            trainer.load_checkpoint(args.checkpoint)
            
        trainer.train(
            num_epochs=config['training']['epochs'], # type: ignore
            save_every=args.save_every, # type: ignore
            validate_every=args.validate_every # type: ignore
        )
        
        logger.info(f"Training completed. Output saved to {args.output_dir}")
        
    def evaluate(self, args):
        config = self.load_config(args.config)
        device = self.setup_device(args.device)
        
        if not args.checkpoint or not Path(args.checkpoint).exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
            
        model = self.create_model(config, args.checkpoint)
        model.eval()
        
        eval_dataset = self.create_dataset(config, args.split)
        
        if args.subset:
            indices = np.random.choice(len(eval_dataset), args.subset, replace=False)
            eval_dataset = torch.utils.data.Subset(eval_dataset, indices) # type: ignore
            
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        evaluator = create_evaluator(args.metrics)
        
        results = []
        predictions = [] if args.save_predictions else None
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(eval_loader, desc="Evaluating")):
                if isinstance(batch, dict):
                    images = batch['image'].to(device)
                    gt_depths = batch['depth'].to(device)
                    masks = batch.get('mask', None)
                else:
                    images, gt_depths = batch[0].to(device), batch[1].to(device)
                    masks = None
                    
                if masks is not None:
                    masks = masks.to(device)
                    
                pred_depths = model(images)
                
                batch_results = evaluator(pred_depths, gt_depths, masks)
                results.append(batch_results)
                
                if predictions is not None:
                    predictions.extend(pred_depths.cpu().numpy())
                    
        final_results = {}
        for key in results[0].keys():
            values = [r[key] for r in results]
            final_results[key] = np.mean(values)
            
        print("\nEvaluation Results:")
        print("=" * 50)
        for metric, value in final_results.items():
            print(f"{metric}: {value:.4f}")
            
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_path = output_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                'config': config,
                'checkpoint': args.checkpoint,
                'split': args.split,
                'metrics': final_results,
                'num_samples': len(eval_dataset)
            }, f, indent=2)
            
        if predictions:
            np.save(output_dir / "predictions.npy", np.array(predictions))
            
        logger.info(f"Evaluation completed. Results saved to {output_dir}")
        
    def inference(self, args):
        config = self.load_config(args.config)
        device = self.setup_device(args.device)
        
        model = self.create_model(config, args.checkpoint)
        model.eval()
        
        if args.image:
            self._single_inference(model, config, args)
        elif args.video:
            self._video_inference(model, config, args)
        elif args.camera:
            self._camera_inference(model, config, args)
        else:
            raise ValueError("Must specify --image, --video, or --camera")
            
    def _single_inference(self, model, config, args):
        if not Path(args.image).exists():
            raise FileNotFoundError(f"Image not found: {args.image}")
            
        engine = InferenceEngine(model, config, self.device)
        
        image = load_image_cv2(args.image)
        
        start_time = time.time()
        depth_map = engine.predict(image)
        inference_time = time.time() - start_time
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_name = Path(args.image).stem
        
        if args.save_raw:
            np.save(output_dir / f"{image_name}_depth.npy", depth_map)
            
        if args.save_colored:
            colored = engine.colorize_depth(depth_map)
            cv2.imwrite(str(output_dir / f"{image_name}_depth_colored.png"), colored)
            
        if args.save_comparison:
            comparison = engine.create_comparison(image, depth_map)
            cv2.imwrite(str(output_dir / f"{image_name}_comparison.png"), comparison)
            
        print(f"Inference completed in {inference_time:.3f}s")
        print(f"Depth range: [{depth_map.min():.3f}, {depth_map.max():.3f}]")
        print(f"Results saved to {output_dir}")
        
    def _video_inference(self, model, config, args):
        if not Path(args.video).exists():
            raise FileNotFoundError(f"Video not found: {args.video}")
            
        engine = InferenceEngine(model, config, self.device)
        
        cap = cv2.VideoCapture(str(args.video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        video_name = Path(args.video).stem
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
        out = cv2.VideoWriter(
            str(output_dir / f"{video_name}_depth.mp4"),
            fourcc, fps, (width, height)
        )
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            depth_map = engine.predict(frame)
            colored_depth = engine.colorize_depth(depth_map)
            
            out.write(colored_depth)
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames")
                
        cap.release()
        out.release()
        
        print(f"Video inference completed. Processed {frame_count} frames")
        print(f"Output saved to {output_dir}")
        
    def _camera_inference(self, model, config, args):
        engine = InferenceEngine(model, config, self.device)
        
        cap = cv2.VideoCapture(args.camera_id)
        
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {args.camera_id}")
            
        print("Press 'q' to quit, 's' to save current frame")
        
        frame_count = 0
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            depth_map = engine.predict(frame)
            colored_depth = engine.colorize_depth(depth_map)
            
            display_frame = np.hstack([frame, colored_depth])
            cv2.imshow('Depth Estimation', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(str(output_dir / f"frame_{timestamp}.png"), frame)
                cv2.imwrite(str(output_dir / f"depth_{timestamp}.png"), colored_depth)
                print(f"Saved frame {frame_count}")
                
            frame_count += 1
            
        cap.release()
        cv2.destroyAllWindows()
        
    def batch_inference(self, args):
        config = self.load_config(args.config)
        device = self.setup_device(args.device)
        
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
            
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [f for f in input_dir.rglob('*') if f.suffix.lower() in image_extensions]
        
        if not image_files:
            raise ValueError(f"No images found in {input_dir}")
            
        model = self.create_model(config, args.checkpoint)
        engine = InferenceEngine(model, config, device)
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        def process_image(image_path):
            try:
                image = load_image_cv2(image_path)
                depth_map = engine.predict(image)
                
                relative_path = image_path.relative_to(input_dir)
                output_path = output_dir / relative_path.parent / relative_path.stem
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                results = {'image': str(relative_path), 'success': True}
                
                if args.save_raw:
                    np.save(str(output_path) + '_depth.npy', depth_map)
                    
                if args.save_colored:
                    colored = engine.colorize_depth(depth_map)
                    cv2.imwrite(str(output_path) + '_depth.png', colored)
                    
                if args.save_comparison:
                    comparison = engine.create_comparison(image, depth_map)
                    cv2.imwrite(str(output_path) + '_comparison.png', comparison)
                    
                results.update({
                    'depth_min': float(depth_map.min()),
                    'depth_max': float(depth_map.max()),
                    'depth_mean': float(depth_map.mean())
                })
                
                return results
                
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                return {'image': str(image_path), 'success': False, 'error': str(e)}
                
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            results = list(tqdm(
                executor.map(process_image, image_files),
                total=len(image_files),
                desc="Processing images"
            ))
            
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        
        report = {
            'total_images': len(image_files),
            'successful': successful,
            'failed': failed,
            'results': results
        }
        
        with open(output_dir / 'batch_report.json', 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"Batch inference completed: {successful}/{len(image_files)} successful")
        print(f"Results saved to {output_dir}")
        
    def export_model(self, args):
        config = self.load_config(args.config)
        device = self.setup_device(args.device)
        
        model = self.create_model(config, args.checkpoint)
        model.eval()
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exporter = ModelExporter(model, config, device) # type: ignore
        
        formats = args.formats if args.formats else ['onnx', 'torchscript']
        
        for format_name in formats:
            try:
                format_enum = ExportFormat[format_name.upper()]
                export_path = output_dir / f"model.{format_name.lower()}"
                
                exporter.export(
                    format_enum,
                    str(export_path), # type: ignore
                    optimize=args.optimize,
                    quantize=args.quantize
                )
                
                logger.info(f"Successfully exported to {export_path}")
                
            except Exception as e:
                logger.error(f"Failed to export to {format_name}: {e}")
                if args.verbose:
                    traceback.print_exc()
                    
    def deploy_plan(self, args):
        config = self.load_config(args.config)
        
        planner = DeploymentPlan(config)
        plan = planner.generate_plan(
            target_platform=args.platform,
            performance_requirements=args.performance,
            constraints=args.constraints
        )
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'deployment_plan.json', 'w') as f:
            json.dump(plan, f, indent=2)
            
        planner.print_plan(plan)
        
    def benchmark(self, args):
        config = self.load_config(args.config)
        device = self.setup_device(args.device)
        
        if args.checkpoint:
            model = self.create_model(config, args.checkpoint)
        else:
            model = self.create_model(config)
            
        benchmarks = []
        
        if 'architecture' in args.benchmark_types:
            arch_benchmark = ModelArchitectureBenchmark(model, config, device) # type: ignore
            results = arch_benchmark.run() # type: ignore
            benchmarks.append(('architecture', results))
            
        if 'export' in args.benchmark_types:
            export_benchmark = ExportFormatBenchmark(model, config, device) # type: ignore
            results = export_benchmark.run() # type: ignore
            benchmarks.append(('export', results))
            
        if 'precision' in args.benchmark_types:
            precision_benchmark = PrecisionBenchmark(model, config, device) # type: ignore
            results = precision_benchmark.run() # type: ignore
            benchmarks.append(('precision', results))
            
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for benchmark_type, results in benchmarks:
            with open(output_dir / f'{benchmark_type}_benchmark.json', 'w') as f:
                json.dump(results, f, indent=2)
                
        print(f"Benchmarks completed. Results saved to {output_dir}")
        
    def compare_models(self, args):
        config = self.load_config(args.config)
        device = self.setup_device(args.device)
        
        models = {}
        for checkpoint in args.checkpoints:
            name = Path(checkpoint).stem
            models[name] = self.create_model(config, checkpoint)
            
        val_dataset = self.create_dataset(config, 'val')
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
        
        evaluator = create_evaluator(args.metrics)
        
        comparison_results = {}
        for name, model in models.items():
            model.eval()
            results = []
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Evaluating {name}"):
                    if isinstance(batch, dict):
                        images = batch['image'].to(device)
                        gt_depths = batch['depth'].to(device)
                        masks = batch.get('mask', None)
                    else:
                        images, gt_depths = batch[0].to(device), batch[1].to(device)
                        masks = None
                        
                    if masks is not None:
                        masks = masks.to(device)
                        
                    pred_depths = model(images)
                    batch_results = evaluator(pred_depths, gt_depths, masks)
                    results.append(batch_results)
                    
            final_results = {}
            for key in results[0].keys():
                values = [r[key] for r in results]
                final_results[key] = np.mean(values)
                
            comparison_results[name] = final_results
            
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'model_comparison.json', 'w') as f:
            json.dump(comparison_results, f, indent=2)
            
        print("\nModel Comparison Results:")
        print("=" * 80)
        for model_name, results in comparison_results.items():
            print(f"\n{model_name}:")
            for metric, value in results.items():
                print(f"  {metric}: {value:.4f}")
                
    def monitor(self, args):
        if args.dashboard:
            run_dashboard(port=args.port, host=args.host)
        else:
            config = self.load_config(args.config)
            monitoring = ModelMonitoringIntegration(config) # type: ignore
            monitoring.start_monitoring()
            
            try:
                while True:
                    time.sleep(10)
                    stats = monitoring.get_current_stats() # type: ignore
                    print(f"CPU: {stats['cpu']:.1f}%, GPU: {stats['gpu']:.1f}%, Memory: {stats['memory']:.1f}%")
            except KeyboardInterrupt:
                monitoring.stop_monitoring()
                
    def test(self, args):
        start_time = time.time()
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
    
        include_patterns = ["test_*.py"]
        exclude_patterns = ["*_helper.py", "*_utilities.py"]
        component_dirs = ["models", "datasets", "training", "utils"]
    
        from tests.test import discover_pytest_tests, run_pytest_tests, print_summary  # Import required functions
        
        test_files = discover_pytest_tests(args.test_dir, include_patterns, exclude_patterns, component_dirs)
        
        if args.test_file:
            test_files = [f for f in test_files if args.test_file in str(f)]
    
        if args.component:
            test_files = [f for f in test_files if args.component in str(f)]
    
        test_results = run_pytest_tests(
            test_files, 
            output_dir, 
            parallel=args.workers, 
            timeout=args.timeout, 
            coverage=not args.no_coverage
        )
        
        print_summary(test_results, start_time)
        
        failed_tests = sum(1 for group in test_results for test in group.get('tests', []) if test['status'] in ["failed", "error"])
        
        if failed_tests > 0:
            logger.error(f"Test execution completed with {failed_tests} failures")
            return 1
        else:
            logger.info("All tests passed successfully")
            return 0
                    
    def config_ops(self, args):
        if args.config_command == 'validate':
            try:
                config = self.load_config(args.config)
                config_manager.validate_config(config) # type: ignore
                print("Configuration is valid")
            except Exception as e:
                print(f"Configuration validation failed: {e}")
                return 1
                
        elif args.config_command == 'create':
            template = config_manager.create_template( # type: ignore
                dataset=args.dataset,
                model=args.model,
                task=args.task
            )
            
            output_path = args.output or f"{args.dataset}_{args.model}_config.yaml"
            with open(output_path, 'w') as f:
                yaml.dump(template, f, default_flow_style=False, indent=2)
            print(f"Configuration template created: {output_path}")
            
        elif args.config_command == 'merge':
            base_config = self.load_config(args.base_config)
            override_config = self.load_config(args.override_config)
            
            merged = config_manager.merge_configs(base_config, override_config) # type: ignore
            
            output_path = args.output or "merged_config.yaml"
            with open(output_path, 'w') as f:
                yaml.dump(merged, f, default_flow_style=False, indent=2)
            print(f"Merged configuration saved: {output_path}")
            
    def dataset_ops(self, args):
        if args.dataset_command == 'info':
            for name, cls in DATASET_REGISTRY.items():
                print(f"{name}: {cls.__doc__ or 'No description'}")
                
        elif args.dataset_command == 'validate':
            config = self.load_config(args.config)
            try:
                dataset = self.create_dataset(config, args.split)
                print(f"Dataset is valid: {len(dataset)} samples")
                
                if args.sample_check:
                    sample = dataset[0]
                    print(f"Sample keys: {list(sample.keys()) if isinstance(sample, dict) else 'tuple'}")
                    
            except Exception as e:
                print(f"Dataset validation failed: {e}")
                return 1
                
        elif args.dataset_command == 'stats':
            config = self.load_config(args.config)
            dataset = self.create_dataset(config, args.split)
            
            print(f"Dataset: {config['dataset']['name']}")
            print(f"Split: {args.split}")
            print(f"Samples: {len(dataset)}")
            
            if hasattr(dataset, 'get_stats'):
                stats = dataset.get_stats() # type: ignore
                for key, value in stats.items():
                    print(f"{key}: {value}")
                    
    def model_ops(self, args):
        if args.model_command == 'list':
            models = get_available_models()
            print("Available models:")
            for model in models:
                print(f"  - {model}")
                
        elif args.model_command == 'info':
            config = self.load_config(args.config) if args.config else {}
            device = self.setup_device(args.device)
            
            model = self.create_model(config, args.checkpoint)
            
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"Model: {model.__class__.__name__}")
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
            
            if args.checkpoint:
                checkpoint = torch.load(args.checkpoint, map_location='cpu')
                if 'epoch' in checkpoint:
                    print(f"Checkpoint epoch: {checkpoint['epoch']}")
                if 'best_metric' in checkpoint:
                    print(f"Best metric: {checkpoint['best_metric']:.4f}")
                    
    def system_info(self, args):
        print("System Information:")
        print(f"Python: {platform.python_version()}")
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                print(f"  GPU {i}: {props.name} ({memory_gb:.1f}GB)")
                
        memory = psutil.virtual_memory()
        print(f"System Memory: {memory.total / (1024**3):.1f}GB")
        print(f"Available Memory: {memory.available / (1024**3):.1f}GB")
        print(f"CPU Cores: {psutil.cpu_count()}")
        
    def profile_model(self, args):
        config = self.load_config(args.config)
        device = self.setup_device(args.device)
        
        model = self.create_model(config, args.checkpoint)
        model.eval()
        
        input_shape = config.get('data', {}).get('img_size', (480, 640))
        dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
        
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            for _ in range(10):
                with torch.no_grad():
                    _ = model(dummy_input)
                    
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        prof.export_chrome_trace(str(output_dir / "profile_trace.json"))
        
        print("Model Profiling Results:")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        
        with open(output_dir / "profile_summary.txt", "w") as f:
            f.write(prof.key_averages().table(sort_by="cuda_time_total"))
            
        print(f"Profile results saved to {output_dir}")
        
    def create_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Comprehensive Depth Estimation CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        parser.add_argument('--config', type=str, help='Configuration file path')
        parser.add_argument('--device', type=str, default='auto', help='Device: cuda, cpu, or auto')
        parser.add_argument('--gpu-ids', type=int, nargs='+', help='GPU IDs to use')
        parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
        parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
        parser.add_argument('--log-file', type=str, help='Log file path')
        parser.add_argument('--quiet', action='store_true', help='Quiet mode')
        parser.add_argument('--verbose', action='store_true', help='Verbose mode')
        parser.add_argument('--num-workers', type=int, default=4, help='Number of data loader workers')
        
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        train_parser = subparsers.add_parser('train', help='Train a model')
        train_parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
        train_parser.add_argument('--checkpoint', type=str, help='Checkpoint path')
        train_parser.add_argument('--distributed', action='store_true', help='Use distributed training')
        train_parser.add_argument('--rank', type=int, default=0, help='Process rank')
        train_parser.add_argument('--world-size', type=int, default=1, help='World size')
        train_parser.add_argument('--mixed-precision', action='store_true', help='Use mixed precision')
        train_parser.add_argument('--monitoring', action='store_true', help='Enable monitoring')
        train_parser.add_argument('--save-every', type=int, default=1, help='Save checkpoint every N epochs')
        train_parser.add_argument('--validate-every', type=int, default=1, help='Validate every N epochs')
        train_parser.add_argument('--config-overrides', type=dict, help='Configuration overrides') # type: ignore
        
        eval_parser = subparsers.add_parser('eval', help='Evaluate a model')
        eval_parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint path')
        eval_parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'])
        eval_parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
        eval_parser.add_argument('--metrics', type=str, nargs='+', default=['all'], help='Metrics to compute')
        eval_parser.add_argument('--save-predictions', action='store_true', help='Save predictions')
        eval_parser.add_argument('--subset', type=int, help='Evaluate on subset')
        
        infer_parser = subparsers.add_parser('inference', help='Run inference')
        infer_parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint path')
        infer_parser.add_argument('--image', type=str, help='Single image path')
        infer_parser.add_argument('--video', type=str, help='Video file path')
        infer_parser.add_argument('--camera', action='store_true', help='Use camera input')
        infer_parser.add_argument('--camera-id', type=int, default=0, help='Camera ID')
        infer_parser.add_argument('--save-raw', action='store_true', help='Save raw depth')
        infer_parser.add_argument('--save-colored', action='store_true', help='Save colored depth')
        infer_parser.add_argument('--save-comparison', action='store_true', help='Save comparison')
        
        batch_parser = subparsers.add_parser('batch-inference', help='Batch inference')
        batch_parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint path')
        batch_parser.add_argument('--input-dir', type=str, required=True, help='Input directory')
        batch_parser.add_argument('--workers', type=int, default=4, help='Number of workers')
        batch_parser.add_argument('--save-raw', action='store_true', help='Save raw depth')
        batch_parser.add_argument('--save-colored', action='store_true', help='Save colored depth')
        batch_parser.add_argument('--save-comparison', action='store_true', help='Save comparison')
        
        export_parser = subparsers.add_parser('export', help='Export model')
        export_parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint path')
        export_parser.add_argument('--formats', type=str, nargs='+', 
                                 choices=['onnx', 'torchscript', 'tensorrt', 'openvino'],
                                 help='Export formats')
        export_parser.add_argument('--optimize', action='store_true', help='Optimize exported model')
        export_parser.add_argument('--quantize', action='store_true', help='Quantize model')
        
        deploy_parser = subparsers.add_parser('deploy-plan', help='Generate deployment plan')
        deploy_parser.add_argument('--platform', type=str, help='Target platform')
        deploy_parser.add_argument('--performance', type=dict, help='Performance requirements') # type: ignore
        deploy_parser.add_argument('--constraints', type=dict, help='Deployment constraints') # type: ignore
        
        benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark model')
        benchmark_parser.add_argument('--checkpoint', type=str, help='Checkpoint path')
        benchmark_parser.add_argument('--benchmark-types', type=str, nargs='+',
                                    choices=['architecture', 'export', 'precision'],
                                    default=['architecture'], help='Benchmark types')
        
        compare_parser = subparsers.add_parser('compare', help='Compare models')
        compare_parser.add_argument('--checkpoints', type=str, nargs='+', required=True, help='Checkpoint paths')
        compare_parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
        compare_parser.add_argument('--metrics', type=str, nargs='+', default=['rmse', 'mae'], help='Metrics')
        
        monitor_parser = subparsers.add_parser('monitor', help='Monitor system')
        monitor_parser.add_argument('--dashboard', action='store_true', help='Launch dashboard')
        monitor_parser.add_argument('--port', type=int, default=8501, help='Dashboard port')
        monitor_parser.add_argument('--host', type=str, default='localhost', help='Dashboard host')
        
        test_parser = subparsers.add_parser('test', help='Run tests')
        test_parser.add_argument('--test-dir', type=str, default='tests', help='Test directory')
        test_parser.add_argument('--output-dir', type=str, default='test_results', help='Output directory for test results')
        test_parser.add_argument('--workers', type=int, default=4, help='Number of workers')
        test_parser.add_argument('--timeout', type=int, default=300, help='Test timeout')
        test_parser.add_argument('--coverage', action='store_true', help='Enable coverage')
        test_parser.add_argument('--fail-fast', action='store_true', help='Stop on first failure')
        test_parser.add_argument('--test-file', type=str, help='Run only a specific test file')
        test_parser.add_argument('--component', type=str, help='Run only a specific component')

        config_parser = subparsers.add_parser('config', help='Configuration operations')
        config_subparsers = config_parser.add_subparsers(dest='config_command')
        
        validate_parser = config_subparsers.add_parser('validate', help='Validate configuration')
        
        create_parser = config_subparsers.add_parser('create', help='Create configuration template')
        create_parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
        create_parser.add_argument('--model', type=str, required=True, help='Model name')
        create_parser.add_argument('--task', type=str, default='depth_estimation', help='Task type')
        create_parser.add_argument('--output', type=str, help='Output path')
        
        merge_parser = config_subparsers.add_parser('merge', help='Merge configurations')
        merge_parser.add_argument('--base-config', type=str, required=True, help='Base configuration')
        merge_parser.add_argument('--override-config', type=str, required=True, help='Override configuration')
        merge_parser.add_argument('--output', type=str, help='Output path')
        
        dataset_parser = subparsers.add_parser('dataset', help='Dataset operations')
        dataset_subparsers = dataset_parser.add_subparsers(dest='dataset_command')
        
        dataset_subparsers.add_parser('info', help='List available datasets')
        
        dataset_validate_parser = dataset_subparsers.add_parser('validate', help='Validate dataset')
        dataset_validate_parser.add_argument('--split', type=str, default='train', help='Dataset split')
        dataset_validate_parser.add_argument('--sample-check', action='store_true', help='Check sample format')
        
        dataset_stats_parser = dataset_subparsers.add_parser('stats', help='Dataset statistics')
        dataset_stats_parser.add_argument('--split', type=str, default='train', help='Dataset split')
        
        model_parser = subparsers.add_parser('model', help='Model operations')
        model_subparsers = model_parser.add_subparsers(dest='model_command')
        
        model_subparsers.add_parser('list', help='List available models')
        
        model_info_parser = model_subparsers.add_parser('info', help='Model information')
        model_info_parser.add_argument('--checkpoint', type=str, help='Checkpoint path')
        
        subparsers.add_parser('system-info', help='System information')
        
        profile_parser = subparsers.add_parser('profile', help='Profile model')
        profile_parser.add_argument('--checkpoint', type=str, help='Checkpoint path')
        
        return parser
        
    def run(self):
        parser = self.create_parser()
        args = parser.parse_args()
        
        if not args.command:
            parser.print_help()
            return 1
            
        self.setup_logging(args.log_level, args.log_file, args.quiet, args.verbose)
        
        try:
            command_map = {
                'train': self.train,
                'eval': self.evaluate,
                'inference': self.inference,
                'batch-inference': self.batch_inference,
                'export': self.export_model,
                'deploy-plan': self.deploy_plan,
                'benchmark': self.benchmark,
                'compare': self.compare_models,
                'monitor': self.monitor,
                'test': self.test,
                'config': self.config_ops,
                'dataset': self.dataset_ops,
                'model': self.model_ops,
                'system-info': self.system_info,
                'profile': self.profile_model
            }
            
            if args.command in command_map:
                return command_map[args.command](args)
            else:
                logger.error(f"Unknown command: {args.command}")
                return 1
                
        except KeyboardInterrupt:
            logger.warning("Interrupted by user")
            return 130
        except Exception as e:
            logger.error(f"Error: {e}")
            if args.verbose:
                traceback.print_exc()
            return 1
            
def main():
    cli = DepthEstimationCLI()
    return cli.run()

if __name__ == '__main__':
    sys.exit(main())