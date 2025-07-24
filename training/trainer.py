# FILE: training/trainer.py
# ehsanasgharzde - TRAINER CLASS
# hosseinsolymanzadeh - PROPER COMMENTING
# ehsanasgharzde, hosseinsolymanzadeh - FIXED REDUNDANT CODE BY EXTRACTING PURE FUNCTIONS AND BASECLASS LEVEL METHODS

import os
import time
import logging
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast # type: ignore
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from collections import defaultdict
from typing import Dict, Optional, Tuple, Any
import numpy as np
from tqdm import tqdm

from losses.factory import create_loss
from metrics.factory import create_evaluator, get_all_metrics

logger = logging.getLogger(__name__)

class State:
    def __init__(self, config: Dict[str, Any]):
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_metrics = {}
        self.patience_counter = 0
        self.config = config
        self.max_patience = config.get('early_stopping_patience', 10)
        self.min_delta = config.get('early_stopping_min_delta', 1e-4)
        self.train_losses = []
        self.val_losses = []
        self.train_metrics_history = []
        self.val_metrics_history = []
        self.epoch_start_time = None
        self.training_start_time = None
        logger.info(f"Training state initialized with config: {config}")

    def update_epoch(self, epoch: int):
        self.current_epoch = epoch
        self.epoch_start_time = time.time()
        
    def update_step(self, step: int):
        self.global_step = step
        
    def should_early_stop(self, val_loss: float) -> bool:
        if val_loss < (self.best_val_loss - self.min_delta):
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.max_patience
            
    def get_epoch_duration(self) -> float:
        if self.epoch_start_time is None:
            return 0.0
        return time.time() - self.epoch_start_time

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device,
        rank: int = 0,
        world_size: int = 1
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.state = State(config)
        
        self.is_distributed = world_size > 1
        if self.is_distributed:
            self.model = DDP(model, device_ids=[rank], find_unused_parameters=True)
            logger.info(f"Initialized DDP on rank {rank}/{world_size}")
        
        loss_config = config.get('loss', {'name': 'SiLogLoss', 'lambda_var': 0.85})
        self.criterion = create_loss(
            loss_config['name'],
            {k: v for k, v in loss_config.items() if k != 'name'}
        )
        logger.info(f"Loss function initialized: {loss_config}")
        
        self.metrics_evaluator = create_evaluator()
        logger.info(f"Metrics evaluator initialized with: {list(get_all_metrics().keys())}")
        
        optimizer_config = config.get('optimizer', {})
        self.optimizer = AdamW(
            model.parameters(),
            lr=optimizer_config.get('lr', 1e-4),
            weight_decay=optimizer_config.get('weight_decay', 1e-2),
            betas=optimizer_config.get('betas', (0.9, 0.999))
        )
        
        scheduler_config = config.get('scheduler', {'type': 'cosine'})
        if scheduler_config['type'] == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config.get('epochs', 100),
                eta_min=scheduler_config.get('eta_min', 1e-6)
            )
        elif scheduler_config['type'] == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 5),
                verbose=True
            )
        else:
            self.scheduler = None
            
        self.use_amp = config.get('mixed_precision', True)
        self.scaler = GradScaler() if self.use_amp else None
        self.checkpoint_dir = config.get('checkpoint_dir', './checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        logger.info("DepthTrainer initialized successfully")

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        epoch_losses = []
        epoch_metrics = defaultdict(list)
        
        if self.rank == 0:
            pbar = tqdm(
                self.train_loader,
                desc=f"Epoch {self.state.current_epoch + 1} Training",
                leave=False
            )
        else:
            pbar = self.train_loader
            
        for batch_idx, batch in enumerate(pbar):
            if isinstance(batch, dict):
                inputs = batch['image'].to(self.device, non_blocking=True)
                targets = batch['depth'].to(self.device, non_blocking=True)
                masks = batch.get('mask', None)
                if masks is not None:
                    masks = masks.to(self.device, non_blocking=True)
            else:
                inputs, targets = batch
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                masks = None
            
            self.optimizer.zero_grad()
            
            if self.use_amp and self.scaler is not None:
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets, masks)
                
                self.scaler.scale(loss).backward()  # type: ignore
                
                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)  # type: ignore
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets, masks)
                loss.backward()
                
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm) # type: ignore
                
                self.optimizer.step()
            
            with torch.no_grad():
                batch_metrics = self.metrics_evaluator(outputs, targets, masks)
                
            epoch_losses.append(loss.item())
            for metric_name, metric_value in batch_metrics.items():
                if isinstance(metric_value, torch.Tensor):
                    metric_value = metric_value.item()
                epoch_metrics[metric_name].append(metric_value)
            
            self.state.update_step(self.state.global_step + 1)
            
            if self.rank == 0:
                pbar.set_postfix({  # type: ignore
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
        
        avg_loss = np.mean(epoch_losses)
        avg_metrics = {name: np.mean(values) for name, values in epoch_metrics.items()}
        self.state.train_losses.append(avg_loss)
        self.state.train_metrics_history.append(avg_metrics)
        return {'loss': avg_loss, **avg_metrics} # type: ignore

    def validate_epoch(self) -> Dict[str, float]:
        self.model.eval()
        epoch_losses = []
        epoch_metrics = defaultdict(list)
        
        if self.rank == 0:
            pbar = tqdm(
                self.val_loader,
                desc=f"Epoch {self.state.current_epoch + 1} Validation",
                leave=False
            )
        else:
            pbar = self.val_loader
            
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                if isinstance(batch, dict):
                    inputs = batch['image'].to(self.device, non_blocking=True)
                    targets = batch['depth'].to(self.device, non_blocking=True)
                    masks = batch.get('mask', None)
                    if masks is not None:
                        masks = masks.to(self.device, non_blocking=True)
                else:
                    inputs, targets = batch
                    inputs = inputs.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)
                    masks = None
                
                if self.use_amp:
                    with autocast():
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets, masks)
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets, masks)
                
                batch_metrics = self.metrics_evaluator(outputs, targets, masks)
                epoch_losses.append(loss.item())
                for metric_name, metric_value in batch_metrics.items():
                    if isinstance(metric_value, torch.Tensor):
                        metric_value = metric_value.item()
                    epoch_metrics[metric_name].append(metric_value)
                
                if self.rank == 0:
                    pbar.set_postfix({'val_loss': f"{loss.item():.4f}"}) # type: ignore
        
        avg_loss = np.mean(epoch_losses)
        avg_metrics = {name: np.mean(values) for name, values in epoch_metrics.items()}
        self.state.val_losses.append(avg_loss)
        self.state.val_metrics_history.append(avg_metrics)
        return {'loss': avg_loss, **avg_metrics} # type: ignore

    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        if self.rank != 0:
            return
            
        model_state = self.model.module.state_dict() if self.is_distributed else self.model.state_dict() # type: ignore
        
        checkpoint = {
            'epoch': self.state.current_epoch,
            'global_step': self.state.global_step,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_val_loss': self.state.best_val_loss,
            'config': self.config,
            'metrics': metrics,
            'training_state': {
                'train_losses': self.state.train_losses,
                'val_losses': self.state.val_losses,
                'train_metrics_history': self.state.train_metrics_history,
                'val_metrics_history': self.state.val_metrics_history
            }
        }
        
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f"checkpoint_epoch_{self.state.current_epoch:03d}.pth"
        )
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved with validation loss: {metrics['loss']:.6f}")
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            if self.is_distributed:
                self.model.module.load_state_dict(checkpoint['model_state_dict']) # type: ignore
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if self.scheduler and checkpoint.get('scheduler_state_dict'):
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            if self.scaler and checkpoint.get('scaler_state_dict'):
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            self.state.current_epoch = checkpoint['epoch']
            self.state.global_step = checkpoint['global_step']
            self.state.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            
            training_state = checkpoint.get('training_state', {})
            self.state.train_losses = training_state.get('train_losses', [])
            self.state.val_losses = training_state.get('val_losses', [])
            self.state.train_metrics_history = training_state.get('train_metrics_history', [])
            self.state.val_metrics_history = training_state.get('val_metrics_history', [])
            
            logger.info(f"Checkpoint loaded successfully from {checkpoint_path}")
            logger.info(f"Resuming from epoch {self.state.current_epoch}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}")
            return False

    def train(self, resume_from_checkpoint: Optional[str] = None) -> Dict[str, Any]:
        start_epoch = 0
        if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
            if self.load_checkpoint(resume_from_checkpoint):
                start_epoch = self.state.current_epoch + 1
        
        num_epochs = self.config.get('epochs', 100)
        save_freq = self.config.get('save_frequency', 5)
        log_freq = self.config.get('log_frequency', 100)
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Training on device: {self.device}")
        logger.info(f"Distributed training: {self.is_distributed}")
        logger.info(f"Mixed precision: {self.use_amp}")
        
        self.state.training_start_time = time.time() # type: ignore
        
        try:
            for epoch in range(start_epoch, num_epochs):
                self.state.update_epoch(epoch)
                
                if self.is_distributed and hasattr(self.train_loader.sampler, 'set_epoch'):
                    self.train_loader.sampler.set_epoch(epoch) # type: ignore
                
                train_metrics = self.train_epoch()
                val_metrics = self.validate_epoch()
                
                if self.scheduler:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['loss'])
                    else:
                        self.scheduler.step()
                
                is_best = val_metrics['loss'] < self.state.best_val_loss
                if is_best:
                    self.state.best_val_loss = val_metrics['loss']
                    self.state.best_metrics = val_metrics.copy()
                
                if (epoch + 1) % save_freq == 0 or is_best:
                    self.save_checkpoint(val_metrics, is_best)
                
                if self.rank == 0:
                    epoch_duration = self.state.get_epoch_duration()
                    
                    logger.info(
                        f"Epoch {epoch + 1:03d}/{num_epochs:03d} | "
                        f"Duration: {epoch_duration:.2f}s | "
                        f"Train Loss: {train_metrics['loss']:.6f} | "
                        f"Val Loss: {val_metrics['loss']:.6f} | "
                        f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                    )
                    
                    if (epoch + 1) % log_freq == 0:
                        logger.info("Training Metrics:")
                        for name, value in train_metrics.items():
                            if name != 'loss':
                                logger.info(f"  {name}: {value:.6f}")
                        
                        logger.info("Validation Metrics:")
                        for name, value in val_metrics.items():
                            if name != 'loss':
                                logger.info(f"  {name}: {value:.6f}")
                
                if self.state.should_early_stop(val_metrics['loss']):
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            if self.rank == 0:
                emergency_path = os.path.join(self.checkpoint_dir, "emergency_checkpoint.pth")
                self.save_checkpoint(val_metrics if 'val_metrics' in locals() else {})
                logger.info(f"Emergency checkpoint saved: {emergency_path}")
        
        except Exception as e:
            logger.error(f"Training failed with error: {str(e)}")
            raise
        
        total_time = time.time() - self.state.training_start_time # type: ignore
        
        training_summary = {
            'total_epochs': self.state.current_epoch + 1,
            'total_time': total_time,
            'best_val_loss': self.state.best_val_loss,
            'best_metrics': self.state.best_metrics,
            'final_lr': self.optimizer.param_groups[0]['lr'],
            'training_history': {
                'train_losses': self.state.train_losses,
                'val_losses': self.state.val_losses,
                'train_metrics': self.state.train_metrics_history,
                'val_metrics': self.state.val_metrics_history
            }
        }
        
        if self.rank == 0:
            logger.info("="*50)
            logger.info("TRAINING COMPLETED")
            logger.info("="*50)
            logger.info(f"Total training time: {total_time:.2f} seconds")
            logger.info(f"Best validation loss: {self.state.best_val_loss:.6f}")
            logger.info(f"Final learning rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            if self.state.best_metrics:
                logger.info("Best validation metrics:")
                for name, value in self.state.best_metrics.items():
                    logger.info(f"  {name}: {value:.6f}")
        
        return training_summary

def setup_logging(log_dir: str = "./logs") -> None:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"Logging setup complete. Log file: {log_file}")

def setup_distributed_training() -> Tuple[int, int, torch.device]:
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        dist.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)
            device = torch.device(f'cuda:{rank}')
        else:
            device = torch.device('cpu')
        
        logger.info(f"Distributed training initialized: rank {rank}/{world_size}")
    else:
        rank = 0
        world_size = 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Single process training on device: {device}")
    
    return rank, world_size, device

def create_sample_config() -> Dict[str, Any]:
    return {
        'epochs': 100,
        'batch_size': 16,
        'learning_rate': 1e-4,
        'weight_decay': 1e-2,
        'model': {
            'backbone_name': 'vit_base_patch16_224',
            'pretrained': True,
            'use_attention': True,
            'final_activation': 'sigmoid'
        },
        'loss': {
            'name': 'SiLogLoss',
            'lambda_var': 0.85,
            'eps': 1e-7
        },
        'optimizer': {
            'lr': 1e-4,
            'weight_decay': 1e-2,
            'betas': (0.9, 0.999)
        },
        'scheduler': {
            'type': 'cosine',
            'eta_min': 1e-6
        },
        'mixed_precision': True,
        'max_grad_norm': 1.0,
        'save_frequency': 5,
        'log_frequency': 10,
        'early_stopping_patience': 10,
        'early_stopping_min_delta': 1e-4,
        'checkpoint_dir': './checkpoints',
        'log_dir': './logs'
    }
