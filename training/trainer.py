# FILE: training/trainer.py
# ehsanasgharzde - TRAINER CLASS
# hosseinsolymanzadeh - PROPER COMMENTING

import os
import time
import logging
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast #type: ignore 
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from collections import defaultdict
from typing import Dict, Optional, List, Tuple, Any
import numpy as np
from tqdm import tqdm

from losses.factory import create_loss
from metrics.metrics import Metrics
from models.edepth import edepth

logger = logging.getLogger(__name__)

class TrainingState:
    def __init__(self, config: Dict[str, Any]):
        # Initialize training state variables
        self.current_epoch = 0  # Current epoch number
        self.global_step = 0    # Total training steps (if used elsewhere)
        self.best_metric = float('inf')  # Best value of the monitored metric so far
        self.best_epoch = 0  # Epoch at which the best metric was achieved
        self.patience_counter = 0  # Counter for early stopping
        self.train_losses = []  # List of training losses per epoch
        self.val_losses = []  # List of validation losses per epoch
        self.train_metrics = defaultdict(list)  # Dict of training metrics (per name) over epochs
        self.val_metrics = defaultdict(list)  # Dict of validation metrics (per name) over epochs
        self.learning_rates = []  # Learning rate values used per epoch
        self.config = config  # Configuration dictionary

    def update_epoch(self, epoch: int, train_loss: float, val_loss: float, 
                     train_metrics: Dict[str, float], val_metrics: Dict[str, float], 
                     lr: float):
        # Update state variables at the end of an epoch
        self.current_epoch = epoch
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.learning_rates.append(lr)

        # Append metrics to their respective lists
        for key, value in train_metrics.items():
            self.train_metrics[key].append(value)
        for key, value in val_metrics.items():
            self.val_metrics[key].append(value)

        # Determine the monitored metric (default: val_loss)
        metric_name = self.config.get('monitor_metric', 'val_loss')
        current_metric = val_loss if metric_name == 'val_loss' else val_metrics.get(metric_name, float('inf'))

        # Check if the current metric is better than the best so far
        if current_metric < self.best_metric:
            self.best_metric = current_metric
            self.best_epoch = epoch
            self.patience_counter = 0  # Reset patience if improvement
            return True
        else:
            self.patience_counter += 1  # Increment patience if no improvement
            return False

    def should_stop_early(self) -> bool:
        # Check if early stopping criteria are met
        patience = self.config.get('early_stopping_patience', 10)
        return self.patience_counter >= patience

    def get_summary(self) -> Dict[str, Any]:
        # Return a summary of the training process
        return {
            'total_epochs': self.current_epoch,
            'best_epoch': self.best_epoch,
            'best_metric': self.best_metric,
            'final_train_loss': self.train_losses[-1] if self.train_losses else 0.0,
            'final_val_loss': self.val_losses[-1] if self.val_losses else 0.0,
            'patience_counter': self.patience_counter
        }

class Trainer:
    def __init__(self, config: Dict[str, Any], model: nn.Module, 
                 train_loader, val_loader, device: torch.device):
        # Initialize configuration, model, data loaders, and device
        self.config = config
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Create loss function
        self.criterion = create_loss(config['loss'])
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        # Use mixed precision training if enabled
        self.scaler = GradScaler() if config.get('use_amp', True) else None
        # Initialize metrics and training state tracker
        self.metrics = Metrics()
        self.state = TrainingState(config)
        
        # Create directory for saving model checkpoints
        self.save_dir = config.get('save_dir', './checkpoints')
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Setup for distributed training if enabled
        self.distributed = config.get('distributed', False)
        if self.distributed:
            self.model = DDP(self.model, device_ids=[device.index])
            
    def _create_optimizer(self) -> torch.optim.Optimizer:
        # Extract optimizer configuration
        optimizer_config = self.config.get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'AdamW')
        lr = optimizer_config.get('lr', 1e-4)
        weight_decay = optimizer_config.get('weight_decay', 1e-5)
        
        # Separate parameter groups if model has backbone and decoder
        if hasattr(self.model, 'backbone') and hasattr(self.model, 'decoder'):
            backbone_params = list(self.model.backbone.parameters()) #type: ignore 
            decoder_params = list(self.model.decoder.parameters()) #type: ignore 
            param_groups = [
                {'params': backbone_params, 'lr': lr * 0.1},
                {'params': decoder_params, 'lr': lr}
            ]
        else:
            param_groups = self.model.parameters()
            
        # Create optimizer (only AdamW supported here)
        if optimizer_type == 'AdamW':
            return AdamW(param_groups, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
            
    def _create_scheduler(self):
        # Extract scheduler configuration
        scheduler_config = self.config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'CosineAnnealingLR')
        
        # Create learning rate scheduler
        if scheduler_type == 'CosineAnnealingLR':
            T_max = scheduler_config.get('T_max', self.config.get('epochs', 100))
            return CosineAnnealingLR(self.optimizer, T_max=T_max)
        elif scheduler_type == 'ReduceLROnPlateau':
            return ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=0.5)
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_type}")
            
    def train_epoch(self) -> Tuple[float, Dict[str, float]]:
        # Set model to training mode
        self.model.train()
        running_loss = 0.0
        num_batches = len(self.train_loader)
        
        all_preds = []
        all_targets = []
        all_masks = []
        
        # Progress bar for training loop
        pbar = tqdm(self.train_loader, desc=f"Training Epoch {self.state.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move input tensors to the appropriate device
            images = batch['image'].to(self.device, non_blocking=True)
            depths = batch['depth'].to(self.device, non_blocking=True)
            masks = batch.get('mask', torch.ones_like(depths)).to(self.device, non_blocking=True)
            
            # Zero out gradients
            self.optimizer.zero_grad()
            
            if self.scaler:
                # Mixed precision forward and backward pass
                with autocast():
                    predictions = self.model(images)
                    loss = self.criterion(predictions, depths, masks)
                    
                # Scale loss and perform backward pass
                self.scaler.scale(loss).backward() #type: ignore 
                
                # Optional gradient clipping
                if self.config.get('grad_clip', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip']) #type: ignore 
                    
                # Optimizer step and scaler update
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard precision forward and backward pass
                predictions = self.model(images)
                loss = self.criterion(predictions, depths, masks)
                loss.backward()
                
                # Optional gradient clipping
                if self.config.get('grad_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip']) #type: ignore 
                    
                # Optimizer step
                self.optimizer.step()
                
            # Accumulate loss and update global step
            running_loss += loss.item()
            self.state.global_step += 1
            
            # Collect predictions and targets for metric calculation
            all_preds.append(predictions.detach())
            all_targets.append(depths.detach())
            all_masks.append(masks.detach())
            
            # Update progress bar with current loss
            pbar.set_postfix({'loss': loss.item()})
            
        # Compute average loss over epoch
        avg_loss = running_loss / num_batches
        
        # Concatenate all prediction and target tensors
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_masks = torch.cat(all_masks, dim=0)
        
        # Compute evaluation metrics
        train_metrics = self.metrics.compute_all_metrics(all_preds, all_targets, all_masks)
        
        return avg_loss, train_metrics #type: ignore 
        
    def validate_epoch(self) -> Tuple[float, Dict[str, float]]:
        self.model.eval()  # Set the model to evaluation mode
        running_loss = 0.0
        num_batches = len(self.val_loader)  # Total number of validation batches
    
        all_preds = []    # Store all predictions
        all_targets = []  # Store all ground truth depths
        all_masks = []    # Store all masks
    
        with torch.no_grad():  # Disable gradient calculation for validation
            pbar = tqdm(self.val_loader, desc=f"Validation Epoch {self.state.current_epoch + 1}")
    
            for batch in pbar:
                images = batch['image'].to(self.device, non_blocking=True)
                depths = batch['depth'].to(self.device, non_blocking=True)
                masks = batch.get('mask', torch.ones_like(depths)).to(self.device, non_blocking=True)
    
                # Use mixed precision if scaler is enabled
                if self.scaler:
                    with autocast():
                        predictions = self.model(images)
                        loss = self.criterion(predictions, depths, masks)
                else:
                    predictions = self.model(images)
                    loss = self.criterion(predictions, depths, masks)
    
                running_loss += loss.item()  # Accumulate batch loss
    
                all_preds.append(predictions)  # Collect predictions
                all_targets.append(depths)     # Collect ground truths
                all_masks.append(masks)        # Collect masks
    
                pbar.set_postfix({'loss': loss.item()})  # Show current batch loss
    
        avg_loss = running_loss / num_batches  # Compute average loss over all batches
    
        # Concatenate all collected tensors
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_masks = torch.cat(all_masks, dim=0)
    
        # Compute validation metrics
        val_metrics = self.metrics.compute_all_metrics(all_preds, all_targets, all_masks)
    
        return avg_loss, val_metrics  #type: ignore
    
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        # Collect model, optimizer, and training state information
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'training_state': self.state.get_summary(),
            'config': self.config
        }
    
        # Save checkpoint file for the current epoch
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint_data, checkpoint_path)
    
        # If this is the best model so far, save a separate copy
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint_data, best_path)
    
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    
    def load_checkpoint(self, checkpoint_path: str):
        # Ensure checkpoint file exists
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
        # Load checkpoint data
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
    
        # Restore model, optimizer, scheduler, and optionally scaler
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
        # Restore training epoch
        self.state.current_epoch = checkpoint['epoch']
    
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint['epoch']
    
    
    def train(self, num_epochs: int, resume_from: Optional[str] = None):
        start_epoch = 0
    
        # Optionally resume training from a saved checkpoint
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from) + 1
    
        logger.info(f"Starting training from epoch {start_epoch}")
    
        for epoch in range(start_epoch, num_epochs):
            self.state.current_epoch = epoch
    
            # Update epoch in distributed training (e.g., for shuffling)
            if self.distributed and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
    
            # Train for one epoch
            train_loss, train_metrics = self.train_epoch()
    
            # Validate after the epoch
            val_loss, val_metrics = self.validate_epoch()
    
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
    
            # Update internal training state and check if this is the best model
            is_best = self.state.update_epoch(epoch, train_loss, val_loss,
                                              train_metrics, val_metrics, current_lr)
    
            # Step the learning rate scheduler
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
    
            # Save checkpoint for this epoch
            self.save_checkpoint(epoch, is_best)
    
            # Log training and validation stats
            logger.info(f"Epoch {epoch + 1}/{num_epochs} - "
                        f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                        f"LR: {current_lr:.2e}")
    
            # Early stopping check
            if self.state.should_stop_early():
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
            
        # Log training completion info
        logger.info(f"Training completed. Best epoch: {self.state.best_epoch + 1}, "
                    f"Best metric: {self.state.best_metric:.4f}")
    
        return self.state.get_summary()


class DistributedTrainer(Trainer):
    def __init__(self, config: Dict[str, Any], model: nn.Module, 
                 train_loader, val_loader, device: torch.device):
        # Initialize base Trainer class
        super().__init__(config, model, train_loader, val_loader, device)
        
        # Get distributed training environment variables
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.rank = int(os.environ.get('RANK', 0))
        
        # Initialize process group if not already initialized
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
        
        # Wrap model with DistributedDataParallel for multi-GPU training
        self.model = DDP(self.model, device_ids=[self.local_rank])
        
    def _sync_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        # Average metrics across all processes
        synced_metrics = {}
        for key, value in metrics.items():
            tensor = torch.tensor(value, device=self.device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            synced_metrics[key] = tensor.item() / self.world_size
        return synced_metrics
        
    def train_epoch(self) -> Tuple[float, Dict[str, float]]:
        # Perform one training epoch using base Trainer logic
        train_loss, train_metrics = super().train_epoch()
        
        # Average loss across all processes
        loss_tensor = torch.tensor(train_loss, device=self.device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        train_loss = loss_tensor.item() / self.world_size
        
        # Synchronize training metrics
        train_metrics = self._sync_metrics(train_metrics)
        
        return train_loss, train_metrics #type: ignore 
        
    def validate_epoch(self) -> Tuple[float, Dict[str, float]]:
        # Perform one validation epoch using base Trainer logic
        val_loss, val_metrics = super().validate_epoch()
        
        # Average loss across all processes
        loss_tensor = torch.tensor(val_loss, device=self.device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        val_loss = loss_tensor.item() / self.world_size
        
        # Synchronize validation metrics
        val_metrics = self._sync_metrics(val_metrics)
        
        return val_loss, val_metrics #type: ignore 
        
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        # Save checkpoint only on the main process
        if self.rank == 0:
            super().save_checkpoint(epoch, is_best)
            
    def cleanup(self):
        # Clean up the distributed process group if initialized
        if dist.is_initialized():
            dist.destroy_process_group()

def create_trainer(config: Dict[str, Any], model: nn.Module, 
                  train_loader, val_loader, device: torch.device) -> Trainer:
    # Create a distributed or regular trainer based on config
    if config.get('distributed', False):
        return DistributedTrainer(config, model, train_loader, val_loader, device)
    else:
        return Trainer(config, model, train_loader, val_loader, device)

def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    # Ensure required config keys are present
    required_keys = ['loss', 'optimizer', 'scheduler', 'epochs']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # Set default values for optional config entries
    if 'save_dir' not in config:
        config['save_dir'] = './checkpoints'
        
    if 'use_amp' not in config:
        config['use_amp'] = True
        
    if 'grad_clip' not in config:
        config['grad_clip'] = 1.0
        
    if 'early_stopping_patience' not in config:
        config['early_stopping_patience'] = 10
        
    if 'monitor_metric' not in config:
        config['monitor_metric'] = 'val_loss'
        
    return config

def setup_logging(log_dir: str = './logs', level: int = logging.INFO):
    # Create logging directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Define log handlers for both console and file output
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(log_dir, 'training.log'))
    ]
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    return logging.getLogger(__name__)

def main():
    # Define training configuration dictionary
    config = {
        'loss': {
            'type': 'MultiLoss',
            'silog_weight': 1.0,
            'smoothness_weight': 0.1,
            'gradient_weight': 0.1
        },
        'optimizer': {
            'type': 'AdamW',
            'lr': 1e-4,
            'weight_decay': 1e-5
        },
        'scheduler': {
            'type': 'CosineAnnealingLR',
            'T_max': 100
        },
        'epochs': 100,
        'use_amp': True,
        'grad_clip': 1.0,
        'early_stopping_patience': 10,
        'monitor_metric': 'val_loss',
        'save_dir': './checkpoints',
        'distributed': False
    }
    
    # Validate and fill in default config entries
    config = validate_config(config)
    
    # Initialize logging system
    setup_logging()
    
    # Set device to CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = edepth(
        backbone_name='vit_base_patch16_224',
        pretrained=True
    )
    
    # Create trainer based on config
    trainer = create_trainer(config, model, train_loader, val_loader, device) #type: ignore 
    
    # Begin training loop
    training_summary = trainer.train(
        num_epochs=config['epochs'],
        resume_from=None
    )
    
    # Output training summary
    print(f"Training completed: {training_summary}")

if __name__ == "__main__":
    main()
