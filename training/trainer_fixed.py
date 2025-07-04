import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
from pathlib import Path
import logging
from typing import Any, Dict
import traceback

class Trainer:
    def __init__(self, model: nn.Module, loss_fn: nn.Module, optimizer: optim.Optimizer, scheduler: Any = None, scaler: GradScaler = None, device: str = 'cuda', checkpoint_dir: str = 'checkpoints', amp: bool = True, grad_clip: float = 1.0):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler if scaler is not None else GradScaler(enabled=amp)
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.amp = amp
        self.grad_clip = grad_clip
        self.start_epoch = 0
        logging.info(f"Trainer initialized on device: {self.device}")

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        try:
            for batch in dataloader:
                rgb, depth = batch['rgb'].to(self.device), batch['depth'].to(self.device)
                self.optimizer.zero_grad()
                with autocast(enabled=self.amp):
                    pred = self.model(rgb)
                    mask = (depth > 0) & torch.isfinite(depth)
                    loss = self.loss_fn(pred, depth, mask)
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                total_loss += loss.item()
            if self.scheduler:
                self.scheduler.step()
            avg_loss = total_loss / len(dataloader)
            logging.info(f"Train epoch completed. Avg loss: {avg_loss:.4f}")
            return avg_loss
        except Exception as e:
            logging.error(f"Error during training epoch: {e}\n{traceback.format_exc()}")
            raise

    def save_checkpoint(self, epoch: int, best: bool = False):
        try:
            state = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scaler_state_dict': self.scaler.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            }
            filename = self.checkpoint_dir / (f'best.pth' if best else f'epoch_{epoch+1}.pth')
            torch.save(state, filename)
            logging.info(f"Saved checkpoint: {filename}")
        except Exception as e:
            logging.error(f"Error saving checkpoint: {e}\n{traceback.format_exc()}")
            raise

    def resume(self, checkpoint_path: str):
        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.scaler.load_state_dict(state['scaler_state_dict'])
        if self.scheduler and state.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(state['scheduler_state_dict'])
        self.start_epoch = state.get('epoch', 0) + 1
        logging.info(f"Resumed from checkpoint: {checkpoint_path} at epoch {self.start_epoch}") 