# FILE: components/test_training.py
# ehsanasgharzde - FIXED REDUNDANT CODE BY EXTRACTING PURE FUNCTIONS AND BASECLASS LEVEL METHODS

import os
import torch
import pytest
import tempfile
import numpy as np
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock, call
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional, Callable
import time

from training.trainer import State, Trainer, setup_logging, setup_distributed_training, create_sample_config

def create_synthetic_depth_data(batch_size: int = 4, height: int = 64, 
                                width: int = 64, channels: int = 3) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    images = torch.randn(batch_size, channels, height, width)
    depths = torch.rand(batch_size, 1, height, width)
    masks = torch.ones(batch_size, 1, height, width)
    return images, depths, masks

def create_synthetic_model(input_channels: int = 3, output_channels: int = 1) -> nn.Module:
    class MockDepthModel(nn.Module):
        def __init__(self, in_ch: int, out_ch: int):
            super().__init__()
            self.conv1 = nn.Conv2d(in_ch, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.conv3 = nn.Conv2d(64, out_ch, 3, padding=1)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.sigmoid(self.conv3(x))
            return x
    
    return MockDepthModel(input_channels, output_channels)

def create_test_config() -> Dict:
    return {
        'epochs': 3,
        'loss': {'name': 'SiLogLoss', 'lambda_var': 0.85},
        'optimizer': {'lr': 1e-3, 'weight_decay': 1e-4, 'betas': (0.9, 0.999)},
        'scheduler': {'type': 'cosine', 'eta_min': 1e-6},
        'mixed_precision': False,
        'save_frequency': 2,
        'log_frequency': 5,
        'checkpoint_dir': './test_checkpoints',
        'max_grad_norm': 1.0,
        'early_stopping_patience': 3,
        'early_stopping_min_delta': 1e-4
    }

def create_test_dataloaders():
    images, depths, masks = create_synthetic_depth_data(batch_size=8)
    train_dataset = TensorDataset(images, depths)
    val_dataset = TensorDataset(images[:4], depths[:4])
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    return train_loader, val_loader

class TestState:
    def test_state_initialization(self):
        config = {
            'early_stopping_patience': 5,
            'early_stopping_min_delta': 1e-3,
            'epochs': 10
        }
        
        state = State(config)
        
        assert state.current_epoch == 0
        assert state.global_step == 0
        assert state.best_val_loss == float('inf')
        assert state.best_metrics == {}
        assert state.patience_counter == 0
        assert state.config == config
        assert state.max_patience == 5
        assert state.min_delta == 1e-3
        assert state.train_losses == []
        assert state.val_losses == []
        assert state.train_metrics_history == []
        assert state.val_metrics_history == []
        assert state.epoch_start_time is None
        assert state.training_start_time is None
    
    def test_state_update_epoch(self):
        state = State({})
        
        start_time = time.time()
        state.update_epoch(5)
        
        assert state.current_epoch == 5
        assert state.epoch_start_time is not None
        assert state.epoch_start_time >= start_time
    
    def test_state_update_step(self):
        state = State({})
        
        state.update_step(100)
        assert state.global_step == 100
    
    def test_state_should_early_stop(self):
        config = {
            'early_stopping_patience': 3,
            'early_stopping_min_delta': 1e-2
        }
        state = State(config)
        
        # First improvement - should not stop
        assert not state.should_early_stop(0.5)
        assert state.best_val_loss == 0.5
        assert state.patience_counter == 0
        
        # Small improvement within min_delta - should increment patience
        assert not state.should_early_stop(0.495)
        assert state.patience_counter == 1
        
        # No improvement - should increment patience
        assert not state.should_early_stop(0.6)
        assert state.patience_counter == 2
        
        # No improvement - should increment patience
        assert not state.should_early_stop(0.55)
        assert state.patience_counter == 3
        
        # Patience exceeded - should stop
        assert state.should_early_stop(0.52)
        
    def test_state_get_epoch_duration(self):
        state = State({})
        
        # No epoch started
        assert state.get_epoch_duration() == 0.0
        
        # Epoch started
        state.update_epoch(0)
        time.sleep(0.1)
        duration = state.get_epoch_duration()
        assert duration >= 0.1
        assert duration < 1.0

class TestTrainer:
    @patch('training.trainer.create_loss')
    @patch('training.trainer.create_evaluator')
    @patch('training.trainer.get_all_metrics')
    def test_trainer_initialization_single_gpu(self, mock_get_metrics, mock_create_evaluator, mock_create_loss):
        # Setup mocks
        mock_loss = Mock(spec=nn.Module)
        mock_create_loss.return_value = mock_loss
        
        mock_evaluator = Mock()
        mock_create_evaluator.return_value = mock_evaluator
        
        mock_get_metrics.return_value = {'rmse': Mock(), 'mae': Mock()}
        
        # Create trainer
        model = create_synthetic_model()
        train_loader, val_loader = create_test_dataloaders()
        config = create_test_config()
        device = torch.device('cpu')
        
        trainer = Trainer(model, train_loader, val_loader, config, device)
        
        # Assertions
        assert trainer.model == model
        assert trainer.train_loader == train_loader
        assert trainer.val_loader == val_loader
        assert trainer.config == config
        assert trainer.device == device
        assert trainer.rank == 0
        assert trainer.world_size == 1
        assert not trainer.is_distributed
        assert isinstance(trainer.state, State)
        assert trainer.criterion == mock_loss
        assert trainer.metrics_evaluator == mock_evaluator
        assert trainer.use_amp == False
        assert trainer.max_grad_norm == 1.0
        
        # Check optimizer
        assert isinstance(trainer.optimizer, torch.optim.AdamW)
        assert trainer.optimizer.param_groups[0]['lr'] == 1e-3
        assert trainer.optimizer.param_groups[0]['weight_decay'] == 1e-4
        
        # Check scheduler
        assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
    
    @patch('training.trainer.create_loss')
    @patch('training.trainer.create_evaluator')
    @patch('training.trainer.get_all_metrics')
    def test_trainer_initialization_distributed(self, mock_get_metrics, mock_create_evaluator, mock_create_loss):
        # Setup mocks
        mock_loss = Mock(spec=nn.Module)
        mock_create_loss.return_value = mock_loss
        
        mock_evaluator = Mock()
        mock_create_evaluator.return_value = mock_evaluator
        
        mock_get_metrics.return_value = {'rmse': Mock(), 'mae': Mock()}
        
        with patch('training.trainer.DDP') as mock_ddp:
            model = create_synthetic_model()
            train_loader, val_loader = create_test_dataloaders()
            config = create_test_config()
            device = torch.device('cpu')
            
            trainer = Trainer(model, train_loader, val_loader, config, device, rank=1, world_size=2)
            
            assert trainer.rank == 1
            assert trainer.world_size == 2
            assert trainer.is_distributed
            mock_ddp.assert_called_once_with(model, device_ids=[1], find_unused_parameters=True)
    
    @patch('training.trainer.create_loss')
    @patch('training.trainer.create_evaluator')
    @patch('training.trainer.get_all_metrics')
    def test_trainer_scheduler_types(self, mock_get_metrics, mock_create_evaluator, mock_create_loss):
        # Setup mocks
        mock_loss = Mock(spec=nn.Module)
        mock_create_loss.return_value = mock_loss
        mock_evaluator = Mock()
        mock_create_evaluator.return_value = mock_evaluator
        mock_get_metrics.return_value = {'rmse': Mock()}
        
        model = create_synthetic_model()
        train_loader, val_loader = create_test_dataloaders()
        device = torch.device('cpu')
        
        # Test cosine scheduler
        config_cosine = create_test_config()
        config_cosine['scheduler'] = {'type': 'cosine', 'eta_min': 1e-6}
        trainer_cosine = Trainer(model, train_loader, val_loader, config_cosine, device)
        assert isinstance(trainer_cosine.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        
        # Test plateau scheduler
        config_plateau = create_test_config()
        config_plateau['scheduler'] = {'type': 'plateau', 'factor': 0.5, 'patience': 5}
        trainer_plateau = Trainer(model, train_loader, val_loader, config_plateau, device)
        assert isinstance(trainer_plateau.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        
        # Test no scheduler
        config_none = create_test_config()
        config_none['scheduler'] = {'type': 'none'}
        trainer_none = Trainer(model, train_loader, val_loader, config_none, device)
        assert trainer_none.scheduler is None
    
    @patch('training.trainer.create_loss')
    @patch('training.trainer.create_evaluator')
    @patch('training.trainer.get_all_metrics')
    def test_train_epoch(self, mock_get_metrics, mock_create_evaluator, mock_create_loss):
        # Setup mocks
        mock_loss = Mock(spec=nn.Module)
        mock_loss.return_value = torch.tensor(0.5, requires_grad=True)
        mock_create_loss.return_value = mock_loss
        
        mock_evaluator = Mock()
        mock_evaluator.return_value = {'rmse': torch.tensor(0.3), 'mae': torch.tensor(0.2)}
        mock_create_evaluator.return_value = mock_evaluator
        
        mock_get_metrics.return_value = {'rmse': Mock(), 'mae': Mock()}
        
        # Create trainer
        model = create_synthetic_model()
        train_loader, val_loader = create_test_dataloaders()
        config = create_test_config()
        device = torch.device('cpu')
        
        trainer = Trainer(model, train_loader, val_loader, config, device)
        
        # Run train epoch
        result = trainer.train_epoch()
        
        # Assertions
        assert 'loss' in result
        assert 'rmse' in result
        assert 'mae' in result
        assert isinstance(result['loss'], float)
        assert result['loss'] >= 0
        assert len(trainer.state.train_losses) == 1
        assert len(trainer.state.train_metrics_history) == 1
        
        # Check that optimizer was called
        assert trainer.state.global_step > 0
    
    @patch('training.trainer.create_loss')
    @patch('training.trainer.create_evaluator')
    @patch('training.trainer.get_all_metrics')
    def test_validate_epoch(self, mock_get_metrics, mock_create_evaluator, mock_create_loss):
        # Setup mocks
        mock_loss = Mock(spec=nn.Module)
        mock_loss.return_value = torch.tensor(0.4)
        mock_create_loss.return_value = mock_loss
        
        mock_evaluator = Mock()
        mock_evaluator.return_value = {'rmse': torch.tensor(0.25), 'mae': torch.tensor(0.15)}
        mock_create_evaluator.return_value = mock_evaluator
        
        mock_get_metrics.return_value = {'rmse': Mock(), 'mae': Mock()}
        
        # Create trainer
        model = create_synthetic_model()
        train_loader, val_loader = create_test_dataloaders()
        config = create_test_config()
        device = torch.device('cpu')
        
        trainer = Trainer(model, train_loader, val_loader, config, device)
        
        # Run validate epoch
        result = trainer.validate_epoch()
        
        # Assertions
        assert 'loss' in result
        assert 'rmse' in result
        assert 'mae' in result
        assert isinstance(result['loss'], float)
        assert result['loss'] >= 0
        assert len(trainer.state.val_losses) == 1
        assert len(trainer.state.val_metrics_history) == 1
    
    @patch('training.trainer.create_loss')
    @patch('training.trainer.create_evaluator')
    @patch('training.trainer.get_all_metrics')
    def test_train_epoch_with_dict_batch(self, mock_get_metrics, mock_create_evaluator, mock_create_loss):
        # Setup mocks
        mock_loss = Mock(spec=nn.Module)
        mock_loss.return_value = torch.tensor(0.5, requires_grad=True)
        mock_create_loss.return_value = mock_loss
        
        mock_evaluator = Mock()
        mock_evaluator.return_value = {'rmse': torch.tensor(0.3)}
        mock_create_evaluator.return_value = mock_evaluator
        
        mock_get_metrics.return_value = {'rmse': Mock()}
        
        # Create dataset with dict format
        images, depths, masks = create_synthetic_depth_data(batch_size=4)
        class DictDataset:
            def __init__(self, images, depths, masks):
                self.images = images
                self.depths = depths
                self.masks = masks
            
            def __len__(self):
                return len(self.images)
            
            def __getitem__(self, idx):
                return {
                    'image': self.images[idx],
                    'depth': self.depths[idx],
                    'mask': self.masks[idx]
                }
        
        dict_dataset = DictDataset(images, depths, masks)
        dict_loader = DataLoader(dict_dataset, batch_size=2) # type: ignore
        _, val_loader = create_test_dataloaders()
        
        model = create_synthetic_model()
        config = create_test_config()
        device = torch.device('cpu')
        
        trainer = Trainer(model, dict_loader, val_loader, config, device)
        
        # Run train epoch
        result = trainer.train_epoch()
        
        # Should handle dict format correctly
        assert 'loss' in result
        assert 'rmse' in result
    
    @patch('training.trainer.create_loss')
    @patch('training.trainer.create_evaluator')
    @patch('training.trainer.get_all_metrics')
    def test_save_checkpoint(self, mock_get_metrics, mock_create_evaluator, mock_create_loss):
        # Setup mocks
        mock_loss = Mock(spec=nn.Module)
        mock_create_loss.return_value = mock_loss
        mock_evaluator = Mock()
        mock_create_evaluator.return_value = mock_evaluator
        mock_get_metrics.return_value = {'rmse': Mock()}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model = create_synthetic_model()
            train_loader, val_loader = create_test_dataloaders()
            config = create_test_config()
            config['checkpoint_dir'] = temp_dir
            device = torch.device('cpu')
            
            trainer = Trainer(model, train_loader, val_loader, config, device)
            trainer.state.current_epoch = 5
            trainer.state.global_step = 100
            trainer.state.best_val_loss = 0.3
            
            metrics = {'loss': 0.4, 'rmse': 0.2}
            
            # Test regular checkpoint save
            trainer.save_checkpoint(metrics, is_best=False)
            
            checkpoint_path = os.path.join(temp_dir, "checkpoint_epoch_005.pth")
            assert os.path.exists(checkpoint_path)
            
            # Test best checkpoint save
            trainer.save_checkpoint(metrics, is_best=True)
            
            best_path = os.path.join(temp_dir, "best_model.pth")
            assert os.path.exists(best_path)
            
            # Verify checkpoint contents
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            assert checkpoint['epoch'] == 5
            assert checkpoint['global_step'] == 100
            assert checkpoint['best_val_loss'] == 0.3
            assert 'model_state_dict' in checkpoint
            assert 'optimizer_state_dict' in checkpoint
            assert 'training_state' in checkpoint
    
    @patch('training.trainer.create_loss')
    @patch('training.trainer.create_evaluator')
    @patch('training.trainer.get_all_metrics')
    def test_load_checkpoint(self, mock_get_metrics, mock_create_evaluator, mock_create_loss):
        # Setup mocks
        mock_loss = Mock(spec=nn.Module)
        mock_create_loss.return_value = mock_loss
        mock_evaluator = Mock()
        mock_create_evaluator.return_value = mock_evaluator
        mock_get_metrics.return_value = {'rmse': Mock()}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model = create_synthetic_model()
            train_loader, val_loader = create_test_dataloaders()
            config = create_test_config()
            config['checkpoint_dir'] = temp_dir
            device = torch.device('cpu')
            
            # Create and save a checkpoint first
            trainer1 = Trainer(model, train_loader, val_loader, config, device)
            trainer1.state.current_epoch = 10
            trainer1.state.global_step = 200
            trainer1.state.best_val_loss = 0.2
            trainer1.state.train_losses = [0.5, 0.4, 0.3]
            trainer1.state.val_losses = [0.6, 0.5, 0.4]
            
            checkpoint_path = os.path.join(temp_dir, "test_checkpoint.pth")
            metrics = {'loss': 0.2, 'rmse': 0.1}
            
            checkpoint = {
                'epoch': trainer1.state.current_epoch,
                'global_step': trainer1.state.global_step,
                'model_state_dict': trainer1.model.state_dict(),
                'optimizer_state_dict': trainer1.optimizer.state_dict(),
                'scheduler_state_dict': trainer1.scheduler.state_dict() if trainer1.scheduler else None,
                'scaler_state_dict': trainer1.scaler.state_dict() if trainer1.scaler else None,
                'best_val_loss': trainer1.state.best_val_loss,
                'config': config,
                'metrics': metrics,
                'training_state': {
                    'train_losses': trainer1.state.train_losses,
                    'val_losses': trainer1.state.val_losses,
                    'train_metrics_history': [],
                    'val_metrics_history': []
                }
            }
            torch.save(checkpoint, checkpoint_path)
            
            # Create new trainer and load checkpoint
            model2 = create_synthetic_model()
            trainer2 = Trainer(model2, train_loader, val_loader, config, device)
            
            success = trainer2.load_checkpoint(checkpoint_path)
            
            assert success
            assert trainer2.state.current_epoch == 10
            assert trainer2.state.global_step == 200
            assert trainer2.state.best_val_loss == 0.2
            assert trainer2.state.train_losses == [0.5, 0.4, 0.3]
            assert trainer2.state.val_losses == [0.6, 0.5, 0.4]
    
    @patch('training.trainer.create_loss')
    @patch('training.trainer.create_evaluator')
    @patch('training.trainer.get_all_metrics')
    def test_load_checkpoint_failure(self, mock_get_metrics, mock_create_evaluator, mock_create_loss):
        # Setup mocks
        mock_loss = Mock(spec=nn.Module)
        mock_create_loss.return_value = mock_loss
        mock_evaluator = Mock()
        mock_create_evaluator.return_value = mock_evaluator
        mock_get_metrics.return_value = {'rmse': Mock()}
        
        model = create_synthetic_model()
        train_loader, val_loader = create_test_dataloaders()
        config = create_test_config()
        device = torch.device('cpu')
        
        trainer = Trainer(model, train_loader, val_loader, config, device)
        
        # Test loading non-existent checkpoint
        success = trainer.load_checkpoint("non_existent_checkpoint.pth")
        assert not success
    
    @patch('training.trainer.create_loss')
    @patch('training.trainer.create_evaluator')
    @patch('training.trainer.get_all_metrics')
    def test_train_full_loop(self, mock_get_metrics, mock_create_evaluator, mock_create_loss):
        # Setup mocks
        mock_loss = Mock(spec=nn.Module)
        mock_loss_values = [torch.tensor(0.8, requires_grad=True), torch.tensor(0.6, requires_grad=True), torch.tensor(0.4, requires_grad=True)]
        mock_loss.side_effect = mock_loss_values * 10  # Repeat for multiple batches
        mock_create_loss.return_value = mock_loss
        
        mock_evaluator = Mock()
        mock_evaluator.return_value = {'rmse': torch.tensor(0.3), 'mae': torch.tensor(0.2)}
        mock_create_evaluator.return_value = mock_evaluator
        
        mock_get_metrics.return_value = {'rmse': Mock(), 'mae': Mock()}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model = create_synthetic_model()
            train_loader, val_loader = create_test_dataloaders()
            config = create_test_config()
            config['epochs'] = 2
            config['checkpoint_dir'] = temp_dir
            config['save_frequency'] = 1
            device = torch.device('cpu')
            
            trainer = Trainer(model, train_loader, val_loader, config, device)
            
            # Run training
            summary = trainer.train()
            
            # Assertions
            assert 'total_epochs' in summary
            assert 'total_time' in summary
            assert 'best_val_loss' in summary
            assert 'best_metrics' in summary
            assert 'final_lr' in summary
            assert 'training_history' in summary
            
            assert summary['total_epochs'] == 2
            assert summary['total_time'] > 0
            assert len(trainer.state.train_losses) == 2
            assert len(trainer.state.val_losses) == 2
    
    @patch('training.trainer.create_loss')
    @patch('training.trainer.create_evaluator')
    @patch('training.trainer.get_all_metrics')
    def test_train_with_early_stopping(self, mock_get_metrics, mock_create_evaluator, mock_create_loss):
        # Setup mocks
        mock_loss = Mock(spec=nn.Module)
        # Simulate no improvement in validation loss
        mock_loss.side_effect = [torch.tensor(0.5, requires_grad=True)] * 100
        mock_create_loss.return_value = mock_loss
        
        mock_evaluator = Mock()
        mock_evaluator.return_value = {'rmse': torch.tensor(0.3)}
        mock_create_evaluator.return_value = mock_evaluator
        
        mock_get_metrics.return_value = {'rmse': Mock()}
        
        model = create_synthetic_model()
        train_loader, val_loader = create_test_dataloaders()
        config = create_test_config()
        config['epochs'] = 10
        config['early_stopping_patience'] = 2
        device = torch.device('cpu')
        
        trainer = Trainer(model, train_loader, val_loader, config, device)
        
        # Run training - should stop early
        summary = trainer.train()
        
        # Should stop before 10 epochs due to early stopping
        assert summary['total_epochs'] < 10
    
    @patch('training.trainer.create_loss')
    @patch('training.trainer.create_evaluator')
    @patch('training.trainer.get_all_metrics')
    def test_train_with_amp(self, mock_get_metrics, mock_create_evaluator, mock_create_loss):
        # Setup mocks
        mock_loss = Mock(spec=nn.Module)
        mock_loss.return_value = torch.tensor(0.5, requires_grad=True)
        mock_create_loss.return_value = mock_loss
        
        mock_evaluator = Mock()
        mock_evaluator.return_value = {'rmse': torch.tensor(0.3)}
        mock_create_evaluator.return_value = mock_evaluator
        
        mock_get_metrics.return_value = {'rmse': Mock()}
        
        model = create_synthetic_model()
        train_loader, val_loader = create_test_dataloaders()
        config = create_test_config()
        config['mixed_precision'] = True
        config['epochs'] = 1
        device = torch.device('cpu')
        
        trainer = Trainer(model, train_loader, val_loader, config, device)
        
        assert trainer.use_amp == True
        assert trainer.scaler is not None
        
        # Run one epoch
        result = trainer.train_epoch()
        assert 'loss' in result

class TestUtilityFunctions:
    def test_setup_logging(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            setup_logging(temp_dir)
            
            # Check if log file was created
            log_files = [f for f in os.listdir(temp_dir) if f.startswith('training_') and f.endswith('.log')]
            assert len(log_files) == 1
    
    @patch.dict(os.environ, {}, clear=True)
    def test_setup_distributed_training_single_process(self):
        rank, world_size, device = setup_distributed_training()
        
        assert rank == 0
        assert world_size == 1
        assert isinstance(device, torch.device)
    
    @patch.dict(os.environ, {'RANK': '1', 'WORLD_SIZE': '2'}, clear=True)
    @patch('training.trainer.dist.init_process_group')
    def test_setup_distributed_training_multi_process(self, mock_init_process_group):
        rank, world_size, device = setup_distributed_training()
        
        assert rank == 1
        assert world_size == 2
        mock_init_process_group.assert_called_once()
    
    def test_create_sample_config(self):
        config = create_sample_config()
        
        required_keys = ['epochs', 'loss', 'optimizer', 'scheduler', 'mixed_precision']
        for key in required_keys:
            assert key in config
        
        assert config['epochs'] == 100
        assert config['loss']['name'] == 'SiLogLoss'
        assert config['optimizer']['lr'] == 1e-4
        assert config['scheduler']['type'] == 'cosine'

class TestEdgeCases:
    @patch('training.trainer.create_loss')
    @patch('training.trainer.create_evaluator')
    @patch('training.trainer.get_all_metrics')
    def test_train_with_keyboard_interrupt(self, mock_get_metrics, mock_create_evaluator, mock_create_loss):
        # Setup mocks
        mock_loss = Mock(spec=nn.Module)
        mock_loss.return_value = torch.tensor(0.5, requires_grad=True)
        mock_create_loss.return_value = mock_loss
        
        mock_evaluator = Mock()
        mock_evaluator.return_value = {'rmse': torch.tensor(0.3)}
        mock_create_evaluator.return_value = mock_evaluator
        
        mock_get_metrics.return_value = {'rmse': Mock()}
        
        model = create_synthetic_model()
        train_loader, val_loader = create_test_dataloaders()
        config = create_test_config()
        config['epochs'] = 1
        device = torch.device('cpu')
        
        trainer = Trainer(model, train_loader, val_loader, config, device)
        
        # Mock train_epoch to raise KeyboardInterrupt
        original_train_epoch = trainer.train_epoch
        def mock_train_epoch():
            raise KeyboardInterrupt("User interrupted")
        
        trainer.train_epoch = mock_train_epoch
        
        # Should handle KeyboardInterrupt gracefully
        summary = trainer.train()
        
        assert 'total_time' in summary
    
    @patch('training.trainer.create_loss')
    @patch('training.trainer.create_evaluator')
    @patch('training.trainer.get_all_metrics')
    def test_save_checkpoint_non_main_rank(self, mock_get_metrics, mock_create_evaluator, mock_create_loss):
        # Setup mocks
        mock_loss = Mock(spec=nn.Module)
        mock_create_loss.return_value = mock_loss
        mock_evaluator = Mock()
        mock_create_evaluator.return_value = mock_evaluator
        mock_get_metrics.return_value = {'rmse': Mock()}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model = create_synthetic_model()
            train_loader, val_loader = create_test_dataloaders()
            config = create_test_config()
            config['checkpoint_dir'] = temp_dir
            device = torch.device('cpu')
            
            # Create trainer with rank != 0
            trainer = Trainer(model, train_loader, val_loader, config, device, rank=1, world_size=2)
            
            metrics = {'loss': 0.4, 'rmse': 0.2}
            
            # Should not save checkpoint when rank != 0
            trainer.save_checkpoint(metrics, is_best=False)
            
            # No checkpoint should be created
            checkpoint_files = [f for f in os.listdir(temp_dir) if f.endswith('.pth')]
            assert len(checkpoint_files) == 0
    
    @patch('training.trainer.create_loss')
    @patch('training.trainer.create_evaluator')
    @patch('training.trainer.get_all_metrics')
    def test_train_epoch_with_gradient_clipping(self, mock_get_metrics, mock_create_evaluator, mock_create_loss):
        # Setup mocks
        mock_loss = Mock(spec=nn.Module)
        mock_loss.return_value = torch.tensor(0.5, requires_grad=True)
        mock_create_loss.return_value = mock_loss
        
        mock_evaluator = Mock()
        mock_evaluator.return_value = {'rmse': torch.tensor(0.3)}
        mock_create_evaluator.return_value = mock_evaluator
        
        mock_get_metrics.return_value = {'rmse': Mock()}
        
        model = create_synthetic_model()
        train_loader, val_loader = create_test_dataloaders()
        config = create_test_config()
        config['max_grad_norm'] = 0.5  # Enable gradient clipping
        device = torch.device('cpu')
        
        trainer = Trainer(model, train_loader, val_loader, config, device)
        
        with patch('torch.nn.utils.clip_grad_norm_') as mock_clip_grad:
            result = trainer.train_epoch()
            
            # Gradient clipping should be called
            assert mock_clip_grad.call_count > 0
            assert 'loss' in result