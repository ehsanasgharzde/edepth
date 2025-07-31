# test_training_pytest.py

import os
import time
import torch
import pytest
import tempfile
import torch.nn as nn
from typing import Dict, Tuple, Any
from torch.utils.data import DataLoader, TensorDataset

from training.trainer import State, Trainer, setup_logging, setup_distributed_training, create_sample_config

def create_synthetic_depth_data(batch_size: int = 4, height: int = 64, 
                                width: int = 64, channels: int = 3) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Generate synthetic image data
    images = torch.randn(batch_size, channels, height, width)
    depths = torch.rand(batch_size, 1, height, width)
    masks = torch.ones(batch_size, 1, height, width)
    return images, depths, masks

def create_test_model(input_channels: int = 3, output_channels: int = 1) -> nn.Module:
    # Simple convolutional model for testing
    class TestDepthModel(nn.Module):
        def __init__(self, in_ch: int, out_ch: int):
            super().__init__()
            self.conv1 = nn.Conv2d(in_ch, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.conv3 = nn.Conv2d(32, out_ch, 3, padding=1)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.sigmoid(self.conv3(x))
            return x
    
    return TestDepthModel(input_channels, output_channels)

def create_test_config() -> Dict[str, Any]:
    # Minimal configuration for testing
    return {
        'epochs': 2,
        'loss': {'name': 'SiLogLoss', 'lambda_var': 0.85},
        'optimizer': {'lr': 1e-3, 'weight_decay': 1e-4, 'betas': (0.9, 0.999)},
        'scheduler': {'type': 'cosine', 'eta_min': 1e-6},
        'mixed_precision': False,
        'save_frequency': 1,
        'log_frequency': 5,
        'checkpoint_dir': './test_checkpoints',
        'max_grad_norm': 1.0,
        'early_stopping_patience': 3,
        'early_stopping_min_delta': 1e-4
    }

def create_test_dataloaders() -> Tuple[DataLoader, DataLoader]:
    # Create simple dataloaders for testing
    images, depths, masks = create_synthetic_depth_data(batch_size=8)
    train_dataset = TensorDataset(images, depths)
    val_dataset = TensorDataset(images[:4], depths[:4])
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    return train_loader, val_loader

def test_state_initialization() -> None:
    # Test State class initialization
    config = {
        'early_stopping_patience': 5,
        'early_stopping_min_delta': 1e-3,
        'epochs': 10
    }
    
    state = State(config)
    
    # Verify initial values
    assert state.current_epoch == 0
    assert state.global_step == 0
    assert state.best_val_loss == float('inf')
    assert len(state.best_metrics) == 0
    assert state.patience_counter == 0
    assert state.max_patience == 5
    assert state.min_delta == 1e-3
    assert len(state.train_losses) == 0
    assert len(state.val_losses) == 0
    assert state.epoch_start_time is None

def test_state_update_operations() -> None:
    # Test State update methods
    state = State({})
    
    # Test epoch update
    start_time = time.time()
    state.update_epoch(5)
    
    assert state.current_epoch == 5
    assert state.epoch_start_time >= start_time
    
    # Test step update
    state.update_step(100)
    assert state.global_step == 100

def test_state_early_stopping_logic() -> None:
    # Test early stopping functionality
    config = {'early_stopping_patience': 3, 'early_stopping_min_delta': 1e-2}
    state = State(config)
    
    # First improvement - should not stop
    stop1 = state.should_early_stop(0.5)
    assert not stop1
    assert state.best_val_loss == 0.5
    assert state.patience_counter == 0
    
    # Small improvement within min_delta - should increment patience
    stop2 = state.should_early_stop(0.495)
    assert not stop2
    assert state.patience_counter == 1
    
    # No improvement - increment patience
    stop3 = state.should_early_stop(0.6)
    assert not stop3
    assert state.patience_counter == 2
    
    # Still no improvement - increment patience
    stop4 = state.should_early_stop(0.55)
    assert not stop4
    assert state.patience_counter == 3
    
    # Patience exceeded - should stop
    stop5 = state.should_early_stop(0.52)
    assert stop5

def test_state_epoch_duration() -> None:
    # Test epoch duration calculation
    state = State({})
    
    # No epoch started
    duration1 = state.get_epoch_duration()
    assert duration1 == 0.0
    
    # Epoch started
    state.update_epoch(0)
    time.sleep(0.01)  # Small sleep to ensure duration > 0
    duration2 = state.get_epoch_duration()
    assert duration2 > 0

def test_trainer_initialization() -> None:
    # Test Trainer initialization with various configurations
    model = create_test_model()
    train_loader, val_loader = create_test_dataloaders()
    config = create_test_config()
    device = torch.device('cpu')
    
    trainer = Trainer(model, train_loader, val_loader, config, device)
    
    # Verify trainer attributes
    assert trainer.model == model
    assert trainer.train_loader == train_loader
    assert trainer.val_loader == val_loader
    assert trainer.config == config
    assert trainer.device == device
    assert trainer.rank == 0
    assert trainer.world_size == 1
    assert not trainer.is_distributed
    assert isinstance(trainer.state, State)
    assert trainer.use_amp == False
    assert trainer.max_grad_norm == 1.0
    assert isinstance(trainer.optimizer, torch.optim.AdamW)
    assert trainer.optimizer.param_groups[0]['lr'] == 1e-3
    assert trainer.optimizer.param_groups[0]['weight_decay'] == 1e-4

def test_trainer_scheduler_configurations() -> None:
    # Test different scheduler configurations
    model = create_test_model()
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

def test_training_epoch_execution() -> None:
    # Test single training epoch execution
    model = create_test_model()
    train_loader, val_loader = create_test_dataloaders()
    config = create_test_config()
    device = torch.device('cpu')
    
    trainer = Trainer(model, train_loader, val_loader, config, device)
    
    # Execute training epoch
    initial_step = trainer.state.global_step
    result = trainer.train_epoch()
    
    # Verify results
    assert 'loss' in result
    assert isinstance(result['loss'], float)
    assert result['loss'] >= 0
    assert len(trainer.state.train_losses) == 1
    assert len(trainer.state.train_metrics_history) == 1
    assert trainer.state.global_step > initial_step

def test_validation_epoch_execution() -> None:
    # Test single validation epoch execution
    model = create_test_model()
    train_loader, val_loader = create_test_dataloaders()
    config = create_test_config()
    device = torch.device('cpu')
    
    trainer = Trainer(model, train_loader, val_loader, config, device)
    
    # Execute validation epoch
    result = trainer.validate_epoch()
    
    # Verify results
    assert 'loss' in result
    assert isinstance(result['loss'], float)
    assert result['loss'] >= 0
    assert len(trainer.state.val_losses) == 1
    assert len(trainer.state.val_metrics_history) == 1

def test_checkpoint_save_and_load() -> None:
    # Test checkpoint saving and loading functionality
    with tempfile.TemporaryDirectory() as temp_dir:
        model = create_test_model()
        train_loader, val_loader = create_test_dataloaders()
        config = create_test_config()
        config['checkpoint_dir'] = temp_dir
        device = torch.device('cpu')
        
        # Create trainer and set some state
        trainer1 = Trainer(model, train_loader, val_loader, config, device)
        trainer1.state.current_epoch = 5
        trainer1.state.global_step = 100
        trainer1.state.best_val_loss = 0.3
        trainer1.state.train_losses = [0.5, 0.4, 0.3]
        
        metrics = {'loss': 0.25, 'rmse': 0.1}
        
        # Save checkpoint
        trainer1.save_checkpoint(metrics, is_best=True)
        
        # Verify files exist
        checkpoint_path = os.path.join(temp_dir, "checkpoint_epoch_005.pth")
        best_path = os.path.join(temp_dir, "best_model.pth")
        
        assert os.path.exists(checkpoint_path)
        assert os.path.exists(best_path)
        
        # Create new trainer and load checkpoint
        model2 = create_test_model()
        trainer2 = Trainer(model2, train_loader, val_loader, config, device)
        
        load_success = trainer2.load_checkpoint(checkpoint_path)
        
        # Verify loaded state
        assert load_success
        assert trainer2.state.current_epoch == 5
        assert trainer2.state.global_step == 100
        assert trainer2.state.best_val_loss == 0.3
        assert trainer2.state.train_losses == [0.5, 0.4, 0.3]

def test_full_training_loop() -> None:
    # Test complete training loop
    with tempfile.TemporaryDirectory() as temp_dir:
        model = create_test_model()
        train_loader, val_loader = create_test_dataloaders()
        config = create_test_config()
        config['epochs'] = 2
        config['checkpoint_dir'] = temp_dir
        config['save_frequency'] = 1
        device = torch.device('cpu')
        
        trainer = Trainer(model, train_loader, val_loader, config, device)
        
        # Run full training
        summary = trainer.train()
        
        # Verify training summary
        required_keys = ['total_epochs', 'total_time', 'best_val_loss', 'best_metrics', 
                       'final_lr', 'training_history']
        
        assert all(key in summary for key in required_keys)
        assert summary['total_epochs'] == 2
        assert summary['total_time'] > 0
        assert len(trainer.state.train_losses) == 2
        assert len(trainer.state.val_losses) == 2

def test_early_stopping_mechanism() -> None:
    # Test early stopping functionality
    model = create_test_model()
    train_loader, val_loader = create_test_dataloaders()
    config = create_test_config()
    config['epochs'] = 10
    config['early_stopping_patience'] = 2
    device = torch.device('cpu')
    
    trainer = Trainer(model, train_loader, val_loader, config, device)
    
    # Manually set validation losses to trigger early stopping
    trainer.state.best_val_loss = 0.5
    
    # Simulate no improvement scenario
    should_stop_1 = trainer.state.should_early_stop(0.6)  # Worse loss
    should_stop_2 = trainer.state.should_early_stop(0.55)  # Still worse
    should_stop_3 = trainer.state.should_early_stop(0.52)  # Still worse - should trigger stop
    
    assert not should_stop_1
    assert not should_stop_2
    assert should_stop_3

def test_utility_functions() -> None:
    # Test utility functions
    # Test logging setup
    with tempfile.TemporaryDirectory() as temp_dir:
        setup_logging(temp_dir)
        log_files = [f for f in os.listdir(temp_dir) if f.startswith('training_') and f.endswith('.log')]
        assert len(log_files) == 1
    
    # Test distributed training setup (single process)
    rank, world_size, device = setup_distributed_training()
    assert rank == 0
    assert world_size == 1
    assert isinstance(device, torch.device)
    
    # Test sample config creation
    config = create_sample_config()
    required_keys = ['epochs', 'loss', 'optimizer', 'scheduler', 'mixed_precision']
    assert all(key in config for key in required_keys)

def test_mixed_precision_configuration() -> None:
    # Test mixed precision configuration
    model = create_test_model()
    train_loader, val_loader = create_test_dataloaders()
    config = create_test_config()
    config['mixed_precision'] = True
    device = torch.device('cpu')
    
    trainer = Trainer(model, train_loader, val_loader, config, device)
    
    # Verify AMP setup
    assert trainer.use_amp == True
    assert trainer.scaler is not None

def test_dict_batch_format() -> None:
    # Test handling of dictionary batch format
    model = create_test_model()
    images, depths, masks = create_synthetic_depth_data(batch_size=4)
    
    # Create dataset with dict format
    class DictDataset:
        def __init__(self, images: torch.Tensor, depths: torch.Tensor, masks: torch.Tensor):
            self.images = images
            self.depths = depths
            self.masks = masks
        
        def __len__(self) -> int:
            return len(self.images)
        
        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            return {
                'image': self.images[idx],
                'depth': self.depths[idx],
                'mask': self.masks[idx]
            }
    
    dict_dataset = DictDataset(images, depths, masks)
    dict_loader = DataLoader(dict_dataset, batch_size=2)
    _, val_loader = create_test_dataloaders()
    
    config = create_test_config()
    device = torch.device('cpu')
    
    trainer = Trainer(model, dict_loader, val_loader, config, device)
    
    # Execute one training epoch
    result = trainer.train_epoch()
    
    assert 'loss' in result
    assert isinstance(result['loss'], float)

# Pytest fixtures for shared test resources
@pytest.fixture
def test_model() -> nn.Module:
    return create_test_model()

@pytest.fixture
def test_dataloaders() -> Tuple[DataLoader, DataLoader]:
    return create_test_dataloaders()

@pytest.fixture
def test_config() -> Dict[str, Any]:
    return create_test_config()

@pytest.fixture
def test_device() -> torch.device:
    return torch.device('cpu')

# Additional parameterized tests using pytest features
@pytest.mark.parametrize("batch_size,height,width,channels", [
    (2, 32, 32, 3),
    (4, 64, 64, 3),
    (1, 128, 128, 1),
])
def test_synthetic_data_generation(batch_size: int, height: int, width: int, channels: int) -> None:
    # Test synthetic data generation with different parameters
    images, depths, masks = create_synthetic_depth_data(batch_size, height, width, channels)
    
    assert images.shape == (batch_size, channels, height, width)
    assert depths.shape == (batch_size, 1, height, width)
    assert masks.shape == (batch_size, 1, height, width)
    assert torch.all(masks == 1.0)

@pytest.mark.parametrize("scheduler_type", ['cosine', 'plateau', 'none'])
def test_scheduler_types(scheduler_type: str) -> None:
    # Test different scheduler types with parametrization
    model = create_test_model()
    train_loader, val_loader = create_test_dataloaders()
    config = create_test_config()
    config['scheduler'] = {'type': scheduler_type}
    device = torch.device('cpu')
    
    trainer = Trainer(model, train_loader, val_loader, config, device)
    
    if scheduler_type == 'cosine':
        assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
    elif scheduler_type == 'plateau':
        assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    else:
        assert trainer.scheduler is None

@pytest.mark.parametrize("mixed_precision", [True, False])
def test_mixed_precision_options(mixed_precision: bool) -> None:
    # Test mixed precision configuration options
    model = create_test_model()
    train_loader, val_loader = create_test_dataloaders()
    config = create_test_config()
    config['mixed_precision'] = mixed_precision
    device = torch.device('cpu')
    
    trainer = Trainer(model, train_loader, val_loader, config, device)
    
    assert trainer.use_amp == mixed_precision
    if mixed_precision:
        assert trainer.scaler is not None
    else:
        assert trainer.scaler is None