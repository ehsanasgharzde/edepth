# FILE: components/test_training.py
# ehsanasgharzde - FIXED REDUNDANT CODE BY EXTRACTING PURE FUNCTIONS AND BASECLASS LEVEL METHODS

import os
import time
import torch
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

def test_state_initialization() -> bool:
    # Test State class initialization
    config = {
        'early_stopping_patience': 5,
        'early_stopping_min_delta': 1e-3,
        'epochs': 10
    }
    
    state = State(config)
    
    # Verify initial values
    checks = [
        state.current_epoch == 0,
        state.global_step == 0,
        state.best_val_loss == float('inf'),
        len(state.best_metrics) == 0,
        state.patience_counter == 0,
        state.max_patience == 5,
        state.min_delta == 1e-3,
        len(state.train_losses) == 0,
        len(state.val_losses) == 0,
        state.epoch_start_time is None
    ]
    
    return all(checks)

def test_state_update_operations() -> bool:
    # Test State update methods
    state = State({})
    
    # Test epoch update
    start_time = time.time()
    state.update_epoch(5)
    
    epoch_check = state.current_epoch == 5 and state.epoch_start_time >= start_time
    
    # Test step update
    state.update_step(100)
    step_check = state.global_step == 100
    
    return epoch_check and step_check

def test_state_early_stopping_logic() -> bool:
    # Test early stopping functionality
    config = {'early_stopping_patience': 3, 'early_stopping_min_delta': 1e-2}
    state = State(config)
    
    # First improvement - should not stop
    stop1 = state.should_early_stop(0.5)
    check1 = not stop1 and state.best_val_loss == 0.5 and state.patience_counter == 0
    
    # Small improvement within min_delta - should increment patience
    stop2 = state.should_early_stop(0.495)
    check2 = not stop2 and state.patience_counter == 1
    
    # No improvement - increment patience
    stop3 = state.should_early_stop(0.6)
    check3 = not stop3 and state.patience_counter == 2
    
    # Still no improvement - increment patience
    stop4 = state.should_early_stop(0.55)
    check4 = not stop4 and state.patience_counter == 3
    
    # Patience exceeded - should stop
    stop5 = state.should_early_stop(0.52)
    check5 = stop5
    
    return all([check1, check2, check3, check4, check5])

def test_state_epoch_duration() -> bool:
    # Test epoch duration calculation
    state = State({})
    
    # No epoch started
    duration1 = state.get_epoch_duration()
    check1 = duration1 == 0.0
    
    # Epoch started
    state.update_epoch(0)
    time.sleep(0.01)  # Small sleep to ensure duration > 0
    duration2 = state.get_epoch_duration()
    check2 = duration2 > 0
    
    return check1 and check2

def test_trainer_initialization() -> bool:
    # Test Trainer initialization with various configurations
    try:
        model = create_test_model()
        train_loader, val_loader = create_test_dataloaders()
        config = create_test_config()
        device = torch.device('cpu')
        
        trainer = Trainer(model, train_loader, val_loader, config, device)
        
        # Verify trainer attributes
        checks = [
            trainer.model == model,
            trainer.train_loader == train_loader,
            trainer.val_loader == val_loader,
            trainer.config == config,
            trainer.device == device,
            trainer.rank == 0,
            trainer.world_size == 1,
            not trainer.is_distributed,
            isinstance(trainer.state, State),
            trainer.use_amp == False,
            trainer.max_grad_norm == 1.0,
            isinstance(trainer.optimizer, torch.optim.AdamW),
            trainer.optimizer.param_groups[0]['lr'] == 1e-3,
            trainer.optimizer.param_groups[0]['weight_decay'] == 1e-4
        ]
        
        return all(checks)
    except Exception:
        return False

def test_trainer_scheduler_configurations() -> bool:
    # Test different scheduler configurations
    try:
        model = create_test_model()
        train_loader, val_loader = create_test_dataloaders()
        device = torch.device('cpu')
        
        # Test cosine scheduler
        config_cosine = create_test_config()
        config_cosine['scheduler'] = {'type': 'cosine', 'eta_min': 1e-6}
        trainer_cosine = Trainer(model, train_loader, val_loader, config_cosine, device)
        cosine_check = isinstance(trainer_cosine.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        
        # Test plateau scheduler
        config_plateau = create_test_config()
        config_plateau['scheduler'] = {'type': 'plateau', 'factor': 0.5, 'patience': 5}
        trainer_plateau = Trainer(model, train_loader, val_loader, config_plateau, device)
        plateau_check = isinstance(trainer_plateau.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        
        # Test no scheduler
        config_none = create_test_config()
        config_none['scheduler'] = {'type': 'none'}
        trainer_none = Trainer(model, train_loader, val_loader, config_none, device)
        none_check = trainer_none.scheduler is None
        
        return cosine_check and plateau_check and none_check
    except Exception:
        return False

def test_training_epoch_execution() -> bool:
    # Test single training epoch execution
    try:
        model = create_test_model()
        train_loader, val_loader = create_test_dataloaders()
        config = create_test_config()
        device = torch.device('cpu')
        
        trainer = Trainer(model, train_loader, val_loader, config, device)
        
        # Execute training epoch
        initial_step = trainer.state.global_step
        result = trainer.train_epoch()
        
        # Verify results
        checks = [
            'loss' in result,
            isinstance(result['loss'], float),
            result['loss'] >= 0,
            len(trainer.state.train_losses) == 1,
            len(trainer.state.train_metrics_history) == 1,
            trainer.state.global_step > initial_step
        ]
        
        return all(checks)
    except Exception:
        return False

def test_validation_epoch_execution() -> bool:
    # Test single validation epoch execution
    try:
        model = create_test_model()
        train_loader, val_loader = create_test_dataloaders()
        config = create_test_config()
        device = torch.device('cpu')
        
        trainer = Trainer(model, train_loader, val_loader, config, device)
        
        # Execute validation epoch
        result = trainer.validate_epoch()
        
        # Verify results
        checks = [
            'loss' in result,
            isinstance(result['loss'], float),
            result['loss'] >= 0,
            len(trainer.state.val_losses) == 1,
            len(trainer.state.val_metrics_history) == 1
        ]
        
        return all(checks)
    except Exception:
        return False

def test_checkpoint_save_and_load() -> bool:
    # Test checkpoint saving and loading functionality
    try:
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
            
            files_exist = os.path.exists(checkpoint_path) and os.path.exists(best_path)
            
            # Create new trainer and load checkpoint
            model2 = create_test_model()
            trainer2 = Trainer(model2, train_loader, val_loader, config, device)
            
            load_success = trainer2.load_checkpoint(checkpoint_path)
            
            # Verify loaded state
            state_checks = [
                load_success,
                trainer2.state.current_epoch == 5,
                trainer2.state.global_step == 100,
                trainer2.state.best_val_loss == 0.3,
                trainer2.state.train_losses == [0.5, 0.4, 0.3]
            ]
            
            return files_exist and all(state_checks)
    except Exception:
        return False

def test_full_training_loop() -> bool:
    # Test complete training loop
    try:
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
            
            summary_checks = [
                all(key in summary for key in required_keys),
                summary['total_epochs'] == 2,
                summary['total_time'] > 0,
                len(trainer.state.train_losses) == 2,
                len(trainer.state.val_losses) == 2
            ]
            
            return all(summary_checks)
    except Exception:
        return False

def test_early_stopping_mechanism() -> bool:
    # Test early stopping functionality
    try:
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
        
        return not should_stop_1 and not should_stop_2 and should_stop_3
    except Exception:
        return False

def test_utility_functions() -> bool:
    # Test utility functions
    try:
        # Test logging setup
        with tempfile.TemporaryDirectory() as temp_dir:
            setup_logging(temp_dir)
            log_files = [f for f in os.listdir(temp_dir) if f.startswith('training_') and f.endswith('.log')]
            logging_check = len(log_files) == 1
        
        # Test distributed training setup (single process)
        rank, world_size, device = setup_distributed_training()
        distributed_check = rank == 0 and world_size == 1 and isinstance(device, torch.device)
        
        # Test sample config creation
        config = create_sample_config()
        required_keys = ['epochs', 'loss', 'optimizer', 'scheduler', 'mixed_precision']
        config_check = all(key in config for key in required_keys)
        
        return logging_check and distributed_check and config_check
    except Exception:
        return False

def test_mixed_precision_configuration() -> bool:
    # Test mixed precision configuration
    try:
        model = create_test_model()
        train_loader, val_loader = create_test_dataloaders()
        config = create_test_config()
        config['mixed_precision'] = True
        device = torch.device('cpu')
        
        trainer = Trainer(model, train_loader, val_loader, config, device)
        
        # Verify AMP setup
        checks = [
            trainer.use_amp == True,
            trainer.scaler is not None
        ]
        
        return all(checks)
    except Exception:
        return False

def test_dict_batch_format() -> bool:
    # Test handling of dictionary batch format
    try:
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
        
        return 'loss' in result and isinstance(result['loss'], float)
    except Exception:
        return False

def run_all_tests() -> Dict[str, bool]:
    # Execute all test functions and return results
    tests = {
        'state_initialization': test_state_initialization,
        'state_update_operations': test_state_update_operations,
        'state_early_stopping_logic': test_state_early_stopping_logic,
        'state_epoch_duration': test_state_epoch_duration,
        'trainer_initialization': test_trainer_initialization,
        'trainer_scheduler_configurations': test_trainer_scheduler_configurations,
        'training_epoch_execution': test_training_epoch_execution,
        'validation_epoch_execution': test_validation_epoch_execution,
        'checkpoint_save_and_load': test_checkpoint_save_and_load,
        'full_training_loop': test_full_training_loop,
        'early_stopping_mechanism': test_early_stopping_mechanism,
        'utility_functions': test_utility_functions,
        'mixed_precision_configuration': test_mixed_precision_configuration,
        'dict_batch_format': test_dict_batch_format
    }
    
    results = {}
    for test_name, test_func in tests.items():
        try:
            results[test_name] = test_func()
        except Exception as e:
            results[test_name] = False
            print(f"Test {test_name} failed with exception: {e}")
    
    return results

def print_test_results(results: Dict[str, bool]) -> None:
    # Print formatted test results
    print("="*60)
    print("FUNCTIONAL TEST RESULTS")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name:<35} | {status}")
    
    print("-"*60)
    print(f"TOTAL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("="*60)