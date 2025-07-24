# FILE: configs/config_loader.py
# ehsanasgharzadeh - CONFIGURATION MANAGER
# hosseinsolymanzadeh - PROPER COMMENTING
# ehsanasgharzde - FIXED REDUNDANT CODE BY EXTRACTING PURE FUNCTIONS AND BASECLASS LEVEL METHODS

import json
import yaml
import logging
import hashlib
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
from datetime import datetime
from omegaconf import OmegaConf
import copy

logger = logging.getLogger(__name__)

class ConfigError(Exception):
    pass

class ConfigValidationError(ConfigError):
    pass

class ConfigLoadError(ConfigError):
    pass

class ConfigSaveError(ConfigError):
    pass

class ConfigMergeError(ConfigError):
    pass

class ConfigLoaderInterface(ABC):
    @abstractmethod
    def load(self, path: Union[str, Path]) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def save(self, config: Dict[str, Any], path: Union[str, Path]) -> None:
        pass
    
    @abstractmethod
    def supports_format(self, file_extension: str) -> bool:
        pass

class BaseConfigLoader(ConfigLoaderInterface):
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def _resolve_path(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        if not path.is_absolute():
            path = self.config_dir / path
        return path
    
    def _validate_path(self, path: Path, for_write: bool = False) -> None:
        if for_write:
            path.parent.mkdir(parents=True, exist_ok=True)
        elif not path.exists():
            raise ConfigLoadError(f"Configuration file not found: {path}")

class YAMLConfigLoader(BaseConfigLoader):
    def load(self, path: Union[str, Path]) -> Dict[str, Any]:
        path = self._resolve_path(path)
        self._validate_path(path)
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            logger.debug(f"Loaded YAML config from {path}")
            return config
        except yaml.YAMLError as e:
            raise ConfigLoadError(f"Failed to parse YAML file {path}: {e}")
        except Exception as e:
            raise ConfigLoadError(f"Failed to load config from {path}: {e}")
    
    def save(self, config: Dict[str, Any], path: Union[str, Path]) -> None:
        path = self._resolve_path(path)
        self._validate_path(path, for_write=True)
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(config, f, default_flow_style=False, indent=2, sort_keys=False)
            logger.debug(f"Saved YAML config to {path}")
        except Exception as e:
            raise ConfigSaveError(f"Failed to save config to {path}: {e}")
    
    def supports_format(self, file_extension: str) -> bool:
        return file_extension.lower() in ['.yaml', '.yml']

class JSONConfigLoader(BaseConfigLoader):
    def load(self, path: Union[str, Path]) -> Dict[str, Any]:
        path = self._resolve_path(path)
        self._validate_path(path)
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.debug(f"Loaded JSON config from {path}")
            return config
        except json.JSONDecodeError as e:
            raise ConfigLoadError(f"Failed to parse JSON file {path}: {e}")
        except Exception as e:
            raise ConfigLoadError(f"Failed to load config from {path}: {e}")
    
    def save(self, config: Dict[str, Any], path: Union[str, Path]) -> None:
        path = self._resolve_path(path)
        self._validate_path(path, for_write=True)
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved JSON config to {path}")
        except Exception as e:
            raise ConfigSaveError(f"Failed to save config to {path}: {e}")
    
    def supports_format(self, file_extension: str) -> bool:
        return file_extension.lower() == '.json'

class OmegaConfLoader(BaseConfigLoader):
    def load(self, path: Union[str, Path]) -> Dict[str, Any]:
        path = self._resolve_path(path)
        self._validate_path(path)
        
        try:
            config = OmegaConf.load(path)
            return OmegaConf.to_container(config, resolve=True) # type:ignore
        except Exception as e:
            raise ConfigLoadError(f"Failed to load config with OmegaConf from {path}: {e}")
    
    def save(self, config: Dict[str, Any], path: Union[str, Path]) -> None:
        path = self._resolve_path(path)
        self._validate_path(path, for_write=True)
        
        try:
            omega_config = OmegaConf.create(config)
            OmegaConf.save(omega_config, path)
            logger.debug(f"Saved OmegaConf config to {path}")
        except Exception as e:
            raise ConfigSaveError(f"Failed to save config with OmegaConf to {path}: {e}")
    
    def supports_format(self, file_extension: str) -> bool:
        return file_extension.lower() in ['.yaml', '.yml', '.json']

class ConfigValidator:
    def __init__(self):
        self.schemas: Dict[str, Dict[str, Any]] = {}
        self.custom_validators: Dict[str, Callable] = {}
    
    def register_schema(self, config_type: str, schema: Dict[str, Any]) -> None:
        self.schemas[config_type] = schema
        logger.debug(f"Registered schema for config type: {config_type}")
    
    def register_validator(self, name: str, validator: Callable) -> None:
        self.custom_validators[name] = validator
        logger.debug(f"Registered custom validator: {name}")
    
    def validate(self, config: Dict[str, Any], config_type: Optional[str] = None) -> bool:
        if config_type and config_type in self.schemas:
            return self._validate_against_schema(config, self.schemas[config_type])
        return self._basic_validation(config)
    
    def _validate_against_schema(self, config: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        errors = []
        
        for field, rules in schema.items():
            if rules.get('required', False) and field not in config:
                errors.append(f"Required field '{field}' is missing")
                continue
                
            if field in config:
                value = config[field]
                field_errors = self._validate_field(field, value, rules)
                errors.extend(field_errors)
        
        if errors:
            raise ConfigValidationError(f"Validation errors: {'; '.join(errors)}")
        
        return True
    
    def _validate_field(self, field: str, value: Any, rules: Dict[str, Any]) -> List[str]:
        errors = []
        
        if 'type' in rules:
            expected_type = rules['type']
            if not isinstance(value, expected_type):
                errors.append(f"Field '{field}' must be of type {expected_type.__name__}")
        
        if 'min_value' in rules and hasattr(value, '__lt__'):
            if value < rules['min_value']:
                errors.append(f"Field '{field}' must be >= {rules['min_value']}")
        
        if 'max_value' in rules and hasattr(value, '__gt__'):
            if value > rules['max_value']:
                errors.append(f"Field '{field}' must be <= {rules['max_value']}")
        
        if 'choices' in rules and value not in rules['choices']:
            errors.append(f"Field '{field}' must be one of {rules['choices']}")
        
        if 'custom_validator' in rules:
            validator_name = rules['custom_validator']
            if validator_name in self.custom_validators:
                try:
                    if not self.custom_validators[validator_name](value):
                        errors.append(f"Field '{field}' failed custom validation")
                except Exception as e:
                    errors.append(f"Field '{field}' custom validation error: {e}")
        
        return errors
    
    def _basic_validation(self, config: Dict[str, Any]) -> bool:
        if not isinstance(config, dict):
            raise ConfigValidationError("Configuration must be a dictionary")
        return True

class ConfigMerger:
    STRATEGIES = ['deep', 'override', 'list_append', 'list_replace']
    
    def merge(self, base: Dict[str, Any], override: Dict[str, Any], 
              strategy: str = 'deep') -> Dict[str, Any]:
        if strategy not in self.STRATEGIES:
            raise ConfigMergeError(f"Unknown merge strategy: {strategy}")
        
        method = getattr(self, f"_{strategy}_merge")
        return method(copy.deepcopy(base), override)
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in override.items():
            if key in base:
                if isinstance(base[key], dict) and isinstance(value, dict):
                    base[key] = self._deep_merge(base[key], value)
                else:
                    base[key] = value
            else:
                base[key] = value
        return base
    
    def _override_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        base.update(override)
        return base
    
    def _list_append_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in override.items():
            if key in base and isinstance(base[key], list) and isinstance(value, list):
                base[key].extend(value)
            else:
                base[key] = value
        return base
    
    def _list_replace_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        return self._override_merge(base, override)

class ConfigCache:
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache: Dict[str, Tuple[Dict[str, Any], float, str]] = {}
        self.access_order: List[str] = []
        self._lock = threading.Lock()
    
    def get(self, key: str, file_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
        with self._lock:
            if key in self.cache:
                config, cached_time, cached_hash = self.cache[key]
                
                if file_path and self._is_file_modified(file_path, cached_time, cached_hash):
                    del self.cache[key]
                    self.access_order.remove(key)
                    return None
                
                self.access_order.remove(key)
                self.access_order.append(key)
                return copy.deepcopy(config)
            return None
    
    def set(self, key: str, config: Dict[str, Any], file_path: Optional[Path] = None) -> None:
        with self._lock:
            if len(self.cache) >= self.max_size and key not in self.cache:
                oldest_key = self.access_order.pop(0)
                del self.cache[oldest_key]
            
            file_hash = ""
            if file_path and file_path.exists():
                file_hash = self._compute_file_hash(file_path)
            
            self.cache[key] = (copy.deepcopy(config), datetime.now().timestamp(), file_hash)
            
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
    
    def clear(self) -> None:
        with self._lock:
            self.cache.clear()
            self.access_order.clear()
    
    def _is_file_modified(self, file_path: Path, cached_time: float, cached_hash: str) -> bool:
        if not file_path.exists():
            return True
        
        file_mtime = file_path.stat().st_mtime
        if file_mtime > cached_time:
            current_hash = self._compute_file_hash(file_path)
            return current_hash != cached_hash
        return False
    
    def _compute_file_hash(self, file_path: Path) -> str:
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

class ConfigHistoryManager:
    def __init__(self, history_dir: str = "config_history"):
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)
    
    def save_version(self, config_name: str, config: Dict[str, Any], 
                    version_tag: Optional[str] = None) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_id = f"{timestamp}_{version_tag}" if version_tag else timestamp
        
        version_dir = self.history_dir / config_name
        version_dir.mkdir(parents=True, exist_ok=True)
        
        version_path = version_dir / f"{version_id}.yaml"
        
        with open(version_path, 'w') as f:
            yaml.safe_dump({
                'version_id': version_id,
                'timestamp': timestamp,
                'tag': version_tag,
                'config': config
            }, f, default_flow_style=False, indent=2)
        
        logger.info(f"Saved config version {version_id} for {config_name}")
        return version_id
    
    def load_version(self, config_name: str, version_id: str) -> Dict[str, Any]:
        version_path = self.history_dir / config_name / f"{version_id}.yaml"
        
        if not version_path.exists():
            raise ConfigLoadError(f"Version {version_id} not found for {config_name}")
        
        with open(version_path, 'r') as f:
            version_data = yaml.safe_load(f)
        
        return version_data['config']
    
    def list_versions(self, config_name: str) -> List[Dict[str, Any]]:
        version_dir = self.history_dir / config_name
        
        if not version_dir.exists():
            return []
        
        versions = []
        for version_file in version_dir.glob("*.yaml"):
            with open(version_file, 'r') as f:
                version_data = yaml.safe_load(f)
                versions.append({
                    'version_id': version_data['version_id'],
                    'timestamp': version_data['timestamp'],
                    'tag': version_data.get('tag'),
                    'file_path': str(version_file)
                })
        
        return sorted(versions, key=lambda x: x['timestamp'], reverse=True)
    
    def rollback(self, config_name: str, version_id: str, 
                target_path: Union[str, Path]) -> None:
        config = self.load_version(config_name, version_id)
        
        loader = YAMLConfigLoader()
        loader.save(config, target_path)
        
        logger.info(f"Rolled back {config_name} to version {version_id}")

class ConfigManager:
    def __init__(self, config_dir: str = "configs", use_cache: bool = True):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.loaders: Dict[str, ConfigLoaderInterface] = {
            '.yaml': YAMLConfigLoader(str(config_dir)),
            '.yml': YAMLConfigLoader(str(config_dir)),
            '.json': JSONConfigLoader(str(config_dir)),
        }
        
        self.validator = ConfigValidator()
        self.merger = ConfigMerger()
        self.cache = ConfigCache() if use_cache else None
        self.history = ConfigHistoryManager()
    
    def register_loader(self, extension: str, loader: ConfigLoaderInterface) -> None:
        self.loaders[extension] = loader
        logger.debug(f"Registered loader for {extension}")
    
    def load(self, config_path: Union[str, Path], use_cache: bool = True, 
             validate: bool = True, config_type: Optional[str] = None) -> Dict[str, Any]:
        config_path = Path(config_path)
        cache_key = str(config_path) if use_cache and self.cache else None
        
        if cache_key:
            cached_config = self.cache.get(cache_key, config_path) # type:ignore
            if cached_config is not None:
                logger.debug(f"Loaded config from cache: {config_path}")
                return cached_config
        
        loader = self._get_loader(config_path)
        config = loader.load(config_path)
        
        if validate:
            self.validator.validate(config, config_type)
        
        if cache_key:
            self.cache.set(cache_key, config, config_path) # type:ignore
        
        logger.info(f"Loaded config: {config_path}")
        return config
    
    def save(self, config: Dict[str, Any], config_path: Union[str, Path], 
             create_backup: bool = True, version_tag: Optional[str] = None) -> None:
        config_path = Path(config_path)
        
        if create_backup and config_path.exists():
            config_name = config_path.stem
            existing_config = self.load(config_path, use_cache=False, validate=False)
            self.history.save_version(config_name, existing_config, version_tag)
        
        loader = self._get_loader(config_path)
        loader.save(config, config_path)
        
        if self.cache:
            cache_key = str(config_path)
            self.cache.set(cache_key, config, config_path)
        
        logger.info(f"Saved config: {config_path}")
    
    def merge_configs(self, base_path: Union[str, Path], 
                     override_path: Union[str, Path],
                     strategy: str = 'deep') -> Dict[str, Any]:
        base_config = self.load(base_path, validate=False)
        override_config = self.load(override_path, validate=False)
        
        merged_config = self.merger.merge(base_config, override_config, strategy)
        logger.info(f"Merged configs: {base_path} + {override_path}")
        return merged_config
    
    def validate_config(self, config: Dict[str, Any], 
                       config_type: Optional[str] = None) -> bool:
        return self.validator.validate(config, config_type)
    
    def list_configs(self, pattern: str = "*.yaml") -> List[Path]:
        return list(self.config_dir.glob(pattern))
    
    def delete_config(self, config_path: Union[str, Path], 
                     create_backup: bool = True) -> None:
        config_path = Path(config_path)
        
        if create_backup and config_path.exists():
            config_name = config_path.stem
            config = self.load(config_path, use_cache=False, validate=False)
            self.history.save_version(config_name, config, "before_delete")
        
        config_path.unlink(missing_ok=True)
        
        if self.cache:
            cache_key = str(config_path)
            if cache_key in self.cache.cache:
                del self.cache.cache[cache_key]
                if cache_key in self.cache.access_order:
                    self.cache.access_order.remove(cache_key)
        
        logger.info(f"Deleted config: {config_path}")
    
    def _get_loader(self, config_path: Path) -> ConfigLoaderInterface:
        extension = config_path.suffix.lower()
        
        if extension not in self.loaders:
            raise ConfigError(f"No loader available for extension: {extension}")
        
        return self.loaders[extension]
    
    def clear_cache(self) -> None:
        if self.cache:
            self.cache.clear()
            logger.info("Cleared configuration cache")

def create_config_manager(config_dir: str = "configs", 
                         use_cache: bool = True) -> ConfigManager:
    return ConfigManager(config_dir, use_cache)

def load_config(config_path: Union[str, Path], 
               config_manager: Optional[ConfigManager] = None,
               **kwargs) -> Dict[str, Any]:
    if config_manager is None:
        config_manager = create_config_manager()
    return config_manager.load(config_path, **kwargs)

def save_config(config: Dict[str, Any], config_path: Union[str, Path], 
               config_manager: Optional[ConfigManager] = None,
               **kwargs) -> None:
    if config_manager is None:
        config_manager = create_config_manager()
    config_manager.save(config, config_path, **kwargs)

def merge_configs(base_path: Union[str, Path], override_path: Union[str, Path], 
                 strategy: str = 'deep',
                 config_manager: Optional[ConfigManager] = None) -> Dict[str, Any]:
    if config_manager is None:
        config_manager = create_config_manager()
    return config_manager.merge_configs(base_path, override_path, strategy)

def setup_config_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )