# FILE: configs/config_loader.py
# ehsanasgharzadeh - CONFIGURATION MANAGER


from omegaconf import OmegaConf, DictConfig
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
from pathlib import Path
import yaml
import json
import os
import logging
from dataclasses import asdict
import copy
from abc import ABC, abstractmethod
import datetime
import shutil
import sys

logger = logging.getLogger(__name__)

class ConfigError(Exception):
    pass

class ConfigValidationError(ConfigError):
    pass

class ConfigLoadError(ConfigError):
    pass

class ConfigSaveError(ConfigError):
    pass

class ConfigLoaderInterface(ABC):
    @abstractmethod
    def load(self, path: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def save(self, config: Dict[str, Any], path: str):
        pass

    @abstractmethod
    def supports_format(self, file_extension: str) -> bool:
        pass

class BaseConfigLoader(ConfigLoaderInterface):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def _ensure_directory(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def _handle_file_error(self, path: str, operation: str, error: Exception):
        error_msg = f"Failed to {operation} config at {path}: {error}"
        self.logger.error(error_msg)
        raise ConfigLoadError(error_msg) if operation == "load" else ConfigSaveError(error_msg)

class YAMLConfigLoader(BaseConfigLoader):
    def load(self, path: str) -> Dict[str, Any]:
        try:
            with open(path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file) or {}
                self.logger.info(f"Loaded YAML config from {path}")
                return config
        except (FileNotFoundError, yaml.YAMLError) as e:
            self._handle_file_error(path, "load", e)

    def save(self, config: Dict[str, Any], path: str):
        try:
            self._ensure_directory(path)
            with open(path, 'w', encoding='utf-8') as file:
                yaml.dump(config, file, default_flow_style=False, allow_unicode=True)
                self.logger.info(f"Saved YAML config to {path}")
        except OSError as e:
            self._handle_file_error(path, "save", e)

    def supports_format(self, file_extension: str) -> bool:
        return file_extension.lower() in ['.yaml', '.yml']

class JSONConfigLoader(BaseConfigLoader):
    def load(self, path: str) -> Dict[str, Any]:
        try:
            with open(path, 'r', encoding='utf-8') as file:
                config = json.load(file)
                self.logger.info(f"Loaded JSON config from {path}")
                return config
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self._handle_file_error(path, "load", e)

    def save(self, config: Dict[str, Any], path: str):
        try:
            self._ensure_directory(path)
            with open(path, 'w', encoding='utf-8') as file:
                json.dump(config, file, indent=4)
                self.logger.info(f"Saved JSON config to {path}")
        except OSError as e:
            self._handle_file_error(path, "save", e)

    def supports_format(self, file_extension: str) -> bool:
        return file_extension.lower() == '.json'

class OmegaConfLoader(BaseConfigLoader):
    def load(self, path: str) -> Dict[str, Any]:
        try:
            conf = OmegaConf.load(path)
            config = OmegaConf.to_container(conf, resolve=True)
            self.logger.info(f"Loaded OmegaConf config from {path}")
            return config #type: ignore 
        except (FileNotFoundError, Exception) as e:
            self._handle_file_error(path, "load", e)

    def save(self, config: Dict[str, Any], path: str):
        try:
            self._ensure_directory(path)
            conf = OmegaConf.create(config)
            OmegaConf.save(config=conf, f=path)
            self.logger.info(f"Saved OmegaConf config to {path}")
        except OSError as e:
            self._handle_file_error(path, "save", e)

    def supports_format(self, file_extension: str) -> bool:
        return file_extension.lower() in ['.yaml', '.yml', '.json']

class ConfigValidator:
    def __init__(self):
        self.schemas = {}
        self._last_errors = []
        self.logger = logging.getLogger(self.__class__.__name__)

    def add_schema(self, config_type: str, schema: Dict[str, Dict[str, Any]]):
        self.schemas[config_type] = schema
        self.logger.debug(f"Added schema for config type: {config_type}")

    def validate(self, config: Dict[str, Any], config_type: str = None) -> bool: #type: ignore 
        valid, _ = self.validate_with_errors(config, config_type)
        return valid

    def validate_with_errors(self, config: Dict[str, Any], config_type: str = None) -> Tuple[bool, List[str]]: #type: ignore 
        self._last_errors = []

        if config_type is None or config_type not in self.schemas:
            return True, []

        schema = self.schemas[config_type]
        self._validate_fields(config, schema)
        self._apply_custom_validators(config, schema)
        
        is_valid = len(self._last_errors) == 0
        if is_valid:
            self.logger.debug(f"Validation passed for config type: {config_type}")
        else:
            self.logger.warning(f"Validation failed for config type: {config_type}, errors: {self._last_errors}")
        
        return is_valid, self._last_errors

    def _validate_fields(self, config: Dict[str, Any], schema: Dict[str, Dict[str, Any]]):
        for field, rules in schema.items():
            if rules.get("required", False) and field not in config:
                self._last_errors.append(f"Missing required field: '{field}'")
                continue

            if field in config:
                self._validate_field_value(field, config[field], rules)

    def _validate_field_value(self, field: str, value: Any, rules: Dict[str, Any]):
        expected_type = rules.get("type")
        if expected_type and not isinstance(value, expected_type):
            self._last_errors.append(f"Field '{field}' expected type {expected_type.__name__}, got {type(value).__name__}")
            return

        if isinstance(value, (int, float)):
            self._validate_numeric_range(field, value, rules)

    def _validate_numeric_range(self, field: str, value: Union[int, float], rules: Dict[str, Any]):
        min_val = rules.get("min")
        max_val = rules.get("max")
        if min_val is not None and value < min_val:
            self._last_errors.append(f"Field '{field}' value {value} below minimum {min_val}")
        if max_val is not None and value > max_val:
            self._last_errors.append(f"Field '{field}' value {value} above maximum {max_val}")

    def _apply_custom_validators(self, config: Dict[str, Any], schema: Dict[str, Dict[str, Any]]):
        for field, rules in schema.items():
            custom_validator = rules.get("custom_validator")
            if callable(custom_validator):
                err = custom_validator(config)
                if err:
                    self._last_errors.append(err)

    def get_validation_errors(self) -> List[str]:
        return self._last_errors.copy()

class ConfigMerger:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def merge(self, base_config: Dict[str, Any], override_config: Dict[str, Any], strategy: str = 'deep_merge') -> Dict[str, Any]:
        merge_func = getattr(self, f'_{strategy}_merge', None)
        if not merge_func:
            raise ValueError(f"Unknown merge strategy: {strategy}")
        
        result = merge_func(base_config, override_config)
        self.logger.debug(f"Merged configs using {strategy} strategy")
        return result

    def _override_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        merged = base.copy()
        merged.update(override)
        return merged

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        merged = base.copy()
        for key, override_value in override.items():
            base_value = merged.get(key)
            if isinstance(base_value, dict) and isinstance(override_value, dict):
                merged[key] = self._deep_merge(base_value, override_value)
            else:
                merged[key] = override_value
        return merged

    def _list_append_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        merged = base.copy()
        for key, override_value in override.items():
            base_value = merged.get(key)
            if isinstance(base_value, list) and isinstance(override_value, list):
                merged[key] = base_value + override_value
            elif isinstance(base_value, dict) and isinstance(override_value, dict):
                merged[key] = self._deep_merge(base_value, override_value)
            else:
                merged[key] = override_value
        return merged

    def _list_replace_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        merged = base.copy()
        for key, override_value in override.items():
            base_value = merged.get(key)
            if isinstance(override_value, list):
                merged[key] = override_value
            elif isinstance(base_value, dict) and isinstance(override_value, dict):
                merged[key] = self._deep_merge(base_value, override_value)
            else:
                merged[key] = override_value
        return merged

class ConfigTemplateManager:
    def __init__(self, template_dir: str = "configs/templates"):
        self.template_dir = Path(template_dir)
        self.templates = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self._initialize_templates()

    def _initialize_templates(self):
        self.template_dir.mkdir(parents=True, exist_ok=True)
        self._load_templates()

    def _load_templates(self):
        self.templates.clear()
        for file_path in self.template_dir.glob("*.yaml"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = yaml.safe_load(f)
                    if content:
                        self.templates[file_path.stem] = content
                        self.logger.debug(f"Loaded template: {file_path.stem}")
            except Exception as e:
                self.logger.warning(f"Failed to load template {file_path.name}: {e}")

    def get_template(self, template_name: str) -> Dict[str, Any]:
        if template_name not in self.templates:
            raise KeyError(f"Template '{template_name}' not found")
        return copy.deepcopy(self.templates[template_name])

    def create_template(self, template_name: str, config: Dict[str, Any]):
        if template_name in self.templates:
            raise KeyError(f"Template '{template_name}' already exists")

        file_path = self.template_dir / f"{template_name}.yaml"
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, sort_keys=False)
            self.templates[template_name] = copy.deepcopy(config)
            self.logger.info(f"Created template: {template_name}")
        except Exception as e:
            self.logger.error(f"Failed to create template {template_name}: {e}")
            raise

    def list_templates(self) -> List[str]:
        return list(self.templates.keys())

    def instantiate_template(self, template_name: str, overrides: Dict[str, Any] = None) -> Dict[str, Any]: #type: ignore 
        config = self.get_template(template_name)
        if overrides:
            config.update(overrides)
        return config

class ConfigManager:
    def __init__(self, config_dir: str = "configs/"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.loaders = self._initialize_loaders()
        self.validator = ConfigValidator()
        self.merger = ConfigMerger()
        self.template_manager = ConfigTemplateManager(str(self.config_dir / "templates"))
        
        self.config_cache = {}
        self.cache_enabled = True
        self.logger = logging.getLogger(self.__class__.__name__)

    def _initialize_loaders(self) -> Dict[str, ConfigLoaderInterface]:
        return {
            '.yaml': YAMLConfigLoader(),
            '.yml': YAMLConfigLoader(),
            '.json': JSONConfigLoader()
        }

    def load(self, config_name: str, use_cache: bool = True) -> Dict[str, Any]:
        if use_cache and config_name in self.config_cache:
            self.logger.debug(f"Loading '{config_name}' from cache")
            return self.config_cache[config_name]

        config_file = self._find_config_file(config_name)
        if not config_file:
            raise FileNotFoundError(f"Configuration file for '{config_name}' not found")

        loader = self._get_loader(config_file)
        config = loader.load(str(config_file))
        
        if self.cache_enabled:
            self.config_cache[config_name] = config

        self.logger.info(f"Loaded configuration '{config_name}'")
        return config

    def save(self, config: Dict[str, Any], config_name: str, format: str = 'yaml'):
        loader = self._get_loader_by_format(format)
        file_path = self.config_dir / f"{config_name}.{format}"
        
        loader.save(config, str(file_path))
        
        if self.cache_enabled:
            self.config_cache[config_name] = copy.deepcopy(config)

    def validate(self, config: Dict[str, Any], config_type: str = None) -> bool: #type: ignore 
        return self.validator.validate(config, config_type)

    def merge(self, base_config_name: str, override_config_name: str, 
              output_name: str = None, strategy: str = 'deep_merge') -> Dict[str, Any]: #type: ignore 
        base_config = self.load(base_config_name)
        override_config = self.load(override_config_name)
        
        merged = self.merger.merge(base_config, override_config, strategy)
        
        if output_name:
            self.save(merged, output_name)
        
        return merged

    def create_from_template(self, template_name: str, config_name: str, 
                           overrides: Dict[str, Any] = None) -> Dict[str, Any]: #type: ignore 
        config = self.template_manager.instantiate_template(template_name, overrides)
        self.save(config, config_name)
        return config

    def list_configs(self) -> List[str]:
        configs = []
        for file_path in self.config_dir.glob("*.*"):
            if file_path.is_file() and file_path.suffix in self.loaders:
                configs.append(file_path.stem)
        return configs

    def delete_config(self, config_name: str):
        deleted = False
        for ext in self.loaders.keys():
            file_path = self.config_dir / f"{config_name}{ext}"
            if file_path.exists():
                file_path.unlink()
                deleted = True
                self.logger.info(f"Deleted configuration file: {file_path}")
        
        if config_name in self.config_cache:
            del self.config_cache[config_name]
        
        if not deleted:
            self.logger.warning(f"Configuration '{config_name}' not found for deletion")

    def backup_config(self, config_name: str, backup_suffix: str = None): #type: ignore 
        config_file = self._find_config_file(config_name)
        if not config_file:
            raise FileNotFoundError(f"Configuration file '{config_name}' not found for backup")

        suffix = backup_suffix or datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        backup_name = f"{config_file.stem}_backup_{suffix}{config_file.suffix}"
        backup_path = self.config_dir / backup_name

        shutil.copy2(config_file, backup_path)
        self.logger.info(f"Backed up '{config_name}' to '{backup_name}'")

    def get_config_info(self, config_name: str) -> Dict[str, Any]:
        config_file = self._find_config_file(config_name)
        if not config_file:
            raise FileNotFoundError(f"Configuration file '{config_name}' not found")

        stat = config_file.stat()
        return {
            "file_name": config_file.name,
            "file_path": str(config_file),
            "size_bytes": stat.st_size,
            "created": datetime.datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "validation_status": self._get_validation_status(config_name)
        }

    def clear_cache(self):
        self.config_cache.clear()
        self.logger.info("Configuration cache cleared")

    def set_cache_enabled(self, enabled: bool):
        self.cache_enabled = enabled
        self.logger.info(f"Configuration cache enabled: {enabled}")

    def _find_config_file(self, config_name: str) -> Optional[Path]:
        for ext in self.loaders.keys():
            candidate = self.config_dir / f"{config_name}{ext}"
            if candidate.exists():
                return candidate
        return None

    def _get_loader(self, config_file: Path) -> ConfigLoaderInterface:
        ext = config_file.suffix.lower()
        loader = self.loaders.get(ext)
        if not loader:
            raise ValueError(f"No loader registered for extension '{ext}'")
        return loader

    def _get_loader_by_format(self, format: str) -> ConfigLoaderInterface:
        ext = f".{format.lower()}"
        loader = self.loaders.get(ext)
        if not loader:
            raise ValueError(f"No loader registered for format '{format}'")
        return loader

    def _get_validation_status(self, config_name: str) -> bool:
        try:
            config = self.load(config_name)
            return self.validate(config, config_name)
        except Exception as e:
            self.logger.error(f"Failed to validate config '{config_name}': {e}")
            return False

class ConfigHistoryManager:
    def __init__(self, history_dir: str = "configs/history"):
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)

    def save_version(self, config_name: str, config: Dict[str, Any], message: str = ""):
        config_history_dir = self.history_dir / config_name
        config_history_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        version_file = config_history_dir / f"{timestamp}.json"
        metadata_file = config_history_dir / f"{timestamp}.meta.json"
        
        with open(version_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)
        
        metadata = {
            "timestamp": timestamp,
            "message": message,
            "version_file": str(version_file.name)
        }
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)
        
        self.logger.info(f"Saved version '{timestamp}' for config '{config_name}'")

    def get_version(self, config_name: str, version: str) -> Dict[str, Any]:
        config_history_dir = self.history_dir / config_name
        version_file = config_history_dir / f"{version}.json"
        
        if not version_file.exists():
            raise FileNotFoundError(f"Version '{version}' not found for config '{config_name}'")
        
        with open(version_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def list_versions(self, config_name: str) -> List[Dict[str, Any]]:
        config_history_dir = self.history_dir / config_name
        if not config_history_dir.exists():
            return []
        
        versions = []
        for meta_path in sorted(config_history_dir.glob("*.meta.json"), reverse=True):
            with open(meta_path, 'r', encoding='utf-8') as f:
                versions.append(json.load(f))
        return versions

    def rollback(self, config_name: str, version: str, target_path: Path):
        config = self.get_version(config_name, version)
        
        with open(target_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)
        
        self.logger.info(f"Rolled back config '{config_name}' to version '{version}'")

class ConfigComparator:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def compare(self, config1: Dict[str, Any], config2: Dict[str, Any]) -> Dict[str, Any]:
        return self._recursive_diff(config1, config2)

    def _recursive_diff(self, d1: Any, d2: Any) -> Dict[str, Any]:
        if isinstance(d1, dict) and isinstance(d2, dict):
            return self._compare_dicts(d1, d2)
        elif isinstance(d1, list) and isinstance(d2, list):
            return {'from': d1, 'to': d2} if d1 != d2 else {}
        else:
            return {'from': d1, 'to': d2} if d1 != d2 else {}

    def _compare_dicts(self, d1: Dict[str, Any], d2: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        keys1, keys2 = set(d1.keys()), set(d2.keys())
        
        removed = {key: d1[key] for key in keys1 - keys2}
        added = {key: d2[key] for key in keys2 - keys1}
        
        changed = {}
        for key in keys1 & keys2:
            diff = self._recursive_diff(d1[key], d2[key])
            if diff:
                changed[key] = diff
        
        if removed: result['removed'] = removed
        if added: result['added'] = added
        if changed: result['changed'] = changed
        
        return result

    def generate_diff_report(self, config1: Dict[str, Any], config2: Dict[str, Any]) -> str:
        diff = self.compare(config1, config2)
        if not diff:
            return "No differences found"
        
        lines = []
        self._format_diff(diff, lines)
        return "\n".join(lines)

    def _format_diff(self, diff: Dict[str, Any], lines: List[str], prefix: str = ""):
        for key, value in diff.items():
            if key == 'added':
                for k, v in value.items():
                    lines.append(f"{prefix}+ Added '{k}': {v}")
            elif key == 'removed':
                for k, v in value.items():
                    lines.append(f"{prefix}- Removed '{k}': {v}")
            elif key == 'changed':
                for k, v in value.items():
                    if isinstance(v, dict) and 'from' in v and 'to' in v:
                        lines.append(f"{prefix}* Changed '{k}': {v['from']} -> {v['to']}")
                    else:
                        lines.append(f"{prefix}* Changed '{k}':")
                        self._format_diff(v, lines, prefix + "  ")

def load_config(path: str) -> Dict[str, Any]:
    loader = _get_loader_by_path(path)
    return loader.load(path)

def save_config(config: Dict[str, Any], path: str):
    loader = _get_loader_by_path(path)
    loader.save(config, path)

def validate_config_file(path: str) -> bool:
    try:
        config = load_config(path)
        return _validator.validate(config)
    except Exception as e:
        logger.error(f"Failed to validate config file '{path}': {e}")
        return False

def merge_config_files(base_path: str, override_path: str, output_path: str = None) -> Dict[str, Any]: #type: ignore     
    base_config = load_config(base_path)
    override_config = load_config(override_path)
    merged = _merger.merge(base_config, override_config)
    
    if output_path:
        save_config(merged, output_path)
    
    return merged

def _get_loader_by_path(path: str) -> ConfigLoaderInterface:
    ext = Path(path).suffix.lower()
    loader = _loaders.get(ext)
    if not loader:
        raise ValueError(f"No loader available for extension '{ext}'")
    return loader

def setup_config_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

_loaders = {
    '.yaml': YAMLConfigLoader(),
    '.yml': YAMLConfigLoader(),
    '.json': JSONConfigLoader()
}

_validator = ConfigValidator()
_merger = ConfigMerger()
config_manager = ConfigManager()