# FILE: configs/config_loader.py
# ehsanasgharzadeh - CONFIGURATION MANAGER
# hosseinsolymanzadeh - PROPER COMMENTING

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

# Base exception for all configuration-related errors
class ConfigError(Exception):
    pass


# Raised when configuration validation fails
class ConfigValidationError(ConfigError):
    pass


# Raised when loading the configuration file fails
class ConfigLoadError(ConfigError):
    pass


# Raised when saving the configuration file fails
class ConfigSaveError(ConfigError):
    pass


# Import the abstract base class functionality
class ConfigLoaderInterface(ABC):
    # Abstract method to load configuration from a file path
    @abstractmethod
    def load(self, path: str) -> Dict[str, Any]:
        pass

    # Abstract method to save a configuration dictionary to a file path
    @abstractmethod
    def save(self, config: Dict[str, Any], path: str):
        pass
    
    # Abstract method to check if a specific file extension is supported
    @abstractmethod
    def supports_format(self, file_extension: str) -> bool:
        pass


class BaseConfigLoader(ConfigLoaderInterface):
    def __init__(self):
        # Initialize a logger using the name of the class
        self.logger = logging.getLogger(self.__class__.__name__)

    def _ensure_directory(self, path: str):
        # Create parent directories for the given path if they don't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def _handle_file_error(self, path: str, operation: str, error: Exception):
        # Format an error message based on the operation (load/save) and log it
        error_msg = f"Failed to {operation} config at {path}: {error}"
        self.logger.error(error_msg)
        # Raise a specific exception based on the operation type
        raise ConfigLoadError(error_msg) if operation == "load" else ConfigSaveError(error_msg)


class YAMLConfigLoader(BaseConfigLoader):
    def load(self, path: str) -> Dict[str, Any]:
        try:
            # Open the YAML file for reading
            with open(path, 'r', encoding='utf-8') as file:
                # Load the YAML content into a dictionary
                config = yaml.safe_load(file) or {}
                # Log successful loading
                self.logger.info(f"Loaded YAML config from {path}")
                return config
        except (FileNotFoundError, yaml.YAMLError) as e:
            # Handle file not found or YAML parsing errors
            self._handle_file_error(path, "load", e)

    def save(self, config: Dict[str, Any], path: str):
        try:
            # Ensure the target directory exists
            self._ensure_directory(path)
            # Open the file for writing the YAML config
            with open(path, 'w', encoding='utf-8') as file:
                # Dump the config dictionary to YAML format
                yaml.dump(config, file, default_flow_style=False, allow_unicode=True)
                # Log successful saving
                self.logger.info(f"Saved YAML config to {path}")
        except OSError as e:
            # Handle file writing-related errors
            self._handle_file_error(path, "save", e)

    def supports_format(self, file_extension: str) -> bool:
        # Check if the file extension is a supported YAML format
        return file_extension.lower() in ['.yaml', '.yml']


class JSONConfigLoader(BaseConfigLoader):
    # Load a JSON configuration file from the given path
    def load(self, path: str) -> Dict[str, Any]:
        try:
            # Open the file in read mode with UTF-8 encoding
            with open(path, 'r', encoding='utf-8') as file:
                # Parse JSON content from the file
                config = json.load(file)
                # Log successful loading
                self.logger.info(f"Loaded JSON config from {path}")
                return config
        except (FileNotFoundError, json.JSONDecodeError) as e:
            # Handle errors related to file not found or invalid JSON format
            self._handle_file_error(path, "load", e)

    # Save a configuration dictionary to a JSON file at the specified path
    def save(self, config: Dict[str, Any], path: str):
        try:
            # Ensure the target directory exists
            self._ensure_directory(path)
            # Open the file in write mode with UTF-8 encoding
            with open(path, 'w', encoding='utf-8') as file:
                # Write the config dictionary as JSON with indentation
                json.dump(config, file, indent=4)
                # Log successful saving
                self.logger.info(f"Saved JSON config to {path}")
        except OSError as e:
            # Handle I/O-related errors
            self._handle_file_error(path, "save", e)

    # Check if the loader supports the given file extension (should be .json)
    def supports_format(self, file_extension: str) -> bool:
        return file_extension.lower() == '.json'


class OmegaConfLoader(BaseConfigLoader):
    def load(self, path: str) -> Dict[str, Any]:
        try:
            # Load configuration file using OmegaConf
            conf = OmegaConf.load(path)
            # Convert the OmegaConf object to a plain dictionary (with variable interpolation resolved)
            config = OmegaConf.to_container(conf, resolve=True)
            # Log successful loading
            self.logger.info(f"Loaded OmegaConf config from {path}")
            return config  # type: ignore
        except (FileNotFoundError, Exception) as e:
            # Handle file-related errors gracefully
            self._handle_file_error(path, "load", e)

    def save(self, config: Dict[str, Any], path: str):
        try:
            # Ensure the target directory exists
            self._ensure_directory(path)
            # Convert dictionary to OmegaConf object
            conf = OmegaConf.create(config)
            # Save the OmegaConf object to the specified path
            OmegaConf.save(config=conf, f=path)
            # Log successful saving
            self.logger.info(f"Saved OmegaConf config to {path}")
        except OSError as e:
            # Handle file-related errors during save
            self._handle_file_error(path, "save", e)

    def supports_format(self, file_extension: str) -> bool:
        # Check if the file extension is one of the supported formats
        return file_extension.lower() in ['.yaml', '.yml', '.json']


class ConfigValidator:
    def __init__(self):
        self.schemas = {}  # Dictionary to hold schemas for different config types
        self._last_errors = []  # Stores the last validation errors
        self.logger = logging.getLogger(self.__class__.__name__)  # Logger for this class

    def add_schema(self, config_type: str, schema: Dict[str, Dict[str, Any]]):
        # Add a schema for a specific config type
        self.schemas[config_type] = schema
        self.logger.debug(f"Added schema for config type: {config_type}")

    def validate(self, config: Dict[str, Any], config_type: str = None) -> bool:  # type: ignore 
        # Perform validation and return only the result (True/False)
        valid, _ = self.validate_with_errors(config, config_type)
        return valid

    def validate_with_errors(self, config: Dict[str, Any], config_type: str = None) -> Tuple[bool, List[str]]:  # type: ignore 
        # Clear previous errors
        self._last_errors = []

        # If config_type is not provided or no schema exists for it, consider config valid
        if config_type is None or config_type not in self.schemas:
            return True, []

        # Retrieve schema for the given config type
        schema = self.schemas[config_type]
        # Validate fields based on schema definitions
        self._validate_fields(config, schema)
        # Apply any custom validation logic defined in schema
        self._apply_custom_validators(config, schema)
        
        # Determine if validation passed based on whether errors were collected
        is_valid = len(self._last_errors) == 0
        # Log the result of validation
        if is_valid:
            self.logger.debug(f"Validation passed for config type: {config_type}")
        else:
            self.logger.warning(f"Validation failed for config type: {config_type}, errors: {self._last_errors}")
        
        # Return validation result and list of errors (if any)
        return is_valid, self._last_errors

    def _validate_fields(self, config: Dict[str, Any], schema: Dict[str, Dict[str, Any]]):
        # Iterate over all fields defined in the schema
        for field, rules in schema.items():
            # Check if the field is required and missing from the config
            if rules.get("required", False) and field not in config:
                self._last_errors.append(f"Missing required field: '{field}'")
                continue

            # If the field exists in the config, validate its value
            if field in config:
                self._validate_field_value(field, config[field], rules)

    def _validate_field_value(self, field: str, value: Any, rules: Dict[str, Any]):
        # Get the expected type for the field
        expected_type = rules.get("type")
        # Check if the value matches the expected type
        if expected_type and not isinstance(value, expected_type):
            self._last_errors.append(
                f"Field '{field}' expected type {expected_type.__name__}, got {type(value).__name__}"
            )
            return

        # If the value is numeric, validate its range
        if isinstance(value, (int, float)):
            self._validate_numeric_range(field, value, rules)


    def _validate_numeric_range(self, field: str, value: Union[int, float], rules: Dict[str, Any]):
        # Get optional minimum and maximum values
        min_val = rules.get("min")
        max_val = rules.get("max")
        # Check if the value is below the allowed minimum
        if min_val is not None and value < min_val:
            self._last_errors.append(f"Field '{field}' value {value} below minimum {min_val}")
        # Check if the value is above the allowed maximum
        if max_val is not None and value > max_val:
            self._last_errors.append(f"Field '{field}' value {value} above maximum {max_val}")

    def _apply_custom_validators(self, config: Dict[str, Any], schema: Dict[str, Dict[str, Any]]):
        # Apply any custom validation functions defined in the schema
        for field, rules in schema.items():
            custom_validator = rules.get("custom_validator")
            # If a callable custom validator is provided, execute it
            if callable(custom_validator):
                err = custom_validator(config)
                # If the validator returns an error message, record it
                if err:
                    self._last_errors.append(err)

    def get_validation_errors(self) -> List[str]:
        # Return a copy of the list of validation errors
        return self._last_errors.copy()


class ConfigMerger:
    def __init__(self):
        # Initialize a logger using the class name
        self.logger = logging.getLogger(self.__class__.__name__)

    def merge(self, base_config: Dict[str, Any], override_config: Dict[str, Any], strategy: str = 'deep_merge') -> Dict[str, Any]:
        # Get the merge function based on the specified strategy
        merge_func = getattr(self, f'_{strategy}_merge', None)
        # Raise an error if the strategy is unknown
        if not merge_func:
            raise ValueError(f"Unknown merge strategy: {strategy}")
        # Merge the configurations using the selected strategy
        result = merge_func(base_config, override_config)
        # Log the merge operation
        self.logger.debug(f"Merged configs using {strategy} strategy")
        # Return the merged configuration
        return result

    def _override_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        # Create a copy of the base config
        merged = base.copy()
        # Override base config values with those from override config
        merged.update(override)
        # Return the result
        return merged

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        # Create a copy of the base config to avoid mutating the original
        merged = base.copy()
        # Iterate through each key-value pair in the override config
        for key, override_value in override.items():
            base_value = merged.get(key)
            # Recursively merge if both values are dictionaries
            if isinstance(base_value, dict) and isinstance(override_value, dict):
                merged[key] = self._deep_merge(base_value, override_value)
            else:
                # Otherwise, override the base value
                merged[key] = override_value
        # Return the merged result
        return merged

    def _list_append_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        # Create a copy of the base config
        merged = base.copy()
        # Iterate through each key-value pair in the override config
        for key, override_value in override.items():
            base_value = merged.get(key)
            # If both base and override values are lists, append override to base
            if isinstance(base_value, list) and isinstance(override_value, list):
                merged[key] = base_value + override_value
            # If both are dicts, perform a deep merge
            elif isinstance(base_value, dict) and isinstance(override_value, dict):
                merged[key] = self._deep_merge(base_value, override_value)
            else:
                # Otherwise, override the base value
                merged[key] = override_value
        # Return the merged result
        return merged

    def _list_replace_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        # Create a copy of the base config
        merged = base.copy()
        # Iterate through each key-value pair in the override config
        for key, override_value in override.items():
            base_value = merged.get(key)
            # If the override value is a list, replace the base list entirely
            if isinstance(override_value, list):
                merged[key] = override_value
            # If both are dicts, perform a deep merge
            elif isinstance(base_value, dict) and isinstance(override_value, dict):
                merged[key] = self._deep_merge(base_value, override_value)
            else:
                # Otherwise, override the base value
                merged[key] = override_value
        # Return the merged result
        return merged


class ConfigTemplateManager:
    def __init__(self, template_dir: str = "configs/templates"):
        # Store the path to the template directory as a Path object
        self.template_dir = Path(template_dir)
        # Dictionary to hold loaded templates keyed by template name
        self.templates = {}
        # Logger instance for this class
        self.logger = logging.getLogger(self.__class__.__name__)
        # Initialize templates by loading them from the directory
        self._initialize_templates()

    def _initialize_templates(self):
        # Create template directory if it doesn't exist
        self.template_dir.mkdir(parents=True, exist_ok=True)
        # Load templates from the directory
        self._load_templates()

    def _load_templates(self):
        # Clear any existing templates in memory
        self.templates.clear()
        # Iterate over all YAML files in the template directory
        for file_path in self.template_dir.glob("*.yaml"):
            try:
                # Open and load the YAML content from the file
                with open(file_path, "r", encoding="utf-8") as f:
                    content = yaml.safe_load(f)
                    # If content is valid, store it with filename stem as key
                    if content:
                        self.templates[file_path.stem] = content
                        # Log debug message about successful loading
                        self.logger.debug(f"Loaded template: {file_path.stem}")
            except Exception as e:
                # Log a warning if loading fails for any file
                self.logger.warning(f"Failed to load template {file_path.name}: {e}")

    def get_template(self, template_name: str) -> Dict[str, Any]:
        # Raise error if requested template does not exist
        if template_name not in self.templates:
            raise KeyError(f"Template '{template_name}' not found")
        # Return a deep copy of the requested template to avoid side effects
        return copy.deepcopy(self.templates[template_name])

    def create_template(self, template_name: str, config: Dict[str, Any]):
        # Prevent overwriting existing templates
        if template_name in self.templates:
            raise KeyError(f"Template '{template_name}' already exists")

        # Construct the full path for the new template file
        file_path = self.template_dir / f"{template_name}.yaml"
        try:
            # Write the provided config dictionary to a YAML file
            with open(file_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, sort_keys=False)
            # Store a deep copy of the config in the templates dictionary
            self.templates[template_name] = copy.deepcopy(config)
            # Log info about successful creation
            self.logger.info(f"Created template: {template_name}")
        except Exception as e:
            # Log an error and re-raise exception if file write fails
            self.logger.error(f"Failed to create template {template_name}: {e}")
            raise

    def list_templates(self) -> List[str]:
        # Return a list of all template names currently loaded
        return list(self.templates.keys())

    def instantiate_template(self, template_name: str, overrides: Dict[str, Any] = None) -> Dict[str, Any]: #type: ignore 
        # Retrieve a deep copy of the template by name
        config = self.get_template(template_name)
        # Apply any overrides to the config dictionary if provided
        if overrides:
            config.update(overrides)
        # Return the resulting config dictionary
        return config


class ConfigManager:
    def __init__(self, config_dir: str = "configs/"):
        # Initialize the path to the configuration directory
        self.config_dir = Path(config_dir)
        # Create the configuration directory if it doesn't exist (including parent dirs)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize loaders for different config file types
        self.loaders = self._initialize_loaders()
        # Validator instance for config validation
        self.validator = ConfigValidator()
        # Merger instance for merging configs
        self.merger = ConfigMerger()
        # Manager for config templates located inside a 'templates' subfolder
        self.template_manager = ConfigTemplateManager(str(self.config_dir / "templates"))
        
        # Cache dictionary to store loaded configs for reuse
        self.config_cache = {}
        # Flag to enable or disable caching
        self.cache_enabled = True
        # Logger instance for this class
        self.logger = logging.getLogger(self.__class__.__name__)

    def _initialize_loaders(self) -> Dict[str, ConfigLoaderInterface]:
        # Return a dictionary mapping file extensions to their respective config loaders
        return {
            '.yaml': YAMLConfigLoader(),
            '.yml': YAMLConfigLoader(),
            '.json': JSONConfigLoader()
        }

    def load(self, config_name: str, use_cache: bool = True) -> Dict[str, Any]:
        # If caching is enabled and the config is cached, return cached config
        if use_cache and config_name in self.config_cache:
            self.logger.debug(f"Loading '{config_name}' from cache")
            return self.config_cache[config_name]

        # Find the config file path based on the config name
        config_file = self._find_config_file(config_name)
        # Raise error if config file is not found
        if not config_file:
            raise FileNotFoundError(f"Configuration file for '{config_name}' not found")

        # Select the appropriate loader based on the config file extension
        loader = self._get_loader(config_file)
        # Load the config data from file
        config = loader.load(str(config_file))
        
        # Cache the loaded config if caching is enabled
        if self.cache_enabled:
            self.config_cache[config_name] = config

        # Log that the config has been loaded successfully
        self.logger.info(f"Loaded configuration '{config_name}'")
        return config

    def save(self, config: Dict[str, Any], config_name: str, format: str = 'yaml'):
     # Get the appropriate loader for the given file format
     loader = self._get_loader_by_format(format)
     # Construct the full file path with config directory and extension
     file_path = self.config_dir / f"{config_name}.{format}"

     # Save the config data to the file using the loader
     loader.save(config, str(file_path))

     # If caching is enabled, update the cache with a deep copy of the config
     if self.cache_enabled:
         self.config_cache[config_name] = copy.deepcopy(config)

    def validate(self, config: Dict[str, Any], config_type: str = None) -> bool:  # type: ignore
        # Validate the given config using the validator; optionally specify config type
        return self.validator.validate(config, config_type)

    def merge(self, base_config_name: str, override_config_name: str, 
              output_name: str = None, strategy: str = 'deep_merge') -> Dict[str, Any]:  # type: ignore
        # Load the base configuration by name
        base_config = self.load(base_config_name)
        # Load the overriding configuration by name
        override_config = self.load(override_config_name)

        # Merge the two configurations according to the given strategy
        merged = self.merger.merge(base_config, override_config, strategy)

        # If an output name is provided, save the merged config under that name
        if output_name:
            self.save(merged, output_name)

        # Return the merged configuration dictionary
        return merged

    def create_from_template(self, template_name: str, config_name: str, 
                           overrides: Dict[str, Any] = None) -> Dict[str, Any]:  # type: ignore
        # Instantiate a config from the given template, applying optional overrides
        config = self.template_manager.instantiate_template(template_name, overrides)
        # Save the instantiated config with the provided config name
        self.save(config, config_name)
        # Return the created config dictionary
        return config

    def list_configs(self) -> List[str]:
        # Initialize empty list to collect config names
        configs = []
        # Iterate over all files in the config directory with any extension
        for file_path in self.config_dir.glob("*.*"):
            # If it's a file and has an extension corresponding to a loader, add its stem (filename without extension)
            if file_path.is_file() and file_path.suffix in self.loaders:
                configs.append(file_path.stem)
        # Return the list of config names available in the config directory
        return configs

    def delete_config(self, config_name: str):
        # Flag to track if any config file was deleted
        deleted = False
        # Iterate over all supported file extensions to find matching config files
        for ext in self.loaders.keys():
            file_path = self.config_dir / f"{config_name}{ext}"
            # If the config file exists, delete it
            if file_path.exists():
                file_path.unlink()
                deleted = True
                self.logger.info(f"Deleted configuration file: {file_path}")
        
        # Remove config from cache if present
        if config_name in self.config_cache:
            del self.config_cache[config_name]
        
        # Log a warning if no config file was found to delete
        if not deleted:
            self.logger.warning(f"Configuration '{config_name}' not found for deletion")

    def backup_config(self, config_name: str, backup_suffix: str = None): #type: ignore 
        # Find the config file path to back up
        config_file = self._find_config_file(config_name)
        # Raise error if config file not found
        if not config_file:
            raise FileNotFoundError(f"Configuration file '{config_name}' not found for backup")

        # Use provided suffix or current timestamp as backup suffix
        suffix = backup_suffix or datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        # Construct backup file name with suffix
        backup_name = f"{config_file.stem}_backup_{suffix}{config_file.suffix}"
        backup_path = self.config_dir / backup_name

        # Copy original config file to backup file
        shutil.copy2(config_file, backup_path)
        self.logger.info(f"Backed up '{config_name}' to '{backup_name}'")

    def get_config_info(self, config_name: str) -> Dict[str, Any]:
        # Find the config file path
        config_file = self._find_config_file(config_name)
        # Raise error if config file not found
        if not config_file:
            raise FileNotFoundError(f"Configuration file '{config_name}' not found")

        # Retrieve file system stats for the config file
        stat = config_file.stat()
        # Return a dictionary with file metadata and validation status
        return {
            "file_name": config_file.name,
            "file_path": str(config_file),
            "size_bytes": stat.st_size,
            "created": datetime.datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "validation_status": self._get_validation_status(config_name)
        }

    def clear_cache(self):
        # Clear all entries from the config cache
        self.config_cache.clear()
        self.logger.info("Configuration cache cleared")

    def set_cache_enabled(self, enabled: bool):
        # Enable or disable caching based on input flag
        self.cache_enabled = enabled
        self.logger.info(f"Configuration cache enabled: {enabled}")

    def _find_config_file(self, config_name: str) -> Optional[Path]:
        # Search for a config file with any supported extension in the config directory
        for ext in self.loaders.keys():
            candidate = self.config_dir / f"{config_name}{ext}"
            if candidate.exists():
                return candidate
        # Return None if no file found
        return None

    def _get_loader(self, config_file: Path) -> ConfigLoaderInterface:
        # Determine loader based on file extension
        ext = config_file.suffix.lower()
        loader = self.loaders.get(ext)
        # Raise error if no loader registered for this extension
        if not loader:
            raise ValueError(f"No loader registered for extension '{ext}'")
        return loader

    def _get_loader_by_format(self, format: str) -> ConfigLoaderInterface:
        # Determine loader based on format string (e.g. 'yaml', 'json')
        ext = f".{format.lower()}"
        loader = self.loaders.get(ext)
        # Raise error if no loader registered for this format
        if not loader:
            raise ValueError(f"No loader registered for format '{format}'")
        return loader

    def _get_validation_status(self, config_name: str) -> bool:
        try:
            # Attempt to load and validate the configuration
            config = self.load(config_name)
            return self.validate(config, config_name)
        except Exception as e:
            # Log error and return False if validation fails
            self.logger.error(f"Failed to validate config '{config_name}': {e}")
            return False


class ConfigHistoryManager:
    def __init__(self, history_dir: str = "configs/history"):
        # Set the base directory where config history will be saved
        self.history_dir = Path(history_dir)
        # Create the directory (and parents) if it doesn't exist
        self.history_dir.mkdir(parents=True, exist_ok=True)
        # Initialize a logger for this class
        self.logger = logging.getLogger(self.__class__.__name__)

    def save_version(self, config_name: str, config: Dict[str, Any], message: str = ""):
        # Directory for the specific config's history versions
        config_history_dir = self.history_dir / config_name
        # Ensure the directory exists
        config_history_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a timestamp string to use as version identifier
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        # Path for saving the actual config JSON
        version_file = config_history_dir / f"{timestamp}.json"
        # Path for saving metadata about the version
        metadata_file = config_history_dir / f"{timestamp}.meta.json"
        
        # Write the config dictionary to the version file in JSON format
        with open(version_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)
        
        # Prepare metadata including timestamp, message, and version filename
        metadata = {
            "timestamp": timestamp,
            "message": message,
            "version_file": str(version_file.name)
        }
        # Write metadata to the metadata file in JSON format
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)
        
        # Log info that a new version has been saved
        self.logger.info(f"Saved version '{timestamp}' for config '{config_name}'")

    def get_version(self, config_name: str, version: str) -> Dict[str, Any]:
        # Directory for the specific config's history versions
        config_history_dir = self.history_dir / config_name
        # Path to the specific version JSON file
        version_file = config_history_dir / f"{version}.json"
        
        # Raise error if version file doesn't exist
        if not version_file.exists():
            raise FileNotFoundError(f"Version '{version}' not found for config '{config_name}'")
        
        # Load and return the JSON content of the version file
        with open(version_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def list_versions(self, config_name: str) -> List[Dict[str, Any]]:
        # Directory for the specific config's history versions
        config_history_dir = self.history_dir / config_name
        # Return empty list if config history directory does not exist
        if not config_history_dir.exists():
            return []
        
        versions = []
        # Iterate over all metadata files sorted in reverse chronological order
        for meta_path in sorted(config_history_dir.glob("*.meta.json"), reverse=True):
            # Load metadata JSON and append to versions list
            with open(meta_path, 'r', encoding='utf-8') as f:
                versions.append(json.load(f))
        # Return list of metadata dicts for all saved versions
        return versions

    def rollback(self, config_name: str, version: str, target_path: Path):
        # Retrieve the specific version of the config
        config = self.get_version(config_name, version)
        
        # Write the config content back to the specified target file path
        with open(target_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)
        
        # Log info about the rollback operation
        self.logger.info(f"Rolled back config '{config_name}' to version '{version}'")


class ConfigComparator:
    def __init__(self):
        # Initialize logger named after this class
        self.logger = logging.getLogger(self.__class__.__name__)

    def compare(self, config1: Dict[str, Any], config2: Dict[str, Any]) -> Dict[str, Any]:
        # Public method to start recursive comparison between two configs
        return self._recursive_diff(config1, config2)

    def _recursive_diff(self, d1: Any, d2: Any) -> Dict[str, Any]:
        # Recursively compare two data structures, dict or list or other
        if isinstance(d1, dict) and isinstance(d2, dict):
            # If both are dicts, compare dictionaries
            return self._compare_dicts(d1, d2)
        elif isinstance(d1, list) and isinstance(d2, list):
            # If both are lists, return diff only if different
            return {'from': d1, 'to': d2} if d1 != d2 else {}
        else:
            # For any other type, return diff only if different
            return {'from': d1, 'to': d2} if d1 != d2 else {}

    def _compare_dicts(self, d1: Dict[str, Any], d2: Dict[str, Any]) -> Dict[str, Any]:
        # Compare two dicts and categorize keys into removed, added, and changed
        result = {}
        keys1, keys2 = set(d1.keys()), set(d2.keys())
        
        # Keys present in d1 but not in d2 => removed
        removed = {key: d1[key] for key in keys1 - keys2}
        # Keys present in d2 but not in d1 => added
        added = {key: d2[key] for key in keys2 - keys1}
        
        # Keys present in both, check if their values differ recursively
        changed = {}
        for key in keys1 & keys2:
            diff = self._recursive_diff(d1[key], d2[key])
            if diff:
                changed[key] = diff
        
        # Add non-empty categories to result
        if removed: result['removed'] = removed
        if added: result['added'] = added
        if changed: result['changed'] = changed
        
        return result

    def generate_diff_report(self, config1: Dict[str, Any], config2: Dict[str, Any]) -> str:
        # Generate a human-readable diff report string between two configs
        diff = self.compare(config1, config2)
        if not diff:
            # If no differences, report as such
            return "No differences found"
        
        lines = []
        # Format the diff dict into lines of text
        self._format_diff(diff, lines)
        # Join all lines with newline characters
        return "\n".join(lines)

    def _format_diff(self, diff: Dict[str, Any], lines: List[str], prefix: str = ""):
        # Helper method to format diff dict recursively into text lines
        for key, value in diff.items():
            if key == 'added':
                # Report added keys with '+ Added' prefix
                for k, v in value.items():
                    lines.append(f"{prefix}+ Added '{k}': {v}")
            elif key == 'removed':
                # Report removed keys with '- Removed' prefix
                for k, v in value.items():
                    lines.append(f"{prefix}- Removed '{k}': {v}")
            elif key == 'changed':
                # For changed keys, check if it's a leaf change or nested dict
                for k, v in value.items():
                    if isinstance(v, dict) and 'from' in v and 'to' in v:
                        # Leaf change, show from -> to
                        lines.append(f"{prefix}* Changed '{k}': {v['from']} -> {v['to']}")
                    else:
                        # Nested changes, print key and recurse with indentation
                        lines.append(f"{prefix}* Changed '{k}':")
                        self._format_diff(v, lines, prefix + "  ")

def load_config(path: str) -> Dict[str, Any]:
    # Determine the appropriate loader based on the file path or extension
    loader = _get_loader_by_path(path)
    # Use the loader to load and return the configuration data
    return loader.load(path)

def save_config(config: Dict[str, Any], path: str):
    # Determine the appropriate loader based on the file path or extension
    loader = _get_loader_by_path(path)
    # Use the loader to save the configuration data to the specified path
    loader.save(config, path)

def validate_config_file(path: str) -> bool:
    try:
        # Attempt to load the config file from the given path
        config = load_config(path)
        # Validate the loaded configuration and return the result
        return _validator.validate(config)
    except Exception as e:
        # Log an error message if loading or validation fails
        logger.error(f"Failed to validate config file '{path}': {e}")
        return False

def merge_config_files(base_path: str, override_path: str, output_path: str = None) -> Dict[str, Any]:  # type: ignore
    # Load base configuration from given file path
    base_config = load_config(base_path)
    # Load override configuration from given file path
    override_config = load_config(override_path)
    # Merge the base and override configurations
    merged = _merger.merge(base_config, override_config)
    
    # If output path is specified, save the merged configuration to the file
    if output_path:
        save_config(merged, output_path)
    
    # Return the merged configuration dictionary
    return merged

def _get_loader_by_path(path: str) -> ConfigLoaderInterface:
    # Extract the file extension from the path, convert to lowercase
    ext = Path(path).suffix.lower()
    # Get the appropriate loader for the file extension
    loader = _loaders.get(ext)
    # Raise an error if no loader found for the extension
    if not loader:
        raise ValueError(f"No loader available for extension '{ext}'")
    # Return the found loader instance
    return loader

def setup_config_logging(level=logging.INFO):
    # Configure logging with specified level and format, output to stdout
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

# Dictionary mapping file extensions to their respective config loader instances
_loaders = {
    '.yaml': YAMLConfigLoader(),
    '.yml': YAMLConfigLoader(),
    '.json': JSONConfigLoader()
}

# Instantiate a configuration validator object
_validator = ConfigValidator()
# Instantiate a configuration merger object
_merger = ConfigMerger()
# Instantiate the main configuration manager
config_manager = ConfigManager()
