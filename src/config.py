"""
Configuration management system for the DeepStream inference system.

This module provides comprehensive YAML configuration loading, validation,
runtime updates, environment variable overrides, and configuration inheritance.
"""

import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Union, List, Type, get_type_hints
from dataclasses import dataclass, field, fields
from enum import Enum
import yaml
from pydantic import BaseModel, ValidationError, field_validator, model_validator, Field
from pydantic.dataclasses import dataclass as pydantic_dataclass
import logging

from .utils.errors import ConfigurationError, PyDSError
from .utils.logging import get_logger


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertLevel(str, Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SourceType(str, Enum):
    """Video source types."""
    FILE = "file"
    WEBCAM = "webcam"
    RTSP = "rtsp"
    WEBRTC = "webrtc"
    NETWORK = "network"
    TEST = "test"
    HTTP = "http"


@pydantic_dataclass
class PipelineConfig:
    """Pipeline configuration settings."""
    batch_size: int = Field(default=4, ge=1, le=16, description="Number of streams to batch together")
    width: int = Field(default=1920, ge=320, le=7680, description="Processing width in pixels")
    height: int = Field(default=1080, ge=240, le=4320, description="Processing height in pixels")
    fps: float = Field(default=30.0, ge=1.0, le=120.0, description="Target frames per second")
    buffer_pool_size: int = Field(default=10, ge=5, le=50, description="Buffer pool size for memory management")
    gpu_id: int = Field(default=0, ge=0, le=7, description="GPU device ID to use")
    batch_timeout_us: int = Field(default=40000, ge=1000, le=1000000, description="Batch timeout in microseconds")
    enable_gpu_inference: bool = Field(default=True, description="Enable GPU acceleration for inference")
    memory_type: str = Field(default="device", pattern="^(device|unified|pinned)$", description="Memory type for GPU buffers")
    num_decode_surfaces: int = Field(default=16, ge=4, le=64, description="Number of decode surfaces")
    processing_mode: str = Field(default="batch", pattern="^(batch|single)$", description="Processing mode")
    
    @field_validator('width', 'height')
    @classmethod
    def validate_dimensions(cls, v):
        """Ensure dimensions are multiples of 16 for GPU efficiency."""
        if v % 16 != 0:
            raise ValueError(f"Dimension must be a multiple of 16 for GPU efficiency")
        return v


@pydantic_dataclass  
class DetectionConfig:
    """Detection engine configuration."""
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum confidence for detections")
    nms_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Non-maximum suppression threshold")
    max_objects: int = Field(default=100, ge=1, le=1000, description="Maximum objects per frame")
    enable_tracking: bool = Field(default=True, description="Enable object tracking")
    tracker_config_file: Optional[str] = Field(default=None, description="Path to tracker configuration file")
    model_engine_file: Optional[str] = Field(default=None, description="Path to TensorRT engine file")
    labels_file: Optional[str] = Field(default=None, description="Path to class labels file")
    input_shape: List[int] = Field(default_factory=lambda: [3, 608, 608], description="Model input shape [C, H, W]")
    output_layers: List[str] = Field(default_factory=lambda: ["output"], description="Model output layer names")
    batch_inference: bool = Field(default=True, description="Enable batch inference")
    tensor_rt_precision: str = Field(default="fp16", pattern="^(fp32|fp16|int8)$", description="TensorRT precision mode")
    
    @field_validator('input_shape')
    @classmethod
    def validate_input_shape(cls, v):
        """Validate input shape format."""
        if len(v) != 3:
            raise ValueError("input_shape must have exactly 3 dimensions [C, H, W]")
        if any(dim <= 0 for dim in v):
            raise ValueError("All input_shape dimensions must be positive")
        return v


@pydantic_dataclass
class AlertConfig:
    """Alert system configuration."""
    enabled: bool = Field(default=True, description="Enable alert system")
    throttle_seconds: int = Field(default=60, ge=1, le=3600, description="Throttle duplicate alerts (seconds)")
    burst_threshold: int = Field(default=3, ge=1, le=20, description="Maximum alerts in burst period")
    max_alerts_per_minute: int = Field(default=10, ge=1, le=1000, description="Rate limit for alerts")
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum confidence for alerts")
    level: AlertLevel = Field(default=AlertLevel.MEDIUM, description="Default alert level")
    handlers: List[str] = Field(default_factory=lambda: ["console"], description="Alert handlers to use")
    retry_attempts: int = Field(default=3, ge=1, le=10, description="Retry attempts for failed alerts")
    retry_delay: float = Field(default=1.0, ge=0.1, le=60.0, description="Delay between retry attempts")


@pydantic_dataclass
class MonitoringConfig:
    """Monitoring and metrics configuration."""
    enabled: bool = Field(default=True, description="Enable monitoring")
    metrics_interval: int = Field(default=30, ge=1, le=300, description="Metrics collection interval (seconds)")
    health_check_interval: int = Field(default=30, ge=5, le=300, description="Health check interval (seconds)")
    profiling_enabled: bool = Field(default=False, description="Enable performance profiling")
    prometheus_enabled: bool = Field(default=False, description="Enable Prometheus metrics")
    prometheus_port: int = Field(default=8000, ge=1024, le=65535, description="Prometheus metrics port")
    prometheus_path: str = Field(default="/metrics", description="Prometheus metrics path")
    log_metrics: bool = Field(default=True, description="Log metrics to structured logs")
    cpu_threshold: float = Field(default=80.0, ge=10.0, le=100.0, description="CPU usage alert threshold (%)")
    memory_threshold: float = Field(default=80.0, ge=10.0, le=100.0, description="Memory usage alert threshold (%)")
    gpu_threshold: float = Field(default=90.0, ge=10.0, le=100.0, description="GPU usage alert threshold (%)")


@pydantic_dataclass
class LoggingConfig:
    """Logging configuration."""
    level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    format: str = Field(default="json", pattern="^(json|text|colored)$", description="Log format")
    include_metrics: bool = Field(default=True, description="Include performance metrics in logs")
    log_file: Optional[str] = Field(default=None, description="Path to log file")
    max_file_size_mb: int = Field(default=100, ge=1, le=1000, description="Maximum log file size (MB)")
    backup_count: int = Field(default=5, ge=1, le=20, description="Number of backup log files")
    console_output: bool = Field(default=True, description="Enable console output")
    json_output: bool = Field(default=False, description="Force JSON output")
    real_time_streaming: bool = Field(default=False, description="Enable real-time log streaming")
    streaming_port: int = Field(default=9999, ge=1024, le=65535, description="Log streaming port")
    filter_spam: bool = Field(default=True, description="Enable spam filtering")
    rate_limit_per_minute: int = Field(default=1000, ge=10, le=10000, description="Rate limit for log messages")


@pydantic_dataclass
class RecoveryConfig:
    """Error recovery configuration."""
    max_retries: int = Field(default=3, ge=1, le=10, description="Maximum retry attempts")
    retry_delay: float = Field(default=2.0, ge=0.1, le=60.0, description="Initial retry delay (seconds)")
    exponential_backoff: bool = Field(default=True, description="Use exponential backoff for retries")
    max_retry_delay: float = Field(default=60.0, ge=1.0, le=600.0, description="Maximum retry delay (seconds)")
    circuit_breaker_enabled: bool = Field(default=True, description="Enable circuit breaker pattern")
    circuit_breaker_threshold: int = Field(default=5, ge=2, le=20, description="Circuit breaker failure threshold")
    circuit_breaker_timeout: int = Field(default=60, ge=10, le=600, description="Circuit breaker timeout (seconds)")


@pydantic_dataclass
class SourceConfig:
    """Video source configuration."""
    id: str = Field(description="Unique source identifier")
    name: str = Field(description="Human-readable source name")
    type: SourceType = Field(description="Source type")
    uri: str = Field(description="Source URI")
    enabled: bool = Field(default=True, description="Enable this source")
    retry_count: int = Field(default=0, ge=0, description="Current retry count")
    max_retries: int = Field(default=5, ge=1, le=20, description="Maximum retry attempts")
    retry_delay: float = Field(default=5.0, ge=1.0, le=300.0, description="Retry delay (seconds)")
    timeout: int = Field(default=30, ge=5, le=300, description="Connection timeout (seconds)")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Source-specific parameters")
    
    @model_validator(mode='after')
    def validate_uri_type_match(self):
        """Validate URI format based on source type."""
        source_type = self.type
        uri = self.uri
        
        if source_type == SourceType.RTSP and not uri.startswith(('rtsp://', 'rtsps://')):
            raise ValueError("RTSP source URI must start with rtsp:// or rtsps://")
        elif source_type == SourceType.HTTP and not uri.startswith(('http://', 'https://')):
            raise ValueError("HTTP source URI must start with http:// or https://")
        elif source_type == SourceType.FILE and not (uri.startswith('file://') or Path(uri).exists()):
            if not uri.startswith('file://'):
                # Check if it's a valid file path
                path = Path(uri)
                if not path.exists() and not path.is_absolute():
                    raise ValueError(f"File source URI must be a valid file path or start with file://: {uri}")
        elif source_type == SourceType.WEBCAM:
            try:
                device_id = int(uri)
                if device_id < 0:
                    raise ValueError("Webcam device ID must be non-negative")
            except ValueError:
                if not uri.startswith('/dev/video'):
                    raise ValueError("Webcam URI must be a device ID (0, 1, 2...) or device path (/dev/video*)")
        
        return self


@pydantic_dataclass
class AppConfig:
    """Main application configuration."""
    # Core configuration sections
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    
    # Application settings
    name: str = Field(default="PyDS Video Analytics", description="Application name")
    version: str = Field(default="0.1.0", description="Application version")
    debug: bool = Field(default=False, description="Enable debug mode")
    environment: str = Field(default="development", pattern="^(development|staging|production)$", description="Environment")
    
    # Sources configuration
    sources: List[SourceConfig] = Field(default_factory=list, description="Video sources configuration")
    
    # Advanced settings
    thread_pool_size: int = Field(default=4, ge=1, le=32, description="Thread pool size")
    max_concurrent_sources: int = Field(default=8, ge=1, le=32, description="Maximum concurrent sources")
    graceful_shutdown_timeout: int = Field(default=30, ge=5, le=300, description="Graceful shutdown timeout (seconds)")
    
    # Custom configuration extensions
    extensions: Dict[str, Any] = Field(default_factory=dict, description="Custom configuration extensions")
    
    @field_validator('sources')
    @classmethod
    def validate_sources(cls, v):
        """Validate sources configuration."""
        if not v:
            return v
        
        # Check for duplicate source IDs
        source_ids = [source.id for source in v]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("Source IDs must be unique")
        
        return v


class ConfigManager:
    """Thread-safe configuration manager with validation and runtime updates."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self._config: Optional[AppConfig] = None
        self._config_path: Optional[Path] = None
        self._lock = threading.RLock()  # Reentrant lock for nested access
        self._logger = get_logger(__name__)
        self._watchers: List[callable] = []
        self._environment_overrides: Dict[str, Any] = {}
        
        if config_path:
            self.load_config(config_path)
        else:
            # Load default configuration
            self._config = AppConfig()
    
    def load_config(self, config_path: Union[str, Path], merge_environment: bool = True) -> AppConfig:
        """
        Load configuration from YAML file with validation.
        
        Args:
            config_path: Path to configuration file
            merge_environment: Whether to merge environment variable overrides
            
        Returns:
            Loaded and validated configuration
            
        Raises:
            ConfigurationError: If configuration is invalid or file cannot be loaded
        """
        with self._lock:
            try:
                config_path = Path(config_path)
                self._config_path = config_path
                
                if not config_path.exists():
                    raise ConfigurationError(
                        f"Configuration file not found: {config_path}",
                        config_path=str(config_path)
                    )
                
                self._logger.info(f"Loading configuration from {config_path}")
                
                # Load YAML content
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                
                if config_data is None:
                    config_data = {}
                
                # Apply environment variable overrides
                if merge_environment:
                    config_data = self._merge_environment_overrides(config_data)
                
                # Validate and create configuration object
                try:
                    self._config = AppConfig(**config_data)
                except ValidationError as e:
                    error_details = self._format_validation_errors(e)
                    raise ConfigurationError(
                        f"Configuration validation failed:\n{error_details}",
                        config_path=str(config_path),
                        context={'validation_errors': error_details}
                    )
                
                self._logger.info(f"Successfully loaded configuration: {self._config.name} v{self._config.version}")
                
                # Notify watchers of configuration change
                self._notify_watchers()
                
                return self._config
                
            except (yaml.YAMLError, IOError) as e:
                raise ConfigurationError(
                    f"Failed to load configuration file {config_path}: {e}",
                    config_path=str(config_path),
                    original_exception=e
                )
    
    def save_config(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """
        Save current configuration to YAML file.
        
        Args:
            config_path: Path to save configuration (defaults to loaded path)
            
        Raises:
            ConfigurationError: If saving fails
        """
        with self._lock:
            if self._config is None:
                raise ConfigurationError("No configuration loaded to save")
            
            save_path = Path(config_path) if config_path else self._config_path
            if save_path is None:
                raise ConfigurationError("No configuration path specified")
            
            try:
                # Convert configuration to dictionary
                config_dict = self._config_to_dict(self._config)
                
                # Ensure parent directory exists
                save_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Write YAML file
                with open(save_path, 'w', encoding='utf-8') as f:
                    yaml.dump(
                        config_dict,
                        f,
                        default_flow_style=False,
                        indent=2,
                        sort_keys=False,
                        allow_unicode=True
                    )
                
                self._logger.info(f"Configuration saved to {save_path}")
                
            except (IOError, yaml.YAMLError) as e:
                raise ConfigurationError(
                    f"Failed to save configuration to {save_path}: {e}",
                    config_path=str(save_path),
                    original_exception=e
                )
    
    def get_config(self) -> AppConfig:
        """
        Get current configuration.
        
        Returns:
            Current configuration object
            
        Raises:
            ConfigurationError: If no configuration is loaded
        """
        with self._lock:
            if self._config is None:
                raise ConfigurationError("No configuration loaded")
            return self._config
    
    def update_config(self, updates: Dict[str, Any], validate: bool = True) -> AppConfig:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
            validate: Whether to validate the updated configuration
            
        Returns:
            Updated configuration object
            
        Raises:
            ConfigurationError: If update fails or validation fails
        """
        with self._lock:
            if self._config is None:
                raise ConfigurationError("No configuration loaded to update")
            
            try:
                # Convert current config to dict
                config_dict = self._config_to_dict(self._config)
                
                # Apply updates recursively
                config_dict = self._deep_merge(config_dict, updates)
                
                # Validate if requested
                if validate:
                    try:
                        new_config = AppConfig(**config_dict)
                    except ValidationError as e:
                        error_details = self._format_validation_errors(e)
                        raise ConfigurationError(
                            f"Configuration update validation failed:\n{error_details}",
                            context={'validation_errors': error_details, 'updates': updates}
                        )
                else:
                    new_config = AppConfig(**config_dict)
                
                # Update configuration
                old_config = self._config
                self._config = new_config
                
                self._logger.info(f"Configuration updated: {len(updates)} changes")
                self._logger.debug(f"Configuration updates: {updates}")
                
                # Notify watchers
                self._notify_watchers()
                
                return self._config
                
            except Exception as e:
                if not isinstance(e, ConfigurationError):
                    raise ConfigurationError(
                        f"Failed to update configuration: {e}",
                        context={'updates': updates},
                        original_exception=e
                    )
                raise
    
    def add_watcher(self, callback: callable) -> None:
        """
        Add a callback to be notified when configuration changes.
        
        Args:
            callback: Function to call when configuration changes
        """
        with self._lock:
            if callback not in self._watchers:
                self._watchers.append(callback)
                self._logger.debug(f"Added configuration watcher: {callback.__name__}")
    
    def remove_watcher(self, callback: callable) -> None:
        """Remove a configuration watcher."""
        with self._lock:
            if callback in self._watchers:
                self._watchers.remove(callback)
                self._logger.debug(f"Removed configuration watcher: {callback.__name__}")
    
    def reload_config(self) -> AppConfig:
        """
        Reload configuration from file.
        
        Returns:
            Reloaded configuration
            
        Raises:
            ConfigurationError: If no config path is set or reload fails
        """
        if self._config_path is None:
            raise ConfigurationError("No configuration file path set for reloading")
        
        return self.load_config(self._config_path)
    
    def validate_config(self, config_dict: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Validate configuration without updating.
        
        Args:
            config_dict: Configuration dictionary to validate (defaults to current)
            
        Returns:
            List of validation errors (empty if valid)
        """
        if config_dict is None:
            if self._config is None:
                return ["No configuration loaded"]
            config_dict = self._config_to_dict(self._config)
        
        try:
            AppConfig(**config_dict)
            return []
        except ValidationError as e:
            return self._format_validation_errors(e).split('\n')
    
    def get_environment_overrides(self) -> Dict[str, Any]:
        """Get current environment variable overrides."""
        return self._environment_overrides.copy()
    
    def _merge_environment_overrides(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Merge environment variable overrides into configuration."""
        overrides = {}
        
        # Common environment variable patterns
        env_mappings = {
            'PYDS_LOG_LEVEL': 'logging.level',
            'PYDS_DEBUG': 'debug',
            'PYDS_ENVIRONMENT': 'environment',
            'PYDS_GPU_ID': 'pipeline.gpu_id',
            'PYDS_BATCH_SIZE': 'pipeline.batch_size',
            'PYDS_CONFIDENCE_THRESHOLD': 'detection.confidence_threshold',
            'PYDS_ALERTS_ENABLED': 'alerts.enabled',
            'PYDS_MONITORING_ENABLED': 'monitoring.enabled',
            'PYDS_PROMETHEUS_PORT': 'monitoring.prometheus_port',
            'PYDS_LOG_FILE': 'logging.log_file',
        }
        
        # Check for environment variables
        for env_var, config_path in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                # Convert string value to appropriate type
                converted_value = self._convert_env_value(value, config_path)
                self._set_nested_value(overrides, config_path, converted_value)
                self._environment_overrides[env_var] = converted_value
        
        # Check for generic PYDS_* environment variables
        for key, value in os.environ.items():
            if key.startswith('PYDS_') and key not in env_mappings:
                # Convert PYDS_SECTION_SETTING to section.setting
                config_path = key[5:].lower().replace('_', '.')  # Remove PYDS_ prefix
                converted_value = self._convert_env_value(value, config_path)
                self._set_nested_value(overrides, config_path, converted_value)
                self._environment_overrides[key] = converted_value
        
        # Merge overrides into config data
        if overrides:
            config_data = self._deep_merge(config_data, overrides)
            self._logger.info(f"Applied {len(self._environment_overrides)} environment overrides")
        
        return config_data
    
    def _convert_env_value(self, value: str, config_path: str) -> Any:
        """Convert environment variable string to appropriate type."""
        # Boolean conversion
        if value.lower() in ('true', 'false', 'yes', 'no', '1', '0'):
            return value.lower() in ('true', 'yes', '1')
        
        # Integer conversion
        try:
            if '.' not in value:
                return int(value)
        except ValueError:
            pass
        
        # Float conversion
        try:
            return float(value)
        except ValueError:
            pass
        
        # List conversion (comma-separated)
        if ',' in value:
            return [item.strip() for item in value.split(',')]
        
        # String value
        return value
    
    def _set_nested_value(self, dictionary: Dict[str, Any], path: str, value: Any) -> None:
        """Set a nested dictionary value using dot notation."""
        keys = path.split('.')
        current = dictionary
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _deep_merge(self, base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge two dictionaries."""
        result = base.copy()
        
        for key, value in updates.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _config_to_dict(self, config: AppConfig) -> Dict[str, Any]:
        """Convert configuration object to dictionary."""
        # Use pydantic's dict() method for proper serialization
        return config.__dict__ if hasattr(config, '__dict__') else {}
    
    def _format_validation_errors(self, error: ValidationError) -> str:
        """Format validation errors into a readable string."""
        errors = []
        for err in error.errors():
            location = ' -> '.join(str(loc) for loc in err['loc'])
            message = err['msg']
            value = err.get('input', 'N/A')
            errors.append(f"  {location}: {message} (got: {value})")
        
        return '\n'.join(errors)
    
    def _notify_watchers(self) -> None:
        """Notify all registered watchers of configuration changes."""
        for watcher in self._watchers:
            try:
                watcher(self._config)
            except Exception as e:
                self._logger.error(f"Error in configuration watcher {watcher.__name__}: {e}")


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def load_config(config_path: Union[str, Path]) -> AppConfig:
    """Load configuration from file using global manager."""
    return get_config_manager().load_config(config_path)


def get_config() -> AppConfig:
    """Get current configuration using global manager."""
    return get_config_manager().get_config()


def update_config(updates: Dict[str, Any]) -> AppConfig:
    """Update configuration using global manager."""
    return get_config_manager().update_config(updates)


def add_config_watcher(callback: callable) -> None:
    """Add configuration watcher using global manager."""
    get_config_manager().add_watcher(callback)


def validate_config_file(config_path: Union[str, Path]) -> List[str]:
    """
    Validate a configuration file without loading it.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        List of validation errors (empty if valid)
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f) or {}
        
        temp_manager = ConfigManager()
        return temp_manager.validate_config(config_data)
        
    except Exception as e:
        return [f"Failed to load configuration file: {e}"]


def validate_config(config: AppConfig) -> List[str]:
    """
    Validate a configuration object.
    
    Args:
        config: Configuration object to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    temp_manager = ConfigManager()
    temp_manager._config = config
    return temp_manager.validate_config()


def save_config(config: AppConfig, config_path: Union[str, Path]) -> None:
    """
    Save configuration to file using global manager.
    
    Args:
        config: Configuration to save
        config_path: Path to save configuration file
    """
    temp_manager = ConfigManager()
    temp_manager._config = config
    temp_manager.save_config(config_path)


def create_default_config(output_path: Union[str, Path]) -> None:
    """
    Create a default configuration file.
    
    Args:
        output_path: Path where to save the default configuration
    """
    default_config = AppConfig()
    temp_manager = ConfigManager()
    temp_manager._config = default_config
    temp_manager.save_config(output_path)