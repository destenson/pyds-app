"""
Tests for configuration management system.

Tests configuration loading, validation, runtime updates, and environment overrides.
"""

import pytest
import os
import yaml
from pathlib import Path
from unittest.mock import patch

from src.config import (
    AppConfig, ConfigManager, PipelineConfig, DetectionConfig,
    AlertConfig, SourceConfig, SourceType, LogLevel, AlertLevel,
    load_config, get_config, update_config, validate_config_file
)
from src.utils.errors import ConfigurationError


class TestConfigModels:
    """Test configuration data models and validation."""
    
    def test_pipeline_config_validation(self):
        """Test pipeline configuration validation."""
        # Valid configuration
        config = PipelineConfig(
            batch_size=4,
            width=1920,
            height=1080,
            fps=30.0
        )
        assert config.batch_size == 4
        assert config.width == 1920
        assert config.height == 1080
        
        # Test dimension validation (must be multiple of 16)
        with pytest.raises(ValueError, match="multiple of 16"):
            PipelineConfig(width=1921, height=1080)
        
        # Test batch size limits
        with pytest.raises(ValueError):
            PipelineConfig(batch_size=17)  # Max is 16
    
    def test_detection_config_validation(self):
        """Test detection configuration validation."""
        # Valid configuration
        config = DetectionConfig(
            confidence_threshold=0.7,
            nms_threshold=0.5,
            max_objects=100
        )
        assert config.confidence_threshold == 0.7
        
        # Test threshold limits
        with pytest.raises(ValueError):
            DetectionConfig(confidence_threshold=1.5)  # Max is 1.0
        
        # Test input shape validation
        with pytest.raises(ValueError, match="3 dimensions"):
            DetectionConfig(input_shape=[640, 640])  # Missing channel dimension
    
    def test_source_config_validation(self):
        """Test source configuration validation."""
        # Valid RTSP source
        source = SourceConfig(
            id="camera1",
            name="Front Camera",
            type=SourceType.RTSP,
            uri="rtsp://192.168.1.100:554/stream"
        )
        assert source.id == "camera1"
        assert source.type == SourceType.RTSP
        
        # Test RTSP URI validation
        with pytest.raises(ValueError, match="must start with rtsp://"):
            SourceConfig(
                id="camera2",
                name="Bad Camera",
                type=SourceType.RTSP,
                uri="http://192.168.1.100:554/stream"  # Wrong protocol
            )
        
        # Test webcam validation
        webcam = SourceConfig(
            id="webcam1",
            name="USB Camera",
            type=SourceType.WEBCAM,
            uri="0"  # Device ID
        )
        assert webcam.uri == "0"
        
        # Test file source validation
        file_source = SourceConfig(
            id="file1",
            name="Test Video",
            type=SourceType.FILE,
            uri="file:///path/to/video.mp4"
        )
        assert file_source.uri.startswith("file://")
    
    def test_app_config_validation(self):
        """Test complete application configuration."""
        # Valid configuration
        config = AppConfig(
            name="Test App",
            version="1.0.0",
            sources=[
                SourceConfig(
                    id="source1",
                    name="Source 1",
                    type=SourceType.TEST,
                    uri="videotestsrc"
                ),
                SourceConfig(
                    id="source2",
                    name="Source 2", 
                    type=SourceType.TEST,
                    uri="videotestsrc"
                )
            ]
        )
        assert len(config.sources) == 2
        
        # Test duplicate source ID validation
        with pytest.raises(ValueError, match="Source IDs must be unique"):
            AppConfig(
                sources=[
                    SourceConfig(id="dup", name="S1", type=SourceType.TEST, uri="test"),
                    SourceConfig(id="dup", name="S2", type=SourceType.TEST, uri="test")
                ]
            )


class TestConfigManager:
    """Test configuration manager functionality."""
    
    def test_load_config_from_yaml(self, test_config_path):
        """Test loading configuration from YAML file."""
        manager = ConfigManager()
        config = manager.load_config(test_config_path)
        
        assert config.name == "PyDS Test"
        assert config.version == "0.1.0"
        assert config.pipeline.batch_size == 2
        assert len(config.sources) == 1
        assert config.sources[0].id == "test_source"
    
    def test_load_invalid_config_file(self, temp_dir):
        """Test loading invalid configuration file."""
        # Non-existent file
        manager = ConfigManager()
        with pytest.raises(ConfigurationError, match="not found"):
            manager.load_config(temp_dir / "nonexistent.yaml")
        
        # Invalid YAML
        invalid_path = temp_dir / "invalid.yaml"
        with open(invalid_path, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        with pytest.raises(ConfigurationError, match="Failed to load"):
            manager.load_config(invalid_path)
    
    def test_save_config(self, temp_dir, test_config):
        """Test saving configuration to file."""
        manager = ConfigManager()
        manager._config = test_config
        
        save_path = temp_dir / "saved_config.yaml"
        manager.save_config(save_path)
        
        assert save_path.exists()
        
        # Load and verify
        with open(save_path, 'r') as f:
            saved_data = yaml.safe_load(f)
        
        assert saved_data['name'] == test_config.name
        assert saved_data['version'] == test_config.version
    
    def test_update_config(self, test_config):
        """Test runtime configuration updates."""
        manager = ConfigManager()
        manager._config = test_config
        
        # Update configuration
        updates = {
            'pipeline': {
                'batch_size': 8,
                'fps': 60.0
            },
            'debug': True
        }
        
        updated = manager.update_config(updates)
        
        assert updated.pipeline.batch_size == 8
        assert updated.pipeline.fps == 60.0
        assert updated.debug is True
        
        # Test invalid update
        with pytest.raises(ConfigurationError, match="validation failed"):
            manager.update_config({'pipeline': {'batch_size': 999}})  # Invalid value
    
    def test_environment_overrides(self, test_config_path):
        """Test environment variable overrides."""
        manager = ConfigManager()
        
        # Set environment variables
        env_vars = {
            'PYDS_LOG_LEVEL': 'debug',
            'PYDS_DEBUG': 'true',
            'PYDS_PIPELINE_BATCH_SIZE': '8',
            'PYDS_ALERTS_ENABLED': 'false'
        }
        
        with patch.dict(os.environ, env_vars):
            config = manager.load_config(test_config_path)
        
        assert config.logging.level == LogLevel.DEBUG
        assert config.debug is True
        assert config.pipeline.batch_size == 8
        assert config.alerts.enabled is False
    
    def test_config_watchers(self, test_config):
        """Test configuration change watchers."""
        manager = ConfigManager()
        manager._config = test_config
        
        # Track watcher calls
        watcher_calls = []
        
        def test_watcher(config):
            watcher_calls.append(config)
        
        manager.add_watcher(test_watcher)
        
        # Update configuration
        manager.update_config({'debug': True})
        
        assert len(watcher_calls) == 1
        assert watcher_calls[0].debug is True
        
        # Remove watcher
        manager.remove_watcher(test_watcher)
        manager.update_config({'debug': False})
        
        assert len(watcher_calls) == 1  # No new calls
    
    def test_validate_config(self):
        """Test configuration validation without loading."""
        manager = ConfigManager()
        
        # Valid configuration
        valid_config = {
            'name': 'Test',
            'pipeline': {'batch_size': 4}
        }
        errors = manager.validate_config(valid_config)
        assert len(errors) == 0
        
        # Invalid configuration
        invalid_config = {
            'name': 'Test',
            'pipeline': {'batch_size': 999}  # Invalid
        }
        errors = manager.validate_config(invalid_config)
        assert len(errors) > 0
        assert any('batch_size' in error for error in errors)


class TestConfigurationFunctions:
    """Test module-level configuration functions."""
    
    def test_load_config_function(self, test_config_path):
        """Test global load_config function."""
        config = load_config(test_config_path)
        assert config.name == "PyDS Test"
    
    def test_get_config_function(self, test_config_path):
        """Test global get_config function."""
        # Load config first
        load_config(test_config_path)
        
        # Get config
        config = get_config()
        assert config.name == "PyDS Test"
    
    def test_update_config_function(self, test_config_path):
        """Test global update_config function."""
        # Load config first
        load_config(test_config_path)
        
        # Update config
        updated = update_config({'debug': True})
        assert updated.debug is True
    
    def test_validate_config_file_function(self, test_config_path):
        """Test validate_config_file function."""
        errors = validate_config_file(test_config_path)
        assert len(errors) == 0
        
        # Test invalid file
        errors = validate_config_file("nonexistent.yaml")
        assert len(errors) > 0


class TestConfigurationScenarios:
    """Test real-world configuration scenarios."""
    
    def test_multi_source_configuration(self, temp_dir):
        """Test configuration with multiple video sources."""
        config_data = {
            'name': 'Multi-Source App',
            'sources': [
                {
                    'id': 'rtsp1',
                    'name': 'Camera 1',
                    'type': 'rtsp',
                    'uri': 'rtsp://192.168.1.100:554/stream1'
                },
                {
                    'id': 'rtsp2',
                    'name': 'Camera 2',
                    'type': 'rtsp',
                    'uri': 'rtsp://192.168.1.101:554/stream2'
                },
                {
                    'id': 'file1',
                    'name': 'Recorded Video',
                    'type': 'file',
                    'uri': 'file:///recordings/video.mp4'
                }
            ]
        }
        
        config_path = temp_dir / "multi_source.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        config = load_config(config_path)
        assert len(config.sources) == 3
        assert config.sources[0].type == SourceType.RTSP
        assert config.sources[2].type == SourceType.FILE
    
    def test_production_configuration(self):
        """Test production configuration settings."""
        config = AppConfig(
            name="Production System",
            environment="production",
            debug=False,
            pipeline=PipelineConfig(
                batch_size=8,
                gpu_id=0,
                enable_gpu_inference=True,
                memory_type="device"
            ),
            monitoring={
                'enabled': True,
                'prometheus_enabled': True,
                'prometheus_port': 8000
            }
        )
        
        assert config.environment == "production"
        assert config.debug is False
        assert config.pipeline.batch_size == 8
        assert config.monitoring.prometheus_enabled is True
    
    def test_development_configuration(self):
        """Test development configuration settings."""
        config = AppConfig(
            name="Dev System",
            environment="development",
            debug=True,
            logging={
                'level': LogLevel.DEBUG,
                'format': 'colored',
                'console_output': True
            }
        )
        
        assert config.environment == "development"
        assert config.debug is True
        assert config.logging.level == LogLevel.DEBUG
        assert config.logging.format == "colored"