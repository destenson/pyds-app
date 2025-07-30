"""
Tests for detection engine and strategies.

Tests detection metadata processing, strategy pattern implementation, and custom detection.
"""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
import numpy as np

from src.detection.engine import (
    DetectionEngine, DetectionMode, MetadataExtractor, StrategyManager
)
from src.detection.models import (
    VideoDetection, BoundingBox, DetectionClass, ConfidenceLevel,
    create_test_detection, filter_overlapping_detections
)
from src.detection.strategies import (
    YOLODetectionStrategy, TemplateMatchingStrategy, FeatureBasedStrategy
)
from src.detection.custom import (
    StrategyRegistry, CustomPattern, StrategyPlugin
)
from src.config import DetectionConfig
from src.utils.errors import DetectionError


class TestDetectionModels:
    """Test detection data models."""
    
    def test_bounding_box_creation(self):
        """Test bounding box creation and validation."""
        # Valid bounding box
        bbox = BoundingBox(x=0.1, y=0.2, width=0.3, height=0.4)
        assert bbox.x == 0.1
        assert bbox.center_x == 0.25  # 0.1 + 0.3/2
        assert bbox.center_y == 0.4   # 0.2 + 0.4/2
        assert bbox.area == 0.12      # 0.3 * 0.4
        
        # Test invalid coordinates
        with pytest.raises(ValueError):
            BoundingBox(x=1.5, y=0.2, width=0.3, height=0.4)  # x > 1
        
        with pytest.raises(ValueError):
            BoundingBox(x=0.8, y=0.2, width=0.3, height=0.4)  # x + width > 1
    
    def test_video_detection_creation(self):
        """Test VideoDetection creation and methods."""
        detection = VideoDetection(
            pattern_name="person",
            confidence=0.85,
            bounding_box=BoundingBox(x=0.1, y=0.1, width=0.2, height=0.3),
            timestamp=datetime.now(),
            frame_number=100,
            source_id="camera_1"
        )
        
        assert detection.pattern_name == "person"
        assert detection.confidence == 0.85
        assert detection.confidence_level == ConfidenceLevel.HIGH
        
        # Test serialization
        detection_dict = detection.to_dict()
        assert detection_dict['pattern_name'] == "person"
        assert detection_dict['confidence'] == 0.85
        
        # Test unique ID generation
        detection2 = create_test_detection()
        assert detection.detection_id != detection2.detection_id
    
    def test_detection_filtering(self):
        """Test overlapping detection filtering."""
        detections = [
            VideoDetection(
                pattern_name="person",
                confidence=0.9,
                bounding_box=BoundingBox(x=0.1, y=0.1, width=0.2, height=0.3),
                timestamp=datetime.now(),
                frame_number=1,
                source_id="test"
            ),
            VideoDetection(
                pattern_name="person",
                confidence=0.8,
                bounding_box=BoundingBox(x=0.15, y=0.15, width=0.2, height=0.3),
                timestamp=datetime.now(),
                frame_number=1,
                source_id="test"
            ),
            VideoDetection(
                pattern_name="car",
                confidence=0.85,
                bounding_box=BoundingBox(x=0.5, y=0.5, width=0.3, height=0.2),
                timestamp=datetime.now(),
                frame_number=1,
                source_id="test"
            )
        ]
        
        # Filter overlapping detections
        filtered = filter_overlapping_detections(detections, iou_threshold=0.5)
        
        # Should keep highest confidence person and the car
        assert len(filtered) == 2
        assert filtered[0].confidence == 0.9  # Highest confidence person
        assert filtered[1].pattern_name == "car"


class TestDetectionEngine:
    """Test core detection engine functionality."""
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self, test_config, mock_deepstream_api):
        """Test detection engine initialization."""
        engine = DetectionEngine(test_config.detection)
        await engine.initialize()
        
        assert engine.is_initialized
        assert engine.confidence_threshold == test_config.detection.confidence_threshold
    
    @pytest.mark.asyncio
    async def test_process_frame_metadata(self, test_config, mock_deepstream_api):
        """Test processing DeepStream frame metadata."""
        engine = DetectionEngine(test_config.detection)
        await engine.initialize()
        
        # Create mock batch metadata
        from tests.conftest import create_mock_batch_meta
        batch_meta = create_mock_batch_meta(num_frames=1, objects_per_frame=3)
        
        # Process metadata
        detections = await engine.process_frame_metadata(
            batch_meta=batch_meta,
            source_id="test_source",
            frame_width=1920,
            frame_height=1080
        )
        
        assert len(detections) >= 2  # At least 2 should pass confidence threshold
        assert all(d.source_id == "test_source" for d in detections)
        assert all(d.confidence >= 0.5 for d in detections)
    
    @pytest.mark.asyncio
    async def test_strategy_registration(self, test_config):
        """Test registering detection strategies."""
        engine = DetectionEngine(test_config.detection)
        
        # Create mock strategy
        mock_strategy = MagicMock()
        mock_strategy.name = "test_strategy"
        mock_strategy.initialize = Mock(return_value=asyncio.coroutine(lambda: True)())
        
        # Register strategy
        await engine.register_strategy(mock_strategy)
        
        assert "test_strategy" in engine._strategy_manager._strategies
    
    @pytest.mark.asyncio
    async def test_detection_with_strategies(self, test_config, mock_deepstream_api):
        """Test detection with multiple strategies."""
        engine = DetectionEngine(test_config.detection)
        await engine.initialize()
        
        # Create mock strategies
        strategy1 = MagicMock()
        strategy1.name = "strategy1"
        strategy1.process = Mock(return_value=asyncio.coroutine(
            lambda det: [det]  # Return same detection
        ))
        
        strategy2 = MagicMock()
        strategy2.name = "strategy2" 
        strategy2.process = Mock(return_value=asyncio.coroutine(
            lambda det: []  # Filter out detection
        ))
        
        await engine.register_strategy(strategy1)
        await engine.register_strategy(strategy2)
        
        # Process detection
        test_detection = create_test_detection()
        results = await engine.apply_strategies(test_detection)
        
        assert len(results) == 2  # Results from both strategies
        assert results[0].strategy_name == "strategy1"
        assert len(results[0].detections) == 1
        assert results[1].strategy_name == "strategy2"
        assert len(results[1].detections) == 0
    
    def test_detection_statistics(self, test_config):
        """Test detection statistics collection."""
        engine = DetectionEngine(test_config.detection)
        
        # Add detection statistics
        for i in range(100):
            engine.update_statistics(
                source_id="test_source",
                detection_count=i % 5,
                processing_time_ms=10.0 + (i % 10)
            )
        
        stats = engine.get_statistics("test_source")
        assert stats.total_detections > 0
        assert stats.average_processing_time_ms > 0
        assert stats.detections_per_second > 0


class TestDetectionStrategies:
    """Test built-in detection strategies."""
    
    @pytest.mark.asyncio
    async def test_yolo_strategy(self, test_config):
        """Test YOLO detection strategy."""
        config = {
            'yolo': {
                'model_path': '/path/to/model.cfg',
                'config_path': '/path/to/config.cfg',
                'confidence_threshold': 0.5
            }
        }
        
        strategy = YOLODetectionStrategy("yolo_v5", config)
        
        # Mock model loading
        with patch('cv2.dnn.readNet') as mock_read_net:
            mock_net = MagicMock()
            mock_read_net.return_value = mock_net
            
            initialized = await strategy.initialize()
            assert initialized is True
        
        # Test detection processing
        test_detection = create_test_detection()
        
        # Mock should_process
        assert strategy.should_process(test_detection) is True
    
    @pytest.mark.asyncio
    async def test_template_matching_strategy(self, test_config, temp_dir):
        """Test template matching strategy."""
        # Create test template
        template_dir = temp_dir / "templates"
        template_dir.mkdir()
        template_file = template_dir / "template1.png"
        
        # Create dummy template image
        import numpy as np
        import cv2
        template_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(template_file), template_img)
        
        config = {
            'template': {
                'template_dir': str(template_dir),
                'match_threshold': 0.8
            }
        }
        
        strategy = TemplateMatchingStrategy("template_match", config)
        initialized = await strategy.initialize()
        assert initialized is True
        assert len(strategy._templates) > 0
    
    @pytest.mark.asyncio
    async def test_feature_based_strategy(self, test_config):
        """Test feature-based detection strategy."""
        config = {
            'feature': {
                'detector_type': 'ORB',  # ORB is patent-free
                'matcher_type': 'BF',
                'min_matches': 10
            }
        }
        
        strategy = FeatureBasedStrategy("feature_detect", config)
        initialized = await strategy.initialize()
        assert initialized is True
        assert strategy._detector is not None
        assert strategy._matcher is not None


class TestCustomDetection:
    """Test custom detection registration and loading."""
    
    def test_strategy_registry(self):
        """Test strategy registry functionality."""
        registry = StrategyRegistry()
        
        # Create mock strategy class
        class CustomStrategy:
            def __init__(self, name, config):
                self.name = name
                self.config = config
        
        # Register plugin
        plugin = StrategyPlugin(
            name="custom_detector",
            version="1.0",
            author="Test Author",
            description="Test custom detector",
            strategy_class=CustomStrategy,
            module_path="test.custom_strategy"
        )
        
        success = registry.register_strategy_plugin(plugin)
        assert success is True
        
        # Create instance
        instance = registry.create_strategy_instance(
            "custom_detector",
            {"threshold": 0.8}
        )
        assert instance is not None
        assert instance.config['threshold'] == 0.8
    
    def test_custom_pattern_registration(self):
        """Test custom pattern registration."""
        registry = StrategyRegistry()
        
        # Register custom pattern
        pattern = CustomPattern(
            name="logo_detector",
            description="Detect company logos",
            pattern_type="template",
            config={
                'template_path': '/path/to/logo.png',
                'scale_invariant': True
            },
            created_by="test_user"
        )
        
        registry.register_pattern(pattern)
        assert "logo_detector" in registry._patterns
        
        # Get pattern
        retrieved = registry.get_pattern("logo_detector")
        assert retrieved.name == "logo_detector"
        assert retrieved.config['scale_invariant'] is True
    
    @pytest.mark.asyncio
    async def test_plugin_loading(self, temp_dir):
        """Test loading strategy plugins from directory."""
        registry = StrategyRegistry()
        
        # Create plugin directory
        plugin_dir = temp_dir / "plugins"
        plugin_dir.mkdir()
        
        # Create simple plugin file
        plugin_file = plugin_dir / "simple_strategy.py"
        plugin_code = """
from src.detection.models import DetectionStrategy, VideoDetection

class SimpleStrategy(DetectionStrategy):
    def __init__(self, name, config):
        super().__init__(name, config)
    
    async def initialize(self):
        return True
    
    def should_process(self, detection):
        return True
    
    async def process(self, detection):
        return [detection]

# Plugin metadata
PLUGIN_INFO = {
    'name': 'simple_strategy',
    'version': '1.0',
    'author': 'Test',
    'description': 'Simple test strategy',
    'strategy_class': SimpleStrategy
}
"""
        plugin_file.write_text(plugin_code)
        
        # Add plugin path and scan
        registry.add_plugin_path(plugin_dir)
        loaded = await registry.scan_and_load_plugins()
        
        assert loaded > 0
        assert "simple_strategy" in registry._plugins
    
    def test_strategy_validation(self):
        """Test strategy configuration validation."""
        registry = StrategyRegistry()
        
        # Define schema
        schema = {
            'threshold': {'type': 'float', 'min': 0.0, 'max': 1.0},
            'min_size': {'type': 'int', 'min': 10}
        }
        
        # Valid config
        valid_config = {'threshold': 0.7, 'min_size': 20}
        is_valid = registry.validate_strategy_config(valid_config, schema)
        assert is_valid is True
        
        # Invalid config
        invalid_config = {'threshold': 1.5, 'min_size': 5}
        is_valid = registry.validate_strategy_config(invalid_config, schema)
        assert is_valid is False