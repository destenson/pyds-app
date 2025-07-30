"""
Pytest configuration and fixtures for PyDS test suite.

Provides mock DeepStream elements, test data generation, and environment setup
for comprehensive testing without requiring actual DeepStream installation.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import sys
import os
from datetime import datetime
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import AppConfig, SourceConfig, SourceType
from src.detection.models import VideoDetection, BoundingBox


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def test_config_path(temp_dir):
    """Create a test configuration file."""
    config_path = temp_dir / "test_config.yaml"
    config_data = {
        "name": "PyDS Test",
        "version": "0.1.0",
        "environment": "testing",
        "pipeline": {
            "batch_size": 2,
            "width": 1280,
            "height": 720,
            "fps": 30.0
        },
        "detection": {
            "confidence_threshold": 0.5,
            "max_objects": 50
        },
        "alerts": {
            "enabled": True,
            "throttle_seconds": 30
        },
        "sources": [
            {
                "id": "test_source",
                "name": "Test Source",
                "type": "test",
                "uri": "videotestsrc",
                "enabled": True
            }
        ]
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f)
    
    return config_path


@pytest.fixture
def test_config():
    """Create a test AppConfig instance."""
    return AppConfig(
        name="PyDS Test",
        version="0.1.0",
        environment="testing",
        debug=True
    )


@pytest.fixture
def test_source():
    """Create a test video source configuration."""
    return SourceConfig(
        id="test_source_1",
        name="Test Video Source",
        type=SourceType.TEST,
        uri="videotestsrc pattern=smpte",
        enabled=True
    )


@pytest.fixture
def mock_gstreamer():
    """Mock GStreamer components."""
    with patch('gi.repository.Gst') as mock_gst:
        # Mock Gst initialization
        mock_gst.init = Mock()
        mock_gst.version_string = Mock(return_value="1.20.0")
        
        # Mock pipeline creation
        mock_pipeline = MagicMock()
        mock_gst.Pipeline.new = Mock(return_value=mock_pipeline)
        
        # Mock element factory
        mock_element = MagicMock()
        mock_gst.ElementFactory.make = Mock(return_value=mock_element)
        
        # Mock state changes
        mock_gst.StateChangeReturn.SUCCESS = 1
        mock_gst.StateChangeReturn.ASYNC = 2
        mock_gst.StateChangeReturn.FAILURE = 3
        
        # Mock states
        mock_gst.State.NULL = 1
        mock_gst.State.READY = 2
        mock_gst.State.PAUSED = 3
        mock_gst.State.PLAYING = 4
        
        # Mock message types
        mock_gst.MessageType.ERROR = 1
        mock_gst.MessageType.WARNING = 2
        mock_gst.MessageType.INFO = 3
        mock_gst.MessageType.STATE_CHANGED = 4
        mock_gst.MessageType.EOS = 5
        
        yield mock_gst


@pytest.fixture
def mock_deepstream_api():
    """Mock DeepStream API components."""
    with patch('src.utils.deepstream.get_deepstream_api') as mock_get_api:
        mock_api = MagicMock()
        
        # Mock batch metadata
        mock_batch_meta = MagicMock()
        mock_api.get_batch_meta = Mock(return_value=mock_batch_meta)
        
        # Mock frame metadata
        mock_frame_meta = MagicMock()
        mock_frame_meta.frame_num = 1
        mock_frame_meta.source_id = 0
        mock_api.cast_frame_meta = Mock(return_value=mock_frame_meta)
        
        # Mock object metadata
        mock_obj_meta = MagicMock()
        mock_obj_meta.confidence = 0.85
        mock_obj_meta.class_id = 0
        mock_obj_meta.obj_label = "person"
        mock_obj_meta.rect_params = MagicMock()
        mock_obj_meta.rect_params.left = 100
        mock_obj_meta.rect_params.top = 100
        mock_obj_meta.rect_params.width = 200
        mock_obj_meta.rect_params.height = 300
        mock_api.cast_object_meta = Mock(return_value=mock_obj_meta)
        
        mock_get_api.return_value = mock_api
        yield mock_api


@pytest.fixture
def mock_deepstream_info():
    """Mock DeepStream version information."""
    with patch('src.utils.deepstream.get_deepstream_info') as mock_get_info:
        from src.utils.deepstream import DeepStreamInfo, DeepStreamVersion, APIType
        
        mock_info = DeepStreamInfo(
            version=DeepStreamVersion.V7,
            version_string="7.0",
            api_type=APIType.GI_REPOSITORY,
            capabilities={"gpu_support": True, "python_bindings": True}
        )
        
        mock_get_info.return_value = mock_info
        yield mock_info


@pytest.fixture
def sample_detection():
    """Create a sample VideoDetection object."""
    return VideoDetection(
        pattern_name="person",
        confidence=0.85,
        bounding_box=BoundingBox(x=0.1, y=0.1, width=0.2, height=0.3),
        timestamp=datetime.now(),
        frame_number=100,
        source_id="test_source_1",
        metadata={"test": True}
    )


@pytest.fixture
def mock_gpu_available():
    """Mock GPU availability check."""
    with patch('src.utils.deepstream.is_gpu_available', return_value=True):
        yield


@pytest.fixture
def mock_metrics_collector():
    """Mock metrics collector."""
    from src.monitoring.metrics import MetricsCollector
    
    collector = MetricsCollector()
    collector.update_fps = Mock()
    collector.update_latency = Mock()
    collector.record_detection = Mock()
    collector.record_alert = Mock()
    
    with patch('src.monitoring.metrics.get_metrics_collector', return_value=collector):
        yield collector


@pytest.fixture
def mock_alert_manager():
    """Mock alert manager."""
    from src.alerts.manager import AlertManager
    
    manager = MagicMock(spec=AlertManager)
    manager.process_detection = Mock(return_value=asyncio.coroutine(lambda x: None))
    manager.start = Mock(return_value=asyncio.coroutine(lambda: None))
    manager.stop = Mock(return_value=asyncio.coroutine(lambda: None))
    
    with patch('src.alerts.manager.get_alert_manager', return_value=manager):
        yield manager


@pytest.fixture(autouse=True)
def cleanup_singletons():
    """Clean up singleton instances between tests."""
    # Clear any global instances
    import src.utils.deepstream as deepstream_module
    deepstream_module._detector = None
    deepstream_module._api = None
    deepstream_module._info = None
    
    yield
    
    # Clean up after test
    deepstream_module._detector = None
    deepstream_module._api = None
    deepstream_module._info = None


def create_mock_gst_buffer():
    """Create a mock GStreamer buffer for testing."""
    buffer = MagicMock()
    buffer.pts = 0
    buffer.dts = 0
    buffer.duration = 33333333  # ~30fps
    return buffer


def create_mock_batch_meta(num_frames=1, objects_per_frame=2):
    """Create mock DeepStream batch metadata."""
    batch_meta = MagicMock()
    batch_meta.num_frames = num_frames
    
    # Create frame metadata list
    frame_list = []
    for frame_idx in range(num_frames):
        frame_meta = MagicMock()
        frame_meta.frame_num = frame_idx
        frame_meta.source_id = 0
        frame_meta.batch_id = frame_idx
        
        # Create object metadata list
        obj_list = []
        for obj_idx in range(objects_per_frame):
            obj_meta = MagicMock()
            obj_meta.class_id = obj_idx
            obj_meta.confidence = 0.5 + (obj_idx * 0.2)
            obj_meta.obj_label = f"object_{obj_idx}"
            
            # Bounding box
            rect = MagicMock()
            rect.left = 100 + (obj_idx * 50)
            rect.top = 100 + (obj_idx * 50)
            rect.width = 100
            rect.height = 150
            obj_meta.rect_params = rect
            
            obj_list.append(obj_meta)
        
        # Link objects
        frame_meta.obj_meta_list = create_linked_list(obj_list)
        frame_list.append(frame_meta)
    
    # Link frames
    batch_meta.frame_meta_list = create_linked_list(frame_list)
    
    return batch_meta


def create_linked_list(items):
    """Create a mock linked list structure for DeepStream metadata."""
    if not items:
        return None
    
    head = MagicMock()
    head.data = items[0]
    current = head
    
    for item in items[1:]:
        next_node = MagicMock()
        next_node.data = item
        current.next = next_node
        current = next_node
    
    current.next = None
    return head


# Performance testing utilities
@pytest.fixture
def performance_monitor():
    """Monitor test performance metrics."""
    import time
    
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.measurements = {}
        
        def start(self, name):
            self.start_time = time.perf_counter()
            return self
        
        def stop(self, name):
            if self.start_time is None:
                return
            elapsed = time.perf_counter() - self.start_time
            self.measurements[name] = elapsed
            self.start_time = None
            return elapsed
        
        def report(self):
            return self.measurements
    
    return PerformanceMonitor()


# Test data generators
def generate_test_video_path(temp_dir, filename="test_video.mp4"):
    """Generate a test video file path."""
    return str(temp_dir / filename)


def generate_test_detections(count=10, source_id="test_source"):
    """Generate test detection objects."""
    detections = []
    for i in range(count):
        detection = VideoDetection(
            pattern_name=f"object_{i % 3}",
            confidence=0.5 + (i % 5) * 0.1,
            bounding_box=BoundingBox(
                x=0.1 + (i % 3) * 0.2,
                y=0.1 + (i % 4) * 0.2,
                width=0.1,
                height=0.15
            ),
            timestamp=datetime.now(),
            frame_number=i * 10,
            source_id=source_id,
            metadata={"index": i}
        )
        detections.append(detection)
    return detections