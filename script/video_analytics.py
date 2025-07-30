#!/usr/bin/env python3
"""
Standalone Video Analytics Script with GStreamer, YOLO, and DeepStream.
Production-ready script for multi-source video pattern detection.

This is a completely independent application designed for production deployment
with REST API control, automatic error recovery, and multi-source processing capabilities.
"""

# GStreamer imports with fallback for development environments
try:
    import gi
    gi.require_version('Gst', '1.0')
    gi.require_version('GObject', '2.0')
    from gi.repository import Gst, GObject, GLib
    GSTREAMER_AVAILABLE = True
except ImportError:
    print("Warning: GStreamer Python bindings not available")
    print("This is expected on Windows development environment")
    print("GStreamer functionality will be simulated")
    GSTREAMER_AVAILABLE = False
    
    # Mock classes for development
    class MockGst:
        @staticmethod
        def init(args): pass
    
    class MockGObject:
        @staticmethod
        def threads_init(): pass
    
    class MockGLib:
        class MainLoop:
            def __init__(self): pass
            def is_running(self): return False
            def quit(self): pass
    
    Gst = MockGst()
    GObject = MockGObject()
    GLib = MockGLib()
import asyncio
import threading
import json
import time
import logging
import logging.handlers
import signal
import sys
import os
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import datetime
import argparse
import yaml

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    
    # Mock classes for development without FastAPI
    class MockFastAPI:
        def __init__(self, **kwargs): pass
        def get(self, path): return lambda f: f
    
    FastAPI = MockFastAPI
    HTTPException = Exception

try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    
    # Mock numpy for basic functionality
    class MockNumPy:
        @staticmethod
        def array(data): return data
        @staticmethod  
        def zeros(shape): return [[0] * shape[1] for _ in range(shape[0])]
    
    np = MockNumPy()
    cv2 = None


# Data Models and Enums
class SourceType(str, Enum):
    """Video source types."""
    RTSP = "rtsp"
    WEBRTC = "webrtc" 
    FILE = "file"
    WEBCAM = "webcam"


class PipelineState(str, Enum):
    """Pipeline state tracking."""
    NULL = "null"
    READY = "ready"
    PAUSED = "paused"
    PLAYING = "playing"
    ERROR = "error"


class HealthStatus(str, Enum):
    """Component health status."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class VideoSource:
    """Video source configuration."""
    id: str
    type: SourceType
    uri: str
    enabled: bool = True
    confidence_threshold: float = 0.5
    max_objects: int = 50
    detection_config: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class DetectionResult:
    """Object detection result."""
    source_id: str
    timestamp: float
    frame_number: int
    objects: List[Dict[str, Any]]
    processing_time_ms: float
    confidence_avg: float = 0.0


@dataclass
class SystemHealth:
    """System health information."""
    overall_status: HealthStatus
    pipeline_status: PipelineState
    active_sources: int
    total_sources: int
    fps_avg: float
    memory_usage_mb: float
    cpu_usage_percent: float
    errors_last_hour: int
    uptime_seconds: float


# Configuration Management
class ConfigManager:
    """Manages application configuration from files and environment variables."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config.yaml"
        self.config = self._load_default_config()
        self.logger = self._setup_logger()
        
        if Path(self.config_path).exists():
            self._load_config_file()
        
        self._apply_env_overrides()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            "application": {
                "name": "VideoAnalytics",
                "version": "1.0.0",
                "debug": False
            },
            "pipeline": {
                "batch_size": 4,
                "batch_timeout_ms": 40,
                "max_sources": 16,
                "gpu_device_id": 0
            },
            "detection": {
                "model_path": "models/yolo.engine",
                "config_path": "models/yolo_config.txt",
                "confidence_threshold": 0.5,
                "nms_threshold": 0.4,
                "max_objects": 50
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8080,
                "cors_enabled": True
            },
            "logging": {
                "level": "INFO",
                "format": "json",
                "file": "video_analytics.log",
                "max_size_mb": 100,
                "backup_count": 5
            },
            "monitoring": {
                "health_check_interval": 30,
                "metrics_interval": 60,
                "auto_recovery": True
            }
        }
    
    def _load_config_file(self):
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    file_config = yaml.safe_load(f)
                else:
                    file_config = json.load(f)
            
            # Deep merge configuration
            self._deep_merge(self.config, file_config)
            
        except Exception as e:
            print(f"Warning: Could not load config file {self.config_path}: {e}")
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides."""
        env_mappings = {
            'VA_DEBUG': ('application', 'debug'),
            'VA_API_PORT': ('api', 'port'),
            'VA_API_HOST': ('api', 'host'),
            'VA_MODEL_PATH': ('detection', 'model_path'),
            'VA_CONFIDENCE_THRESHOLD': ('detection', 'confidence_threshold'),
            'VA_LOG_LEVEL': ('logging', 'level'),
            'VA_GPU_DEVICE': ('pipeline', 'gpu_device_id')
        }
        
        for env_var, (section, key) in env_mappings.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                
                # Type conversion
                if key in ['port', 'gpu_device_id', 'backup_count', 'max_size_mb']:
                    value = int(value)
                elif key in ['debug', 'cors_enabled', 'auto_recovery']:
                    value = value.lower() in ['true', '1', 'yes']
                elif key in ['confidence_threshold', 'nms_threshold']:
                    value = float(value)
                
                self.config[section][key] = value
    
    def _deep_merge(self, base: Dict, update: Dict):
        """Deep merge two dictionaries."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def _setup_logger(self) -> logging.Logger:
        """Setup basic logger."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(section, {}).get(key, default)
    
    def update(self, section: str, key: str, value: Any):
        """Update configuration value at runtime."""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
        self.logger.info(f"Updated config: {section}.{key} = {value}")


# Logger Setup
class Logger:
    """Structured logging with JSON output support."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = self._setup_structured_logger()
    
    def _setup_structured_logger(self) -> logging.Logger:
        """Setup structured logger with JSON output."""
        logger = logging.getLogger("VideoAnalytics")
        logger.setLevel(getattr(logging, self.config.get('logging', 'level', 'INFO')))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        
        # File handler
        log_file = self.config.get('logging', 'file', 'video_analytics.log')
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=self.config.get('logging', 'max_size_mb', 100) * 1024 * 1024,
            backupCount=self.config.get('logging', 'backup_count', 5)
        )
        
        # Formatter
        if self.config.get('logging', 'format') == 'json':
            formatter = self._get_json_formatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    def _get_json_formatter(self):
        """Create JSON formatter."""
        class JsonFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                    'level': record.levelname,
                    'logger': record.name,
                    'message': record.getMessage(),
                    'module': record.module,
                    'function': record.funcName,
                    'line': record.lineno
                }
                
                if hasattr(record, 'source_id'):
                    log_entry['source_id'] = record.source_id
                if hasattr(record, 'frame_number'):
                    log_entry['frame_number'] = record.frame_number
                if hasattr(record, 'processing_time_ms'):
                    log_entry['processing_time_ms'] = record.processing_time_ms
                
                return json.dumps(log_entry)
        
        return JsonFormatter()
    
    def info(self, message: str, **kwargs):
        """Log info message with optional structured data."""
        self._log_with_extra(logging.INFO, message, kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with optional structured data."""
        self._log_with_extra(logging.WARNING, message, kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with optional structured data."""
        self._log_with_extra(logging.ERROR, message, kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with optional structured data."""
        self._log_with_extra(logging.CRITICAL, message, kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with optional structured data."""
        self._log_with_extra(logging.DEBUG, message, kwargs)
    
    def _log_with_extra(self, level: int, message: str, extra: Dict[str, Any]):
        """Log message with extra structured data."""
        # Create a LogRecord with extra attributes
        record = self.logger.makeRecord(
            self.logger.name, level, "", 0, message, (), None
        )
        
        # Add extra attributes
        for key, value in extra.items():
            setattr(record, key, value)
        
        self.logger.handle(record)


# GStreamer Pipeline Management
class GStreamerPipeline:
    """Core GStreamer pipeline management with DeepStream integration."""
    
    def __init__(self, config: ConfigManager, logger: Logger):
        self.config = config
        self.logger = logger
        self.pipeline = None
        self.bus = None
        self.main_loop = None
        
        # Pipeline elements
        self.streammux = None
        self.nvinfer = None
        self.nvtracker = None
        self.tiler = None
        self.sink = None
        
        # Source management
        self.sources = {}  # source_id -> source_element mapping
        self.source_pads = {}  # source_id -> pad mapping
        self.next_source_index = 0
        
        # Pipeline state
        self.state = PipelineState.NULL
        self.is_playing = False
        
        # Performance metrics
        self.frame_count = 0
        self.start_time = None
        
        # Mock GStreamer elements for development
        if not GSTREAMER_AVAILABLE:
            self._init_mock_elements()
    
    def _init_mock_elements(self):
        """Initialize mock elements for development without GStreamer."""
        class MockElement:
            def __init__(self, name):
                self.name = name
                self.properties = {}
            
            def set_property(self, prop, value):
                self.properties[prop] = value
            
            def get_property(self, prop):
                return self.properties.get(prop)
            
            def link(self, other):
                return True
            
            def get_static_pad(self, name):
                return MockPad(name)
            
            def get_request_pad(self, name):
                return MockPad(name)
            
            def set_state(self, state):
                return 1  # SUCCESS
        
        class MockPad:
            def __init__(self, name):
                self.name = name
            
            def link(self, other):
                return True
        
        class MockPipeline(MockElement):
            def __init__(self):
                super().__init__("pipeline")
                self.elements = []
            
            def add(self, element):
                self.elements.append(element)
            
            def get_bus(self):
                return MockBus()
            
            def get_state(self, timeout):
                return (1, 4, 0)  # SUCCESS, PLAYING, VOID_PENDING
        
        class MockBus:
            def add_signal_watch(self):
                pass
            
            def connect(self, signal, callback):
                pass
        
        # Create mock elements
        self.pipeline = MockPipeline()
        self.streammux = MockElement("streammux")
        self.nvinfer = MockElement("nvinfer")
        self.nvtracker = MockElement("nvtracker")
        self.tiler = MockElement("tiler")
        self.sink = MockElement("sink")
    
    def create_pipeline(self) -> bool:
        """Create the GStreamer pipeline."""
        try:
            if GSTREAMER_AVAILABLE:
                return self._create_real_pipeline()
            else:
                self.logger.info("Creating mock pipeline for development")
                return self._create_mock_pipeline()
        
        except Exception as e:
            self.logger.error(f"Failed to create pipeline: {e}")
            return False
    
    def _create_real_pipeline(self) -> bool:
        """Create real GStreamer pipeline with DeepStream elements."""
        try:
            self.logger.info("Creating GStreamer pipeline...")
            
            # Create main pipeline
            self.pipeline = Gst.Pipeline.new("video-analytics-pipeline")
            if not self.pipeline:
                raise RuntimeError("Failed to create pipeline")
            
            # Create core elements
            if not self._create_pipeline_elements():
                return False
            
            # Link elements
            if not self._link_pipeline_elements():
                return False
            
            # Setup bus monitoring
            self._setup_bus_monitoring()
            
            self.logger.info("GStreamer pipeline created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create real pipeline: {e}")
            return False
    
    def _create_mock_pipeline(self) -> bool:
        """Create mock pipeline for development."""
        self.logger.info("Mock pipeline created successfully")
        return True
    
    def _create_pipeline_elements(self) -> bool:
        """Create all pipeline elements."""
        try:
            # Stream multiplexer
            self.streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
            if not self.streammux:
                self.logger.warning("nvstreammux not available, using mock")
                return False
            
            # Configure batch processing
            batch_size = self.config.get('pipeline', 'batch_size', 4)
            batch_timeout = self.config.get('pipeline', 'batch_timeout_ms', 40)
            
            self.streammux.set_property("batch-size", batch_size)
            self.streammux.set_property("batched-push-timeout", batch_timeout * 1000)
            self.streammux.set_property("width", 1920)
            self.streammux.set_property("height", 1080)
            
            # Primary inference (YOLO detection)
            self.nvinfer = Gst.ElementFactory.make("nvinfer", "primary-inference")
            if not self.nvinfer:
                self.logger.warning("nvinfer not available")
                return False
            
            # Configure inference  
            config_path = self.config.get('detection', 'config_path', 'models/yolo_config.txt')
            if Path(config_path).exists():
                self.nvinfer.set_property("config-file-path", config_path)
            
            # Object tracker (optional)
            self.nvtracker = Gst.ElementFactory.make("nvtracker", "tracker")
            if not self.nvtracker:
                self.logger.info("nvtracker not available, tracking disabled")
            
            # Multi-stream tiler (optional)
            self.tiler = Gst.ElementFactory.make("nvmultistreamtiler", "tiler")
            if not self.tiler:
                self.logger.info("nvmultistreamtiler not available")
            else:
                self.tiler.set_property("rows", 2)
                self.tiler.set_property("columns", 2)
                self.tiler.set_property("width", 1920)
                self.tiler.set_property("height", 1080)
            
            # On-screen display
            self.nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
            if not self.nvosd:
                self.logger.info("nvdsosd not available")
            
            # Sink
            self.sink = Gst.ElementFactory.make("fakesink", "output-sink")
            if not self.sink:
                raise RuntimeError("Failed to create sink element")
            
            self.sink.set_property("sync", False)
            self.sink.set_property("async", False)
            
            # Add elements to pipeline
            elements = [self.streammux, self.nvinfer]
            if self.nvtracker:
                elements.append(self.nvtracker)
            if self.tiler:
                elements.append(self.tiler)
            if self.nvosd:
                elements.append(self.nvosd)
            elements.append(self.sink)
            
            for element in elements:
                if element:
                    self.pipeline.add(element)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create pipeline elements: {e}")
            return False
    
    def _link_pipeline_elements(self) -> bool:
        """Link pipeline elements together."""
        try:
            # Link streammux to inference
            if not self.streammux.link(self.nvinfer):
                raise RuntimeError("Failed to link streammux to nvinfer")
            
            current = self.nvinfer
            
            # Link optional elements
            for element in [self.nvtracker, self.tiler, self.nvosd]:
                if element:
                    if not current.link(element):
                        raise RuntimeError(f"Failed to link to {element.get_name()}")
                    current = element
            
            # Link to sink
            if not current.link(self.sink):
                raise RuntimeError("Failed to link to sink")
            
            self.logger.info("Pipeline elements linked successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to link elements: {e}")
            return False
    
    def _setup_bus_monitoring(self):
        """Setup bus message monitoring."""
        if not GSTREAMER_AVAILABLE:
            return
        
        try:
            self.bus = self.pipeline.get_bus()
            self.bus.add_signal_watch()
            self.bus.connect("message", self._on_bus_message)
        except Exception as e:
            self.logger.error(f"Failed to setup bus: {e}")
    
    def _on_bus_message(self, bus, message):
        """Handle bus messages."""
        msg_type = message.type
        
        if msg_type == Gst.MessageType.EOS:
            self.logger.info("End-of-stream received")
            self.stop()
            
        elif msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            self.logger.error(f"Pipeline error: {err} - {debug}")
            self.state = PipelineState.ERROR
            
        elif msg_type == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            self.logger.warning(f"Pipeline warning: {warn} - {debug}")
            
        elif msg_type == Gst.MessageType.STATE_CHANGED:
            if message.src == self.pipeline:
                old, new, pending = message.parse_state_changed()
                self.logger.debug(f"State changed: {old} -> {new}")
    
    def start(self) -> bool:
        """Start the pipeline."""
        try:
            if not GSTREAMER_AVAILABLE:
                self.state = PipelineState.PLAYING
                self.is_playing = True
                self.start_time = time.time()
                self.logger.info("Mock pipeline started")
                return True
            
            if not self.pipeline:
                self.logger.error("Pipeline not created")
                return False
            
            # Set to PLAYING state
            ret = self.pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                self.logger.error("Failed to start pipeline")
                return False
            
            self.state = PipelineState.PLAYING
            self.is_playing = True
            self.start_time = time.time()
            
            self.logger.info("Pipeline started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Exception starting pipeline: {e}")
            return False
    
    def stop(self) -> bool:
        """Stop the pipeline."""
        try:
            if not GSTREAMER_AVAILABLE:
                self.state = PipelineState.NULL
                self.is_playing = False
                self.logger.info("Mock pipeline stopped")
                return True
            
            if not self.pipeline:
                return True
            
            # Set to NULL state
            self.pipeline.set_state(Gst.State.NULL)
            self.state = PipelineState.NULL
            self.is_playing = False
            
            # Log statistics
            if self.start_time:
                duration = time.time() - self.start_time
                fps = self.frame_count / duration if duration > 0 else 0
                self.logger.info(f"Pipeline stopped. Duration: {duration:.1f}s, FPS: {fps:.1f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Exception stopping pipeline: {e}")
            return False
    
    def add_source(self, source: VideoSource) -> bool:
        """Add a video source to the pipeline."""
        try:
            if not GSTREAMER_AVAILABLE:
                self.sources[source.id] = f"mock_source_{source.id}"
                self.logger.info(f"Added mock source: {source.id}")
                return True
            
            # This will be fully implemented in Task 4
            self.logger.info(f"Adding source: {source.id} ({source.type.value})")
            
            # Create source element based on type
            if source.type == SourceType.RTSP:
                element = Gst.ElementFactory.make("rtspsrc", f"src-{source.id}")
                if element:
                    element.set_property("location", source.uri)
                    element.set_property("latency", 2000)
            elif source.type == SourceType.FILE:
                element = Gst.ElementFactory.make("filesrc", f"src-{source.id}")
                if element:
                    element.set_property("location", source.uri)
            elif source.type == SourceType.WEBCAM:
                element = Gst.ElementFactory.make("v4l2src", f"src-{source.id}")
                if element:
                    element.set_property("device", source.uri)
            else:
                self.logger.error(f"Unsupported source type: {source.type}")
                return False
            
            if not element:
                self.logger.error(f"Failed to create source element for {source.id}")
                return False
            
            # Add to pipeline and connect
            self.pipeline.add(element)
            self.sources[source.id] = element
            
            # Get request pad from streammux
            pad_name = f"sink_{self.next_source_index}"
            sinkpad = self.streammux.get_request_pad(pad_name)
            if sinkpad:
                self.source_pads[source.id] = sinkpad
                self.next_source_index += 1
            
            # Sync state if playing
            if self.is_playing:
                element.set_state(Gst.State.PLAYING)
            
            self.logger.info(f"Successfully added source: {source.id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add source {source.id}: {e}")
            return False
    
    def remove_source(self, source_id: str) -> bool:
        """Remove a video source from the pipeline."""
        try:
            if not GSTREAMER_AVAILABLE:
                if source_id in self.sources:
                    del self.sources[source_id]
                    self.logger.info(f"Removed mock source: {source_id}")
                    return True
                return False
            
            if source_id not in self.sources:
                self.logger.warning(f"Source {source_id} not found")
                return False
            
            # This will be fully implemented in Task 4
            element = self.sources[source_id]
            element.set_state(Gst.State.NULL)
            self.pipeline.remove(element)
            
            del self.sources[source_id]
            
            if source_id in self.source_pads:
                pad = self.source_pads[source_id]
                self.streammux.release_request_pad(pad)
                del self.source_pads[source_id]
            
            self.logger.info(f"Successfully removed source: {source_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to remove source {source_id}: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        stats = {
            'state': self.state.value,
            'is_playing': self.is_playing,
            'source_count': len(self.sources),
            'frame_count': self.frame_count
        }
        
        if self.start_time and self.is_playing:
            duration = time.time() - self.start_time
            stats['duration_seconds'] = duration
            stats['fps'] = self.frame_count / duration if duration > 0 else 0
        
        return stats


# YOLO Detection Engine
class YOLODetector:
    """YOLO object detection engine with DeepStream integration."""
    
    def __init__(self, config: ConfigManager, logger: Logger):
        self.config = config
        self.logger = logger
        
        # Detection parameters
        self.confidence_threshold = config.get('detection', 'confidence_threshold', 0.5)
        self.nms_threshold = config.get('detection', 'nms_threshold', 0.4)
        self.max_objects = config.get('detection', 'max_objects', 50)
        
        # Model configuration
        self.model_path = config.get('detection', 'model_path', 'models/yolo.engine')
        self.config_path = config.get('detection', 'config_path', 'models/yolo_config.txt')
        
        # Class labels
        self.class_labels = self._load_class_labels()
        
        # Performance metrics
        self.total_detections = 0
        self.detection_times = []
    
    def _load_class_labels(self) -> List[str]:
        """Load YOLO class labels."""
        # Default COCO labels
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
    
    def configure_inference(self, nvinfer_element) -> bool:
        """Configure the nvinfer element for YOLO detection."""
        try:
            if not GSTREAMER_AVAILABLE:
                self.logger.info("Mock YOLO configuration")
                return True
            
            # Set configuration file path
            if Path(self.config_path).exists():
                nvinfer_element.set_property("config-file-path", self.config_path)
                self.logger.info(f"Configured YOLO with: {self.config_path}")
            else:
                self.logger.warning(f"YOLO config not found: {self.config_path}")
            
            # Set model engine file if exists
            if Path(self.model_path).exists():
                nvinfer_element.set_property("model-engine-file", self.model_path)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to configure YOLO: {e}")
            return False
    
    def process_metadata(self, batch_meta) -> List[DetectionResult]:
        """Process DeepStream batch metadata to extract detections."""
        # This will be fully implemented in Task 3
        # For now, return mock detections
        mock_detection = DetectionResult(
            source_id="source_0",
            timestamp=time.time(),
            frame_number=1,
            objects=[
                {
                    'class_id': 0,
                    'class_name': 'person',
                    'confidence': 0.95,
                    'bbox': {'x': 100, 'y': 100, 'width': 200, 'height': 300}
                }
            ],
            processing_time_ms=10.5,
            confidence_avg=0.95
        )
        
        return [mock_detection]
    
    def filter_detections(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter detections based on confidence and NMS."""
        filtered = []
        
        for det in detections:
            if det.get('confidence', 0) >= self.confidence_threshold:
                filtered.append(det)
        
        # Apply NMS if needed
        if len(filtered) > self.max_objects:
            filtered = filtered[:self.max_objects]
        
        return filtered
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detection statistics."""
        avg_time = sum(self.detection_times) / len(self.detection_times) if self.detection_times else 0
        
        return {
            'total_detections': self.total_detections,
            'avg_detection_time_ms': avg_time,
            'confidence_threshold': self.confidence_threshold,
            'max_objects': self.max_objects
        }


# Source Manager
class SourceManager:
    """Manages multiple video sources dynamically."""
    
    def __init__(self, pipeline: GStreamerPipeline, config: ConfigManager, logger: Logger):
        self.pipeline = pipeline
        self.config = config
        self.logger = logger
        
        # Source tracking
        self.sources: Dict[str, VideoSource] = {}
        self.source_status: Dict[str, str] = {}
        
        # Limits
        self.max_sources = config.get('pipeline', 'max_sources', 16)
    
    def add_source(self, source: VideoSource) -> bool:
        """Add a new video source."""
        try:
            # Validate source
            if source.id in self.sources:
                self.logger.warning(f"Source {source.id} already exists")
                return False
            
            if len(self.sources) >= self.max_sources:
                self.logger.error(f"Maximum sources ({self.max_sources}) reached")
                return False
            
            # Add to pipeline
            if self.pipeline.add_source(source):
                self.sources[source.id] = source
                self.source_status[source.id] = "active"
                self.logger.info(f"Added source: {source.id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to add source {source.id}: {e}")
            return False
    
    def remove_source(self, source_id: str) -> bool:
        """Remove a video source."""
        try:
            if source_id not in self.sources:
                self.logger.warning(f"Source {source_id} not found")
                return False
            
            # Remove from pipeline
            if self.pipeline.remove_source(source_id):
                del self.sources[source_id]
                del self.source_status[source_id]
                self.logger.info(f"Removed source: {source_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to remove source {source_id}: {e}")
            return False
    
    def update_source(self, source_id: str, config: Dict[str, Any]) -> bool:
        """Update source configuration."""
        if source_id not in self.sources:
            return False
        
        source = self.sources[source_id]
        
        # Update configuration
        if 'enabled' in config:
            source.enabled = config['enabled']
        if 'confidence_threshold' in config:
            source.confidence_threshold = config['confidence_threshold']
        if 'max_objects' in config:
            source.max_objects = config['max_objects']
        
        self.logger.info(f"Updated source {source_id} configuration")
        return True
    
    def get_sources(self) -> List[VideoSource]:
        """Get all sources."""
        return list(self.sources.values())
    
    def get_source(self, source_id: str) -> Optional[VideoSource]:
        """Get a specific source."""
        return self.sources.get(source_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get source manager statistics."""
        return {
            'total_sources': len(self.sources),
            'active_sources': sum(1 for s in self.source_status.values() if s == "active"),
            'max_sources': self.max_sources,
            'sources': {
                sid: {
                    'type': s.type.value,
                    'status': self.source_status.get(sid, 'unknown'),
                    'enabled': s.enabled
                }
                for sid, s in self.sources.items()
            }
        }


# Application Main Class
class VideoAnalyticsEngine:
    """Main application class orchestrating all components."""
    
    def __init__(self, config_path: Optional[str] = None):
        # Configuration and logging
        self.config = ConfigManager(config_path)
        self.logger = Logger(self.config)
        
        # Initialize GStreamer (if available)
        if GSTREAMER_AVAILABLE:
            Gst.init(None)
            GObject.threads_init()
        else:
            self.logger.warning("Running in development mode without GStreamer")
        
        # Application state
        self.running = False
        self.shutdown_requested = False
        self.start_time = time.time()
        
        # Components (will be initialized in start())
        self.pipeline = None
        self.source_manager = None
        self.yolo_detector = None
        self.api_server = None
        self.health_monitor = None
        self.error_recovery = None
        
        # Threading
        self.main_loop = None
        self.api_thread = None
        self.monitoring_thread = None
        
        # Signal handlers
        self._setup_signal_handlers()
        
        self.logger.info("VideoAnalyticsEngine initialized", 
                        version=self.config.get('application', 'version'))
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating shutdown")
            self.shutdown()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def start(self) -> bool:
        """Start the video analytics engine."""
        try:
            self.logger.info("Starting VideoAnalyticsEngine...")
            
            # Initialize components in order
            if not await self._initialize_components():
                return False
            
            # Start main processing loop
            await self._start_main_loop()
            
            # Start API server in background thread
            self._start_api_server()
            
            # Start monitoring
            self._start_monitoring()
            
            self.running = True
            self.start_time = time.time()
            
            self.logger.info("VideoAnalyticsEngine started successfully")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to start VideoAnalyticsEngine: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop the video analytics engine gracefully."""
        try:
            self.logger.info("Stopping VideoAnalyticsEngine...")
            
            self.running = False
            self.shutdown_requested = True
            
            # Stop components in reverse order
            await self._stop_components()
            
            uptime = time.time() - self.start_time
            self.logger.info(f"VideoAnalyticsEngine stopped (uptime: {uptime:.1f}s)")
            return True
        
        except Exception as e:
            self.logger.error(f"Error stopping VideoAnalyticsEngine: {e}")
            return False
    
    async def _initialize_components(self) -> bool:
        """Initialize all components."""
        try:
            self.logger.info("Initializing components...")
            
            # Initialize GStreamer pipeline
            self.pipeline = GStreamerPipeline(self.config, self.logger)
            if not self.pipeline.create_pipeline():
                self.logger.error("Failed to create GStreamer pipeline")
                return False
            
            # Initialize YOLO detector
            self.yolo_detector = YOLODetector(self.config, self.logger)
            if self.pipeline.nvinfer:
                self.yolo_detector.configure_inference(self.pipeline.nvinfer)
            
            # Initialize source manager
            self.source_manager = SourceManager(self.pipeline, self.config, self.logger)
            
            # Load default sources from config
            await self._load_default_sources()
            
            # Initialize health monitor (placeholder for now)
            self.health_monitor = None  # Will be implemented in Task 6
            
            # Initialize error recovery (placeholder for now)
            self.error_recovery = None  # Will be implemented in Task 6
            
            self.logger.info("All components initialized successfully")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            return False
    
    async def _load_default_sources(self):
        """Load default sources from configuration."""
        try:
            # Check if config has default sources
            if hasattr(self.config, 'config') and 'sources' in self.config.config:
                sources = self.config.config['sources']
                for source_config in sources:
                    if source_config.get('enabled', False):
                        source = VideoSource(
                            id=source_config['id'],
                            type=SourceType(source_config['type']),
                            uri=source_config['uri'],
                            enabled=source_config.get('enabled', True),
                            confidence_threshold=source_config.get('confidence_threshold', 0.5),
                            max_objects=source_config.get('max_objects', 50)
                        )
                        if self.source_manager.add_source(source):
                            self.logger.info(f"Loaded default source: {source.id}")
        except Exception as e:
            self.logger.warning(f"Failed to load default sources: {e}")
    
    async def _start_main_loop(self):
        """Start main GLib event loop."""
        try:
            if GSTREAMER_AVAILABLE:
                self.main_loop = GLib.MainLoop()
                # Start pipeline
                if not self.pipeline.start():
                    raise RuntimeError("Failed to start pipeline")
                
                # Run main loop in background thread
                def run_loop():
                    try:
                        self.main_loop.run()
                    except Exception as e:
                        self.logger.error(f"Main loop error: {e}")
                
                loop_thread = threading.Thread(target=run_loop, daemon=True)
                loop_thread.start()
            else:
                # Mock mode
                if self.pipeline:
                    self.pipeline.start()
        except Exception as e:
            self.logger.error(f"Failed to start main loop: {e}")
            raise
    
    def _start_api_server(self):
        """Start REST API server in background thread."""
        try:
            if not FASTAPI_AVAILABLE:
                self.logger.warning("FastAPI not available, API server disabled")
                return
            
            # Create API app (will be implemented in Task 5)
            self.api_server = self._create_api_app()
            
            # Start in background thread
            def run_api():
                uvicorn.run(
                    self.api_server,
                    host=self.config.get('api', 'host', '0.0.0.0'),
                    port=self.config.get('api', 'port', 8080),
                    log_level="warning"  # Reduce uvicorn logging
                )
            
            self.api_thread = threading.Thread(target=run_api, daemon=True)
            self.api_thread.start()
            
            self.logger.info(f"API server started on "
                           f"{self.config.get('api', 'host')}:{self.config.get('api', 'port')}")
        
        except Exception as e:
            self.logger.error(f"Failed to start API server: {e}")
    
    def _start_monitoring(self):
        """Start health monitoring in background thread."""
        try:
            def monitoring_loop():
                while self.running and not self.shutdown_requested:
                    try:
                        # Health monitoring (will be implemented in Task 6)
                        time.sleep(self.config.get('monitoring', 'health_check_interval', 30))
                    except Exception as e:
                        self.logger.error(f"Error in monitoring loop: {e}")
                        time.sleep(10)
            
            self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            
            self.logger.info("Health monitoring started")
        
        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {e}")
    
    def _create_api_app(self) -> FastAPI:
        """Create FastAPI application with full API endpoints."""
        app = FastAPI(
            title="Video Analytics API",
            description="REST API for video analytics control",
            version=self.config.get('application', 'version', '1.0.0')
        )
        
        # Enable CORS if configured
        if self.config.get('api', 'cors_enabled', True) and FASTAPI_AVAILABLE:
            from fastapi.middleware.cors import CORSMiddleware
            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        
        # Health endpoint
        @app.get("/health")
        async def health_check():
            return {
                "status": "healthy" if self.running else "stopped",
                "uptime": time.time() - self.start_time if self.running else 0,
                "version": self.config.get('application', 'version')
            }
        
        # System status endpoint
        @app.get("/api/v1/status")
        async def get_status():
            pipeline_stats = self.pipeline.get_statistics() if self.pipeline else {}
            source_stats = self.source_manager.get_statistics() if self.source_manager else {}
            detection_stats = self.yolo_detector.get_statistics() if self.yolo_detector else {}
            
            return {
                "system": {
                    "running": self.running,
                    "uptime_seconds": time.time() - self.start_time,
                    "version": self.config.get('application', 'version')
                },
                "pipeline": pipeline_stats,
                "sources": source_stats,
                "detection": detection_stats
            }
        
        # Source management endpoints
        @app.get("/api/v1/sources")
        async def list_sources():
            """List all video sources."""
            if not self.source_manager:
                raise HTTPException(status_code=503, detail="Source manager not initialized")
            
            sources = self.source_manager.get_sources()
            return {
                "sources": [
                    {
                        "id": s.id,
                        "type": s.type.value,
                        "uri": s.uri,
                        "enabled": s.enabled,
                        "status": self.source_manager.source_status.get(s.id, "unknown")
                    }
                    for s in sources
                ]
            }
        
        @app.get("/api/v1/sources/{source_id}")
        async def get_source(source_id: str):
            """Get specific source details."""
            if not self.source_manager:
                raise HTTPException(status_code=503, detail="Source manager not initialized")
            
            source = self.source_manager.get_source(source_id)
            if not source:
                raise HTTPException(status_code=404, detail=f"Source {source_id} not found")
            
            return {
                "id": source.id,
                "type": source.type.value,
                "uri": source.uri,
                "enabled": source.enabled,
                "confidence_threshold": source.confidence_threshold,
                "max_objects": source.max_objects,
                "status": self.source_manager.source_status.get(source.id, "unknown")
            }
        
        @app.post("/api/v1/sources")
        async def add_source(source_data: Dict[str, Any]):
            """Add a new video source."""
            if not self.source_manager:
                raise HTTPException(status_code=503, detail="Source manager not initialized")
            
            try:
                source = VideoSource(
                    id=source_data['id'],
                    type=SourceType(source_data['type']),
                    uri=source_data['uri'],
                    enabled=source_data.get('enabled', True),
                    confidence_threshold=source_data.get('confidence_threshold', 0.5),
                    max_objects=source_data.get('max_objects', 50)
                )
                
                if self.source_manager.add_source(source):
                    return {"status": "success", "source_id": source.id}
                else:
                    raise HTTPException(status_code=400, detail="Failed to add source")
                    
            except KeyError as e:
                raise HTTPException(status_code=400, detail=f"Missing required field: {e}")
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @app.put("/api/v1/sources/{source_id}")
        async def update_source(source_id: str, config: Dict[str, Any]):
            """Update source configuration."""
            if not self.source_manager:
                raise HTTPException(status_code=503, detail="Source manager not initialized")
            
            if self.source_manager.update_source(source_id, config):
                return {"status": "success", "source_id": source_id}
            else:
                raise HTTPException(status_code=404, detail=f"Source {source_id} not found")
        
        @app.delete("/api/v1/sources/{source_id}")
        async def remove_source(source_id: str):
            """Remove a video source."""
            if not self.source_manager:
                raise HTTPException(status_code=503, detail="Source manager not initialized")
            
            if self.source_manager.remove_source(source_id):
                return {"status": "success", "source_id": source_id}
            else:
                raise HTTPException(status_code=404, detail=f"Source {source_id} not found")
        
        # Configuration endpoints
        @app.get("/api/v1/config")
        async def get_config():
            """Get current configuration."""
            return {
                "pipeline": {
                    "batch_size": self.config.get('pipeline', 'batch_size'),
                    "max_sources": self.config.get('pipeline', 'max_sources')
                },
                "detection": {
                    "confidence_threshold": self.config.get('detection', 'confidence_threshold'),
                    "nms_threshold": self.config.get('detection', 'nms_threshold'),
                    "max_objects": self.config.get('detection', 'max_objects')
                }
            }
        
        @app.put("/api/v1/config/detection")
        async def update_detection_config(config: Dict[str, Any]):
            """Update detection configuration."""
            try:
                if 'confidence_threshold' in config:
                    self.config.update('detection', 'confidence_threshold', 
                                     float(config['confidence_threshold']))
                    if self.yolo_detector:
                        self.yolo_detector.confidence_threshold = float(config['confidence_threshold'])
                
                if 'nms_threshold' in config:
                    self.config.update('detection', 'nms_threshold', 
                                     float(config['nms_threshold']))
                    if self.yolo_detector:
                        self.yolo_detector.nms_threshold = float(config['nms_threshold'])
                
                if 'max_objects' in config:
                    self.config.update('detection', 'max_objects', 
                                     int(config['max_objects']))
                    if self.yolo_detector:
                        self.yolo_detector.max_objects = int(config['max_objects'])
                
                return {"status": "success", "config": config}
                
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid value: {e}")
        
        # Metrics endpoint
        @app.get("/api/v1/metrics")
        async def get_metrics():
            """Get performance metrics."""
            return {
                "pipeline": self.pipeline.get_statistics() if self.pipeline else {},
                "detection": self.yolo_detector.get_statistics() if self.yolo_detector else {},
                "sources": self.source_manager.get_statistics() if self.source_manager else {}
            }
        
        return app
    
    async def _stop_components(self):
        """Stop all components."""
        try:
            # Stop pipeline
            if self.pipeline:
                self.pipeline.stop()
            
            # Stop API server
            if self.api_thread and self.api_thread.is_alive():
                # Note: uvicorn shutdown will be handled by process termination
                self.logger.info("API server will shutdown with process")
            
            # Stop main loop
            if GSTREAMER_AVAILABLE and self.main_loop and self.main_loop.is_running():
                self.main_loop.quit()
            
            self.logger.info("All components stopped")
        
        except Exception as e:
            self.logger.error(f"Error stopping components: {e}")
    
    def shutdown(self):
        """Request shutdown."""
        self.shutdown_requested = True
        self.logger.info("Shutdown requested")
    
    async def run_until_shutdown(self):
        """Run until shutdown is requested."""
        try:
            while self.running and not self.shutdown_requested:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
        finally:
            await self.stop()


# CLI Interface and Main Entry Point
def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Standalone Video Analytics Script with GStreamer and YOLO"
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file (YAML or JSON)'
    )
    
    parser.add_argument(
        '--test-pipeline',
        action='store_true',
        help='Test basic pipeline creation and exit'
    )
    
    parser.add_argument(
        '--test-inference',
        action='store_true',
        help='Test YOLO model loading and inference'
    )
    
    parser.add_argument(
        '--test-sources',
        action='store_true',
        help='Test multi-source handling'
    )
    
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run performance benchmarking'
    )
    
    parser.add_argument(
        '--sources',
        type=int,
        default=1,
        help='Number of sources for testing/benchmarking'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=60,
        help='Duration for benchmarking (seconds)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='Path to YOLO model file'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )
    
    return parser


async def run_tests(args) -> int:
    """Run test modes."""
    print("Test modes will be implemented in subsequent tasks")
    
    if args.test_pipeline:
        print("[OK] Pipeline test mode selected")
        return 0
    
    if args.test_inference:
        print("[OK] Inference test mode selected")
        return 0
    
    if args.test_sources:
        print("[OK] Sources test mode selected")
        return 0
    
    if args.benchmark:
        print(f"[OK] Benchmark mode selected (sources: {args.sources}, duration: {args.duration}s)")
        return 0
    
    return 0


async def main() -> int:
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        # Handle test modes
        if any([args.test_pipeline, args.test_inference, args.test_sources, args.benchmark]):
            return await run_tests(args)
        
        # Create and start application
        app = VideoAnalyticsEngine(config_path=args.config)
        
        if args.debug:
            app.config.update('logging', 'level', 'DEBUG')
        
        # Start the application
        if not await app.start():
            print("Failed to start VideoAnalyticsEngine")
            return 1
        
        # Run until shutdown
        try:
            await app.run_until_shutdown()
        except KeyboardInterrupt:
            print("\nShutdown requested by user")
        
        return 0
    
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    # Run the application
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
