name: "Robust DeepStream Inference System"
description: |

## Purpose
Implement a production-ready, multi-source video analytics system using NVIDIA DeepStream and GStreamer for real-time pattern detection. The system must handle custom detection strategies, manage alerts with intelligent throttling, provide automatic error recovery, and support backwards compatibility across DeepStream versions 5.x through 7.x.

## Core Principles
1. **Context is King**: Leverage proven DeepStream patterns and GStreamer best practices
2. **Validation Loops**: Comprehensive testing with real video sources and synthetic data
3. **Information Dense**: Use established NVIDIA community patterns and proven architectures
4. **Progressive Success**: Build incrementally from basic pipeline to full multi-source system
5. **Backwards Compatibility**: Support DeepStream 5.x through 7.x with runtime version detection

---

## Goal

Build a robust, standalone Python application that processes multiple video streams simultaneously using NVIDIA DeepStream, detecting custom patterns with configurable strategies, managing alerts with intelligent throttling, and providing automatic recovery from failures. The system must achieve real-time performance (>30 FPS), support dynamic source management, and be extensible for new detection algorithms.

## Why

- **Production Deployment**: Enterprise-ready system with comprehensive error handling and monitoring
- **Multi-Source Scalability**: Process RTSP, WebRTC, file, and webcam sources concurrently without performance degradation
- **Backwards Compatibility**: Support existing NVIDIA infrastructure across DeepStream versions 5.x, 6.x, and 7.x
- **Extensible Architecture**: Plugin-based detection strategies allow easy addition of new algorithms
- **Operational Excellence**: Built-in monitoring, logging, and automatic recovery for 24/7 operation
- **Developer Experience**: Clear APIs and comprehensive examples for integration and customization

## What

A Python-based video analytics application with the following capabilities:

- **Multi-Source Pipeline Management**: Simultaneous processing of RTSP streams, WebRTC sources, video files, and webcam inputs
- **Dynamic Source Control**: Runtime addition/removal of video sources without pipeline restart
- **Custom Detection Engine**: Extensible strategy pattern supporting YOLO, custom models, and template matching
- **Intelligent Alert System**: Configurable throttling, multiple output handlers, and spam prevention
- **Automatic Recovery**: Fault tolerance with exponential backoff and graceful degradation
- **Performance Monitoring**: Real-time FPS tracking, memory usage, and processing latency metrics
- **Version Compatibility**: Automatic DeepStream version detection with appropriate API selection
- **Configuration Management**: YAML-based configuration with runtime updates and validation

### Success Criteria

- [ ] Process 4+ concurrent video sources at >30 FPS with GPU acceleration
- [ ] Support DeepStream 5.x, 6.x, and 7.x with automatic version detection and API adaptation
- [ ] Implement extensible detection strategies with custom model integration
- [ ] Achieve <100ms alert processing latency with configurable throttling
- [ ] Maintain <5% CPU overhead for pipeline management and monitoring
- [ ] Recover automatically from source failures within 10 seconds
- [ ] Support dynamic source addition/removal without affecting other sources
- [ ] Provide comprehensive logging and metrics for production monitoring
- [ ] Handle video format variations (H.264, H.265, different resolutions) seamlessly
- [ ] Maintain stable memory usage during 24+ hour continuous operation

## All Needed Context

### Documentation & References
```yaml
# MUST READ - Include these in your context window
- url: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Python_Sample_Apps.html
  why: Official DeepStream Python bindings documentation with complete API reference
  
- url: https://github.com/NVIDIA-AI-IOT/deepstream_python_apps
  why: Production-ready reference implementations for all major DeepStream features and patterns
  
- url: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Overview.html
  why: DeepStream architecture fundamentals - pipeline concepts, memory management, plugin ecosystem
  
- url: https://brettviren.github.io/pygst-tutorial-org/pygst-tutorial.html
  why: Comprehensive GStreamer Python tutorial covering pipeline construction, bus handling, and state management
  
- url: https://galliot.us/blog/deepstream-python-bindings-customization/
  why: Advanced DeepStream Python customization patterns for production deployments
  
- url: https://sahilchachra.medium.com/all-you-want-to-get-started-with-gstreamer-in-python-2276d9ed548e
  why: GStreamer Python integration patterns with PyGObject and real-world examples
  
- url: https://docs.astral.sh/uv/
  why: Modern Python package manager for dependency management and project configuration
  
- url: https://medium.com/@tzongwei2/nvidia-deepstream-gstreamer-pipeline-in-python-simplified-507907a91c6e
  section: Pipeline Construction and Error Handling
  critical: Production-ready pipeline setup patterns with proper error recovery
  
- file: TODO.md
  why: Complete requirements specification including performance, extensibility, and reliability requirements
  
- file: PRPs/vision-deepstream-python-implementation.md
  why: Existing implementation plan with detailed architecture and compatibility strategies
```

### Current Codebase tree
```bash
./
├── CLAUDE.md                                 # Project guidance and development commands
├── TODO.md                                   # Complete requirements specification
├── main.py                                   # Placeholder entry point (needs implementation)
└── PRPs/
    ├── templates/
    │   └── prp_base.md                      # PRP template for implementation guidance
    └── vision-deepstream-python-implementation.md  # Existing architectural plan
```

### Desired Codebase tree with files to be added and responsibility of file
```bash
./
├── pyproject.toml                           # uv project configuration with DeepStream dependencies
├── README.md                                # Installation guide and usage documentation  
├── CLAUDE.md                                # Keep existing project guidance
├── TODO.md                                  # Keep existing requirements (reference)
├── main.py                                  # CLI application entry point with Click interface
├── src/
│   ├── __init__.py                          # Package initialization with version info
│   ├── app.py                               # Main application class and lifecycle orchestration
│   ├── config.py                            # Configuration management with YAML validation
│   ├── pipeline/
│   │   ├── __init__.py                      # Pipeline package exports and version compatibility
│   │   ├── manager.py                       # GStreamer pipeline lifecycle and state management
│   │   ├── sources.py                       # Multi-source video input abstraction (RTSP/WebRTC/file/webcam)
│   │   ├── elements.py                      # DeepStream element creation with version compatibility
│   │   └── factory.py                       # Pipeline factory for different source configurations
│   ├── detection/
│   │   ├── __init__.py                      # Detection package exports and strategy registry
│   │   ├── engine.py                        # Core detection engine with metadata processing
│   │   ├── strategies.py                    # Built-in detection strategies (YOLO, template matching)
│   │   ├── custom.py                        # Custom detection strategy interface and loading
│   │   └── models.py                        # Detection result data models and validation
│   ├── alerts/
│   │   ├── __init__.py                      # Alert package exports and handler registry
│   │   ├── manager.py                       # Alert management with intelligent throttling
│   │   ├── handlers.py                      # Output handlers (console, file, webhook, email)
│   │   └── throttling.py                    # Advanced throttling algorithms and state management
│   ├── monitoring/
│   │   ├── __init__.py                      # Monitoring package exports and metrics registry
│   │   ├── metrics.py                       # Performance metrics collection and aggregation
│   │   ├── health.py                        # Health monitoring and automatic recovery
│   │   └── profiling.py                     # Performance profiling and bottleneck detection
│   └── utils/
│       ├── __init__.py                      # Utilities package exports
│       ├── deepstream.py                    # DeepStream version detection and compatibility layer
│       ├── errors.py                        # Custom exception hierarchy and error handling
│       ├── logging.py                       # Structured logging with performance metrics
│       └── async_utils.py                   # Async utilities and thread-safe operations
├── tests/
│   ├── __init__.py                          # Test package initialization
│   ├── conftest.py                          # Pytest fixtures and test environment setup
│   ├── test_config.py                       # Configuration management and validation tests
│   ├── test_pipeline.py                     # Pipeline lifecycle and state management tests
│   ├── test_detection.py                    # Detection engine and strategy tests
│   ├── test_alerts.py                       # Alert system and throttling tests
│   ├── test_monitoring.py                   # Monitoring and health check tests
│   └── integration/
│       ├── __init__.py                      # Integration test package
│       ├── test_multi_source.py             # Multi-source pipeline integration tests
│       ├── test_recovery.py                 # Error recovery and fault tolerance tests
│       └── test_performance.py              # Performance benchmarking and load tests
├── examples/
│   ├── basic_detection.py                   # Simple single-source detection with detailed comments
│   ├── multi_source_rtsp.py                 # Multiple RTSP stream processing example
│   ├── custom_detection_strategy.py         # Custom detection algorithm implementation
│   ├── alert_configuration.py               # Alert system configuration and testing
│   └── performance_monitoring.py            # Monitoring and metrics collection example
├── configs/
│   ├── default.yaml                         # Comprehensive default configuration template
│   ├── development.yaml                     # Development environment optimized settings
│   ├── production.yaml                      # Production deployment configuration
│   └── testing.yaml                         # Test environment configuration
└── scripts/
    ├── setup_environment.py                 # Environment validation and DeepStream setup
    ├── benchmark_performance.py             # Performance benchmarking utilities
    └── validate_installation.py             # Installation verification and diagnostics
```

### Known Gotchas of our codebase & Library Quirks
```python
# CRITICAL: DeepStream Version Compatibility
# DeepStream 5.x uses pyds module directly with different metadata APIs
# DeepStream 6.x+ uses gi.repository.NvDs with updated function signatures
# MUST implement runtime version detection and API abstraction layer
version = detect_deepstream_version()
if version >= (6, 0):
    from gi.repository import NvDs
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
else:
    import pyds
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(buffer)

# CRITICAL: GStreamer Python Integration
# MUST call gi.require_version('Gst', '1.0') before any GStreamer imports
# MUST initialize with GObject.threads_init() and Gst.init(None) for thread safety
# Pipeline state changes are ASYNCHRONOUS - must handle GST_STATE_CHANGE_ASYNC properly
# Bus message handling requires proper event loop integration with threading

# CRITICAL: DeepStream Memory Management
# GPU buffers must be properly mapped/unmapped to prevent memory leaks
# DeepStream metadata has reference counting - improper handling causes crashes
# String properties require pyds.get_string() and proper buffer management
# NvBufSurface objects need explicit cleanup and memory type awareness

# CRITICAL: Performance Considerations
# Probe functions execute in GStreamer threads - minimize Python processing
# Python GIL limits multi-threaded performance - use async patterns
# Batch processing is essential for GPU efficiency - configure nvstreammux properly
# Memory allocation in probe callbacks can cause pipeline stalls

# CRITICAL: Thread Safety Requirements
# GStreamer callbacks execute in different threads from main application
# Alert broadcasting must be thread-safe across multiple video sources
# Configuration updates during runtime require proper locking mechanisms
# Pipeline state management needs synchronization between threads

# CRITICAL: Error Recovery Patterns
# Pipeline failures require graceful state transitions: PLAYING→PAUSED→READY→NULL
# Source disconnections need automatic retry with exponential backoff
# Bus error messages must be handled to prevent silent failures
# Resource cleanup is critical during error conditions to prevent leaks

# CRITICAL: uv Package Management
# Use pyproject.toml exclusively (no setup.py or requirements.txt)
# DeepStream Python bindings require specific CUDA compatibility markers
# GPU dependencies must specify compatible CUDA and driver versions
# Development dependencies separate from runtime requirements
```

## Implementation Blueprint

### Data models and structure

Create the core data models ensuring type safety and consistency with video processing best practices.

```python
# Core data structures following DeepStream patterns
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List, Callable, Union
from pathlib import Path
import asyncio

@dataclass
class VideoDetection:
    """Represents a detected pattern in video frame with normalized coordinates"""
    pattern_name: str
    confidence: float
    bounding_box: tuple[float, float, float, float]  # x, y, width, height (normalized 0-1)
    timestamp: datetime
    frame_number: int
    source_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate detection data consistency"""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be 0-1, got {self.confidence}")
        if len(self.bounding_box) != 4:
            raise ValueError("Bounding box must have 4 coordinates")

@dataclass
class VideoSource:
    """Video source configuration with validation"""
    id: str
    name: str
    source_type: 'SourceType'
    uri: str
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    last_error: Optional[str] = None
    
    def __post_init__(self):
        """Validate source configuration"""
        if not self.id or not self.uri:
            raise ValueError("Source ID and URI are required")

class SourceType(Enum):
    """Supported video source types with URI validation patterns"""
    FILE = "file"
    WEBCAM = "webcam" 
    RTSP = "rtsp"
    WEBRTC = "webrtc"
    NETWORK = "network"
    TEST = "test"
    HTTP = "http"

class AlertLevel(Enum):
    """Alert severity levels for prioritization"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class AlertConfig:
    """Alert configuration with intelligent throttling"""
    enabled: bool = True
    throttle_seconds: int = 60
    min_confidence: float = 0.5
    level: AlertLevel = AlertLevel.MEDIUM
    handlers: List[str] = field(default_factory=lambda: ["console"])
    max_alerts_per_minute: int = 10
    burst_threshold: int = 3
    
@dataclass
class PipelineState:
    """Pipeline state tracking for monitoring"""
    source_id: str
    gst_state: str
    fps: float
    frame_count: int
    error_count: int
    last_activity: datetime
    memory_usage_mb: float
    gpu_utilization: float
```

### List of tasks to be completed to fulfill the PRP in the order they should be completed

```yaml
Task 1: Setup project infrastructure and dependencies
CREATE ./pyproject.toml:
  - CONFIGURE uv project with Python 3.8+ for DeepStream 5+ compatibility
  - ADD dependencies: pygobject>=3.40.0, gst-python, numpy>=1.19.0, opencv-python>=4.5.0
  - ADD DeepStream bindings with version compatibility markers (pyds, gi.repository)
  - ADD development dependencies: pytest, pytest-asyncio, pytest-cov, ruff, mypy, black
  - CONFIGURE CLI entry points and package metadata
  - SET CUDA compatibility constraints for GPU acceleration

CREATE ./README.md:
  - DOCUMENT installation requirements for different DeepStream versions
  - ADD step-by-step setup instructions for Ubuntu/Windows
  - INCLUDE troubleshooting guide for common DeepStream issues
  - PROVIDE usage examples and configuration guidance

Task 2: Implement core utilities and error handling
CREATE ./src/utils/errors.py:
  - DEFINE custom exception hierarchy: PipelineError, DetectionError, AlertError
  - ADD DeepStream-specific error types with version compatibility
  - IMPLEMENT error recovery strategies with exponential backoff
  - INCLUDE comprehensive error logging and context preservation

CREATE ./src/utils/logging.py:
  - CONFIGURE structured logging with JSON output for production
  - ADD performance metrics integration (FPS, latency, memory)
  - IMPLEMENT log level configuration and rotation
  - SUPPORT real-time log streaming for monitoring systems

CREATE ./src/utils/deepstream.py:
  - IMPLEMENT DeepStream version detection with feature capability mapping
  - ADD automatic API selection: pyds vs gi.repository based on version
  - HANDLE differences in metadata access patterns between versions
  - PROVIDE compatibility shims for common operations

CREATE ./src/utils/async_utils.py:
  - IMPLEMENT thread-safe async utilities for GStreamer integration
  - ADD queue management for inter-thread communication
  - PROVIDE async context managers for resource cleanup
  - HANDLE graceful shutdown coordination across components

Task 3: Implement configuration management system
CREATE ./src/config.py:
  - IMPLEMENT YAML configuration loading with comprehensive validation
  - ADD runtime configuration updates with thread-safe access
  - SUPPORT environment variable overrides and CLI parameter integration
  - INCLUDE configuration schema validation with detailed error messages
  - HANDLE configuration inheritance and environment-specific overrides

CREATE ./configs/default.yaml:
  - DEFINE comprehensive default configuration covering all components
  - ADD detailed documentation comments for each configuration section
  - INCLUDE DeepStream pipeline parameters and performance tuning options
  - SET reasonable defaults for development and testing environments

CREATE ./configs/production.yaml:
  - OPTIMIZE configuration for production deployment
  - CONFIGURE higher batch sizes and GPU memory allocation
  - SET aggressive error recovery and monitoring settings
  - INCLUDE production-specific alert and logging configurations

Task 4: Implement GStreamer pipeline management
CREATE ./src/pipeline/manager.py:
  - IMPLEMENT GStreamer pipeline lifecycle with proper state management
  - ADD bus message handling with comprehensive error recovery
  - SUPPORT dynamic pipeline modification and source management
  - HANDLE asynchronous state transitions with timeout handling
  - INTEGRATE performance monitoring and health checks

CREATE ./src/pipeline/sources.py:
  - IMPLEMENT video source abstraction for different input types
  - ADD source validation and availability checking with retry logic
  - SUPPORT dynamic source addition/removal during pipeline operation
  - HANDLE source-specific configurations and error recovery
  - INTEGRATE source health monitoring and automatic reconnection

CREATE ./src/pipeline/elements.py:
  - IMPLEMENT DeepStream element creation with version compatibility
  - ADD proper element linking with capability negotiation
  - SUPPORT GPU memory management and CUDA context handling
  - HANDLE different DeepStream plugin configurations across versions
  - INTEGRATE element performance monitoring and optimization

CREATE ./src/pipeline/factory.py:
  - IMPLEMENT pipeline factory patterns for different source configurations
  - ADD template-based pipeline construction for common use cases
  - SUPPORT custom pipeline modification and extension
  - HANDLE batch processing optimization for multi-source scenarios

Task 5: Implement detection engine with strategy pattern
CREATE ./src/detection/models.py:
  - IMPLEMENT detection result data models with validation
  - ADD confidence thresholding and bounding box normalization
  - SUPPORT metadata attachment and custom field extensions
  - HANDLE detection result serialization and persistence

CREATE ./src/detection/engine.py:
  - IMPLEMENT core detection engine with DeepStream metadata processing
  - ADD strategy pattern support for multiple detection algorithms
  - SUPPORT confidence filtering and result aggregation
  - HANDLE detection result routing to alert system
  - INTEGRATE performance profiling and bottleneck detection

CREATE ./src/detection/strategies.py:
  - IMPLEMENT built-in detection strategies: YOLO, template matching, feature-based
  - ADD DeepStream nvinfer integration with model loading
  - SUPPORT custom model configuration and parameter tuning
  - HANDLE different input/output tensor formats and preprocessing
  - INTEGRATE strategy performance benchmarking

CREATE ./src/detection/custom.py:
  - IMPLEMENT custom detection strategy interface and registration system
  - ADD dynamic strategy loading from external modules
  - SUPPORT strategy configuration validation and lifecycle management
  - HANDLE strategy error isolation and fallback mechanisms

Task 6: Implement intelligent alert management
CREATE ./src/alerts/throttling.py:
  - IMPLEMENT advanced throttling algorithms: token bucket, sliding window
  - ADD burst detection and adaptive throttling
  - SUPPORT per-source and per-pattern throttling configuration
  - HANDLE throttling state persistence and recovery

CREATE ./src/alerts/manager.py:
  - IMPLEMENT alert management with multi-threaded processing
  - ADD intelligent throttling with configurable policies
  - SUPPORT alert prioritization and routing rules
  - HANDLE alert persistence and delivery guarantees
  - INTEGRATE alert analytics and reporting

CREATE ./src/alerts/handlers.py:
  - IMPLEMENT various alert output handlers: console, file, webhook, email
  - ADD handler error handling and retry mechanisms
  - SUPPORT custom handler registration and configuration
  - HANDLE handler failover and load balancing
  - INTEGRATE handler performance monitoring

Task 7: Implement monitoring and health management
CREATE ./src/monitoring/metrics.py:
  - IMPLEMENT comprehensive performance metrics collection
  - ADD real-time FPS tracking, latency measurement, memory usage monitoring
  - SUPPORT metrics aggregation and historical data storage
  - HANDLE metrics export in multiple formats (JSON, Prometheus, InfluxDB)
  - INTEGRATE alerting on performance thresholds

CREATE ./src/monitoring/health.py:
  - IMPLEMENT health monitoring for all system components
  - ADD automatic recovery mechanisms for common failure scenarios
  - SUPPORT health check endpoints and status reporting
  - HANDLE graceful degradation under resource constraints
  - INTEGRATE predictive failure detection

CREATE ./src/monitoring/profiling.py:
  - IMPLEMENT performance profiling and bottleneck detection
  - ADD CPU and GPU utilization tracking
  - SUPPORT pipeline optimization recommendations
  - HANDLE performance regression detection

Task 8: Implement main application and CLI interface
CREATE ./src/app.py:
  - IMPLEMENT main application class with complete lifecycle management
  - ADD proper initialization sequence with dependency injection
  - SUPPORT graceful shutdown with resource cleanup
  - HANDLE multi-source coordination and load balancing
  - INTEGRATE comprehensive error handling and recovery

MODIFY ./main.py:
  - IMPLEMENT modern CLI interface using Click with comprehensive subcommands
  - ADD configuration file validation and testing commands
  - SUPPORT interactive mode for development and debugging
  - INCLUDE comprehensive help documentation and usage examples
  - INTEGRATE logging configuration and output formatting

Task 9: Create comprehensive test suite
CREATE ./tests/conftest.py:
  - IMPLEMENT pytest fixtures for testing environment setup
  - ADD mock DeepStream elements and pipeline components
  - SUPPORT test data generation and synthetic video sources
  - HANDLE test isolation and cleanup automation
  - INTEGRATE performance test utilities

CREATE ./tests/test_*.py files:
  - IMPLEMENT unit tests for all major components with >90% coverage
  - ADD integration tests for pipeline and detection workflows
  - SUPPORT parameterized tests for different DeepStream versions
  - HANDLE async testing for multi-threaded components
  - INTEGRATE property-based testing for edge cases

CREATE ./tests/integration/test_performance.py:
  - IMPLEMENT performance benchmarking with automated regression detection
  - ADD load testing for multi-source scenarios
  - SUPPORT memory leak detection and resource usage validation
  - HANDLE performance profiling and optimization validation

Task 10: Create examples and deployment scripts
CREATE ./examples/*.py:
  - IMPLEMENT comprehensive examples covering all major use cases
  - ADD detailed comments explaining DeepStream integration patterns
  - SUPPORT different complexity levels from basic to advanced
  - INCLUDE performance optimization examples and troubleshooting guides

CREATE ./scripts/setup_environment.py:
  - IMPLEMENT automated environment validation and setup
  - ADD DeepStream installation verification and guidance
  - SUPPORT dependency checking and compatibility validation
  - HANDLE different operating system requirements and troubleshooting
```

### Per task pseudocode as needed added to each task

```python
# Task 4: Pipeline Management Pseudocode
class PipelineManager:
    def __init__(self, config: Config):
        # CRITICAL: Initialize GStreamer with thread safety
        import gi
        gi.require_version('Gst', '1.0')
        from gi.repository import Gst, GObject
        GObject.threads_init()
        Gst.init(None)
        
        self.config = config
        self.pipelines: Dict[str, Gst.Pipeline] = {}
        self.pipeline_states: Dict[str, PipelineState] = {}
        self.bus_handlers: Dict[str, Any] = {}
        self.metrics_collector = MetricsCollector()
        
    async def create_multi_source_pipeline(self, sources: List[VideoSource]) -> str:
        # PATTERN: Multi-source pipeline with nvstreammux
        pipeline = Gst.Pipeline.new(f"analytics-pipeline-{uuid4()}")
        
        # Create stream muxer for batching
        streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
        if not streammux:
            raise PipelineError("Failed to create nvstreammux")
            
        # Configure for performance
        streammux.set_property('width', self.config.processing.width)
        streammux.set_property('height', self.config.processing.height)
        streammux.set_property('batch-size', len(sources))
        streammux.set_property('batched-push-timeout', self.config.processing.batch_timeout)
        
        # Add sources dynamically
        for i, source in enumerate(sources):
            source_bin = await self._create_source_bin(source, i)
            pipeline.add(source_bin)
            
            # Link to muxer with proper pad management
            src_pad = source_bin.get_static_pad("src")
            sink_pad = streammux.get_request_pad(f"sink_{i}")
            if src_pad.link(sink_pad) != Gst.PadLinkReturn.OK:
                raise PipelineError(f"Failed to link source {source.id}")
                
        # CRITICAL: Set up bus message handling
        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)
        
        # Add detection probe
        await self._setup_detection_probe(pipeline)
        
        return pipeline_id

    def _on_bus_message(self, bus, message):
        # PATTERN: Comprehensive bus message handling
        t = message.type
        source_name = message.src.get_name() if message.src else "unknown"
        
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error(f"Pipeline error from {source_name}: {err}")
            asyncio.create_task(self._handle_pipeline_error(source_name, err, debug))
            
        elif t == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            logger.warning(f"Pipeline warning from {source_name}: {warn}")
            
        elif t == Gst.MessageType.STATE_CHANGED:
            old_state, new_state, pending = message.parse_state_changed()
            self._update_pipeline_state(source_name, new_state)
            
        elif t == Gst.MessageType.EOS:
            logger.info(f"End of stream from {source_name}")
            asyncio.create_task(self._handle_eos(source_name))
            
        return True

# Task 5: Detection Engine Pseudocode
class DetectionEngine:
    def __init__(self, config: DetectionConfig):
        self.strategies: Dict[str, DetectionStrategy] = {}
        self.confidence_threshold = config.confidence_threshold
        self.deepstream_version = detect_deepstream_version()
        self.metrics = MetricsCollector()
        
    async def process_frame_metadata(self, batch_meta, source_id: str) -> List[VideoDetection]:
        # PATTERN: Version-compatible metadata processing
        detections = []
        processing_start = time.perf_counter()
        
        try:
            # CRITICAL: Navigate DeepStream metadata hierarchy safely
            frame_meta_list = batch_meta.frame_meta_list
            while frame_meta_list is not None:
                frame_meta = pyds.NvDsFrameMeta.cast(frame_meta_list.data)
                
                # Process object metadata with confidence filtering
                obj_meta_list = frame_meta.obj_meta_list
                while obj_meta_list is not None:
                    obj_meta = pyds.NvDsObjectMeta.cast(obj_meta_list.data)
                    
                    if obj_meta.confidence >= self.confidence_threshold:
                        detection = VideoDetection(
                            pattern_name=obj_meta.obj_label,
                            confidence=obj_meta.confidence,
                            bounding_box=self._normalize_bbox(obj_meta.rect_params, frame_meta),
                            timestamp=datetime.now(),
                            frame_number=frame_meta.frame_num,
                            source_id=source_id,
                            metadata=self._extract_additional_metadata(obj_meta)
                        )
                        detections.append(detection)
                        
                        # Apply custom detection strategies
                        for strategy in self.strategies.values():
                            if strategy.should_process(detection):
                                enhanced = await strategy.enhance_detection(detection, frame_meta)
                                if enhanced:
                                    detections.append(enhanced)
                                    
                    obj_meta_list = obj_meta_list.next
                frame_meta_list = frame_meta_list.next
                
        except Exception as e:
            logger.error(f"Error processing DeepStream metadata: {e}")
            self.metrics.increment_error_count("metadata_processing")
            
        finally:
            processing_time = time.perf_counter() - processing_start
            self.metrics.record_processing_time(processing_time)
            
        return detections

# Task 6: Alert Management Pseudocode
class AlertManager:
    def __init__(self, config: AlertConfig):
        self.config = config
        self.throttling = ThrottlingManager(config)
        self.handlers: List[AlertHandler] = []
        self.alert_queue = asyncio.Queue(maxsize=1000)
        self.metrics = MetricsCollector()
        self._processing_task = None
        
    async def start(self):
        """Start alert processing task"""
        self._processing_task = asyncio.create_task(self._process_alert_queue())
        
    async def process_detection(self, detection: VideoDetection):
        # PATTERN: Intelligent throttling with multiple strategies
        try:
            # Apply throttling rules
            should_alert, throttle_reason = await self.throttling.should_alert(detection)
            if not should_alert:
                self.metrics.increment_throttled_alerts(throttle_reason)
                return
                
            # Create alert with context
            alert = Alert(
                detection=detection,
                timestamp=datetime.now(),
                alert_id=str(uuid4()),
                level=self._determine_alert_level(detection),
                context=self._build_alert_context(detection)
            )
            
            # Queue for processing (non-blocking)
            try:
                self.alert_queue.put_nowait(alert)
                self.metrics.increment_queued_alerts()
            except asyncio.QueueFull:
                logger.warning("Alert queue full, dropping alert")
                self.metrics.increment_dropped_alerts()
                
        except Exception as e:
            logger.error(f"Error processing detection for alerts: {e}")
            
    async def _process_alert_queue(self):
        """Background task for alert processing"""
        while True:
            try:
                alert = await self.alert_queue.get()
                
                # Broadcast to all enabled handlers
                handler_tasks = []
                for handler in self.handlers:
                    if handler.should_handle(alert):
                        handler_tasks.append(handler.handle_alert(alert))
                        
                # Process handlers concurrently with error isolation
                results = await asyncio.gather(*handler_tasks, return_exceptions=True)
                
                # Log handler failures without stopping processing
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Alert handler {self.handlers[i].name} failed: {result}")
                        self.metrics.increment_handler_errors(self.handlers[i].name)
                        
                self.alert_queue.task_done()
                self.metrics.increment_processed_alerts()
                
            except Exception as e:
                logger.error(f"Error in alert processing loop: {e}")
                await asyncio.sleep(1)  # Prevent tight error loop
```

### Integration Points
```yaml
DEPENDENCIES:
  - pyproject.toml: "Add pygobject>=3.40.0 for GStreamer Python bindings"
  - pyproject.toml: "Add DeepStream bindings with CUDA compatibility: pyds>=1.1.0"
  - pyproject.toml: "Add numpy>=1.19.0, opencv-python>=4.5.0 for image processing"
  - pyproject.toml: "Add asyncio-glib for GStreamer event loop integration"
  - pyproject.toml: "Add click>=8.0.0 for modern CLI interface"
  - pyproject.toml: "Add pyyaml>=6.0, pydantic>=2.0 for configuration management"
  
DEEPSTREAM_COMPATIBILITY:
  - version_detection: "Runtime capability detection for API selection"
  - metadata_access: "Version-specific metadata extraction patterns"
  - element_creation: "Plugin availability and configuration differences"
  - memory_management: "GPU buffer handling variations across versions"
  
GSTREAMER_INTEGRATION:
  - pipeline_construction: "Element factory patterns with error checking"
  - bus_handling: "Comprehensive message handling with async integration"
  - probe_callbacks: "Performance-optimized buffer probes with minimal overhead"
  - state_management: "Proper state transitions with timeout handling"
  
PERFORMANCE_OPTIMIZATION:
  - batching: "nvstreammux configuration for optimal GPU utilization"
  - memory_pools: "Pre-allocated buffer pools for high-throughput scenarios"
  - thread_management: "Minimal thread overhead with async coordination"
  - gpu_acceleration: "CUDA stream optimization and memory type selection"
```

## Validation Loop

### Level 1: Syntax & Style
```bash
# Run these FIRST - fix any errors before proceeding
uv run ruff check src/ tests/ examples/ --fix    # Auto-fix what's possible
uv run mypy src/                                  # Type checking with strict mode
uv run black src/ tests/ examples/ --check       # Code formatting validation

# Expected: No errors. If errors, READ the error message and fix the root cause.
```

### Level 2: Unit Tests each new feature/file/function use existing test patterns
```python
# CREATE comprehensive test suite with realistic scenarios:
@pytest.mark.asyncio
async def test_multi_source_pipeline_creation():
    """Test concurrent multi-source pipeline with different input types"""
    config = Config.load_from_file("configs/testing.yaml")
    manager = PipelineManager(config)
    
    sources = [
        VideoSource(id="rtsp-1", name="RTSP Camera", source_type=SourceType.RTSP, 
                   uri="rtsp://demo.url/stream1"),
        VideoSource(id="file-1", name="Test Video", source_type=SourceType.FILE,
                   uri="file:///test/data/sample.mp4"),
        VideoSource(id="test-1", name="Synthetic", source_type=SourceType.TEST,
                   uri="videotestsrc pattern=smpte")
    ]
    
    pipeline_id = await manager.create_multi_source_pipeline(sources)
    assert pipeline_id is not None
    assert len(manager.pipelines) == 1
    
    # Verify pipeline can start
    await manager.start_pipeline(pipeline_id)
    await asyncio.sleep(2)  # Allow initialization
    
    # Check health metrics
    state = manager.get_pipeline_state(pipeline_id)
    assert state.gst_state == "PLAYING"
    assert state.fps > 0

@pytest.mark.asyncio
async def test_detection_engine_with_mock_metadata():
    """Test detection processing with realistic DeepStream metadata"""
    engine = DetectionEngine(DetectionConfig(confidence_threshold=0.7))
    
    # Create mock metadata that matches real DeepStream structure
    mock_metadata = create_mock_deepstream_batch_meta([
        {"class_id": 0, "confidence": 0.85, "bbox": (100, 100, 200, 150)},
        {"class_id": 1, "confidence": 0.92, "bbox": (300, 200, 100, 120)},
        {"class_id": 0, "confidence": 0.65, "bbox": (50, 300, 80, 100)}  # Below threshold
    ])
    
    detections = await engine.process_frame_metadata(mock_metadata, "test-source")
    
    # Verify filtering and processing
    assert len(detections) == 2  # One filtered out by confidence
    assert all(d.confidence >= 0.7 for d in detections)
    assert all(0 <= coord <= 1 for d in detections for coord in d.bounding_box)

@pytest.mark.asyncio
async def test_alert_throttling_prevents_spam():
    """Test intelligent alert throttling with burst detection"""
    config = AlertConfig(throttle_seconds=5, burst_threshold=3, max_alerts_per_minute=10)
    manager = AlertManager(config)
    mock_handler = MockAlertHandler()
    manager.add_handler(mock_handler)
    
    await manager.start()
    
    detection = VideoDetection(
        pattern_name="person",
        confidence=0.9,
        bounding_box=(0.1, 0.1, 0.2, 0.2),
        timestamp=datetime.now(),
        frame_number=1,
        source_id="test-camera"
    )
    
    # Test normal throttling
    await manager.process_detection(detection)
    await asyncio.sleep(0.1)  # Allow processing
    assert len(mock_handler.alerts) == 1
    
    # Immediate duplicate should be throttled
    await manager.process_detection(detection)
    await asyncio.sleep(0.1)
    assert len(mock_handler.alerts) == 1
    
    # Test burst detection
    for _ in range(5):
        detection.frame_number += 1
        detection.timestamp = datetime.now()
        await manager.process_detection(detection)
    
    await asyncio.sleep(0.5)  # Allow processing
    # Should not exceed burst threshold
    assert len(mock_handler.alerts) <= config.burst_threshold + 1

def test_deepstream_version_compatibility():
    """Test version detection and API compatibility layer"""
    version_info = detect_deepstream_version()
    assert version_info.major >= 5  # Minimum supported version
    
    # Test API abstraction works for detected version
    api = get_deepstream_api(version_info)
    assert hasattr(api, 'get_batch_meta')
    assert hasattr(api, 'cast_frame_meta')
    assert hasattr(api, 'extract_object_meta')
    
    # Test metadata processing compatibility
    mock_buffer = create_mock_gst_buffer()
    batch_meta = api.get_batch_meta(mock_buffer)
    assert batch_meta is not None

@pytest.mark.asyncio
async def test_pipeline_error_recovery():
    """Test automatic recovery from pipeline failures"""
    config = Config.load_from_file("configs/testing.yaml")
    config.recovery.max_retries = 2
    config.recovery.retry_delay = 1.0
    
    manager = PipelineManager(config)
    
    # Create pipeline with source that will fail
    failing_source = VideoSource(
        id="failing-rtsp", 
        name="Failing RTSP", 
        source_type=SourceType.RTSP,
        uri="rtsp://nonexistent.server/stream"
    )
    
    pipeline_id = await manager.create_multi_source_pipeline([failing_source])
    
    # Monitor recovery attempts
    recovery_attempts = 0
    def on_recovery(source_id, attempt):
        nonlocal recovery_attempts
        recovery_attempts = attempt
    
    manager.on_recovery_attempt = on_recovery
    
    # Start pipeline (will fail and trigger recovery)
    with pytest.raises(PipelineError):
        await manager.start_pipeline(pipeline_id)
    
    # Wait for recovery attempts
    await asyncio.sleep(5)
    assert recovery_attempts == config.recovery.max_retries
```

```bash
# Run and iterate until passing:
uv run pytest tests/ -v --cov=src --cov-report=html
# Target: >90% test coverage with all tests passing
# If failing: Read error, understand root cause, fix code, re-run
```

### Level 3: Integration Test
```bash
# Test with synthetic video source (always available)
uv run python main.py start --sources configs/test-sources.yaml --config configs/testing.yaml --verbose

# Expected: Multi-source pipeline starts, processes test frames, shows real-time metrics
# Should show: FPS tracking, detection processing, alert throttling, memory usage

# Test configuration validation
uv run python main.py validate-config configs/production.yaml

# Expected: Configuration validation passes with component availability report
# Should check: DeepStream installation, GPU availability, model files, network connectivity

# Test performance benchmarking
uv run python scripts/benchmark_performance.py --sources 4 --duration 60 --profile

# Expected: Performance metrics within acceptable ranges (>30 FPS, <200ms latency)
# Should generate: Performance report, bottleneck analysis, optimization recommendations

# Test error recovery with simulated failures
uv run python main.py start --sources configs/unreliable-sources.yaml --enable-chaos-testing

# Expected: Graceful handling of source failures with automatic recovery
# Should demonstrate: Retry mechanisms, degraded mode operation, health monitoring

# Test RTSP source with real stream (if available)
uv run python main.py start --source rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4 --max-duration 30

# Expected: RTSP stream processing with network error handling
# Should handle: Network interruptions, stream format changes, reconnection logic
```

## Final validation Checklist

- [ ] All tests pass with >90% coverage: `uv run pytest tests/ -v --cov=src --cov-report=html`
- [ ] No linting errors: `uv run ruff check src/ tests/ examples/`
- [ ] No type errors: `uv run mypy src/`
- [ ] Code formatting consistent: `uv run black --check src/ tests/ examples/`
- [ ] Multi-source pipeline operates at >30 FPS with 4 concurrent sources
- [ ] DeepStream version detection works across 5.x, 6.x, 7.x installations
- [ ] Alert throttling prevents spam while maintaining responsiveness
- [ ] Error recovery mechanisms handle pipeline failures gracefully
- [ ] Memory usage remains stable during 1+ hour continuous operation
- [ ] Performance metrics collection provides actionable insights
- [ ] Configuration validation catches common setup errors
- [ ] Examples demonstrate all major features with clear documentation
- [ ] Installation script works on clean Ubuntu 20.04+ and Windows 11 systems

---

## Anti-Patterns to Avoid

- ❌ Don't skip GStreamer initialization sequence - causes segmentation faults
- ❌ Don't ignore DeepStream version differences - APIs are incompatible between major versions  
- ❌ Don't use blocking operations in GStreamer probe callbacks - causes pipeline stalls
- ❌ Don't assume GPU is always available - implement graceful CPU fallback
- ❌ Don't forget proper metadata lifecycle management - leads to memory leaks and crashes
- ❌ Don't hardcode DeepStream plugin configurations - parameters vary between versions
- ❌ Don't skip comprehensive bus message handling - critical errors may go unnoticed
- ❌ Don't use synchronous I/O in async contexts - blocks event loops and degrades performance
- ❌ Don't ignore threading requirements - GStreamer callbacks execute in different threads
- ❌ Don't implement custom retry logic without exponential backoff - causes resource exhaustion
- ❌ Don't skip input validation in public APIs - leads to unclear error messages
- ❌ Don't log sensitive information or excessive debug data in production

## Confidence Score: 9/10

This PRP provides exceptional implementation guidance with:
- ✅ Comprehensive DeepStream version compatibility strategy (5.x through 7.x)
- ✅ Production-ready GStreamer integration patterns with robust error handling
- ✅ Real-world performance optimization techniques from NVIDIA community examples
- ✅ Modern Python development practices using uv package manager and async/await
- ✅ Extensive test strategy covering unit, integration, and performance testing
- ✅ Proven multi-source video processing patterns with GPU acceleration
- ✅ Enterprise-grade error recovery and monitoring capabilities
- ✅ Detailed implementation tasks with specific pseudocode and gotchas
- ✅ Clear validation gates with executable commands and expected outcomes
- ✅ Comprehensive documentation references from official NVIDIA sources

The score is 9/10 due to the high implementation success probability enabled by:
1. Detailed research from official NVIDIA DeepStream examples
2. Proven architecture patterns from production deployments  
3. Comprehensive error handling and version compatibility strategies
4. Clear implementation order with validation at each step
5. Extensive context from official documentation and community best practices

The implementation should achieve working code through iterative refinement with the provided validation loops and comprehensive context.