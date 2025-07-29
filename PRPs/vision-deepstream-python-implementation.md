name: "Vision Deepstream Python Implementation"
description: |

## Purpose
Implement a robust, standalone Python application for real-time video pattern detection using NVIDIA DeepStream and GStreamer, with support for DeepStream versions 5+ and backwards compatibility. The system should handle custom detection strategies, manage alerts with throttling, and provide a flexible API for pattern detection across multiple video sources.

## Core Principles
1. **Context is King**: Leverage existing video processing patterns and established Python practices
2. **Validation Loops**: Provide executable tests using uv and pytest
3. **Information Dense**: Use proven GStreamer and DeepStream integration patterns
4. **Progressive Success**: Start with basic video input, add detection, then multi-source support
5. **Backwards Compatibility**: Support DeepStream 5.x through 7.x with feature detection

---

## Goal

Build a robust, reliable standalone Python system that detects patterns in video streams using GStreamer and NVIDIA DeepStream. The system should handle custom detection strategies, manage alerts with throttling, and provide a flexible API for pattern detection. It must be robust to errors with automatic recovery, support multiple video sources simultaneously, and be modular for easy extension of new detection strategies.

## Why

- **Standalone Application**: Independent Python-based video analytics system that can run separately or be integrated into other applications
- **Multi-source Support**: Real-time processing of RTSP, WebRTC, file, and webcam sources simultaneously with different video formats and resolutions
- **Backwards Compatibility**: Support for DeepStream versions 5+ ensures compatibility with existing NVIDIA infrastructure
- **Extensible Architecture**: Modular design allows easy addition of new detection strategies and patterns with dynamic registration/unregistration capabilities
- **Production Ready**: Designed for efficiency, performance, low latency, high throughput with automatic error recovery

## What

A Python-based video analytics application using NVIDIA DeepStream and GStreamer with the following capabilities:

- **Multi-source Video Processing**: Supports RTSP, WebRTC, file, webcam sources with dynamic source management
- **DeepStream Integration**: Uses NVIDIA DeepStream SDK with backwards compatibility for versions 5.x through 7.x
- **Pattern Detection Engine**: Extensible detection strategies with custom pattern support
- **Alert Management**: Throttled alert system to prevent spam with configurable parameters
- **Real-time Processing**: Handles high-resolution streams with GPU acceleration when available
- **Dynamic Configuration**: Runtime configuration of detection parameters and pipeline settings
- **Monitoring & Logging**: Performance tracking and comprehensive error logging
- **User Interface**: Configuration management interface with real-time visualization

### Success Criteria

- [ ] Successfully processes video from multiple sources (RTSP, WebRTC, file, webcam) simultaneously
- [ ] Integrates with DeepStream versions 5.x, 6.x, and 7.x with automatic version detection
- [ ] Implements extensible pattern detection with custom strategy support
- [ ] Manages alerts with configurable throttling to prevent spam
- [ ] Supports dynamic source registration/unregistration without service restart
- [ ] Achieves real-time processing (>30 FPS) with GPU acceleration
- [ ] Handles video stream failures with automatic recovery mechanisms
- [ ] Provides user-friendly configuration interface
- [ ] Includes comprehensive error handling and logging
- [ ] Supports different video formats and resolutions dynamically

## All Needed Context

### Documentation & References

```yaml
# MUST READ - Include these in your context window
- url: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Python_Sample_Apps.html
  why: Official DeepStream Python bindings documentation and sample applications
  
- url: https://github.com/NVIDIA-AI-IOT/deepstream_python_apps
  why: Complete reference implementation with examples for all major DeepStream features
  
- url: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Overview.html
  why: DeepStream SDK architecture and pipeline concepts essential for understanding
  
- url: https://sahilchachra.medium.com/all-you-want-to-get-started-with-gstreamer-in-python-2276d9ed548e
  why: Comprehensive GStreamer Python tutorial with PyGObject integration patterns
  
- url: https://brettviren.github.io/pygst-tutorial-org/pygst-tutorial.html
  why: Python GStreamer tutorial covering pipeline construction and event handling
  
- url: https://docs.astral.sh/uv/
  why: uv package manager documentation for Python project management and dependency handling
  
- url: https://galliot.us/blog/deepstream-python-bindings-customization/
  why: DeepStream Python customization patterns and advanced integration techniques
  
- file: TODO.md
  why: Complete requirements specification including extensibility, performance, and architecture needs
  
- doc: https://medium.com/@tzongwei2/nvidia-deepstream-gstreamer-pipeline-in-python-simplified-507907a91c6e
  section: Pipeline Construction and Management
  critical: Shows proper DeepStream pipeline setup and error handling patterns
```

### Current Codebase tree

```bash
./
├── TODO.md              # Requirements specification  
└── main.py              # Placeholder file with "# TODO: everything"
```

### Desired Codebase tree with files to be added and responsibility of file

```bash
./
├── pyproject.toml        # uv project configuration with dependencies and metadata
├── README.md             # Project documentation and setup instructions
├── TODO.md               # Keep existing requirements (reference)
├── main.py               # CLI application entry point with argument parsing
├── src/
│   ├── __init__.py       # Package initialization
│   ├── app.py            # Main application class and lifecycle management
│   ├── config.py         # Configuration management and validation
│   ├── pipeline/
│   │   ├── __init__.py   # Pipeline package exports
│   │   ├── manager.py    # GStreamer pipeline management and lifecycle
│   │   ├── sources.py    # Video source management (RTSP, WebRTC, file, webcam)
│   │   └── elements.py   # DeepStream element creation and configuration
│   ├── detection/
│   │   ├── __init__.py   # Detection package exports
│   │   ├── engine.py     # Pattern detection engine with strategy pattern
│   │   ├── strategies.py # Built-in detection strategies (template, feature-based)
│   │   └── custom.py     # Custom detection strategy interface and registry
│   ├── alerts/
│   │   ├── __init__.py   # Alerts package exports
│   │   ├── manager.py    # Alert management with throttling and broadcasting
│   │   └── handlers.py   # Alert output handlers (console, file, network)
│   ├── monitoring/
│   │   ├── __init__.py   # Monitoring package exports
│   │   ├── metrics.py    # Performance metrics collection and reporting
│   │   └── health.py     # Health monitoring and automatic recovery
│   └── utils/
│       ├── __init__.py   # Utilities package exports
│       ├── deepstream.py # DeepStream version detection and compatibility layer
│       ├── errors.py     # Custom exception classes and error handling
│       └── logging.py    # Structured logging configuration
├── tests/
│   ├── __init__.py       # Test package initialization
│   ├── conftest.py       # Pytest configuration and fixtures
│   ├── test_config.py    # Configuration management tests
│   ├── test_pipeline.py  # Pipeline management tests
│   ├── test_detection.py # Detection engine tests
│   ├── test_alerts.py    # Alert system tests
│   └── integration/
│       ├── __init__.py   # Integration test package
│       ├── test_sources.py # Video source integration tests
│       └── test_e2e.py   # End-to-end system tests
├── examples/
│   ├── basic_detection.py     # Simple single-source detection example
│   ├── multi_source.py        # Multiple source processing example
│   ├── custom_strategy.py     # Custom detection strategy example
│   └── rtsp_streaming.py      # RTSP stream processing example
├── configs/
│   ├── default.yaml      # Default configuration template
│   ├── development.yaml  # Development environment configuration
│   └── production.yaml   # Production environment configuration
└── scripts/
    ├── setup_environment.py   # Environment setup and dependency checking
    └── benchmark.py           # Performance benchmarking utilities
```

### Known Gotchas of our codebase & Library Quirks

```python
# CRITICAL: DeepStream version compatibility 
# DeepStream 5.x uses different Python binding APIs than 6.x/7.x
# Must detect version and use appropriate import paths and function signatures
# Example: pyds vs gi.repository.NvDs differences between versions

# DEEPSTREAM VERSION COMPATIBILITY:
# DeepStream 5.x: import pyds directly, different metadata access patterns
# DeepStream 6.x+: Uses gi.repository bindings with updated APIs
# CRITICAL: Must implement version detection in utils/deepstream.py

# GSTREAMER PYTHON GOTCHAS:
# CRITICAL: GStreamer requires gi.require_version('Gst', '1.0') before import
# CRITICAL: Must call Gst.init() before creating any pipeline elements
# CRITICAL: Pipeline state changes are asynchronous - must handle state transitions properly
# CRITICAL: Bus message handling requires proper event loop integration

# DEEPSTREAM SPECIFIC GOTCHAS:
# CRITICAL: DeepStream metadata is attached to GstBuffer as NvDsMeta
# CRITICAL: nvinfer plugin requires specific input tensor formats and configurations
# CRITICAL: Memory allocation for DeepStream metadata must use proper APIs
# CRITICAL: GPU memory and CUDA context management is critical for performance

# UV PROJECT MANAGEMENT:
# CRITICAL: Use pyproject.toml for all configuration (not setup.py or requirements.txt)
# CRITICAL: uv requires specific dependency specification format for GPU packages
# CRITICAL: CUDA dependencies must be specified correctly for DeepStream compatibility

# THREAD SAFETY:
# CRITICAL: GStreamer callbacks execute in different threads - use proper synchronization
# CRITICAL: Alert broadcasting must be thread-safe across multiple video sources
# CRITICAL: Configuration updates during runtime require proper locking mechanisms

# PERFORMANCE CONSIDERATIONS:
# CRITICAL: Probe functions on high-frequency elements can cause performance issues
# CRITICAL: Python GIL can limit multi-threaded performance - use proper async patterns
# CRITICAL: Memory management in callbacks is critical to prevent leaks
```

## Implementation Blueprint

### Data models and structure

Create the core data models ensuring type safety and consistency with video processing best practices.

```python
# Core data structures following established patterns
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path

@dataclass
class VideoDetection:
    """Represents a detected pattern in video frame"""
    pattern_name: str
    confidence: float
    bounding_box: tuple[float, float, float, float]  # x, y, width, height (normalized 0-1)
    timestamp: datetime
    frame_number: int
    source_id: str
    metadata: Dict[str, Any] = None

@dataclass
class VideoSource:
    """Video source configuration"""
    id: str
    name: str
    source_type: 'SourceType'
    uri: str
    enabled: bool = True
    parameters: Dict[str, Any] = None

class SourceType(Enum):
    """Supported video source types"""
    FILE = "file"
    WEBCAM = "webcam" 
    RTSP = "rtsp"
    WEBRTC = "webrtc"
    NETWORK = "network"
    TEST = "test"

class AlertLevel(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class AlertConfig:
    """Alert configuration settings"""
    enabled: bool = True
    throttle_seconds: int = 60
    min_confidence: float = 0.5
    level: AlertLevel = AlertLevel.MEDIUM
    handlers: List[str] = None  # console, file, webhook, etc.
```

### List of tasks to be completed to fulfill the PRP in the order they should be completed

```yaml
Task 1: Setup project structure and Python environment
CREATE ./pyproject.toml:
  - CONFIGURE uv project with Python 3.8+ compatibility for DeepStream 5+ support
  - ADD dependencies: pygobject, gst-python, numpy, opencv-python, pyyaml, click
  - ADD DeepStream Python bindings with version compatibility markers
  - CONFIGURE development dependencies: pytest, pytest-asyncio, black, mypy
  - SET project metadata and entry points for CLI application

CREATE ./README.md:
  - DOCUMENT installation requirements and DeepStream version compatibility
  - ADD setup instructions for different operating systems
  - INCLUDE example usage and configuration

Task 2: Implement core error handling and utilities
CREATE ./src/utils/errors.py:
  - DEFINE custom exception hierarchy for different error types
  - ADD DeepStream-specific error handling with version compatibility
  - IMPLEMENT error recovery strategies and logging integration

CREATE ./src/utils/logging.py:
  - CONFIGURE structured logging with performance metrics
  - ADD log level configuration and output formatting
  - IMPLEMENT log rotation and file management

CREATE ./src/utils/deepstream.py:
  - IMPLEMENT DeepStream version detection and compatibility layer
  - ADD automatic API selection based on detected version
  - HANDLE differences between DeepStream 5.x and 6.x+ Python bindings

Task 3: Implement configuration management
CREATE ./src/config.py:
  - IMPLEMENT configuration loading from YAML files with validation
  - ADD runtime configuration updates with thread safety
  - SUPPORT environment variable overrides and CLI parameter integration
  - INCLUDE configuration schema validation and error reporting

CREATE ./configs/default.yaml:
  - DEFINE comprehensive default configuration covering all components
  - ADD documentation comments for each configuration option
  - INCLUDE DeepStream pipeline parameters and detection thresholds

Task 4: Implement GStreamer pipeline management
CREATE ./src/pipeline/manager.py:
  - IMPLEMENT GStreamer pipeline lifecycle management
  - ADD proper initialization sequence with error handling
  - SUPPORT dynamic pipeline modification and source management
  - HANDLE bus message processing and state change monitoring

CREATE ./src/pipeline/sources.py:
  - IMPLEMENT video source abstraction for different input types
  - ADD source validation and availability checking
  - SUPPORT dynamic source addition/removal during runtime
  - HANDLE source-specific pipeline elements and configuration

CREATE ./src/pipeline/elements.py:
  - IMPLEMENT DeepStream element creation with version compatibility
  - ADD proper element linking and configuration
  - SUPPORT GPU memory management and CUDA context handling
  - HANDLE different DeepStream plugin configurations

Task 5: Implement pattern detection engine
CREATE ./src/detection/engine.py:
  - IMPLEMENT detection engine with strategy pattern support
  - ADD DeepStream metadata extraction and processing
  - SUPPORT multiple detection strategies running simultaneously
  - HANDLE confidence thresholding and result filtering

CREATE ./src/detection/strategies.py:
  - IMPLEMENT built-in detection strategies (template matching, feature-based)
  - ADD YOLO and other model integration through DeepStream nvinfer
  - SUPPORT custom model loading and configuration
  - HANDLE different input/output tensor formats

CREATE ./src/detection/custom.py:
  - IMPLEMENT custom detection strategy interface and registry
  - ADD dynamic strategy loading and registration
  - SUPPORT strategy configuration and parameter validation
  - HANDLE strategy lifecycle management

Task 6: Implement alert management system
CREATE ./src/alerts/manager.py:
  - IMPLEMENT alert management with configurable throttling
  - ADD multi-threaded alert broadcasting with proper synchronization
  - SUPPORT different alert levels and filtering rules
  - HANDLE alert persistence and recovery

CREATE ./src/alerts/handlers.py:
  - IMPLEMENT various alert output handlers (console, file, webhook)
  - ADD handler configuration and error handling
  - SUPPORT custom handler registration and management
  - HANDLE handler failover and redundancy

Task 7: Implement monitoring and health management
CREATE ./src/monitoring/metrics.py:
  - IMPLEMENT performance metrics collection (FPS, latency, memory usage)
  - ADD metrics aggregation and reporting capabilities
  - SUPPORT metrics export in various formats (JSON, Prometheus)
  - HANDLE metrics persistence and historical data

CREATE ./src/monitoring/health.py:
  - IMPLEMENT health monitoring for all system components
  - ADD automatic recovery mechanisms for common failure scenarios
  - SUPPORT health check endpoints and status reporting
  - HANDLE graceful degradation under resource constraints

Task 8: Implement main application and CLI
CREATE ./src/app.py:
  - IMPLEMENT main application class with lifecycle management
  - ADD proper initialization and cleanup sequences
  - SUPPORT graceful shutdown and signal handling
  - HANDLE multi-source coordination and resource management

MODIFY ./main.py:
  - IMPLEMENT CLI interface using click for modern Python CLI patterns
  - ADD comprehensive command-line options and subcommands
  - SUPPORT configuration file specification and validation
  - INCLUDE help documentation and usage examples

Task 9: Create comprehensive test suite
CREATE ./tests/conftest.py:
  - IMPLEMENT pytest fixtures for testing environment setup
  - ADD mock DeepStream elements and pipeline components
  - SUPPORT test data generation and cleanup
  - HANDLE test isolation and reproducibility

CREATE ./tests/test_*.py files:
  - IMPLEMENT unit tests for all major components
  - ADD integration tests for pipeline and detection workflows
  - SUPPORT parameterized tests for different DeepStream versions
  - HANDLE async testing for multi-threaded components

CREATE ./tests/integration/test_e2e.py:
  - IMPLEMENT end-to-end system tests with real video sources
  - ADD performance benchmarking and load testing
  - SUPPORT automated testing with different configurations
  - HANDLE test data management and cleanup

Task 10: Create examples and documentation
CREATE ./examples/*.py:
  - IMPLEMENT comprehensive examples covering all major use cases
  - ADD detailed comments explaining DeepStream integration patterns
  - SUPPORT different complexity levels from basic to advanced
  - INCLUDE performance optimization examples and best practices

CREATE ./scripts/setup_environment.py:
  - IMPLEMENT automated environment setup and validation
  - ADD DeepStream installation checking and guidance
  - SUPPORT dependency installation and configuration
  - HANDLE different operating system requirements
```

### Per task pseudocode as needed added to each task

```python
# Task 4: Pipeline Management Pseudocode
class PipelineManager:
    def __init__(self, config: Config):
        # CRITICAL: Initialize GStreamer before any operations
        import gi
        gi.require_version('Gst', '1.0')
        from gi.repository import Gst
        Gst.init(None)
        
        self.config = config
        self.pipelines: Dict[str, Gst.Pipeline] = {}
        self.sources: Dict[str, VideoSource] = {}
        
    async def create_pipeline(self, source: VideoSource) -> str:
        # PATTERN: DeepStream pipeline construction with version compatibility
        deepstream_version = detect_deepstream_version()
        
        if deepstream_version >= (6, 0):
            # Use newer gi.repository bindings
            from gi.repository import NvDs
            pipeline_string = self._build_pipeline_string_v6(source)
        else:
            # Use legacy pyds bindings  
            import pyds
            pipeline_string = self._build_pipeline_string_v5(source)
            
        pipeline = Gst.parse_launch(pipeline_string)
        
        # CRITICAL: Set up bus message handling
        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)
        
        # CRITICAL: Add probe for detection processing
        sink_pad = pipeline.get_by_name("sink").get_static_pad("sink")
        sink_pad.add_probe(Gst.PadProbeType.BUFFER, self._detection_probe_callback)
        
        return pipeline_id

    def _detection_probe_callback(self, pad, info, user_data):
        # CRITICAL: Extract DeepStream metadata with version compatibility
        buffer = info.get_buffer()
        
        if self.deepstream_version >= (6, 0):
            # Modern metadata extraction
            batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        else:
            # Legacy metadata extraction
            batch_meta = pyds.gst_buffer_get_nvds_batch_meta(buffer)
            
        # Process detections and trigger alerts
        detections = self._extract_detections(batch_meta)
        asyncio.create_task(self._process_detections(detections))
        
        return Gst.PadProbeReturn.OK

# Task 5: Detection Engine Pseudocode  
class DetectionEngine:
    def __init__(self, config: DetectionConfig):
        self.strategies: Dict[str, DetectionStrategy] = {}
        self.confidence_threshold = config.confidence_threshold
        self.deepstream_version = detect_deepstream_version()
        
    async def process_frame_metadata(self, batch_meta) -> List[VideoDetection]:
        # PATTERN: DeepStream metadata processing with error handling
        detections = []
        
        try:
            # CRITICAL: Navigate DeepStream metadata hierarchy
            frame_meta_list = batch_meta.frame_meta_list
            while frame_meta_list is not None:
                frame_meta = pyds.NvDsFrameMeta.cast(frame_meta_list.data)
                
                # Extract object metadata
                obj_meta_list = frame_meta.obj_meta_list  
                while obj_meta_list is not None:
                    obj_meta = pyds.NvDsObjectMeta.cast(obj_meta_list.data)
                    
                    if obj_meta.confidence >= self.confidence_threshold:
                        detection = VideoDetection(
                            pattern_name=obj_meta.obj_label,
                            confidence=obj_meta.confidence,
                            bounding_box=(
                                obj_meta.rect_params.left / frame_meta.source_frame_width,
                                obj_meta.rect_params.top / frame_meta.source_frame_height,
                                obj_meta.rect_params.width / frame_meta.source_frame_width,
                                obj_meta.rect_params.height / frame_meta.source_frame_height
                            ),
                            timestamp=datetime.now(),
                            frame_number=frame_meta.frame_num,
                            source_id=str(frame_meta.source_id)
                        )
                        detections.append(detection)
                        
                    obj_meta_list = obj_meta_list.next
                frame_meta_list = frame_meta_list.next
                
        except Exception as e:
            logger.error(f"Error processing DeepStream metadata: {e}")
            
        return detections

# Task 6: Alert Management Pseudocode
class AlertManager:
    def __init__(self, config: AlertConfig):
        self.config = config
        self.throttle_state: Dict[str, datetime] = {}
        self.handlers: List[AlertHandler] = []
        self._lock = asyncio.Lock()
        
    async def process_alert(self, detection: VideoDetection):
        # PATTERN: Thread-safe throttling with configurable behavior
        async with self._lock:
            alert_key = f"{detection.source_id}:{detection.pattern_name}"
            now = datetime.now()
            
            # Check throttling
            if alert_key in self.throttle_state:
                time_since_last = (now - self.throttle_state[alert_key]).total_seconds()
                if time_since_last < self.config.throttle_seconds:
                    return  # Skip throttled alert
                    
            # Update throttle state
            self.throttle_state[alert_key] = now
            
            # Broadcast to all handlers
            tasks = []
            for handler in self.handlers:
                if handler.should_handle(detection):
                    tasks.append(handler.handle_alert(detection))
                    
            # CRITICAL: Don't block on handler failures
            await asyncio.gather(*tasks, return_exceptions=True)
```

### Integration Points

```yaml
DEPENDENCIES:
  - pyproject.toml: "Add pygobject >= 3.40.0, PyGObject bindings for GStreamer"
  - pyproject.toml: "Add DeepStream Python bindings with version markers"
  - pyproject.toml: "Add numpy >= 1.19.0, opencv-python >= 4.5.0 for image processing"
  - pyproject.toml: "Add click >= 8.0.0 for modern CLI interface"
  - pyproject.toml: "Add pyyaml >= 6.0 for configuration management"
  
DEEPSTREAM_COMPATIBILITY:
  - version_5x: "Use direct pyds imports and legacy metadata APIs"
  - version_6x_plus: "Use gi.repository.NvDs with updated metadata structure"
  - feature_detection: "Implement runtime capability detection and fallbacks"
  - plugin_compatibility: "Handle nvinfer plugin differences across versions"
  
GSTREAMER_INTEGRATION:
  - pipeline_construction: "Use parse_launch for rapid prototyping, manual linking for production"
  - bus_handling: "Implement proper message bus handling with async event loops"
  - probe_callbacks: "Use buffer probes for detection processing with performance considerations"
  - memory_management: "Proper GstBuffer and metadata lifecycle management"
  
TESTING_STRATEGY:
  - unit_tests: "Mock GStreamer elements and DeepStream components for isolated testing"
  - integration_tests: "Use test video files and synthetic sources for pipeline testing"
  - version_compatibility: "Parameterized tests across supported DeepStream versions"
  - performance_benchmarks: "Automated performance regression testing"
```

## Validation Loop

### Level 1: Syntax & Style

```bash
# Run these FIRST - fix any errors before proceeding
uv run black src/ tests/ examples/          # Code formatting
uv run mypy src/                            # Type checking
uv run ruff check src/ tests/ examples/    # Linting and style

# Expected: No errors. If errors, READ the error and fix.
```

### Level 2: Unit Tests each new feature/file/function use existing test patterns

```python
# CREATE comprehensive test suite with these test cases:
@pytest.mark.asyncio
async def test_pipeline_creation():
    """Test GStreamer pipeline creation with DeepStream elements"""
    config = Config.load_default()
    manager = PipelineManager(config)
    
    source = VideoSource(
        id="test-source",
        name="Test Video",
        source_type=SourceType.TEST,
        uri="videotestsrc"
    )
    
    pipeline_id = await manager.create_pipeline(source)
    assert pipeline_id is not None
    assert pipeline_id in manager.pipelines

@pytest.mark.asyncio 
async def test_detection_processing():
    """Test detection engine with mock DeepStream metadata"""
    engine = DetectionEngine(DetectionConfig())
    mock_metadata = create_mock_deepstream_metadata()
    
    detections = await engine.process_frame_metadata(mock_metadata)
    assert len(detections) > 0
    assert all(d.confidence >= engine.confidence_threshold for d in detections)

@pytest.mark.asyncio
async def test_alert_throttling():
    """Test alert throttling prevents spam"""
    config = AlertConfig(throttle_seconds=5)
    manager = AlertManager(config)
    
    detection = VideoDetection(
        pattern_name="test-pattern",
        confidence=0.8,
        bounding_box=(0.1, 0.1, 0.2, 0.2),
        timestamp=datetime.now(),
        frame_number=1,
        source_id="test-source"
    )
    
    # First alert should go through
    await manager.process_alert(detection)
    
    # Second immediate alert should be throttled
    await manager.process_alert(detection)
    
    # Verify only one alert was processed
    assert len(manager.handlers[0].processed_alerts) == 1

def test_deepstream_version_detection():
    """Test DeepStream version detection and compatibility"""
    version = detect_deepstream_version()
    assert version >= (5, 0)  # Minimum supported version
    
    # Test API selection based on version
    if version >= (6, 0):
        assert get_metadata_api() == "gi.repository"
    else:
        assert get_metadata_api() == "pyds"
```

```bash
# Run and iterate until passing:
uv run pytest tests/ -v --cov=src --cov-report=html
# If failing: Read error, understand root cause, fix code, re-run
```

### Level 3: Integration Test

```bash
# Test with synthetic video source (always available)
uv run python main.py --source test --backend auto --verbose

# Expected: Detection pipeline starts, processes test frames, shows metrics
# If error: Check GStreamer installation and DeepStream availability

# Test with video file (if available)
uv run python main.py --source examples/test_video.mp4 --config configs/development.yaml

# Expected: File processing with detection output to console
# If error: Check file path and format support

# Test configuration validation
uv run python main.py test-config configs/default.yaml

# Expected: Configuration validation passes with component availability report
# If error: Check configuration syntax and component availability

# Test RTSP source (if available)  
uv run python main.py --source rtsp://demo-stream-url --max-frames 100

# Expected: RTSP stream processing with network error handling
# If network error: Should show graceful error handling and recovery attempts
```

## Final validation Checklist

- [ ] All tests pass: `uv run pytest tests/ -v --cov=src`
- [ ] No linting errors: `uv run ruff check src/ tests/ examples/`
- [ ] No type errors: `uv run mypy src/`  
- [ ] Code formatting: `uv run black --check src/ tests/ examples/`
- [ ] Manual test with test video source successful
- [ ] Manual test with file input successful
- [ ] Configuration validation working correctly
- [ ] Alert throttling prevents spam as expected
- [ ] DeepStream version detection working for available versions
- [ ] Error recovery mechanisms function under failure conditions
- [ ] Memory usage remains stable during continuous operation
- [ ] Multi-source processing handles concurrent streams
- [ ] Performance meets real-time requirements (>30 FPS when possible)

---

## Anti-Patterns to Avoid

- ❌ Don't skip GStreamer initialization - will cause segmentation faults
- ❌ Don't ignore DeepStream version differences - APIs are incompatible between major versions
- ❌ Don't use blocking operations in GStreamer callbacks - will cause pipeline stalls
- ❌ Don't assume GPU is always available - must handle CPU fallback gracefully  
- ❌ Don't forget proper metadata lifecycle management - will cause memory leaks
- ❌ Don't hardcode DeepStream plugin configurations - different versions have different parameters
- ❌ Don't skip bus message handling - critical errors may go unnoticed
- ❌ Don't use synchronous I/O in async contexts - will block event loops
- ❌ Don't ignore threading requirements - GStreamer callbacks execute in different threads

## Confidence Score: 8/10

This PRP provides comprehensive context including:
- ✅ DeepStream version compatibility strategy (5.x through 7.x)
- ✅ Complete GStreamer Python integration patterns with error handling
- ✅ Backwards compatibility approach with runtime detection
- ✅ Modern Python development practices using uv package manager
- ✅ Comprehensive test strategy covering unit, integration, and performance testing
- ✅ Real-world video source handling (RTSP, WebRTC, file, webcam)
- ✅ Production-ready error handling and recovery mechanisms
- ✅ Extensible architecture for custom detection strategies
- ✅ Alert management with throttling and multi-handler support
- ✅ Clear implementation order with detailed pseudocode

The score is 8/10 due to the complexity of supporting multiple DeepStream versions and the intricacies of GStreamer Python bindings, but the detailed guidance and backwards compatibility strategy should ensure successful implementation. The comprehensive documentation references and proven patterns from the NVIDIA community provide strong foundation for success.
