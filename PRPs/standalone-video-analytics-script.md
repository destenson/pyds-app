---
name: "Standalone Video Analytics Script"
description: |
  
## Purpose
Implement a robust, standalone Python script for real-time pattern detection in video streams using GStreamer, YOLO, and DeepStream. This is a completely independent application designed for production deployment with REST API control, automatic error recovery, and multi-source processing capabilities.

## Core Principles
1. **Independence**: Standalone script with no dependencies on existing codebase
2. **Robustness**: Automatic recovery from failures and comprehensive error handling
3. **Performance**: High-throughput, low-latency processing with GPU acceleration
4. **Flexibility**: Dynamic source management and configurable detection parameters
5. **Production-Ready**: Monitoring, logging, and deployment capabilities

---

## Goal
Create a production-ready standalone video analytics script that:
- **Processes multiple video sources** simultaneously (RTSP, WebRTC, file, webcam)
- **Implements YOLO object detection** with DeepStream GPU acceleration
- **Provides REST API control** for dynamic source and parameter management
- **Handles automatic recovery** from pipeline failures and network issues
- **Supports custom models** and configurable detection parameters
- **Includes comprehensive monitoring** and logging capabilities

## Why
- **Standalone Deployment**: Independent script that can be deployed anywhere
- **Production Robustness**: Auto-recovery from common video streaming failures
- **API Integration**: Easy integration with larger systems via REST API
- **Performance**: GPU-accelerated processing for high-throughput scenarios
- **Flexibility**: Support for various video sources and custom detection models
- **Operational Excellence**: Comprehensive logging and monitoring for production use

## What
A single Python script that provides:
1. **Multi-source video processing** pipeline using GStreamer + DeepStream
2. **YOLO object detection** with configurable models and parameters  
3. **REST API server** for dynamic source and configuration management
4. **Automatic error recovery** and pipeline health monitoring
5. **Comprehensive logging** and performance metrics
6. **Docker deployment** support with proper GPU access
7. **Configuration management** via files and environment variables

### Success Criteria
- [ ] Process 4+ concurrent video sources at 30 FPS with <100ms latency
- [ ] Automatic recovery from network disconnections within 30 seconds
- [ ] REST API for source management (add/remove/configure) with <500ms response
- [ ] Support for custom YOLO models with configurable confidence thresholds
- [ ] Zero-downtime source addition/removal during operation
- [ ] Comprehensive error logging with structured JSON output
- [ ] GPU memory usage optimization for high-resolution streams
- [ ] Docker deployment with proper NVIDIA runtime integration

## All Needed Context

### Documentation & References
```yaml
# MUST READ - Include these in your context window
- url: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Python_Sample_Apps.html
  why: Official DeepStream Python development patterns and examples
  critical: Understanding pipeline creation and metadata handling
  section: Multi-stream processing and dynamic source management

- url: https://github.com/NVIDIA-AI-IOT/deepstream_python_apps
  why: Official NVIDIA DeepStream Python examples with multi-source patterns
  critical: deepstream-test3 for multi-stream, runtime_source_add_delete for dynamic management
  focus: apps/deepstream-test3/ and apps/runtime_source_add_delete/

- url: https://github.com/marcoslucianops/DeepStream-Yolo
  why: Comprehensive YOLO integration with DeepStream supporting multiple YOLO versions
  critical: Custom post-processing and configuration management
  focus: Model configuration files and inference engine setup

- url: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_RestServer.html
  why: DeepStream REST API patterns for runtime configuration
  critical: Dynamic pipeline management and parameter updates
  section: Application integration patterns and API endpoints

- url: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_docker_containers.html
  why: Docker deployment patterns for DeepStream applications
  critical: Production deployment with GPU access and environment setup

- url: https://gstreamer.freedesktop.org/documentation/application-development/advanced/pipeline-manipulation.html
  why: Dynamic pipeline manipulation patterns for robust applications
  critical: Adding/removing sources during runtime and error recovery
```

### Current Technology Stack
```bash
# Core Dependencies
GStreamer 1.0              # Media framework
NVIDIA DeepStream 6.3+     # GPU-accelerated video analytics
Python 3.8+                # Runtime environment
PyGObject (gi)             # GStreamer Python bindings
FastAPI                    # REST API framework
OpenCV                     # Computer vision utilities
NumPy                      # Numerical processing

# YOLO Model Support
YOLOv5, YOLOv7, YOLOv8     # Object detection models
TensorRT                   # GPU inference optimization
CUDA 11.0+                 # GPU acceleration
```

### Standalone Script Architecture
```bash
video_analytics_script.py   # Main standalone script
├── VideoAnalyticsEngine    # Core processing engine
├── GStreamerPipeline       # Pipeline management
├── YOLODetector           # YOLO inference handler
├── SourceManager          # Multi-source management  
├── RestAPIServer          # FastAPI server
├── ErrorRecovery          # Auto-recovery mechanisms
├── HealthMonitor          # Pipeline health checks
├── ConfigManager          # Configuration handling
└── Logger                 # Structured logging
```

### Known Gotchas & Critical Patterns

**CRITICAL: GStreamer Initialization Pattern**
- Must call `Gst.init()` before any GStreamer operations
- Requires `gi.require_version('Gst', '1.0')` before import
- Thread-safe initialization for multi-threaded applications

**CRITICAL: DeepStream Pipeline Pattern**
```python
# Standard DeepStream multi-source pipeline
nvstreammux -> nvinfer -> nvtracker -> nvdsosd -> nvmultistreamtiler
```

**CRITICAL: Dynamic Source Management**
- Use `nvstreammux` request pads for dynamic source addition
- Proper pad linking/unlinking for source removal
- State management during runtime modifications

**CRITICAL: Error Recovery Patterns**
- Monitor bus messages for ERROR, WARNING, and EOS
- Implement pipeline recreation for network source failures
- Use probe callbacks for monitoring data flow

**CRITICAL: Memory Management**
- Use NVMM memory for zero-copy GPU processing
- Proper metadata allocation/deallocation
- Batch processing for GPU efficiency

**CRITICAL: REST API Integration**
- Run FastAPI server in separate thread from GStreamer
- Use thread-safe queues for command communication
- Implement proper graceful shutdown

## Implementation Blueprint

### Core Architecture Pattern
```python
#!/usr/bin/env python3
"""
Standalone Video Analytics Script with GStreamer, YOLO, and DeepStream.
Production-ready script for multi-source video pattern detection.
"""

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GObject', '2.0')

from gi.repository import Gst, GObject, GLib
import asyncio
import threading
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import logging
import json
import time
from dataclasses import dataclass
from enum import Enum
```

### Data Models
```python
class SourceType(str, Enum):
    RTSP = "rtsp"
    WEBRTC = "webrtc" 
    FILE = "file"
    WEBCAM = "webcam"

@dataclass
class VideoSource:
    id: str
    type: SourceType
    uri: str
    enabled: bool = True
    detection_config: Dict[str, Any] = None

@dataclass 
class DetectionResult:
    source_id: str
    timestamp: float
    objects: List[Dict[str, Any]]
    frame_number: int
```

### List of Tasks to Complete (in order)

```yaml
Task 1: "Setup project structure and dependencies"
CREATE video_analytics_script.py:
  - Setup GStreamer and DeepStream imports with fallback handling
  - Initialize logging system with structured JSON output
  - Create main application class VideoAnalyticsEngine
  - Implement basic configuration loading from JSON/YAML
  - Add Docker deployment configuration

Task 2: "Implement core GStreamer pipeline management"
IMPLEMENT GStreamerPipeline class:
  - Create basic multi-source pipeline with nvstreammux
  - Add YOLO inference pipeline with nvinfer element
  - Implement pipeline state management (NULL, READY, PLAYING)
  - Add bus message handling for errors and state changes
  - Create probe callbacks for metadata extraction

Task 3: "Implement YOLO detection engine"
IMPLEMENT YOLODetector class:
  - Configure nvinfer element with YOLO model
  - Implement metadata parsing for object detection results
  - Add confidence threshold filtering
  - Create detection result formatting
  - Support for multiple YOLO model formats (v5, v7, v8)

Task 4: "Implement dynamic source management"
IMPLEMENT SourceManager class:
  - Add methods for runtime source addition/removal
  - Implement proper pad management for nvstreammux
  - Add source health monitoring and status tracking
  - Create source configuration validation
  - Handle source-specific parameters (resolution, FPS)

Task 5: "Implement REST API server"
IMPLEMENT RestAPIServer class:
  - Create FastAPI application with source management endpoints
  - Add endpoints: GET/POST/DELETE /sources, GET /status, POST /config
  - Implement thread-safe communication with pipeline
  - Add request validation and error handling
  - Create OpenAPI documentation

Task 6: "Implement error recovery and health monitoring"
IMPLEMENT ErrorRecovery and HealthMonitor classes:
  - Add pipeline failure detection and automatic restart
  - Implement source-specific recovery strategies
  - Create health check endpoints and metrics
  - Add performance monitoring (FPS, latency, memory)
  - Implement graceful shutdown handling

Task 7: "Implement configuration management"
IMPLEMENT ConfigManager class:
  - Support for JSON/YAML configuration files
  - Environment variable overrides
  - Runtime configuration updates via API
  - Model configuration management
  - Validation and schema checking

Task 8: "Add comprehensive logging and monitoring"
IMPLEMENT Logger class:
  - Structured JSON logging with correlation IDs
  - Performance metrics collection
  - Error tracking and alerting
  - Pipeline state logging
  - Integration with external monitoring systems

Task 9: "Create Docker deployment configuration"
CREATE Docker deployment:
  - Multi-stage Dockerfile with DeepStream base image
  - docker-compose.yml for development and production
  - NVIDIA runtime configuration
  - Volume mounting for models and configuration
  - Health check configuration

Task 10: "Add comprehensive testing and validation"
CREATE test suite:
  - Unit tests for core components
  - Integration tests with mock video sources
  - Performance benchmarking scripts
  - Docker deployment validation
  - API endpoint testing
```

### Implementation Approach

**Task 2 - GStreamer Pipeline Pattern:**
```python
def create_pipeline(self):
    # Create pipeline elements
    self.pipeline = Gst.Pipeline("video-analytics")
    self.streammux = Gst.ElementFactory.make("nvstreammux", "muxer")
    self.nvinfer = Gst.ElementFactory.make("nvinfer", "infer")
    self.tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    
    # Configure batch processing
    self.streammux.set_property("batch-size", 4)
    self.streammux.set_property("batched-push-timeout", 40000)
```

**Task 3 - YOLO Configuration Pattern:**
```python
def configure_yolo_inference(self, model_path: str, config_path: str):
    self.nvinfer.set_property("config-file-path", config_path)
    self.nvinfer.set_property("model-engine-file", model_path)
    self.nvinfer.set_property("batch-size", 4)
```

**Task 4 - Dynamic Source Management:**
```python
def add_source(self, source_config: VideoSource):
    # Create source element based on type
    if source_config.type == SourceType.RTSP:
        source = Gst.ElementFactory.make("rtspsrc", f"src-{source_config.id}")
        source.set_property("location", source_config.uri)
    
    # Get request pad from streammux
    sinkpad = self.streammux.get_request_pad(f"sink_{len(self.sources)}")
    
    # Link source to muxer
    srcpad = source.get_static_pad("src")
    srcpad.link(sinkpad)
```

**Task 5 - REST API Pattern:**
```python
@app.post("/api/v1/sources")
async def add_source(source: VideoSource):
    try:
        success = await self.engine.add_source(source)
        if success:
            return {"status": "success", "source_id": source.id}
        else:
            raise HTTPException(status_code=400, detail="Failed to add source")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Integration Points
```yaml
GSTREAMER_PIPELINE:
  - multi_source_input: nvstreammux element with request pads
  - yolo_inference: nvinfer element with custom model configuration
  - metadata_extraction: probe callbacks on inference output
  - display_output: nvdsosd for overlay, nvegltransform for display

DEEPSTREAM_INTEGRATION:
  - batch_metadata: NvDsBatchMeta for multi-source processing
  - object_metadata: NvDsObjectMeta for detection results
  - memory_management: NVMM memory for GPU direct access
  - custom_models: Support for TensorRT optimized YOLO models

REST_API_ENDPOINTS:
  - source_management: CRUD operations for video sources
  - configuration: Runtime parameter updates
  - monitoring: Health checks and performance metrics
  - model_management: Dynamic model loading and switching

ERROR_RECOVERY:
  - pipeline_recreation: Full restart for complex failures
  - source_reconnection: Individual source recovery
  - health_monitoring: Continuous pipeline state checking
  - graceful_shutdown: Proper cleanup and resource management
```

## Validation Loop

### Level 1: Environment and Dependencies
```bash
# Verify DeepStream installation
deepstream-app --version

# Check GStreamer plugins
gst-inspect-1.0 nvstreammux
gst-inspect-1.0 nvinfer

# Python dependencies
python -c "import gi; gi.require_version('Gst', '1.0'); from gi.repository import Gst; print('GStreamer OK')"

# Expected: All dependencies available, no import errors
```

### Level 2: Basic Pipeline Creation
```bash
# Test basic pipeline creation and state transitions
python video_analytics_script.py --test-pipeline

# Test YOLO model loading
python video_analytics_script.py --test-inference --model /path/to/yolo.engine

# Test multi-source handling
python video_analytics_script.py --test-sources --sources test_sources.json

# Expected: Pipeline creates successfully, model loads, sources connect
```

### Level 3: REST API and Dynamic Management
```bash
# Start the application
python video_analytics_script.py --config config.json

# Test REST API endpoints
curl -X POST http://localhost:8080/api/v1/sources -H "Content-Type: application/json" -d '{"id": "test", "type": "rtsp", "uri": "rtsp://example.com/stream"}'

curl -X GET http://localhost:8080/api/v1/status

# Test dynamic source management
curl -X DELETE http://localhost:8080/api/v1/sources/test

# Expected: API responds correctly, sources added/removed dynamically
```

### Level 4: Error Recovery and Performance
```bash
# Test error recovery by disconnecting network
python video_analytics_script.py --config config.json &
# Disconnect network, reconnect after 30 seconds

# Performance benchmarking
python video_analytics_script.py --benchmark --sources 4 --duration 60

# Memory usage monitoring
python video_analytics_script.py --config config.json --monitor-memory

# Expected: Automatic recovery, acceptable performance metrics, stable memory usage
```

## Final Validation Checklist
- [ ] Pipeline processes multiple video sources simultaneously: `python video_analytics_script.py --test-multi-source`
- [ ] YOLO detection works with configurable parameters: `curl -X PUT /api/v1/config/detection`
- [ ] REST API enables dynamic source management: `curl -X POST /api/v1/sources`
- [ ] Automatic error recovery from network failures: Network disconnect test
- [ ] Performance meets requirements (30 FPS, <100ms latency): `--benchmark` mode
- [ ] Docker deployment works with GPU access: `docker-compose up`
- [ ] Logging provides structured output: Check JSON log format
- [ ] Configuration management via files and API: Test config updates

---

## Anti-Patterns to Avoid
- ❌ Don't create tight coupling with existing codebase - keep it standalone
- ❌ Don't ignore GStreamer message bus - implement proper error handling
- ❌ Don't use blocking operations in main thread - use async patterns
- ❌ Don't forget NVMM memory management - leads to memory leaks
- ❌ Don't skip proper pipeline state transitions - causes instability
- ❌ Don't ignore thread safety between GStreamer and REST API
- ❌ Don't hardcode paths or parameters - make everything configurable
- ❌ Don't skip graceful shutdown handling - can corrupt streams

## Confidence Score: 8/10

This PRP provides comprehensive context for implementing a production-grade standalone video analytics script. The high confidence comes from:
- **Rich Documentation**: Detailed NVIDIA DeepStream and GStreamer references
- **Real-world Patterns**: Based on proven production implementations
- **Comprehensive Testing**: Multiple validation levels ensure robustness
- **Production Focus**: Addresses real deployment and operational concerns
- **Error Recovery**: Detailed patterns for handling common failure modes

Risk areas: Complex GStreamer/DeepStream integration and potential hardware-specific issues, but these are mitigated through comprehensive documentation and fallback patterns.