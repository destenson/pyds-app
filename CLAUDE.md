# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based video analytics application using NVIDIA DeepStream and GStreamer for real-time pattern detection across multiple video sources. The project is in early development stage with comprehensive planning documented in PRPs (Project Requirement Proposals).

## Key Commands

### Development Setup
```bash
# Install dependencies using uv (modern Python package manager)
uv pip install -e .

# Run the application
python main.py

# Run specific module
python -m src.app
```

### Testing
```bash
# Run all tests
pytest
python -m pytest

# Run specific test file
pytest tests/test_config.py

# Run with coverage
pytest --cov=src
```

### Code Quality
```bash
# Linting
ruff check
ruff check --fix  # Auto-fix issues

# Type checking (if configured)
python -m mypy src/
```

## Architecture Overview

### Core Components

1. **Pipeline Management** (`src/pipeline/`)
   - GStreamer pipeline lifecycle management
   - Video source abstraction for RTSP, WebRTC, file, and webcam inputs
   - DeepStream element configuration and management

2. **Detection Engine** (`src/detection/`)
   - Strategy pattern for extensible detection algorithms
   - Custom pattern registration and management
   - Frame analysis and metadata extraction

3. **Alert System** (`src/alerts/`)
   - Throttled alert broadcasting to prevent spam
   - Multiple output handlers (console, file, network)
   - Thread-safe alert queue management

4. **Monitoring** (`src/monitoring/`)
   - Performance metrics collection
   - Health checks and automatic recovery
   - Pipeline state tracking

### Critical Implementation Notes

1. **DeepStream Version Compatibility**
   - Support for DeepStream 5.x through 7.x
   - Version detection required before imports
   - Different Python binding APIs between versions (pyds vs gi.repository.NvDs)

2. **GStreamer Integration**
   - Must call `gi.require_version('Gst', '1.0')` before imports
   - Initialize with `Gst.init()` before creating pipeline elements
   - Handle asynchronous state transitions properly
   - Bus message handling requires event loop integration

3. **Thread Safety**
   - GStreamer callbacks execute in different threads
   - Alert broadcasting must be thread-safe
   - Configuration updates require proper locking

4. **Performance Considerations**
   - Minimize probe function usage on high-frequency elements
   - Manage Python GIL impact with proper async patterns
   - Critical memory management in callbacks to prevent leaks

## Project Resources

- **PRPs/**: Contains detailed implementation plans following structured templates
- **TODO.md**: Comprehensive requirements specification
- **.claude/settings.local.json**: Allowed bash commands for development

## Dependencies

Key dependencies include:
- pygobject (GStreamer Python bindings)
- gst-python
- numpy
- opencv-python
- pyyaml
- click (CLI interface)
- pytest (testing)
- ruff (linting)

## Development Workflow

1. Always refer to the vision-deepstream-python-implementation.md PRP for detailed implementation guidance
2. Follow the modular architecture outlined in the PRP
3. Implement proper error handling and recovery mechanisms
4. Use structured logging throughout the application
5. Write tests for new functionality using pytest
6. Ensure DeepStream version compatibility in all components