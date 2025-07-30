# PyDS User Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [Basic Usage](#basic-usage)
3. [Configuration](#configuration)
4. [Video Sources](#video-sources)
5. [Detection Strategies](#detection-strategies)
6. [Alert Management](#alert-management)
7. [Monitoring](#monitoring)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Features](#advanced-features)

## Quick Start

### 1. Installation

First, ensure you have the prerequisites:
- NVIDIA GPU with CUDA support
- NVIDIA DeepStream SDK (5.x, 6.x, or 7.x)
- Python 3.8+

Install PyDS:
```bash
# Clone the repository
git clone https://github.com/your-org/pyds-app.git
cd pyds-app

# Install with UV (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Or install with pip
pip install -e .
```

### 2. Verify Installation

```bash
# Check environment
python scripts/setup_environment.py --check

# Test with synthetic video
python main.py run --source test
```

### 3. Your First Detection

Create a simple configuration file `my-config.yaml`:
```yaml
sources:
  - id: webcam
    name: "USB Camera"
    type: webcam
    uri: "0"
    enabled: true

alerts:
  enabled: true
  handlers:
    - console
```

Run detection:
```bash
python main.py run --config my-config.yaml
```

## Basic Usage

### Command Line Interface

The main interface is through the `main.py` command:

```bash
# Show help
python main.py --help

# Run with default settings
python main.py run

# Run with specific source
python main.py run --source rtsp://camera.ip/stream

# Run with configuration file
python main.py run --config configs/production.yaml

# Check system status
python main.py status

# Validate configuration
python main.py validate-config my-config.yaml
```

### Basic Commands

| Command | Description |
|---------|-------------|
| `run` | Start the video analytics system |
| `status` | Show system status and health |
| `validate-config` | Check configuration file validity |
| `monitor` | Launch interactive monitoring dashboard |
| `benchmark` | Run performance benchmarks |

## Configuration

### Configuration Structure

PyDS uses YAML configuration files with the following main sections:

```yaml
# Application settings
name: "My Video Analytics"
environment: development  # development, staging, production
debug: false

# Pipeline configuration
pipeline:
  batch_size: 4
  width: 1920
  height: 1080
  fps: 30.0
  gpu_id: 0

# Detection settings
detection:
  confidence_threshold: 0.7
  max_objects: 100
  strategies:
    - name: yolo
      enabled: true

# Alert configuration
alerts:
  enabled: true
  throttle_seconds: 60
  handlers:
    - console
    - file

# Video sources
sources:
  - id: camera1
    name: "Front Camera"
    type: rtsp
    uri: "rtsp://192.168.1.100/stream"
```

### Environment Variables

Override configuration with environment variables:

```bash
export PYDS_LOG_LEVEL=debug
export PYDS_GPU_ID=1
export PYDS_CONFIDENCE_THRESHOLD=0.8
export PYDS_ALERTS_ENABLED=true
```

### Configuration Validation

Always validate your configuration:

```bash
python main.py validate-config my-config.yaml
```

## Video Sources

### Supported Source Types

#### 1. RTSP Cameras
```yaml
sources:
  - id: rtsp_camera
    name: "IP Camera"
    type: rtsp
    uri: "rtsp://username:password@192.168.1.100:554/stream1"
    parameters:
      latency: 200
      timeout: 30
```

#### 2. USB Webcams
```yaml
sources:
  - id: webcam
    name: "USB Camera"
    type: webcam
    uri: "0"  # Device index or /dev/video0
    parameters:
      width: 1280
      height: 720
      framerate: 30
```

#### 3. Video Files
```yaml
sources:
  - id: video_file
    name: "Recorded Video"
    type: file
    uri: "file:///path/to/video.mp4"
    parameters:
      loop: true
```

#### 4. Test Patterns
```yaml
sources:
  - id: test_source
    name: "Test Pattern"
    type: test
    uri: "videotestsrc pattern=ball"
```

### Multi-Source Setup

Process multiple sources simultaneously:

```yaml
sources:
  - id: front_door
    name: "Front Door Camera"
    type: rtsp
    uri: "rtsp://192.168.1.100/stream"
    
  - id: back_yard
    name: "Backyard Camera"
    type: rtsp
    uri: "rtsp://192.168.1.101/stream"
    
  - id: parking_lot
    name: "Parking Lot Camera"
    type: rtsp
    uri: "rtsp://192.168.1.102/stream"

pipeline:
  batch_size: 3  # Process all 3 sources together
```

### Source Health Monitoring

PyDS automatically monitors source health:
- Connection status
- Frame rate stability
- Automatic reconnection
- Error tracking

## Detection Strategies

### Built-in Strategies

#### 1. YOLO Detection
```yaml
detection:
  strategies:
    - name: yolo
      enabled: true
      config:
        model_path: "/path/to/yolo.engine"
        confidence_threshold: 0.7
        nms_threshold: 0.4
```

#### 2. Template Matching
```yaml
detection:
  strategies:
    - name: template
      enabled: true
      config:
        template_dir: "/path/to/templates/"
        match_threshold: 0.8
```

#### 3. Feature-Based Detection
```yaml
detection:
  strategies:
    - name: feature
      enabled: true
      config:
        detector_type: "ORB"
        min_matches: 10
```

### Custom Detection Strategies

Create your own detection strategy:

```python
# my_detection.py
from src.detection.models import DetectionStrategy, VideoDetection

class LogoDetector(DetectionStrategy):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.logo_template = None
    
    async def initialize(self):
        # Load your model/templates
        return True
    
    def should_process(self, detection):
        return detection.confidence > 0.8
    
    async def process(self, detection):
        # Your detection logic here
        return [detection]  # Return modified detections
```

Register in configuration:
```yaml
detection:
  strategies:
    - name: logo_detector
      enabled: true
      module: "my_detection"
      class: "LogoDetector"
      config:
        template_path: "/path/to/logo.png"
```

## Alert Management

### Alert Handlers

#### Console Output
```yaml
alerts:
  handlers:
    - name: console
      enabled: true
      config:
        colored_output: true
        include_timestamp: true
```

#### File Logging
```yaml
alerts:
  handlers:
    - name: file
      enabled: true
      config:
        log_file: "/var/log/pyds-alerts.log"
        max_size_mb: 100
        backup_count: 5
```

#### Webhook Notifications
```yaml
alerts:
  handlers:
    - name: webhook
      enabled: true
      config:
        url: "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
        timeout: 30
        retry_attempts: 3
```

#### Email Alerts
```yaml
alerts:
  handlers:
    - name: email
      enabled: true
      config:
        smtp_server: "smtp.gmail.com"
        smtp_port: 587
        username: "your-email@gmail.com"
        password: "your-password"
        to_addresses:
          - "admin@company.com"
          - "security@company.com"
```

### Alert Throttling

Prevent alert spam with intelligent throttling:

```yaml
alerts:
  # Basic throttling
  throttle_seconds: 60  # Minimum time between identical alerts
  
  # Burst protection
  burst_threshold: 5    # Max alerts in burst period
  max_alerts_per_minute: 20
  
  # Confidence filtering
  min_confidence: 0.6   # Only alert on high-confidence detections
```

## Monitoring

### Performance Metrics

PyDS collects comprehensive metrics:
- Frame processing rate (FPS)
- Detection latency
- CPU and memory usage
- GPU utilization
- Source health status

### Real-time Monitoring

Launch the monitoring dashboard:
```bash
python main.py monitor --dashboard
```

### Prometheus Integration

Export metrics to Prometheus:
```yaml
monitoring:
  prometheus_enabled: true
  prometheus_port: 8000
  prometheus_path: "/metrics"
```

Access metrics at: `http://localhost:8000/metrics`

### Health Checks

Check system health:
```bash
# Basic health check
python main.py health-check

# Detailed health report
python main.py health-check --detailed

# Health check endpoint
curl http://localhost:8000/health
```

## Troubleshooting

### Common Issues

#### 1. "DeepStream not found"
```bash
# Check DeepStream installation
python scripts/setup_environment.py --validate-gpu

# Install DeepStream SDK
# Follow NVIDIA's installation guide
```

#### 2. "GPU out of memory"
Reduce memory usage:
```yaml
pipeline:
  batch_size: 2        # Reduce from 4
  width: 1280         # Reduce resolution
  height: 720
  memory_type: "unified"
```

#### 3. "Low FPS performance"
Optimize for performance:
```yaml
pipeline:
  batch_size: 8            # Increase batching
  enable_gpu_inference: true
  processing_mode: "batch"

detection:
  tensor_rt_precision: "fp16"  # Use FP16 for speed
```

#### 4. "RTSP connection failed"
```bash
# Test RTSP manually
gst-launch-1.0 rtspsrc location=rtsp://your.stream.url ! fakesink

# Check network connectivity
ping camera.ip
telnet camera.ip 554
```

### Debug Mode

Enable detailed logging:
```bash
export PYDS_LOG_LEVEL=debug
python main.py run --config my-config.yaml --verbose
```

### Performance Profiling

Run performance analysis:
```bash
# Benchmark system performance
python scripts/benchmark_performance.py --sources 4 --duration 60

# Profile memory usage
python scripts/benchmark_performance.py --profile
```

## Advanced Features

### Custom Pipeline Configuration

Create custom pipeline templates:
```python
# custom_pipeline.py
from src.pipeline.factory import PipelineBuilder

class CustomPipeline(PipelineBuilder):
    async def build_pipeline(self, sources, **kwargs):
        # Your custom pipeline logic
        pass
```

### Plugin Development

Develop detection plugins:
```python
# plugins/my_plugin.py
PLUGIN_INFO = {
    'name': 'my_detector',
    'version': '1.0',
    'author': 'Your Name',
    'description': 'Custom detection plugin'
}

class MyDetectionStrategy(DetectionStrategy):
    # Implementation here
    pass
```

### API Integration

PyDS can be integrated into larger systems:
```python
from src.app import PyDSApp
from src.config import AppConfig

# Programmatic usage
config = AppConfig(name="My App")
app = PyDSApp(config)

await app.initialize()
await app.start()
# Your integration code
await app.stop()
```

### Batch Processing

Process video files in batch:
```bash
# Process multiple files
for file in /path/to/videos/*.mp4; do
    python main.py run --source "file://$file" --duration 300
done
```

### Deployment

#### Docker Deployment
```dockerfile
FROM nvcr.io/nvidia/deepstream:7.0-devel
COPY . /app
WORKDIR /app
RUN pip install -e .
CMD ["python", "main.py", "run", "--config", "configs/production.yaml"]
```

#### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pyds-app
spec:
  template:
    spec:
      containers:
      - name: pyds
        image: pyds-app:latest
        resources:
          limits:
            nvidia.com/gpu: 1
```

---

For more detailed information, see:
- [API Reference](api-reference.md)
- [Configuration Reference](configuration-reference.md)
- [Developer Guide](developer-guide.md)
- [Troubleshooting Guide](troubleshooting.md)