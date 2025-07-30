# PyDS-App: Robust DeepStream Inference System

A production-ready, multi-source video analytics system using NVIDIA DeepStream and GStreamer for real-time pattern detection with intelligent alerting and automatic error recovery.

## 🚀 Features

- **Multi-Source Processing**: Concurrent RTSP, WebRTC, file, and webcam input handling
- **DeepStream Compatibility**: Support for DeepStream 5.x, 6.x, and 7.x with automatic version detection
- **Extensible Detection**: Plugin-based detection strategies for YOLO, custom models, and template matching
- **Intelligent Alerting**: Configurable throttling with burst detection and spam prevention
- **Automatic Recovery**: Fault tolerance with exponential backoff and graceful degradation
- **Real-time Monitoring**: FPS tracking, memory usage, and processing latency metrics
- **Production Ready**: Enterprise-grade error handling, logging, and health monitoring

## 📋 Requirements

### System Requirements

**Minimum:**
- NVIDIA GPU with CUDA Compute Capability 6.0+
- CUDA 11.4+ or 12.x
- Python 3.8+
- 8GB RAM
- Ubuntu 20.04+ or Windows 11

**Recommended:**
- NVIDIA RTX series GPU
- CUDA 12.x
- Python 3.10+
- 16GB+ RAM
- Ubuntu 22.04 LTS

### DeepStream Compatibility Matrix

| DeepStream Version | CUDA Version | Python API | Status |
|-------------------|---------------|------------|--------|
| 5.0 - 5.1         | 11.4+        | `pyds`     | ✅ Supported |
| 6.0 - 6.4         | 11.4+        | `gi.repository` | ✅ Supported |
| 7.0+              | 12.x         | `gi.repository` | ✅ Supported |

## 🛠️ Installation

### Quick Start with UV (Recommended)

```bash
# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone https://github.com/deepstream-analytics/pyds-app.git
cd pyds-app

# Install dependencies
uv sync

# Validate installation
uv run pyds-validate
```

### Manual Installation

#### Ubuntu 20.04+ Installation

1. **Install NVIDIA Driver and CUDA**
   ```bash
   # Install NVIDIA driver
   sudo apt update
   sudo apt install nvidia-driver-535
   
   # Install CUDA Toolkit 12.x
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
   sudo dpkg -i cuda-keyring_1.0-1_all.deb
   sudo apt update
   sudo apt install cuda-toolkit-12-0
   
   # Add CUDA to PATH
   echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
   source ~/.bashrc
   ```

2. **Install DeepStream SDK**
   ```bash
   # Download DeepStream 7.0 (or latest version)
   wget https://developer.nvidia.com/downloads/deepstream-7.0/deepstream_7.0-1_amd64.deb
   sudo apt install ./deepstream_7.0-1_amd64.deb
   
   # Install DeepStream Python bindings
   cd /opt/nvidia/deepstream/deepstream-7.0/sources/deepstream_python_apps/bindings
   python3 setup.py build_ext --inplace
   pip3 install ./
   ```

3. **Install GStreamer Dependencies**
   ```bash
   sudo apt install \
       libgstreamer1.0-dev \
       libgstreamer-plugins-base1.0-dev \
       libgstreamer-plugins-bad1.0-dev \
       gstreamer1.0-plugins-base \
       gstreamer1.0-plugins-good \
       gstreamer1.0-plugins-bad \
       gstreamer1.0-plugins-ugly \
       gstreamer1.0-libav \
       gstreamer1.0-tools \
       gstreamer1.0-x \
       gstreamer1.0-alsa \
       gstreamer1.0-gl \
       gstreamer1.0-gtk3 \
       gstreamer1.0-qt5 \
       gstreamer1.0-pulseaudio \
       python3-gi \
       python3-gi-cairo \
       gir1.2-gstreamer-1.0
   ```

4. **Install Python Dependencies**
   ```bash
   pip3 install -e .
   ```

#### Windows 11 Installation

1. **Install NVIDIA Driver and CUDA**
   - Download and install latest NVIDIA driver from [nvidia.com](https://www.nvidia.com/drivers/)
   - Download and install CUDA Toolkit 12.x from [NVIDIA Developer](https://developer.nvidia.com/cuda-downloads)

2. **Install DeepStream SDK**
   - Download DeepStream 7.0 Windows installer
   - Run installer with administrator privileges
   - Add DeepStream to system PATH

3. **Install Python and Dependencies**
   ```powershell
   # Install Python 3.10+ from python.org
   # Install PyDS-App
   pip install -e .
   ```

### Docker Installation (Recommended for Production)

```bash
# Use NVIDIA DeepStream base image
docker pull nvcr.io/nvidia/deepstream:7.0-devel

# Build application container
docker build -t pyds-app .

# Run with GPU support
docker run --gpus all -it pyds-app
```

## 🚀 Quick Start

### Basic Usage

```bash
# Start with test video source
uv run pyds-app start --source test --config configs/development.yaml

# Process video file
uv run pyds-app start --source file:///path/to/video.mp4

# Process RTSP stream
uv run pyds-app start --source rtsp://camera.ip/stream

# Multi-source processing
uv run pyds-app start --sources configs/multi-source.yaml
```

### Configuration

Create a configuration file `my-config.yaml`:

```yaml
# DeepStream Pipeline Configuration
pipeline:
  batch_size: 4
  width: 1920
  height: 1080
  fps: 30

# Detection Configuration  
detection:
  confidence_threshold: 0.7
  strategies:
    - name: "yolo"
      model_path: "/path/to/yolo.engine"
      enabled: true
    - name: "custom"
      module: "my_detection.py"
      enabled: false

# Alert Configuration
alerts:
  enabled: true
  throttle_seconds: 60
  handlers:
    - console
    - file
    - webhook

# Monitoring Configuration
monitoring:
  metrics_enabled: true
  health_check_interval: 30
  profiling_enabled: false
```

Run with custom configuration:
```bash
uv run pyds-app start --config my-config.yaml --sources my-sources.yaml
```

## 📖 Advanced Usage

### Custom Detection Strategies

Create a custom detection strategy:

```python
# my_detection.py
from src.detection.custom import DetectionStrategy
from src.detection.models import VideoDetection

class MyCustomStrategy(DetectionStrategy):
    def __init__(self, config):
        self.config = config
        
    async def detect(self, frame_meta, source_id: str) -> list[VideoDetection]:
        # Implement custom detection logic
        detections = []
        # ... your detection code here ...
        return detections
        
    def should_process(self, detection: VideoDetection) -> bool:
        return detection.confidence > 0.8
```

Register and use:
```yaml
detection:
  strategies:
    - name: "my_custom"
      module: "my_detection"
      class: "MyCustomStrategy"
      enabled: true
      config:
        threshold: 0.8
```

### Custom Alert Handlers

```python
# my_alerts.py
from src.alerts.handlers import AlertHandler

class SlackHandler(AlertHandler):
    async def handle_alert(self, alert):
        # Send alert to Slack
        await self.send_to_slack(alert.message)
```

### Performance Monitoring

```bash
# Real-time performance monitoring
uv run pyds-app monitor --dashboard

# Performance benchmarking
uv run pyds-benchmark --sources 4 --duration 300 --profile

# Health check
uv run pyds-app health-check
```

## 🔧 Configuration Reference

### Pipeline Configuration

```yaml
pipeline:
  batch_size: 4              # Number of streams to batch together
  width: 1920               # Processing width
  height: 1080              # Processing height
  fps: 30                   # Target FPS
  buffer_pool_size: 10      # Buffer pool size for memory management
  gpu_id: 0                 # GPU device ID to use
```

### Detection Configuration

```yaml
detection:
  confidence_threshold: 0.7  # Minimum confidence for detections
  nms_threshold: 0.5        # Non-maximum suppression threshold
  max_objects: 100          # Maximum objects per frame
  strategies:               # Detection strategies to use
    - name: "yolo"
      model_path: "/path/to/model.engine"
      labels_path: "/path/to/labels.txt"
      enabled: true
      config:
        input_shape: [3, 608, 608]
        output_layers: ["output"]
```

### Alert Configuration

```yaml
alerts:
  enabled: true
  throttle_seconds: 60       # Throttle duplicate alerts
  burst_threshold: 3         # Max alerts in burst period
  max_alerts_per_minute: 10  # Rate limit
  handlers:
    - name: "console"
      enabled: true
    - name: "file"
      enabled: true
      config:
        log_file: "/var/log/pyds-alerts.log"
        max_size_mb: 100
    - name: "webhook"
      enabled: false
      config:
        url: "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
        timeout: 30
```

## 🐛 Troubleshooting

### Common Issues

#### 1. DeepStream Installation Issues

**Error**: `ImportError: No module named 'pyds'`

**Solution**:
```bash
# For DeepStream 5.x
cd /opt/nvidia/deepstream/deepstream-5.1/sources/deepstream_python_apps/bindings
python3 setup.py build_ext --inplace
pip3 install ./

# For DeepStream 6.x+
# pyds is replaced with gi.repository bindings
pip3 install pygobject
```

#### 2. GStreamer Plugin Not Found

**Error**: `gst-plugin-scanner: could not find plugin`

**Solution**:
```bash
# Ubuntu
sudo apt install gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly

# Set plugin path
export GST_PLUGIN_PATH=/opt/nvidia/deepstream/deepstream/lib/gst-plugins/
```

#### 3. CUDA Out of Memory

**Error**: `CUDA out of memory`

**Solutions**:
- Reduce batch size in configuration
- Lower processing resolution
- Enable memory optimization:

```yaml
pipeline:
  batch_size: 2              # Reduce from 4
  width: 1280               # Reduce from 1920
  height: 720               # Reduce from 1080
  memory_type: "unified"     # Use unified memory
```

#### 4. Low FPS Performance

**Diagnosis**:
```bash
# Check GPU utilization
nvidia-smi

# Profile application
uv run pyds-benchmark --profile --sources 1 --duration 60
```

**Solutions**:
- Enable GPU acceleration
- Optimize detection models
- Adjust pipeline configuration:

```yaml
pipeline:
  batch_size: 4              # Increase batching
  buffer_pool_size: 20       # Increase buffer pool
  enable_gpu_inference: true # Enable GPU inference
```

#### 5. Network Stream Issues

**Error**: `Could not connect to RTSP source`

**Solutions**:
```bash
# Test RTSP connection
gst-launch-1.0 rtspsrc location=rtsp://your.stream.url ! fakesink

# Check firewall and network connectivity
ping your.stream.ip
telnet your.stream.ip 554
```

### Debug Mode

Enable debug logging:
```bash
# Set debug environment variables
export GST_DEBUG=3
export PYDS_DEBUG=1

# Run with verbose logging
uv run pyds-app start --config configs/debug.yaml --verbose
```

### Performance Optimization

#### GPU Memory Optimization

```yaml
pipeline:
  memory_type: "device"      # Use device memory for better performance
  buffer_pool_size: 15       # Optimize buffer pool size
  batch_timeout_ms: 40       # Reduce batch timeout

detection:
  tensor_rt_precision: "fp16" # Use FP16 for faster inference
  max_batch_size: 8          # Optimize batch size for your GPU
```

#### CPU Optimization

```yaml
monitoring:
  cpu_affinity: [0, 1, 2, 3] # Pin to specific CPU cores
  thread_pool_size: 4        # Optimize thread pool
  
pipeline:
  num_decode_surfaces: 16    # Increase decode surfaces
  processing_width: 1280     # Balance quality vs performance
  processing_height: 720
```

## 📊 Monitoring and Metrics

### Built-in Metrics

The system provides comprehensive monitoring:

- **Pipeline Metrics**: FPS, latency, memory usage
- **Detection Metrics**: Objects detected, confidence distribution
- **Alert Metrics**: Alerts sent, throttled, failed
- **System Metrics**: CPU usage, GPU utilization, memory consumption

### Prometheus Integration

Enable Prometheus metrics:
```yaml
monitoring:
  prometheus:
    enabled: true
    port: 8000
    path: "/metrics"
```

### Health Checks

```bash
# Check system health
curl http://localhost:8000/health

# Detailed health report
uv run pyds-app health-check --detailed
```

## 🧪 Testing

### Run Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test categories
uv run pytest -m "not slow"           # Skip slow tests
uv run pytest -m "unit"               # Run only unit tests
uv run pytest -m "integration"        # Run integration tests
```

### Test Categories

- **Unit Tests**: Fast, isolated component tests
- **Integration Tests**: Multi-component interaction tests
- **Performance Tests**: Load and performance validation
- **GPU Tests**: Tests requiring CUDA/GPU (marked with `@pytest.mark.gpu`)

## 🚀 Deployment

### Docker Deployment

```dockerfile
# Dockerfile
FROM nvcr.io/nvidia/deepstream:7.0-devel

WORKDIR /app
COPY . .
RUN pip install -e .

EXPOSE 8000
CMD ["pyds-app", "start", "--config", "configs/production.yaml"]
```

### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pyds-app
spec:
  replicas: 2
  selector:
    matchLabels:
      app: pyds-app
  template:
    metadata:
      labels:
        app: pyds-app
    spec:
      containers:
      - name: pyds-app
        image: pyds-app:latest
        resources:
          requests:
            nvidia.com/gpu: 1
          limits:
            nvidia.com/gpu: 1
```

### Production Checklist

- [ ] Configure log rotation and retention
- [ ] Set up monitoring and alerting
- [ ] Configure automatic restart on failure
- [ ] Set resource limits and requests
- [ ] Enable security scanning and updates
- [ ] Configure backup and disaster recovery
- [ ] Set up performance monitoring
- [ ] Configure network security and firewalls

## 🤝 Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NVIDIA DeepStream team for the excellent SDK and documentation
- GStreamer community for the robust multimedia framework
- Contributors and users who provide feedback and improvements

## 📞 Support

- 📖 [Documentation](https://pyds-app.readthedocs.io/)
- 🐛 [Issues](https://github.com/deepstream-analytics/pyds-app/issues)
- 💬 [Discussions](https://github.com/deepstream-analytics/pyds-app/discussions)
- 📧 Email: support@deepstream-analytics.com

---

**Made with ❤️ for the computer vision and video analytics community**