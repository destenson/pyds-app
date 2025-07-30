# Standalone Video Analytics Script

A production-ready Python script for real-time pattern detection in video streams using GStreamer, YOLO, and NVIDIA DeepStream.

## Features

- **Multi-source video processing**: RTSP, WebRTC, file, and webcam support
- **YOLO object detection**: GPU-accelerated inference with DeepStream
- **REST API control**: Dynamic source and configuration management
- **Automatic error recovery**: Handles network failures and pipeline errors
- **Production monitoring**: Structured JSON logging and performance metrics
- **Docker deployment**: Ready for containerized deployment with GPU support

## Quick Start

### Development Mode (without GPU/DeepStream)

```bash
# Install minimal dependencies for development
pip install fastapi uvicorn pyyaml

# Run in mock mode (no GPU required)
python video_analytics_script.py --config config.yaml

# Test pipeline creation
python video_analytics_script.py --test-pipeline
```

### Production Mode (with GPU/DeepStream)

```bash
# Requires NVIDIA GPU, CUDA, and DeepStream installed
docker-compose up

# Or run directly with DeepStream environment
python video_analytics.py --config config.yaml
```

## REST API Endpoints

The script provides a comprehensive REST API on port 8080:

- `GET /health` - Health check endpoint
- `GET /api/v1/status` - System status and statistics
- `GET /api/v1/sources` - List all video sources
- `POST /api/v1/sources` - Add a new video source
- `PUT /api/v1/sources/{id}` - Update source configuration
- `DELETE /api/v1/sources/{id}` - Remove a video source
- `GET /api/v1/config` - Get current configuration
- `PUT /api/v1/config/detection` - Update detection parameters
- `GET /api/v1/metrics` - Get performance metrics

### Example: Add a Video Source

```bash
curl -X POST http://localhost:8080/api/v1/sources \
  -H "Content-Type: application/json" \
  -d '{
    "id": "camera1",
    "type": "rtsp",
    "uri": "rtsp://192.168.1.100:554/stream",
    "enabled": true,
    "confidence_threshold": 0.6
  }'
```

## Configuration

The script uses YAML configuration with environment variable overrides:

```yaml
# config.yaml
application:
  name: "VideoAnalytics"
  version: "1.0.0"

pipeline:
  batch_size: 4
  max_sources: 16
  gpu_device_id: 0

detection:
  model_path: "models/yolo.engine"
  confidence_threshold: 0.5
  nms_threshold: 0.4
  max_objects: 50
```

### Environment Variables

- `VA_DEBUG` - Enable debug mode
- `VA_API_PORT` - API server port (default: 8080)
- `VA_MODEL_PATH` - Path to YOLO model
- `VA_CONFIDENCE_THRESHOLD` - Detection confidence threshold
- `VA_LOG_LEVEL` - Logging level (DEBUG, INFO, WARNING, ERROR)

## Architecture

The script is organized into modular components:

- **VideoAnalyticsEngine** - Main application orchestrator
- **GStreamerPipeline** - Media pipeline management
- **YOLODetector** - Object detection engine
- **SourceManager** - Dynamic video source management
- **ConfigManager** - Configuration and environment handling
- **Logger** - Structured JSON logging

## Performance Optimization

- Batch processing for GPU efficiency
- Zero-copy NVMM memory usage
- Configurable frame skipping
- Dynamic source management without pipeline restart
- Automatic recovery from transient failures

## Monitoring and Debugging

### Structured Logging

All logs are output in JSON format for easy parsing:

```json
{
  "timestamp": "2024-01-15T10:30:45.123456",
  "level": "INFO",
  "logger": "VideoAnalytics",
  "message": "Added source: camera1",
  "source_id": "camera1",
  "module": "source_manager"
}
```

### Performance Metrics

Access real-time metrics via the API:

```bash
curl http://localhost:8080/api/v1/metrics
```

## Production Deployment

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f

# Scale to multiple instances
docker-compose up -d --scale video-analytics=3
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: video-analytics
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: video-analytics
        image: video-analytics:latest
        resources:
          limits:
            nvidia.com/gpu: 1
```

## Troubleshooting

### Common Issues

1. **GStreamer not found**: Install GStreamer development packages or run in Docker
2. **CUDA/DeepStream errors**: Ensure NVIDIA drivers and DeepStream are properly installed
3. **API not accessible**: Check firewall rules and ensure port 8080 is open
4. **Memory issues**: Reduce batch_size or max_sources in configuration

### Debug Mode

Run with debug logging for detailed troubleshooting:

```bash
python video_analytics_script.py --debug
```

## License

This is a standalone implementation following NVIDIA DeepStream best practices.
