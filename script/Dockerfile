# Multi-stage Dockerfile for Video Analytics Script
# Production-ready Docker image with DeepStream and NVIDIA GPU support

# Stage 1: Development base with DeepStream
FROM nvcr.io/nvidia/deepstream:6.3-devel as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    python3-gi \
    python3-gi-cairo \
    gir1.2-gstreamer-1.0 \
    gir1.2-gst-plugins-base-1.0 \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    libgirepository1.0-dev \
    libcairo2-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Python dependencies
FROM base as python-deps

# Install Python packages
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Stage 3: Production image
FROM python-deps as production

# Set working directory
WORKDIR /app

# Copy application files
COPY video_analytics_script.py /app/
COPY config.yaml /app/
COPY models/ /app/models/
COPY sample_videos/ /app/sample_videos/

# Create necessary directories
RUN mkdir -p /app/logs /app/data

# Set environment variables
ENV PYTHONPATH=/app
ENV GST_PLUGIN_PATH=/opt/nvidia/deepstream/deepstream/lib/gst-plugins/
ENV LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream/lib/
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose API port
EXPOSE 8080

# Run as non-root user for security
RUN useradd -m -u 1000 videoanalytics
RUN chown -R videoanalytics:videoanalytics /app
USER videoanalytics

# Default command
CMD ["python3", "video_analytics_script.py", "--config", "config.yaml"]