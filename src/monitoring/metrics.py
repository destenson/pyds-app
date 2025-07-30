"""
Comprehensive performance metrics collection and export system.

This module provides real-time FPS tracking, latency measurement, resource monitoring,
and metrics export in multiple formats for complete system observability.
"""

import asyncio
import time
import json
import csv
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable, Union, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, deque
import statistics
import gc
import sys
import platform

try:
    import GPUtil
    GPU_MONITORING_AVAILABLE = True
except ImportError:
    GPU_MONITORING_AVAILABLE = False

try:
    import nvidia_ml_py3 as nvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

from ..config import AppConfig
from ..utils.errors import MonitoringError, handle_error
from ..utils.logging import get_logger, performance_context
from ..utils.async_utils import get_task_manager, PeriodicTaskRunner


class MetricType(Enum):
    """Types of metrics collected."""
    COUNTER = "counter"          # Cumulative value that increases
    GAUGE = "gauge"             # Current value that can go up/down
    HISTOGRAM = "histogram"      # Distribution of values
    TIMER = "timer"             # Duration measurements
    RATE = "rate"               # Rate of change over time


class ExportFormat(Enum):
    """Supported export formats."""
    JSON = "json"
    CSV = "csv"
    PROMETHEUS = "prometheus"
    INFLUXDB = "influxdb"
    GRAFANA = "grafana"


@dataclass
class MetricPoint:
    """Individual metric data point."""
    name: str
    value: Union[int, float]
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp,
            'labels': self.labels,
            'metadata': self.metadata
        }


@dataclass
class MetricSeries:
    """Time series of metric points."""
    name: str
    metric_type: MetricType
    description: str
    unit: str = ""
    labels: Dict[str, str] = field(default_factory=dict)
    points: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def add_point(self, value: Union[int, float], timestamp: Optional[float] = None, **labels):
        """Add a new data point."""
        if timestamp is None:
            timestamp = time.time()
        
        point_labels = {**self.labels, **labels}
        point = MetricPoint(self.name, value, timestamp, point_labels)
        self.points.append(point)
    
    def get_current_value(self) -> Optional[Union[int, float]]:
        """Get the most recent value."""
        return self.points[-1].value if self.points else None
    
    def get_average(self, window_seconds: float = 60.0) -> Optional[float]:
        """Get average value over time window."""
        cutoff_time = time.time() - window_seconds
        values = [p.value for p in self.points if p.timestamp >= cutoff_time]
        return statistics.mean(values) if values else None
    
    def get_rate(self, window_seconds: float = 60.0) -> Optional[float]:
        """Get rate of change over time window."""
        cutoff_time = time.time() - window_seconds
        recent_points = [p for p in self.points if p.timestamp >= cutoff_time]
        
        if len(recent_points) < 2:
            return None
        
        # Calculate rate as change per second
        oldest = recent_points[0]
        newest = recent_points[-1]
        
        time_diff = newest.timestamp - oldest.timestamp
        value_diff = newest.value - oldest.value
        
        return value_diff / time_diff if time_diff > 0 else None


@dataclass
class SystemResourceMetrics:
    """System resource usage metrics."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_available_mb: float = 0.0
    disk_used_percent: float = 0.0
    disk_free_gb: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    process_count: int = 0
    load_average: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    uptime_seconds: float = 0.0


@dataclass
class GPUResourceMetrics:
    """GPU resource usage metrics."""
    gpu_id: int
    gpu_name: str = ""
    utilization_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    memory_percent: float = 0.0
    temperature_c: float = 0.0
    power_draw_w: float = 0.0
    power_limit_w: float = 0.0


@dataclass
class PipelineMetrics:
    """Pipeline-specific performance metrics."""
    pipeline_id: str
    source_count: int = 0
    fps_input: float = 0.0
    fps_processing: float = 0.0
    fps_output: float = 0.0
    latency_ms: float = 0.0
    queue_size: int = 0
    dropped_frames: int = 0
    error_count: int = 0
    uptime_seconds: float = 0.0


@dataclass
class DetectionMetrics:
    """Detection engine performance metrics."""
    total_detections: int = 0
    detections_per_second: float = 0.0
    average_confidence: float = 0.0
    inference_time_ms: float = 0.0
    post_processing_time_ms: float = 0.0
    strategy_count: int = 0
    active_strategies: List[str] = field(default_factory=list)


@dataclass
class AlertMetrics:
    """Alert system performance metrics."""
    total_alerts: int = 0
    alerts_per_second: float = 0.0
    throttled_alerts: int = 0
    delivered_alerts: int = 0
    failed_alerts: int = 0
    queue_size: int = 0
    average_delivery_time_ms: float = 0.0
    handler_count: int = 0


class ResourceMonitor:
    """Monitors system and GPU resources."""
    
    def __init__(self):
        """Initialize resource monitor."""
        self.logger = get_logger(__name__)
        self._gpu_initialized = False
        self._nvml_initialized = False
        
        # Initialize GPU monitoring
        self._init_gpu_monitoring()
        
        # Get system info
        self._system_info = self._get_system_info()
    
    def _init_gpu_monitoring(self):
        """Initialize GPU monitoring libraries."""
        if NVML_AVAILABLE:
            try:
                nvml.nvmlInit()
                self._nvml_initialized = True
                self.logger.info("NVML initialized for GPU monitoring")
            except Exception as e:
                self.logger.warning(f"Failed to initialize NVML: {e}")
        
        if GPU_MONITORING_AVAILABLE:
            try:
                GPUtil.getGPUs()  # Test access
                self._gpu_initialized = True
                self.logger.info("GPUtil initialized for GPU monitoring")
            except Exception as e:
                self.logger.warning(f"Failed to initialize GPUtil: {e}")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        try:
            return {
                'platform': platform.system(),
                'platform_release': platform.release(),
                'platform_version': platform.version(),
                'architecture': platform.machine(),
                'processor': platform.processor(),
                'python_version': sys.version,
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3)
            }
        except Exception as e:
            self.logger.error(f"Error getting system info: {e}")
            return {}
    
    def get_system_metrics(self) -> SystemResourceMetrics:
        """Get current system resource metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / (1024**2)
            memory_available_mb = memory.available / (1024**2)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_used_percent = (disk.used / disk.total) * 100
            disk_free_gb = disk.free / (1024**3)
            
            # Network metrics
            network = psutil.net_io_counters()
            network_bytes_sent = network.bytes_sent
            network_bytes_recv = network.bytes_recv
            
            # Process metrics
            process_count = len(psutil.pids())
            
            # Load average (Unix-like systems)
            try:
                load_average = psutil.getloadavg()
            except (AttributeError, OSError):
                load_average = (0.0, 0.0, 0.0)
            
            # System uptime
            uptime_seconds = time.time() - psutil.boot_time()
            
            return SystemResourceMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_mb=memory_used_mb,
                memory_available_mb=memory_available_mb,
                disk_used_percent=disk_used_percent,
                disk_free_gb=disk_free_gb,
                network_bytes_sent=network_bytes_sent,
                network_bytes_recv=network_bytes_recv,
                process_count=process_count,
                load_average=load_average,
                uptime_seconds=uptime_seconds
            )
        
        except Exception as e:
            self.logger.error(f"Error getting system metrics: {e}")
            return SystemResourceMetrics()
    
    def get_gpu_metrics(self) -> List[GPUResourceMetrics]:
        """Get GPU resource metrics."""
        gpu_metrics = []
        
        try:
            if self._nvml_initialized:
                gpu_metrics.extend(self._get_nvml_metrics())
            elif self._gpu_initialized:
                gpu_metrics.extend(self._get_gputil_metrics())
        
        except Exception as e:
            self.logger.error(f"Error getting GPU metrics: {e}")
        
        return gpu_metrics
    
    def _get_nvml_metrics(self) -> List[GPUResourceMetrics]:
        """Get GPU metrics using NVML."""
        metrics = []
        
        try:
            device_count = nvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = nvml.nvmlDeviceGetHandleByIndex(i)
                
                # Basic info
                name = nvml.nvmlDeviceGetName(handle).decode('utf-8')
                
                # Memory info
                mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                memory_used_mb = mem_info.used / (1024**2)
                memory_total_mb = mem_info.total / (1024**2)
                memory_percent = (mem_info.used / mem_info.total) * 100
                
                # Utilization
                util = nvml.nvmlDeviceGetUtilizationRates(handle)
                utilization_percent = util.gpu
                
                # Temperature
                try:
                    temperature_c = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                except:
                    temperature_c = 0.0
                
                # Power
                try:
                    power_draw_w = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    power_limit_w = nvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[1] / 1000.0
                except:
                    power_draw_w = 0.0
                    power_limit_w = 0.0
                
                metrics.append(GPUResourceMetrics(
                    gpu_id=i,
                    gpu_name=name,
                    utilization_percent=utilization_percent,
                    memory_used_mb=memory_used_mb,
                    memory_total_mb=memory_total_mb,
                    memory_percent=memory_percent,
                    temperature_c=temperature_c,
                    power_draw_w=power_draw_w,
                    power_limit_w=power_limit_w
                ))
        
        except Exception as e:
            self.logger.error(f"Error getting NVML metrics: {e}")
        
        return metrics
    
    def _get_gputil_metrics(self) -> List[GPUResourceMetrics]:
        """Get GPU metrics using GPUtil."""
        metrics = []
        
        try:
            gpus = GPUtil.getGPUs()
            
            for gpu in gpus:
                metrics.append(GPUResourceMetrics(
                    gpu_id=gpu.id,
                    gpu_name=gpu.name,
                    utilization_percent=gpu.load * 100,
                    memory_used_mb=gpu.memoryUsed,
                    memory_total_mb=gpu.memoryTotal,
                    memory_percent=gpu.memoryUtil * 100,
                    temperature_c=gpu.temperature,
                    power_draw_w=0.0,  # Not available in GPUtil
                    power_limit_w=0.0
                ))
        
        except Exception as e:
            self.logger.error(f"Error getting GPUtil metrics: {e}")
        
        return metrics
    
    def get_process_metrics(self, pid: Optional[int] = None) -> Dict[str, Any]:
        """Get metrics for specific process."""
        try:
            if pid is None:
                pid = os.getpid()
            
            process = psutil.Process(pid)
            
            return {
                'pid': pid,
                'name': process.name(),
                'status': process.status(),
                'cpu_percent': process.cpu_percent(),
                'memory_percent': process.memory_percent(),
                'memory_info': process.memory_info()._asdict(),
                'num_threads': process.num_threads(),
                'create_time': process.create_time(),
                'cmdline': ' '.join(process.cmdline()) if process.cmdline() else ''
            }
        
        except Exception as e:
            self.logger.error(f"Error getting process metrics: {e}")
            return {}


class FPSTracker:
    """Tracks frames per second for various components."""
    
    def __init__(self, name: str, window_size: int = 100):
        """
        Initialize FPS tracker.
        
        Args:
            name: Tracker name
            window_size: Number of samples to keep for calculation
        """
        self.name = name
        self.window_size = window_size
        self._frame_times: deque = deque(maxlen=window_size)
        self._frame_count = 0
        self._start_time = time.time()
        self._last_update = self._start_time
        self._lock = threading.Lock()
    
    def record_frame(self, timestamp: Optional[float] = None):
        """Record a processed frame."""
        if timestamp is None:
            timestamp = time.time()
        
        with self._lock:
            self._frame_times.append(timestamp)
            self._frame_count += 1
            self._last_update = timestamp
    
    def get_fps(self, window_seconds: float = 10.0) -> float:
        """Calculate FPS over the specified time window."""
        with self._lock:
            if not self._frame_times:
                return 0.0
            
            current_time = time.time()
            cutoff_time = current_time - window_seconds
            
            # Count frames in the time window
            frame_count = sum(1 for t in self._frame_times if t >= cutoff_time)
            
            return frame_count / window_seconds if window_seconds > 0 else 0.0
    
    def get_average_fps(self) -> float:
        """Get average FPS since tracker started."""
        with self._lock:
            elapsed = self._last_update - self._start_time
            return self._frame_count / elapsed if elapsed > 0 else 0.0
    
    def get_instantaneous_fps(self) -> float:
        """Get instantaneous FPS based on recent frames."""
        with self._lock:
            if len(self._frame_times) < 2:
                return 0.0
            
            # Use last few frames to calculate instantaneous FPS
            recent_frames = list(self._frame_times)[-min(10, len(self._frame_times)):]
            if len(recent_frames) < 2:
                return 0.0
            
            time_span = recent_frames[-1] - recent_frames[0]
            return (len(recent_frames) - 1) / time_span if time_span > 0 else 0.0
    
    def reset(self):
        """Reset the tracker."""
        with self._lock:
            self._frame_times.clear()
            self._frame_count = 0
            self._start_time = time.time()
            self._last_update = self._start_time


class LatencyTracker:
    """Tracks end-to-end latency measurements."""
    
    def __init__(self, name: str, max_samples: int = 1000):
        """
        Initialize latency tracker.
        
        Args:
            name: Tracker name
            max_samples: Maximum number of samples to keep
        """
        self.name = name
        self._latencies: deque = deque(maxlen=max_samples)
        self._pending_operations: Dict[str, float] = {}
        self._lock = threading.Lock()
    
    def start_operation(self, operation_id: str, timestamp: Optional[float] = None) -> str:
        """Start tracking an operation."""
        if timestamp is None:
            timestamp = time.time()
        
        with self._lock:
            self._pending_operations[operation_id] = timestamp
        
        return operation_id
    
    def end_operation(self, operation_id: str, timestamp: Optional[float] = None) -> Optional[float]:
        """End tracking an operation and return latency."""
        if timestamp is None:
            timestamp = time.time()
        
        with self._lock:
            start_time = self._pending_operations.pop(operation_id, None)
            if start_time is None:
                return None
            
            latency = (timestamp - start_time) * 1000  # Convert to milliseconds
            self._latencies.append(latency)
            return latency
    
    def record_latency(self, latency_ms: float):
        """Directly record a latency measurement."""
        with self._lock:
            self._latencies.append(latency_ms)
    
    def get_average_latency(self) -> float:
        """Get average latency in milliseconds."""
        with self._lock:
            return statistics.mean(self._latencies) if self._latencies else 0.0
    
    def get_percentile_latency(self, percentile: float) -> float:
        """Get percentile latency in milliseconds."""
        with self._lock:
            if not self._latencies:
                return 0.0
            
            sorted_latencies = sorted(self._latencies)
            index = int((percentile / 100.0) * len(sorted_latencies))
            return sorted_latencies[min(index, len(sorted_latencies) - 1)]
    
    def get_latency_stats(self) -> Dict[str, float]:
        """Get comprehensive latency statistics."""
        with self._lock:
            if not self._latencies:
                return {
                    'count': 0,
                    'average': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'p50': 0.0,
                    'p95': 0.0,
                    'p99': 0.0,
                    'stddev': 0.0
                }
            
            latencies = list(self._latencies)
            return {
                'count': len(latencies),
                'average': statistics.mean(latencies),
                'min': min(latencies),
                'max': max(latencies),
                'p50': self.get_percentile_latency(50),
                'p95': self.get_percentile_latency(95),
                'p99': self.get_percentile_latency(99),
                'stddev': statistics.stdev(latencies) if len(latencies) > 1 else 0.0
            }


class MetricsExporter:
    """Exports metrics in various formats."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize metrics exporter."""
        self.config = config or {}
        self.logger = get_logger(__name__)
    
    async def export_json(self, metrics: Dict[str, Any], file_path: Path) -> bool:
        """Export metrics as JSON."""
        try:
            # Prepare metrics for JSON serialization
            serializable_metrics = self._prepare_for_json(metrics)
            
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write JSON file
            with open(file_path, 'w') as f:
                json.dump(serializable_metrics, f, indent=2, default=str)
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error exporting JSON metrics: {e}")
            return False
    
    async def export_csv(self, metrics: Dict[str, Any], file_path: Path) -> bool:
        """Export metrics as CSV."""
        try:
            # Flatten metrics for CSV format
            flattened_metrics = self._flatten_metrics(metrics)
            
            if not flattened_metrics:
                return False
            
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write CSV file
            with open(file_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=flattened_metrics[0].keys())
                writer.writeheader()
                writer.writerows(flattened_metrics)
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error exporting CSV metrics: {e}")
            return False
    
    async def export_prometheus(self, metrics: Dict[str, Any], file_path: Path) -> bool:
        """Export metrics in Prometheus format."""
        try:
            prometheus_text = self._format_prometheus(metrics)
            
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write Prometheus file
            with open(file_path, 'w') as f:
                f.write(prometheus_text)
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error exporting Prometheus metrics: {e}")
            return False
    
    def _prepare_for_json(self, data: Any) -> Any:
        """Prepare data for JSON serialization."""
        if isinstance(data, dict):
            return {key: self._prepare_for_json(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._prepare_for_json(item) for item in data]
        elif isinstance(data, deque):
            return [self._prepare_for_json(item) for item in data]
        elif hasattr(data, '__dict__'):
            return self._prepare_for_json(asdict(data) if hasattr(data, '__dataclass_fields__') else data.__dict__)
        elif isinstance(data, (datetime, time.struct_time)):
            return str(data)
        else:
            return data
    
    def _flatten_metrics(self, metrics: Dict[str, Any], prefix: str = "") -> List[Dict[str, Any]]:
        """Flatten nested metrics for CSV export."""
        flattened = []
        
        def _flatten_recursive(obj: Any, current_prefix: str):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_prefix = f"{current_prefix}_{key}" if current_prefix else key
                    _flatten_recursive(value, new_prefix)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    new_prefix = f"{current_prefix}_{i}" if current_prefix else f"item_{i}"
                    _flatten_recursive(item, new_prefix)
            else:
                flattened.append({
                    'metric': current_prefix,
                    'value': str(obj),
                    'timestamp': time.time()
                })
        
        _flatten_recursive(metrics, prefix)
        return flattened
    
    def _format_prometheus(self, metrics: Dict[str, Any]) -> str:
        """Format metrics in Prometheus exposition format."""
        lines = []
        
        def _format_recursive(obj: Any, prefix: str = ""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_prefix = f"{prefix}_{key}" if prefix else key
                    _format_recursive(value, new_prefix)
            elif isinstance(obj, (int, float)):
                # Clean up metric name for Prometheus
                metric_name = prefix.lower().replace('-', '_').replace(' ', '_')
                lines.append(f"# TYPE {metric_name} gauge")
                lines.append(f"{metric_name} {obj}")
            elif isinstance(obj, list) and obj:
                # Handle list of numeric values
                if all(isinstance(x, (int, float)) for x in obj):
                    metric_name = prefix.lower().replace('-', '_').replace(' ', '_')
                    lines.append(f"# TYPE {metric_name} gauge")
                    for i, value in enumerate(obj):
                        lines.append(f"{metric_name}{{index=\"{i}\"}} {value}")
        
        _format_recursive(metrics)
        return '\n'.join(lines)


class MetricsCollector:
    """
    Comprehensive metrics collection system.
    
    Provides real-time FPS tracking, latency measurement, resource monitoring,
    and metrics export in multiple formats for complete system observability.
    """
    
    def __init__(self, config: AppConfig):
        """
        Initialize metrics collector.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Core components
        self._resource_monitor = ResourceMonitor()
        self._exporter = MetricsExporter()
        
        # Metric series storage
        self._metrics: Dict[str, MetricSeries] = {}
        self._metrics_lock = threading.RLock()
        
        # FPS trackers
        self._fps_trackers: Dict[str, FPSTracker] = {}
        
        # Latency trackers
        self._latency_trackers: Dict[str, LatencyTracker] = {}
        
        # Performance counters
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, Union[int, float]] = {}
        self._timers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Collection control
        self._collecting = False
        self._collection_interval = 1.0  # seconds
        self._periodic_runner = PeriodicTaskRunner()
        
        # Export settings
        self._export_interval = 60.0  # seconds
        self._export_formats = ['json']
        self._export_path = Path("data/metrics")
        
        # Historical data
        self._history_retention_hours = 24
        self._history: deque = deque(maxlen=int(3600 * 24 / self._collection_interval))  # 24 hours
        
        self.logger.info("MetricsCollector initialized")
    
    async def start(self) -> bool:
        """
        Start metrics collection.
        
        Returns:
            True if started successfully
        """
        try:
            self.logger.info("Starting MetricsCollector...")
            
            # Initialize default metrics
            self._initialize_default_metrics()
            
            # Start periodic collection
            await self._start_periodic_collection()
            
            self._collecting = True
            
            self.logger.info("MetricsCollector started successfully")
            return True
        
        except Exception as e:
            error = handle_error(e, context={'component': 'metrics_collector'})
            self.logger.error(f"Failed to start MetricsCollector: {error}")
            return False
    
    async def stop(self) -> bool:
        """
        Stop metrics collection.
        
        Returns:
            True if stopped successfully
        """
        try:
            self.logger.info("Stopping MetricsCollector...")
            
            self._collecting = False
            
            # Stop periodic tasks
            await self._periodic_runner.stop_all_tasks()
            
            # Export final metrics
            await self._export_current_metrics()
            
            self.logger.info("MetricsCollector stopped successfully")
            return True
        
        except Exception as e:
            self.logger.error(f"Error stopping MetricsCollector: {e}")
            return False
    
    def _initialize_default_metrics(self):
        """Initialize default metric series."""
        default_metrics = [
            # System metrics
            ('system_cpu_percent', MetricType.GAUGE, 'CPU utilization percentage', '%'),
            ('system_memory_percent', MetricType.GAUGE, 'Memory utilization percentage', '%'),
            ('system_disk_percent', MetricType.GAUGE, 'Disk utilization percentage', '%'),
            ('system_network_bytes_sent', MetricType.COUNTER, 'Network bytes sent', 'bytes'),
            ('system_network_bytes_recv', MetricType.COUNTER, 'Network bytes received', 'bytes'),
            
            # GPU metrics
            ('gpu_utilization_percent', MetricType.GAUGE, 'GPU utilization percentage', '%'),
            ('gpu_memory_percent', MetricType.GAUGE, 'GPU memory utilization percentage', '%'),
            ('gpu_temperature_c', MetricType.GAUGE, 'GPU temperature', '°C'),
            
            # Pipeline metrics
            ('pipeline_fps_input', MetricType.GAUGE, 'Pipeline input FPS', 'fps'),
            ('pipeline_fps_processing', MetricType.GAUGE, 'Pipeline processing FPS', 'fps'),
            ('pipeline_fps_output', MetricType.GAUGE, 'Pipeline output FPS', 'fps'),
            ('pipeline_latency_ms', MetricType.GAUGE, 'Pipeline end-to-end latency', 'ms'),
            ('pipeline_queue_size', MetricType.GAUGE, 'Pipeline queue size', 'frames'),
            ('pipeline_dropped_frames', MetricType.COUNTER, 'Dropped frames count', 'frames'),
            
            # Detection metrics
            ('detection_total_count', MetricType.COUNTER, 'Total detections', 'detections'),
            ('detection_rate', MetricType.RATE, 'Detections per second', 'det/s'),
            ('detection_inference_time_ms', MetricType.GAUGE, 'Inference time', 'ms'),
            ('detection_average_confidence', MetricType.GAUGE, 'Average detection confidence', ''),
            
            # Alert metrics  
            ('alert_total_count', MetricType.COUNTER, 'Total alerts', 'alerts'),
            ('alert_rate', MetricType.RATE, 'Alerts per second', 'alerts/s'),
            ('alert_delivery_time_ms', MetricType.GAUGE, 'Alert delivery time', 'ms'),
            ('alert_queue_size', MetricType.GAUGE, 'Alert queue size', 'alerts')
        ]
        
        with self._metrics_lock:
            for name, metric_type, description, unit in default_metrics:
                self._metrics[name] = MetricSeries(
                    name=name,
                    metric_type=metric_type,
                    description=description,
                    unit=unit
                )
    
    async def _start_periodic_collection(self):
        """Start periodic metrics collection tasks."""
        # Main collection task
        await self._periodic_runner.start_periodic_task(
            "collect_metrics",
            self._collect_all_metrics,
            interval=self._collection_interval
        )
        
        # Export task
        await self._periodic_runner.start_periodic_task(
            "export_metrics",
            self._export_current_metrics,
            interval=self._export_interval
        )
        
        # Cleanup task
        await self._periodic_runner.start_periodic_task(
            "cleanup_metrics",
            self._cleanup_old_metrics,
            interval=3600.0  # Every hour
        )
    
    async def _collect_all_metrics(self):
        """Collect all system metrics."""
        try:
            timestamp = time.time()
            
            # Collect system metrics
            system_metrics = self._resource_monitor.get_system_metrics()
            self._record_system_metrics(system_metrics, timestamp)
            
            # Collect GPU metrics
            gpu_metrics = self._resource_monitor.get_gpu_metrics()
            self._record_gpu_metrics(gpu_metrics, timestamp)
            
            # Calculate derived metrics
            self._calculate_derived_metrics(timestamp)
            
            # Store snapshot in history
            snapshot = self._create_metrics_snapshot(timestamp)
            self._history.append(snapshot)
            
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")
    
    def _record_system_metrics(self, metrics: SystemResourceMetrics, timestamp: float):
        """Record system resource metrics."""
        with self._metrics_lock:
            self._metrics['system_cpu_percent'].add_point(metrics.cpu_percent, timestamp)
            self._metrics['system_memory_percent'].add_point(metrics.memory_percent, timestamp)
            self._metrics['system_disk_percent'].add_point(metrics.disk_used_percent, timestamp)
            self._metrics['system_network_bytes_sent'].add_point(metrics.network_bytes_sent, timestamp)
            self._metrics['system_network_bytes_recv'].add_point(metrics.network_bytes_recv, timestamp)
    
    def _record_gpu_metrics(self, gpu_metrics: List[GPUResourceMetrics], timestamp: float):
        """Record GPU resource metrics."""
        with self._metrics_lock:
            for gpu in gpu_metrics:
                labels = {'gpu_id': str(gpu.gpu_id), 'gpu_name': gpu.gpu_name}
                
                # Create GPU-specific metric series if they don't exist
                for metric_name in ['gpu_utilization_percent', 'gpu_memory_percent', 'gpu_temperature_c']:
                    gpu_metric_name = f"{metric_name}_gpu_{gpu.gpu_id}"
                    if gpu_metric_name not in self._metrics:
                        base_metric = self._metrics[metric_name]
                        self._metrics[gpu_metric_name] = MetricSeries(
                            name=gpu_metric_name,
                            metric_type=base_metric.metric_type,
                            description=f"{base_metric.description} for GPU {gpu.gpu_id}",
                            unit=base_metric.unit,
                            labels=labels
                        )
                
                # Record values
                self._metrics[f'gpu_utilization_percent_gpu_{gpu.gpu_id}'].add_point(
                    gpu.utilization_percent, timestamp
                )
                self._metrics[f'gpu_memory_percent_gpu_{gpu.gpu_id}'].add_point(
                    gpu.memory_percent, timestamp
                )
                self._metrics[f'gpu_temperature_c_gpu_{gpu.gpu_id}'].add_point(
                    gpu.temperature_c, timestamp
                )
    
    def _calculate_derived_metrics(self, timestamp: float):
        """Calculate derived metrics from collected data."""
        with self._metrics_lock:
            # Calculate rates for counters
            for name, metric in self._metrics.items():
                if metric.metric_type == MetricType.COUNTER:
                    rate = metric.get_rate(window_seconds=60.0)
                    if rate is not None:
                        rate_metric_name = f"{name}_rate"
                        if rate_metric_name not in self._metrics:
                            self._metrics[rate_metric_name] = MetricSeries(
                                name=rate_metric_name,
                                metric_type=MetricType.RATE,
                                description=f"Rate of {metric.description}",
                                unit=f"{metric.unit}/s"
                            )
                        self._metrics[rate_metric_name].add_point(rate, timestamp)
    
    def _create_metrics_snapshot(self, timestamp: float) -> Dict[str, Any]:
        """Create snapshot of current metrics."""
        snapshot = {
            'timestamp': timestamp,
            'system': asdict(self._resource_monitor.get_system_metrics()),
            'gpu': [asdict(gpu) for gpu in self._resource_monitor.get_gpu_metrics()],
            'counters': dict(self._counters),
            'gauges': dict(self._gauges),
            'fps_trackers': {
                name: {
                    'current_fps': tracker.get_fps(),
                    'average_fps': tracker.get_average_fps(),
                    'instantaneous_fps': tracker.get_instantaneous_fps()
                } for name, tracker in self._fps_trackers.items()
            },
            'latency_trackers': {
                name: tracker.get_latency_stats()
                for name, tracker in self._latency_trackers.items()
            }
        }
        
        return snapshot
    
    async def _export_current_metrics(self):
        """Export current metrics to configured formats."""
        try:
            timestamp = datetime.now()
            
            # Create comprehensive metrics report
            metrics_report = {
                'timestamp': timestamp.isoformat(),
                'collection_info': {
                    'collector_uptime_seconds': time.time() - getattr(self, '_start_time', time.time()),
                    'metrics_count': len(self._metrics),
                    'history_count': len(self._history),
                    'collection_interval': self._collection_interval
                },
                'current_snapshot': self._create_metrics_snapshot(time.time()),
                'historical_averages': self._calculate_historical_averages()
            }
            
            # Export in configured formats
            base_filename = f"metrics_{timestamp.strftime('%Y%m%d_%H%M%S')}"
            
            for format_type in self._export_formats:
                if format_type == 'json':
                    file_path = self._export_path / f"{base_filename}.json"
                    await self._exporter.export_json(metrics_report, file_path)
                
                elif format_type == 'csv':
                    file_path = self._export_path / f"{base_filename}.csv"
                    await self._exporter.export_csv(metrics_report, file_path)
                
                elif format_type == 'prometheus':
                    file_path = self._export_path / f"{base_filename}.prom"
                    await self._exporter.export_prometheus(metrics_report['current_snapshot'], file_path)
            
            self.logger.debug(f"Exported metrics in {len(self._export_formats)} formats")
        
        except Exception as e:
            self.logger.error(f"Error exporting metrics: {e}")
    
    def _calculate_historical_averages(self) -> Dict[str, Any]:
        """Calculate historical averages from stored snapshots."""
        if not self._history:
            return {}
        
        averages = {}
        
        try:
            # Calculate averages for key metrics
            cpu_values = []
            memory_values = []
            gpu_util_values = []
            
            for snapshot in self._history:
                if 'system' in snapshot:
                    cpu_values.append(snapshot['system'].get('cpu_percent', 0))
                    memory_values.append(snapshot['system'].get('memory_percent', 0))
                
                if 'gpu' in snapshot and snapshot['gpu']:
                    gpu_util_values.extend([gpu.get('utilization_percent', 0) for gpu in snapshot['gpu']])
            
            if cpu_values:
                averages['avg_cpu_percent_24h'] = statistics.mean(cpu_values)
            if memory_values:
                averages['avg_memory_percent_24h'] = statistics.mean(memory_values)
            if gpu_util_values:
                averages['avg_gpu_utilization_24h'] = statistics.mean(gpu_util_values)
        
        except Exception as e:
            self.logger.error(f"Error calculating historical averages: {e}")
        
        return averages
    
    async def _cleanup_old_metrics(self):
        """Clean up old metric data points."""
        try:
            cutoff_time = time.time() - (self._history_retention_hours * 3600)
            
            with self._metrics_lock:
                for metric in self._metrics.values():
                    # Remove old points (deque maxlen handles this automatically)
                    # but we can clean up very old points manually if needed
                    while metric.points and metric.points[0].timestamp < cutoff_time:
                        metric.points.popleft()
            
            self.logger.debug("Cleaned up old metric data points")
        
        except Exception as e:
            self.logger.error(f"Error cleaning up metrics: {e}")
    
    # Public API methods
    
    def get_fps_tracker(self, name: str) -> FPSTracker:
        """Get or create FPS tracker."""
        if name not in self._fps_trackers:
            self._fps_trackers[name] = FPSTracker(name)
        return self._fps_trackers[name]
    
    def get_latency_tracker(self, name: str) -> LatencyTracker:
        """Get or create latency tracker."""
        if name not in self._latency_trackers:
            self._latency_trackers[name] = LatencyTracker(name)
        return self._latency_trackers[name]
    
    def increment_counter(self, name: str, value: int = 1, **labels):
        """Increment a counter metric."""
        self._counters[name] += value
        
        with self._metrics_lock:
            if name in self._metrics:
                self._metrics[name].add_point(self._counters[name], **labels)
    
    def set_gauge(self, name: str, value: Union[int, float], **labels):
        """Set a gauge metric value."""
        self._gauges[name] = value
        
        with self._metrics_lock:
            if name in self._metrics:
                self._metrics[name].add_point(value, **labels)
    
    def record_timer(self, name: str, duration_ms: float, **labels):
        """Record a timing measurement."""
        self._timers[name].append(duration_ms)
        
        with self._metrics_lock:
            if name in self._metrics:
                self._metrics[name].add_point(duration_ms, **labels)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        return self._create_metrics_snapshot(time.time())
    
    def get_metric_history(self, metric_name: str, hours: int = 1) -> List[MetricPoint]:
        """Get historical data for a specific metric."""
        with self._metrics_lock:
            if metric_name not in self._metrics:
                return []
            
            cutoff_time = time.time() - (hours * 3600)
            return [p for p in self._metrics[metric_name].points if p.timestamp >= cutoff_time]
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get system performance summary."""
        system_metrics = self._resource_monitor.get_system_metrics()
        gpu_metrics = self._resource_monitor.get_gpu_metrics()
        
        return {
            'system': asdict(system_metrics),
            'gpu': [asdict(gpu) for gpu in gpu_metrics],
            'fps_summary': {
                name: {
                    'current': tracker.get_fps(),
                    'average': tracker.get_average_fps()
                } for name, tracker in self._fps_trackers.items()
            },
            'latency_summary': {
                name: tracker.get_latency_stats()
                for name, tracker in self._latency_trackers.items()
            },
            'collection_status': {
                'is_collecting': self._collecting,
                'metrics_count': len(self._metrics),
                'history_size': len(self._history)
            }
        }


# Global metrics collector instance
_global_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector(config: Optional[AppConfig] = None) -> Optional[MetricsCollector]:
    """Get global metrics collector instance."""
    global _global_metrics_collector
    if _global_metrics_collector is None and config is not None:
        _global_metrics_collector = MetricsCollector(config)
    return _global_metrics_collector