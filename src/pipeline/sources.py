"""
Video source abstraction and management for multi-source video analytics.

This module provides comprehensive video source management including RTSP, WebRTC,
file, webcam, and test sources with health monitoring and dynamic management.
"""

import asyncio
import time
import re
from typing import Dict, List, Optional, Set, Callable, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from abc import ABC, abstractmethod
import threading
from urllib.parse import urlparse
import socket
import subprocess

from ..config import SourceConfig, SourceType
from ..utils.errors import SourceError, NetworkSourceError, handle_error, recovery_strategy
from ..utils.logging import get_logger, performance_context, log_pipeline_event
from ..utils.async_utils import get_task_manager, ThreadSafeAsyncQueue, PeriodicTaskRunner


class SourceStatus(Enum):
    """Video source status enumeration."""
    UNKNOWN = "unknown"
    AVAILABLE = "available"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    STREAMING = "streaming"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    DISABLED = "disabled"


class SourceHealth(Enum):
    """Video source health levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class SourceMetrics:
    """Metrics for video source monitoring."""
    source_id: str
    frames_received: int = 0
    frames_dropped: int = 0
    bytes_received: int = 0
    connection_attempts: int = 0
    successful_connections: int = 0
    last_frame_time: float = 0.0
    average_fps: float = 0.0
    current_fps: float = 0.0
    latency_ms: Optional[float] = None
    bitrate_kbps: Optional[float] = None
    resolution: Optional[Tuple[int, int]] = None
    last_error: Optional[str] = None
    error_count: int = 0
    uptime_seconds: float = 0.0
    reconnection_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'source_id': self.source_id,
            'frames_received': self.frames_received,
            'frames_dropped': self.frames_dropped,
            'bytes_received': self.bytes_received,
            'connection_attempts': self.connection_attempts,
            'successful_connections': self.successful_connections,
            'last_frame_time': self.last_frame_time,
            'average_fps': self.average_fps,
            'current_fps': self.current_fps,
            'latency_ms': self.latency_ms,
            'bitrate_kbps': self.bitrate_kbps,
            'resolution': self.resolution,
            'last_error': self.last_error,
            'error_count': self.error_count,
            'uptime_seconds': self.uptime_seconds,
            'reconnection_count': self.reconnection_count
        }


@dataclass
class SourceState:
    """Complete state information for a video source."""
    config: SourceConfig
    status: SourceStatus = SourceStatus.UNKNOWN
    health: SourceHealth = SourceHealth.HEALTHY
    metrics: SourceMetrics = field(init=False)
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    next_retry: Optional[float] = None
    retry_count: int = 0
    is_active: bool = False
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """Initialize metrics after creation."""
        self.metrics = SourceMetrics(source_id=self.config.id)
    
    def update_status(self, status: SourceStatus, error_message: Optional[str] = None):
        """Update source status and timestamp."""
        self.status = status
        self.last_updated = time.time()
        if error_message:
            self.error_message = error_message
            self.metrics.error_count += 1
            self.metrics.last_error = error_message


class VideoSource(ABC):
    """Abstract base class for video sources."""
    
    def __init__(self, config: SourceConfig):
        """
        Initialize video source.
        
        Args:
            config: Source configuration
        """
        self.config = config
        self.logger = get_logger(f"{__name__}.{config.type.value}")
        self.state = SourceState(config)
        self._callbacks: List[Callable] = []
        self._stop_event = asyncio.Event()
        
    @abstractmethod
    async def validate(self) -> bool:
        """
        Validate source availability and configuration.
        
        Returns:
            True if source is valid and available
        """
        pass
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to the source.
        
        Returns:
            True if connection successful
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """
        Disconnect from the source.
        
        Returns:
            True if disconnection successful
        """
        pass
    
    @abstractmethod
    async def get_capabilities(self) -> Dict[str, Any]:
        """
        Get source capabilities and format information.
        
        Returns:
            Dictionary of capabilities
        """
        pass
    
    def add_callback(self, callback: Callable):
        """Add callback for source events."""
        self._callbacks.append(callback)
    
    async def _notify_callbacks(self, event_type: str, **kwargs):
        """Notify all callbacks of an event."""
        event_data = {
            'source_id': self.config.id,
            'event_type': event_type,
            'timestamp': time.time(),
            **kwargs
        }
        
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event_data)
                else:
                    callback(event_data)
            except Exception as e:
                self.logger.error(f"Error in source callback: {e}")
    
    def update_metrics(self, **kwargs):
        """Update source metrics."""
        for key, value in kwargs.items():
            if hasattr(self.state.metrics, key):
                setattr(self.state.metrics, key, value)
        
        self.state.last_updated = time.time()
    
    async def start_monitoring(self):
        """Start monitoring the source."""
        self._stop_event.clear()
        await self._notify_callbacks('monitoring_started')
    
    async def stop_monitoring(self):
        """Stop monitoring the source."""
        self._stop_event.set()
        await self._notify_callbacks('monitoring_stopped')


class RTSPSource(VideoSource):
    """RTSP video source implementation."""
    
    def __init__(self, config: SourceConfig):
        super().__init__(config)
        self._connection_timeout = config.timeout or 30
        self._rtsp_session = None
    
    async def validate(self) -> bool:
        """Validate RTSP source configuration and connectivity."""
        try:
            # Validate URI format
            if not self.config.uri.startswith(('rtsp://', 'rtsps://')):
                raise SourceError("Invalid RTSP URI format", source_uri=self.config.uri)
            
            # Parse URI to check host and port
            parsed = urlparse(self.config.uri)
            if not parsed.hostname:
                raise SourceError("Invalid RTSP hostname", source_uri=self.config.uri)
            
            port = parsed.port or (554 if parsed.scheme == 'rtsp' else 322)
            
            # Test network connectivity
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5.0)
                result = sock.connect_ex((parsed.hostname, port))
                sock.close()
                
                if result != 0:
                    self.logger.warning(f"RTSP server {parsed.hostname}:{port} not reachable")
                    return False
            
            except Exception as e:
                self.logger.warning(f"Network connectivity test failed: {e}")
                return False
            
            self.state.update_status(SourceStatus.AVAILABLE)
            return True
        
        except Exception as e:
            error_msg = f"RTSP validation failed: {e}"
            self.state.update_status(SourceStatus.ERROR, error_msg)
            self.logger.error(error_msg)
            return False
    
    async def connect(self) -> bool:
        """Connect to RTSP source."""
        try:
            self.state.update_status(SourceStatus.CONNECTING)
            self.state.metrics.connection_attempts += 1
            
            # Use GStreamer's rtspsrc element for actual connection
            # This is a placeholder for the actual implementation
            await asyncio.sleep(0.1)  # Simulate connection time
            
            self.state.update_status(SourceStatus.CONNECTED)
            self.state.metrics.successful_connections += 1
            self.state.is_active = True
            
            await self._notify_callbacks('connected')
            self.logger.info(f"Connected to RTSP source {self.config.id}")
            return True
        
        except Exception as e:
            error_msg = f"RTSP connection failed: {e}"
            self.state.update_status(SourceStatus.ERROR, error_msg)
            await self._notify_callbacks('connection_failed', error=str(e))
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from RTSP source."""
        try:
            self.state.update_status(SourceStatus.DISCONNECTED)
            self.state.is_active = False
            
            await self._notify_callbacks('disconnected')
            self.logger.info(f"Disconnected from RTSP source {self.config.id}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error disconnecting RTSP source: {e}")
            return False
    
    async def get_capabilities(self) -> Dict[str, Any]:
        """Get RTSP source capabilities."""
        return {
            'type': 'rtsp',
            'protocols': ['rtsp', 'rtsps'],
            'supports_seek': False,
            'supports_pause': False,
            'live_source': True,
            'network_source': True,
            'supported_codecs': ['h264', 'h265', 'mjpeg'],
            'max_resolution': None,  # Unknown until connected
            'audio_support': True
        }


class FileSource(VideoSource):
    """File-based video source implementation."""
    
    def __init__(self, config: SourceConfig):
        super().__init__(config)
        self._file_path = Path(config.uri.replace("file://", ""))
    
    async def validate(self) -> bool:
        """Validate file source."""
        try:
            # Check if file exists and is readable
            if not self._file_path.exists():
                raise SourceError(f"Video file not found: {self._file_path}")
            
            if not self._file_path.is_file():
                raise SourceError(f"Path is not a file: {self._file_path}")
            
            # Check file permissions
            if not os.access(self._file_path, os.R_OK):
                raise SourceError(f"File not readable: {self._file_path}")
            
            # Check file format (basic validation)
            supported_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv', '.wmv'}
            if self._file_path.suffix.lower() not in supported_extensions:
                self.logger.warning(f"Unsupported file extension: {self._file_path.suffix}")
            
            # Get file size
            file_size = self._file_path.stat().st_size
            if file_size == 0:
                raise SourceError(f"Empty video file: {self._file_path}")
            
            self.state.update_status(SourceStatus.AVAILABLE)
            self.update_metrics(bytes_received=file_size)
            return True
        
        except Exception as e:
            error_msg = f"File validation failed: {e}"
            self.state.update_status(SourceStatus.ERROR, error_msg)
            self.logger.error(error_msg)
            return False
    
    async def connect(self) -> bool:
        """Connect to file source."""
        try:
            self.state.update_status(SourceStatus.CONNECTING)
            self.state.metrics.connection_attempts += 1
            
            # File sources connect immediately
            self.state.update_status(SourceStatus.CONNECTED)
            self.state.metrics.successful_connections += 1
            self.state.is_active = True
            
            await self._notify_callbacks('connected')
            self.logger.info(f"Connected to file source {self.config.id}")
            return True
        
        except Exception as e:
            error_msg = f"File connection failed: {e}"
            self.state.update_status(SourceStatus.ERROR, error_msg)
            await self._notify_callbacks('connection_failed', error=str(e))
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from file source."""
        try:
            self.state.update_status(SourceStatus.DISCONNECTED)
            self.state.is_active = False
            
            await self._notify_callbacks('disconnected')
            self.logger.info(f"Disconnected from file source {self.config.id}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error disconnecting file source: {e}")
            return False
    
    async def get_capabilities(self) -> Dict[str, Any]:
        """Get file source capabilities."""
        return {
            'type': 'file',
            'protocols': ['file'],
            'supports_seek': True,
            'supports_pause': True,
            'live_source': False,
            'network_source': False,
            'supported_codecs': ['h264', 'h265', 'mpeg4', 'mjpeg'],
            'file_path': str(self._file_path),
            'file_size': self._file_path.stat().st_size if self._file_path.exists() else 0,
            'audio_support': True
        }


class WebcamSource(VideoSource):
    """Webcam video source implementation."""
    
    def __init__(self, config: SourceConfig):
        super().__init__(config)
        self._device_id = self._parse_device_id(config.uri)
    
    def _parse_device_id(self, uri: str) -> Union[int, str]:
        """Parse webcam device ID from URI."""
        if uri.startswith('/dev/video'):
            return uri
        try:
            return int(uri)
        except ValueError:
            return uri
    
    async def validate(self) -> bool:
        """Validate webcam source."""
        try:
            # Check if device exists (Linux-specific)
            if isinstance(self._device_id, str) and self._device_id.startswith('/dev/video'):
                device_path = Path(self._device_id)
                if not device_path.exists():
                    raise SourceError(f"Webcam device not found: {self._device_id}")
                
                # Check device permissions
                if not os.access(device_path, os.R_OK | os.W_OK):
                    raise SourceError(f"Insufficient permissions for webcam: {self._device_id}")
            
            # Try to query device capabilities using v4l2-ctl (Linux)
            try:
                if isinstance(self._device_id, str):
                    result = subprocess.run(
                        ['v4l2-ctl', '--device', self._device_id, '--list-formats-ext'],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0:
                        self.logger.debug(f"Webcam capabilities: {result.stdout[:200]}...")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                # v4l2-ctl not available, continue without capability check
                pass
            
            self.state.update_status(SourceStatus.AVAILABLE)
            return True
        
        except Exception as e:
            error_msg = f"Webcam validation failed: {e}"
            self.state.update_status(SourceStatus.ERROR, error_msg)
            self.logger.error(error_msg)
            return False
    
    async def connect(self) -> bool:
        """Connect to webcam source."""
        try:
            self.state.update_status(SourceStatus.CONNECTING)
            self.state.metrics.connection_attempts += 1
            
            # Webcam connection is handled by GStreamer v4l2src
            await asyncio.sleep(0.1)  # Simulate connection time
            
            self.state.update_status(SourceStatus.CONNECTED)
            self.state.metrics.successful_connections += 1
            self.state.is_active = True
            
            await self._notify_callbacks('connected')
            self.logger.info(f"Connected to webcam source {self.config.id}")
            return True
        
        except Exception as e:
            error_msg = f"Webcam connection failed: {e}"
            self.state.update_status(SourceStatus.ERROR, error_msg)
            await self._notify_callbacks('connection_failed', error=str(e))
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from webcam source."""
        try:
            self.state.update_status(SourceStatus.DISCONNECTED)
            self.state.is_active = False
            
            await self._notify_callbacks('disconnected')
            self.logger.info(f"Disconnected from webcam source {self.config.id}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error disconnecting webcam source: {e}")
            return False
    
    async def get_capabilities(self) -> Dict[str, Any]:
        """Get webcam source capabilities."""
        return {
            'type': 'webcam',
            'protocols': ['v4l2', 'dshow'],
            'supports_seek': False,
            'supports_pause': True,
            'live_source': True,
            'network_source': False,
            'device_id': self._device_id,
            'supported_codecs': ['raw', 'mjpeg', 'h264'],
            'audio_support': False  # Video only by default
        }


class TestSource(VideoSource):
    """Test video source implementation."""
    
    def __init__(self, config: SourceConfig):
        super().__init__(config)
        self._pattern = config.parameters.get('pattern', 'smpte')
    
    async def validate(self) -> bool:
        """Validate test source."""
        try:
            # Test sources are always available
            self.state.update_status(SourceStatus.AVAILABLE)
            return True
        
        except Exception as e:
            error_msg = f"Test source validation failed: {e}"
            self.state.update_status(SourceStatus.ERROR, error_msg)
            self.logger.error(error_msg)
            return False
    
    async def connect(self) -> bool:
        """Connect to test source."""
        try:
            self.state.update_status(SourceStatus.CONNECTING)
            self.state.metrics.connection_attempts += 1
            
            # Test sources connect immediately
            self.state.update_status(SourceStatus.CONNECTED)
            self.state.metrics.successful_connections += 1
            self.state.is_active = True
            
            await self._notify_callbacks('connected')
            self.logger.info(f"Connected to test source {self.config.id}")
            return True
        
        except Exception as e:
            error_msg = f"Test source connection failed: {e}"
            self.state.update_status(SourceStatus.ERROR, error_msg)
            await self._notify_callbacks('connection_failed', error=str(e))
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from test source."""
        try:
            self.state.update_status(SourceStatus.DISCONNECTED)
            self.state.is_active = False
            
            await self._notify_callbacks('disconnected')
            self.logger.info(f"Disconnected from test source {self.config.id}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error disconnecting test source: {e}")
            return False
    
    async def get_capabilities(self) -> Dict[str, Any]:
        """Get test source capabilities."""
        return {
            'type': 'test',
            'protocols': ['test'],
            'supports_seek': False,
            'supports_pause': True,
            'live_source': True,
            'network_source': False,
            'pattern': self._pattern,
            'supported_codecs': ['raw'],
            'audio_support': False
        }


class SourceFactory:
    """Factory for creating video source instances."""
    
    @staticmethod
    def create_source(config: SourceConfig) -> VideoSource:
        """
        Create appropriate video source based on configuration.
        
        Args:
            config: Source configuration
            
        Returns:
            VideoSource instance
            
        Raises:
            SourceError: If source type is not supported
        """
        source_map = {
            SourceType.RTSP: RTSPSource,
            SourceType.FILE: FileSource,
            SourceType.WEBCAM: WebcamSource,
            SourceType.TEST: TestSource,
            # Add more source types as needed
        }
        
        source_class = source_map.get(config.type)
        if not source_class:
            raise SourceError(f"Unsupported source type: {config.type}")
        
        return source_class(config)


class SourceHealthMonitor:
    """Monitors health of video sources."""
    
    def __init__(self, check_interval: float = 30.0):
        """
        Initialize health monitor.
        
        Args:
            check_interval: Interval between health checks in seconds
        """
        self.check_interval = check_interval
        self.logger = get_logger(__name__)
        self._sources: Dict[str, VideoSource] = {}
        self._health_callbacks: List[Callable] = []
        self._periodic_runner = PeriodicTaskRunner(check_interval)
        self._lock = threading.Lock()
    
    def add_source(self, source: VideoSource):
        """Add source to health monitoring."""
        with self._lock:
            self._sources[source.config.id] = source
            self.logger.debug(f"Added source {source.config.id} to health monitoring")
    
    def remove_source(self, source_id: str):
        """Remove source from health monitoring."""
        with self._lock:
            if source_id in self._sources:
                del self._sources[source_id]
                self.logger.debug(f"Removed source {source_id} from health monitoring")
    
    def add_health_callback(self, callback: Callable):
        """Add callback for health events."""
        self._health_callbacks.append(callback)
    
    async def start_monitoring(self):
        """Start health monitoring."""
        await self._periodic_runner.start_periodic_task(
            "health_check",
            self._perform_health_check
        )
        self.logger.info("Started source health monitoring")
    
    async def stop_monitoring(self):
        """Stop health monitoring."""
        await self._periodic_runner.stop_all_tasks()
        self.logger.info("Stopped source health monitoring")
    
    async def _perform_health_check(self):
        """Perform health check on all sources."""
        with self._lock:
            sources = list(self._sources.values())
        
        for source in sources:
            try:
                await self._check_source_health(source)
            except Exception as e:
                self.logger.error(f"Error checking health of source {source.config.id}: {e}")
    
    async def _check_source_health(self, source: VideoSource):
        """Check health of individual source."""
        try:
            current_time = time.time()
            metrics = source.state.metrics
            
            # Calculate health based on various factors
            health_score = 100.0
            
            # Check frame reception
            time_since_last_frame = current_time - metrics.last_frame_time
            if time_since_last_frame > 30.0:  # No frames for 30 seconds
                health_score -= 40.0
            elif time_since_last_frame > 10.0:  # No frames for 10 seconds
                health_score -= 20.0
            
            # Check error rate
            if metrics.connection_attempts > 0:
                success_rate = metrics.successful_connections / metrics.connection_attempts
                health_score *= success_rate
            
            # Check reconnection frequency
            if metrics.reconnection_count > 5:
                health_score -= 20.0
            
            # Determine health level
            if health_score >= 80:
                new_health = SourceHealth.HEALTHY
            elif health_score >= 60:
                new_health = SourceHealth.DEGRADED
            elif health_score >= 30:
                new_health = SourceHealth.UNHEALTHY
            else:
                new_health = SourceHealth.CRITICAL
            
            # Update health if changed
            old_health = source.state.health
            if new_health != old_health:
                source.state.health = new_health
                source.state.last_updated = current_time
                
                # Notify callbacks
                for callback in self._health_callbacks:
                    try:
                        await callback({
                            'source_id': source.config.id,
                            'old_health': old_health.value,
                            'new_health': new_health.value,
                            'health_score': health_score,
                            'metrics': metrics.to_dict()
                        })
                    except Exception as e:
                        self.logger.error(f"Error in health callback: {e}")
                
                self.logger.info(
                    f"Source {source.config.id} health changed: "
                    f"{old_health.value} -> {new_health.value} (score: {health_score:.1f})"
                )
        
        except Exception as e:
            self.logger.error(f"Error in health check for {source.config.id}: {e}")


class VideoSourceManager:
    """
    Comprehensive video source management system.
    
    Manages multiple video sources with health monitoring, dynamic addition/removal,
    and automatic retry mechanisms.
    """
    
    def __init__(self):
        """Initialize video source manager."""
        self.logger = get_logger(__name__)
        self._sources: Dict[str, VideoSource] = {}
        self._source_callbacks: List[Callable] = []
        self._health_monitor = SourceHealthMonitor()
        self._task_manager = get_task_manager()
        self._lock = threading.Lock()
        
        # Configuration
        self._retry_attempts = 3
        self._retry_delay = 5.0
        self._health_check_interval = 30.0
        
        # Event queue for async processing
        self._event_queue = ThreadSafeAsyncQueue(maxsize=1000)
        
        self.logger.info("VideoSourceManager initialized")
    
    async def add_source(self, config: SourceConfig) -> bool:
        """
        Add a new video source.
        
        Args:
            config: Source configuration
            
        Returns:
            True if source added successfully
        """
        try:
            if config.id in self._sources:
                raise SourceError(f"Source {config.id} already exists")
            
            self.logger.info(f"Adding source {config.id} (type: {config.type.value})")
            
            # Create source instance
            source = SourceFactory.create_source(config)
            
            # Add callbacks
            source.add_callback(self._on_source_event)
            
            # Validate source
            if not await source.validate():
                raise SourceError(f"Source validation failed for {config.id}")
            
            # Store source
            with self._lock:
                self._sources[config.id] = source
            
            # Add to health monitoring
            self._health_monitor.add_source(source)
            
            # Start monitoring
            await source.start_monitoring()
            
            # Notify callbacks
            await self._notify_source_callbacks('source_added', config.id, source.state)
            
            self.logger.info(f"Successfully added source {config.id}")
            return True
        
        except Exception as e:
            error_msg = f"Failed to add source {config.id}: {e}"
            self.logger.error(error_msg)
            await self._notify_source_callbacks('source_add_failed', config.id, None, error=str(e))
            return False
    
    async def remove_source(self, source_id: str) -> bool:
        """
        Remove a video source.
        
        Args:
            source_id: ID of source to remove
            
        Returns:
            True if source removed successfully
        """
        try:
            with self._lock:
                if source_id not in self._sources:
                    self.logger.warning(f"Source {source_id} not found for removal")
                    return True
                
                source = self._sources[source_id]
            
            self.logger.info(f"Removing source {source_id}")
            
            # Stop monitoring
            await source.stop_monitoring()
            
            # Disconnect if connected
            if source.state.is_active:
                await source.disconnect()
            
            # Remove from health monitoring
            self._health_monitor.remove_source(source_id)
            
            # Remove from sources
            with self._lock:
                del self._sources[source_id]
            
            # Notify callbacks
            await self._notify_source_callbacks('source_removed', source_id, source.state)
            
            self.logger.info(f"Successfully removed source {source_id}")
            return True
        
        except Exception as e:
            error_msg = f"Failed to remove source {source_id}: {e}"
            self.logger.error(error_msg)
            return False
    
    async def connect_source(self, source_id: str) -> bool:
        """
        Connect to a specific source.
        
        Args:
            source_id: ID of source to connect
            
        Returns:
            True if connection successful
        """
        with self._lock:
            if source_id not in self._sources:
                raise SourceError(f"Source {source_id} not found")
            source = self._sources[source_id]
        
        return await source.connect()
    
    async def disconnect_source(self, source_id: str) -> bool:
        """
        Disconnect from a specific source.
        
        Args:
            source_id: ID of source to disconnect
            
        Returns:
            True if disconnection successful
        """
        with self._lock:
            if source_id not in self._sources:
                raise SourceError(f"Source {source_id} not found")
            source = self._sources[source_id]
        
        return await source.disconnect()
    
    async def connect_all_sources(self) -> Dict[str, bool]:
        """
        Connect to all sources.
        
        Returns:
            Dictionary mapping source IDs to connection success
        """
        results = {}
        
        with self._lock:
            sources = list(self._sources.items())
        
        for source_id, source in sources:
            try:
                if source.config.enabled:
                    results[source_id] = await source.connect()
                else:
                    results[source_id] = False
                    self.logger.debug(f"Skipping disabled source {source_id}")
            except Exception as e:
                self.logger.error(f"Error connecting to source {source_id}: {e}")
                results[source_id] = False
        
        connected_count = sum(results.values())
        self.logger.info(f"Connected to {connected_count}/{len(results)} sources")
        
        return results
    
    async def disconnect_all_sources(self) -> Dict[str, bool]:
        """
        Disconnect from all sources.
        
        Returns:
            Dictionary mapping source IDs to disconnection success
        """
        results = {}
        
        with self._lock:
            sources = list(self._sources.items())
        
        for source_id, source in sources:
            try:
                results[source_id] = await source.disconnect()
            except Exception as e:
                self.logger.error(f"Error disconnecting from source {source_id}: {e}")
                results[source_id] = False
        
        return results
    
    def get_source(self, source_id: str) -> Optional[VideoSource]:
        """Get source by ID."""
        with self._lock:
            return self._sources.get(source_id)
    
    def get_all_sources(self) -> Dict[str, VideoSource]:
        """Get all sources."""
        with self._lock:
            return self._sources.copy()
    
    def get_active_sources(self) -> Dict[str, VideoSource]:
        """Get all active (connected) sources."""
        with self._lock:
            return {
                source_id: source
                for source_id, source in self._sources.items()
                if source.state.is_active
            }
    
    def get_source_states(self) -> Dict[str, SourceState]:
        """Get states of all sources."""
        with self._lock:
            return {
                source_id: source.state
                for source_id, source in self._sources.items()
            }
    
    def get_source_metrics(self) -> Dict[str, SourceMetrics]:
        """Get metrics for all sources."""
        with self._lock:
            return {
                source_id: source.state.metrics
                for source_id, source in self._sources.items()
            }
    
    async def _on_source_event(self, event_data: Dict[str, Any]):
        """Handle events from individual sources."""
        try:
            source_id = event_data['source_id']
            event_type = event_data['event_type']
            
            # Queue event for async processing
            self._event_queue.put_nowait(event_data)
            
            # Handle specific events
            if event_type == 'connection_failed':
                await self._handle_connection_failure(source_id, event_data.get('error'))
            elif event_type == 'disconnected':
                await self._handle_disconnection(source_id)
        
        except Exception as e:
            self.logger.error(f"Error handling source event: {e}")
    
    async def _handle_connection_failure(self, source_id: str, error: Optional[str]):
        """Handle source connection failures with retry logic."""
        try:
            with self._lock:
                if source_id not in self._sources:
                    return
                source = self._sources[source_id]
            
            # Check if we should retry
            if source.state.retry_count < source.config.max_retries:
                source.state.retry_count += 1
                retry_delay = source.config.retry_delay * (2 ** (source.state.retry_count - 1))  # Exponential backoff
                source.state.next_retry = time.time() + retry_delay
                
                self.logger.info(
                    f"Scheduling retry {source.state.retry_count}/{source.config.max_retries} "
                    f"for source {source_id} in {retry_delay:.1f}s"
                )
                
                # Schedule retry
                await self._task_manager.create_task(
                    self._retry_connection(source_id, retry_delay),
                    name=f"retry_{source_id}_{source.state.retry_count}",
                    description=f"Retry connection to source {source_id}"
                )
            else:
                self.logger.error(
                    f"Max retries exceeded for source {source_id}, marking as failed"
                )
                source.state.update_status(SourceStatus.ERROR, "Max retries exceeded")
                await self._notify_source_callbacks('source_failed', source_id, source.state)
        
        except Exception as e:
            self.logger.error(f"Error handling connection failure for {source_id}: {e}")
    
    async def _retry_connection(self, source_id: str, delay: float):
        """Retry connection to a source after delay."""
        try:
            await asyncio.sleep(delay)
            
            with self._lock:
                if source_id not in self._sources:
                    return
                source = self._sources[source_id]
            
            self.logger.info(f"Retrying connection to source {source_id}")
            
            if await source.connect():
                source.state.retry_count = 0  # Reset retry count on success
                source.state.next_retry = None
                source.state.metrics.reconnection_count += 1
                await self._notify_source_callbacks('source_reconnected', source_id, source.state)
        
        except Exception as e:
            self.logger.error(f"Error retrying connection to {source_id}: {e}")
    
    async def _handle_disconnection(self, source_id: str):
        """Handle source disconnections."""
        try:
            self.logger.info(f"Source {source_id} disconnected")
            
            # For network sources, attempt reconnection
            with self._lock:
                if source_id not in self._sources:
                    return
                source = self._sources[source_id]
            
            if source.config.type in [SourceType.RTSP, SourceType.WEBRTC]:
                await self._handle_connection_failure(source_id, "Connection lost")
        
        except Exception as e:
            self.logger.error(f"Error handling disconnection for {source_id}: {e}")
    
    def add_source_callback(self, callback: Callable):
        """Add callback for source events."""
        self._source_callbacks.append(callback)
    
    async def _notify_source_callbacks(self, event_type: str, source_id: str, state: Optional[SourceState], **kwargs):
        """Notify all source callbacks."""
        event_data = {
            'event_type': event_type,
            'source_id': source_id,
            'state': state,
            'timestamp': time.time(),
            **kwargs
        }
        
        for callback in self._source_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event_data)
                else:
                    callback(event_data)
            except Exception as e:
                self.logger.error(f"Error in source callback: {e}")
    
    async def start_health_monitoring(self):
        """Start health monitoring for all sources."""
        await self._health_monitor.start_monitoring()
        self.logger.info("Started source health monitoring")
    
    async def stop_health_monitoring(self):
        """Stop health monitoring."""
        await self._health_monitor.stop_monitoring()
        self.logger.info("Stopped source health monitoring")
    
    async def stop(self):
        """Stop source manager and clean up resources."""
        self.logger.info("Stopping VideoSourceManager...")
        
        try:
            # Stop health monitoring
            await self.stop_health_monitoring()
            
            # Disconnect all sources
            await self.disconnect_all_sources()
            
            # Stop monitoring for all sources
            with self._lock:
                sources = list(self._sources.values())
            
            for source in sources:
                await source.stop_monitoring()
            
            # Clear sources
            with self._lock:
                self._sources.clear()
            
            self.logger.info("VideoSourceManager stopped successfully")
        
        except Exception as e:
            self.logger.error(f"Error stopping VideoSourceManager: {e}")


# Global source manager instance
_source_manager: Optional[VideoSourceManager] = None


def get_source_manager() -> VideoSourceManager:
    """Get global source manager instance."""
    global _source_manager
    if _source_manager is None:
        _source_manager = VideoSourceManager()
    return _source_manager