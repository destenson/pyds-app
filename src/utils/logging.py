"""
Structured logging system for the DeepStream inference system.

This module provides comprehensive logging with JSON output, performance metrics
integration, configurable levels, rotation, and real-time streaming capabilities.
"""

import json
import sys
import os
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union, List, Callable
from enum import Enum
import logging
import logging.handlers
from contextlib import contextmanager
import structlog
import psutil
from dataclasses import dataclass, asdict


class LogLevel(Enum):
    """Log levels for structured logging."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class LogFormat(Enum):
    """Available log formats."""
    JSON = "json"
    TEXT = "text"
    COLORED = "colored"


@dataclass
class PerformanceMetrics:
    """Performance metrics to include in logs."""
    timestamp: float
    fps: Optional[float] = None
    latency_ms: Optional[float] = None
    cpu_percent: Optional[float] = None
    memory_mb: Optional[float] = None
    gpu_utilization: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    pipeline_state: Optional[str] = None
    source_id: Optional[str] = None
    detection_count: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}


class MetricsCollector:
    """Collects performance metrics for logging."""
    
    def __init__(self):
        self._process = psutil.Process()
        self._metrics_lock = threading.Lock()
        self._current_metrics = PerformanceMetrics(timestamp=time.time())
        
        # Try to import NVIDIA ML for GPU metrics
        try:
            import pynvml
            pynvml.nvmlInit()
            self._has_gpu = True
            self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except (ImportError, Exception):
            self._has_gpu = False
            self._gpu_handle = None
    
    def update_fps(self, fps: float, source_id: Optional[str] = None):
        """Update FPS metric."""
        with self._metrics_lock:
            self._current_metrics.fps = fps
            self._current_metrics.source_id = source_id
            self._current_metrics.timestamp = time.time()
    
    def update_latency(self, latency_ms: float):
        """Update latency metric."""
        with self._metrics_lock:
            self._current_metrics.latency_ms = latency_ms
            self._current_metrics.timestamp = time.time()
    
    def update_pipeline_state(self, state: str):
        """Update pipeline state."""
        with self._metrics_lock:
            self._current_metrics.pipeline_state = state
            self._current_metrics.timestamp = time.time()
    
    def update_detection_count(self, count: int):
        """Update detection count."""
        with self._metrics_lock:
            self._current_metrics.detection_count = count
            self._current_metrics.timestamp = time.time()
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        with self._metrics_lock:
            # Update system metrics
            try:
                cpu_percent = self._process.cpu_percent()
                memory_info = self._process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                
                self._current_metrics.cpu_percent = cpu_percent
                self._current_metrics.memory_mb = memory_mb
                
                # Update GPU metrics if available
                if self._has_gpu and self._gpu_handle:
                    try:
                        import pynvml
                        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                        gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                        
                        self._current_metrics.gpu_utilization = gpu_util.gpu
                        self._current_metrics.gpu_memory_mb = gpu_memory.used / 1024 / 1024
                    except Exception:
                        pass  # GPU metrics not critical
                        
            except Exception:
                pass  # System metrics not critical
            
            # Return a copy of current metrics
            return PerformanceMetrics(
                timestamp=self._current_metrics.timestamp,
                fps=self._current_metrics.fps,
                latency_ms=self._current_metrics.latency_ms,
                cpu_percent=self._current_metrics.cpu_percent,
                memory_mb=self._current_metrics.memory_mb,
                gpu_utilization=self._current_metrics.gpu_utilization,
                gpu_memory_mb=self._current_metrics.gpu_memory_mb,
                pipeline_state=self._current_metrics.pipeline_state,
                source_id=self._current_metrics.source_id,
                detection_count=self._current_metrics.detection_count
            )


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def __init__(self, include_metrics: bool = False):
        super().__init__()
        self.include_metrics = include_metrics
        self.metrics_collector = MetricsCollector() if include_metrics else None
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Base log structure
        log_obj = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname.lower(),
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'thread_name': record.threadName,
            'process': record.process
        }
        
        # Add exception info if present
        if record.exc_info:
            log_obj['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields from record
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in ('name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'message'):
                extra_fields[key] = value
        
        if extra_fields:
            log_obj['extra'] = extra_fields
        
        # Add performance metrics if enabled
        if self.include_metrics and self.metrics_collector:
            try:
                metrics = self.metrics_collector.get_current_metrics()
                log_obj['metrics'] = metrics.to_dict()
            except Exception:
                pass  # Don't fail logging due to metrics
        
        return json.dumps(log_obj, default=str, ensure_ascii=False)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m'    # Magenta
    }
    RESET = '\033[0m'
    
    def __init__(self, include_metrics: bool = False):
        super().__init__()
        self.include_metrics = include_metrics
        self.metrics_collector = MetricsCollector() if include_metrics else None
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        # Color the level name
        color = self.COLORS.get(record.levelname, '')
        colored_level = f"{color}{record.levelname}{self.RESET}"
        
        # Base format
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        base_msg = f"[{timestamp}] {colored_level} {record.name}: {record.getMessage()}"
        
        # Add location info for errors
        if record.levelno >= logging.ERROR:
            base_msg += f" ({record.filename}:{record.lineno})"
        
        # Add performance metrics if enabled
        if self.include_metrics and self.metrics_collector:
            try:
                metrics = self.metrics_collector.get_current_metrics()
                if metrics.fps is not None:
                    base_msg += f" [FPS: {metrics.fps:.1f}]"
                if metrics.latency_ms is not None:
                    base_msg += f" [Latency: {metrics.latency_ms:.1f}ms]"
                if metrics.cpu_percent is not None:
                    base_msg += f" [CPU: {metrics.cpu_percent:.1f}%]"
                if metrics.memory_mb is not None:
                    base_msg += f" [Mem: {metrics.memory_mb:.0f}MB]"
            except Exception:
                pass  # Don't fail logging due to metrics
        
        return base_msg


class LoggingConfig:
    """Configuration for the logging system."""
    
    def __init__(
        self,
        level: LogLevel = LogLevel.INFO,
        format: LogFormat = LogFormat.COLORED,
        include_metrics: bool = True,
        log_file: Optional[str] = None,
        max_file_size_mb: int = 100,
        backup_count: int = 5,
        console_output: bool = True,
        json_output: bool = False,
        real_time_streaming: bool = False,
        streaming_port: int = 9999,
        filter_spam: bool = True,
        rate_limit_per_minute: int = 1000
    ):
        self.level = level
        self.format = format
        self.include_metrics = include_metrics
        self.log_file = log_file
        self.max_file_size_mb = max_file_size_mb
        self.backup_count = backup_count
        self.console_output = console_output
        self.json_output = json_output
        self.real_time_streaming = real_time_streaming
        self.streaming_port = streaming_port
        self.filter_spam = filter_spam
        self.rate_limit_per_minute = rate_limit_per_minute


class SpamFilter(logging.Filter):
    """Filter to prevent log spam by rate limiting similar messages."""
    
    def __init__(self, rate_limit_per_minute: int = 1000):
        super().__init__()
        self.rate_limit_per_minute = rate_limit_per_minute
        self.message_counts = {}
        self.last_cleanup = time.time()
        self.cleanup_interval = 60  # Clean up every minute
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log records to prevent spam."""
        now = time.time()
        
        # Clean up old counts periodically
        if now - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_counts(now)
            self.last_cleanup = now
        
        # Create message key
        msg_key = f"{record.levelname}:{record.name}:{record.getMessage()[:100]}"
        
        # Track message count
        if msg_key not in self.message_counts:
            self.message_counts[msg_key] = {'count': 0, 'first_seen': now, 'last_seen': now}
        
        msg_info = self.message_counts[msg_key]
        msg_info['count'] += 1
        msg_info['last_seen'] = now
        
        # Calculate rate (messages per minute)
        time_span = max(now - msg_info['first_seen'], 1)  # Avoid division by zero
        rate = (msg_info['count'] / time_span) * 60
        
        # Allow message if under rate limit
        if rate <= self.rate_limit_per_minute:
            return True
        
        # For rate-limited messages, only allow every Nth occurrence
        # This ensures we don't completely silence important repeated errors
        nth_occurrence = max(1, int(rate / self.rate_limit_per_minute))
        return msg_info['count'] % nth_occurrence == 0
    
    def _cleanup_old_counts(self, now: float):
        """Remove old message counts to prevent memory growth."""
        cutoff_time = now - 300  # Remove counts older than 5 minutes
        keys_to_remove = []
        
        for key, info in self.message_counts.items():
            if info['last_seen'] < cutoff_time:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.message_counts[key]


class StreamingHandler(logging.Handler):
    """Handler for real-time log streaming over network."""
    
    def __init__(self, port: int = 9999):
        super().__init__()
        self.port = port
        self.clients = []
        self.server_socket = None
        self.server_thread = None
        self.running = False
        
        # Start server in background thread
        self._start_server()
    
    def _start_server(self):
        """Start TCP server for log streaming."""
        import socket
        import threading
        
        def server_worker():
            try:
                self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.server_socket.bind(('localhost', self.port))
                self.server_socket.listen(5)
                self.running = True
                
                while self.running:
                    try:
                        client_socket, addr = self.server_socket.accept()
                        self.clients.append(client_socket)
                    except Exception:
                        break
            except Exception as e:
                print(f"Failed to start log streaming server: {e}")
        
        self.server_thread = threading.Thread(target=server_worker, daemon=True)
        self.server_thread.start()
    
    def emit(self, record: logging.LogRecord):
        """Send log record to all connected clients."""
        if not self.clients:
            return
        
        try:
            msg = self.format(record) + '\n'
            msg_bytes = msg.encode('utf-8')
            
            # Send to all clients, removing disconnected ones
            clients_to_remove = []
            for client in self.clients:
                try:
                    client.send(msg_bytes)
                except Exception:
                    clients_to_remove.append(client)
            
            # Remove disconnected clients
            for client in clients_to_remove:
                self.clients.remove(client)
                try:
                    client.close()
                except Exception:
                    pass
                    
        except Exception:
            pass  # Don't fail logging due to streaming issues
    
    def close(self):
        """Close streaming handler."""
        self.running = False
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception:
                pass
        
        for client in self.clients:
            try:
                client.close()
            except Exception:
                pass
        
        super().close()


class PyDSLogger:
    """Main logging interface for the PyDS application."""
    
    def __init__(self, config: Optional[LoggingConfig] = None):
        self.config = config or LoggingConfig()
        self.metrics_collector = MetricsCollector()
        self._setup_logging()
    
    def _setup_logging(self):
        """Set up structured logging based on configuration."""
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer() if self.config.json_output else structlog.dev.ConsoleRenderer(colors=True),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        # Get root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.level.value.upper()))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Add console handler
        if self.config.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            
            if self.config.format == LogFormat.JSON:
                formatter = JSONFormatter(include_metrics=self.config.include_metrics)
            elif self.config.format == LogFormat.COLORED:
                formatter = ColoredFormatter(include_metrics=self.config.include_metrics)
            else:
                formatter = logging.Formatter(
                    '[%(asctime)s] %(levelname)s %(name)s: %(message)s'
                )
            
            console_handler.setFormatter(formatter)
            
            # Add spam filter if enabled
            if self.config.filter_spam:
                console_handler.addFilter(
                    SpamFilter(self.config.rate_limit_per_minute)
                )
            
            root_logger.addHandler(console_handler)
        
        # Add file handler
        if self.config.log_file:
            log_path = Path(self.config.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                self.config.log_file,
                maxBytes=self.config.max_file_size_mb * 1024 * 1024,
                backupCount=self.config.backup_count
            )
            
            # Always use JSON format for file output
            file_formatter = JSONFormatter(include_metrics=self.config.include_metrics)
            file_handler.setFormatter(file_formatter)
            
            # Add spam filter if enabled
            if self.config.filter_spam:
                file_handler.addFilter(
                    SpamFilter(self.config.rate_limit_per_minute)
                )
            
            root_logger.addHandler(file_handler)
        
        # Add streaming handler
        if self.config.real_time_streaming:
            streaming_handler = StreamingHandler(self.config.streaming_port)
            streaming_formatter = JSONFormatter(include_metrics=self.config.include_metrics)
            streaming_handler.setFormatter(streaming_formatter)
            root_logger.addHandler(streaming_handler)
    
    def get_logger(self, name: str) -> structlog.BoundLogger:
        """Get a structured logger instance."""
        return structlog.get_logger(name)
    
    def update_metrics(
        self,
        fps: Optional[float] = None,
        latency_ms: Optional[float] = None,
        pipeline_state: Optional[str] = None,
        source_id: Optional[str] = None,
        detection_count: Optional[int] = None
    ):
        """Update performance metrics for logging."""
        if fps is not None:
            self.metrics_collector.update_fps(fps, source_id)
        if latency_ms is not None:
            self.metrics_collector.update_latency(latency_ms)
        if pipeline_state is not None:
            self.metrics_collector.update_pipeline_state(pipeline_state)
        if detection_count is not None:
            self.metrics_collector.update_detection_count(detection_count)
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        return self.metrics_collector.get_current_metrics()
    
    @contextmanager
    def performance_context(self, operation_name: str):
        """Context manager for measuring operation performance."""
        start_time = time.time()
        logger = structlog.get_logger("performance")
        
        try:
            logger.debug("Operation started", operation=operation_name)
            yield
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(
                "Operation failed",
                operation=operation_name,
                duration_ms=duration_ms,
                error=str(e)
            )
            raise
            
        else:
            duration_ms = (time.time() - start_time) * 1000
            logger.info(
                "Operation completed",
                operation=operation_name,
                duration_ms=duration_ms
            )
            
            # Update latency metric
            self.update_metrics(latency_ms=duration_ms)


# Global logger instance
_global_logger: Optional[PyDSLogger] = None


def setup_logging(config: Optional[LoggingConfig] = None) -> PyDSLogger:
    """Set up global logging configuration."""
    global _global_logger
    _global_logger = PyDSLogger(config)
    return _global_logger


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a logger instance with the given name."""
    if _global_logger is None:
        setup_logging()
    return _global_logger.get_logger(name)


def update_metrics(**kwargs):
    """Update performance metrics in the global logger."""
    if _global_logger is not None:
        _global_logger.update_metrics(**kwargs)


def performance_context(operation_name: str):
    """Context manager for measuring performance."""
    if _global_logger is None:
        setup_logging()
    return _global_logger.performance_context(operation_name)


# Convenience functions for common logging patterns
def log_pipeline_event(
    logger: structlog.BoundLogger,
    event: str,
    pipeline_id: Optional[str] = None,
    source_id: Optional[str] = None,
    **context
):
    """Log a pipeline event with standard context."""
    log_context = {
        'event_type': 'pipeline',
        'event': event,
        **context
    }
    
    if pipeline_id:
        log_context['pipeline_id'] = pipeline_id
    if source_id:
        log_context['source_id'] = source_id
    
    logger.info("Pipeline event", **log_context)


def log_detection_event(
    logger: structlog.BoundLogger,
    detection_count: int,
    source_id: str,
    confidence_avg: Optional[float] = None,
    **context
):
    """Log a detection event with standard context."""
    log_context = {
        'event_type': 'detection',
        'detection_count': detection_count,
        'source_id': source_id,
        **context
    }
    
    if confidence_avg:
        log_context['confidence_avg'] = confidence_avg
    
    logger.info("Detection event", **log_context)
    
    # Update global metrics
    update_metrics(detection_count=detection_count)


def log_alert_event(
    logger: structlog.BoundLogger,
    alert_type: str,
    source_id: str,
    handler: str,
    success: bool = True,
    **context
):
    """Log an alert event with standard context."""
    log_context = {
        'event_type': 'alert',
        'alert_type': alert_type,
        'source_id': source_id,
        'handler': handler,
        'success': success,
        **context
    }
    
    if success:
        logger.info("Alert sent", **log_context)
    else:
        logger.error("Alert failed", **log_context)


def log_error_with_context(
    logger: structlog.BoundLogger,
    error: Exception,
    operation: str,
    **context
):
    """Log an error with comprehensive context."""
    log_context = {
        'event_type': 'error',
        'operation': operation,
        'error_type': type(error).__name__,
        'error_message': str(error),
        **context
    }
    
    logger.error("Operation failed", **log_context, exc_info=True)