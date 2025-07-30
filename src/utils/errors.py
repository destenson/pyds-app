"""
Error handling and exception hierarchy for the DeepStream inference system.

This module provides a comprehensive exception hierarchy with context preservation,
error recovery strategies, and DeepStream version compatibility.
"""

import time
from typing import Any, Dict, Optional, Union, Type
from enum import Enum
import traceback
from datetime import datetime
import asyncio


class ErrorSeverity(Enum):
    """Error severity levels for logging and alerting."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors for classification and handling."""
    CONFIGURATION = "configuration"
    PIPELINE = "pipeline"
    DETECTION = "detection"
    ALERT = "alert"
    MONITORING = "monitoring"
    NETWORK = "network"
    SYSTEM = "system"
    DEEPSTREAM = "deepstream"
    GSTREAMER = "gstreamer"


class RecoveryAction(Enum):
    """Actions that can be taken to recover from errors."""
    RETRY = "retry"
    RESTART = "restart"
    SKIP = "skip"
    FAIL = "fail"
    RESET = "reset"


class PyDSError(Exception):
    """
    Base exception class for all PyDS application errors.
    
    Provides context preservation, error categorization, and recovery hints.
    """
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        source_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        recoverable: bool = True,
        retry_delay: float = 1.0,
        max_retries: int = 3,
        original_exception: Optional[Exception] = None
    ):
        """
        Initialize PyDS error with comprehensive context.
        
        Args:
            message: Human-readable error description
            category: Error category for classification
            severity: Error severity level
            source_id: Identifier of the source that caused the error
            context: Additional context information
            recoverable: Whether the error can be recovered from
            retry_delay: Initial retry delay in seconds
            max_retries: Maximum number of retry attempts
            original_exception: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.source_id = source_id
        self.context = context or {}
        self.recoverable = recoverable
        self.retry_delay = retry_delay
        self.max_retries = max_retries
        self.original_exception = original_exception
        self.timestamp = datetime.now()
        self.traceback_str = traceback.format_exc()
        
        # Add system context
        self.context.update({
            'timestamp': self.timestamp.isoformat(),
            'category': self.category.value,
            'severity': self.severity.value,
            'recoverable': self.recoverable,
            'retry_delay': self.retry_delay,
            'max_retries': self.max_retries
        })
        
        if original_exception:
            self.context['original_error'] = str(original_exception)
            self.context['original_type'] = type(original_exception).__name__

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging and serialization."""
        return {
            'error_type': type(self).__name__,
            'message': self.message,
            'category': self.category.value,
            'severity': self.severity.value,
            'source_id': self.source_id,
            'context': self.context,
            'recoverable': self.recoverable,
            'timestamp': self.timestamp.isoformat(),
            'traceback': self.traceback_str
        }

    def __str__(self) -> str:
        base_msg = f"[{self.category.value.upper()}] {self.message}"
        if self.source_id:
            base_msg += f" (source: {self.source_id})"
        return base_msg


class ConfigurationError(PyDSError):
    """Errors related to configuration loading, validation, or processing."""
    
    def __init__(self, message: str, config_path: Optional[str] = None, **kwargs):
        kwargs.setdefault('category', ErrorCategory.CONFIGURATION)
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        kwargs.setdefault('recoverable', False)  # Config errors usually require manual fix
        
        context = kwargs.get('context', {})
        if config_path:
            context['config_path'] = config_path
        kwargs['context'] = context
        
        super().__init__(message, **kwargs)


class PipelineError(PyDSError):
    """Errors related to GStreamer pipeline operations."""
    
    def __init__(
        self, 
        message: str, 
        pipeline_id: Optional[str] = None,
        element_name: Optional[str] = None,
        gst_state: Optional[str] = None,
        **kwargs
    ):
        kwargs.setdefault('category', ErrorCategory.PIPELINE)
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        kwargs.setdefault('recoverable', True)
        kwargs.setdefault('retry_delay', 2.0)
        kwargs.setdefault('max_retries', 3)
        
        context = kwargs.get('context', {})
        if pipeline_id:
            context['pipeline_id'] = pipeline_id
        if element_name:
            context['element_name'] = element_name
        if gst_state:
            context['gst_state'] = gst_state
        kwargs['context'] = context
        
        super().__init__(message, **kwargs)


class SourceError(PipelineError):
    """Errors specific to video source handling."""
    
    def __init__(
        self, 
        message: str, 
        source_uri: Optional[str] = None,
        source_type: Optional[str] = None,
        **kwargs
    ):
        kwargs.setdefault('severity', ErrorSeverity.MEDIUM)
        kwargs.setdefault('retry_delay', 5.0)  # Longer delay for source reconnection
        kwargs.setdefault('max_retries', 5)    # More retries for network sources
        
        context = kwargs.get('context', {})
        if source_uri:
            context['source_uri'] = source_uri
        if source_type:
            context['source_type'] = source_type
        kwargs['context'] = context
        
        super().__init__(message, **kwargs)


class NetworkSourceError(SourceError):
    """Errors specific to network-based video sources (RTSP, WebRTC, etc.)."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.NETWORK)
        kwargs.setdefault('retry_delay', 10.0)  # Longer delay for network issues
        kwargs.setdefault('max_retries', 10)    # More retries for network sources
        super().__init__(message, **kwargs)


class DetectionError(PyDSError):
    """Errors related to detection engine and strategies."""
    
    def __init__(
        self, 
        message: str, 
        strategy_name: Optional[str] = None,
        model_path: Optional[str] = None,
        **kwargs
    ):
        kwargs.setdefault('category', ErrorCategory.DETECTION)
        kwargs.setdefault('severity', ErrorSeverity.MEDIUM)
        kwargs.setdefault('recoverable', True)
        kwargs.setdefault('retry_delay', 1.0)
        
        context = kwargs.get('context', {})
        if strategy_name:
            context['strategy_name'] = strategy_name
        if model_path:
            context['model_path'] = model_path
        kwargs['context'] = context
        
        super().__init__(message, **kwargs)


class ModelLoadError(DetectionError):
    """Errors related to loading detection models."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        kwargs.setdefault('recoverable', False)  # Model load errors usually require manual fix
        super().__init__(message, **kwargs)


class AlertError(PyDSError):
    """Errors related to alert processing and delivery."""
    
    def __init__(
        self, 
        message: str, 
        handler_name: Optional[str] = None,
        alert_id: Optional[str] = None,
        **kwargs
    ):
        kwargs.setdefault('category', ErrorCategory.ALERT)
        kwargs.setdefault('severity', ErrorSeverity.LOW)  # Alert errors shouldn't stop detection
        kwargs.setdefault('recoverable', True)
        kwargs.setdefault('retry_delay', 0.5)
        kwargs.setdefault('max_retries', 2)
        
        context = kwargs.get('context', {})
        if handler_name:
            context['handler_name'] = handler_name
        if alert_id:
            context['alert_id'] = alert_id
        kwargs['context'] = context
        
        super().__init__(message, **kwargs)


class ApplicationError(PyDSError):
    """General application-level errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.SYSTEM)
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        kwargs.setdefault('recoverable', False)
        
        super().__init__(message, **kwargs)


class MonitoringError(PyDSError):
    """Errors related to monitoring and metrics collection."""
    
    def __init__(self, message: str, metric_name: Optional[str] = None, **kwargs):
        kwargs.setdefault('category', ErrorCategory.MONITORING)
        kwargs.setdefault('severity', ErrorSeverity.LOW)
        kwargs.setdefault('recoverable', True)
        
        context = kwargs.get('context', {})
        if metric_name:
            context['metric_name'] = metric_name
        kwargs['context'] = context
        
        super().__init__(message, **kwargs)


class HealthError(PyDSError):
    """Errors related to health monitoring and checks."""
    
    def __init__(self, message: str, component_name: Optional[str] = None, **kwargs):
        kwargs.setdefault('category', ErrorCategory.MONITORING)
        kwargs.setdefault('severity', ErrorSeverity.MEDIUM)
        kwargs.setdefault('recoverable', True)
        
        context = kwargs.get('context', {})
        if component_name:
            context['component_name'] = component_name
        kwargs['context'] = context
        
        super().__init__(message, **kwargs)


class ProfilingError(PyDSError):
    """Errors related to performance profiling."""
    
    def __init__(self, message: str, profiler_name: Optional[str] = None, **kwargs):
        kwargs.setdefault('category', ErrorCategory.MONITORING)
        kwargs.setdefault('severity', ErrorSeverity.LOW)
        kwargs.setdefault('recoverable', True)
        
        context = kwargs.get('context', {})
        if profiler_name:
            context['profiler_name'] = profiler_name
        kwargs['context'] = context
        
        super().__init__(message, **kwargs)


class DeepStreamError(PyDSError):
    """Errors specific to DeepStream operations and version compatibility."""
    
    def __init__(
        self, 
        message: str, 
        deepstream_version: Optional[str] = None,
        plugin_name: Optional[str] = None,
        **kwargs
    ):
        kwargs.setdefault('category', ErrorCategory.DEEPSTREAM)
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        kwargs.setdefault('recoverable', True)
        kwargs.setdefault('retry_delay', 1.0)
        
        context = kwargs.get('context', {})
        if deepstream_version:
            context['deepstream_version'] = deepstream_version
        if plugin_name:
            context['plugin_name'] = plugin_name
        kwargs['context'] = context
        
        super().__init__(message, **kwargs)


class GStreamerError(PyDSError):
    """Errors specific to GStreamer operations."""
    
    def __init__(
        self, 
        message: str, 
        element_name: Optional[str] = None,
        gst_message: Optional[str] = None,
        **kwargs
    ):
        kwargs.setdefault('category', ErrorCategory.GSTREAMER)
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        kwargs.setdefault('recoverable', True)
        kwargs.setdefault('retry_delay', 2.0)
        
        context = kwargs.get('context', {})
        if element_name:
            context['element_name'] = element_name
        if gst_message:
            context['gst_message'] = gst_message
        kwargs['context'] = context
        
        super().__init__(message, **kwargs)


class VersionCompatibilityError(DeepStreamError):
    """Errors related to DeepStream version compatibility."""
    
    def __init__(
        self, 
        message: str, 
        required_version: Optional[str] = None,
        detected_version: Optional[str] = None,
        **kwargs
    ):
        kwargs.setdefault('severity', ErrorSeverity.CRITICAL)
        kwargs.setdefault('recoverable', False)  # Version issues require manual resolution
        
        context = kwargs.get('context', {})
        if required_version:
            context['required_version'] = required_version
        if detected_version:
            context['detected_version'] = detected_version
        kwargs['context'] = context
        
        super().__init__(message, **kwargs)


class ResourceExhaustionError(PyDSError):
    """Errors related to resource exhaustion (memory, GPU, CPU)."""
    
    def __init__(
        self, 
        message: str, 
        resource_type: str,
        current_usage: Optional[float] = None,
        max_available: Optional[float] = None,
        **kwargs
    ):
        kwargs.setdefault('category', ErrorCategory.SYSTEM)
        kwargs.setdefault('severity', ErrorSeverity.CRITICAL)
        kwargs.setdefault('recoverable', True)
        kwargs.setdefault('retry_delay', 5.0)  # Wait longer for resources to free up
        kwargs.setdefault('max_retries', 2)    # Fewer retries for resource issues
        
        context = kwargs.get('context', {})
        context.update({
            'resource_type': resource_type,
            'current_usage': current_usage,
            'max_available': max_available
        })
        kwargs['context'] = context
        
        super().__init__(message, **kwargs)


class ErrorRecoveryStrategy:
    """
    Implements error recovery strategies with exponential backoff.
    """
    
    def __init__(self):
        self.retry_counts: Dict[str, int] = {}
        self.last_retry_time: Dict[str, float] = {}
    
    def should_retry(self, error: PyDSError, error_key: Optional[str] = None) -> bool:
        """
        Determine if an error should be retried based on its properties and history.
        
        Args:
            error: The error to evaluate
            error_key: Unique key for tracking retry attempts (defaults to error type)
            
        Returns:
            True if the error should be retried
        """
        if not error.recoverable:
            return False
            
        key = error_key or f"{type(error).__name__}:{error.source_id or 'global'}"
        retry_count = self.retry_counts.get(key, 0)
        
        return retry_count < error.max_retries
    
    def get_retry_delay(self, error: PyDSError, error_key: Optional[str] = None) -> float:
        """
        Calculate retry delay using exponential backoff.
        
        Args:
            error: The error being retried
            error_key: Unique key for tracking retry attempts
            
        Returns:
            Delay in seconds before retry
        """
        key = error_key or f"{type(error).__name__}:{error.source_id or 'global'}"
        retry_count = self.retry_counts.get(key, 0)
        
        # Exponential backoff with jitter
        import random
        base_delay = error.retry_delay * (2 ** retry_count)
        jitter = random.uniform(0.1, 0.3) * base_delay
        
        return min(base_delay + jitter, 60.0)  # Cap at 60 seconds
    
    def record_retry(self, error: PyDSError, error_key: Optional[str] = None) -> None:
        """
        Record a retry attempt for tracking.
        
        Args:
            error: The error being retried
            error_key: Unique key for tracking retry attempts
        """
        key = error_key or f"{type(error).__name__}:{error.source_id or 'global'}"
        self.retry_counts[key] = self.retry_counts.get(key, 0) + 1
        self.last_retry_time[key] = time.time()
    
    def reset_retry_count(self, error: PyDSError, error_key: Optional[str] = None) -> None:
        """
        Reset retry count after successful recovery.
        
        Args:
            error: The error that was recovered from
            error_key: Unique key for tracking retry attempts
        """
        key = error_key or f"{type(error).__name__}:{error.source_id or 'global'}"
        self.retry_counts.pop(key, None)
        self.last_retry_time.pop(key, None)
    
    async def retry_with_backoff(
        self, 
        func: callable, 
        error_key: Optional[str] = None,
        *args, 
        **kwargs
    ) -> Any:
        """
        Execute a function with automatic retry and exponential backoff.
        
        Args:
            func: Function to execute
            error_key: Unique key for tracking retry attempts
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Result of successful function execution
            
        Raises:
            PyDSError: If all retry attempts are exhausted
        """
        last_error = None
        
        while True:
            try:
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                
                # Success - reset retry count
                if last_error:
                    self.reset_retry_count(last_error, error_key)
                
                return result
                
            except PyDSError as e:
                last_error = e
                
                if not self.should_retry(e, error_key):
                    raise e
                
                delay = self.get_retry_delay(e, error_key)
                self.record_retry(e, error_key)
                
                # Log retry attempt
                print(f"Retrying after error: {e}. Delay: {delay:.2f}s, Attempt: {self.retry_counts.get(error_key or f'{type(e).__name__}:{e.source_id or "global"}', 0)}")

                await asyncio.sleep(delay)
                
            except Exception as e:
                # Convert non-PyDS errors to PyDSError for consistent handling
                pyds_error = PyDSError(
                    f"Unexpected error: {str(e)}",
                    original_exception=e,
                    recoverable=True
                )
                last_error = pyds_error
                
                if not self.should_retry(pyds_error, error_key):
                    raise pyds_error
                
                delay = self.get_retry_delay(pyds_error, error_key)
                self.record_retry(pyds_error, error_key)
                
                await asyncio.sleep(delay)


# Global error recovery strategy instance
recovery_strategy = ErrorRecoveryStrategy()


def handle_error(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    source_id: Optional[str] = None,
    recoverable: bool = True
) -> PyDSError:
    """
    Convert any exception to a PyDSError with appropriate context.
    
    Args:
        error: Original exception
        context: Additional context information
        source_id: Source identifier where error occurred
        recoverable: Whether the error can be recovered from
        
    Returns:
        PyDSError with full context
    """
    if isinstance(error, PyDSError):
        return error
    
    # Map common exception types to appropriate PyDS errors
    error_message = str(error)
    error_type = type(error).__name__
    
    if "configuration" in error_message.lower() or "config" in error_message.lower():
        return ConfigurationError(
            error_message,
            context=context,
            source_id=source_id,
            original_exception=error,
            recoverable=recoverable
        )
    elif "pipeline" in error_message.lower() or "gstreamer" in error_message.lower():
        return PipelineError(
            error_message,
            context=context,
            source_id=source_id,
            original_exception=error,
            recoverable=recoverable
        )
    elif "deepstream" in error_message.lower() or "pyds" in error_message.lower():
        return DeepStreamError(
            error_message,
            context=context,
            source_id=source_id,
            original_exception=error,
            recoverable=recoverable
        )
    elif "network" in error_message.lower() or "rtsp" in error_message.lower():
        return NetworkSourceError(
            error_message,
            context=context,
            source_id=source_id,
            original_exception=error,
            recoverable=recoverable
        )
    else:
        return PyDSError(
            f"{error_type}: {error_message}",
            context=context,
            source_id=source_id,
            original_exception=error,
            recoverable=recoverable
        )


# Convenience decorators for error handling
def handle_exceptions(
    error_type: Type[PyDSError] = PyDSError,
    source_id: Optional[str] = None,
    recoverable: bool = True
):
    """
    Decorator to automatically handle exceptions and convert them to PyDSError.
    
    Args:
        error_type: Type of PyDSError to create
        source_id: Source identifier for error context
        recoverable: Whether errors should be marked as recoverable
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except PyDSError:
                raise  # Re-raise PyDS errors as-is
            except Exception as e:
                context = {
                    'function': func.__name__,
                    'args': str(args)[:200],  # Truncate long args
                    'kwargs': str(kwargs)[:200]
                }
                raise error_type(
                    f"Error in {func.__name__}: {str(e)}",
                    context=context,
                    source_id=source_id,
                    original_exception=e,
                    recoverable=recoverable
                )
        return wrapper
    return decorator


def async_handle_exceptions(
    error_type: Type[PyDSError] = PyDSError,
    source_id: Optional[str] = None,
    recoverable: bool = True
):
    """
    Async version of the exception handling decorator.
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except PyDSError:
                raise  # Re-raise PyDS errors as-is
            except Exception as e:
                context = {
                    'function': func.__name__,
                    'args': str(args)[:200],  # Truncate long args
                    'kwargs': str(kwargs)[:200]
                }
                raise error_type(
                    f"Error in {func.__name__}: {str(e)}",
                    context=context,
                    source_id=source_id,
                    original_exception=e,
                    recoverable=recoverable
                )
        return wrapper
    return decorator
