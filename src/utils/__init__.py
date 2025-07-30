"""
Utilities package for the DeepStream inference system.

This package provides core utilities including error handling, logging,
DeepStream compatibility, and async utilities.
"""

from .errors import (
    PyDSError,
    ConfigurationError,
    PipelineError,
    DetectionError,
    AlertError,
    DeepStreamError,
    VersionCompatibilityError,
    ErrorRecoveryStrategy,
    handle_error
)

from .logging import (
    setup_logging,
    get_logger,
    update_metrics,
    performance_context,
    LoggingConfig
)

from .deepstream import (
    get_deepstream_info,
    get_deepstream_api,
    check_version_compatibility,
    is_gpu_available
)

from .async_utils import (
    ThreadSafeAsyncQueue,
    AsyncTaskManager,
    GracefulShutdownManager,
    get_shutdown_manager,
    get_task_manager,
    create_managed_task
)

__all__ = [
    # Error handling
    'PyDSError',
    'ConfigurationError', 
    'PipelineError',
    'DetectionError',
    'AlertError',
    'DeepStreamError',
    'VersionCompatibilityError',
    'ErrorRecoveryStrategy',
    'handle_error',
    
    # Logging
    'setup_logging',
    'get_logger',
    'update_metrics',
    'performance_context',
    'LoggingConfig',
    
    # DeepStream
    'get_deepstream_info',
    'get_deepstream_api',
    'check_version_compatibility',
    'is_gpu_available',
    
    # Async utilities
    'ThreadSafeAsyncQueue',
    'AsyncTaskManager', 
    'GracefulShutdownManager',
    'get_shutdown_manager',
    'get_task_manager',
    'create_managed_task'
]