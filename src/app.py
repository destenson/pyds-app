"""
Main PyDS Application class with complete lifecycle management and dependency injection.

This module provides the primary application orchestrator that coordinates all subsystems
including pipeline management, detection, alerts, and monitoring with graceful shutdown
and comprehensive error handling.
"""

import asyncio
import signal
import time
import sys
import os
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from pathlib import Path
import threading
import atexit

from .config import AppConfig, load_config, validate_config
from .utils.errors import ApplicationError, handle_error, RecoveryAction
from .utils.logging import get_logger, setup_logging, LogLevel
from .utils.async_utils import get_task_manager, TaskManager, graceful_shutdown

# Import all major components
from .pipeline.manager import get_pipeline_manager, PipelineManager
from .pipeline.sources import get_source_manager, VideoSourceManager
from .pipeline.elements import get_elements_manager, DeepStreamElementsManager
from .detection.engine import get_detection_engine, DetectionEngine
from .alerts.manager import get_alert_manager, AlertManager
from .alerts.handlers import create_default_handlers
from .monitoring.metrics import get_metrics_collector, MetricsCollector
from .monitoring.health import get_health_monitor, HealthMonitor
from .monitoring.profiling import get_performance_profiler, PerformanceProfiler


class ApplicationState(Enum):
    """Application lifecycle states."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class ShutdownReason(Enum):
    """Reasons for application shutdown."""
    USER_REQUEST = "user_request"
    SIGNAL = "signal"
    ERROR = "error"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    HEALTH_CHECK_FAILURE = "health_check_failure"


@dataclass
class ComponentInfo:
    """Information about a system component."""
    name: str
    instance: Any
    start_order: int
    stop_order: int
    critical: bool = True
    dependencies: List[str] = field(default_factory=list)
    started: bool = False
    start_time: Optional[float] = None
    error_count: int = 0
    last_error: Optional[Exception] = None


@dataclass
class ApplicationStatus:
    """Current application status."""
    state: ApplicationState
    uptime_seconds: float
    component_count: int
    running_components: int
    failed_components: int
    total_sources: int
    active_sources: int
    total_detections: int
    total_alerts: int
    memory_usage_mb: float
    cpu_usage_percent: float
    last_error: Optional[str] = None
    shutdown_reason: Optional[ShutdownReason] = None


class PyDSApp:
    """
    Main PyDS Application with comprehensive lifecycle management.
    
    Provides complete orchestration of all subsystems including pipeline management,
    detection engines, alert systems, and monitoring with graceful shutdown and
    comprehensive error handling.
    """
    
    def __init__(self, config_path: Optional[str] = None, config: Optional[AppConfig] = None):
        """
        Initialize PyDS Application.
        
        Args:
            config_path: Path to configuration file
            config: Pre-loaded configuration object
        """
        # Core application state
        self._state = ApplicationState.UNINITIALIZED
        self._start_time = time.time()
        self._shutdown_requested = False
        self._shutdown_reason: Optional[ShutdownReason] = None
        
        # Configuration
        self._config_path = config_path
        self._config = config
        
        # Components registry
        self._components: Dict[str, ComponentInfo] = {}
        self._component_instances: Dict[str, Any] = {}
        
        # Task management
        self._task_manager: Optional[TaskManager] = None
        self._background_tasks: List[asyncio.Task] = []
        
        # Shutdown handling
        self._shutdown_callbacks: List[Callable] = []
        self._cleanup_callbacks: List[Callable] = []
        
        # Thread safety
        self._state_lock = threading.RLock()
        
        # Initialize logger (will be reconfigured after config load)
        self.logger = get_logger(__name__)
        
        # Register signal handlers
        self._register_signal_handlers()
        
        # Register cleanup at exit
        atexit.register(self._emergency_cleanup)
        
        self.logger.info("PyDSApp initialized")
    
    async def initialize(self) -> bool:
        """
        Initialize the application.
        
        Returns:
            True if initialization successful
        """
        try:
            with self._state_lock:
                if self._state != ApplicationState.UNINITIALIZED:
                    self.logger.warning(f"Application already in state: {self._state}")
                    return False
                
                self._state = ApplicationState.INITIALIZING
            
            self.logger.info("Initializing PyDSApp...")
            
            # Load and validate configuration
            if not await self._load_configuration():
                return False
            
            # Setup logging with configuration
            setup_logging(
                level=LogLevel(self._config.logging.level),
                log_file=self._config.logging.file,
                max_file_size_mb=self._config.logging.max_file_size_mb,
                backup_count=self._config.logging.backup_count
            )
            
            # Initialize task manager
            self._task_manager = get_task_manager()
            
            # Register all components
            self._register_components()
            
            # Initialize components in dependency order
            if not await self._initialize_components():
                return False
            
            self.logger.info("PyDSApp initialization completed successfully")
            return True
        
        except Exception as e:
            error = handle_error(e, context={'component': 'app_initialization'})
            self.logger.error(f"Failed to initialize PyDSApp: {error}")
            
            with self._state_lock:
                self._state = ApplicationState.ERROR
            
            return False
    
    async def start(self) -> bool:
        """
        Start the application.
        
        Returns:
            True if started successfully
        """
        try:
            with self._state_lock:
                if self._state != ApplicationState.INITIALIZING:
                    self.logger.error(f"Cannot start from state: {self._state}")
                    return False
                
                self._state = ApplicationState.STARTING
            
            self.logger.info("Starting PyDSApp...")
            
            # Start components in proper order
            if not await self._start_components():
                return False
            
            # Start background tasks
            await self._start_background_tasks()
            
            # Mark as running
            with self._state_lock:
                self._state = ApplicationState.RUNNING
                self._start_time = time.time()
            
            self.logger.info("PyDSApp started successfully")
            
            # Log startup summary
            await self._log_startup_summary()
            
            return True
        
        except Exception as e:
            error = handle_error(e, context={'component': 'app_start'})
            self.logger.error(f"Failed to start PyDSApp: {error}")
            
            with self._state_lock:
                self._state = ApplicationState.ERROR
            
            return False
    
    async def stop(self, reason: ShutdownReason = ShutdownReason.USER_REQUEST) -> bool:
        """
        Stop the application gracefully.
        
        Args:
            reason: Reason for shutdown
            
        Returns:
            True if stopped successfully
        """
        try:
            with self._state_lock:
                if self._state in [ApplicationState.STOPPING, ApplicationState.STOPPED]:
                    self.logger.info("Application already stopping/stopped")
                    return True
                
                self._state = ApplicationState.STOPPING
                self._shutdown_requested = True
                self._shutdown_reason = reason
            
            self.logger.info(f"Stopping PyDSApp (reason: {reason.value})...")
            
            # Execute shutdown callbacks
            await self._execute_shutdown_callbacks()
            
            # Stop background tasks
            await self._stop_background_tasks()
            
            # Stop components in reverse order
            await self._stop_components()
            
            # Cleanup resources
            await self._cleanup_resources()
            
            # Mark as stopped
            with self._state_lock:
                self._state = ApplicationState.STOPPED
            
            # Calculate uptime
            uptime = time.time() - self._start_time
            self.logger.info(f"PyDSApp stopped successfully (uptime: {uptime:.1f}s)")
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            return False
    
    async def run(self) -> int:
        """
        Run the application until shutdown.
        
        Returns:
            Exit code (0 for success, non-zero for error)
        """
        try:
            # Initialize application
            if not await self.initialize():
                self.logger.error("Failed to initialize application")
                return 1
            
            # Start application
            if not await self.start():
                self.logger.error("Failed to start application")
                return 1
            
            # Wait for shutdown signal
            try:
                await self._wait_for_shutdown()
            except KeyboardInterrupt:
                self.logger.info("Received keyboard interrupt")
                self._shutdown_reason = ShutdownReason.SIGNAL
            
            # Stop application
            if not await self.stop(self._shutdown_reason or ShutdownReason.USER_REQUEST):
                self.logger.error("Failed to stop application gracefully")
                return 1
            
            return 0
        
        except Exception as e:
            self.logger.error(f"Unhandled error in application: {e}")
            return 1
    
    async def _load_configuration(self) -> bool:
        """Load and validate configuration."""
        try:
            if self._config is None:
                if self._config_path:
                    self._config = load_config(self._config_path)
                else:
                    # Try default locations
                    for config_file in ['config/config.yaml', 'config.yaml', 'config/default.yaml']:
                        if Path(config_file).exists():
                            self._config = load_config(config_file)
                            self._config_path = config_file
                            break
                    
                    if self._config is None:
                        raise ApplicationError("No configuration file found")
            
            # Validate configuration
            validation_errors = validate_config(self._config)
            if validation_errors:
                for error in validation_errors:
                    self.logger.error(f"Configuration error: {error}")
                return False
            
            self.logger.info(f"Loaded configuration from: {self._config_path or 'programmatic'}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            return False
    
    def _register_components(self):
        """Register all system components."""
        # Define component startup/shutdown order
        # Lower numbers start first, higher numbers stop first
        
        self._components = {
            # Core infrastructure (start first)
            'task_manager': ComponentInfo(
                name='task_manager',
                instance=None,
                start_order=1,
                stop_order=10,
                critical=True
            ),
            
            # Monitoring components
            'metrics_collector': ComponentInfo(
                name='metrics_collector',
                instance=None,
                start_order=2,
                stop_order=9,
                critical=False
            ),
            
            'health_monitor': ComponentInfo(
                name='health_monitor',
                instance=None,
                start_order=3,
                stop_order=8,
                critical=False
            ),
            
            'performance_profiler': ComponentInfo(
                name='performance_profiler',
                instance=None,
                start_order=4,
                stop_order=7,
                critical=False
            ),
            
            # Pipeline components
            'elements_manager': ComponentInfo(
                name='elements_manager',
                instance=None,
                start_order=5,
                stop_order=6,
                critical=True
            ),
            
            'source_manager': ComponentInfo(
                name='source_manager',
                instance=None,
                start_order=6,
                stop_order=5,
                critical=True,
                dependencies=['elements_manager']
            ),
            
            'pipeline_manager': ComponentInfo(
                name='pipeline_manager',
                instance=None,
                start_order=7,
                stop_order=4,
                critical=True,
                dependencies=['elements_manager', 'source_manager']
            ),
            
            # Detection system
            'detection_engine': ComponentInfo(
                name='detection_engine',
                instance=None,
                start_order=8,
                stop_order=3,
                critical=True,
                dependencies=['pipeline_manager']
            ),
            
            # Alert system (start last, depends on everything)
            'alert_manager': ComponentInfo(
                name='alert_manager',
                instance=None,
                start_order=9,
                stop_order=2,
                critical=True,
                dependencies=['detection_engine']
            )
        }
    
    async def _initialize_components(self) -> bool:
        """Initialize all components."""
        try:
            # Create component instances
            component_instances = {
                'task_manager': self._task_manager,
                'metrics_collector': get_metrics_collector(self._config),
                'health_monitor': get_health_monitor(self._config),
                'performance_profiler': get_performance_profiler(self._config),
                'elements_manager': get_elements_manager(self._config),
                'source_manager': get_source_manager(self._config),
                'pipeline_manager': get_pipeline_manager(self._config),
                'detection_engine': get_detection_engine(self._config),
                'alert_manager': get_alert_manager(self._config)
            }
            
            # Store instances in components
            for name, instance in component_instances.items():
                if name in self._components:
                    self._components[name].instance = instance
                    self._component_instances[name] = instance
            
            # Initialize alert handlers
            await self._initialize_alert_handlers()
            
            self.logger.info(f"Initialized {len(self._components)} components")
            return True
        
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            return False
    
    async def _initialize_alert_handlers(self):
        """Initialize alert handlers."""
        try:
            alert_manager = self._component_instances.get('alert_manager')
            if alert_manager:
                # Create and register default handlers
                handlers = await create_default_handlers(self._config)
                
                for handler in handlers:
                    success = alert_manager.register_handler(handler)
                    if success:
                        self.logger.info(f"Registered alert handler: {handler.name}")
                    else:
                        self.logger.warning(f"Failed to register alert handler: {handler.name}")
        
        except Exception as e:
            self.logger.error(f"Error initializing alert handlers: {e}")
    
    async def _start_components(self) -> bool:
        """Start all components in proper order."""
        try:
            # Sort components by start order
            sorted_components = sorted(
                self._components.values(),
                key=lambda c: c.start_order
            )
            
            for component in sorted_components:
                if not await self._start_component(component):
                    if component.critical:
                        self.logger.error(f"Failed to start critical component: {component.name}")
                        return False
                    else:
                        self.logger.warning(f"Failed to start non-critical component: {component.name}")
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error starting components: {e}")
            return False
    
    async def _start_component(self, component: ComponentInfo) -> bool:
        """Start a single component."""
        try:
            self.logger.info(f"Starting component: {component.name}")
            
            # Check dependencies
            for dep_name in component.dependencies:
                if dep_name in self._components:
                    dep_component = self._components[dep_name]
                    if not dep_component.started:
                        self.logger.error(f"Dependency {dep_name} not started for {component.name}")
                        return False
            
            # Start the component
            instance = component.instance
            if instance and hasattr(instance, 'start'):
                start_time = time.time()
                success = await instance.start()
                
                if success:
                    component.started = True
                    component.start_time = start_time
                    self.logger.info(f"Started component: {component.name}")
                    return True
                else:
                    component.error_count += 1
                    self.logger.error(f"Component start failed: {component.name}")
                    return False
            else:
                # Component doesn't need explicit start
                component.started = True
                component.start_time = time.time()
                return True
        
        except Exception as e:
            component.error_count += 1
            component.last_error = e
            self.logger.error(f"Error starting component {component.name}: {e}")
            return False
    
    async def _stop_components(self):
        """Stop all components in reverse order."""
        try:
            # Sort components by stop order (reverse of start order)
            sorted_components = sorted(
                self._components.values(),
                key=lambda c: c.stop_order,
                reverse=True
            )
            
            for component in sorted_components:
                if component.started:
                    await self._stop_component(component)
        
        except Exception as e:
            self.logger.error(f"Error stopping components: {e}")
    
    async def _stop_component(self, component: ComponentInfo):
        """Stop a single component."""
        try:
            self.logger.info(f"Stopping component: {component.name}")
            
            instance = component.instance
            if instance and hasattr(instance, 'stop'):
                success = await instance.stop()
                
                if success:
                    self.logger.info(f"Stopped component: {component.name}")
                else:
                    self.logger.warning(f"Component stop failed: {component.name}")
            
            component.started = False
        
        except Exception as e:
            self.logger.error(f"Error stopping component {component.name}: {e}")
    
    async def _start_background_tasks(self):
        """Start background monitoring and maintenance tasks."""
        try:
            # Application status monitoring
            status_task = asyncio.create_task(self._status_monitoring_loop())
            self._background_tasks.append(status_task)
            
            # Health monitoring integration
            health_task = asyncio.create_task(self._health_monitoring_loop())
            self._background_tasks.append(health_task)
            
            # Resource cleanup task
            cleanup_task = asyncio.create_task(self._periodic_cleanup_loop())
            self._background_tasks.append(cleanup_task)
            
            self.logger.info(f"Started {len(self._background_tasks)} background tasks")
        
        except Exception as e:
            self.logger.error(f"Error starting background tasks: {e}")
    
    async def _stop_background_tasks(self):
        """Stop all background tasks."""
        try:
            # Cancel all tasks
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for cancellation
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
            self._background_tasks.clear()
            self.logger.info("Stopped all background tasks")
        
        except Exception as e:
            self.logger.error(f"Error stopping background tasks: {e}")
    
    async def _status_monitoring_loop(self):
        """Background task for monitoring application status."""
        try:
            while not self._shutdown_requested:
                try:
                    # Update component health
                    await self._update_component_health()
                    
                    # Log status periodically
                    if int(time.time()) % 300 == 0:  # Every 5 minutes
                        status = self.get_status()
                        self.logger.info(f"Application Status: {status.state.value}, "
                                       f"Components: {status.running_components}/{status.component_count}, "
                                       f"Memory: {status.memory_usage_mb:.1f}MB")
                    
                    await asyncio.sleep(30)  # Check every 30 seconds
                
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Error in status monitoring: {e}")
                    await asyncio.sleep(10)
        
        except asyncio.CancelledError:
            pass
    
    async def _health_monitoring_loop(self):
        """Background task for health monitoring integration."""
        try:
            health_monitor = self._component_instances.get('health_monitor')
            if not health_monitor:
                return
            
            while not self._shutdown_requested:
                try:
                    # Get system health
                    system_health = health_monitor.get_system_health()
                    
                    # Check for critical issues
                    if system_health.overall_status.value == 'critical':
                        critical_components = system_health.critical_components
                        self.logger.critical(f"Critical health issues detected: {critical_components}")
                        
                        # Consider emergency shutdown if too many critical issues
                        if len(critical_components) >= 3:
                            self.logger.critical("Multiple critical health issues - initiating emergency shutdown")
                            await self.stop(ShutdownReason.HEALTH_CHECK_FAILURE)
                            break
                    
                    await asyncio.sleep(60)  # Check every minute
                
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Error in health monitoring: {e}")
                    await asyncio.sleep(30)
        
        except asyncio.CancelledError:
            pass
    
    async def _periodic_cleanup_loop(self):
        """Background task for periodic cleanup."""
        try:
            while not self._shutdown_requested:
                try:
                    # Force garbage collection
                    import gc
                    gc.collect()
                    
                    # Execute cleanup callbacks
                    for callback in self._cleanup_callbacks:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback()
                            else:
                                callback()
                        except Exception as e:
                            self.logger.error(f"Error in cleanup callback: {e}")
                    
                    await asyncio.sleep(600)  # Every 10 minutes
                
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Error in periodic cleanup: {e}")
                    await asyncio.sleep(300)
        
        except asyncio.CancelledError:
            pass
    
    async def _update_component_health(self):
        """Update health status of all components."""
        try:
            for component in self._components.values():
                # Check if component is responsive
                if component.started and component.instance:
                    try:
                        # Check if component has a health check method
                        if hasattr(component.instance, 'get_statistics'):
                            stats = component.instance.get_statistics()
                            # Component is responsive if it can return stats
                        
                        # Reset error count on successful check
                        if component.error_count > 0:
                            component.error_count = max(0, component.error_count - 1)
                    
                    except Exception as e:
                        component.error_count += 1
                        component.last_error = e
                        
                        if component.error_count > 5:
                            self.logger.error(f"Component {component.name} appears unhealthy: {e}")
        
        except Exception as e:
            self.logger.error(f"Error updating component health: {e}")
    
    async def _wait_for_shutdown(self):
        """Wait for shutdown signal."""
        try:
            while not self._shutdown_requested:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            self._shutdown_requested = True
            self._shutdown_reason = ShutdownReason.SIGNAL
    
    async def _execute_shutdown_callbacks(self):
        """Execute registered shutdown callbacks."""
        for callback in self._shutdown_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                self.logger.error(f"Error in shutdown callback: {e}")
    
    async def _cleanup_resources(self):
        """Cleanup application resources."""
        try:
            # Close task manager
            if self._task_manager:
                await self._task_manager.cleanup()
            
            # Clear component references
            self._component_instances.clear()
            
            self.logger.info("Cleaned up application resources")
        
        except Exception as e:
            self.logger.error(f"Error cleaning up resources: {e}")
    
    def _register_signal_handlers(self):
        """Register signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}")
            self._shutdown_requested = True
            self._shutdown_reason = ShutdownReason.SIGNAL
        
        try:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            # Windows doesn't have SIGHUP
            if hasattr(signal, 'SIGHUP'):
                signal.signal(signal.SIGHUP, signal_handler)
        
        except Exception as e:
            self.logger.warning(f"Could not register signal handlers: {e}")
    
    def _emergency_cleanup(self):
        """Emergency cleanup called on exit."""
        try:
            if self._state == ApplicationState.RUNNING:
                self.logger.warning("Emergency cleanup - application did not shutdown cleanly")
                
                # Try to stop components synchronously
                for component in self._components.values():
                    if component.started and component.instance:
                        try:
                            if hasattr(component.instance, 'stop'):
                                # Try sync stop first
                                if not asyncio.iscoroutinefunction(component.instance.stop):
                                    component.instance.stop()
                        except Exception:
                            pass
        except Exception:
            pass  # Silent fail in emergency cleanup
    
    async def _log_startup_summary(self):
        """Log application startup summary."""
        try:
            status = self.get_status()
            
            summary = f"""
=== PyDS Application Started ===
State: {status.state.value}
Components: {status.running_components}/{status.component_count} running
Configuration: {self._config_path or 'programmatic'}
Memory Usage: {status.memory_usage_mb:.1f} MB
CPU Usage: {status.cpu_usage_percent:.1f}%
Sources Configured: {status.total_sources}
Active Sources: {status.active_sources}
=====================================
"""
            self.logger.info(summary.strip())
        
        except Exception as e:
            self.logger.error(f"Error logging startup summary: {e}")
    
    # Public API
    
    def get_status(self) -> ApplicationStatus:
        """Get current application status."""
        try:
            import psutil
            
            with self._state_lock:
                # Count component states
                running_components = sum(1 for c in self._components.values() if c.started)
                failed_components = sum(1 for c in self._components.values() if c.error_count > 0)
                
                # Get resource usage
                process = psutil.Process()
                memory_usage_mb = process.memory_info().rss / (1024 * 1024)
                cpu_usage_percent = process.cpu_percent()
                
                # Get source information
                source_manager = self._component_instances.get('source_manager')
                total_sources = 0
                active_sources = 0
                if source_manager and hasattr(source_manager, 'get_source_count'):
                    try:
                        total_sources = source_manager.get_source_count()
                        active_sources = source_manager.get_active_source_count()
                    except:
                        pass
                
                # Get detection/alert counts
                total_detections = 0
                total_alerts = 0
                
                detection_engine = self._component_instances.get('detection_engine')
                if detection_engine and hasattr(detection_engine, 'get_statistics'):
                    try:
                        stats = detection_engine.get_statistics()
                        total_detections = getattr(stats, 'total_detections', 0)
                    except:
                        pass
                
                alert_manager = self._component_instances.get('alert_manager')
                if alert_manager and hasattr(alert_manager, 'get_statistics'):
                    try:
                        stats = alert_manager.get_statistics()
                        total_alerts = stats.get('total_alerts_received', 0)
                    except:
                        pass
                
                return ApplicationStatus(
                    state=self._state,
                    uptime_seconds=time.time() - self._start_time,
                    component_count=len(self._components),
                    running_components=running_components,
                    failed_components=failed_components,
                    total_sources=total_sources,
                    active_sources=active_sources,
                    total_detections=total_detections,
                    total_alerts=total_alerts,
                    memory_usage_mb=memory_usage_mb,
                    cpu_usage_percent=cpu_usage_percent,
                    shutdown_reason=self._shutdown_reason
                )
        
        except Exception as e:
            self.logger.error(f"Error getting application status: {e}")
            return ApplicationStatus(
                state=self._state,
                uptime_seconds=0,
                component_count=0,
                running_components=0,
                failed_components=0,
                total_sources=0,
                active_sources=0,
                total_detections=0,
                total_alerts=0,
                memory_usage_mb=0,
                cpu_usage_percent=0
            )
    
    def get_component_status(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed status of all components."""
        component_status = {}
        
        for name, component in self._components.items():
            component_status[name] = {
                'name': component.name,
                'started': component.started,
                'start_time': component.start_time,
                'uptime_seconds': time.time() - component.start_time if component.start_time else 0,
                'error_count': component.error_count,
                'last_error': str(component.last_error) if component.last_error else None,
                'critical': component.critical,
                'dependencies': component.dependencies
            }
        
        return component_status
    
    def get_config(self) -> Optional[AppConfig]:
        """Get application configuration."""
        return self._config
    
    def add_shutdown_callback(self, callback: Callable):
        """Add callback to be executed during shutdown."""
        self._shutdown_callbacks.append(callback)
    
    def add_cleanup_callback(self, callback: Callable):
        """Add callback to be executed during periodic cleanup."""
        self._cleanup_callbacks.append(callback)
    
    def request_shutdown(self, reason: ShutdownReason = ShutdownReason.USER_REQUEST):
        """Request application shutdown."""
        self._shutdown_requested = True
        self._shutdown_reason = reason
        self.logger.info(f"Shutdown requested: {reason.value}")
    
    def is_running(self) -> bool:
        """Check if application is running."""
        with self._state_lock:
            return self._state == ApplicationState.RUNNING
    
    def is_healthy(self) -> bool:
        """Check if application is healthy."""
        with self._state_lock:
            if self._state != ApplicationState.RUNNING:
                return False
            
            # Check if critical components are running
            critical_components = [c for c in self._components.values() if c.critical]
            failed_critical = [c for c in critical_components if not c.started or c.error_count > 3]
            
            return len(failed_critical) == 0
    
    async def restart_component(self, component_name: str) -> bool:
        """Restart a specific component."""
        try:
            if component_name not in self._components:
                self.logger.error(f"Component not found: {component_name}")
                return False
            
            component = self._components[component_name]
            
            self.logger.info(f"Restarting component: {component_name}")
            
            # Stop component
            if component.started:
                await self._stop_component(component)
            
            # Start component
            success = await self._start_component(component)
            
            if success:
                self.logger.info(f"Successfully restarted component: {component_name}")
            else:
                self.logger.error(f"Failed to restart component: {component_name}")
            
            return success
        
        except Exception as e:
            self.logger.error(f"Error restarting component {component_name}: {e}")
            return False


# Global application instance
_global_app: Optional[PyDSApp] = None


def get_app() -> Optional[PyDSApp]:
    """Get global application instance."""
    return _global_app


def set_app(app: PyDSApp):
    """Set global application instance."""
    global _global_app
    _global_app = app


async def create_app(config_path: Optional[str] = None, config: Optional[AppConfig] = None) -> PyDSApp:
    """
    Create and initialize PyDS application.
    
    Args:
        config_path: Path to configuration file
        config: Pre-loaded configuration
        
    Returns:
        Initialized PyDSApp instance
    """
    app = PyDSApp(config_path=config_path, config=config)
    
    # Initialize the application
    success = await app.initialize()
    if not success:
        raise ApplicationError("Failed to initialize PyDSApp")
    
    # Set as global instance
    set_app(app)
    
    return app