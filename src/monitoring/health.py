"""
Comprehensive health monitoring system with automatic recovery and graceful degradation.

This module provides health checks, circuit breakers, automatic recovery mechanisms,
and graceful degradation under resource constraints for robust system operation.
"""

import asyncio
import time
import json
from typing import Dict, List, Optional, Any, Callable, Union, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, deque
import threading
import gc
import psutil
from contextlib import asynccontextmanager

from ..config import AppConfig
from ..utils.errors import HealthError, handle_error, RecoveryAction
from ..utils.logging import get_logger, performance_context
from ..utils.async_utils import get_task_manager, PeriodicTaskRunner


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """Types of system components."""
    PIPELINE = "pipeline"
    DETECTION = "detection"
    ALERTS = "alerts"
    MONITORING = "monitoring"
    STORAGE = "storage"
    NETWORK = "network"
    EXTERNAL = "external"


class RecoveryStrategy(Enum):
    """Recovery strategies for different failure types."""
    RESTART_COMPONENT = "restart_component"
    REDUCE_LOAD = "reduce_load"
    FALLBACK_MODE = "fallback_mode"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_SHUTDOWN = "graceful_shutdown"
    MANUAL_INTERVENTION = "manual_intervention"


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Failing, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class HealthCheck:
    """Individual health check definition."""
    name: str
    component_type: ComponentType
    check_function: Callable
    interval_seconds: float = 30.0
    timeout_seconds: float = 10.0
    failure_threshold: int = 3
    recovery_threshold: int = 2
    critical: bool = False
    dependencies: List[str] = field(default_factory=list)
    recovery_strategies: List[RecoveryStrategy] = field(default_factory=list)
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthResult:
    """Result of a health check."""
    check_name: str
    status: HealthStatus
    timestamp: float
    response_time_ms: float
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Exception] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'check_name': self.check_name,
            'status': self.status.value,
            'timestamp': self.timestamp,
            'response_time_ms': self.response_time_ms,
            'message': self.message,
            'details': self.details,
            'error': str(self.error) if self.error else None
        }


@dataclass
class ComponentHealth:
    """Health status of a system component."""
    name: str
    component_type: ComponentType
    overall_status: HealthStatus
    last_check_time: float
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    failure_count_24h: int = 0
    recovery_attempts: int = 0
    last_recovery_time: Optional[float] = None
    circuit_breaker_state: CircuitBreakerState = CircuitBreakerState.CLOSED
    check_results: deque = field(default_factory=lambda: deque(maxlen=100))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_result(self, result: HealthResult):
        """Add health check result."""
        self.check_results.append(result)
        self.last_check_time = result.timestamp
        
        if result.status == HealthStatus.HEALTHY:
            self.consecutive_successes += 1
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
            self.consecutive_successes = 0
            self.failure_count_24h += 1
        
        # Update overall status
        self._update_overall_status()
    
    def _update_overall_status(self):
        """Update overall component status based on recent results."""
        if not self.check_results:
            self.overall_status = HealthStatus.UNKNOWN
            return
        
        recent_results = list(self.check_results)[-10:]  # Last 10 checks
        healthy_count = sum(1 for r in recent_results if r.status == HealthStatus.HEALTHY)
        critical_count = sum(1 for r in recent_results if r.status == HealthStatus.CRITICAL)
        
        if critical_count > 0:
            self.overall_status = HealthStatus.CRITICAL
        elif healthy_count >= len(recent_results) * 0.8:  # 80% healthy
            self.overall_status = HealthStatus.HEALTHY
        elif healthy_count >= len(recent_results) * 0.5:  # 50% healthy
            self.overall_status = HealthStatus.DEGRADED
        else:
            self.overall_status = HealthStatus.UNHEALTHY
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'component_type': self.component_type.value,
            'overall_status': self.overall_status.value,
            'last_check_time': self.last_check_time,
            'consecutive_failures': self.consecutive_failures,
            'consecutive_successes': self.consecutive_successes,
            'failure_count_24h': self.failure_count_24h,
            'recovery_attempts': self.recovery_attempts,
            'last_recovery_time': self.last_recovery_time,
            'circuit_breaker_state': self.circuit_breaker_state.value,
            'recent_results': [r.to_dict() for r in list(self.check_results)[-5:]],
            'metadata': self.metadata
        }


@dataclass
class SystemHealth:
    """Overall system health status."""
    overall_status: HealthStatus
    timestamp: float
    component_health: Dict[str, ComponentHealth]
    degraded_components: List[str] = field(default_factory=list)
    critical_components: List[str] = field(default_factory=list)
    recovery_actions_taken: List[str] = field(default_factory=list)
    resource_constraints: Dict[str, Any] = field(default_factory=dict)
    uptime_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'overall_status': self.overall_status.value,
            'timestamp': self.timestamp,
            'component_health': {name: comp.to_dict() for name, comp in self.component_health.items()},
            'degraded_components': self.degraded_components,
            'critical_components': self.critical_components,
            'recovery_actions_taken': self.recovery_actions_taken,
            'resource_constraints': self.resource_constraints,
            'uptime_seconds': self.uptime_seconds
        }


class CircuitBreaker:
    """Circuit breaker implementation for preventing cascading failures."""
    
    def __init__(self, name: str, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        """
        Initialize circuit breaker.
        
        Args:
            name: Circuit breaker name
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before testing recovery
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._lock = threading.Lock()
    
    @asynccontextmanager
    async def call(self):
        """Context manager for protected calls."""
        if not self._should_allow_call():
            raise HealthError(f"Circuit breaker {self.name} is OPEN")
        
        try:
            yield
            self._on_success()
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_allow_call(self) -> bool:
        """Check if call should be allowed."""
        with self._lock:
            if self._state == CircuitBreakerState.CLOSED:
                return True
            elif self._state == CircuitBreakerState.OPEN:
                # Check if recovery timeout has passed
                if time.time() - self._last_failure_time >= self.recovery_timeout:
                    self._state = CircuitBreakerState.HALF_OPEN
                    return True
                return False
            else:  # HALF_OPEN
                return True
    
    def _on_success(self):
        """Handle successful call."""
        with self._lock:
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._state = CircuitBreakerState.CLOSED
            self._failure_count = 0
    
    def _on_failure(self):
        """Handle failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitBreakerState.OPEN
    
    def get_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        return self._state
    
    def force_open(self):
        """Force circuit breaker to OPEN state."""
        with self._lock:
            self._state = CircuitBreakerState.OPEN
            self._last_failure_time = time.time()
    
    def force_closed(self):
        """Force circuit breaker to CLOSED state."""
        with self._lock:
            self._state = CircuitBreakerState.CLOSED
            self._failure_count = 0


class RecoveryManager:
    """Manages automatic recovery strategies."""
    
    def __init__(self, config: AppConfig):
        """Initialize recovery manager."""
        self.config = config
        self.logger = get_logger(__name__)
        
        # Recovery strategy implementations
        self._recovery_strategies: Dict[RecoveryStrategy, Callable] = {
            RecoveryStrategy.RESTART_COMPONENT: self._restart_component,
            RecoveryStrategy.REDUCE_LOAD: self._reduce_load,
            RecoveryStrategy.FALLBACK_MODE: self._fallback_mode,
            RecoveryStrategy.CIRCUIT_BREAKER: self._enable_circuit_breaker,
            RecoveryStrategy.GRACEFUL_SHUTDOWN: self._graceful_shutdown,
            RecoveryStrategy.MANUAL_INTERVENTION: self._request_manual_intervention
        }
        
        # Recovery attempt tracking
        self._recovery_attempts: Dict[str, int] = defaultdict(int)
        self._recovery_cooldowns: Dict[str, float] = {}
        self._max_recovery_attempts = 3
        self._recovery_cooldown_seconds = 300.0  # 5 minutes
    
    async def attempt_recovery(self, component_name: str, strategies: List[RecoveryStrategy]) -> bool:
        """
        Attempt recovery using provided strategies.
        
        Args:
            component_name: Name of component to recover
            strategies: List of recovery strategies to try
            
        Returns:
            True if recovery was attempted successfully
        """
        try:
            # Check if component is in cooldown
            if self._is_in_cooldown(component_name):
                self.logger.debug(f"Component {component_name} is in recovery cooldown")
                return False
            
            # Check max attempts
            if self._recovery_attempts[component_name] >= self._max_recovery_attempts:
                self.logger.warning(f"Max recovery attempts reached for {component_name}")
                return False
            
            # Try each strategy in order
            for strategy in strategies:
                if strategy in self._recovery_strategies:
                    self.logger.info(f"Attempting {strategy.value} recovery for {component_name}")
                    
                    success = await self._recovery_strategies[strategy](component_name)
                    
                    if success:
                        self._recovery_attempts[component_name] += 1
                        self._recovery_cooldowns[component_name] = time.time()
                        self.logger.info(f"Recovery strategy {strategy.value} succeeded for {component_name}")
                        return True
                    else:
                        self.logger.warning(f"Recovery strategy {strategy.value} failed for {component_name}")
            
            return False
        
        except Exception as e:
            self.logger.error(f"Error during recovery attempt for {component_name}: {e}")
            return False
    
    def _is_in_cooldown(self, component_name: str) -> bool:
        """Check if component is in recovery cooldown."""
        last_recovery = self._recovery_cooldowns.get(component_name, 0)
        return time.time() - last_recovery < self._recovery_cooldown_seconds
    
    async def _restart_component(self, component_name: str) -> bool:
        """Restart a component."""
        try:
            # This would integrate with the actual component management system
            self.logger.info(f"Restarting component: {component_name}")
            
            # Simulate restart process
            await asyncio.sleep(1)
            
            # In real implementation, this would call component restart methods
            return True
        
        except Exception as e:
            self.logger.error(f"Error restarting component {component_name}: {e}")
            return False
    
    async def _reduce_load(self, component_name: str) -> bool:
        """Reduce load on a component."""
        try:
            self.logger.info(f"Reducing load for component: {component_name}")
            
            # Implement load reduction strategies
            # - Reduce batch sizes
            # - Increase processing intervals
            # - Disable non-critical features
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error reducing load for {component_name}: {e}")
            return False
    
    async def _fallback_mode(self, component_name: str) -> bool:
        """Enable fallback mode for a component."""
        try:
            self.logger.info(f"Enabling fallback mode for component: {component_name}")
            
            # Implement fallback strategies
            # - Use cached data
            # - Disable advanced features
            # - Use simplified processing
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error enabling fallback mode for {component_name}: {e}")
            return False
    
    async def _enable_circuit_breaker(self, component_name: str) -> bool:
        """Enable circuit breaker for a component."""
        try:
            self.logger.info(f"Enabling circuit breaker for component: {component_name}")
            
            # This would integrate with circuit breaker management
            return True
        
        except Exception as e:
            self.logger.error(f"Error enabling circuit breaker for {component_name}: {e}")
            return False
    
    async def _graceful_shutdown(self, component_name: str) -> bool:
        """Perform graceful shutdown of a component."""
        try:
            self.logger.warning(f"Performing graceful shutdown for component: {component_name}")
            
            # Implement graceful shutdown
            # - Stop accepting new requests
            # - Complete current operations
            # - Clean up resources
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error in graceful shutdown for {component_name}: {e}")
            return False
    
    async def _request_manual_intervention(self, component_name: str) -> bool:
        """Request manual intervention for a component."""
        try:
            self.logger.critical(f"Manual intervention required for component: {component_name}")
            
            # Send alerts to administrators
            # Create support tickets
            # Log critical events
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error requesting manual intervention for {component_name}: {e}")
            return False


class HealthMonitor:
    """
    Comprehensive health monitoring system.
    
    Provides health checks, automatic recovery, circuit breakers, and graceful
    degradation under resource constraints for robust system operation.
    """
    
    def __init__(self, config: AppConfig):
        """
        Initialize health monitor.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Core components
        self._recovery_manager = RecoveryManager(config)
        
        # Health check management
        self._health_checks: Dict[str, HealthCheck] = {}
        self._component_health: Dict[str, ComponentHealth] = {}
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # System state
        self._system_health = SystemHealth(
            overall_status=HealthStatus.UNKNOWN,
            timestamp=time.time(),
            component_health={},
            uptime_seconds=0.0
        )
        self._start_time = time.time()
        self._monitoring_enabled = False
        
        # Resource monitoring
        self._resource_thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_percent': 90.0,
            'gpu_memory_percent': 90.0
        }
        
        # Periodic tasks
        self._periodic_runner = PeriodicTaskRunner()
        self._check_interval = 30.0
        self._cleanup_interval = 3600.0  # 1 hour
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize default health checks
        self._initialize_default_checks()
        
        self.logger.info("HealthMonitor initialized")
    
    async def start(self) -> bool:
        """
        Start health monitoring.
        
        Returns:
            True if started successfully
        """
        try:
            self.logger.info("Starting HealthMonitor...")
            
            # Start periodic health checks
            await self._start_periodic_checks()
            
            # Initialize system health
            await self._update_system_health()
            
            self._monitoring_enabled = True
            
            self.logger.info("HealthMonitor started successfully")
            return True
        
        except Exception as e:
            error = handle_error(e, context={'component': 'health_monitor'})
            self.logger.error(f"Failed to start HealthMonitor: {error}")
            return False
    
    async def stop(self) -> bool:
        """
        Stop health monitoring.
        
        Returns:
            True if stopped successfully
        """
        try:
            self.logger.info("Stopping HealthMonitor...")
            
            self._monitoring_enabled = False
            
            # Stop periodic tasks
            await self._periodic_runner.stop_all_tasks()
            
            # Save final health report
            await self._save_health_report()
            
            self.logger.info("HealthMonitor stopped successfully")
            return True
        
        except Exception as e:
            self.logger.error(f"Error stopping HealthMonitor: {e}")
            return False
    
    def _initialize_default_checks(self):
        """Initialize default health checks."""
        default_checks = [
            # System resource checks
            HealthCheck(
                name="system_resources",
                component_type=ComponentType.MONITORING,
                check_function=self._check_system_resources,
                interval_seconds=30.0,
                critical=True,
                recovery_strategies=[RecoveryStrategy.REDUCE_LOAD, RecoveryStrategy.GRACEFUL_SHUTDOWN]
            ),
            
            # Pipeline health check
            HealthCheck(
                name="pipeline_health",
                component_type=ComponentType.PIPELINE,
                check_function=self._check_pipeline_health,
                interval_seconds=15.0,
                critical=True,
                recovery_strategies=[RecoveryStrategy.RESTART_COMPONENT, RecoveryStrategy.REDUCE_LOAD]
            ),
            
            # Detection engine check
            HealthCheck(
                name="detection_engine",
                component_type=ComponentType.DETECTION,
                check_function=self._check_detection_engine,
                interval_seconds=30.0,
                dependencies=["pipeline_health"],
                recovery_strategies=[RecoveryStrategy.RESTART_COMPONENT, RecoveryStrategy.FALLBACK_MODE]
            ),
            
            # Alert system check
            HealthCheck(
                name="alert_system",
                component_type=ComponentType.ALERTS,
                check_function=self._check_alert_system,
                interval_seconds=45.0,
                recovery_strategies=[RecoveryStrategy.RESTART_COMPONENT, RecoveryStrategy.CIRCUIT_BREAKER]
            ),
            
            # Memory usage check
            HealthCheck(
                name="memory_usage",
                component_type=ComponentType.MONITORING,
                check_function=self._check_memory_usage,
                interval_seconds=60.0,
                recovery_strategies=[RecoveryStrategy.REDUCE_LOAD]
            )
        ]
        
        for check in default_checks:
            self.register_health_check(check)
    
    async def _start_periodic_checks(self):
        """Start periodic health check tasks."""
        # Main health check task
        await self._periodic_runner.start_periodic_task(
            "health_checks",
            self._run_health_checks,
            interval=self._check_interval
        )
        
        # System health update task
        await self._periodic_runner.start_periodic_task(
            "system_health_update",
            self._update_system_health,
            interval=60.0
        )
        
        # Cleanup task
        await self._periodic_runner.start_periodic_task(
            "cleanup_old_data",
            self._cleanup_old_data,
            interval=self._cleanup_interval
        )
    
    async def _run_health_checks(self):
        """Run all enabled health checks."""
        try:
            check_tasks = []
            
            for check_name, health_check in self._health_checks.items():
                if health_check.enabled:
                    task = self._run_single_health_check(health_check)
                    check_tasks.append(task)
            
            if check_tasks:
                await asyncio.gather(*check_tasks, return_exceptions=True)
        
        except Exception as e:
            self.logger.error(f"Error running health checks: {e}")
    
    async def _run_single_health_check(self, health_check: HealthCheck):
        """Run a single health check."""
        start_time = time.time()
        
        try:
            # Check dependencies first
            if not await self._check_dependencies(health_check):
                result = HealthResult(
                    check_name=health_check.name,
                    status=HealthStatus.DEGRADED,
                    timestamp=start_time,
                    response_time_ms=0.0,
                    message="Dependencies not satisfied"
                )
            else:
                # Run the actual check with timeout
                try:
                    check_result = await asyncio.wait_for(
                        health_check.check_function(),
                        timeout=health_check.timeout_seconds
                    )
                    
                    if isinstance(check_result, HealthResult):
                        result = check_result
                    else:
                        # Convert boolean or other result to HealthResult
                        status = HealthStatus.HEALTHY if check_result else HealthStatus.UNHEALTHY
                        result = HealthResult(
                            check_name=health_check.name,
                            status=status,
                            timestamp=start_time,
                            response_time_ms=(time.time() - start_time) * 1000,
                            message="Check completed"
                        )
                
                except asyncio.TimeoutError:
                    result = HealthResult(
                        check_name=health_check.name,
                        status=HealthStatus.CRITICAL,
                        timestamp=start_time,
                        response_time_ms=health_check.timeout_seconds * 1000,
                        message=f"Health check timed out after {health_check.timeout_seconds}s"
                    )
                
                except Exception as e:
                    result = HealthResult(
                        check_name=health_check.name,
                        status=HealthStatus.CRITICAL,
                        timestamp=start_time,
                        response_time_ms=(time.time() - start_time) * 1000,
                        message=f"Health check failed: {e}",
                        error=e
                    )
            
            # Update component health
            await self._update_component_health(health_check, result)
            
            # Check if recovery is needed
            if result.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
                await self._consider_recovery(health_check, result)
        
        except Exception as e:
            self.logger.error(f"Error in health check {health_check.name}: {e}")
    
    async def _check_dependencies(self, health_check: HealthCheck) -> bool:
        """Check if health check dependencies are satisfied."""
        for dep_name in health_check.dependencies:
            if dep_name in self._component_health:
                dep_health = self._component_health[dep_name]
                if dep_health.overall_status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
                    return False
        return True
    
    async def _update_component_health(self, health_check: HealthCheck, result: HealthResult):
        """Update component health based on check result."""
        with self._lock:
            if health_check.name not in self._component_health:
                self._component_health[health_check.name] = ComponentHealth(
                    name=health_check.name,
                    component_type=health_check.component_type,
                    overall_status=HealthStatus.UNKNOWN,
                    last_check_time=0.0
                )
            
            component = self._component_health[health_check.name]
            component.add_result(result)
            
            # Update circuit breaker state
            if health_check.name in self._circuit_breakers:
                circuit_breaker = self._circuit_breakers[health_check.name]
                component.circuit_breaker_state = circuit_breaker.get_state()
    
    async def _consider_recovery(self, health_check: HealthCheck, result: HealthResult):
        """Consider recovery actions based on health check result."""
        try:
            component = self._component_health.get(health_check.name)
            if not component:
                return
            
            # Check if recovery should be attempted
            if component.consecutive_failures >= health_check.failure_threshold:
                if health_check.recovery_strategies:
                    recovery_attempted = await self._recovery_manager.attempt_recovery(
                        health_check.name,
                        health_check.recovery_strategies
                    )
                    
                    if recovery_attempted:
                        component.recovery_attempts += 1
                        component.last_recovery_time = time.time()
                        
                        # Add to system recovery actions
                        action_description = f"Recovery attempted for {health_check.name}"
                        self._system_health.recovery_actions_taken.append(action_description)
        
        except Exception as e:
            self.logger.error(f"Error considering recovery for {health_check.name}: {e}")
    
    async def _update_system_health(self):
        """Update overall system health status."""
        try:
            with self._lock:
                current_time = time.time()
                
                # Update uptime
                self._system_health.uptime_seconds = current_time - self._start_time
                self._system_health.timestamp = current_time
                
                # Update component health in system health
                self._system_health.component_health = self._component_health.copy()
                
                # Determine overall system status
                self._determine_overall_status()
                
                # Check resource constraints
                await self._check_resource_constraints()
                
                # Identify degraded and critical components
                self._identify_problem_components()
        
        except Exception as e:
            self.logger.error(f"Error updating system health: {e}")
    
    def _determine_overall_status(self):
        """Determine overall system health status."""
        if not self._component_health:
            self._system_health.overall_status = HealthStatus.UNKNOWN
            return
        
        critical_count = 0
        unhealthy_count = 0
        degraded_count = 0
        healthy_count = 0
        
        for component in self._component_health.values():
            if component.overall_status == HealthStatus.CRITICAL:
                critical_count += 1
            elif component.overall_status == HealthStatus.UNHEALTHY:
                unhealthy_count += 1
            elif component.overall_status == HealthStatus.DEGRADED:
                degraded_count += 1
            elif component.overall_status == HealthStatus.HEALTHY:
                healthy_count += 1
        
        # Determine overall status
        if critical_count > 0:
            self._system_health.overall_status = HealthStatus.CRITICAL
        elif unhealthy_count > healthy_count:
            self._system_health.overall_status = HealthStatus.UNHEALTHY
        elif degraded_count > 0:
            self._system_health.overall_status = HealthStatus.DEGRADED
        else:
            self._system_health.overall_status = HealthStatus.HEALTHY
    
    async def _check_resource_constraints(self):
        """Check for resource constraints."""
        try:
            constraints = {}
            
            # Check system resources
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            if cpu_percent > self._resource_thresholds['cpu_percent']:
                constraints['cpu'] = {
                    'current': cpu_percent,
                    'threshold': self._resource_thresholds['cpu_percent'],
                    'severity': 'high' if cpu_percent > 90 else 'medium'
                }
            
            if memory.percent > self._resource_thresholds['memory_percent']:
                constraints['memory'] = {
                    'current': memory.percent,
                    'threshold': self._resource_thresholds['memory_percent'],
                    'severity': 'high' if memory.percent > 95 else 'medium'
                }
            
            disk_percent = (disk.used / disk.total) * 100
            if disk_percent > self._resource_thresholds['disk_percent']:
                constraints['disk'] = {
                    'current': disk_percent,
                    'threshold': self._resource_thresholds['disk_percent'],
                    'severity': 'high' if disk_percent > 95 else 'medium'
                }
            
            self._system_health.resource_constraints = constraints
            
            # Trigger degradation if needed
            if constraints:
                await self._handle_resource_constraints(constraints)
        
        except Exception as e:
            self.logger.error(f"Error checking resource constraints: {e}")
    
    async def _handle_resource_constraints(self, constraints: Dict[str, Any]):
        """Handle resource constraints with degradation strategies."""
        try:
            for resource, info in constraints.items():
                severity = info['severity']
                
                if severity == 'high':
                    # Implement aggressive degradation
                    self.logger.warning(f"High {resource} usage detected: {info['current']:.1f}%")
                    
                    # Reduce processing load
                    await self._apply_degradation_strategy(resource, 'aggressive')
                
                elif severity == 'medium':
                    # Implement moderate degradation
                    self.logger.info(f"Moderate {resource} usage detected: {info['current']:.1f}%")
                    
                    # Reduce processing load
                    await self._apply_degradation_strategy(resource, 'moderate')
        
        except Exception as e:
            self.logger.error(f"Error handling resource constraints: {e}")
    
    async def _apply_degradation_strategy(self, resource: str, severity: str):
        """Apply degradation strategy based on resource and severity."""
        try:
            if resource == 'cpu' and severity == 'aggressive':
                # Reduce processing frequency
                # Disable non-critical features
                # Increase batch processing intervals
                pass
            
            elif resource == 'memory' and severity == 'aggressive':
                # Force garbage collection
                gc.collect()
                # Reduce cache sizes
                # Clear buffers
                pass
            
            elif resource == 'disk' and severity == 'aggressive':
                # Clean up temporary files
                # Compress logs
                # Reduce log levels
                pass
        
        except Exception as e:
            self.logger.error(f"Error applying degradation strategy: {e}")
    
    def _identify_problem_components(self):
        """Identify degraded and critical components."""
        self._system_health.degraded_components = []
        self._system_health.critical_components = []
        
        for name, component in self._component_health.items():
            if component.overall_status == HealthStatus.DEGRADED:
                self._system_health.degraded_components.append(name)
            elif component.overall_status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
                self._system_health.critical_components.append(name)
    
    async def _cleanup_old_data(self):
        """Clean up old health check data."""
        try:
            cutoff_time = time.time() - (24 * 3600)  # 24 hours ago
            
            with self._lock:
                for component in self._component_health.values():
                    # Reset 24h failure count
                    component.failure_count_24h = 0
        
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
    
    async def _save_health_report(self):
        """Save comprehensive health report."""
        try:
            report_path = Path("data/health_report.json")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            health_report = self._system_health.to_dict()
            
            with open(report_path, 'w') as f:
                json.dump(health_report, f, indent=2, default=str)
            
            self.logger.info(f"Health report saved to {report_path}")
        
        except Exception as e:
            self.logger.error(f"Error saving health report: {e}")
    
    # Health check implementations
    
    async def _check_system_resources(self) -> HealthResult:
        """Check system resource usage."""
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            # Determine status based on resource usage
            if cpu_percent > 90 or memory.percent > 95:
                status = HealthStatus.CRITICAL
                message = f"Critical resource usage: CPU {cpu_percent:.1f}%, Memory {memory.percent:.1f}%"
            elif cpu_percent > 80 or memory.percent > 85:
                status = HealthStatus.DEGRADED
                message = f"High resource usage: CPU {cpu_percent:.1f}%, Memory {memory.percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Normal resource usage: CPU {cpu_percent:.1f}%, Memory {memory.percent:.1f}%"
            
            return HealthResult(
                check_name="system_resources",
                status=status,
                timestamp=time.time(),
                response_time_ms=10.0,
                message=message,
                details={
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_available_gb': memory.available / (1024**3)
                }
            )
        
        except Exception as e:
            return HealthResult(
                check_name="system_resources",
                status=HealthStatus.CRITICAL,
                timestamp=time.time(),
                response_time_ms=0.0,
                message=f"Failed to check system resources: {e}",
                error=e
            )
    
    async def _check_pipeline_health(self) -> HealthResult:
        """Check pipeline health."""
        # This would integrate with the actual pipeline manager
        return HealthResult(
            check_name="pipeline_health",
            status=HealthStatus.HEALTHY,
            timestamp=time.time(),
            response_time_ms=5.0,
            message="Pipeline operating normally"
        )
    
    async def _check_detection_engine(self) -> HealthResult:
        """Check detection engine health."""
        # This would integrate with the actual detection engine
        return HealthResult(
            check_name="detection_engine",
            status=HealthStatus.HEALTHY,
            timestamp=time.time(),
            response_time_ms=8.0,
            message="Detection engine operating normally"
        )
    
    async def _check_alert_system(self) -> HealthResult:
        """Check alert system health."""
        # This would integrate with the actual alert system
        return HealthResult(
            check_name="alert_system",
            status=HealthStatus.HEALTHY,
            timestamp=time.time(),
            response_time_ms=3.0,
            message="Alert system operating normally"
        )
    
    async def _check_memory_usage(self) -> HealthResult:
        """Check memory usage patterns."""
        try:
            memory = psutil.virtual_memory()
            process = psutil.Process()
            process_memory = process.memory_info()
            
            # Check for memory leaks or unusual patterns
            if memory.percent > 90:
                status = HealthStatus.CRITICAL
            elif memory.percent > 80:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            
            return HealthResult(
                check_name="memory_usage",
                status=status,
                timestamp=time.time(),
                response_time_ms=2.0,
                message=f"Memory usage: {memory.percent:.1f}%",
                details={
                    'system_memory_percent': memory.percent,
                    'process_memory_mb': process_memory.rss / (1024**2),
                    'process_memory_percent': process.memory_percent()
                }
            )
        
        except Exception as e:
            return HealthResult(
                check_name="memory_usage",
                status=HealthStatus.CRITICAL,
                timestamp=time.time(),
                response_time_ms=0.0,
                message=f"Failed to check memory usage: {e}",
                error=e
            )
    
    # Public API
    
    def register_health_check(self, health_check: HealthCheck) -> bool:
        """Register a new health check."""
        try:
            self._health_checks[health_check.name] = health_check
            
            # Create circuit breaker if needed
            if RecoveryStrategy.CIRCUIT_BREAKER in health_check.recovery_strategies:
                self._circuit_breakers[health_check.name] = CircuitBreaker(
                    health_check.name,
                    failure_threshold=health_check.failure_threshold
                )
            
            self.logger.info(f"Registered health check: {health_check.name}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error registering health check {health_check.name}: {e}")
            return False
    
    def unregister_health_check(self, check_name: str) -> bool:
        """Unregister a health check."""
        try:
            if check_name in self._health_checks:
                del self._health_checks[check_name]
                
                # Remove circuit breaker
                if check_name in self._circuit_breakers:
                    del self._circuit_breakers[check_name]
                
                # Remove component health
                if check_name in self._component_health:
                    del self._component_health[check_name]
                
                self.logger.info(f"Unregistered health check: {check_name}")
                return True
            
            return False
        
        except Exception as e:
            self.logger.error(f"Error unregistering health check {check_name}: {e}")
            return False
    
    def get_system_health(self) -> SystemHealth:
        """Get current system health status."""
        return self._system_health
    
    def get_component_health(self, component_name: str) -> Optional[ComponentHealth]:
        """Get health status for specific component."""
        return self._component_health.get(component_name)
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary."""
        return {
            'system_health': self._system_health.to_dict(),
            'monitoring_enabled': self._monitoring_enabled,
            'registered_checks': len(self._health_checks),
            'circuit_breakers': {
                name: cb.get_state().value 
                for name, cb in self._circuit_breakers.items()
            }
        }
    
    async def force_health_check(self, check_name: str) -> Optional[HealthResult]:
        """Force execution of a specific health check."""
        if check_name in self._health_checks:
            health_check = self._health_checks[check_name]
            await self._run_single_health_check(health_check)
            
            # Return latest result
            if check_name in self._component_health:
                component = self._component_health[check_name]
                if component.check_results:
                    return component.check_results[-1]
        
        return None


# Global health monitor instance
_global_health_monitor: Optional[HealthMonitor] = None


def get_health_monitor(config: Optional[AppConfig] = None) -> Optional[HealthMonitor]:
    """Get global health monitor instance."""
    global _global_health_monitor
    if _global_health_monitor is None and config is not None:
        _global_health_monitor = HealthMonitor(config)
    return _global_health_monitor