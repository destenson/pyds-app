"""
Comprehensive alert management system with intelligent processing and delivery.

This module provides the core AlertManager class that coordinates alert processing,
throttling, prioritization, and delivery through multiple handlers with guaranteed
delivery and comprehensive error handling.
"""

import asyncio
import time
import json
from typing import Dict, List, Optional, Any, Callable, Set, Tuple, Union, Type
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor
import uuid
import hashlib

from ..config import AppConfig, AlertConfig, AlertLevel
from ..utils.errors import AlertError, handle_error, recovery_strategy
from ..utils.logging import get_logger, performance_context, log_alert_event
from ..utils.async_utils import get_task_manager, ThreadSafeAsyncQueue, PeriodicTaskRunner
from ..detection.models import VideoDetection, DetectionResult
from .throttling import ThrottlingManager, get_throttling_manager


class AlertStatus(Enum):
    """Alert processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    THROTTLED = "throttled"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"
    EXPIRED = "expired"


class AlertPriority(Enum):
    """Alert priority levels for processing order."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


class DeliveryStatus(Enum):
    """Delivery attempt status."""
    SUCCESS = "success"
    TEMPORARY_FAILURE = "temporary_failure"
    PERMANENT_FAILURE = "permanent_failure"
    TIMEOUT = "timeout"
    HANDLER_ERROR = "handler_error"


@dataclass
class AlertMessage:
    """Core alert message structure."""
    alert_id: str
    detection: VideoDetection
    alert_level: AlertLevel
    priority: AlertPriority
    message: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.expires_at is None:
            # Default expiration of 1 hour
            self.expires_at = self.created_at + timedelta(hours=1)
    
    def is_expired(self) -> bool:
        """Check if alert has expired."""
        return datetime.now() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'alert_id': self.alert_id,
            'detection': asdict(self.detection),
            'alert_level': self.alert_level.value,
            'priority': self.priority.value,
            'message': self.message,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'metadata': self.metadata,
            'tags': list(self.tags)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AlertMessage':
        """Create from dictionary."""
        return cls(
            alert_id=data['alert_id'],
            detection=VideoDetection(**data['detection']),
            alert_level=AlertLevel(data['alert_level']),
            priority=AlertPriority(data['priority']),
            message=data['message'],
            created_at=datetime.fromisoformat(data['created_at']),
            expires_at=datetime.fromisoformat(data['expires_at']) if data.get('expires_at') else None,
            metadata=data.get('metadata', {}),
            tags=set(data.get('tags', []))
        )


@dataclass
class DeliveryAttempt:
    """Record of alert delivery attempt."""
    attempt_id: str
    alert_id: str
    handler_name: str
    status: DeliveryStatus
    attempted_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    next_retry_at: Optional[datetime] = None
    delivery_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'attempt_id': self.attempt_id,
            'alert_id': self.alert_id,
            'handler_name': self.handler_name,
            'status': self.status.value,
            'attempted_at': self.attempted_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error_message': self.error_message,
            'retry_count': self.retry_count,
            'next_retry_at': self.next_retry_at.isoformat() if self.next_retry_at else None,
            'delivery_time_ms': self.delivery_time_ms
        }


@dataclass
class AlertTracker:
    """Tracks alert processing lifecycle."""
    alert: AlertMessage
    status: AlertStatus
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    processing_started_at: Optional[float] = None
    throttle_reason: Optional[str] = None
    throttle_metadata: Optional[Dict[str, Any]] = None
    delivery_attempts: List[DeliveryAttempt] = field(default_factory=list)
    retry_count: int = 0
    next_retry_at: Optional[float] = None
    total_processing_time_ms: float = 0.0
    
    def update_status(self, status: AlertStatus, metadata: Optional[Dict[str, Any]] = None):
        """Update alert status with timestamp."""
        self.status = status
        self.updated_at = time.time()
        
        if status == AlertStatus.PROCESSING and self.processing_started_at is None:
            self.processing_started_at = time.time()
        elif status in [AlertStatus.DELIVERED, AlertStatus.FAILED, AlertStatus.EXPIRED]:
            if self.processing_started_at:
                self.total_processing_time_ms = (time.time() - self.processing_started_at) * 1000
    
    def add_delivery_attempt(self, attempt: DeliveryAttempt):
        """Add delivery attempt record."""
        self.delivery_attempts.append(attempt)
        self.updated_at = time.time()
    
    def get_successful_deliveries(self) -> List[DeliveryAttempt]:
        """Get successful delivery attempts."""
        return [attempt for attempt in self.delivery_attempts if attempt.status == DeliveryStatus.SUCCESS]
    
    def get_failed_deliveries(self) -> List[DeliveryAttempt]:
        """Get failed delivery attempts."""
        return [
            attempt for attempt in self.delivery_attempts 
            if attempt.status in [DeliveryStatus.PERMANENT_FAILURE, DeliveryStatus.TIMEOUT, DeliveryStatus.HANDLER_ERROR]
        ]
    
    def should_retry(self, max_retries: int) -> bool:
        """Check if alert should be retried."""
        if self.retry_count >= max_retries:
            return False
        
        if self.next_retry_at and time.time() < self.next_retry_at:
            return False
        
        return self.status == AlertStatus.FAILED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'alert': self.alert.to_dict(),
            'status': self.status.value,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'processing_started_at': self.processing_started_at,
            'throttle_reason': self.throttle_reason,
            'throttle_metadata': self.throttle_metadata,
            'delivery_attempts': [attempt.to_dict() for attempt in self.delivery_attempts],
            'retry_count': self.retry_count,
            'next_retry_at': self.next_retry_at,
            'total_processing_time_ms': self.total_processing_time_ms
        }


class PriorityQueue:
    """Thread-safe priority queue for alerts."""
    
    def __init__(self, maxsize: int = 10000):
        """Initialize priority queue."""
        self._queues: Dict[AlertPriority, deque] = {
            priority: deque() for priority in AlertPriority
        }
        self._maxsize = maxsize
        self._size = 0
        self._lock = threading.RLock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)
    
    def put(self, alert_tracker: AlertTracker, block: bool = True, timeout: Optional[float] = None) -> bool:
        """Put alert in priority queue."""
        with self._not_full:
            if not block:
                if self._size >= self._maxsize:
                    return False
            else:
                while self._size >= self._maxsize:
                    if not self._not_full.wait(timeout):
                        return False
            
            self._queues[alert_tracker.alert.priority].append(alert_tracker)
            self._size += 1
            self._not_empty.notify()
            return True
    
    def get(self, block: bool = True, timeout: Optional[float] = None) -> Optional[AlertTracker]:
        """Get highest priority alert from queue."""
        with self._not_empty:
            if not block:
                if self._size == 0:
                    return None
            else:
                while self._size == 0:
                    if not self._not_empty.wait(timeout):
                        return None
            
            # Get from highest priority queue first
            for priority in sorted(AlertPriority, key=lambda p: p.value, reverse=True):
                queue = self._queues[priority]
                if queue:
                    alert_tracker = queue.popleft()
                    self._size -= 1
                    self._not_full.notify()
                    return alert_tracker
            
            return None
    
    def qsize(self) -> int:
        """Get total queue size."""
        with self._lock:
            return self._size
    
    def qsize_by_priority(self) -> Dict[AlertPriority, int]:
        """Get queue sizes by priority."""
        with self._lock:
            return {priority: len(queue) for priority, queue in self._queues.items()}
    
    def empty(self) -> bool:
        """Check if queue is empty."""
        with self._lock:
            return self._size == 0
    
    def full(self) -> bool:
        """Check if queue is full."""
        with self._lock:
            return self._size >= self._maxsize


class AlertHandler:
    """Base class for alert handlers."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize handler."""
        self.name = name
        self.config = config
        self.logger = get_logger(f"{__name__}.{name}")
        self.enabled = config.get('enabled', True)
        self.timeout_seconds = config.get('timeout_seconds', 30.0)
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay_seconds = config.get('retry_delay_seconds', 5.0)
        
        # Statistics
        self._stats = {
            'total_alerts': 0,
            'successful_deliveries': 0,
            'failed_deliveries': 0,
            'timeout_deliveries': 0,
            'average_delivery_time_ms': 0.0,
            'last_delivery_at': None
        }
        self._delivery_times: deque = deque(maxlen=100)
        self._stats_lock = threading.Lock()
    
    async def handle_alert(self, alert: AlertMessage) -> DeliveryAttempt:
        """Handle alert delivery."""
        attempt_id = str(uuid.uuid4())
        start_time = time.time()
        
        attempt = DeliveryAttempt(
            attempt_id=attempt_id,
            alert_id=alert.alert_id,
            handler_name=self.name,
            status=DeliveryStatus.TEMPORARY_FAILURE,
            attempted_at=datetime.now()
        )
        
        try:
            with performance_context(f"alert_delivery_{self.name}"):
                # Check if handler is enabled
                if not self.enabled:
                    attempt.status = DeliveryStatus.PERMANENT_FAILURE
                    attempt.error_message = f"Handler {self.name} is disabled"
                    return attempt
                
                # Perform the actual delivery
                success = await self._deliver_alert(alert)
                
                if success:
                    attempt.status = DeliveryStatus.SUCCESS
                    attempt.completed_at = datetime.now()
                    
                    # Update statistics
                    delivery_time_ms = (time.time() - start_time) * 1000
                    attempt.delivery_time_ms = delivery_time_ms
                    self._update_stats(True, delivery_time_ms)
                    
                else:
                    attempt.status = DeliveryStatus.TEMPORARY_FAILURE
                    attempt.error_message = "Delivery failed"
                    self._update_stats(False, 0)
        
        except asyncio.TimeoutError:
            attempt.status = DeliveryStatus.TIMEOUT
            attempt.error_message = f"Delivery timeout after {self.timeout_seconds}s"
            self._update_stats(False, 0, timeout=True)
        
        except Exception as e:
            attempt.status = DeliveryStatus.HANDLER_ERROR
            attempt.error_message = str(e)
            attempt.completed_at = datetime.now()
            self.logger.error(f"Handler error in {self.name}: {e}")
            self._update_stats(False, 0)
        
        return attempt
    
    async def _deliver_alert(self, alert: AlertMessage) -> bool:
        """Override this method to implement actual delivery logic."""
        raise NotImplementedError("Subclasses must implement _deliver_alert")
    
    def _update_stats(self, success: bool, delivery_time_ms: float, timeout: bool = False):
        """Update handler statistics."""
        with self._stats_lock:
            self._stats['total_alerts'] += 1
            
            if success:
                self._stats['successful_deliveries'] += 1
                self._delivery_times.append(delivery_time_ms)
                
                # Update average delivery time
                if self._delivery_times:
                    self._stats['average_delivery_time_ms'] = sum(self._delivery_times) / len(self._delivery_times)
                
                self._stats['last_delivery_at'] = datetime.now().isoformat()
            
            elif timeout:
                self._stats['timeout_deliveries'] += 1
                self._stats['failed_deliveries'] += 1
            
            else:
                self._stats['failed_deliveries'] += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get handler statistics."""
        with self._stats_lock:
            stats = self._stats.copy()
            
            if stats['total_alerts'] > 0:
                stats['success_rate'] = stats['successful_deliveries'] / stats['total_alerts']
                stats['failure_rate'] = stats['failed_deliveries'] / stats['total_alerts']
                stats['timeout_rate'] = stats['timeout_deliveries'] / stats['total_alerts']
            else:
                stats['success_rate'] = 0.0
                stats['failure_rate'] = 0.0
                stats['timeout_rate'] = 0.0
            
            return stats
    
    def reset_statistics(self):
        """Reset handler statistics."""
        with self._stats_lock:
            self._stats = {
                'total_alerts': 0,
                'successful_deliveries': 0,
                'failed_deliveries': 0,
                'timeout_deliveries': 0,
                'average_delivery_time_ms': 0.0,
                'last_delivery_at': None
            }
            self._delivery_times.clear()


class AlertManager:
    """
    Comprehensive alert management system with intelligent processing.
    
    Provides multi-threaded alert processing, intelligent throttling,
    prioritization, delivery guarantees, and comprehensive monitoring.
    """
    
    def __init__(self, config: AppConfig):
        """
        Initialize alert manager.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.alert_config = config.alerts
        self.logger = get_logger(__name__)
        
        # Core components
        self._throttling_manager = get_throttling_manager(config)
        if not self._throttling_manager:
            self._throttling_manager = ThrottlingManager(config)
        
        # Processing queues
        self._processing_queue = PriorityQueue(maxsize=self.alert_config.max_queue_size)
        self._retry_queue = PriorityQueue(maxsize=1000)
        
        # Alert tracking
        self._active_alerts: Dict[str, AlertTracker] = {}
        self._alert_history: deque = deque(maxlen=10000)
        self._tracking_lock = threading.RLock()
        
        # Handler management
        self._handlers: Dict[str, AlertHandler] = {}
        self._handler_groups: Dict[str, List[str]] = {}
        
        # Processing control
        self._processing_enabled = False
        self._worker_threads: List[threading.Thread] = []
        self._thread_pool = ThreadPoolExecutor(
            max_workers=self.alert_config.max_workers,
            thread_name_prefix="AlertWorker"
        )
        
        # Statistics and monitoring
        self._statistics = {
            'total_alerts_received': 0,
            'alerts_processed': 0,
            'alerts_throttled': 0,
            'alerts_delivered': 0,
            'alerts_failed': 0,
            'alerts_expired': 0,
            'average_processing_time_ms': 0.0,
            'queue_high_watermark': 0
        }
        self._stats_lock = threading.Lock()
        
        # Periodic tasks
        self._periodic_runner = PeriodicTaskRunner()
        self._cleanup_interval = 300.0  # 5 minutes
        self._stats_interval = 60.0     # 1 minute
        
        # State persistence
        self._state_file = Path("data/alert_manager_state.json")
        self._state_save_interval = 30.0
        
        self.logger.info("AlertManager initialized")
    
    async def start(self) -> bool:
        """
        Start the alert manager.
        
        Returns:
            True if started successfully
        """
        try:
            self.logger.info("Starting AlertManager...")
            
            # Start throttling manager
            if hasattr(self._throttling_manager, 'start_periodic_tasks'):
                await self._throttling_manager.start_periodic_tasks()
            
            # Start worker threads
            self._start_worker_threads()
            
            # Start periodic tasks
            await self._start_periodic_tasks()
            
            # Load saved state
            await self._load_state()
            
            self._processing_enabled = True
            
            self.logger.info("AlertManager started successfully")
            return True
        
        except Exception as e:
            error = handle_error(e, context={'component': 'alert_manager'})
            self.logger.error(f"Failed to start AlertManager: {error}")
            return False
    
    async def stop(self) -> bool:
        """
        Stop the alert manager.
        
        Returns:
            True if stopped successfully
        """
        try:
            self.logger.info("Stopping AlertManager...")
            
            self._processing_enabled = False
            
            # Stop periodic tasks
            await self._periodic_runner.stop_all_tasks()
            
            # Stop worker threads
            self._stop_worker_threads()
            
            # Shutdown thread pool
            self._thread_pool.shutdown(wait=True, timeout=30)
            
            # Stop throttling manager
            if hasattr(self._throttling_manager, 'stop_periodic_tasks'):
                await self._throttling_manager.stop_periodic_tasks()
            
            # Save final state
            await self._save_state()
            
            self.logger.info("AlertManager stopped successfully")
            return True
        
        except Exception as e:
            self.logger.error(f"Error stopping AlertManager: {e}")
            return False
    
    def _start_worker_threads(self):
        """Start worker threads for alert processing."""
        num_workers = min(self.alert_config.max_workers, 8)
        
        for i in range(num_workers):
            worker = threading.Thread(
                target=self._worker_thread,
                name=f"AlertWorker-{i}",
                daemon=True
            )
            worker.start()
            self._worker_threads.append(worker)
        
        self.logger.info(f"Started {num_workers} alert worker threads")
    
    def _stop_worker_threads(self):
        """Stop worker threads."""
        # Workers will stop when processing_enabled becomes False
        for worker in self._worker_threads:
            worker.join(timeout=5.0)
        
        self._worker_threads.clear()
        self.logger.info("Stopped alert worker threads")
    
    def _worker_thread(self):
        """Worker thread for processing alerts."""
        thread_name = threading.current_thread().name
        self.logger.debug(f"Started {thread_name}")
        
        while self._processing_enabled:
            try:
                # Get alert from queue with timeout
                alert_tracker = self._processing_queue.get(timeout=1.0)
                if alert_tracker is None:
                    continue
                
                # Process the alert
                asyncio.run(self._process_alert(alert_tracker))
            
            except Exception as e:
                self.logger.error(f"Error in {thread_name}: {e}")
                time.sleep(0.1)  # Prevent tight error loop
        
        self.logger.debug(f"Stopped {thread_name}")
    
    async def submit_alert(self, detection: VideoDetection, message: str, level: AlertLevel = AlertLevel.MEDIUM) -> str:
        """
        Submit alert for processing.
        
        Args:
            detection: Detection that triggered the alert
            message: Alert message
            level: Alert level
            
        Returns:
            Alert ID
        """
        try:
            # Create alert message
            alert_id = self._generate_alert_id(detection)
            priority = self._map_level_to_priority(level)
            
            alert = AlertMessage(
                alert_id=alert_id,
                detection=detection,
                alert_level=level,
                priority=priority,
                message=message,
                created_at=datetime.now(),
                metadata={
                    'source_id': detection.source_id,
                    'pattern_name': detection.pattern_name,
                    'confidence': detection.confidence
                }
            )
            
            # Check throttling
            should_allow, reason, throttle_metadata = await self._throttling_manager.should_allow_alert(
                detection, level
            )
            
            with self._stats_lock:
                self._statistics['total_alerts_received'] += 1
            
            # Create alert tracker
            alert_tracker = AlertTracker(
                alert=alert,
                status=AlertStatus.PENDING if should_allow else AlertStatus.THROTTLED
            )
            
            if not should_allow:
                # Alert is throttled
                alert_tracker.throttle_reason = reason
                alert_tracker.throttle_metadata = throttle_metadata
                
                with self._stats_lock:
                    self._statistics['alerts_throttled'] += 1
                
                self.logger.debug(f"Alert {alert_id} throttled: {reason}")
                
                # Still track throttled alerts for monitoring
                with self._tracking_lock:
                    self._active_alerts[alert_id] = alert_tracker
                
                return alert_id
            
            # Queue alert for processing
            queued = self._processing_queue.put(alert_tracker, block=False)
            
            if queued:
                with self._tracking_lock:
                    self._active_alerts[alert_id] = alert_tracker
                
                # Update queue statistics
                queue_size = self._processing_queue.qsize()
                with self._stats_lock:
                    self._statistics['queue_high_watermark'] = max(
                        self._statistics['queue_high_watermark'], 
                        queue_size
                    )
                
                log_alert_event(
                    self.logger,
                    alert_id,
                    detection.source_id,
                    level.value,
                    f"Queued for processing (queue size: {queue_size})"
                )
            else:
                # Queue full - mark as failed
                alert_tracker.update_status(AlertStatus.FAILED)
                alert_tracker.throttle_reason = "queue_full"
                
                with self._stats_lock:
                    self._statistics['alerts_failed'] += 1
                
                self.logger.warning(f"Alert queue full, dropping alert {alert_id}")
            
            return alert_id
        
        except Exception as e:
            error = handle_error(e, context={'detection_id': str(detection.detection_id)})
            self.logger.error(f"Error submitting alert: {error}")
            raise AlertError(f"Failed to submit alert: {error}")
    
    async def _process_alert(self, alert_tracker: AlertTracker):
        """Process a single alert through all handlers."""
        alert = alert_tracker.alert
        alert_id = alert.alert_id
        
        try:
            # Check if alert has expired
            if alert.is_expired():
                alert_tracker.update_status(AlertStatus.EXPIRED)
                with self._stats_lock:
                    self._statistics['alerts_expired'] += 1
                
                self.logger.debug(f"Alert {alert_id} expired")
                self._finalize_alert(alert_tracker)
                return
            
            # Update status to processing
            alert_tracker.update_status(AlertStatus.PROCESSING)
            
            # Get applicable handlers
            handlers = self._get_handlers_for_alert(alert)
            
            if not handlers:
                self.logger.warning(f"No handlers available for alert {alert_id}")
                alert_tracker.update_status(AlertStatus.FAILED)
                self._finalize_alert(alert_tracker)
                return
            
            # Process through handlers concurrently
            delivery_tasks = []
            for handler in handlers:
                task = handler.handle_alert(alert)
                delivery_tasks.append(task)
            
            # Wait for all deliveries to complete
            delivery_attempts = await asyncio.gather(*delivery_tasks, return_exceptions=True)
            
            successful_deliveries = 0
            for attempt in delivery_attempts:
                if isinstance(attempt, DeliveryAttempt):
                    alert_tracker.add_delivery_attempt(attempt)
                    if attempt.status == DeliveryStatus.SUCCESS:
                        successful_deliveries += 1
                elif isinstance(attempt, Exception):
                    self.logger.error(f"Handler execution error: {attempt}")
            
            # Determine final status
            if successful_deliveries > 0:
                alert_tracker.update_status(AlertStatus.DELIVERED)
                with self._stats_lock:
                    self._statistics['alerts_delivered'] += 1
                
                log_alert_event(
                    self.logger,
                    alert_id,
                    alert.detection.source_id,
                    alert.alert_level.value,
                    f"Delivered successfully ({successful_deliveries}/{len(handlers)} handlers)"
                )
            else:
                # All handlers failed - check if we should retry
                if alert_tracker.should_retry(self.alert_config.max_retries):
                    await self._schedule_retry(alert_tracker)
                else:
                    alert_tracker.update_status(AlertStatus.FAILED)
                    with self._stats_lock:
                        self._statistics['alerts_failed'] += 1
                    
                    self.logger.error(f"Alert {alert_id} failed to deliver to all handlers")
            
            # Update processing statistics
            with self._stats_lock:
                self._statistics['alerts_processed'] += 1
                
                # Update average processing time
                total_processed = self._statistics['alerts_processed']
                current_avg = self._statistics['average_processing_time_ms']
                new_time = alert_tracker.total_processing_time_ms
                
                self._statistics['average_processing_time_ms'] = (
                    (current_avg * (total_processed - 1) + new_time) / total_processed
                )
            
            self._finalize_alert(alert_tracker)
        
        except Exception as e:
            error = handle_error(e, context={'alert_id': alert_id})
            self.logger.error(f"Error processing alert {alert_id}: {error}")
            
            alert_tracker.update_status(AlertStatus.FAILED)
            self._finalize_alert(alert_tracker)
    
    def _finalize_alert(self, alert_tracker: AlertTracker):
        """Finalize alert processing and move to history."""
        alert_id = alert_tracker.alert.alert_id
        
        with self._tracking_lock:
            # Move from active to history
            if alert_id in self._active_alerts:
                del self._active_alerts[alert_id]
            
            self._alert_history.append(alert_tracker)
    
    async def _schedule_retry(self, alert_tracker: AlertTracker):
        """Schedule alert for retry."""
        alert_tracker.retry_count += 1
        alert_tracker.update_status(AlertStatus.RETRYING)
        
        # Calculate retry delay with exponential backoff
        base_delay = self.alert_config.retry_delay_seconds
        delay = base_delay * (2 ** (alert_tracker.retry_count - 1))
        delay = min(delay, self.alert_config.max_retry_delay_seconds)
        
        alert_tracker.next_retry_at = time.time() + delay
        
        # Add to retry queue
        self._retry_queue.put(alert_tracker, block=False)
        
        self.logger.info(
            f"Scheduled alert {alert_tracker.alert.alert_id} for retry "
            f"#{alert_tracker.retry_count} in {delay:.1f}s"
        )
    
    def _get_handlers_for_alert(self, alert: AlertMessage) -> List[AlertHandler]:
        """Get applicable handlers for alert based on configuration."""
        applicable_handlers = []
        
        for handler_name, handler in self._handlers.items():
            if not handler.enabled:
                continue
            
            # Check handler-specific filters
            if self._should_handler_process_alert(handler, alert):
                applicable_handlers.append(handler)
        
        return applicable_handlers
    
    def _should_handler_process_alert(self, handler: AlertHandler, alert: AlertMessage) -> bool:
        """Check if handler should process the alert."""
        # Check alert level filter
        min_level = handler.config.get('min_alert_level')
        if min_level and AlertLevel(min_level).value > alert.alert_level.value:
            return False
        
        # Check source filter
        source_filter = handler.config.get('source_filter')
        if source_filter and alert.detection.source_id not in source_filter:
            return False
        
        # Check pattern filter
        pattern_filter = handler.config.get('pattern_filter')
        if pattern_filter and alert.detection.pattern_name not in pattern_filter:
            return False
        
        # Check tag filter
        tag_filter = handler.config.get('tag_filter')
        if tag_filter and not alert.tags.intersection(set(tag_filter)):
            return False
        
        return True
    
    def _generate_alert_id(self, detection: VideoDetection) -> str:
        """Generate unique alert ID."""
        # Create deterministic ID based on detection properties
        data = f"{detection.detection_id}:{detection.source_id}:{detection.timestamp}:{time.time()}"
        return hashlib.md5(data.encode()).hexdigest()[:16]
    
    def _map_level_to_priority(self, level: AlertLevel) -> AlertPriority:
        """Map alert level to processing priority."""
        mapping = {
            AlertLevel.LOW: AlertPriority.LOW,
            AlertLevel.MEDIUM: AlertPriority.MEDIUM,
            AlertLevel.HIGH: AlertPriority.HIGH,
            AlertLevel.CRITICAL: AlertPriority.CRITICAL
        }
        return mapping.get(level, AlertPriority.MEDIUM)
    
    async def _start_periodic_tasks(self):
        """Start periodic maintenance tasks."""
        await self._periodic_runner.start_periodic_task(
            "cleanup_expired_alerts",
            self._cleanup_expired_alerts,
            interval=self._cleanup_interval
        )
        
        await self._periodic_runner.start_periodic_task(
            "process_retry_queue",
            self._process_retry_queue,
            interval=10.0  # Check every 10 seconds
        )
        
        await self._periodic_runner.start_periodic_task(
            "save_state",
            self._save_state,
            interval=self._state_save_interval
        )
        
        self.logger.info("Started alert manager periodic tasks")
    
    async def _cleanup_expired_alerts(self):
        """Clean up expired alerts and old history."""
        try:
            current_time = time.time()
            expired_count = 0
            
            with self._tracking_lock:
                # Clean up active alerts that have expired
                expired_alerts = []
                for alert_id, tracker in self._active_alerts.items():
                    if tracker.alert.is_expired():
                        expired_alerts.append(alert_id)
                
                for alert_id in expired_alerts:
                    tracker = self._active_alerts[alert_id]
                    tracker.update_status(AlertStatus.EXPIRED)
                    self._finalize_alert(tracker)
                    expired_count += 1
                
                # Clean up old history entries (older than 24 hours)
                cutoff_time = current_time - (24 * 3600)
                while self._alert_history and self._alert_history[0].created_at < cutoff_time:
                    self._alert_history.popleft()
            
            if expired_count > 0:
                self.logger.info(f"Cleaned up {expired_count} expired alerts")
        
        except Exception as e:
            self.logger.error(f"Error in cleanup task: {e}")
    
    async def _process_retry_queue(self):
        """Process alerts scheduled for retry."""
        try:
            current_time = time.time()
            processed_retries = 0
            
            # Process all ready retries
            while not self._retry_queue.empty():
                alert_tracker = self._retry_queue.get(block=False)
                if alert_tracker is None:
                    break
                
                # Check if it's time to retry
                if alert_tracker.next_retry_at and current_time >= alert_tracker.next_retry_at:
                    # Reset status and requeue for processing
                    alert_tracker.update_status(AlertStatus.PENDING)
                    alert_tracker.next_retry_at = None
                    
                    if self._processing_queue.put(alert_tracker, block=False):
                        processed_retries += 1
                    else:
                        # Queue full - mark as failed
                        alert_tracker.update_status(AlertStatus.FAILED)
                        self._finalize_alert(alert_tracker)
                else:
                    # Not ready yet - put back in retry queue
                    self._retry_queue.put(alert_tracker, block=False)
                    break
            
            if processed_retries > 0:
                self.logger.debug(f"Processed {processed_retries} retry alerts")
        
        except Exception as e:
            self.logger.error(f"Error processing retry queue: {e}")
    
    def register_handler(self, handler: AlertHandler) -> bool:
        """Register alert handler."""
        try:
            self._handlers[handler.name] = handler
            self.logger.info(f"Registered alert handler: {handler.name}")
            return True
        except Exception as e:
            self.logger.error(f"Error registering handler {handler.name}: {e}")
            return False
    
    def unregister_handler(self, handler_name: str) -> bool:
        """Unregister alert handler."""
        try:
            if handler_name in self._handlers:
                del self._handlers[handler_name]
                self.logger.info(f"Unregistered alert handler: {handler_name}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error unregistering handler {handler_name}: {e}")
            return False
    
    def get_alert_status(self, alert_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific alert."""
        with self._tracking_lock:
            # Check active alerts
            if alert_id in self._active_alerts:
                return self._active_alerts[alert_id].to_dict()
            
            # Check history
            for tracker in self._alert_history:
                if tracker.alert.alert_id == alert_id:
                    return tracker.to_dict()
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive alert manager statistics."""
        with self._stats_lock:
            stats = self._statistics.copy()
        
        # Add queue statistics
        stats['queue_size'] = self._processing_queue.qsize()
        stats['queue_size_by_priority'] = {
            priority.name: size for priority, size in self._processing_queue.qsize_by_priority().items()
        }
        stats['retry_queue_size'] = self._retry_queue.qsize()
        
        # Add active alerts count
        with self._tracking_lock:
            stats['active_alerts_count'] = len(self._active_alerts)
            stats['alert_history_count'] = len(self._alert_history)
        
        # Add handler statistics
        stats['handlers'] = {
            name: handler.get_statistics() for name, handler in self._handlers.items()
        }
        
        # Add throttling statistics
        if self._throttling_manager:
            stats['throttling'] = self._throttling_manager.get_throttling_statistics()
        
        return stats
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get list of currently active alerts."""
        with self._tracking_lock:
            return [tracker.to_dict() for tracker in self._active_alerts.values()]
    
    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent alert history."""
        with self._tracking_lock:
            recent_alerts = list(self._alert_history)[-limit:]
            return [tracker.to_dict() for tracker in recent_alerts]
    
    async def _save_state(self):
        """Save alert manager state to file."""
        try:
            state_data = {
                'timestamp': time.time(),
                'statistics': self._statistics,
                'active_alerts_count': len(self._active_alerts),
                'queue_size': self._processing_queue.qsize(),
                'retry_queue_size': self._retry_queue.qsize()
            }
            
            # Ensure data directory exists
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write state file
            with open(self._state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            self.logger.debug("Saved alert manager state")
        
        except Exception as e:
            self.logger.error(f"Error saving state: {e}")
    
    async def _load_state(self):
        """Load alert manager state from file."""
        try:
            if not self._state_file.exists():
                self.logger.debug("No state file found, starting fresh")
                return
            
            with open(self._state_file, 'r') as f:
                state_data = json.load(f)
            
            # Load statistics (but don't overwrite current session stats)
            self.logger.info("Loaded alert manager state from file")
        
        except Exception as e:
            self.logger.error(f"Error loading state: {e}")


# Global alert manager instance
_global_alert_manager: Optional[AlertManager] = None


def get_alert_manager(config: Optional[AppConfig] = None) -> Optional[AlertManager]:
    """Get global alert manager instance."""
    global _global_alert_manager
    if _global_alert_manager is None and config is not None:
        _global_alert_manager = AlertManager(config)
    return _global_alert_manager