"""
Async utilities and thread-safe operations for GStreamer integration.

This module provides thread-safe async utilities, queue management, context managers,
and graceful shutdown coordination for the DeepStream inference system.
"""

import asyncio
import threading
import time
import weakref
from typing import Any, Callable, Optional, Dict, List, Union, TypeVar, Generic, Awaitable
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from enum import Enum
import logging
import signal
import functools
from concurrent.futures import ThreadPoolExecutor, as_completed

from .errors import PyDSError
from .logging import get_logger


T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


class ShutdownStage(Enum):
    """Stages of the shutdown process."""
    RUNNING = "running"
    SHUTDOWN_INITIATED = "shutdown_initiated"
    STOPPING_COMPONENTS = "stopping_components"
    CLEANING_UP = "cleaning_up"
    SHUTDOWN_COMPLETE = "shutdown_complete"


@dataclass
class TaskInfo:
    """Information about a managed async task."""
    task: asyncio.Task
    name: str
    created_at: float
    description: Optional[str] = None
    cleanup_callback: Optional[Callable] = None


class ThreadSafeAsyncQueue(Generic[T]):
    """Thread-safe queue for communication between async and sync contexts."""
    
    def __init__(self, maxsize: int = 0):
        self._queue = asyncio.Queue(maxsize=maxsize)
        self._loop = None
        self._lock = threading.Lock()
    
    def _ensure_loop(self):
        """Ensure we have access to the event loop."""
        if self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                # No running loop, we'll need to create one
                pass
    
    async def put(self, item: T) -> None:
        """Put item in queue (async)."""
        await self._queue.put(item)
    
    def put_nowait(self, item: T) -> None:
        """Put item in queue without waiting (thread-safe)."""
        with self._lock:
            try:
                self._queue.put_nowait(item)
            except asyncio.QueueFull:
                raise
    
    def put_from_thread(self, item: T) -> None:
        """Put item in queue from a different thread."""
        self._ensure_loop()
        if self._loop is not None:
            # Schedule the put operation in the event loop
            asyncio.run_coroutine_threadsafe(self.put(item), self._loop)
        else:
            # Fallback to nowait if no loop available
            self.put_nowait(item)
    
    async def get(self) -> T:
        """Get item from queue (async)."""
        return await self._queue.get()
    
    def get_nowait(self) -> T:
        """Get item from queue without waiting."""
        return self._queue.get_nowait()
    
    async def get_with_timeout(self, timeout: float) -> Optional[T]:
        """Get item from queue with timeout."""
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
    
    def task_done(self) -> None:
        """Mark task as done."""
        self._queue.task_done()
    
    async def join(self) -> None:
        """Wait for all tasks to be done."""
        await self._queue.join()
    
    def qsize(self) -> int:
        """Get queue size."""
        return self._queue.qsize()
    
    def empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()
    
    def full(self) -> bool:
        """Check if queue is full."""
        return self._queue.full()


class AsyncTaskManager:
    """Manages async tasks with proper cleanup and error handling."""
    
    def __init__(self):
        self._tasks: Dict[str, TaskInfo] = {}
        self._lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()
        self._logger = get_logger(__name__)
    
    async def create_task(
        self,
        coro: Awaitable[T],
        name: str,
        description: Optional[str] = None,
        cleanup_callback: Optional[Callable] = None
    ) -> asyncio.Task[T]:
        """
        Create and register a managed task.
        
        Args:
            coro: Coroutine to execute
            name: Unique name for the task
            description: Optional description
            cleanup_callback: Optional cleanup function to call when task completes
            
        Returns:
            Created asyncio.Task
        """
        async with self._lock:
            if name in self._tasks:
                raise ValueError(f"Task with name '{name}' already exists")
            
            task = asyncio.create_task(coro, name=name)
            task_info = TaskInfo(
                task=task,
                name=name,
                created_at=time.time(),
                description=description,
                cleanup_callback=cleanup_callback
            )
            
            self._tasks[name] = task_info
            
            # Add done callback for cleanup
            task.add_done_callback(
                lambda t: asyncio.create_task(self._handle_task_completion(name))
            )
            
            self._logger.debug(f"Created task '{name}': {description or 'No description'}")
            return task
    
    async def _handle_task_completion(self, task_name: str):
        """Handle task completion and cleanup."""
        async with self._lock:
            if task_name not in self._tasks:
                return
            
            task_info = self._tasks[task_name]
            task = task_info.task
            
            try:
                if task.cancelled():
                    self._logger.debug(f"Task '{task_name}' was cancelled")
                elif task.exception():
                    self._logger.error(
                        f"Task '{task_name}' failed with exception: {task.exception()}"
                    )
                else:
                    self._logger.debug(f"Task '{task_name}' completed successfully")
                
                # Call cleanup callback if provided
                if task_info.cleanup_callback:
                    try:
                        if asyncio.iscoroutinefunction(task_info.cleanup_callback):
                            await task_info.cleanup_callback()
                        else:
                            task_info.cleanup_callback()
                    except Exception as e:
                        self._logger.error(f"Cleanup callback for task '{task_name}' failed: {e}")
                
            finally:
                # Remove task from tracking
                del self._tasks[task_name]
    
    async def cancel_task(self, name: str, timeout: float = 5.0) -> bool:
        """
        Cancel a specific task with timeout.
        
        Args:
            name: Name of task to cancel
            timeout: Maximum time to wait for cancellation
            
        Returns:
            True if task was cancelled successfully
        """
        async with self._lock:
            if name not in self._tasks:
                return True  # Task doesn't exist, consider it cancelled
            
            task = self._tasks[name].task
        
        if task.done():
            return True
        
        task.cancel()
        
        try:
            await asyncio.wait_for(task, timeout=timeout)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        
        return task.cancelled() or task.done()
    
    async def cancel_all_tasks(self, timeout: float = 10.0) -> int:
        """
        Cancel all managed tasks.
        
        Args:
            timeout: Maximum time to wait for all cancellations
            
        Returns:
            Number of tasks that were cancelled
        """
        async with self._lock:
            task_names = list(self._tasks.keys())
        
        if not task_names:
            return 0
        
        self._logger.info(f"Cancelling {len(task_names)} tasks...")
        
        # Cancel all tasks
        cancelled_count = 0
        for name in task_names:
            if await self.cancel_task(name, timeout=timeout / len(task_names)):
                cancelled_count += 1
        
        return cancelled_count
    
    async def wait_for_tasks(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for all tasks to complete.
        
        Args:
            timeout: Maximum time to wait
            
        Returns:
            True if all tasks completed within timeout
        """
        async with self._lock:
            if not self._tasks:
                return True
            
            tasks = [info.task for info in self._tasks.values()]
        
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout
            )
            return True
        except asyncio.TimeoutError:
            return False
    
    def get_task_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all managed tasks."""
        info = {}
        for name, task_info in self._tasks.items():
            info[name] = {
                'name': name,
                'description': task_info.description,
                'created_at': task_info.created_at,
                'age_seconds': time.time() - task_info.created_at,
                'done': task_info.task.done(),
                'cancelled': task_info.task.cancelled()
            }
            
            if task_info.task.done() and not task_info.task.cancelled():
                try:
                    exception = task_info.task.exception()
                    info[name]['exception'] = str(exception) if exception else None
                except Exception:
                    info[name]['exception'] = "Unknown"
        
        return info


class GracefulShutdownManager:
    """Manages graceful shutdown of the entire application."""
    
    def __init__(self):
        self._stage = ShutdownStage.RUNNING
        self._shutdown_callbacks: List[Callable] = []
        self._components: Dict[str, Any] = weakref.WeakValueDictionary()
        self._lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()
        self._task_manager = AsyncTaskManager()
        self._logger = get_logger(__name__)
        
        # Register signal handlers
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self._logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.shutdown())
        
        try:
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)
        except ValueError:
            # Signal handling not available (e.g., on Windows or in threads)
            self._logger.debug("Signal handlers not available in this context")
    
    def register_component(self, name: str, component: Any):
        """Register a component for shutdown management."""
        self._components[name] = component
        self._logger.debug(f"Registered component for shutdown: {name}")
    
    def register_shutdown_callback(self, callback: Callable):
        """Register a callback to be called during shutdown."""
        self._shutdown_callbacks.append(callback)
    
    async def shutdown(self, timeout: float = 30.0):
        """
        Perform graceful shutdown of all components.
        
        Args:
            timeout: Maximum time to wait for shutdown
        """
        async with self._lock:
            if self._stage != ShutdownStage.RUNNING:
                self._logger.warning("Shutdown already in progress")
                return
            
            self._stage = ShutdownStage.SHUTDOWN_INITIATED
            self._shutdown_event.set()
            
            self._logger.info("Starting graceful shutdown...")
            start_time = time.time()
            
            try:
                # Stage 1: Stop components
                self._stage = ShutdownStage.STOPPING_COMPONENTS
                await self._stop_components(timeout * 0.6)
                
                # Stage 2: Run shutdown callbacks
                await self._run_shutdown_callbacks(timeout * 0.2)
                
                # Stage 3: Cancel all managed tasks
                self._stage = ShutdownStage.CLEANING_UP
                await self._task_manager.cancel_all_tasks(timeout * 0.2)
                
                # Stage 4: Final cleanup
                await self._final_cleanup()
                
                self._stage = ShutdownStage.SHUTDOWN_COMPLETE
                elapsed = time.time() - start_time
                self._logger.info(f"Graceful shutdown completed in {elapsed:.2f} seconds")
                
            except Exception as e:
                self._logger.error(f"Error during shutdown: {e}")
                self._stage = ShutdownStage.SHUTDOWN_COMPLETE
    
    async def _stop_components(self, timeout: float):
        """Stop all registered components."""
        if not self._components:
            return
        
        self._logger.info(f"Stopping {len(self._components)} components...")
        
        # Try to stop components gracefully
        for name, component in list(self._components.items()):
            try:
                if hasattr(component, 'stop'):
                    if asyncio.iscoroutinefunction(component.stop):
                        await asyncio.wait_for(component.stop(), timeout=timeout / len(self._components))
                    else:
                        component.stop()
                    self._logger.debug(f"Stopped component: {name}")
            except Exception as e:
                self._logger.error(f"Error stopping component {name}: {e}")
    
    async def _run_shutdown_callbacks(self, timeout: float):
        """Run all registered shutdown callbacks."""
        if not self._shutdown_callbacks:
            return
        
        self._logger.info(f"Running {len(self._shutdown_callbacks)} shutdown callbacks...")
        
        for i, callback in enumerate(self._shutdown_callbacks):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await asyncio.wait_for(callback(), timeout=timeout / len(self._shutdown_callbacks))
                else:
                    callback()
            except Exception as e:
                self._logger.error(f"Error in shutdown callback {i}: {e}")
    
    async def _final_cleanup(self):
        """Perform final cleanup tasks."""
        # Clear components and callbacks
        self._components.clear()
        self._shutdown_callbacks.clear()
        
        self._logger.debug("Final cleanup completed")
    
    @property
    def is_shutting_down(self) -> bool:
        """Check if shutdown is in progress."""
        return self._stage != ShutdownStage.RUNNING
    
    @property
    def shutdown_stage(self) -> ShutdownStage:
        """Get current shutdown stage."""
        return self._stage
    
    async def wait_for_shutdown(self):
        """Wait for shutdown to be initiated."""
        await self._shutdown_event.wait()


class AsyncContextManager:
    """Base class for async context managers with cleanup."""
    
    def __init__(self, name: str):
        self.name = name
        self._logger = get_logger(__name__)
        self._cleanup_callbacks: List[Callable] = []
    
    def add_cleanup(self, callback: Callable):
        """Add a cleanup callback."""
        self._cleanup_callbacks.append(callback)
    
    async def __aenter__(self):
        """Enter async context."""
        await self._setup()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context with cleanup."""
        try:
            await self._cleanup()
        except Exception as e:
            self._logger.error(f"Error during cleanup of {self.name}: {e}")
    
    async def _setup(self):
        """Override in subclasses for setup logic."""
        pass
    
    async def _cleanup(self):
        """Run all cleanup callbacks."""
        for callback in reversed(self._cleanup_callbacks):  # Reverse order
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                self._logger.error(f"Error in cleanup callback for {self.name}: {e}")


@asynccontextmanager
async def managed_resource(
    setup_func: Callable,
    cleanup_func: Callable,
    name: str = "resource"
):
    """
    Context manager for resources that need setup and cleanup.
    
    Args:
        setup_func: Function to call for setup (can be async)
        cleanup_func: Function to call for cleanup (can be async)
        name: Name for logging
    """
    logger = get_logger(__name__)
    resource = None
    
    try:
        logger.debug(f"Setting up resource: {name}")
        if asyncio.iscoroutinefunction(setup_func):
            resource = await setup_func()
        else:
            resource = setup_func()
        
        yield resource
        
    except Exception as e:
        logger.error(f"Error with resource {name}: {e}")
        raise
    finally:
        if resource is not None:
            try:
                logger.debug(f"Cleaning up resource: {name}")
                if asyncio.iscoroutinefunction(cleanup_func):
                    await cleanup_func(resource)
                else:
                    cleanup_func(resource)
            except Exception as e:
                logger.error(f"Error cleaning up resource {name}: {e}")


def run_in_thread_pool(
    func: Callable,
    *args,
    executor: Optional[ThreadPoolExecutor] = None,
    **kwargs
) -> asyncio.Future:
    """
    Run a function in a thread pool.
    
    Args:
        func: Function to run
        *args: Function arguments
        executor: Optional custom executor
        **kwargs: Function keyword arguments
        
    Returns:
        Future representing the result
    """
    loop = asyncio.get_event_loop()
    
    if executor is None:
        return loop.run_in_executor(None, functools.partial(func, **kwargs), *args)
    else:
        return loop.run_in_executor(executor, functools.partial(func, **kwargs), *args)


def sync_to_async(func: F) -> F:
    """
    Decorator to run synchronous function in thread pool.
    
    Args:
        func: Synchronous function to wrap
        
    Returns:
        Async version of the function
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        return await run_in_thread_pool(func, *args, **kwargs)
    
    return wrapper


def rate_limit(calls_per_second: float):
    """
    Decorator to rate limit function calls.
    
    Args:
        calls_per_second: Maximum calls per second
    """
    min_interval = 1.0 / calls_per_second
    last_called = {}
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            now = time.time()
            key = id(func)
            
            if key in last_called:
                elapsed = now - last_called[key]
                if elapsed < min_interval:
                    await asyncio.sleep(min_interval - elapsed)
            
            last_called[key] = time.time()
            return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            now = time.time()
            key = id(func)
            
            if key in last_called:
                elapsed = now - last_called[key]
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)
            
            last_called[key] = time.time()
            return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


class PeriodicTaskRunner:
    """Runs periodic tasks with proper async handling."""
    
    def __init__(self, interval: float, task_manager: Optional[AsyncTaskManager] = None):
        self.interval = interval
        self.task_manager = task_manager or AsyncTaskManager()
        self._tasks: Dict[str, asyncio.Task] = {}
        self._stop_events: Dict[str, asyncio.Event] = {}
        self._logger = get_logger(__name__)
    
    async def start_periodic_task(
        self,
        name: str,
        coro_func: Callable[[], Awaitable[Any]],
        interval: Optional[float] = None
    ):
        """
        Start a periodic task.
        
        Args:
            name: Unique name for the task
            coro_func: Async function to run periodically
            interval: Override default interval
        """
        if name in self._tasks:
            raise ValueError(f"Periodic task '{name}' already running")
        
        task_interval = interval or self.interval
        stop_event = asyncio.Event()
        self._stop_events[name] = stop_event
        
        async def periodic_wrapper():
            while not stop_event.is_set():
                try:
                    await coro_func()
                except Exception as e:
                    self._logger.error(f"Error in periodic task '{name}': {e}")
                
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=task_interval)
                    break  # Stop event was set
                except asyncio.TimeoutError:
                    continue  # Timeout, continue with next iteration
        
        task = await self.task_manager.create_task(
            periodic_wrapper(),
            name=f"periodic_{name}",
            description=f"Periodic task: {name} (interval: {task_interval}s)"
        )
        
        self._tasks[name] = task
        self._logger.info(f"Started periodic task '{name}' with {task_interval}s interval")
    
    async def stop_periodic_task(self, name: str, timeout: float = 5.0) -> bool:
        """
        Stop a specific periodic task.
        
        Args:
            name: Name of task to stop
            timeout: Maximum time to wait
            
        Returns:
            True if task was stopped successfully
        """
        if name not in self._tasks:
            return True
        
        # Set stop event
        self._stop_events[name].set()
        
        # Wait for task to complete
        success = await self.task_manager.cancel_task(f"periodic_{name}", timeout)
        
        # Cleanup
        if name in self._tasks:
            del self._tasks[name]
        if name in self._stop_events:
            del self._stop_events[name]
        
        if success:
            self._logger.info(f"Stopped periodic task '{name}'")
        else:
            self._logger.warning(f"Failed to stop periodic task '{name}' within timeout")
        
        return success
    
    async def stop_all_tasks(self, timeout: float = 10.0) -> int:
        """Stop all periodic tasks."""
        task_names = list(self._tasks.keys())
        stopped_count = 0
        
        for name in task_names:
            if await self.stop_periodic_task(name, timeout / len(task_names) if task_names else timeout):
                stopped_count += 1
        
        return stopped_count


# Global instances
_shutdown_manager: Optional[GracefulShutdownManager] = None
_task_manager: Optional[AsyncTaskManager] = None


def get_shutdown_manager() -> GracefulShutdownManager:
    """Get global shutdown manager."""
    global _shutdown_manager
    if _shutdown_manager is None:
        _shutdown_manager = GracefulShutdownManager()
    return _shutdown_manager


def get_task_manager() -> AsyncTaskManager:
    """Get global task manager."""
    global _task_manager
    if _task_manager is None:
        _task_manager = AsyncTaskManager()
    return _task_manager


# Convenience functions
async def create_managed_task(
    coro: Awaitable[T],
    name: str,
    description: Optional[str] = None
) -> asyncio.Task[T]:
    """Create a managed task using the global task manager."""
    return await get_task_manager().create_task(coro, name, description)


def register_for_shutdown(name: str, component: Any):
    """Register a component for graceful shutdown."""
    get_shutdown_manager().register_component(name, component)


def add_shutdown_callback(callback: Callable):
    """Add a callback to be called during shutdown."""
    get_shutdown_manager().register_shutdown_callback(callback)


async def graceful_shutdown():
    """Perform graceful shutdown of all registered components."""
    return await get_shutdown_manager().shutdown()


# Aliases for backwards compatibility
TaskManager = AsyncTaskManager