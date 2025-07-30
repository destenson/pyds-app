"""
Core detection engine with DeepStream metadata processing.

This module provides the central detection engine that processes DeepStream metadata,
manages detection strategies, and routes results to the alert system.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Callable, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import threading
from collections import defaultdict, deque

from ..config import AppConfig, DetectionConfig
from ..utils.errors import DetectionError, handle_error, recovery_strategy
from ..utils.logging import get_logger, performance_context, log_detection_event
from ..utils.async_utils import get_task_manager, ThreadSafeAsyncQueue, PeriodicTaskRunner
from ..utils.deepstream import get_deepstream_api, get_deepstream_info, DeepStreamVersion
from .models import (
    VideoDetection, DetectionResult, DetectionStatistics, BoundingBox, 
    DetectionMetadata, DetectionStrategy, create_test_detection,
    filter_overlapping_detections
)


class DetectionMode(Enum):
    """Detection processing modes."""
    REAL_TIME = "real_time"
    BATCH = "batch"
    INTERVAL = "interval"


class ProcessingStage(Enum):
    """Detection processing stages."""
    PRE_PROCESSING = "pre_processing"
    INFERENCE = "inference"
    POST_PROCESSING = "post_processing"
    STRATEGY_PROCESSING = "strategy_processing"
    RESULT_AGGREGATION = "result_aggregation"


@dataclass
class DetectionFrame:
    """Container for frame data and metadata."""
    frame_number: int
    source_id: str
    timestamp: datetime
    gst_buffer: Any  # GStreamer buffer
    batch_meta: Any  # DeepStream batch metadata
    width: int
    height: int
    processing_start: float = field(default_factory=time.perf_counter)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyResult:
    """Result from a detection strategy."""
    strategy_name: str
    detections: List[VideoDetection]
    processing_time_ms: float
    confidence_scores: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None


class MetadataExtractor:
    """Extracts detection metadata from DeepStream batch metadata."""
    
    def __init__(self, deepstream_api, deepstream_info):
        """
        Initialize metadata extractor.
        
        Args:
            deepstream_api: DeepStream API instance
            deepstream_info: DeepStream version information
        """
        self.deepstream_api = deepstream_api
        self.deepstream_info = deepstream_info
        self.logger = get_logger(__name__)
    
    async def extract_detections(
        self, 
        batch_meta: Any, 
        source_id: str, 
        frame_width: int, 
        frame_height: int
    ) -> List[VideoDetection]:
        """
        Extract detection results from DeepStream batch metadata.
        
        Args:
            batch_meta: DeepStream batch metadata
            source_id: Source identifier
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            
        Returns:
            List of extracted detections
        """
        detections = []
        
        try:
            # This is a comprehensive implementation for DeepStream metadata extraction
            # The actual implementation depends on the DeepStream version and API
            
            if self.deepstream_info.major_version >= 6:
                detections = await self._extract_detections_v6plus(
                    batch_meta, source_id, frame_width, frame_height
                )
            else:
                detections = await self._extract_detections_v5(
                    batch_meta, source_id, frame_width, frame_height
                )
            
            self.logger.debug(f"Extracted {len(detections)} detections from {source_id}")
            return detections
        
        except Exception as e:
            self.logger.error(f"Error extracting detections: {e}")
            return []
    
    async def _extract_detections_v6plus(
        self, 
        batch_meta: Any, 
        source_id: str, 
        frame_width: int, 
        frame_height: int
    ) -> List[VideoDetection]:
        """Extract detections for DeepStream 6.x+."""
        detections = []
        
        try:
            # Note: This is a simplified version. The actual implementation would
            # require importing the appropriate DeepStream Python bindings and
            # iterating through the metadata structures.
            
            # Simulate metadata extraction for demonstration
            # In real implementation, this would iterate through:
            # - batch_meta.frame_meta_list
            # - frame_meta.obj_meta_list
            # - Extract bounding boxes, confidence scores, class labels
            
            # Placeholder detection for testing
            detection = VideoDetection(
                pattern_name="test_object",
                confidence=0.85,
                bounding_box=BoundingBox(x=0.1, y=0.1, width=0.2, height=0.3),
                frame_number=1,
                source_id=source_id,
                timestamp=datetime.now(),
                metadata=DetectionMetadata(
                    class_id=0,
                    class_name="test_object"
                )
            )
            detections.append(detection)
            
        except Exception as e:
            self.logger.error(f"Error in v6+ metadata extraction: {e}")
        
        return detections
    
    async def _extract_detections_v5(
        self, 
        batch_meta: Any, 
        source_id: str, 
        frame_width: int, 
        frame_height: int
    ) -> List[VideoDetection]:
        """Extract detections for DeepStream 5.x."""
        detections = []
        
        try:
            # Simplified v5 extraction
            # Real implementation would use pyds module directly
            
            # Placeholder detection for testing
            detection = VideoDetection(
                pattern_name="test_object_v5",
                confidence=0.75,
                bounding_box=BoundingBox(x=0.15, y=0.15, width=0.25, height=0.35),
                frame_number=1,
                source_id=source_id,
                timestamp=datetime.now(),
                metadata=DetectionMetadata(
                    class_id=0,
                    class_name="test_object_v5"
                )
            )
            detections.append(detection)
            
        except Exception as e:
            self.logger.error(f"Error in v5 metadata extraction: {e}")
        
        return detections
    
    def normalize_bounding_box(
        self, 
        left: float, 
        top: float, 
        width: float, 
        height: float,
        frame_width: int, 
        frame_height: int
    ) -> BoundingBox:
        """
        Normalize bounding box coordinates.
        
        Args:
            left: Left coordinate in pixels
            top: Top coordinate in pixels
            width: Width in pixels
            height: Height in pixels
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            
        Returns:
            Normalized bounding box
        """
        try:
            return BoundingBox(
                x=left / frame_width,
                y=top / frame_height,
                width=width / frame_width,
                height=height / frame_height
            )
        
        except (ValueError, ZeroDivisionError) as e:
            self.logger.error(f"Error normalizing bounding box: {e}")
            # Return a default valid bounding box
            return BoundingBox(x=0.0, y=0.0, width=0.1, height=0.1)


class StrategyManager:
    """Manages detection strategies and their execution."""
    
    def __init__(self, config: DetectionConfig):
        """
        Initialize strategy manager.
        
        Args:
            config: Detection configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        self._strategies: Dict[str, DetectionStrategy] = {}
        self._strategy_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._enabled_strategies: Set[str] = set()
        self._lock = threading.RLock()
    
    def register_strategy(self, strategy: DetectionStrategy) -> bool:
        """
        Register a detection strategy.
        
        Args:
            strategy: Detection strategy to register
            
        Returns:
            True if registration successful
        """
        try:
            with self._lock:
                if strategy.name in self._strategies:
                    self.logger.warning(f"Strategy {strategy.name} already registered, replacing")
                
                self._strategies[strategy.name] = strategy
                
                if strategy.enabled:
                    self._enabled_strategies.add(strategy.name)
                
                self.logger.info(f"Registered detection strategy: {strategy.name}")
                return True
        
        except Exception as e:
            self.logger.error(f"Error registering strategy {strategy.name}: {e}")
            return False
    
    def unregister_strategy(self, strategy_name: str) -> bool:
        """
        Unregister a detection strategy.
        
        Args:
            strategy_name: Name of strategy to unregister
            
        Returns:
            True if unregistration successful
        """
        try:
            with self._lock:
                if strategy_name in self._strategies:
                    strategy = self._strategies[strategy_name]
                    
                    # Clean up strategy
                    asyncio.create_task(strategy.cleanup())
                    
                    del self._strategies[strategy_name]
                    self._enabled_strategies.discard(strategy_name)
                    
                    self.logger.info(f"Unregistered detection strategy: {strategy_name}")
                    return True
                
                return False
        
        except Exception as e:
            self.logger.error(f"Error unregistering strategy {strategy_name}: {e}")
            return False
    
    def enable_strategy(self, strategy_name: str) -> bool:
        """Enable a detection strategy."""
        try:
            with self._lock:
                if strategy_name in self._strategies:
                    self._strategies[strategy_name].enabled = True
                    self._enabled_strategies.add(strategy_name)
                    self.logger.info(f"Enabled strategy: {strategy_name}")
                    return True
                
                return False
        
        except Exception as e:
            self.logger.error(f"Error enabling strategy {strategy_name}: {e}")
            return False
    
    def disable_strategy(self, strategy_name: str) -> bool:
        """Disable a detection strategy."""
        try:
            with self._lock:
                if strategy_name in self._strategies:
                    self._strategies[strategy_name].enabled = False
                    self._enabled_strategies.discard(strategy_name)
                    self.logger.info(f"Disabled strategy: {strategy_name}")
                    return True
                
                return False
        
        except Exception as e:
            self.logger.error(f"Error disabling strategy {strategy_name}: {e}")
            return False
    
    async def process_detections(
        self, 
        detections: List[VideoDetection], 
        frame_data: DetectionFrame
    ) -> List[StrategyResult]:
        """
        Process detections through all enabled strategies.
        
        Args:
            detections: List of base detections
            frame_data: Frame data and metadata
            
        Returns:
            List of strategy results
        """
        results = []
        
        with self._lock:
            enabled_strategies = list(self._enabled_strategies)
        
        if not enabled_strategies:
            self.logger.debug("No enabled strategies for processing")
            return results
        
        # Process strategies concurrently
        strategy_tasks = []
        for strategy_name in enabled_strategies:
            strategy = self._strategies.get(strategy_name)
            if strategy:
                task = self._process_strategy(strategy, detections, frame_data)
                strategy_tasks.append(task)
        
        if strategy_tasks:
            strategy_results = await asyncio.gather(*strategy_tasks, return_exceptions=True)
            
            for result in strategy_results:
                if isinstance(result, StrategyResult):
                    results.append(result)
                elif isinstance(result, Exception):
                    self.logger.error(f"Strategy processing error: {result}")
        
        return results
    
    async def _process_strategy(
        self, 
        strategy: DetectionStrategy, 
        detections: List[VideoDetection],
        frame_data: DetectionFrame
    ) -> StrategyResult:
        """Process detections through a single strategy."""
        start_time = time.perf_counter()
        
        try:
            # Filter detections that should be processed by this strategy
            applicable_detections = [
                detection for detection in detections 
                if strategy.should_process(detection)
            ]
            
            if not applicable_detections:
                return StrategyResult(
                    strategy_name=strategy.name,
                    detections=[],
                    processing_time_ms=0,
                    confidence_scores=[]
                )
            
            # Process detections through strategy
            enhanced_detections = []
            confidence_scores = []
            
            for detection in applicable_detections:
                try:
                    # In a real implementation, this would call the strategy's detect method
                    # For now, we'll simulate strategy processing
                    enhanced_detection = await self._simulate_strategy_processing(
                        strategy, detection, frame_data
                    )
                    
                    if enhanced_detection:
                        enhanced_detections.append(enhanced_detection)
                        confidence_scores.append(enhanced_detection.confidence)
                
                except Exception as e:
                    self.logger.error(f"Error processing detection in strategy {strategy.name}: {e}")
            
            processing_time_ms = (time.perf_counter() - start_time) * 1000
            
            # Update strategy statistics
            with self._lock:
                stats = self._strategy_stats[strategy.name]
                stats['total_processed'] += len(applicable_detections)
                stats['total_enhanced'] += len(enhanced_detections)
                stats['total_time_ms'] += processing_time_ms
                stats['avg_time_ms'] = stats['total_time_ms'] / max(stats['total_processed'], 1)
            
            return StrategyResult(
                strategy_name=strategy.name,
                detections=enhanced_detections,
                processing_time_ms=processing_time_ms,
                confidence_scores=confidence_scores,
                success=True
            )
        
        except Exception as e:
            error_msg = f"Strategy {strategy.name} processing failed: {e}"
            self.logger.error(error_msg)
            
            return StrategyResult(
                strategy_name=strategy.name,
                detections=[],
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
                confidence_scores=[],
                success=False,
                error_message=error_msg
            )
    
    async def _simulate_strategy_processing(
        self, 
        strategy: DetectionStrategy, 
        detection: VideoDetection,
        frame_data: DetectionFrame
    ) -> Optional[VideoDetection]:
        """Simulate strategy processing for demonstration."""
        try:
            # In real implementation, this would call strategy.detect() or enhance_detection()
            # For now, we'll create an enhanced version of the detection
            
            enhanced_detection = VideoDetection(
                detection_id=detection.detection_id,
                pattern_name=f"{detection.pattern_name}_{strategy.name}",
                confidence=min(detection.confidence * 1.1, 1.0),  # Slight confidence boost
                bounding_box=detection.bounding_box,
                timestamp=detection.timestamp,
                frame_number=detection.frame_number,
                source_id=detection.source_id,
                source_name=detection.source_name,
                metadata=DetectionMetadata(
                    **detection.metadata.__dict__,
                    model_name=strategy.name,
                    preprocessing_time_ms=1.0,
                    inference_time_ms=5.0
                )
            )
            
            return enhanced_detection
        
        except Exception as e:
            self.logger.error(f"Error in strategy simulation: {e}")
            return None
    
    def get_strategy_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all strategies."""
        with self._lock:
            return dict(self._strategy_stats)
    
    def get_enabled_strategies(self) -> Set[str]:
        """Get set of enabled strategy names."""
        with self._lock:
            return self._enabled_strategies.copy()
    
    def get_strategy(self, name: str) -> Optional[DetectionStrategy]:
        """Get strategy by name."""
        with self._lock:
            return self._strategies.get(name)


class DetectionEngine:
    """
    Core detection engine with comprehensive metadata processing.
    
    Processes DeepStream metadata, manages detection strategies, applies filters,
    and routes results to downstream systems.
    """
    
    def __init__(self, config: AppConfig):
        """
        Initialize detection engine.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.detection_config = config.detection
        self.logger = get_logger(__name__)
        
        # DeepStream integration
        self._deepstream_api = get_deepstream_api()
        self._deepstream_info = get_deepstream_info()
        self._metadata_extractor = MetadataExtractor(self._deepstream_api, self._deepstream_info)
        
        # Strategy management
        self._strategy_manager = StrategyManager(self.detection_config)
        
        # Detection processing
        self._processing_queue = ThreadSafeAsyncQueue(maxsize=1000)
        self._result_callbacks: List[Callable] = []
        self._detection_filters: List[Callable] = []
        
        # Statistics and monitoring
        self._statistics = DetectionStatistics()
        self._performance_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Processing control
        self._processing_enabled = True
        self._processing_tasks: List[asyncio.Task] = []
        self._task_manager = get_task_manager()
        
        # Configuration
        self._confidence_threshold = self.detection_config.confidence_threshold
        self._nms_threshold = self.detection_config.nms_threshold
        self._max_objects = self.detection_config.max_objects
        
        self.logger.info(f"DetectionEngine initialized (DeepStream {self._deepstream_info.version_string})")
    
    async def start(self) -> bool:
        """
        Start the detection engine.
        
        Returns:
            True if started successfully
        """
        try:
            self.logger.info("Starting DetectionEngine...")
            
            # Start processing tasks
            await self._start_processing_tasks()
            
            # Initialize strategies
            await self._initialize_default_strategies()
            
            self._processing_enabled = True
            
            self.logger.info("DetectionEngine started successfully")
            return True
        
        except Exception as e:
            error = handle_error(e, context={'component': 'detection_engine'})
            self.logger.error(f"Failed to start DetectionEngine: {error}")
            return False
    
    async def stop(self) -> bool:
        """
        Stop the detection engine.
        
        Returns:
            True if stopped successfully
        """
        try:
            self.logger.info("Stopping DetectionEngine...")
            
            self._processing_enabled = False
            
            # Stop processing tasks
            for task in self._processing_tasks:
                task.cancel()
            
            if self._processing_tasks:
                await asyncio.gather(*self._processing_tasks, return_exceptions=True)
            
            # Clean up strategies
            await self._cleanup_strategies()
            
            self.logger.info("DetectionEngine stopped successfully")
            return True
        
        except Exception as e:
            self.logger.error(f"Error stopping DetectionEngine: {e}")
            return False
    
    async def _start_processing_tasks(self):
        """Start background processing tasks."""
        # Main detection processing task
        processing_task = await self._task_manager.create_task(
            self._detection_processing_loop(),
            name="detection_processing",
            description="Main detection processing loop"
        )
        self._processing_tasks.append(processing_task)
        
        # Performance monitoring task
        metrics_task = await self._task_manager.create_task(
            self._performance_monitoring_loop(),
            name="detection_metrics",
            description="Detection performance monitoring"
        )
        self._processing_tasks.append(metrics_task)
    
    async def _initialize_default_strategies(self):
        """Initialize default detection strategies."""
        try:
            # In a real implementation, this would load and initialize
            # the actual detection strategies (YOLO, template matching, etc.)
            # For now, we'll create placeholder strategies
            
            from .strategies import create_default_strategies
            strategies = await create_default_strategies(self.config)
            
            for strategy in strategies:
                await strategy.initialize()
                self._strategy_manager.register_strategy(strategy)
            
            self.logger.info(f"Initialized {len(strategies)} default detection strategies")
        
        except Exception as e:
            self.logger.error(f"Error initializing default strategies: {e}")
    
    async def _cleanup_strategies(self):
        """Clean up all registered strategies."""
        try:
            strategies = list(self._strategy_manager._strategies.keys())
            for strategy_name in strategies:
                self._strategy_manager.unregister_strategy(strategy_name)
            
            self.logger.info("Cleaned up all detection strategies")
        
        except Exception as e:
            self.logger.error(f"Error cleaning up strategies: {e}")
    
    async def process_frame(self, frame_data: DetectionFrame) -> DetectionResult:
        """
        Process a single frame for detections.
        
        Args:
            frame_data: Frame data and metadata
            
        Returns:
            Detection results
        """
        processing_start = time.perf_counter()
        
        try:
            with performance_context(f"process_frame_{frame_data.source_id}_{frame_data.frame_number}"):
                # Extract base detections from DeepStream metadata
                base_detections = await self._metadata_extractor.extract_detections(
                    frame_data.batch_meta,
                    frame_data.source_id,
                    frame_data.width,
                    frame_data.height
                )
                
                # Apply confidence filtering
                filtered_detections = self._apply_confidence_filter(base_detections)
                
                # Apply NMS filtering
                nms_detections = filter_overlapping_detections(
                    filtered_detections, 
                    self._nms_threshold
                )
                
                # Limit number of objects
                limited_detections = nms_detections[:self._max_objects]
                
                # Process through strategies
                strategy_results = await self._strategy_manager.process_detections(
                    limited_detections, 
                    frame_data
                )
                
                # Aggregate strategy results
                final_detections = self._aggregate_strategy_results(
                    limited_detections, 
                    strategy_results
                )
                
                # Apply custom filters
                filtered_final = self._apply_custom_filters(final_detections)
                
                # Create result
                processing_time_ms = (time.perf_counter() - processing_start) * 1000
                
                result = DetectionResult(
                    detections=filtered_final,
                    frame_number=frame_data.frame_number,
                    source_id=frame_data.source_id,
                    timestamp=frame_data.timestamp,
                    processing_time_ms=processing_time_ms
                )
                
                # Update statistics
                self._statistics.update(result)
                
                # Record performance metrics
                self._record_performance_metrics(result, strategy_results)
                
                # Log detection event
                log_detection_event(
                    self.logger,
                    len(filtered_final),
                    frame_data.source_id,
                    confidence_avg=result.average_confidence
                )
                
                return result
        
        except Exception as e:
            error = handle_error(e, context={
                'source_id': frame_data.source_id,
                'frame_number': frame_data.frame_number
            })
            self.logger.error(f"Error processing frame: {error}")
            
            # Return empty result on error
            return DetectionResult(
                detections=[],
                frame_number=frame_data.frame_number,
                source_id=frame_data.source_id,
                timestamp=frame_data.timestamp,
                processing_time_ms=(time.perf_counter() - processing_start) * 1000
            )
    
    def _apply_confidence_filter(self, detections: List[VideoDetection]) -> List[VideoDetection]:
        """Apply confidence threshold filtering."""
        return [
            detection for detection in detections 
            if detection.confidence >= self._confidence_threshold
        ]
    
    def _apply_custom_filters(self, detections: List[VideoDetection]) -> List[VideoDetection]:
        """Apply custom detection filters."""
        filtered = detections
        
        for filter_func in self._detection_filters:
            try:
                filtered = filter_func(filtered)
            except Exception as e:
                self.logger.error(f"Error in custom filter: {e}")
        
        return filtered
    
    def _aggregate_strategy_results(
        self, 
        base_detections: List[VideoDetection],
        strategy_results: List[StrategyResult]
    ) -> List[VideoDetection]:
        """Aggregate results from multiple strategies."""
        aggregated = base_detections.copy()
        
        # Add enhanced detections from strategies
        for result in strategy_results:
            if result.success:
                aggregated.extend(result.detections)
        
        # Remove duplicates and apply final NMS
        deduplicated = filter_overlapping_detections(aggregated, self._nms_threshold)
        
        return deduplicated
    
    def _record_performance_metrics(self, result: DetectionResult, strategy_results: List[StrategyResult]):
        """Record performance metrics for monitoring."""
        try:
            # Record frame processing metrics
            self._performance_metrics['frame_processing_time'].append(result.processing_time_ms)
            self._performance_metrics['detections_per_frame'].append(len(result.detections))
            self._performance_metrics['confidence_scores'].extend([d.confidence for d in result.detections])
            
            # Record strategy metrics
            for strategy_result in strategy_results:
                strategy_key = f"strategy_{strategy_result.strategy_name}_time"
                self._performance_metrics[strategy_key].append(strategy_result.processing_time_ms)
        
        except Exception as e:
            self.logger.error(f"Error recording performance metrics: {e}")
    
    async def _detection_processing_loop(self):
        """Main detection processing loop."""
        self.logger.info("Started detection processing loop")
        
        while self._processing_enabled:
            try:
                # Get frame data from queue
                frame_data = await self._processing_queue.get_with_timeout(1.0)
                if frame_data is None:
                    continue
                
                # Process frame
                result = await self.process_frame(frame_data)
                
                # Notify result callbacks
                await self._notify_result_callbacks(result)
                
                # Mark task as done
                self._processing_queue.task_done()
            
            except Exception as e:
                self.logger.error(f"Error in detection processing loop: {e}")
                await asyncio.sleep(0.1)  # Prevent tight error loop
        
        self.logger.info("Detection processing loop stopped")
    
    async def _performance_monitoring_loop(self):
        """Performance monitoring loop."""
        self.logger.info("Started detection performance monitoring")
        
        while self._processing_enabled:
            try:
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
                # Calculate and log performance metrics
                await self._calculate_performance_metrics()
            
            except Exception as e:
                self.logger.error(f"Error in performance monitoring: {e}")
        
        self.logger.info("Detection performance monitoring stopped")
    
    async def _calculate_performance_metrics(self):
        """Calculate and log performance metrics."""
        try:
            metrics = {}
            
            # Calculate average processing times
            if self._performance_metrics['frame_processing_time']:
                avg_frame_time = sum(self._performance_metrics['frame_processing_time']) / len(self._performance_metrics['frame_processing_time'])
                metrics['avg_frame_processing_ms'] = avg_frame_time
                metrics['processing_fps'] = 1000.0 / avg_frame_time if avg_frame_time > 0 else 0
            
            # Calculate detection statistics
            if self._performance_metrics['detections_per_frame']:
                avg_detections = sum(self._performance_metrics['detections_per_frame']) / len(self._performance_metrics['detections_per_frame'])
                metrics['avg_detections_per_frame'] = avg_detections
            
            # Calculate confidence statistics
            if self._performance_metrics['confidence_scores']:
                confidences = self._performance_metrics['confidence_scores']
                metrics['avg_confidence'] = sum(confidences) / len(confidences)
                metrics['min_confidence'] = min(confidences)
                metrics['max_confidence'] = max(confidences)
            
            # Log strategy performance
            strategy_stats = self._strategy_manager.get_strategy_statistics()
            for strategy_name, stats in strategy_stats.items():
                metrics[f'strategy_{strategy_name}_avg_time_ms'] = stats.get('avg_time_ms', 0)
                metrics[f'strategy_{strategy_name}_processed'] = stats.get('total_processed', 0)
            
            # Log overall statistics
            self.logger.info(f"Detection Performance Metrics: {metrics}")
        
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
    
    async def queue_frame_for_processing(self, frame_data: DetectionFrame):
        """
        Queue frame data for processing.
        
        Args:
            frame_data: Frame data to process
        """
        try:
            self._processing_queue.put_nowait(frame_data)
        except asyncio.QueueFull:
            self.logger.warning(f"Detection queue full, dropping frame {frame_data.frame_number}")
    
    def add_result_callback(self, callback: Callable):
        """Add callback for detection results."""
        self._result_callbacks.append(callback)
    
    def add_detection_filter(self, filter_func: Callable):
        """Add custom detection filter."""
        self._detection_filters.append(filter_func)
    
    async def _notify_result_callbacks(self, result: DetectionResult):
        """Notify all result callbacks."""
        for callback in self._result_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(result)
                else:
                    callback(result)
            except Exception as e:
                self.logger.error(f"Error in result callback: {e}")
    
    def register_strategy(self, strategy: DetectionStrategy) -> bool:
        """Register a detection strategy."""
        return self._strategy_manager.register_strategy(strategy)
    
    def unregister_strategy(self, strategy_name: str) -> bool:
        """Unregister a detection strategy."""
        return self._strategy_manager.unregister_strategy(strategy_name)
    
    def enable_strategy(self, strategy_name: str) -> bool:
        """Enable a detection strategy."""
        return self._strategy_manager.enable_strategy(strategy_name)
    
    def disable_strategy(self, strategy_name: str) -> bool:
        """Disable a detection strategy."""
        return self._strategy_manager.disable_strategy(strategy_name)
    
    def get_statistics(self) -> DetectionStatistics:
        """Get detection statistics."""
        return self._statistics
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return {
            name: list(values) for name, values in self._performance_metrics.items()
        }
    
    def get_strategy_manager(self) -> StrategyManager:
        """Get strategy manager instance."""
        return self._strategy_manager
    
    def update_confidence_threshold(self, threshold: float):
        """Update confidence threshold."""
        if 0.0 <= threshold <= 1.0:
            self._confidence_threshold = threshold
            self.logger.info(f"Updated confidence threshold to {threshold}")
    
    def update_nms_threshold(self, threshold: float):
        """Update NMS threshold."""
        if 0.0 <= threshold <= 1.0:
            self._nms_threshold = threshold
            self.logger.info(f"Updated NMS threshold to {threshold}")


# Global detection engine instance
_global_engine: Optional[DetectionEngine] = None


def get_detection_engine(config: Optional[AppConfig] = None) -> Optional[DetectionEngine]:
    """Get global detection engine instance."""
    global _global_engine
    if _global_engine is None and config is not None:
        _global_engine = DetectionEngine(config)
    return _global_engine


def create_detection_frame(
    frame_number: int,
    source_id: str,
    gst_buffer: Any,
    batch_meta: Any,
    width: int,
    height: int,
    **metadata
) -> DetectionFrame:
    """
    Create a detection frame for processing.
    
    Args:
        frame_number: Frame number
        source_id: Source identifier
        gst_buffer: GStreamer buffer
        batch_meta: DeepStream batch metadata
        width: Frame width
        height: Frame height
        **metadata: Additional metadata
        
    Returns:
        DetectionFrame ready for processing
    """
    return DetectionFrame(
        frame_number=frame_number,
        source_id=source_id,
        timestamp=datetime.now(),
        gst_buffer=gst_buffer,
        batch_meta=batch_meta,
        width=width,
        height=height,
        metadata=metadata
    )