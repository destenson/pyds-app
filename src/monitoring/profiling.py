"""
Advanced performance profiling with bottleneck detection and optimization recommendations.

This module provides CPU/GPU utilization tracking, performance bottleneck identification,
and actionable optimization recommendations for system performance tuning.
"""

import asyncio
import time
import cProfile
import pstats
import io  
import threading
import psutil
import gc
from typing import Dict, List, Optional, Any, Callable, Union, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, deque, Counter
import statistics
import json
import tracemalloc
import linecache
import sys
import resource

try:
    import py_spy
    PY_SPY_AVAILABLE = True
except ImportError:
    PY_SPY_AVAILABLE = False

try:
    import memory_profiler
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False

try:
    import nvidia_ml_py3 as nvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

from ..config import AppConfig
from ..utils.errors import ProfilingError, handle_error
from ..utils.logging import get_logger, performance_context
from ..utils.async_utils import get_task_manager, PeriodicTaskRunner


class ProfilerType(Enum):
    """Types of profilers available."""
    CPROFILE = "cprofile"
    MEMORY = "memory"
    GPU = "gpu"
    SYSTEM = "system"
    CUSTOM = "custom"


class BottleneckType(Enum):
    """Types of performance bottlenecks."""
    CPU_BOUND = "cpu_bound"
    MEMORY_BOUND = "memory_bound"
    IO_BOUND = "io_bound"
    GPU_BOUND = "gpu_bound"
    NETWORK_BOUND = "network_bound"
    CONTENTION = "contention"
    ALGORITHMIC = "algorithmic"


class PerformanceLevel(Enum):
    """Performance levels for comparison."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class ProfilePoint:
    """Single profiling data point."""
    timestamp: float
    function_name: str
    module_name: str
    line_number: int
    cpu_time_ms: float
    wall_time_ms: float
    memory_mb: float = 0.0
    call_count: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BottleneckInfo:
    """Information about a detected bottleneck."""
    bottleneck_type: BottleneckType
    severity: PerformanceLevel
    location: str
    description: str
    impact_score: float
    recommendations: List[str]
    metrics: Dict[str, Any] = field(default_factory=dict)
    detected_at: float = field(default_factory=time.time)


@dataclass
class OptimizationRecommendation:
    """Optimization recommendation."""
    category: str
    priority: PerformanceLevel
    title: str
    description: str
    implementation_effort: str  # low, medium, high
    estimated_improvement: str  # percentage or description
    code_example: Optional[str] = None
    references: List[str] = field(default_factory=list)


@dataclass
class PerformanceReport:
    """Comprehensive performance analysis report."""
    timestamp: float
    duration_seconds: float
    total_samples: int
    cpu_utilization: Dict[str, float]
    memory_utilization: Dict[str, float]
    gpu_utilization: Dict[str, float]
    bottlenecks: List[BottleneckInfo]
    recommendations: List[OptimizationRecommendation]
    hot_spots: List[Dict[str, Any]]
    performance_trends: Dict[str, List[float]]
    summary: Dict[str, Any]


class CPUProfiler:
    """CPU performance profiler using cProfile."""
    
    def __init__(self, name: str):
        """Initialize CPU profiler."""
        self.name = name
        self.logger = get_logger(__name__)
        self._profiler: Optional[cProfile.Profile] = None
        self._profile_data: Dict[str, Any] = {}
        self._is_profiling = False
        self._lock = threading.Lock()
    
    def start_profiling(self):
        """Start CPU profiling."""
        with self._lock:
            if self._is_profiling:
                return
            
            self._profiler = cProfile.Profile()
            self._profiler.enable()
            self._is_profiling = True
            self.logger.debug(f"Started CPU profiling: {self.name}")
    
    def stop_profiling(self) -> Dict[str, Any]:
        """Stop CPU profiling and return results."""
        with self._lock:
            if not self._is_profiling or not self._profiler:
                return {}
            
            self._profiler.disable()
            self._is_profiling = False
            
            # Capture profile statistics
            stats_stream = io.StringIO()
            stats = pstats.Stats(self._profiler, stream=stats_stream)
            stats.sort_stats('cumulative')
            stats.print_stats(50)  # Top 50 functions
            
            # Parse profile data
            profile_data = self._parse_profile_stats(stats)
            self.logger.debug(f"Stopped CPU profiling: {self.name}")
            
            return profile_data
    
    def _parse_profile_stats(self, stats: pstats.Stats) -> Dict[str, Any]:
        """Parse cProfile statistics into structured data."""
        try:
            profile_data = {
                'total_calls': stats.total_calls,
                'total_time': stats.total_tt,
                'functions': []
            }
            
            # Extract function statistics
            for func_info, (cc, nc, tt, ct, callers) in stats.stats.items():
                filename, line_num, func_name = func_info
                
                function_data = {
                    'filename': filename,
                    'line_number': line_num,
                    'function_name': func_name,
                    'call_count': cc,
                    'native_calls': nc,
                    'total_time': tt,
                    'cumulative_time': ct,
                    'time_per_call': tt / cc if cc > 0 else 0,
                    'callers': len(callers) if callers else 0
                }
                
                profile_data['functions'].append(function_data)
            
            # Sort by cumulative time
            profile_data['functions'].sort(key=lambda x: x['cumulative_time'], reverse=True)
            
            return profile_data
        
        except Exception as e:
            self.logger.error(f"Error parsing profile stats: {e}")
            return {}


class MemoryProfiler:
    """Memory usage profiler."""
    
    def __init__(self, name: str):
        """Initialize memory profiler."""
        self.name = name
        self.logger = get_logger(__name__)
        self._tracemalloc_started = False
        self._memory_snapshots: List[Any] = []
        self._baseline_snapshot: Optional[Any] = None
    
    def start_profiling(self):
        """Start memory profiling."""
        try:
            if not self._tracemalloc_started:
                tracemalloc.start(10)  # Keep 10 frames
                self._tracemalloc_started = True
                self._baseline_snapshot = tracemalloc.take_snapshot()
                self.logger.debug(f"Started memory profiling: {self.name}")
        
        except Exception as e:
            self.logger.error(f"Error starting memory profiling: {e}")
    
    def stop_profiling(self) -> Dict[str, Any]:
        """Stop memory profiling and return results."""
        try:
            if not self._tracemalloc_started:
                return {}
            
            # Take final snapshot
            current_snapshot = tracemalloc.take_snapshot()
            
            # Calculate memory differences
            memory_data = self._analyze_memory_usage(current_snapshot)
            
            tracemalloc.stop()
            self._tracemalloc_started = False
            
            self.logger.debug(f"Stopped memory profiling: {self.name}")
            return memory_data
        
        except Exception as e:
            self.logger.error(f"Error stopping memory profiling: {e}")
            return {}
    
    def _analyze_memory_usage(self, snapshot) -> Dict[str, Any]:
        """Analyze memory usage from snapshots."""
        try:
            memory_data = {
                'total_memory_mb': 0.0,
                'peak_memory_mb': 0.0,
                'top_allocators': [],
                'memory_trends': []
            }
            
            # Get current memory statistics
            if self._baseline_snapshot:
                top_stats = snapshot.compare_to(self._baseline_snapshot, 'lineno')
                
                total_size = 0
                for stat in top_stats[:20]:  # Top 20 allocators
                    total_size += stat.size
                    
                    allocator_info = {
                        'filename': stat.traceback.format()[0] if stat.traceback else 'unknown',
                        'size_mb': stat.size / (1024 * 1024),
                        'count': stat.count,
                        'size_diff_mb': stat.size_diff / (1024 * 1024) if hasattr(stat, 'size_diff') else 0
                    }
                    memory_data['top_allocators'].append(allocator_info)
                
                memory_data['total_memory_mb'] = total_size / (1024 * 1024)
            
            # Get system memory info
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_data['peak_memory_mb'] = memory_info.rss / (1024 * 1024)
            
            return memory_data
        
        except Exception as e:
            self.logger.error(f"Error analyzing memory usage: {e}")
            return {}


class GPUProfiler:
    """GPU performance profiler."""
    
    def __init__(self, name: str):
        """Initialize GPU profiler."""
        self.name = name
        self.logger = get_logger(__name__)
        self._profiling_data: List[Dict[str, Any]] = []
        self._is_profiling = False
        self._nvml_initialized = False
        
        # Initialize NVML if available
        if NVML_AVAILABLE:
            try:
                nvml.nvmlInit()
                self._nvml_initialized = True
            except Exception as e:
                self.logger.warning(f"Failed to initialize NVML: {e}")
    
    def start_profiling(self):
        """Start GPU profiling."""
        self._is_profiling = True
        self._profiling_data.clear()
        self.logger.debug(f"Started GPU profiling: {self.name}")
    
    def stop_profiling(self) -> Dict[str, Any]:
        """Stop GPU profiling and return results."""
        self._is_profiling = False
        
        if not self._profiling_data:
            return {}
        
        # Analyze GPU profiling data
        gpu_data = self._analyze_gpu_data()
        self.logger.debug(f"Stopped GPU profiling: {self.name}")
        
        return gpu_data
    
    def sample_gpu_metrics(self):
        """Sample current GPU metrics."""
        if not self._is_profiling or not self._nvml_initialized:
            return
        
        try:
            device_count = nvml.nvmlDeviceGetCount()
            timestamp = time.time()
            
            for i in range(device_count):
                handle = nvml.nvmlDeviceGetHandleByIndex(i)
                
                # Get GPU metrics
                util = nvml.nvmlDeviceGetUtilizationRates(handle)
                memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                temperature = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                
                try:
                    power = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                except:
                    power = 0.0
                
                gpu_sample = {
                    'timestamp': timestamp,
                    'gpu_id': i,
                    'utilization_percent': util.gpu,
                    'memory_utilization_percent': util.memory,
                    'memory_used_mb': memory_info.used / (1024 * 1024),
                    'memory_total_mb': memory_info.total / (1024 * 1024),
                    'temperature_c': temperature,
                    'power_watts': power
                }
                
                self._profiling_data.append(gpu_sample)
        
        except Exception as e:
            self.logger.error(f"Error sampling GPU metrics: {e}")
    
    def _analyze_gpu_data(self) -> Dict[str, Any]:
        """Analyze collected GPU profiling data."""
        if not self._profiling_data:
            return {}
        
        try:
            # Group data by GPU ID
            gpu_data_by_id = defaultdict(list)
            for sample in self._profiling_data:
                gpu_data_by_id[sample['gpu_id']].append(sample)
            
            analysis = {
                'gpu_count': len(gpu_data_by_id),
                'gpus': {}
            }
            
            for gpu_id, samples in gpu_data_by_id.items():
                utilizations = [s['utilization_percent'] for s in samples]
                memory_utils = [s['memory_utilization_percent'] for s in samples]
                temperatures = [s['temperature_c'] for s in samples]
                
                gpu_analysis = {
                    'avg_utilization': statistics.mean(utilizations),
                    'max_utilization': max(utilizations),
                    'avg_memory_utilization': statistics.mean(memory_utils),
                    'max_memory_utilization': max(memory_utils),
                    'avg_temperature': statistics.mean(temperatures),
                    'max_temperature': max(temperatures),
                    'total_samples': len(samples)
                }
                
                analysis['gpus'][gpu_id] = gpu_analysis
            
            return analysis
        
        except Exception as e:
            self.logger.error(f"Error analyzing GPU data: {e}")
            return {}


class BottleneckDetector:
    """Detects performance bottlenecks and provides recommendations."""
    
    def __init__(self):
        """Initialize bottleneck detector."""
        self.logger = get_logger(__name__)
        
        # Thresholds for bottleneck detection
        self._cpu_threshold = 80.0  # CPU usage percentage
        self._memory_threshold = 85.0  # Memory usage percentage
        self._gpu_threshold = 90.0  # GPU usage percentage
        self._io_wait_threshold = 20.0  # IO wait percentage
    
    def analyze_performance_data(self, 
                                cpu_data: Dict[str, Any],
                                memory_data: Dict[str, Any],
                                gpu_data: Dict[str, Any],
                                system_data: Dict[str, Any]) -> List[BottleneckInfo]:
        """Analyze performance data and detect bottlenecks."""
        bottlenecks = []
        
        # CPU bottleneck detection
        cpu_bottlenecks = self._detect_cpu_bottlenecks(cpu_data, system_data)
        bottlenecks.extend(cpu_bottlenecks)
        
        # Memory bottleneck detection
        memory_bottlenecks = self._detect_memory_bottlenecks(memory_data, system_data)
        bottlenecks.extend(memory_bottlenecks)
        
        # GPU bottleneck detection
        gpu_bottlenecks = self._detect_gpu_bottlenecks(gpu_data)
        bottlenecks.extend(gpu_bottlenecks)
        
        # IO bottleneck detection
        io_bottlenecks = self._detect_io_bottlenecks(system_data)
        bottlenecks.extend(io_bottlenecks)
        
        return bottlenecks
    
    def _detect_cpu_bottlenecks(self, cpu_data: Dict[str, Any], system_data: Dict[str, Any]) -> List[BottleneckInfo]:
        """Detect CPU-related bottlenecks."""
        bottlenecks = []
        
        try:
            # Check overall CPU usage
            cpu_percent = system_data.get('cpu_percent', 0)
            if cpu_percent > self._cpu_threshold:
                severity = PerformanceLevel.CRITICAL if cpu_percent > 95 else PerformanceLevel.POOR
                
                bottleneck = BottleneckInfo(
                    bottleneck_type=BottleneckType.CPU_BOUND,
                    severity=severity,
                    location="system_cpu",
                    description=f"High CPU utilization: {cpu_percent:.1f}%",
                    impact_score=cpu_percent / 100.0,
                    recommendations=[
                        "Consider reducing processing load",
                        "Optimize CPU-intensive algorithms",
                        "Use multi-threading for parallel processing",
                        "Profile and optimize hot code paths"
                    ],
                    metrics={'cpu_percent': cpu_percent}
                )
                bottlenecks.append(bottleneck)
            
            # Check for specific function hotspots
            if 'functions' in cpu_data:
                for func in cpu_data['functions'][:5]:  # Top 5 functions
                    if func['cumulative_time'] > 1.0:  # More than 1 second
                        bottleneck = BottleneckInfo(
                            bottleneck_type=BottleneckType.ALGORITHMIC,
                            severity=PerformanceLevel.FAIR,
                            location=f"{func['filename']}:{func['function_name']}",
                            description=f"Function consuming {func['cumulative_time']:.2f}s CPU time",
                            impact_score=func['cumulative_time'] / 10.0,
                            recommendations=[
                                f"Optimize {func['function_name']} function",
                                "Consider algorithmic improvements",
                                "Profile function internals",
                                "Add caching if applicable"
                            ],
                            metrics=func
                        )
                        bottlenecks.append(bottleneck)
        
        except Exception as e:
            self.logger.error(f"Error detecting CPU bottlenecks: {e}")
        
        return bottlenecks
    
    def _detect_memory_bottlenecks(self, memory_data: Dict[str, Any], system_data: Dict[str, Any]) -> List[BottleneckInfo]:
        """Detect memory-related bottlenecks."""
        bottlenecks = []
        
        try:
            # Check system memory usage
            memory_percent = system_data.get('memory_percent', 0)
            if memory_percent > self._memory_threshold:
                severity = PerformanceLevel.CRITICAL if memory_percent > 95 else PerformanceLevel.POOR
                
                bottleneck = BottleneckInfo(
                    bottleneck_type=BottleneckType.MEMORY_BOUND,
                    severity=severity,
                    location="system_memory",
                    description=f"High memory utilization: {memory_percent:.1f}%",
                    impact_score=memory_percent / 100.0,
                    recommendations=[
                        "Optimize memory usage patterns",
                        "Implement object pooling",
                        "Reduce memory allocation frequency",
                        "Consider memory profiling for leaks"
                    ],
                    metrics={'memory_percent': memory_percent}
                )
                bottlenecks.append(bottleneck)
            
            # Check for memory leaks
            if 'top_allocators' in memory_data:
                for allocator in memory_data['top_allocators'][:3]:
                    if allocator['size_mb'] > 100:  # More than 100MB
                        bottleneck = BottleneckInfo(
                            bottleneck_type=BottleneckType.MEMORY_BOUND,
                            severity=PerformanceLevel.FAIR,
                            location=allocator['filename'],
                            description=f"Large memory allocation: {allocator['size_mb']:.1f}MB",
                            impact_score=allocator['size_mb'] / 1000.0,
                            recommendations=[
                                "Review memory allocation patterns",
                                "Consider streaming or chunked processing",
                                "Implement garbage collection optimization",
                                "Use memory-efficient data structures"
                            ],
                            metrics=allocator
                        )
                        bottlenecks.append(bottleneck)
        
        except Exception as e:
            self.logger.error(f"Error detecting memory bottlenecks: {e}")
        
        return bottlenecks
    
    def _detect_gpu_bottlenecks(self, gpu_data: Dict[str, Any]) -> List[BottleneckInfo]:
        """Detect GPU-related bottlenecks."""
        bottlenecks = []
        
        try:
            if 'gpus' not in gpu_data:
                return bottlenecks
            
            for gpu_id, gpu_info in gpu_data['gpus'].items():
                # Check GPU utilization
                gpu_util = gpu_info.get('avg_utilization', 0)
                if gpu_util > self._gpu_threshold:
                    severity = PerformanceLevel.CRITICAL if gpu_util > 95 else PerformanceLevel.POOR
                    
                    bottleneck = BottleneckInfo(
                        bottleneck_type=BottleneckType.GPU_BOUND,
                        severity=severity,
                        location=f"gpu_{gpu_id}",
                        description=f"High GPU utilization: {gpu_util:.1f}%",
                        impact_score=gpu_util / 100.0,
                        recommendations=[
                            "Optimize GPU kernel execution",
                            "Reduce batch sizes if memory constrained",
                            "Consider multiple GPU processing",
                            "Profile GPU memory usage patterns"
                        ],
                        metrics=gpu_info
                    )
                    bottlenecks.append(bottleneck)
                
                # Check GPU memory utilization
                gpu_mem_util = gpu_info.get('avg_memory_utilization', 0)
                if gpu_mem_util > 90:
                    bottleneck = BottleneckInfo(
                        bottleneck_type=BottleneckType.GPU_BOUND,
                        severity=PerformanceLevel.POOR,
                        location=f"gpu_{gpu_id}_memory",
                        description=f"High GPU memory utilization: {gpu_mem_util:.1f}%",
                        impact_score=gpu_mem_util / 100.0,
                        recommendations=[
                            "Reduce GPU memory usage",
                            "Optimize tensor sizes",
                            "Use gradient checkpointing",
                            "Consider model quantization"
                        ],
                        metrics=gpu_info
                    )
                    bottlenecks.append(bottleneck)
        
        except Exception as e:
            self.logger.error(f"Error detecting GPU bottlenecks: {e}")
        
        return bottlenecks
    
    def _detect_io_bottlenecks(self, system_data: Dict[str, Any]) -> List[BottleneckInfo]:
        """Detect IO-related bottlenecks."""
        bottlenecks = []
        
        try:
            # Check disk usage
            disk_percent = system_data.get('disk_percent', 0)
            if disk_percent > 90:
                bottleneck = BottleneckInfo(
                    bottleneck_type=BottleneckType.IO_BOUND,
                    severity=PerformanceLevel.POOR,
                    location="disk_storage",
                    description=f"High disk utilization: {disk_percent:.1f}%",
                    impact_score=disk_percent / 100.0,
                    recommendations=[
                        "Clean up temporary files",
                        "Archive old data",
                        "Consider faster storage",
                        "Implement data compression"
                    ],
                    metrics={'disk_percent': disk_percent}
                )
                bottlenecks.append(bottleneck)
        
        except Exception as e:
            self.logger.error(f"Error detecting IO bottlenecks: {e}")
        
        return bottlenecks


class RecommendationEngine:
    """Generates optimization recommendations based on performance analysis."""
    
    def __init__(self):
        """Initialize recommendation engine."""
        self.logger = get_logger(__name__)
    
    def generate_recommendations(self, 
                                bottlenecks: List[BottleneckInfo],
                                performance_data: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Generate recommendations based on bottlenecks
        for bottleneck in bottlenecks:
            bottleneck_recommendations = self._get_bottleneck_recommendations(bottleneck)
            recommendations.extend(bottleneck_recommendations)
        
        # Generate general recommendations
        general_recommendations = self._get_general_recommendations(performance_data)
        recommendations.extend(general_recommendations)
        
        # Sort by priority and remove duplicates
        recommendations = self._deduplicate_recommendations(recommendations)
        recommendations.sort(key=lambda x: self._priority_score(x.priority), reverse=True)
        
        return recommendations
    
    def _get_bottleneck_recommendations(self, bottleneck: BottleneckInfo) -> List[OptimizationRecommendation]:
        """Get recommendations for specific bottleneck."""
        recommendations = []
        
        if bottleneck.bottleneck_type == BottleneckType.CPU_BOUND:
            recommendations.extend([
                OptimizationRecommendation(
                    category="CPU Optimization",
                    priority=PerformanceLevel.CRITICAL,
                    title="Implement Multi-threading",
                    description="Use threading or multiprocessing to parallelize CPU-intensive tasks",
                    implementation_effort="medium",
                    estimated_improvement="20-40%",
                    code_example="""
import concurrent.futures
import threading

def process_parallel(data_chunks):
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_chunk, chunk) for chunk in data_chunks]
        results = [future.result() for future in futures]
    return results
""",
                    references=["https://docs.python.org/3/library/concurrent.futures.html"]
                ),
                OptimizationRecommendation(
                    category="Algorithm Optimization",
                    priority=PerformanceLevel.GOOD,
                    title="Optimize Hot Code Paths",
                    description="Profile and optimize the most frequently executed code paths",
                    implementation_effort="high",
                    estimated_improvement="10-30%"
                )
            ])
        
        elif bottleneck.bottleneck_type == BottleneckType.MEMORY_BOUND:
            recommendations.extend([
                OptimizationRecommendation(
                    category="Memory Management",
                    priority=PerformanceLevel.CRITICAL,
                    title="Implement Object Pooling",
                    description="Reuse objects to reduce memory allocation overhead",
                    implementation_effort="medium",
                    estimated_improvement="15-25%",
                    code_example="""
class ObjectPool:
    def __init__(self, create_func, max_size=100):
        self._create_func = create_func
        self._pool = []
        self._max_size = max_size
    
    def get(self):
        if self._pool:
            return self._pool.pop()
        return self._create_func()
    
    def put(self, obj):
        if len(self._pool) < self._max_size:
            self._pool.append(obj)
"""
                ),
                OptimizationRecommendation(
                    category="Memory Management",
                    priority=PerformanceLevel.GOOD,
                    title="Use Memory Mapping",
                    description="Use memory-mapped files for large datasets",
                    implementation_effort="low",
                    estimated_improvement="10-20%"
                )
            ])
        
        elif bottleneck.bottleneck_type == BottleneckType.GPU_BOUND:
            recommendations.extend([
                OptimizationRecommendation(
                    category="GPU Optimization",
                    priority=PerformanceLevel.CRITICAL,
                    title="Optimize Batch Processing",
                    description="Adjust batch sizes for optimal GPU utilization",
                    implementation_effort="low",
                    estimated_improvement="20-50%"
                ),
                OptimizationRecommendation(
                    category="GPU Optimization",
                    priority=PerformanceLevel.GOOD,
                    title="Use Mixed Precision",
                    description="Implement mixed precision training/inference",
                    implementation_effort="medium",
                    estimated_improvement="30-60%"
                )
            ])
        
        return recommendations
    
    def _get_general_recommendations(self, performance_data: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """Get general optimization recommendations."""
        recommendations = [
            OptimizationRecommendation(
                category="General",
                priority=PerformanceLevel.FAIR,
                title="Enable Performance Monitoring",
                description="Implement continuous performance monitoring",
                implementation_effort="low",
                estimated_improvement="Better visibility"
            ),
            OptimizationRecommendation(
                category="General",
                priority=PerformanceLevel.FAIR,
                title="Implement Caching",
                description="Add caching for frequently accessed data",
                implementation_effort="medium",
                estimated_improvement="10-30%"
            ),
            OptimizationRecommendation(
                category="General",
                priority=PerformanceLevel.FAIR,
                title="Optimize Configuration",
                description="Review and optimize system configuration parameters",
                implementation_effort="low",
                estimated_improvement="5-15%"
            )
        ]
        
        return recommendations
    
    def _deduplicate_recommendations(self, recommendations: List[OptimizationRecommendation]) -> List[OptimizationRecommendation]:
        """Remove duplicate recommendations."""
        seen_titles = set()
        unique_recommendations = []
        
        for rec in recommendations:
            if rec.title not in seen_titles:
                seen_titles.add(rec.title)
                unique_recommendations.append(rec)
        
        return unique_recommendations
    
    def _priority_score(self, priority: PerformanceLevel) -> int:
        """Convert priority to numeric score."""
        priority_scores = {
            PerformanceLevel.CRITICAL: 5,
            PerformanceLevel.POOR: 4,
            PerformanceLevel.FAIR: 3,
            PerformanceLevel.GOOD: 2,
            PerformanceLevel.EXCELLENT: 1
        }
        return priority_scores.get(priority, 0)


class PerformanceProfiler:
    """
    Comprehensive performance profiler with bottleneck detection.
    
    Provides CPU/GPU utilization tracking, performance bottleneck identification,
    and actionable optimization recommendations for system performance tuning.
    """
    
    def __init__(self, config: AppConfig):
        """
        Initialize performance profiler.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Core components
        self._bottleneck_detector = BottleneckDetector()
        self._recommendation_engine = RecommendationEngine()
        
        # Profiler instances
        self._profilers: Dict[str, Union[CPUProfiler, MemoryProfiler, GPUProfiler]] = {}
        self._active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self._performance_history: deque = deque(maxlen=1000)
        self._baseline_metrics: Optional[Dict[str, Any]] = None
        
        # Configuration
        self._profiling_enabled = False
        self._auto_profiling_interval = 300.0  # 5 minutes
        self._report_generation_interval = 3600.0  # 1 hour
        
        # Periodic tasks
        self._periodic_runner = PeriodicTaskRunner()
        
        # Thread safety
        self._lock = threading.RLock()
        
        self.logger.info("PerformanceProfiler initialized")
    
    async def start(self) -> bool:
        """
        Start performance profiling.
        
        Returns:
            True if started successfully
        """
        try:
            self.logger.info("Starting PerformanceProfiler...")
            
            # Capture baseline metrics
            await self._capture_baseline_metrics()
            
            # Start periodic profiling
            await self._start_periodic_profiling()
            
            self._profiling_enabled = True
            
            self.logger.info("PerformanceProfiler started successfully")
            return True
        
        except Exception as e:
            error = handle_error(e, context={'component': 'performance_profiler'})
            self.logger.error(f"Failed to start PerformanceProfiler: {error}")
            return False
    
    async def stop(self) -> bool:
        """
        Stop performance profiling.
        
        Returns:
            True if stopped successfully
        """
        try:
            self.logger.info("Stopping PerformanceProfiler...")
            
            self._profiling_enabled = False
            
            # Stop all active profiling sessions
            await self._stop_all_sessions()
            
            # Stop periodic tasks
            await self._periodic_runner.stop_all_tasks()
            
            # Generate final report
            await self._generate_final_report()
            
            self.logger.info("PerformanceProfiler stopped successfully")
            return True
        
        except Exception as e:
            self.logger.error(f"Error stopping PerformanceProfiler: {e}")
            return False
    
    async def _capture_baseline_metrics(self):
        """Capture baseline performance metrics."""
        try:
            baseline_metrics = {
                'timestamp': time.time(),
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory': psutil.virtual_memory()._asdict(),
                'disk': psutil.disk_usage('/')._asdict(),
                'network': psutil.net_io_counters()._asdict(),
                'process': psutil.Process().as_dict()
            }
            
            self._baseline_metrics = baseline_metrics
            self.logger.info("Captured baseline performance metrics")
        
        except Exception as e:
            self.logger.error(f"Error capturing baseline metrics: {e}")
    
    async def _start_periodic_profiling(self):
        """Start periodic profiling tasks."""
        # Auto profiling task
        await self._periodic_runner.start_periodic_task(
            "auto_profiling",
            self._auto_profiling_task,
            interval=self._auto_profiling_interval
        )
        
        # Report generation task
        await self._periodic_runner.start_periodic_task(
            "report_generation",
            self._generate_performance_report,
            interval=self._report_generation_interval
        )
        
        # GPU sampling task (if available)
        if NVML_AVAILABLE:
            await self._periodic_runner.start_periodic_task(
                "gpu_sampling",
                self._sample_gpu_metrics,
                interval=5.0  # Every 5 seconds
            )
    
    async def _auto_profiling_task(self):
        """Automatic profiling task."""
        try:
            session_id = f"auto_{int(time.time())}"
            
            # Start profiling session
            await self.start_profiling_session(session_id, [
                ProfilerType.CPROFILE,
                ProfilerType.MEMORY,
                ProfilerType.GPU
            ])
            
            # Let it run for 60 seconds
            await asyncio.sleep(60)
            
            # Stop and analyze
            report = await self.stop_profiling_session(session_id)
            
            if report:
                # Store in history
                self._performance_history.append({
                    'timestamp': time.time(),
                    'session_id': session_id,
                    'report': report
                })
                
                # Check for critical bottlenecks
                critical_bottlenecks = [
                    b for b in report.bottlenecks 
                    if b.severity == PerformanceLevel.CRITICAL
                ]
                
                if critical_bottlenecks:
                    self.logger.warning(
                        f"Detected {len(critical_bottlenecks)} critical performance bottlenecks"
                    )
        
        except Exception as e:
            self.logger.error(f"Error in auto profiling task: {e}")
    
    async def _sample_gpu_metrics(self):
        """Sample GPU metrics for active GPU profilers."""
        for session_id, session_info in self._active_sessions.items():
            if 'gpu' in session_info['profilers']:
                gpu_profiler = session_info['profilers']['gpu']
                gpu_profiler.sample_gpu_metrics()
    
    async def start_profiling_session(self, 
                                     session_id: str, 
                                     profiler_types: List[ProfilerType],
                                     duration_seconds: Optional[float] = None) -> bool:
        """
        Start a profiling session.
        
        Args:
            session_id: Unique session identifier
            profiler_types: List of profiler types to use
            duration_seconds: Optional duration limit
            
        Returns:
            True if session started successfully
        """
        try:
            with self._lock:
                if session_id in self._active_sessions:
                    self.logger.warning(f"Profiling session {session_id} already active")
                    return False
                
                # Create profilers
                profilers = {}
                for profiler_type in profiler_types:
                    if profiler_type == ProfilerType.CPROFILE:
                        profilers['cpu'] = CPUProfiler(f"{session_id}_cpu")
                    elif profiler_type == ProfilerType.MEMORY:
                        profilers['memory'] = MemoryProfiler(f"{session_id}_memory")
                    elif profiler_type == ProfilerType.GPU:
                        profilers['gpu'] = GPUProfiler(f"{session_id}_gpu")
                
                # Start all profilers
                for profiler in profilers.values():
                    profiler.start_profiling()
                
                # Store session info
                self._active_sessions[session_id] = {
                    'start_time': time.time(),
                    'duration_seconds': duration_seconds,
                    'profilers': profilers,
                    'profiler_types': profiler_types
                }
                
                self.logger.info(f"Started profiling session: {session_id}")
                
                # Auto-stop if duration specified
                if duration_seconds:
                    asyncio.create_task(self._auto_stop_session(session_id, duration_seconds))
                
                return True
        
        except Exception as e:
            self.logger.error(f"Error starting profiling session {session_id}: {e}")
            return False
    
    async def stop_profiling_session(self, session_id: str) -> Optional[PerformanceReport]:
        """
        Stop profiling session and generate report.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Performance report or None if failed
        """
        try:
            with self._lock:
                if session_id not in self._active_sessions:
                    self.logger.warning(f"Profiling session {session_id} not found")
                    return None
                
                session_info = self._active_sessions[session_id]
                start_time = session_info['start_time']
                duration = time.time() - start_time
                
                # Stop all profilers and collect data
                profiling_data = {}
                for profiler_name, profiler in session_info['profilers'].items():
                    profiling_data[profiler_name] = profiler.stop_profiling()
                
                # Remove session
                del self._active_sessions[session_id]
                
                self.logger.info(f"Stopped profiling session: {session_id}")
                
                # Generate performance report
                report = await self._generate_session_report(
                    session_id, duration, profiling_data
                )
                
                return report
        
        except Exception as e:
            self.logger.error(f"Error stopping profiling session {session_id}: {e}")
            return None
    
    async def _auto_stop_session(self, session_id: str, duration_seconds: float):
        """Automatically stop session after duration."""
        await asyncio.sleep(duration_seconds)
        await self.stop_profiling_session(session_id)
    
    async def _stop_all_sessions(self):
        """Stop all active profiling sessions."""
        session_ids = list(self._active_sessions.keys())
        for session_id in session_ids:
            await self.stop_profiling_session(session_id)
    
    async def _generate_session_report(self, 
                                       session_id: str,
                                       duration: float,
                                       profiling_data: Dict[str, Any]) -> PerformanceReport:
        """Generate comprehensive performance report for session."""
        try:
            # Get current system metrics
            system_data = {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': (psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100
            }
            
            # Detect bottlenecks
            bottlenecks = self._bottleneck_detector.analyze_performance_data(
                profiling_data.get('cpu', {}),
                profiling_data.get('memory', {}),
                profiling_data.get('gpu', {}),
                system_data
            )
            
            # Generate recommendations
            recommendations = self._recommendation_engine.generate_recommendations(
                bottlenecks, profiling_data
            )
            
            # Create performance report
            report = PerformanceReport(
                timestamp=time.time(),
                duration_seconds=duration,
                total_samples=sum(len(data.get('functions', [])) for data in profiling_data.values()),
                cpu_utilization=self._extract_cpu_utilization(profiling_data.get('cpu', {})),
                memory_utilization=self._extract_memory_utilization(profiling_data.get('memory', {})),
                gpu_utilization=self._extract_gpu_utilization(profiling_data.get('gpu', {})),
                bottlenecks=bottlenecks,
                recommendations=recommendations,
                hot_spots=self._identify_hot_spots(profiling_data),
                performance_trends=self._calculate_performance_trends(),
                summary=self._generate_summary(bottlenecks, recommendations)
            )
            
            # Save report
            await self._save_report(session_id, report)
            
            return report
        
        except Exception as e:
            self.logger.error(f"Error generating session report: {e}")
            return PerformanceReport(
                timestamp=time.time(),
                duration_seconds=duration,
                total_samples=0,
                cpu_utilization={},
                memory_utilization={},
                gpu_utilization={},
                bottlenecks=[],
                recommendations=[],
                hot_spots=[],
                performance_trends={},
                summary={}
            )
    
    def _extract_cpu_utilization(self, cpu_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract CPU utilization metrics."""
        cpu_metrics = {}
        
        if 'functions' in cpu_data:
            total_time = cpu_data.get('total_time', 0)
            cpu_metrics['total_time'] = total_time
            
            # Top functions by time
            functions = cpu_data['functions'][:10]
            for i, func in enumerate(functions):
                cpu_metrics[f'top_function_{i+1}_time'] = func.get('cumulative_time', 0)
        
        return cpu_metrics
    
    def _extract_memory_utilization(self, memory_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract memory utilization metrics."""
        memory_metrics = {}
        
        if 'total_memory_mb' in memory_data:
            memory_metrics['total_memory_mb'] = memory_data['total_memory_mb']
        
        if 'peak_memory_mb' in memory_data:
            memory_metrics['peak_memory_mb'] = memory_data['peak_memory_mb']
        
        return memory_metrics
    
    def _extract_gpu_utilization(self, gpu_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract GPU utilization metrics."""
        gpu_metrics = {}
        
        if 'gpus' in gpu_data:
            for gpu_id, gpu_info in gpu_data['gpus'].items():
                gpu_metrics[f'gpu_{gpu_id}_avg_util'] = gpu_info.get('avg_utilization', 0)
                gpu_metrics[f'gpu_{gpu_id}_max_util'] = gpu_info.get('max_utilization', 0)
                gpu_metrics[f'gpu_{gpu_id}_avg_memory'] = gpu_info.get('avg_memory_utilization', 0)
        
        return gpu_metrics
    
    def _identify_hot_spots(self, profiling_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify performance hot spots."""
        hot_spots = []
        
        # CPU hot spots
        cpu_data = profiling_data.get('cpu', {})
        if 'functions' in cpu_data:
            for func in cpu_data['functions'][:5]:
                hot_spots.append({
                    'type': 'cpu',
                    'location': f"{func['filename']}:{func['function_name']}",
                    'metric': 'cumulative_time',
                    'value': func['cumulative_time'],
                    'impact': 'high' if func['cumulative_time'] > 1.0 else 'medium'
                })
        
        # Memory hot spots
        memory_data = profiling_data.get('memory', {})
        if 'top_allocators' in memory_data:
            for allocator in memory_data['top_allocators'][:3]:
                hot_spots.append({
                    'type': 'memory',
                    'location': allocator['filename'],
                    'metric': 'size_mb',
                    'value': allocator['size_mb'],
                    'impact': 'high' if allocator['size_mb'] > 100 else 'medium'
                })
        
        return hot_spots
    
    def _calculate_performance_trends(self) -> Dict[str, List[float]]:
        """Calculate performance trends from history."""
        trends = defaultdict(list)
        
        for entry in self._performance_history:
            report = entry.get('report')
            if report:
                # CPU trend
                cpu_util = sum(report.cpu_utilization.values())
                trends['cpu_utilization'].append(cpu_util)
                
                # Memory trend
                memory_util = sum(report.memory_utilization.values())
                trends['memory_utilization'].append(memory_util)
                
                # Bottleneck count trend
                trends['bottleneck_count'].append(len(report.bottlenecks))
        
        return dict(trends)
    
    def _generate_summary(self, 
                         bottlenecks: List[BottleneckInfo],
                         recommendations: List[OptimizationRecommendation]) -> Dict[str, Any]:
        """Generate performance summary."""
        summary = {
            'overall_performance': 'good',
            'bottleneck_count': len(bottlenecks),
            'critical_issues': len([b for b in bottlenecks if b.severity == PerformanceLevel.CRITICAL]),
            'recommendation_count': len(recommendations),
            'high_priority_recommendations': len([r for r in recommendations if r.priority == PerformanceLevel.CRITICAL])
        }
        
        # Determine overall performance level
        critical_count = summary['critical_issues']
        if critical_count > 2:
            summary['overall_performance'] = 'critical'
        elif critical_count > 0:
            summary['overall_performance'] = 'poor'
        elif len(bottlenecks) > 5:
            summary['overall_performance'] = 'fair'
        
        return summary
    
    async def _save_report(self, session_id: str, report: PerformanceReport):
        """Save performance report to file."""
        try:
            report_path = Path(f"data/performance_reports/report_{session_id}.json")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert report to dictionary
            report_dict = {
                'timestamp': report.timestamp,
                'duration_seconds': report.duration_seconds,
                'total_samples': report.total_samples,
                'cpu_utilization': report.cpu_utilization,
                'memory_utilization': report.memory_utilization,
                'gpu_utilization': report.gpu_utilization,
                'bottlenecks': [asdict(b) for b in report.bottlenecks],
                'recommendations': [asdict(r) for r in report.recommendations],
                'hot_spots': report.hot_spots,
                'performance_trends': report.performance_trends,
                'summary': report.summary
            }
            
            with open(report_path, 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)
            
            self.logger.info(f"Saved performance report: {report_path}")
        
        except Exception as e:
            self.logger.error(f"Error saving report: {e}")
    
    async def _generate_performance_report(self):
        """Generate periodic performance report."""
        try:
            if not self._performance_history:
                return
            
            # Analyze recent performance data
            recent_reports = list(self._performance_history)[-10:]  # Last 10 reports
            
            # Generate trend analysis
            trend_analysis = self._analyze_performance_trends(recent_reports)
            
            self.logger.info(f"Generated performance trend analysis: {trend_analysis}")
        
        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}")
    
    def _analyze_performance_trends(self, reports: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance trends from multiple reports."""
        if not reports:
            return {}
        
        trend_analysis = {
            'period_start': reports[0]['timestamp'],
            'period_end': reports[-1]['timestamp'],
            'report_count': len(reports),
            'trends': {}
        }
        
        # Analyze bottleneck trends
        bottleneck_counts = [len(r['report'].bottlenecks) for r in reports]
        if bottleneck_counts:
            trend_analysis['trends']['bottlenecks'] = {
                'average': statistics.mean(bottleneck_counts),
                'trend': 'increasing' if bottleneck_counts[-1] > bottleneck_counts[0] else 'decreasing'
            }
        
        return trend_analysis
    
    async def _generate_final_report(self):
        """Generate final comprehensive report."""
        try:
            final_report_path = Path("data/performance_final_report.json")
            final_report_path.parent.mkdir(parents=True, exist_ok=True)
            
            final_report = {
                'timestamp': time.time(),
                'total_sessions': len(self._performance_history),
                'baseline_metrics': self._baseline_metrics,
                'performance_history': [
                    {
                        'timestamp': entry['timestamp'],
                        'session_id': entry['session_id'],
                        'summary': entry['report'].summary
                    } for entry in self._performance_history
                ],
                'overall_trends': self._calculate_performance_trends()
            }
            
            with open(final_report_path, 'w') as f:
                json.dump(final_report, f, indent=2, default=str)
            
            self.logger.info(f"Generated final performance report: {final_report_path}")
        
        except Exception as e:
            self.logger.error(f"Error generating final report: {e}")
    
    # Public API
    
    def get_active_sessions(self) -> List[str]:
        """Get list of active profiling sessions."""
        with self._lock:
            return list(self._active_sessions.keys())
    
    def get_performance_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent performance history."""
        return list(self._performance_history)[-limit:]
    
    def get_baseline_metrics(self) -> Optional[Dict[str, Any]]:
        """Get baseline performance metrics."""
        return self._baseline_metrics
    
    async def run_quick_profile(self, duration_seconds: float = 30) -> Optional[PerformanceReport]:
        """Run a quick performance profile."""
        session_id = f"quick_{int(time.time())}"
        
        success = await self.start_profiling_session(
            session_id,
            [ProfilerType.CPROFILE, ProfilerType.MEMORY],
            duration_seconds
        )
        
        if success:
            await asyncio.sleep(duration_seconds)
            return await self.stop_profiling_session(session_id)
        
        return None


# Global performance profiler instance
_global_profiler: Optional[PerformanceProfiler] = None


def get_performance_profiler(config: Optional[AppConfig] = None) -> Optional[PerformanceProfiler]:
    """Get global performance profiler instance."""
    global _global_profiler
    if _global_profiler is None and config is not None:
        _global_profiler = PerformanceProfiler(config)
    return _global_profiler