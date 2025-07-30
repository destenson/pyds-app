#!/usr/bin/env python3
"""
Performance Benchmarking Utility for PyDS System

This script provides comprehensive performance testing including:
- FPS benchmarking across multiple sources
- Memory and CPU usage profiling  
- GPU utilization monitoring
- Latency measurement
- Load testing with different configurations
- Optimization recommendations

Usage:
    python scripts/benchmark_performance.py --sources 4 --duration 60
    python scripts/benchmark_performance.py --profile --config configs/production.yaml
    python scripts/benchmark_performance.py --stress-test --max-sources 16
"""

import asyncio
import sys
import time
import json
import csv
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import statistics
import psutil
import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TaskID
from rich.live import Live

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.app import PyDSApp
from src.config import AppConfig, SourceConfig, SourceType, create_default_config
from src.utils.logging import setup_logging, LogLevel
from src.monitoring.metrics import MetricsCollector, get_metrics_collector
from src.monitoring.profiling import PerformanceProfiler

console = Console()


@dataclass
class BenchmarkResult:
    """Results from a performance benchmark."""
    test_name: str
    duration_seconds: float
    sources_count: int
    configuration: Dict[str, Any]
    
    # Performance metrics
    average_fps: float
    min_fps: float
    max_fps: float
    fps_std_dev: float
    
    # Resource usage
    average_cpu_percent: float
    peak_cpu_percent: float
    average_memory_mb: float
    peak_memory_mb: float
    average_gpu_percent: Optional[float] = None
    peak_gpu_percent: Optional[float] = None
    
    # Processing metrics
    total_frames: int
    total_detections: int
    total_alerts: int
    average_latency_ms: float
    max_latency_ms: float
    
    # Error metrics
    pipeline_errors: int
    source_reconnects: int
    dropped_frames: int
    
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return asdict(self)
    
    def get_performance_score(self) -> float:
        """Calculate overall performance score (0-100)."""
        # Base score from FPS (30 FPS = 50 points)
        fps_score = min(self.average_fps / 30.0 * 50, 50)
        
        # CPU efficiency (lower is better, 50% = 25 points)
        cpu_score = max(25 - (self.average_cpu_percent / 50.0 * 25), 0)
        
        # Memory efficiency (2GB = 15 points)
        memory_score = max(15 - (self.average_memory_mb / 2048.0 * 15), 0)
        
        # Stability (fewer errors = higher score)
        error_penalty = min(self.pipeline_errors * 2, 10)
        stability_score = max(10 - error_penalty, 0)
        
        return min(fps_score + cpu_score + memory_score + stability_score, 100)


class PerformanceBenchmark:
    """Performance benchmarking utility."""
    
    def __init__(self):
        self.console = Console()
        self.results: List[BenchmarkResult] = []
        self.current_metrics: Dict[str, List[float]] = {
            'fps': [],
            'cpu': [],
            'memory': [],
            'gpu': [],
            'latency': []
        }
    
    async def run_fps_benchmark(
        self,
        sources_count: int,
        duration: int,
        config: AppConfig = None
    ) -> BenchmarkResult:
        """
        Run FPS benchmark with specified number of sources.
        
        Args:
            sources_count: Number of video sources to test
            duration: Test duration in seconds
            config: Optional custom configuration
            
        Returns:
            Benchmark results
        """
        test_name = f"FPS_Benchmark_{sources_count}_sources"
        self.console.print(f"\n[blue]Running {test_name}[/blue]")
        
        # Create test configuration
        if config is None:
            config = self._create_benchmark_config(sources_count)
        
        # Reset metrics
        self.current_metrics = {key: [] for key in self.current_metrics}
        
        # Create and initialize app
        app = PyDSApp(config)
        start_time = time.time()
        
        try:
            await app.initialize()
            
            # Start monitoring task
            monitor_task = asyncio.create_task(
                self._monitor_performance(app, duration)
            )
            
            # Start application
            await app.start()
            
            # Wait for benchmark duration
            await asyncio.sleep(duration)
            
            # Stop monitoring
            monitor_task.cancel()
            
            # Collect final metrics
            status = app.get_status()
            
            # Calculate statistics
            fps_stats = self._calculate_stats(self.current_metrics['fps'])
            cpu_stats = self._calculate_stats(self.current_metrics['cpu'])
            memory_stats = self._calculate_stats(self.current_metrics['memory'])
            latency_stats = self._calculate_stats(self.current_metrics['latency'])
            
            gpu_avg = gpu_peak = None
            if self.current_metrics['gpu']:
                gpu_stats = self._calculate_stats(self.current_metrics['gpu'])
                gpu_avg = gpu_stats['mean']
                gpu_peak = gpu_stats['max']
            
            # Create result
            result = BenchmarkResult(
                test_name=test_name,
                duration_seconds=time.time() - start_time,
                sources_count=sources_count,
                configuration=self._extract_config_summary(config),
                average_fps=fps_stats['mean'],
                min_fps=fps_stats['min'],
                max_fps=fps_stats['max'],
                fps_std_dev=fps_stats['std_dev'],
                average_cpu_percent=cpu_stats['mean'],
                peak_cpu_percent=cpu_stats['max'],
                average_memory_mb=memory_stats['mean'],
                peak_memory_mb=memory_stats['max'],
                average_gpu_percent=gpu_avg,
                peak_gpu_percent=gpu_peak,
                total_frames=status.total_frames if hasattr(status, 'total_frames') else 0,
                total_detections=status.total_detections,
                total_alerts=status.total_alerts,
                average_latency_ms=latency_stats['mean'],
                max_latency_ms=latency_stats['max'],
                pipeline_errors=status.failed_components,
                source_reconnects=0,  # TODO: Track from source manager
                dropped_frames=0  # TODO: Track from pipeline
            )
            
            self.results.append(result)
            return result
        
        finally:
            await app.stop()
    
    async def run_stress_test(
        self,
        max_sources: int,
        duration: int,
        step_size: int = 2
    ) -> List[BenchmarkResult]:
        """
        Run stress test with increasing number of sources.
        
        Args:
            max_sources: Maximum number of sources to test
            duration: Duration for each test step
            step_size: Increment between tests
            
        Returns:
            List of benchmark results
        """
        self.console.print(f"\n[red]Running Stress Test[/red] (1 to {max_sources} sources)")
        
        results = []
        
        with Progress() as progress:
            task = progress.add_task("Stress Testing...", total=max_sources)
            
            for sources in range(1, max_sources + 1, step_size):
                progress.update(task, description=f"Testing {sources} sources...")
                
                try:
                    result = await self.run_fps_benchmark(sources, duration)
                    results.append(result)
                    
                    # Check if system is overloaded
                    if result.average_fps < 10 or result.average_cpu_percent > 95:
                        self.console.print(f"[yellow]System overloaded at {sources} sources, stopping stress test[/yellow]")
                        break
                    
                    progress.advance(task, step_size)
                    
                    # Brief pause between tests
                    await asyncio.sleep(2)
                    
                except Exception as e:
                    self.console.print(f"[red]Error testing {sources} sources: {e}[/red]")
                    break
        
        return results
    
    async def run_memory_profile(
        self,
        duration: int,
        sources_count: int = 4
    ) -> BenchmarkResult:
        """
        Run detailed memory profiling.
        
        Args:
            duration: Profile duration in seconds
            sources_count: Number of sources to use
            
        Returns:
            Benchmark results with memory details
        """
        self.console.print(f"\n[green]Running Memory Profile[/green] ({duration}s)")
        
        # Enable memory profiling
        config = self._create_benchmark_config(sources_count)
        config.monitoring.profiling_enabled = True
        
        # Run with profiler
        app = PyDSApp(config)
        
        try:
            profiler = PerformanceProfiler(config)
            await profiler.start_profiling()
            
            # Run benchmark
            result = await self.run_fps_benchmark(sources_count, duration, config)
            
            # Get profiling results
            profile_data = await profiler.stop_profiling()
            
            # Add profiling information to result
            result.configuration['profiling'] = {
                'memory_peak_mb': profile_data.get('memory_peak_mb', 0),
                'memory_leaks': profile_data.get('memory_leaks', []),
                'hotspots': profile_data.get('cpu_hotspots', [])
            }
            
            return result
        
        finally:
            await app.stop()
    
    def _create_benchmark_config(self, sources_count: int) -> AppConfig:
        """Create optimized configuration for benchmarking."""
        # Create test sources
        sources = []
        for i in range(sources_count):
            source = SourceConfig(
                id=f"benchmark_source_{i}",
                name=f"Benchmark Source {i}",
                type=SourceType.TEST,
                uri="videotestsrc pattern=ball is-live=true"
            )
            sources.append(source)
        
        # Create configuration optimized for performance
        config = AppConfig(
            name="Performance Benchmark",
            sources=sources,
            pipeline={
                'batch_size': min(sources_count, 8),
                'width': 1280,  # Lower resolution for benchmarking
                'height': 720,
                'fps': 30.0,
                'gpu_id': 0,
                'enable_gpu_inference': True,
                'buffer_pool_size': 20,
                'processing_mode': 'batch'
            },
            detection={
                'confidence_threshold': 0.7,
                'max_objects': 20,  # Fewer objects for consistent performance
                'batch_inference': True
            },
            alerts={
                'enabled': False,  # Disable alerts for pure performance testing
                'handlers': []
            },
            monitoring={
                'enabled': True,
                'metrics_interval': 1,  # High frequency for detailed metrics
                'profiling_enabled': False
            }
        )
        
        return config
    
    async def _monitor_performance(self, app: PyDSApp, duration: int):
        """Monitor performance metrics during benchmark."""
        start_time = time.time()
        
        while time.time() - start_time < duration:
            try:
                # Get application status
                status = app.get_status()
                
                # System metrics
                process = psutil.Process()
                cpu_percent = process.cpu_percent()
                memory_mb = process.memory_info().rss / 1024 / 1024
                
                # Pipeline metrics
                pipeline_states = app.pipeline_manager.get_pipeline_states()
                
                total_fps = 0
                total_latency = 0
                active_pipelines = 0
                
                for state in pipeline_states:
                    if state.get('fps', 0) > 0:
                        total_fps += state['fps']
                        total_latency += state.get('latency_ms', 0)
                        active_pipelines += 1
                
                avg_fps = total_fps / max(active_pipelines, 1)
                avg_latency = total_latency / max(active_pipelines, 1)
                
                # Record metrics
                self.current_metrics['fps'].append(avg_fps)
                self.current_metrics['cpu'].append(cpu_percent)
                self.current_metrics['memory'].append(memory_mb)
                self.current_metrics['latency'].append(avg_latency)
                
                # GPU metrics (if available)
                try:
                    import pynvml
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self.current_metrics['gpu'].append(gpu_util.gpu)
                except:
                    pass
                
                await asyncio.sleep(1)  # Sample every second
                
            except Exception as e:
                # Don't let monitoring errors stop the benchmark
                pass
    
    def _calculate_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate statistics for a list of values."""
        if not values:
            return {'mean': 0, 'min': 0, 'max': 0, 'std_dev': 0}
        
        return {
            'mean': statistics.mean(values),
            'min': min(values),
            'max': max(values),
            'std_dev': statistics.stdev(values) if len(values) > 1 else 0
        }
    
    def _extract_config_summary(self, config: AppConfig) -> Dict[str, Any]:
        """Extract key configuration parameters."""
        return {
            'batch_size': config.pipeline.batch_size,
            'resolution': f"{config.pipeline.width}x{config.pipeline.height}",
            'target_fps': config.pipeline.fps,
            'gpu_enabled': config.pipeline.enable_gpu_inference,
            'detection_threshold': config.detection.confidence_threshold
        }
    
    def generate_report(self, output_file: str = None) -> str:
        """Generate comprehensive performance report."""
        if not self.results:
            return "No benchmark results available"
        
        # Create summary table
        table = Table(title="Performance Benchmark Results")
        table.add_column("Test")
        table.add_column("Sources")
        table.add_column("Avg FPS", justify="right")
        table.add_column("CPU %", justify="right")
        table.add_column("Memory MB", justify="right")
        table.add_column("Score", justify="right")
        
        for result in self.results:
            score = result.get_performance_score()
            score_color = "green" if score > 80 else "yellow" if score > 60 else "red"
            
            table.add_row(
                result.test_name,
                str(result.sources_count),
                f"{result.average_fps:.1f}",
                f"{result.average_cpu_percent:.1f}",
                f"{result.average_memory_mb:.0f}",
                f"[{score_color}]{score:.0f}[/{score_color}]"
            )
        
        # Export results if requested
        if output_file:
            self._export_results(output_file)
        
        return table
    
    def _export_results(self, output_file: str):
        """Export results to file."""
        output_path = Path(output_file)
        
        if output_path.suffix.lower() == '.json':
            # Export as JSON
            with open(output_path, 'w') as f:
                json.dump([r.to_dict() for r in self.results], f, indent=2)
        
        elif output_path.suffix.lower() == '.csv':
            # Export as CSV
            if self.results:
                with open(output_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=self.results[0].to_dict().keys())
                    writer.writeheader()
                    for result in self.results:
                        writer.writerow(result.to_dict())
        
        console.print(f"Results exported to: {output_path}")


@click.command()
@click.option('--sources', '-s', type=int, default=4,
              help='Number of video sources to test')
@click.option('--duration', '-d', type=int, default=60,
              help='Test duration in seconds')
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Custom configuration file')
@click.option('--stress-test', is_flag=True,
              help='Run stress test with increasing sources')
@click.option('--max-sources', type=int, default=16,
              help='Maximum sources for stress test')
@click.option('--profile', is_flag=True,
              help='Enable detailed profiling')
@click.option('--output', '-o', type=click.Path(),
              help='Output file for results (JSON or CSV)')
@click.option('--quiet', '-q', is_flag=True,
              help='Suppress verbose output')
def main(sources, duration, config, stress_test, max_sources, profile, output, quiet):
    """
    PyDS Performance Benchmarking Utility
    
    Run comprehensive performance tests on the PyDS video analytics system.
    """
    if not quiet:
        console.print("[bold blue]PyDS Performance Benchmark[/bold blue]\n")
    
    # Setup logging
    log_level = LogLevel.WARNING if quiet else LogLevel.INFO
    setup_logging(log_level)
    
    # Create benchmark runner
    benchmark = PerformanceBenchmark()
    
    async def run_benchmarks():
        try:
            if stress_test:
                # Run stress test
                await benchmark.run_stress_test(max_sources, duration)
            
            elif profile:
                # Run memory profile
                await benchmark.run_memory_profile(duration, sources)
            
            else:
                # Run standard FPS benchmark
                config_obj = None
                if config:
                    from src.config import load_config
                    config_obj = load_config(config)
                
                await benchmark.run_fps_benchmark(sources, duration, config_obj)
            
            # Generate and display report
            if not quiet:
                report = benchmark.generate_report(output)
                console.print(report)
            else:
                # Just export results
                if output:
                    benchmark._export_results(output)
        
        except KeyboardInterrupt:
            console.print("\n[yellow]Benchmark interrupted by user[/yellow]")
        except Exception as e:
            console.print(f"\n[red]Benchmark failed: {e}[/red]")
            import traceback
            traceback.print_exc()
    
    # Run benchmarks
    asyncio.run(run_benchmarks())


if __name__ == "__main__":
    main()