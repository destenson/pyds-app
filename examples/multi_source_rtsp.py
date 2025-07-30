#!/usr/bin/env python3
"""
Multi-Source RTSP Streaming Example

This example demonstrates:
- Processing multiple RTSP streams simultaneously
- Load balancing across sources
- Automatic reconnection on stream failure
- Aggregated performance metrics
- Alert throttling for multiple sources

Usage:
    python examples/multi_source_rtsp.py
    python examples/multi_source_rtsp.py --sources camera1=rtsp://ip1/stream camera2=rtsp://ip2/stream
    python examples/multi_source_rtsp.py --config configs/multi_rtsp.yaml
"""

import asyncio
import sys
from pathlib import Path
import click
from datetime import datetime
from typing import Dict, List
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.app import PyDSApp
from src.config import AppConfig, SourceConfig, SourceType, load_config
from src.utils.logging import setup_logging, LogLevel
from src.pipeline.sources import SourceHealth


def create_rtsp_config(rtsp_sources: Dict[str, str]) -> AppConfig:
    """
    Create configuration for multiple RTSP sources.
    
    Args:
        rtsp_sources: Dictionary of source_name -> rtsp_url mappings
        
    Returns:
        AppConfig with RTSP sources configured
    """
    # Create source configurations
    sources = []
    for name, url in rtsp_sources.items():
        source = SourceConfig(
            id=f"rtsp_{name}",
            name=f"RTSP Camera: {name}",
            type=SourceType.RTSP,
            uri=url,
            enabled=True,
            max_retries=10,  # More retries for network sources
            retry_delay=5.0,
            parameters={
                'latency': 200,  # Buffer for network jitter
                'buffer-size': 2048,
                'protocols': 'tcp',  # Use TCP for reliability
                'timeout': 10000000  # 10 second timeout
            }
        )
        sources.append(source)
    
    # Create optimized configuration for multiple sources
    config = AppConfig(
        name="Multi-Source RTSP Analytics",
        environment="production",
        pipeline={
            'batch_size': min(len(sources), 8),  # Batch up to 8 sources
            'width': 1920,
            'height': 1080,
            'fps': 30.0,
            'gpu_id': 0,
            'enable_gpu_inference': True,
            'buffer_pool_size': 20,  # Larger pool for multiple sources
            'processing_mode': 'batch'
        },
        detection={
            'confidence_threshold': 0.7,
            'max_objects': 50,
            'batch_inference': True,
            'enable_tracking': True
        },
        alerts={
            'enabled': True,
            'throttle_seconds': 60,  # Throttle per source
            'burst_threshold': 5,
            'handlers': ['console', 'file'],
            'max_alerts_per_minute': 20  # Total across all sources
        },
        monitoring={
            'enabled': True,
            'metrics_interval': 10,
            'health_check_interval': 30,
            'cpu_threshold': 80.0,
            'memory_threshold': 85.0,
            'gpu_threshold': 90.0
        },
        sources=sources,
        max_concurrent_sources=len(sources),
        thread_pool_size=max(4, len(sources) // 2)
    )
    
    return config


async def monitor_sources(app: PyDSApp):
    """
    Monitor health and status of all video sources.
    
    Args:
        app: PyDS application instance
    """
    print("\n=== Source Health Monitor ===")
    
    while app.is_running():
        await asyncio.sleep(10)  # Check every 10 seconds
        
        source_manager = app.source_manager
        all_sources = source_manager.get_all_sources()
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Source Status:")
        print("-" * 60)
        print(f"{'Source ID':<20} {'Status':<15} {'Health':<10} {'FPS':<8} {'Frames':<10}")
        print("-" * 60)
        
        total_fps = 0
        total_frames = 0
        healthy_count = 0
        
        for source_id, source_state in all_sources.items():
            metrics = source_state.metrics
            health = source_manager.get_source_health(source_id)
            
            # Determine health symbol
            if health == SourceHealth.HEALTHY:
                health_symbol = "✓"
                healthy_count += 1
            elif health == SourceHealth.DEGRADED:
                health_symbol = "⚠"
            else:
                health_symbol = "✗"
            
            print(f"{source_id:<20} {source_state.status.value:<15} "
                  f"{health_symbol} {health.value:<9} {metrics.current_fps:>6.1f} "
                  f"{metrics.frames_received:>10}")
            
            total_fps += metrics.current_fps
            total_frames += metrics.frames_received
        
        print("-" * 60)
        print(f"Total: {healthy_count}/{len(all_sources)} healthy | "
              f"FPS: {total_fps:.1f} | Frames: {total_frames}")
        
        # Check for unhealthy sources
        unhealthy = [s for s, state in all_sources.items() 
                    if source_manager.get_source_health(s) == SourceHealth.UNHEALTHY]
        
        if unhealthy:
            print(f"\n⚠️  Unhealthy sources detected: {', '.join(unhealthy)}")
            print("   Automatic reconnection will be attempted...")


async def multi_rtsp_example(
    rtsp_sources: Dict[str, str] = None,
    config_path: str = None,
    duration: int = 0
):
    """
    Run multi-source RTSP streaming example.
    
    Args:
        rtsp_sources: Dictionary of RTSP sources
        config_path: Optional configuration file
        duration: Run duration in seconds (0 for infinite)
    """
    print("=== PyDS Multi-Source RTSP Example ===\n")
    
    # Setup logging
    setup_logging(LogLevel.INFO)
    
    # Load or create configuration
    if config_path:
        print(f"Loading configuration from: {config_path}")
        config = load_config(config_path)
    elif rtsp_sources:
        print(f"Configuring {len(rtsp_sources)} RTSP sources")
        config = create_rtsp_config(rtsp_sources)
    else:
        # Default example sources (public test streams)
        print("Using example RTSP sources (public test streams)")
        default_sources = {
            "example1": "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4",
            "example2": "rtsp://demo:demo@ipvmdemo.dyndns.org:5541/onvif-media/media.amp",
        }
        config = create_rtsp_config(default_sources)
    
    print(f"\nConfigured Sources:")
    for source in config.sources:
        print(f"  - {source.name}: {source.uri}")
    
    print(f"\nPipeline Configuration:")
    print(f"  Batch Size: {config.pipeline.batch_size}")
    print(f"  Resolution: {config.pipeline.width}x{config.pipeline.height}")
    print(f"  Target FPS: {config.pipeline.fps}")
    
    # Create application
    app = PyDSApp(config)
    
    try:
        # Initialize
        print("\nInitializing multi-source pipeline...")
        await app.initialize()
        print("✓ Pipeline initialized successfully")
        
        # Start monitoring task
        monitor_task = asyncio.create_task(monitor_sources(app))
        
        # Start application
        print("\nStarting multi-source analytics...")
        print("Press Ctrl+C to stop\n")
        
        await app.start()
        
        # Run for specified duration or until interrupted
        if duration > 0:
            print(f"Running for {duration} seconds...")
            await asyncio.sleep(duration)
            print(f"\nReached specified duration of {duration} seconds")
        else:
            # Wait for shutdown signal
            await app.wait_for_shutdown()
        
    except KeyboardInterrupt:
        print("\n\nShutdown requested by user...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("\nStopping multi-source pipeline...")
        monitor_task.cancel()
        await app.stop()
        print("✓ Pipeline stopped successfully")
        
        # Summary statistics
        status = app.get_status()
        source_manager = app.source_manager
        aggregated = source_manager.get_aggregated_metrics()
        
        print(f"\n=== Session Summary ===")
        print(f"Runtime: {status.uptime_seconds:.1f} seconds")
        print(f"Total Sources: {aggregated['total_sources']}")
        print(f"Active Sources: {aggregated['active_sources']}")
        print(f"Total Frames: {aggregated['total_frames']}")
        print(f"Average FPS: {aggregated['average_fps']:.1f}")
        print(f"Total Detections: {status.total_detections}")
        print(f"Total Alerts: {status.total_alerts}")


@click.command()
@click.option('--sources', '-s', multiple=True,
              help='RTSP sources in format name=url (can specify multiple)')
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Configuration file with sources')
@click.option('--duration', '-d', type=int, default=0,
              help='Run duration in seconds (0 for infinite)')
@click.option('--save-config', type=click.Path(),
              help='Save generated configuration to file')
def main(sources, config, duration, save_config):
    """
    PyDS Multi-Source RTSP Streaming Example
    
    Process multiple RTSP streams simultaneously with load balancing and monitoring.
    
    Examples:
    
        # Use public test streams
        python multi_source_rtsp.py
        
        # Specify custom RTSP sources
        python multi_source_rtsp.py -s cam1=rtsp://192.168.1.100/stream -s cam2=rtsp://192.168.1.101/stream
        
        # Load from configuration file
        python multi_source_rtsp.py -c configs/cameras.yaml
    """
    # Parse RTSP sources
    rtsp_sources = {}
    if sources:
        for source in sources:
            if '=' in source:
                name, url = source.split('=', 1)
                rtsp_sources[name] = url
            else:
                print(f"Invalid source format: {source}")
                print("Expected format: name=rtsp://url")
                sys.exit(1)
    
    # Save configuration if requested
    if save_config and rtsp_sources:
        config = create_rtsp_config(rtsp_sources)
        config_dict = config.dict()
        
        save_path = Path(save_config)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        print(f"Configuration saved to: {save_path}")
        return
    
    # Run example
    try:
        asyncio.run(multi_rtsp_example(rtsp_sources, config, duration))
    except KeyboardInterrupt:
        print("\nExample terminated by user")
    except Exception as e:
        print(f"\nExample failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()