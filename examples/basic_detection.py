#!/usr/bin/env python3
"""
Basic Detection Example - Single Source with Console Alerts

This example demonstrates:
- Setting up a single video source (test pattern)
- Running basic object detection
- Displaying alerts in the console
- Monitoring performance metrics

Usage:
    python examples/basic_detection.py
    python examples/basic_detection.py --source rtsp://camera.ip/stream
    python examples/basic_detection.py --config configs/custom.yaml
"""

import asyncio
import sys
from pathlib import Path
import click
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.app import PyDSApp
from src.config import AppConfig, SourceConfig, SourceType, load_config
from src.utils.logging import setup_logging, LogLevel


async def basic_detection_example(source_uri: str = None, config_path: str = None):
    """
    Run basic detection example with a single source.
    
    Args:
        source_uri: Optional video source URI (defaults to test pattern)
        config_path: Optional configuration file path
    """
    print("=== PyDS Basic Detection Example ===\n")
    
    # Setup logging
    setup_logging(LogLevel.INFO)
    
    # Load or create configuration
    if config_path:
        print(f"Loading configuration from: {config_path}")
        config = load_config(config_path)
    else:
        print("Using default configuration")
        config = AppConfig(
            name="Basic Detection Example",
            debug=True,
            pipeline={
                'batch_size': 1,
                'width': 1280,
                'height': 720,
                'fps': 30.0
            },
            detection={
                'confidence_threshold': 0.6,
                'max_objects': 50
            },
            alerts={
                'enabled': True,
                'handlers': ['console'],
                'throttle_seconds': 10  # Limit duplicate alerts
            },
            monitoring={
                'enabled': True,
                'metrics_interval': 5  # Report metrics every 5 seconds
            }
        )
    
    # Configure video source
    if source_uri:
        # Parse source type from URI
        if source_uri.startswith('rtsp://'):
            source_type = SourceType.RTSP
        elif source_uri.startswith('file://') or Path(source_uri).exists():
            source_type = SourceType.FILE
        elif source_uri.isdigit() or source_uri.startswith('/dev/video'):
            source_type = SourceType.WEBCAM
        else:
            source_type = SourceType.TEST
        
        source = SourceConfig(
            id="primary_source",
            name="Primary Video Source",
            type=source_type,
            uri=source_uri
        )
    else:
        # Default test source
        print("Using test video pattern (no source specified)")
        source = SourceConfig(
            id="test_source",
            name="Test Pattern",
            type=SourceType.TEST,
            uri="videotestsrc pattern=ball"
        )
    
    # Update configuration with source
    config.sources = [source]
    
    print(f"\nSource Configuration:")
    print(f"  Type: {source.type.value}")
    print(f"  URI: {source.uri}")
    print(f"  ID: {source.id}")
    
    # Create and initialize application
    print("\nInitializing PyDS application...")
    app = PyDSApp(config)
    
    try:
        # Initialize components
        await app.initialize()
        print("✓ Application initialized successfully")
        
        # Set up performance monitoring
        start_time = datetime.now()
        last_stats_time = start_time
        
        async def monitor_performance():
            """Monitor and display performance metrics."""
            nonlocal last_stats_time
            
            while app.is_running():
                await asyncio.sleep(5)  # Check every 5 seconds
                
                # Get current statistics
                stats = app.get_status()
                uptime = datetime.now() - start_time
                
                print(f"\n--- Performance Report ({uptime.seconds}s uptime) ---")
                print(f"State: {stats.state.value}")
                print(f"Active Sources: {stats.active_sources}/{stats.total_sources}")
                print(f"Total Detections: {stats.total_detections}")
                print(f"Total Alerts: {stats.total_alerts}")
                print(f"CPU Usage: {stats.cpu_usage_percent:.1f}%")
                print(f"Memory Usage: {stats.memory_usage_mb:.1f} MB")
                
                # Get pipeline metrics
                pipeline_states = app.pipeline_manager.get_pipeline_states()
                for state in pipeline_states:
                    print(f"\nPipeline '{state['pipeline_id']}':")
                    print(f"  FPS: {state['fps']:.1f}")
                    print(f"  Frames: {state['frame_count']}")
                    print(f"  State: {state['state']}")
                
                last_stats_time = datetime.now()
        
        # Start monitoring task
        monitor_task = asyncio.create_task(monitor_performance())
        
        # Start the application
        print("\nStarting video analytics...")
        print("Press Ctrl+C to stop\n")
        
        await app.start()
        
        # Wait for shutdown signal
        await app.wait_for_shutdown()
        
    except KeyboardInterrupt:
        print("\n\nShutdown requested by user...")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        # Clean shutdown
        print("\nStopping application...")
        monitor_task.cancel()
        await app.stop()
        print("✓ Application stopped successfully")
        
        # Final statistics
        final_stats = app.get_status()
        print(f"\n=== Final Statistics ===")
        print(f"Total Runtime: {final_stats.uptime_seconds:.1f} seconds")
        print(f"Total Detections: {final_stats.total_detections}")
        print(f"Total Alerts: {final_stats.total_alerts}")
        print(f"Failed Components: {final_stats.failed_components}")


@click.command()
@click.option('--source', '-s', help='Video source URI (RTSP, file, webcam, or test)')
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--duration', '-d', type=int, help='Run duration in seconds (0 for infinite)')
def main(source, config, duration):
    """
    PyDS Basic Detection Example
    
    Run a simple video analytics pipeline with object detection and console alerts.
    """
    # Create and run async example
    async def run_with_duration():
        if duration and duration > 0:
            # Run with timeout
            try:
                await asyncio.wait_for(
                    basic_detection_example(source, config),
                    timeout=duration
                )
            except asyncio.TimeoutError:
                print(f"\nReached specified duration of {duration} seconds")
        else:
            # Run indefinitely
            await basic_detection_example(source, config)
    
    # Run the async function
    try:
        asyncio.run(run_with_duration())
    except KeyboardInterrupt:
        print("\nExample terminated by user")
    except Exception as e:
        print(f"\nExample failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()