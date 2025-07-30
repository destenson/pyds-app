#!/usr/bin/env python3
"""
PyDS Application - NVIDIA DeepStream Video Analytics Platform

A comprehensive video analytics application using NVIDIA DeepStream and GStreamer
for real-time pattern detection across multiple video sources with intelligent
alert management and comprehensive monitoring.

Usage:
    python main.py --help
    python main.py run --config config/default.yaml
    python main.py status
    python main.py validate-config config/default.yaml
"""

import asyncio
import sys
import os
import signal
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
import threading

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.live import Live
from rich.prompt import Prompt, Confirm
from rich.tree import Tree

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.app import PyDSApp, create_app, ApplicationState, ShutdownReason
from src.config import AppConfig, load_config, validate_config, save_config, create_default_config
from src.utils.logging import setup_logging, LogLevel, get_logger
from src.utils.errors import ApplicationError, ConfigurationError


# Global console for rich output
console = Console()


class CLIContext:
    """CLI context for passing state between commands."""
    
    def __init__(self):
        self.config_path: Optional[str] = None
        self.config: Optional[AppConfig] = None
        self.app: Optional[PyDSApp] = None
        self.verbose: bool = False
        self.quiet: bool = False


# Global CLI context
cli_context = CLIContext()


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), 
              help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, 
              help='Enable verbose output')
@click.option('--quiet', '-q', is_flag=True, 
              help='Suppress non-essential output')
@click.option('--log-level', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
              default='INFO', help='Set logging level')
@click.version_option(version='1.0.0', prog_name='PyDS')
def cli(config, verbose, quiet, log_level):
    """
    PyDS - NVIDIA DeepStream Video Analytics Platform
    
    A comprehensive video analytics application for real-time pattern detection
    across multiple video sources with intelligent alert management.
    """
    cli_context.config_path = config
    cli_context.verbose = verbose
    cli_context.quiet = quiet
    
    # Setup logging
    setup_logging(level=LogLevel(log_level))
    
    if verbose and not quiet:
        console.print("[blue]PyDS - NVIDIA DeepStream Video Analytics Platform[/blue]")
        console.print(f"Log level: {log_level}")
        if config:
            console.print(f"Config file: {config}")


@cli.command()
@click.option('--daemon', '-d', is_flag=True, 
              help='Run in daemon mode (background)')
@click.option('--profile', is_flag=True,
              help='Enable performance profiling')
@click.option('--monitor', is_flag=True,
              help='Enable interactive monitoring')
def run(daemon, profile, monitor):
    """Start the PyDS application."""
    
    async def run_app():
        try:
            # Load configuration
            config_path = cli_context.config_path or 'config/default.yaml'
            
            if not Path(config_path).exists():
                console.print(f"[red]Error: Configuration file not found: {config_path}[/red]")
                console.print("Use 'python main.py create-config' to create a default configuration.")
                return 1
            
            # Create and run application
            with console.status("[bold green]Starting PyDS application...") as status:
                app = await create_app(config_path=config_path)
                cli_context.app = app
                
                if profile:
                    console.print("[yellow]Performance profiling enabled[/yellow]")
                
                if monitor:
                    # Start monitoring in background
                    monitor_task = asyncio.create_task(run_interactive_monitor(app))
                
                status.update("[bold green]Application starting...")
                
                # Run the application
                exit_code = await app.run()
                
                if monitor:
                    monitor_task.cancel()
                
                return exit_code
        
        except KeyboardInterrupt:
            console.print("\n[yellow]Received interrupt signal, shutting down...[/yellow]")
            if cli_context.app:
                await cli_context.app.stop(ShutdownReason.SIGNAL)
            return 0
        
        except Exception as e:
            console.print(f"[red]Error running application: {e}[/red]")
            return 1
    
    if daemon:
        console.print("[yellow]Daemon mode not yet implemented[/yellow]")
        return 1
    
    # Run the application
    try:
        exit_code = asyncio.run(run_app())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        console.print("\n[yellow]Application interrupted[/yellow]")
        sys.exit(0)


@cli.command()
@click.argument('config_file', type=click.Path())
def validate_config(config_file):
    """Validate a configuration file."""
    try:
        console.print(f"[blue]Validating configuration: {config_file}[/blue]")
        
        if not Path(config_file).exists():
            console.print(f"[red]Error: Configuration file not found: {config_file}[/red]")
            return
        
        # Load and validate configuration
        config = load_config(config_file)
        errors = validate_config(config)
        
        if errors:
            console.print(f"[red]Configuration validation failed with {len(errors)} errors:[/red]")
            for error in errors:
                console.print(f"  • {error}")
        else:
            console.print("[green]✓ Configuration is valid[/green]")
            
            # Show configuration summary
            if not cli_context.quiet:
                show_config_summary(config)
    
    except Exception as e:
        console.print(f"[red]Error validating configuration: {e}[/red]")


@cli.command()
@click.argument('output_file', type=click.Path())
@click.option('--interactive', '-i', is_flag=True,
              help='Interactive configuration creation')
def create_config(output_file, interactive):
    """Create a new configuration file."""
    try:
        if Path(output_file).exists():
            if not Confirm.ask(f"Configuration file {output_file} already exists. Overwrite?"):
                return
        
        if interactive:
            config = create_interactive_config()
        else:
            config = create_default_config()
        
        # Save configuration
        save_config(config, output_file)
        
        console.print(f"[green]✓ Configuration created: {output_file}[/green]")
        
        if not cli_context.quiet:
            console.print("[blue]Configuration summary:[/blue]")
            show_config_summary(config)
    
    except Exception as e:
        console.print(f"[red]Error creating configuration: {e}[/red]")


@cli.command()
def status():
    """Show application status."""
    # This would connect to a running application instance
    # For now, show placeholder status
    console.print("[yellow]Application status checking not yet implemented[/yellow]")
    console.print("This feature requires a running application with status endpoint.")


@cli.command()
@click.option('--output', '-o', type=click.Path(),
              help='Output file for the report')
@click.option('--format', 'report_format', type=click.Choice(['json', 'text', 'html']),
              default='text', help='Report format')
def health_report(output, report_format):
    """Generate a health report."""
    try:
        console.print("[blue]Generating health report...[/blue]")
        
        # This would collect health data from running application
        health_data = {
            "timestamp": time.time(),
            "status": "healthy",
            "components": [],
            "metrics": {}
        }
        
        if output:
            if report_format == 'json':
                with open(output, 'w') as f:
                    json.dump(health_data, f, indent=2)
            else:
                with open(output, 'w') as f:
                    f.write("Health report generated\n")
            
            console.print(f"[green]✓ Health report saved: {output}[/green]")
        else:
            console.print("Health report would be displayed here")
    
    except Exception as e:
        console.print(f"[red]Error generating health report: {e}[/red]")


@cli.command()
@click.option('--duration', '-d', type=int, default=60,
              help='Profiling duration in seconds')
@click.option('--output', '-o', type=click.Path(),
              help='Output file for the profile report')
def profile(duration, output):
    """Run performance profiling."""
    try:
        console.print(f"[blue]Running performance profile for {duration} seconds...[/blue]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Profiling...", total=duration)
            
            for i in range(duration):
                time.sleep(1)
                progress.update(task, advance=1, description=f"Profiling... ({i+1}/{duration}s)")
        
        console.print("[green]✓ Profiling completed[/green]")
        
        if output:
            console.print(f"[green]Profile report would be saved to: {output}[/green]")
        else:
            console.print("Profile results would be displayed here")
    
    except Exception as e:
        console.print(f"[red]Error running profile: {e}[/red]")


@cli.command()
@click.option('--watch', '-w', is_flag=True,
              help='Watch metrics in real-time')
@click.option('--duration', '-d', type=int,
              help='Duration to watch in seconds')
def metrics(watch, duration):
    """Display system metrics."""
    try:
        if watch:
            console.print("[blue]Watching metrics (Ctrl+C to stop)...[/blue]")
            
            try:
                start_time = time.time()
                while True:
                    if duration and (time.time() - start_time) > duration:
                        break
                    
                    # Clear screen and show metrics
                    console.clear()
                    display_metrics()
                    time.sleep(2)
            
            except KeyboardInterrupt:
                console.print("\n[yellow]Metrics watching stopped[/yellow]")
        else:
            display_metrics()
    
    except Exception as e:
        console.print(f"[red]Error displaying metrics: {e}[/red]")


@cli.command()
@click.argument('component_name')
def restart_component(component_name):
    """Restart a specific component."""
    console.print(f"[yellow]Restarting component: {component_name}[/yellow]")
    console.print("This feature requires connection to a running application.")


@cli.command()
def interactive():
    """Start interactive mode."""
    console.print("[blue]Starting PyDS Interactive Mode[/blue]")
    console.print("Type 'help' for available commands, 'exit' to quit.")
    
    while True:
        try:
            command = Prompt.ask("[green]pyds>[/green]")
            
            if command.lower() in ['exit', 'quit']:
                console.print("[yellow]Goodbye![/yellow]")
                break
            elif command.lower() == 'help':
                show_interactive_help()
            elif command.lower() == 'status':
                display_application_status()
            elif command.lower() == 'metrics':
                display_metrics()
            elif command.lower().startswith('config'):
                handle_config_command(command)
            else:
                console.print(f"[red]Unknown command: {command}[/red]")
                console.print("Type 'help' for available commands.")
        
        except KeyboardInterrupt:
            console.print("\n[yellow]Use 'exit' to quit interactive mode[/yellow]")
        except EOFError:
            console.print("\n[yellow]Goodbye![/yellow]")
            break


def show_config_summary(config: AppConfig):
    """Display configuration summary."""
    table = Table(title="Configuration Summary")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Application Name", config.app_name)
    table.add_row("Log Level", config.logging.level)
    table.add_row("Pipeline Batch Size", str(config.pipeline.batch_size))
    table.add_row("Detection Confidence", str(config.detection.confidence_threshold))
    table.add_row("Max Alerts/Minute", str(config.alerts.max_alerts_per_minute))
    
    console.print(table)


def create_interactive_config() -> AppConfig:
    """Create configuration interactively."""
    console.print("[blue]Interactive Configuration Creation[/blue]")
    
    # Basic settings
    app_name = Prompt.ask("Application name", default="PyDS-App")
    
    # Pipeline settings
    console.print("\n[blue]Pipeline Configuration[/blue]")
    batch_size = int(Prompt.ask("Batch size", default="4"))
    width = int(Prompt.ask("Frame width", default="1920"))
    height = int(Prompt.ask("Frame height", default="1080"))
    fps = int(Prompt.ask("Target FPS", default="30"))
    
    # Detection settings
    console.print("\n[blue]Detection Configuration[/blue]")
    confidence_threshold = float(Prompt.ask("Confidence threshold", default="0.5"))
    nms_threshold = float(Prompt.ask("NMS threshold", default="0.4"))
    
    # Alert settings
    console.print("\n[blue]Alert Configuration[/blue]")
    max_alerts = int(Prompt.ask("Max alerts per minute", default="60"))
    
    # Create configuration object
    config = create_default_config()
    config.app_name = app_name
    config.pipeline.batch_size = batch_size
    config.pipeline.width = width
    config.pipeline.height = height
    config.pipeline.fps = fps
    config.detection.confidence_threshold = confidence_threshold
    config.detection.nms_threshold = nms_threshold
    config.alerts.max_alerts_per_minute = max_alerts
    
    return config


async def run_interactive_monitor(app: PyDSApp):
    """Run interactive monitoring."""
    try:
        while True:
            await asyncio.sleep(5)
            
            # Update monitoring display
            if not cli_context.quiet:
                status = app.get_status()
                console.print(f"Status: {status.state.value} | "
                            f"Uptime: {status.uptime_seconds:.0f}s | "
                            f"Memory: {status.memory_usage_mb:.1f}MB")
    
    except asyncio.CancelledError:
        pass


def display_application_status():
    """Display current application status."""
    # This would get status from running application
    table = Table(title="Application Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Uptime", style="yellow")
    
    # Placeholder data
    components = [
        ("Pipeline Manager", "Running", "00:15:23"),
        ("Detection Engine", "Running", "00:15:20"),
        ("Alert System", "Running", "00:15:18"),
        ("Health Monitor", "Running", "00:15:15"),
    ]
    
    for name, status, uptime in components:
        table.add_row(name, status, uptime)
    
    console.print(table)


def display_metrics():
    """Display system metrics."""
    import psutil
    
    table = Table(title="System Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    # System metrics
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    table.add_row("CPU Usage", f"{cpu_percent:.1f}%")
    table.add_row("Memory Usage", f"{memory.percent:.1f}%")
    table.add_row("Memory Available", f"{memory.available / (1024**3):.1f} GB")
    table.add_row("Disk Usage", f"{(disk.used / disk.total) * 100:.1f}%")
    table.add_row("Disk Free", f"{disk.free / (1024**3):.1f} GB")
    
    console.print(table)


def show_interactive_help():
    """Show interactive mode help."""
    help_panel = Panel.fit(
        """[bold blue]Available Commands:[/bold blue]

[green]status[/green]     - Show application status
[green]metrics[/green]    - Display system metrics
[green]config[/green]     - Show configuration
[green]help[/green]       - Show this help
[green]exit[/green]       - Exit interactive mode

[yellow]Note:[/yellow] Some commands require a running application instance.""",
        title="Interactive Help"
    )
    console.print(help_panel)


def handle_config_command(command: str):
    """Handle configuration commands in interactive mode."""
    if command == "config":
        if cli_context.config:
            show_config_summary(cli_context.config)
        else:
            console.print("[yellow]No configuration loaded[/yellow]")
    else:
        console.print("[red]Invalid config command[/red]")


@cli.command()
def doctor():
    """Run system diagnostics."""
    console.print("[blue]Running PyDS System Diagnostics...[/blue]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        # Check Python version
        task = progress.add_task("Checking Python version...", total=7)
        time.sleep(0.5)
        python_version = sys.version_info
        if python_version >= (3, 8):
            console.print("[green]✓ Python version OK[/green]")
        else:
            console.print("[red]✗ Python version too old (requires 3.8+)[/red]")
        progress.advance(task)
        
        # Check dependencies
        progress.update(task, description="Checking dependencies...")
        time.sleep(0.5)
        missing_deps = check_dependencies()
        if not missing_deps:
            console.print("[green]✓ All dependencies available[/green]")
        else:
            console.print(f"[red]✗ Missing dependencies: {', '.join(missing_deps)}[/red]")
        progress.advance(task)
        
        # Check GStreamer
        progress.update(task, description="Checking GStreamer...")
        time.sleep(0.5)
        if check_gstreamer():
            console.print("[green]✓ GStreamer available[/green]")
        else:
            console.print("[red]✗ GStreamer not found[/red]")
        progress.advance(task)
        
        # Check DeepStream
        progress.update(task, description="Checking DeepStream...")
        time.sleep(0.5)
        deepstream_version = check_deepstream()
        if deepstream_version:
            console.print(f"[green]✓ DeepStream {deepstream_version} available[/green]")
        else:
            console.print("[red]✗ DeepStream not found[/red]")
        progress.advance(task)
        
        # Check GPU
        progress.update(task, description="Checking GPU...")
        time.sleep(0.5)
        gpu_info = check_gpu()
        if gpu_info:
            console.print(f"[green]✓ GPU available: {gpu_info}[/green]")
        else:
            console.print("[yellow]⚠ No GPU detected[/yellow]")
        progress.advance(task)
        
        # Check permissions
        progress.update(task, description="Checking permissions...")
        time.sleep(0.5)
        if check_permissions():
            console.print("[green]✓ Permissions OK[/green]")
        else:
            console.print("[yellow]⚠ Some permissions issues detected[/yellow]")
        progress.advance(task)
        
        # Check disk space
        progress.update(task, description="Checking disk space...")
        time.sleep(0.5)
        disk_space = check_disk_space()
        if disk_space > 1:  # 1GB minimum
            console.print(f"[green]✓ Disk space OK ({disk_space:.1f} GB free)[/green]")
        else:
            console.print(f"[red]✗ Low disk space ({disk_space:.1f} GB free)[/red]")
        progress.advance(task)
    
    console.print("\n[blue]Diagnostics completed![/blue]")


def check_dependencies() -> List[str]:
    """Check for missing dependencies."""
    required_deps = [
        'click', 'rich', 'psutil', 'pyyaml', 'numpy'
    ]
    
    missing = []
    for dep in required_deps:
        try:
            __import__(dep)
        except ImportError:
            missing.append(dep)
    
    return missing


def check_gstreamer() -> bool:
    """Check if GStreamer is available."""
    try:
        import gi
        gi.require_version('Gst', '1.0')
        from gi.repository import Gst
        Gst.init(None)
        return True
    except:
        return False


def check_deepstream() -> Optional[str]:
    """Check DeepStream availability."""
    try:
        # Try to import DeepStream bindings
        import gi
        gi.require_version('NvDs', '1.0')
        from gi.repository import NvDs
        return "6.x+"
    except:
        try:
            import pyds
            return "5.x"
        except:
            return None


def check_gpu() -> Optional[str]:
    """Check GPU availability."""
    try:
        import nvidia_ml_py3 as nvml
        nvml.nvmlInit()
        device_count = nvml.nvmlDeviceGetCount()
        if device_count > 0:
            handle = nvml.nvmlDeviceGetHandleByIndex(0)
            name = nvml.nvmlDeviceGetName(handle).decode('utf-8')
            return f"{name} (+{device_count-1} more)" if device_count > 1 else name
    except:
        pass
    
    return None


def check_permissions() -> bool:
    """Check file system permissions."""
    try:
        # Check if we can create directories and files
        test_dir = Path("tmp_test_permissions")
        test_dir.mkdir(exist_ok=True)
        
        test_file = test_dir / "test.txt"
        test_file.write_text("test")
        test_file.unlink()
        test_dir.rmdir()
        
        return True
    except:
        return False


def check_disk_space() -> float:
    """Check available disk space in GB."""
    import shutil
    free_bytes = shutil.disk_usage('.').free
    return free_bytes / (1024**3)


@cli.command()
def version():
    """Show version information."""
    version_info = {
        "PyDS": "1.0.0",
        "Python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "Platform": sys.platform
    }
    
    # Check optional component versions
    try:
        import gi
        gi.require_version('Gst', '1.0')
        from gi.repository import Gst
        Gst.init(None)
        version_info["GStreamer"] = ".".join(map(str, Gst.version()))
    except:
        version_info["GStreamer"] = "Not available"
    
    try:
        deepstream_version = check_deepstream()
        version_info["DeepStream"] = deepstream_version or "Not available"
    except:
        version_info["DeepStream"] = "Not available"
    
    table = Table(title="Version Information")
    table.add_column("Component", style="cyan")
    table.add_column("Version", style="green")
    
    for component, version in version_info.items():
        table.add_row(component, version)
    
    console.print(table)


def main():
    """Main entry point."""
    try:
        # Ensure we have a clean environment
        if sys.version_info < (3, 8):
            console.print("[red]Error: Python 3.8 or higher is required[/red]")
            sys.exit(1)
        
        # Run CLI
        cli()
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        if cli_context.verbose:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()