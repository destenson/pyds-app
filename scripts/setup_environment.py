#!/usr/bin/env python3
"""
Environment Setup and Validation Script for PyDS

This script validates and sets up the PyDS environment including:
- DeepStream SDK installation verification
- GStreamer plugins availability
- GPU/CUDA setup validation
- Python dependencies checking
- System requirements verification
- Environment configuration

Usage:
    python scripts/setup_environment.py --check
    python scripts/setup_environment.py --install-deps
    python scripts/setup_environment.py --validate-gpu
    python scripts/setup_environment.py --full-setup
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.deepstream import get_deepstream_info, DeepStreamDetector
from src.utils.logging import setup_logging, LogLevel

console = Console()


class EnvironmentValidator:
    """Validates and sets up the PyDS environment."""
    
    def __init__(self):
        self.console = Console()
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = []
        self.errors = []
        
        # System information
        self.system_info = {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'architecture': platform.machine(),
            'python_version': platform.python_version(),
            'python_path': sys.executable
        }
    
    def run_full_validation(self) -> bool:
        """Run complete environment validation."""
        self.console.print("[bold blue]PyDS Environment Validation[/bold blue]\n")
        
        # System requirements
        self._check_system_requirements()
        
        # Python environment
        self._check_python_environment()
        
        # CUDA and GPU
        self._check_cuda_installation()
        self._check_gpu_availability()
        
        # GStreamer
        self._check_gstreamer_installation()
        
        # DeepStream
        self._check_deepstream_installation()
        
        # Python dependencies
        self._check_python_dependencies()
        
        # Permissions and paths
        self._check_permissions()
        
        # Generate summary
        return self._generate_summary()
    
    def _check_system_requirements(self):
        """Check basic system requirements."""
        self.console.print("[yellow]Checking System Requirements...[/yellow]")
        
        # Operating system
        if self.system_info['platform'] not in ['Linux', 'Windows']:
            self._add_error(f"Unsupported platform: {self.system_info['platform']}")
        else:
            self._add_success(f"Platform: {self.system_info['platform']}")
        
        # Architecture
        if self.system_info['architecture'] not in ['x86_64', 'AMD64']:
            self._add_error(f"Unsupported architecture: {self.system_info['architecture']}")
        else:
            self._add_success(f"Architecture: {self.system_info['architecture']}")
        
        # Memory
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
            if memory_gb < 8:
                self._add_warning(f"Low system memory: {memory_gb:.1f}GB (8GB+ recommended)")
            else:
                self._add_success(f"System memory: {memory_gb:.1f}GB")
        except ImportError:
            self._add_warning("Could not check system memory (psutil not available)")
        
        # Disk space
        try:
            disk_usage = shutil.disk_usage(Path.cwd())
            free_gb = disk_usage.free / (1024**3)
            if free_gb < 10:
                self._add_warning(f"Low disk space: {free_gb:.1f}GB free (10GB+ recommended)")
            else:
                self._add_success(f"Disk space: {free_gb:.1f}GB free")
        except Exception:
            self._add_warning("Could not check disk space")
    
    def _check_python_environment(self):
        """Check Python version and virtual environment."""
        self.console.print("[yellow]Checking Python Environment...[/yellow]")
        
        # Python version
        python_version = tuple(map(int, platform.python_version().split('.')))
        if python_version < (3, 8):
            self._add_error(f"Python {platform.python_version()} is too old (3.8+ required)")
        else:
            self._add_success(f"Python version: {platform.python_version()}")
        
        # Virtual environment
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            self._add_success("Virtual environment detected")
        else:
            self._add_warning("No virtual environment detected (recommended for isolation)")
        
        # Package manager
        if shutil.which('uv'):
            self._add_success("UV package manager available")
        elif shutil.which('pip'):
            self._add_success("pip package manager available")
        else:
            self._add_error("No package manager found")
    
    def _check_cuda_installation(self):
        """Check CUDA installation and version."""
        self.console.print("[yellow]Checking CUDA Installation...[/yellow]")
        
        # Check nvidia-smi
        if shutil.which('nvidia-smi'):
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version,cuda_version', '--format=csv,noheader'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if lines and lines[0]:
                        driver_version, cuda_version = lines[0].split(', ')
                        self._add_success(f"NVIDIA Driver: {driver_version}")
                        self._add_success(f"CUDA Version: {cuda_version}")
                    else:
                        self._add_error("Could not parse nvidia-smi output")
                else:
                    self._add_error("nvidia-smi failed to run")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                self._add_error("nvidia-smi not working properly")
        else:
            self._add_error("nvidia-smi not found - NVIDIA drivers not installed")
        
        # Check CUDA toolkit
        cuda_paths = [
            '/usr/local/cuda',
            '/opt/cuda',
            'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\*'
        ]
        
        cuda_found = False
        for cuda_path in cuda_paths:
            if Path(cuda_path).exists() or any(Path(p).exists() for p in Path(cuda_path).parent.glob(Path(cuda_path).name) if '*' in cuda_path):
                self._add_success(f"CUDA toolkit found at: {cuda_path}")
                cuda_found = True
                break
        
        if not cuda_found:
            self._add_warning("CUDA toolkit not found in standard locations")
    
    def _check_gpu_availability(self):
        """Check GPU availability and compute capability."""
        self.console.print("[yellow]Checking GPU Availability...[/yellow]")
        
        try:
            import pynvml
            pynvml.nvmlInit()
            
            device_count = pynvml.nvmlDeviceGetCount()
            self._add_success(f"Found {device_count} NVIDIA GPU(s)")
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle).decode()
                
                # Memory
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                mem_total_gb = mem_info.total / (1024**3)
                
                # Compute capability (if available)
                try:
                    major = pynvml.nvmlDeviceGetCudaComputeCapability(handle)[0]
                    minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)[1]
                    compute_cap = f"{major}.{minor}"
                    
                    if major < 6:  # Compute capability 6.0+ required for DeepStream
                        self._add_warning(f"GPU {i} ({name}): Compute capability {compute_cap} may be too old")
                    else:
                        self._add_success(f"GPU {i} ({name}): {mem_total_gb:.1f}GB, Compute {compute_cap}")
                
                except:
                    self._add_success(f"GPU {i} ({name}): {mem_total_gb:.1f}GB memory")
        
        except ImportError:
            self._add_warning("pynvml not available - cannot check detailed GPU info")
        except Exception as e:
            self._add_error(f"Error checking GPU: {e}")
    
    def _check_gstreamer_installation(self):
        """Check GStreamer installation and plugins."""
        self.console.print("[yellow]Checking GStreamer Installation...[/yellow]")
        
        # Check gst-launch command
        gst_commands = ['gst-launch-1.0', 'gst-launch']
        gst_found = False
        
        for cmd in gst_commands:
            if shutil.which(cmd):
                try:
                    result = subprocess.run([cmd, '--version'], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        version_line = result.stdout.split('\n')[0]
                        self._add_success(f"GStreamer: {version_line}")
                        gst_found = True
                        break
                except:
                    pass
        
        if not gst_found:
            self._add_error("GStreamer not found")
            return
        
        # Check essential plugins
        essential_plugins = [
            'coreelements', 'videotestsrc', 'videoconvert', 'videoscale',
            'autodetect', 'playback'
        ]
        
        for plugin in essential_plugins:
            if self._check_gst_plugin(plugin):
                self._add_success(f"GStreamer plugin: {plugin}")
            else:
                self._add_error(f"Missing GStreamer plugin: {plugin}")
    
    def _check_deepstream_installation(self):
        """Check DeepStream SDK installation."""
        self.console.print("[yellow]Checking DeepStream Installation...[/yellow]")
        
        try:
            detector = DeepStreamDetector()
            info = detector.detect_deepstream()
            
            self._add_success(f"DeepStream {info.version_string} found")
            self._add_success(f"API Type: {info.api_type.value}")
            
            if info.install_path:
                self._add_success(f"Install path: {info.install_path}")
            
            if info.python_bindings_path:
                self._add_success(f"Python bindings: {info.python_bindings_path}")
            
            # Check capabilities
            if info.capabilities:
                working_caps = [cap for cap, working in info.capabilities.items() if working]
                self._add_success(f"Working capabilities: {', '.join(working_caps)}")
                
                broken_caps = [cap for cap, working in info.capabilities.items() if not working]
                if broken_caps:
                    self._add_warning(f"Non-working capabilities: {', '.join(broken_caps)}")
        
        except Exception as e:
            self._add_error(f"DeepStream not found or not working: {e}")
    
    def _check_python_dependencies(self):
        """Check Python package dependencies."""
        self.console.print("[yellow]Checking Python Dependencies...[/yellow]")
        
        # Core packages
        core_packages = [
            ('numpy', 'Core numerical computing'),
            ('opencv-python', 'Computer vision'),
            ('pyyaml', 'Configuration files'),
            ('click', 'Command line interface'),
            ('rich', 'Rich console output'),
            ('psutil', 'System monitoring'),
            ('aiofiles', 'Async file operations'),
            ('aiohttp', 'Async HTTP client')
        ]
        
        for package, description in core_packages:
            if self._check_python_package(package):
                self._add_success(f"Python package: {package} ({description})")
            else:
                self._add_error(f"Missing Python package: {package} ({description})")
        
        # Optional packages
        optional_packages = [
            ('pynvml', 'NVIDIA GPU monitoring'),
            ('prometheus-client', 'Metrics export'),
            ('memory-profiler', 'Memory profiling')
        ]
        
        for package, description in optional_packages:
            if self._check_python_package(package):
                self._add_success(f"Optional package: {package} ({description})")
            else:
                self._add_warning(f"Optional package missing: {package} ({description})")
    
    def _check_permissions(self):
        """Check file permissions and paths."""
        self.console.print("[yellow]Checking Permissions and Paths...[/yellow]")
        
        # Write permissions in current directory
        try:
            test_file = Path('test_write_permission.tmp')
            test_file.write_text('test')
            test_file.unlink()
            self._add_success("Write permissions in current directory")
        except:
            self._add_error("No write permissions in current directory")
        
        # Check if running as root (not recommended)
        if os.geteuid() == 0 if hasattr(os, 'geteuid') else False:
            self._add_warning("Running as root (not recommended for development)")
        
        # Check PATH
        path_dirs = os.environ.get('PATH', '').split(os.pathsep)
        if len(path_dirs) > 0:
            self._add_success(f"PATH contains {len(path_dirs)} directories")
        else:
            self._add_error("PATH environment variable is empty")
    
    def _check_gst_plugin(self, plugin_name: str) -> bool:
        """Check if a GStreamer plugin is available."""
        try:
            result = subprocess.run(['gst-inspect-1.0', plugin_name],
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except:
            return False
    
    def _check_python_package(self, package_name: str) -> bool:
        """Check if a Python package is installed."""
        try:
            __import__(package_name.replace('-', '_'))
            return True
        except ImportError:
            return False
    
    def _add_success(self, message: str):
        """Add a success message."""
        self.console.print(f"[green]✓[/green] {message}")
        self.checks_passed += 1
    
    def _add_warning(self, message: str):
        """Add a warning message."""
        self.console.print(f"[yellow]⚠[/yellow] {message}")
        self.warnings.append(message)
    
    def _add_error(self, message: str):
        """Add an error message."""
        self.console.print(f"[red]✗[/red] {message}")
        self.errors.append(message)
        self.checks_failed += 1
    
    def _generate_summary(self) -> bool:
        """Generate validation summary."""
        self.console.print("\n" + "="*60)
        
        # Summary statistics
        total_checks = self.checks_passed + self.checks_failed
        success_rate = (self.checks_passed / total_checks * 100) if total_checks > 0 else 0
        
        summary_table = Table(title="Environment Validation Summary")
        summary_table.add_column("Category", style="cyan")
        summary_table.add_column("Status", justify="center")
        summary_table.add_column("Count", justify="right")
        
        summary_table.add_row("Passed", "[green]✓[/green]", str(self.checks_passed))
        summary_table.add_row("Warnings", "[yellow]⚠[/yellow]", str(len(self.warnings)))
        summary_table.add_row("Failed", "[red]✗[/red]", str(self.checks_failed))
        summary_table.add_row("Success Rate", "", f"{success_rate:.1f}%")
        
        self.console.print(summary_table)
        
        # Overall status
        if self.checks_failed == 0:
            if len(self.warnings) == 0:
                status = Panel("[bold green]Environment is fully ready for PyDS![/bold green]", 
                             style="green")
            else:
                status = Panel("[bold yellow]Environment is mostly ready. Check warnings above.[/bold yellow]", 
                             style="yellow")
        else:
            status = Panel("[bold red]Environment has critical issues that must be resolved.[/bold red]", 
                         style="red")
        
        self.console.print(status)
        
        # Recommendations
        if self.errors or self.warnings:
            self.console.print("\n[bold]Recommendations:[/bold]")
            
            if "DeepStream not found" in str(self.errors):
                self.console.print("• Install NVIDIA DeepStream SDK from https://developer.nvidia.com/deepstream-sdk")
            
            if "nvidia-smi not found" in str(self.errors):
                self.console.print("• Install NVIDIA GPU drivers")
            
            if "GStreamer not found" in str(self.errors):
                self.console.print("• Install GStreamer development packages")
            
            if any("Python package" in error for error in self.errors):
                self.console.print("• Install missing Python packages: pip install -e .")
        
        return self.checks_failed == 0
    
    def install_python_dependencies(self):
        """Install Python dependencies."""
        self.console.print("[bold blue]Installing Python Dependencies[/bold blue]\n")
        
        # Check for package managers
        if shutil.which('uv'):
            cmd = ['uv', 'sync']
        elif shutil.which('pip'):
            cmd = ['pip', 'install', '-e', '.']
        else:
            self._add_error("No package manager found")
            return False
        
        try:
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
                task = progress.add_task("Installing dependencies...", total=None)
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    self._add_success("Dependencies installed successfully")
                    return True
                else:
                    self._add_error(f"Dependency installation failed: {result.stderr}")
                    return False
        
        except subprocess.TimeoutExpired:
            self._add_error("Dependency installation timed out")
            return False
        except Exception as e:
            self._add_error(f"Error installing dependencies: {e}")
            return False
    
    def export_system_info(self, output_file: str):
        """Export system information to file."""
        # Collect all system information
        system_report = {
            'timestamp': str(datetime.now()),
            'system_info': self.system_info,
            'validation_results': {
                'checks_passed': self.checks_passed,
                'checks_failed': self.checks_failed,
                'warnings': self.warnings,
                'errors': self.errors
            }
        }
        
        # Try to get DeepStream info
        try:
            info = get_deepstream_info()
            system_report['deepstream_info'] = {
                'version': info.version_string,
                'api_type': info.api_type.value,
                'install_path': info.install_path,
                'capabilities': info.capabilities
            }
        except:
            system_report['deepstream_info'] = None
        
        # Export to file
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump(system_report, f, indent=2)
        
        self.console.print(f"System information exported to: {output_path}")


@click.command()
@click.option('--check', is_flag=True, help='Run environment validation')
@click.option('--install-deps', is_flag=True, help='Install Python dependencies')
@click.option('--validate-gpu', is_flag=True, help='Validate GPU and CUDA setup only')
@click.option('--export-info', type=click.Path(), help='Export system info to file')
@click.option('--full-setup', is_flag=True, help='Run full setup and validation')
@click.option('--quiet', '-q', is_flag=True, help='Suppress verbose output')
def main(check, install_deps, validate_gpu, export_info, full_setup, quiet):
    """
    PyDS Environment Setup and Validation
    
    Validate system requirements and set up the PyDS development environment.
    """
    # Setup logging
    log_level = LogLevel.WARNING if quiet else LogLevel.INFO
    setup_logging(log_level)
    
    validator = EnvironmentValidator()
    
    try:
        if full_setup:
            # Full setup: install dependencies then validate
            if not quiet:
                console.print("[bold blue]Running Full PyDS Environment Setup[/bold blue]\n")
            
            success = validator.install_python_dependencies()
            if success:
                validator.run_full_validation()
        
        elif check:
            # Just run validation
            validator.run_full_validation()
        
        elif install_deps:
            # Just install dependencies
            validator.install_python_dependencies()
        
        elif validate_gpu:
            # Just validate GPU setup
            console.print("[bold blue]GPU/CUDA Validation[/bold blue]\n")
            validator._check_cuda_installation()
            validator._check_gpu_availability()
        
        else:
            # Default: run basic validation
            validator.run_full_validation()
        
        # Export system info if requested
        if export_info:
            validator.export_system_info(export_info)
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Setup interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Setup failed: {e}[/red]")
        if not quiet:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()