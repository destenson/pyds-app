"""
Vulture whitelist for legitimate "unused" code patterns in PyDS application.

This file contains patterns that Vulture should not flag as dead code,
including dynamic imports, GStreamer callbacks, plugin discovery, etc.

See: https://vulture.readthedocs.io/en/latest/
"""

# Dynamic imports through entry points (legitimate plugin discovery)
entry_points = None  # pkg_resources.iter_entry_points()
load_entry_point = None  # pkg_resources.load_entry_point()

# Unused imports that are actually used in type hints or dynamic contexts
get_type_hints = None  # Used in type validation
NamedTuple = None  # Used in data structures
root_validator = None  # Pydantic v1 compatibility
RecoveryAction = None  # Used in error handling strategies
graceful_shutdown = None  # Used in shutdown handlers
MonitoringError = None  # Custom exception types
ProfilingError = None  # Custom exception types
Counter = None  # Used in metrics collection
linecache = None  # Used in debugging/profiling
py_spy = None  # External profiling tool
memory_profiler = None  # Memory profiling tool
as_completed = None  # AsyncIO utility
create_test_detection = None  # Used in test utilities

# GStreamer callback function signatures (called by GStreamer C library)
def gst_pad_probe_callback(pad, probe_info):
    """GStreamer probe callback - called from C code."""
    pass

def gst_bus_callback(bus, message):
    """GStreamer bus message callback - called from C code."""
    pass

def gst_element_callback(element, *args):
    """Generic GStreamer element callback - called from C code.""" 
    pass

# DeepStream specific callbacks
def nvinfer_callback(l_frame, obj_meta_list):
    """DeepStream inference callback - called from NVIDIA libraries."""
    pass

def nvds_meta_callback(batch_meta, frame_meta):
    """DeepStream metadata callback - called from NVIDIA libraries."""
    pass

# Mock classes and compatibility shims (when GSTREAMER_AVAILABLE=False)
class MockGstElement:
    """Mock GStreamer element for testing without GStreamer."""
    def set_property(self, name, value): pass
    def get_property(self, name): return None
    def link(self, other): return True
    def set_state(self, state): return True

class MockGstPipeline:
    """Mock GStreamer pipeline for testing without GStreamer.""" 
    def add(self, *elements): pass
    def remove(self, *elements): pass
    def get_bus(self): return MockGstBus()
    def set_state(self, state): return True

class MockGstBus:
    """Mock GStreamer bus for testing without GStreamer."""
    def add_signal_watch(self): pass
    def connect(self, signal, callback): pass

# Test fixtures and pytest markers (used by pytest framework)
def pytest_configure(config):
    """Pytest configuration hook."""
    pass

def pytest_collection_modifyitems(config, items):
    """Pytest collection hook."""
    pass

# CLI command functions (entry points from pyproject.toml)
def main():
    """Main entry point - called from console scripts."""
    pass

def benchmark_main():
    """Benchmark entry point - called from console scripts."""
    pass

def setup_main():
    """Setup entry point - called from console scripts."""
    pass

def validate_main():
    """Validation entry point - called from console scripts."""
    pass

# Detection strategy entry points (dynamic loading via entry_points)
class YOLOStrategy:
    """YOLO detection strategy - loaded dynamically."""
    pass

class TemplateMatchingStrategy:
    """Template matching strategy - loaded dynamically."""
    pass

class FeatureBasedStrategy:
    """Feature-based detection strategy - loaded dynamically."""
    pass

# Alert handler entry points (dynamic loading via entry_points) 
class ConsoleHandler:
    """Console alert handler - loaded dynamically."""
    pass

class FileHandler:
    """File alert handler - loaded dynamically."""
    pass

class WebhookHandler:
    """Webhook alert handler - loaded dynamically."""
    pass

class EmailHandler:
    """Email alert handler - loaded dynamically."""
    pass

# Configuration and environment variables (accessed via getattr/os.environ)
DEBUG = None
TESTING = None
GSTREAMER_AVAILABLE = None
DEEPSTREAM_AVAILABLE = None
CUDA_AVAILABLE = None

# Version compatibility handling
DEEPSTREAM_VERSION = None
GSTREAMER_VERSION = None

# Error handling and logging patterns
def log_and_ignore_error(func):
    """Decorator for logging errors without raising."""
    pass

# Async context patterns
async def async_context_manager_enter(self):
    """Async context manager entry."""
    return self

async def async_context_manager_exit(self, exc_type, exc_val, exc_tb):
    """Async context manager exit.""" 
    pass

# Configuration file handling (dynamic attribute access)
config_attr = None  # getattr(config, 'attr', default)

# Signal handlers (registered but may not be directly called)
def signal_handler(signum, frame):
    """Signal handler for graceful shutdown."""
    pass

# Thread-local storage patterns
thread_local_storage = None

# Rich console formatting (accessed via getattr patterns)
console_style = None
console_theme = None

# Pydantic model fields (dynamic validation)
model_fields = None
model_config = None

# Profiling and debugging hooks
def profile_function(func):
    """Profiling decorator."""
    return func

def debug_hook(*args, **kwargs):
    """Debug hook function."""
    pass

# Platform-specific code (conditional imports)
if hasattr(object, 'windows_specific'):
    windows_specific = None

if hasattr(object, 'linux_specific'):
    linux_specific = None

# Type checking variables (only used by mypy)
TYPE_CHECKING = False
if TYPE_CHECKING:
    type_checking_only_import = None

# Backward compatibility aliases
LegacyClassName = None
deprecated_function = None

# Abstract base class methods (implemented by subclasses)
def abstract_method(self):
    """Abstract method implemented by subclasses."""
    raise NotImplementedError

# Property methods (accessed via getattr)
def property_getter(self):
    """Property getter method.""" 
    pass

def property_setter(self, value):
    """Property setter method."""
    pass

# Magic methods that appear unused but are called by Python
def __str__(self):
    """String representation."""
    pass

def __repr__(self):
    """Object representation."""
    pass

def __eq__(self, other):
    """Equality comparison."""
    pass

def __hash__(self):
    """Hash function."""
    pass

# Metaclass patterns
class MetaClass(type):
    """Metaclass for dynamic class creation."""
    pass

# Import hooks and finders
class ImportHook:
    """Custom import hook."""
    def find_spec(self, name, path, target=None):
        return None