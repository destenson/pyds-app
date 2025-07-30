"""
DeepStream version detection and compatibility layer.

This module provides automatic version detection, API selection, and compatibility
shims to support DeepStream 5.x, 6.x, and 7.x across different platforms.
"""

import os
import sys
import re
import subprocess
from typing import Optional, Dict, Any, Tuple, Union, Type
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path

from .errors import DeepStreamError, VersionCompatibilityError


class DeepStreamVersion(Enum):
    """DeepStream major version identifiers."""
    V5 = "5"
    V6 = "6" 
    V7 = "7"
    UNKNOWN = "unknown"


class APIType(Enum):
    """DeepStream Python API types."""
    PYDS = "pyds"           # DeepStream 5.x direct Python bindings
    GI_REPOSITORY = "gi"    # DeepStream 6.x+ GObject Introspection bindings
    UNKNOWN = "unknown"


@dataclass
class DeepStreamInfo:
    """Information about detected DeepStream installation."""
    version: DeepStreamVersion
    version_string: str
    api_type: APIType
    install_path: Optional[str] = None
    python_bindings_path: Optional[str] = None
    capabilities: Dict[str, bool] = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = {}
    
    @property
    def major_version(self) -> int:
        """Get major version as integer."""
        if self.version == DeepStreamVersion.V5:
            return 5
        elif self.version == DeepStreamVersion.V6:
            return 6
        elif self.version == DeepStreamVersion.V7:
            return 7
        else:
            return 0
    
    @property
    def minor_version(self) -> int:
        """Get minor version as integer."""
        try:
            match = re.search(r'(\d+)\.(\d+)', self.version_string)
            if match:
                return int(match.group(2))
        except Exception:
            pass
        return 0
    
    def is_compatible_with(self, min_version: str) -> bool:
        """Check if current version is compatible with minimum required version."""
        try:
            min_major, min_minor = map(int, min_version.split('.'))
            return (self.major_version > min_major or 
                   (self.major_version == min_major and self.minor_version >= min_minor))
        except Exception:
            return False


class DeepStreamDetector:
    """Detects DeepStream installation and capabilities."""
    
    # Common DeepStream installation paths
    INSTALL_PATHS = [
        "/opt/nvidia/deepstream/deepstream",
        "/opt/nvidia/deepstream/deepstream-7.0",
        "/opt/nvidia/deepstream/deepstream-6.4",
        "/opt/nvidia/deepstream/deepstream-6.3", 
        "/opt/nvidia/deepstream/deepstream-6.2",
        "/opt/nvidia/deepstream/deepstream-6.1",
        "/opt/nvidia/deepstream/deepstream-6.0",
        "/opt/nvidia/deepstream/deepstream-5.1",
        "/opt/nvidia/deepstream/deepstream-5.0",
        "C:\\Program Files\\NVIDIA Corporation\\DeepStream SDK",
        "C:\\DeepStream",
    ]
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._cached_info: Optional[DeepStreamInfo] = None
    
    def detect_deepstream(self, force_refresh: bool = False) -> DeepStreamInfo:
        """
        Detect DeepStream installation and return version information.
        
        Args:
            force_refresh: Force re-detection even if cached
            
        Returns:
            DeepStreamInfo with detected version and capabilities
            
        Raises:
            VersionCompatibilityError: If DeepStream is not found or unsupported
        """
        if self._cached_info and not force_refresh:
            return self._cached_info
        
        self.logger.info("Detecting DeepStream installation...")
        
        # Try multiple detection methods
        detection_methods = [
            self._detect_from_python_import,
            self._detect_from_install_path,
            self._detect_from_environment,
            self._detect_from_system_packages
        ]
        
        for method in detection_methods:
            try:
                info = method()
                if info and info.version != DeepStreamVersion.UNKNOWN:
                    self.logger.info(f"Detected DeepStream {info.version_string} using {method.__name__}")
                    info.capabilities = self._detect_capabilities(info)
                    self._cached_info = info
                    return info
            except Exception as e:
                self.logger.debug(f"Detection method {method.__name__} failed: {e}")
                continue
        
        # No DeepStream found
        raise VersionCompatibilityError(
            "DeepStream SDK not found. Please install DeepStream 5.0+ and ensure Python bindings are available.",
            required_version="5.0+",
            detected_version="none"
        )
    
    def _detect_from_python_import(self) -> Optional[DeepStreamInfo]:
        """Try to detect DeepStream by importing Python bindings."""
        # Try pyds (DeepStream 5.x)
        try:
            import pyds
            version_string = getattr(pyds, '__version__', 'unknown')
            
            # Try to get more specific version info
            if hasattr(pyds, 'get_nvds_version'):
                try:
                    version_string = pyds.get_nvds_version()
                except Exception:
                    pass
            
            return DeepStreamInfo(
                version=self._parse_version(version_string),
                version_string=version_string,
                api_type=APIType.PYDS,
                python_bindings_path=pyds.__file__ if hasattr(pyds, '__file__') else None
            )
        except ImportError:
            pass
        
        # Try gi.repository (DeepStream 6.x+)
        try:
            import gi
            gi.require_version('Gst', '1.0')
            from gi.repository import Gst
            
            # Initialize GStreamer to access DeepStream plugins
            Gst.init(None)
            
            # Try to find DeepStream plugins to determine version
            registry = Gst.Registry.get()
            
            # Look for DeepStream-specific plugins
            ds_plugins = ['nvstreammux', 'nvinfer', 'nvtracker', 'nvdsosd']
            found_plugins = []
            
            for plugin_name in ds_plugins:
                plugin = registry.find_plugin(plugin_name)
                if plugin:
                    found_plugins.append(plugin_name)
            
            if found_plugins:
                # Assume 6.x+ if we found GStreamer-based DeepStream plugins
                version_string = "6.0+"  # Default assumption
                
                # Try to get more specific version from plugin info
                nvstreammux = registry.find_plugin('nvstreammux')
                if nvstreammux:
                    version_string = nvstreammux.get_version() or version_string
                
                return DeepStreamInfo(
                    version=self._parse_version(version_string),
                    version_string=version_string,
                    api_type=APIType.GI_REPOSITORY
                )
        except Exception:
            pass
        
        return None
    
    def _detect_from_install_path(self) -> Optional[DeepStreamInfo]:
        """Try to detect DeepStream from installation paths."""
        for install_path in self.INSTALL_PATHS:
            path = Path(install_path)
            if path.exists():
                # Try to find version from path name
                version_match = re.search(r'deepstream-?(\d+\.\d+)', str(path))
                if version_match:
                    version_string = version_match.group(1)
                    
                    # Check for Python bindings
                    bindings_paths = [
                        path / "sources" / "deepstream_python_apps" / "bindings",
                        path / "lib" / "python3" / "site-packages",
                        path / "python"
                    ]
                    
                    python_bindings_path = None
                    for bindings_path in bindings_paths:
                        if bindings_path.exists():
                            python_bindings_path = str(bindings_path)
                            break
                    
                    # Determine API type based on version
                    major_version = int(version_string.split('.')[0])
                    api_type = APIType.PYDS if major_version == 5 else APIType.GI_REPOSITORY
                    
                    return DeepStreamInfo(
                        version=self._parse_version(version_string),
                        version_string=version_string,
                        api_type=api_type,
                        install_path=str(path),
                        python_bindings_path=python_bindings_path
                    )
        
        return None
    
    def _detect_from_environment(self) -> Optional[DeepStreamInfo]:
        """Try to detect DeepStream from environment variables."""
        # Check common environment variables
        env_vars = ['DEEPSTREAM_DIR', 'DEEPSTREAM_PATH', 'DS_SDK_ROOT']
        
        for env_var in env_vars:
            path = os.environ.get(env_var)
            if path and Path(path).exists():
                # Try to extract version from path or version file
                version_file = Path(path) / "version"
                if version_file.exists():
                    try:
                        version_string = version_file.read_text().strip()
                        return DeepStreamInfo(
                            version=self._parse_version(version_string),
                            version_string=version_string,
                            api_type=self._infer_api_type(version_string),
                            install_path=path
                        )
                    except Exception:
                        pass
        
        return None
    
    def _detect_from_system_packages(self) -> Optional[DeepStreamInfo]:
        """Try to detect DeepStream from system package manager."""
        try:
            # Try dpkg on Debian/Ubuntu systems
            result = subprocess.run(
                ['dpkg', '-l', 'deepstream*'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                # Parse dpkg output to find DeepStream packages
                for line in result.stdout.split('\n'):
                    if 'deepstream' in line.lower():
                        parts = line.split()
                        if len(parts) >= 3:
                            package_name = parts[1]
                            version_string = parts[2]
                            
                            # Extract version from package name or version string
                            version_match = re.search(r'(\d+\.\d+)', package_name + version_string)
                            if version_match:
                                version = version_match.group(1)
                                return DeepStreamInfo(
                                    version=self._parse_version(version),
                                    version_string=version,
                                    api_type=self._infer_api_type(version)
                                )
        except Exception:
            pass
        
        # Try rpm on RedHat/CentOS systems
        try:
            result = subprocess.run(
                ['rpm', '-qa', 'deepstream*'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout.strip():
                # Parse rpm output
                for line in result.stdout.split('\n'):
                    if 'deepstream' in line.lower():
                        version_match = re.search(r'(\d+\.\d+)', line)
                        if version_match:
                            version = version_match.group(1)
                            return DeepStreamInfo(
                                version=self._parse_version(version),
                                version_string=version,
                                api_type=self._infer_api_type(version)
                            )
        except Exception:
            pass
        
        return None
    
    def _parse_version(self, version_string: str) -> DeepStreamVersion:
        """Parse version string to DeepStreamVersion enum."""
        if not version_string or version_string == 'unknown':
            return DeepStreamVersion.UNKNOWN
        
        # Extract major version number
        match = re.search(r'(\d+)', version_string)
        if match:
            major_version = int(match.group(1))
            if major_version == 5:
                return DeepStreamVersion.V5
            elif major_version == 6:
                return DeepStreamVersion.V6
            elif major_version >= 7:
                return DeepStreamVersion.V7
        
        return DeepStreamVersion.UNKNOWN
    
    def _infer_api_type(self, version_string: str) -> APIType:
        """Infer API type from version string."""
        version = self._parse_version(version_string)
        if version == DeepStreamVersion.V5:
            return APIType.PYDS
        elif version in [DeepStreamVersion.V6, DeepStreamVersion.V7]:
            return APIType.GI_REPOSITORY
        else:
            return APIType.UNKNOWN
    
    def _detect_capabilities(self, info: DeepStreamInfo) -> Dict[str, bool]:
        """Detect available capabilities for the DeepStream installation."""
        capabilities = {}
        
        try:
            if info.api_type == APIType.PYDS:
                # Test pyds capabilities
                import pyds
                capabilities.update({
                    'metadata_extraction': hasattr(pyds, 'gst_buffer_get_nvds_batch_meta'),
                    'object_meta': hasattr(pyds, 'NvDsObjectMeta'),
                    'frame_meta': hasattr(pyds, 'NvDsFrameMeta'),
                    'batch_meta': hasattr(pyds, 'NvDsBatchMeta'),
                    'user_meta': hasattr(pyds, 'alloc_nvds_user_meta'),
                    'string_handling': hasattr(pyds, 'get_string'),
                    'buffer_utils': hasattr(pyds, 'get_nvds_buf_surface'),
                    'analytics_meta': hasattr(pyds, 'NvDsAnalyticsObjInfo'),
                    'tracker_meta': hasattr(pyds, 'NvDsTracker'),
                })
                
            elif info.api_type == APIType.GI_REPOSITORY:
                # Test gi.repository capabilities
                try:
                    import gi
                    gi.require_version('Gst', '1.0')
                    from gi.repository import Gst
                    
                    # Initialize GStreamer
                    Gst.init(None)
                    registry = Gst.Registry.get()
                    
                    # Check for DeepStream plugins
                    ds_plugins = {
                        'nvstreammux': 'stream_muxer',
                        'nvinfer': 'inference',
                        'nvtracker': 'tracking',
                        'nvdsosd': 'on_screen_display',
                        'nvmsgconv': 'message_converter',
                        'nvmsgbroker': 'message_broker',
                        'nvdsanalytics': 'analytics',
                        'nvv4l2decoder': 'video_decoder',
                        'nvv4l2h264enc': 'h264_encoder',
                        'nvv4l2h265enc': 'h265_encoder'
                    }
                    
                    for plugin, capability in ds_plugins.items():
                        plugin_obj = registry.find_plugin(plugin)
                        capabilities[capability] = plugin_obj is not None
                        
                except Exception as e:
                    self.logger.debug(f"Error detecting GI capabilities: {e}")
            
            # Common capabilities
            capabilities.update({
                'gpu_support': self._check_gpu_support(),
                'python_bindings': True,  # If we got here, bindings are available
                'version_compatible': info.major_version >= 5
            })
            
        except Exception as e:
            self.logger.warning(f"Error detecting capabilities: {e}")
        
        return capabilities
    
    def _check_gpu_support(self) -> bool:
        """Check if GPU/CUDA support is available."""
        try:
            # Try to import and initialize NVML
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            return device_count > 0
        except Exception:
            pass
        
        try:
            # Check for CUDA runtime
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
            return result.returncode == 0
        except Exception:
            pass
        
        return False


class DeepStreamAPI:
    """Unified API abstraction for different DeepStream versions."""
    
    def __init__(self, info: DeepStreamInfo):
        self.info = info
        self.logger = logging.getLogger(__name__)
        self._pyds_module = None
        self._gi_modules = {}
        
        # Initialize appropriate bindings
        if info.api_type == APIType.PYDS:
            self._init_pyds()
        elif info.api_type == APIType.GI_REPOSITORY:
            self._init_gi()
        else:
            raise VersionCompatibilityError(
                f"Unsupported API type: {info.api_type}",
                detected_version=info.version_string
            )
    
    def _init_pyds(self):
        """Initialize pyds bindings for DeepStream 5.x."""
        try:
            import pyds
            self._pyds_module = pyds
            self.logger.debug("Initialized pyds bindings")
        except ImportError as e:
            raise DeepStreamError(
                f"Failed to import pyds module: {e}",
                deepstream_version=self.info.version_string,
                recoverable=False
            )
    
    def _init_gi(self):
        """Initialize GObject Introspection bindings for DeepStream 6.x+."""
        try:
            import gi
            gi.require_version('Gst', '1.0')
            gi.require_version('GObject', '2.0')
            
            from gi.repository import Gst, GObject, GLib
            
            # Initialize GStreamer
            GObject.threads_init()
            Gst.init(None)
            
            self._gi_modules = {
                'Gst': Gst,
                'GObject': GObject,
                'GLib': GLib
            }
            
            self.logger.debug("Initialized GI bindings")
        except Exception as e:
            raise DeepStreamError(
                f"Failed to initialize GI bindings: {e}",
                deepstream_version=self.info.version_string,
                recoverable=False
            )
    
    def get_batch_meta(self, gst_buffer) -> Any:
        """Get batch metadata from GStreamer buffer."""
        if self.info.api_type == APIType.PYDS:
            return self._pyds_module.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        else:
            # For GI bindings, metadata access may be different
            # This is a placeholder - actual implementation depends on DeepStream 6.x+ API
            raise NotImplementedError("GI batch meta extraction not yet implemented")
    
    def cast_frame_meta(self, frame_meta_data) -> Any:
        """Cast frame metadata to appropriate type."""
        if self.info.api_type == APIType.PYDS:
            return self._pyds_module.NvDsFrameMeta.cast(frame_meta_data)
        else:
            # GI equivalent
            raise NotImplementedError("GI frame meta casting not yet implemented")
    
    def cast_object_meta(self, obj_meta_data) -> Any:
        """Cast object metadata to appropriate type."""
        if self.info.api_type == APIType.PYDS:
            return self._pyds_module.NvDsObjectMeta.cast(obj_meta_data)
        else:
            # GI equivalent
            raise NotImplementedError("GI object meta casting not yet implemented")
    
    def get_string(self, string_ptr) -> str:
        """Get string from C string pointer."""
        if self.info.api_type == APIType.PYDS:
            return self._pyds_module.get_string(string_ptr)
        else:
            # GI string handling might be different
            return str(string_ptr) if string_ptr else ""
    
    def alloc_user_meta(self) -> Any:
        """Allocate user metadata."""
        if self.info.api_type == APIType.PYDS:
            return self._pyds_module.alloc_nvds_user_meta()
        else:
            raise NotImplementedError("GI user meta allocation not yet implemented")
    
    def get_surface(self, gst_buffer, frame_meta) -> Any:
        """Get NvBufSurface from buffer."""
        if self.info.api_type == APIType.PYDS:
            return self._pyds_module.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
        else:
            raise NotImplementedError("GI surface access not yet implemented")
    
    @property
    def constants(self) -> Dict[str, Any]:
        """Get API constants."""
        if self.info.api_type == APIType.PYDS:
            return {
                'NVBUF_MEM_CUDA_UNIFIED': getattr(self._pyds_module, 'NVBUF_MEM_CUDA_UNIFIED', 0),
                'NVBUF_MEM_CUDA_DEVICE': getattr(self._pyds_module, 'NVBUF_MEM_CUDA_DEVICE', 1),
                'NVDS_META_TYPE_USER': getattr(self._pyds_module, 'NvDsMetaType', {}).get('NVDS_USER_META', 0)
            }
        else:
            return {}


# Global DeepStream detector and API instances
_detector: Optional[DeepStreamDetector] = None
_api: Optional[DeepStreamAPI] = None
_info: Optional[DeepStreamInfo] = None


def get_deepstream_info(force_refresh: bool = False) -> DeepStreamInfo:
    """Get DeepStream installation information."""
    global _detector, _info
    
    if _detector is None:
        _detector = DeepStreamDetector()
    
    if _info is None or force_refresh:
        _info = _detector.detect_deepstream(force_refresh)
    
    return _info


def get_deepstream_api(force_refresh: bool = False) -> DeepStreamAPI:
    """Get unified DeepStream API instance."""
    global _api
    
    if _api is None or force_refresh:
        info = get_deepstream_info(force_refresh)
        _api = DeepStreamAPI(info)
    
    return _api


def check_version_compatibility(min_version: str) -> bool:
    """Check if detected DeepStream version meets minimum requirements."""
    try:
        info = get_deepstream_info()
        return info.is_compatible_with(min_version)
    except Exception:
        return False


def get_version_string() -> str:
    """Get DeepStream version string."""
    try:
        info = get_deepstream_info()
        return info.version_string
    except Exception:
        return "unknown"


def is_gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    try:
        info = get_deepstream_info()
        return info.capabilities.get('gpu_support', False)
    except Exception:
        return False


# Convenience functions for common version checks
def is_deepstream_5x() -> bool:
    """Check if running on DeepStream 5.x."""
    try:
        info = get_deepstream_info()
        return info.version == DeepStreamVersion.V5
    except Exception:
        return False


def is_deepstream_6x() -> bool:
    """Check if running on DeepStream 6.x."""
    try:
        info = get_deepstream_info()
        return info.version == DeepStreamVersion.V6
    except Exception:
        return False


def is_deepstream_7x() -> bool:
    """Check if running on DeepStream 7.x."""
    try:
        info = get_deepstream_info()
        return info.version == DeepStreamVersion.V7
    except Exception:
        return False


def uses_pyds_api() -> bool:
    """Check if using pyds API (DeepStream 5.x)."""
    try:
        info = get_deepstream_info()
        return info.api_type == APIType.PYDS
    except Exception:
        return False


def uses_gi_api() -> bool:
    """Check if using GI API (DeepStream 6.x+)."""
    try:
        info = get_deepstream_info()
        return info.api_type == APIType.GI_REPOSITORY
    except Exception:
        return False