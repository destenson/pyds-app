"""
DeepStream element creation and management with version compatibility.

This module provides version-compatible DeepStream element creation, GPU memory
management, and plugin configuration across DeepStream 5.x through 7.x.
"""

import time
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
# Try to import GStreamer dependencies with fallback
try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst, GObject
    GSTREAMER_AVAILABLE = True
except (ImportError, ValueError):
    # Mock GStreamer for development without actual installation
    GSTREAMER_AVAILABLE = False
    
    class MockElement:
        def __init__(self, name=None):
            self.props = type('props', (), {})()
            self.name = name
    
    class MockGstClass:
        def __init__(self):
            self.Element = MockElement
        
        def __getattr__(self, name):
            return type(name, (), {})
    
    class MockGObject:
        pass
    
    Gst = MockGstClass()
    GObject = MockGObject()

from ..config import AppConfig, PipelineConfig, DetectionConfig
from ..utils.errors import DeepStreamError, PipelineError, handle_error
from ..utils.logging import get_logger, performance_context
from ..utils.deepstream import get_deepstream_info, get_deepstream_api, DeepStreamVersion, APIType


class MemoryType(Enum):
    """GPU memory types for NVIDIA buffers."""
    DEVICE = "device"           # NVBUF_MEM_CUDA_DEVICE
    UNIFIED = "unified"         # NVBUF_MEM_CUDA_UNIFIED  
    PINNED = "pinned"          # NVBUF_MEM_CUDA_PINNED
    DEFAULT = "default"         # Use system default


class ElementType(Enum):
    """DeepStream element types."""
    STREAMMUX = "nvstreammux"
    INFERENCE = "nvinfer"
    TRACKER = "nvtracker"
    OSD = "nvdsosd"
    CONVERTER = "nvvideoconvert"
    ENCODER = "nvv4l2h264enc"
    DECODER = "nvv4l2decoder"
    ANALYTICS = "nvdsanalytics"
    MSGCONV = "nvmsgconv"
    MSGBROKER = "nvmsgbroker"
    TRANSFORM = "nvegltransform"


@dataclass
class ElementProperties:
    """Properties for DeepStream elements."""
    element_type: ElementType
    properties: Dict[str, Any] = field(default_factory=dict)
    gpu_id: int = 0
    memory_type: MemoryType = MemoryType.DEFAULT
    batch_size: int = 1
    required_plugins: List[str] = field(default_factory=list)
    version_specific: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class ElementInfo:
    """Information about created DeepStream elements."""
    element: Gst.Element
    element_type: ElementType
    name: str
    gpu_id: int
    memory_type: MemoryType
    properties: Dict[str, Any]
    created_at: float = field(default_factory=time.time)
    performance_stats: Dict[str, float] = field(default_factory=dict)


class PluginManager:
    """Manages DeepStream plugin availability and loading."""
    
    def __init__(self):
        """Initialize plugin manager."""
        self.logger = get_logger(__name__)
        self._registry = Gst.Registry.get()
        self._available_plugins: Set[str] = set()
        self._deepstream_plugins: Dict[str, bool] = {}
        self._plugin_versions: Dict[str, str] = {}
        
        # Scan available plugins
        self._scan_plugins()
    
    def _scan_plugins(self):
        """Scan and catalog available plugins."""
        try:
            self.logger.debug("Scanning available GStreamer plugins...")
            
            # Core DeepStream plugins to check
            deepstream_plugins = {
                'nvstreammux': 'Stream multiplexer',
                'nvinfer': 'Inference engine',
                'nvtracker': 'Object tracker',
                'nvdsosd': 'On-screen display',
                'nvvideoconvert': 'Video converter',
                'nvv4l2h264enc': 'H.264 encoder',
                'nvv4l2h265enc': 'H.265 encoder',
                'nvv4l2decoder': 'Video decoder',
                'nvdsanalytics': 'Analytics plugin',
                'nvmsgconv': 'Message converter',
                'nvmsgbroker': 'Message broker',
                'nvegltransform': 'EGL transform',
                'nvjpegenc': 'JPEG encoder',
                'nvjpegdec': 'JPEG decoder'
            }
            
            for plugin_name, description in deepstream_plugins.items():
                plugin = self._registry.find_plugin(plugin_name)
                is_available = plugin is not None
                
                self._deepstream_plugins[plugin_name] = is_available
                if is_available:
                    self._available_plugins.add(plugin_name)
                    version = plugin.get_version() if plugin else "unknown"
                    self._plugin_versions[plugin_name] = version
                    self.logger.debug(f"Found {plugin_name} v{version}: {description}")
                else:
                    self.logger.warning(f"Missing {plugin_name}: {description}")
            
            self.logger.info(f"Found {len(self._available_plugins)} DeepStream plugins")
        
        except Exception as e:
            self.logger.error(f"Error scanning plugins: {e}")
    
    def is_plugin_available(self, plugin_name: str) -> bool:
        """Check if a plugin is available."""
        return plugin_name in self._available_plugins
    
    def get_plugin_version(self, plugin_name: str) -> Optional[str]:
        """Get version of a plugin."""
        return self._plugin_versions.get(plugin_name)
    
    def get_available_plugins(self) -> Set[str]:
        """Get set of available plugin names."""
        return self._available_plugins.copy()
    
    def get_missing_plugins(self, required_plugins: List[str]) -> List[str]:
        """Get list of missing required plugins."""
        return [plugin for plugin in required_plugins if not self.is_plugin_available(plugin)]
    
    def verify_deepstream_installation(self) -> Tuple[bool, List[str]]:
        """
        Verify DeepStream installation completeness.
        
        Returns:
            Tuple of (is_complete, missing_plugins)
        """
        critical_plugins = ['nvstreammux', 'nvinfer', 'nvdsosd']
        missing = self.get_missing_plugins(critical_plugins)
        return len(missing) == 0, missing


class GPUMemoryManager:
    """Manages GPU memory allocation and optimization."""
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize GPU memory manager.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        self._memory_pools: Dict[MemoryType, int] = {}
        self._allocated_memory: Dict[str, int] = {}
        self._deepstream_info = get_deepstream_info()
        
        # Initialize memory type mappings
        self._init_memory_mappings()
    
    def _init_memory_mappings(self):
        """Initialize memory type mappings for different DeepStream versions."""
        self._memory_type_values = {
            MemoryType.DEVICE: 0,    # NVBUF_MEM_CUDA_DEVICE
            MemoryType.UNIFIED: 1,   # NVBUF_MEM_CUDA_UNIFIED
            MemoryType.PINNED: 2,    # NVBUF_MEM_CUDA_PINNED
            MemoryType.DEFAULT: 0    # Default to device memory
        }
        
        # Version-specific adjustments
        if self._deepstream_info.major_version >= 6:
            # DeepStream 6.x+ may have different memory type values
            pass
    
    def get_memory_type_value(self, memory_type: MemoryType) -> int:
        """Get numeric value for memory type."""
        return self._memory_type_values.get(memory_type, 0)
    
    def get_optimal_memory_type(self, element_type: ElementType, batch_size: int) -> MemoryType:
        """
        Determine optimal memory type for element.
        
        Args:
            element_type: Type of DeepStream element
            batch_size: Processing batch size
            
        Returns:
            Optimal memory type
        """
        # For high-throughput scenarios, prefer device memory
        if batch_size > 4:
            return MemoryType.DEVICE
        
        # For inference elements, prefer device memory
        if element_type in [ElementType.INFERENCE, ElementType.ANALYTICS]:
            return MemoryType.DEVICE
        
        # For display/output elements, unified memory may be better
        if element_type in [ElementType.OSD, ElementType.TRANSFORM]:
            return MemoryType.UNIFIED
        
        # Default based on configuration
        config_memory = getattr(self.config, 'memory_type', 'device')
        return MemoryType(config_memory)
    
    def estimate_memory_usage(self, width: int, height: int, batch_size: int, fps: float) -> int:
        """
        Estimate memory usage in MB.
        
        Args:
            width: Frame width
            height: Frame height
            batch_size: Batch size
            fps: Frames per second
            
        Returns:
            Estimated memory usage in MB
        """
        # Estimate based on frame size and processing requirements
        bytes_per_pixel = 1.5  # NV12 format
        frame_size = width * height * bytes_per_pixel
        
        # Account for multiple buffers in pipeline
        buffer_multiplier = 3  # Input, processing, output buffers
        memory_per_stream = frame_size * buffer_multiplier
        
        # Total memory for batch
        total_memory = memory_per_stream * batch_size
        
        # Add overhead for metadata and processing
        overhead_multiplier = 1.2
        
        return int((total_memory * overhead_multiplier) / (1024 * 1024))  # Convert to MB
    
    def configure_memory_pools(self, elements: List[ElementInfo]):
        """Configure memory pools for optimal performance."""
        try:
            total_memory_mb = 0
            
            for element_info in elements:
                estimated_mb = self.estimate_memory_usage(
                    self.config.width,
                    self.config.height,
                    self.config.batch_size,
                    self.config.fps
                )
                total_memory_mb += estimated_mb
                self._allocated_memory[element_info.name] = estimated_mb
            
            self.logger.info(f"Estimated total GPU memory usage: {total_memory_mb} MB")
            
            # Log memory allocation per element
            for name, memory_mb in self._allocated_memory.items():
                self.logger.debug(f"Element {name}: ~{memory_mb} MB")
        
        except Exception as e:
            self.logger.error(f"Error configuring memory pools: {e}")


class DeepStreamElementFactory:
    """Factory for creating DeepStream elements with version compatibility."""
    
    def __init__(self, config: AppConfig):
        """
        Initialize element factory.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        self._plugin_manager = PluginManager()
        self._memory_manager = GPUMemoryManager(config.pipeline)
        self._deepstream_info = get_deepstream_info()
        self._deepstream_api = get_deepstream_api()
        
        # Verify DeepStream installation
        is_complete, missing = self._plugin_manager.verify_deepstream_installation()
        if not is_complete:
            self.logger.warning(f"Incomplete DeepStream installation. Missing: {missing}")
    
    def create_element(self, element_type: ElementType, name: str, **kwargs) -> ElementInfo:
        """
        Create a DeepStream element with version compatibility.
        
        Args:
            element_type: Type of element to create
            name: Element name
            **kwargs: Additional element properties
            
        Returns:
            ElementInfo with created element and metadata
            
        Raises:
            DeepStreamError: If element creation fails
        """
        plugin_name = element_type.value
        
        # Check plugin availability
        if not self._plugin_manager.is_plugin_available(plugin_name):
            raise DeepStreamError(
                f"Plugin {plugin_name} not available",
                plugin_name=plugin_name,
                deepstream_version=self._deepstream_info.version_string
            )
        
        try:
            with performance_context(f"create_element_{name}"):
                # Create GStreamer element
                element = Gst.ElementFactory.make(plugin_name, name)
                if not element:
                    raise DeepStreamError(f"Failed to create element {plugin_name}")
                
                # Configure element properties
                properties = self._get_default_properties(element_type)
                properties.update(kwargs)
                
                # Apply version-specific configurations
                self._apply_version_specific_config(element, element_type, properties)
                
                # Set common properties
                self._set_common_properties(element, element_type, properties)
                
                # Configure GPU and memory settings
                memory_type = self._memory_manager.get_optimal_memory_type(
                    element_type, 
                    self.config.pipeline.batch_size
                )
                
                element_info = ElementInfo(
                    element=element,
                    element_type=element_type,
                    name=name,
                    gpu_id=self.config.pipeline.gpu_id,
                    memory_type=memory_type,
                    properties=properties
                )
                
                self.logger.debug(f"Created {plugin_name} element: {name}")
                return element_info
        
        except Exception as e:
            raise DeepStreamError(
                f"Failed to create {plugin_name} element: {e}",
                plugin_name=plugin_name,
                element_name=name,
                original_exception=e
            )
    
    def _get_default_properties(self, element_type: ElementType) -> Dict[str, Any]:
        """Get default properties for element type."""
        defaults = {
            ElementType.STREAMMUX: {
                'width': self.config.pipeline.width,
                'height': self.config.pipeline.height,
                'batch-size': self.config.pipeline.batch_size,
                'batched-push-timeout': self.config.pipeline.batch_timeout_us,
                'gpu-id': self.config.pipeline.gpu_id,
                'live-source': True
            },
            ElementType.INFERENCE: {
                'config-file-path': '',  # Must be set by caller
                'batch-size': self.config.pipeline.batch_size,
                'unique-id': 1,
                'gpu-id': self.config.pipeline.gpu_id,
                'interval': 0,  # Process every frame
                'gie-unique-id': 1
            },
            ElementType.TRACKER: {
                'tracker-width': 640,
                'tracker-height': 384,
                'gpu-id': self.config.pipeline.gpu_id,
                'll-lib-file': '/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so',
                'll-config-file': '',  # Must be set by caller
                'enable-batch-process': True,
                'enable-past-frame': True
            },
            ElementType.OSD: {
                'process-mode': 0,  # CPU mode
                'display-text': True,
                'display-bbox': True,
                'display-mask': False,
                'display-clock': True,
                'gpu-id': self.config.pipeline.gpu_id,
                'text-size': 15,
                'text-color': 1,  # White
                'text-bg-color': 0,  # Black
                'font': 'Serif',
                'show-clock': True
            },
            ElementType.CONVERTER: {
                'gpu-id': self.config.pipeline.gpu_id,
                'nvbuf-memory-type': 0  # Device memory
            },
            ElementType.ENCODER: {
                'bitrate': 4000000,  # 4 Mbps
                'gpu-id': self.config.pipeline.gpu_id,
                'profile': 0,  # Baseline
                'insert-sps-pps': True,
                'bufapi-version': True
            },
            ElementType.ANALYTICS: {
                'config-file': '',  # Must be set by caller
                'gpu-id': self.config.pipeline.gpu_id,
                'unique-id': 1
            },
            ElementType.MSGCONV: {
                'config': '',  # Must be set by caller
                'payload-type': 0,  # JSON
                'msg2p-newapi': True,
                'frame-interval': 1
            },
            ElementType.MSGBROKER: {
                'config': '',  # Must be set by caller
                'proto-lib': '/opt/nvidia/deepstream/deepstream/lib/libnvds_kafka_proto.so',
                'sync': False
            },
            ElementType.TRANSFORM: {
                'gpu-id': self.config.pipeline.gpu_id
            }
        }
        
        return defaults.get(element_type, {})
    
    def _apply_version_specific_config(self, element: Gst.Element, element_type: ElementType, properties: Dict[str, Any]):
        """Apply version-specific configurations."""
        version = self._deepstream_info.major_version
        
        try:
            if element_type == ElementType.STREAMMUX:
                if version >= 6:
                    # DeepStream 6.x+ has different property names
                    if 'nvbuf-memory-type' not in properties:
                        memory_type = self._memory_manager.get_memory_type_value(MemoryType.DEFAULT)
                        properties['nvbuf-memory-type'] = memory_type
                    
                    # Set interpolation method for better quality
                    if hasattr(element.props, 'interpolation_method'):
                        properties['interpolation-method'] = 1  # Bilinear
                
                elif version == 5:
                    # DeepStream 5.x specific configurations
                    if 'enable-padding' not in properties:
                        properties['enable-padding'] = False
            
            elif element_type == ElementType.INFERENCE:
                if version >= 6:
                    # DeepStream 6.x+ supports more inference configurations
                    if 'infer-on-gie-id' not in properties:
                        properties['infer-on-gie-id'] = -1  # Infer on all
                    
                    if 'operate-on-gie-id' not in properties:
                        properties['operate-on-gie-id'] = -1
                
                # Set tensor output format based on version
                if version >= 7:
                    properties['output-tensor-meta'] = True
            
            elif element_type == ElementType.TRACKER:
                if version >= 6:
                    # Use newer tracker library if available
                    new_tracker_lib = '/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so'
                    if Path(new_tracker_lib).exists():
                        properties['ll-lib-file'] = new_tracker_lib
                    
                    # Enable batch processing for better performance
                    properties['enable-batch-process'] = True
            
            elif element_type == ElementType.OSD:
                if version >= 6:
                    # DeepStream 6.x+ has improved OSD features
                    properties['hw-blend-color-attr'] = True
                    properties['border-width'] = 3
        
        except Exception as e:
            self.logger.warning(f"Error applying version-specific config: {e}")
    
    def _set_common_properties(self, element: Gst.Element, element_type: ElementType, properties: Dict[str, Any]):
        """Set common properties on element."""
        try:
            for prop_name, prop_value in properties.items():
                try:
                    # Check if property exists
                    if hasattr(element.props, prop_name.replace('-', '_')):
                        element.set_property(prop_name, prop_value)
                        self.logger.debug(f"Set {prop_name}={prop_value} on {element.get_name()}")
                    else:
                        self.logger.debug(f"Property {prop_name} not available on {element.get_name()}")
                
                except Exception as e:
                    self.logger.warning(f"Failed to set property {prop_name}={prop_value}: {e}")
        
        except Exception as e:
            self.logger.error(f"Error setting element properties: {e}")
    
    def create_inference_element(self, name: str, config_file: str, model_engine: Optional[str] = None, **kwargs) -> ElementInfo:
        """
        Create inference element with model configuration.
        
        Args:
            name: Element name
            config_file: Path to inference configuration file
            model_engine: Optional path to TensorRT engine file
            **kwargs: Additional properties
            
        Returns:
            ElementInfo for inference element
        """
        properties = {
            'config-file-path': config_file,
            **kwargs
        }
        
        if model_engine:
            properties['model-engine-file'] = model_engine
        
        # Set detection-specific properties
        detection_config = self.config.detection
        properties.update({
            'batch-size': min(detection_config.max_objects, self.config.pipeline.batch_size),
            'interval': 0 if detection_config.batch_inference else 1,
            'gpu-id': self.config.pipeline.gpu_id
        })
        
        return self.create_element(ElementType.INFERENCE, name, **properties)
    
    def create_tracker_element(self, name: str, config_file: str, **kwargs) -> ElementInfo:
        """
        Create tracker element with configuration.
        
        Args:
            name: Element name
            config_file: Path to tracker configuration file
            **kwargs: Additional properties
            
        Returns:
            ElementInfo for tracker element
        """
        properties = {
            'll-config-file': config_file,
            **kwargs
        }
        
        return self.create_element(ElementType.TRACKER, name, **properties)
    
    def create_osd_element(self, name: str, **kwargs) -> ElementInfo:
        """
        Create OSD element for visualization.
        
        Args:
            name: Element name
            **kwargs: Additional properties
            
        Returns:
            ElementInfo for OSD element
        """
        # Set OSD-specific configurations
        properties = {
            'display-text': True,
            'display-bbox': True,
            'display-clock': True,
            'process-mode': 0,  # CPU mode for compatibility
            **kwargs
        }
        
        return self.create_element(ElementType.OSD, name, **properties)
    
    def get_plugin_manager(self) -> PluginManager:
        """Get plugin manager instance."""
        return self._plugin_manager
    
    def get_memory_manager(self) -> GPUMemoryManager:
        """Get memory manager instance."""
        return self._memory_manager


class DeepStreamElementsManager:
    """
    High-level manager for DeepStream elements and pipelines.
    
    Provides unified interface for creating, configuring, and managing
    DeepStream elements with version compatibility and optimization.
    """
    
    def __init__(self, config: AppConfig):
        """
        Initialize elements manager.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        self._factory = DeepStreamElementFactory(config)
        self._elements: Dict[str, ElementInfo] = {}
        self._element_groups: Dict[str, List[str]] = {}
        
        self.logger.info("DeepStreamElementsManager initialized")
    
    def create_detection_pipeline_elements(self, group_name: str = "detection") -> List[ElementInfo]:
        """
        Create standard detection pipeline elements.
        
        Args:
            group_name: Group name for element management
            
        Returns:
            List of created elements
        """
        elements = []
        
        try:
            # Create stream muxer
            streammux = self._factory.create_element(
                ElementType.STREAMMUX,
                "stream-muxer"
            )
            elements.append(streammux)
            
            # Create inference element
            if self.config.detection.model_engine_file:
                nvinfer = self._factory.create_inference_element(
                    "primary-inference",
                    config_file="config/yolo_config.txt",  # Default config
                    model_engine=self.config.detection.model_engine_file
                )
            else:
                nvinfer = self._factory.create_inference_element(
                    "primary-inference",
                    config_file="config/yolo_config.txt"
                )
            elements.append(nvinfer)
            
            # Create tracker if enabled
            if self.config.detection.enable_tracking:
                tracker = self._factory.create_tracker_element(
                    "tracker",
                    config_file=self.config.detection.tracker_config_file or "config/tracker_config.yml"
                )
                elements.append(tracker)
            
            # Create OSD for visualization
            osd = self._factory.create_osd_element("onscreendisplay")
            elements.append(osd)
            
            # Create converter for output
            converter = self._factory.create_element(
                ElementType.CONVERTER,
                "nvvidconv"
            )
            elements.append(converter)
            
            # Store elements
            element_names = []
            for element_info in elements:
                self._elements[element_info.name] = element_info
                element_names.append(element_info.name)
            
            self._element_groups[group_name] = element_names
            
            self.logger.info(f"Created detection pipeline with {len(elements)} elements")
            return elements
        
        except Exception as e:
            # Clean up partially created elements
            for element_info in elements:
                try:
                    if element_info.name in self._elements:
                        del self._elements[element_info.name]
                except Exception:
                    pass
            
            raise DeepStreamError(f"Failed to create detection pipeline elements: {e}", original_exception=e)
    
    def create_analytics_pipeline_elements(self, group_name: str = "analytics") -> List[ElementInfo]:
        """
        Create analytics pipeline elements.
        
        Args:
            group_name: Group name for element management
            
        Returns:
            List of created elements
        """
        elements = []
        
        try:
            # Create analytics element
            analytics = self._factory.create_element(
                ElementType.ANALYTICS,
                "analytics",
                config_file="config/analytics_config.txt"
            )
            elements.append(analytics)
            
            # Create message converter
            msgconv = self._factory.create_element(
                ElementType.MSGCONV,
                "msgconv",
                config="config/msgconv_config.txt"
            )
            elements.append(msgconv)
            
            # Create message broker
            msgbroker = self._factory.create_element(
                ElementType.MSGBROKER,
                "msgbroker",
                config="config/msgbroker_config.txt"
            )
            elements.append(msgbroker)
            
            # Store elements
            element_names = []
            for element_info in elements:
                self._elements[element_info.name] = element_info
                element_names.append(element_info.name)
            
            self._element_groups[group_name] = element_names
            
            self.logger.info(f"Created analytics pipeline with {len(elements)} elements")
            return elements
        
        except Exception as e:
            # Clean up partially created elements
            for element_info in elements:
                try:
                    if element_info.name in self._elements:
                        del self._elements[element_info.name]
                except Exception:
                    pass
            
            raise DeepStreamError(f"Failed to create analytics pipeline elements: {e}", original_exception=e)
    
    def link_elements(self, elements: List[ElementInfo]) -> bool:
        """
        Link elements in sequence.
        
        Args:
            elements: List of elements to link
            
        Returns:
            True if all elements linked successfully
        """
        try:
            for i in range(len(elements) - 1):
                src_element = elements[i].element
                dst_element = elements[i + 1].element
                
                if not src_element.link(dst_element):
                    raise PipelineError(
                        f"Failed to link {src_element.get_name()} to {dst_element.get_name()}"
                    )
                
                self.logger.debug(f"Linked {src_element.get_name()} -> {dst_element.get_name()}")
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error linking elements: {e}")
            return False
    
    def get_element(self, name: str) -> Optional[ElementInfo]:
        """Get element by name."""
        return self._elements.get(name)
    
    def get_elements_by_group(self, group_name: str) -> List[ElementInfo]:
        """Get all elements in a group."""
        element_names = self._element_groups.get(group_name, [])
        return [self._elements[name] for name in element_names if name in self._elements]
    
    def get_all_elements(self) -> Dict[str, ElementInfo]:
        """Get all managed elements."""
        return self._elements.copy()
    
    def remove_element(self, name: str) -> bool:
        """
        Remove element from management.
        
        Args:
            name: Element name
            
        Returns:
            True if element removed successfully
        """
        try:
            if name in self._elements:
                element_info = self._elements[name]
                
                # Remove from groups
                for group_name, element_names in self._element_groups.items():
                    if name in element_names:
                        element_names.remove(name)
                
                # Remove from elements
                del self._elements[name]
                
                self.logger.debug(f"Removed element {name}")
                return True
            
            return False
        
        except Exception as e:
            self.logger.error(f"Error removing element {name}: {e}")
            return False
    
    def remove_group(self, group_name: str) -> bool:
        """
        Remove all elements in a group.
        
        Args:
            group_name: Group name
            
        Returns:
            True if group removed successfully
        """
        try:
            if group_name not in self._element_groups:
                return True
            
            element_names = self._element_groups[group_name].copy()
            
            for name in element_names:
                self.remove_element(name)
            
            del self._element_groups[group_name]
            
            self.logger.info(f"Removed element group {group_name}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error removing group {group_name}: {e}")
            return False
    
    def optimize_elements_for_performance(self, elements: List[ElementInfo]):
        """Optimize elements for performance."""
        try:
            memory_manager = self._factory.get_memory_manager()
            memory_manager.configure_memory_pools(elements)
            
            # Apply performance optimizations
            for element_info in elements:
                element = element_info.element
                element_type = element_info.element_type
                
                # Set performance-related properties
                if element_type == ElementType.STREAMMUX:
                    # Optimize batching
                    if hasattr(element.props, 'batched_push_timeout'):
                        element.set_property('batched-push-timeout', 4000)  # 4ms
                    
                    # Enable live source optimizations
                    if hasattr(element.props, 'live_source'):
                        element.set_property('live-source', True)
                
                elif element_type == ElementType.INFERENCE:
                    # Enable batch processing
                    if hasattr(element.props, 'batch_size'):
                        element.set_property('batch-size', self.config.pipeline.batch_size)
                    
                    # Optimize interval
                    if hasattr(element.props, 'interval'):
                        element.set_property('interval', 0)  # Process every frame
                
                elif element_type == ElementType.TRACKER:
                    # Enable batch processing for tracker
                    if hasattr(element.props, 'enable_batch_process'):
                        element.set_property('enable-batch-process', True)
                
                self.logger.debug(f"Applied performance optimizations to {element.get_name()}")
            
            self.logger.info(f"Optimized {len(elements)} elements for performance")
        
        except Exception as e:
            self.logger.error(f"Error optimizing elements: {e}")
    
    def get_element_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all elements."""
        stats = {}
        
        for name, element_info in self._elements.items():
            stats[name] = {
                'type': element_info.element_type.value,
                'gpu_id': element_info.gpu_id,
                'memory_type': element_info.memory_type.value,
                'created_at': element_info.created_at,
                'uptime_seconds': time.time() - element_info.created_at,
                'properties': element_info.properties,
                'performance_stats': element_info.performance_stats
            }
        
        return stats
    
    def cleanup(self):
        """Clean up all managed elements."""
        try:
            self.logger.info("Cleaning up DeepStream elements...")
            
            # Remove all groups
            group_names = list(self._element_groups.keys())
            for group_name in group_names:
                self.remove_group(group_name)
            
            # Clear remaining elements
            self._elements.clear()
            
            self.logger.info("DeepStream elements cleanup completed")
        
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


# Module-level factory instance for global access
_global_factory: Optional[DeepStreamElementFactory] = None
_global_manager: Optional[DeepStreamElementsManager] = None


def get_element_factory(config: Optional[AppConfig] = None) -> DeepStreamElementFactory:
    """Get global element factory instance."""
    global _global_factory
    if _global_factory is None and config is not None:
        _global_factory = DeepStreamElementFactory(config)
    return _global_factory


def get_elements_manager(config: Optional[AppConfig] = None) -> DeepStreamElementsManager:
    """Get global elements manager instance."""
    global _global_manager
    if _global_manager is None and config is not None:
        _global_manager = DeepStreamElementsManager(config)
    return _global_manager