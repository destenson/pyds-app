"""
GStreamer pipeline lifecycle management with DeepStream integration.

This module provides comprehensive pipeline management including multi-source support,
state management, bus message handling, and automatic error recovery.
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Callable, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
# Try to import GStreamer dependencies with fallback
try:
    import gi
    gi.require_version('Gst', '1.0')
    gi.require_version('GObject', '2.0')
    from gi.repository import Gst, GObject, GLib
    GSTREAMER_AVAILABLE = True
except (ImportError, ValueError):
    # Mock objects for development without GStreamer
    GSTREAMER_AVAILABLE = False
    
    class MockElement:
        def __init__(self, name=None):
            self.props = type('props', (), {})()
            self.name = name
            
        def set_property(self, name, value):
            setattr(self.props, name, value)
            
        def get_property(self, name):
            return getattr(self.props, name, None)
            
        def link(self, other):
            return True
            
        def get_static_pad(self, name):
            return MockPad()
            
        def get_request_pad(self, name):
            return MockPad()
            
        def add(self, element):
            pass
    
    class MockBin(MockElement):
        def __init__(self, name=None):
            super().__init__(name)
    
    class MockPad:
        def __init__(self):
            pass
            
        def link(self, other):
            return True
    
    class MockGstClass:
        def __init__(self):
            self.State = type('State', (), {
                'NULL': 'null', 'READY': 'ready', 'PAUSED': 'paused', 'PLAYING': 'playing'
            })()
            self.StateChangeReturn = type('StateChangeReturn', (), {
                'SUCCESS': 'success', 'ASYNC': 'async', 'FAILURE': 'failure'
            })()
            self.MessageType = type('MessageType', (), {
                'ERROR': 'error', 'WARNING': 'warning', 'INFO': 'info', 'EOS': 'eos'
            })()
            self.PadProbeReturn = type('PadProbeReturn', (), {
                'OK': 'ok', 'DROP': 'drop', 'REMOVE': 'remove'
            })()
            self.Pad = MockPad
            self.Element = MockElement
            self.Bin = MockBin
            self.ElementFactory = type('ElementFactory', (), {
                'make': staticmethod(lambda factory_name, element_name=None: MockElement(element_name))
            })()
        
        def __getattr__(self, name):
            # Return a generic mock class for any missing attribute
            return type(name, (), {})
        
        def init(self, args=None):
            pass
            
        def Pipeline(self, name=None):
            return MockPipeline()
    
    class MockGObject:
        @staticmethod
        def timeout_add(interval, callback, *args):
            return 1
            
        @staticmethod
        def source_remove(source_id):
            pass
    
    class MockGLib:
        @staticmethod
        def MainLoop():
            return MockMainLoop()
    
    class MockPipeline:
        def __init__(self):
            self.state = 'null'
            
        def get_by_name(self, name):
            return MockElement()
            
        def set_state(self, state):
            return MockGst.StateChangeReturn.SUCCESS
            
        def get_state(self, timeout=None):
            return (MockGst.StateChangeReturn.SUCCESS, self.state, 'pending')
            
        def get_bus(self):
            return MockBus()
            
        def add(self, element):
            pass
            
        def link(self, other):
            return True
    
    class MockElement:
        def __init__(self):
            self.props = type('props', (), {})()
            
        def set_property(self, name, value):
            setattr(self.props, name, value)
            
        def get_property(self, name):
            return getattr(self.props, name, None)
            
        def link(self, other):
            return True
    
    class MockBus:
        def add_signal_watch(self):
            pass
            
        def connect(self, signal, callback):
            pass
    
    class MockMainLoop:
        def run(self):
            pass
            
        def quit(self):
            pass
    
    Gst = MockGstClass()
    GObject = MockGObject
    GLib = MockGLib

from ..config import AppConfig, SourceConfig
from ..utils.errors import PipelineError, SourceError, DeepStreamError, handle_error, recovery_strategy
from ..utils.logging import get_logger, performance_context, log_pipeline_event
from ..utils.async_utils import get_task_manager, register_for_shutdown, ThreadSafeAsyncQueue
from ..utils.deepstream import get_deepstream_api, get_deepstream_info
from ..detection.models import VideoDetection, DetectionResult


class PipelineState(Enum):
    """Pipeline state enumeration."""
    NULL = "null"
    READY = "ready"
    PAUSED = "paused"
    PLAYING = "playing"
    ERROR = "error"


@dataclass
class PipelineInfo:
    """Information about a managed pipeline."""
    pipeline_id: str
    pipeline: Gst.Pipeline
    state: PipelineState = PipelineState.NULL
    sources: List[SourceConfig] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    error_count: int = 0
    fps: float = 0.0
    frame_count: int = 0
    memory_usage_mb: float = 0.0
    gpu_utilization: float = 0.0
    bus_watch_id: Optional[int] = None
    detection_probe_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class SourcePadInfo:
    """Information about source pads and their connections."""
    source_id: str
    pad: Gst.Pad
    sink_pad: Gst.Pad
    connected: bool = False
    last_frame_time: float = 0.0
    frame_count: int = 0
    error_count: int = 0


class PipelineManager:
    """
    Manages GStreamer pipelines with DeepStream integration.
    
    Provides multi-source pipeline management, state transitions, error recovery,
    and integration with detection and monitoring systems.
    """
    
    def __init__(self, config: AppConfig):
        """
        Initialize pipeline manager.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        self._lock = Lock()
        
        # Pipeline management
        self._pipelines: Dict[str, PipelineInfo] = {}
        self._source_pads: Dict[str, SourcePadInfo] = {}
        self._active_sources: Set[str] = set()
        
        # DeepStream integration
        self._deepstream_api = None
        self._deepstream_info = None
        
        # Callbacks and event handling
        self._detection_callbacks: List[Callable] = []
        self._state_change_callbacks: List[Callable] = []
        self._error_callbacks: List[Callable] = []
        
        # Performance monitoring
        self._fps_tracking: Dict[str, List[Tuple[float, int]]] = {}
        self._performance_metrics: Dict[str, Dict[str, float]] = {}
        
        # Async integration
        self._event_queue = ThreadSafeAsyncQueue(maxsize=1000)
        self._task_manager = get_task_manager()
        
        # Initialize GStreamer and DeepStream
        self._initialize_gstreamer()
        self._initialize_deepstream()
        
        # Register for shutdown
        register_for_shutdown("pipeline_manager", self)
        
        self.logger.info("PipelineManager initialized successfully")
    
    def _initialize_gstreamer(self):
        """Initialize GStreamer with thread safety."""
        try:
            # Initialize GObject threading
            GObject.threads_init()
            
            # Initialize GStreamer
            if not Gst.is_initialized():
                Gst.init(None)
                self.logger.info("GStreamer initialized")
            
            # Verify GStreamer version
            version = Gst.version()
            self.logger.info(f"GStreamer version: {version.major}.{version.minor}.{version.micro}")
            
            # Check for required plugins
            required_plugins = [
                'coreelements', 'videotestsrc', 'videoconvert', 'videoscale',
                'audiotestsrc', 'autodetect', 'playback'
            ]
            
            registry = Gst.Registry.get()
            missing_plugins = []
            
            for plugin_name in required_plugins:
                plugin = registry.find_plugin(plugin_name)
                if not plugin:
                    missing_plugins.append(plugin_name)
            
            if missing_plugins:
                self.logger.warning(f"Missing GStreamer plugins: {missing_plugins}")
            
        except Exception as e:
            raise PipelineError(
                f"Failed to initialize GStreamer: {e}",
                original_exception=e,
                recoverable=False
            )
    
    def _initialize_deepstream(self):
        """Initialize DeepStream API and check compatibility."""
        try:
            self._deepstream_info = get_deepstream_info()
            self._deepstream_api = get_deepstream_api()
            
            self.logger.info(
                f"DeepStream {self._deepstream_info.version_string} initialized "
                f"(API: {self._deepstream_info.api_type.value})"
            )
            
            # Log capabilities
            capabilities = self._deepstream_info.capabilities
            self.logger.info(f"DeepStream capabilities: {list(capabilities.keys())}")
            
            if not capabilities.get('gpu_support', False):
                self.logger.warning("GPU support not available - performance may be degraded")
            
        except Exception as e:
            raise DeepStreamError(
                f"Failed to initialize DeepStream: {e}",
                original_exception=e,
                recoverable=False
            )
    
    async def create_multi_source_pipeline(
        self, 
        sources: List[SourceConfig],
        pipeline_id: Optional[str] = None
    ) -> str:
        """
        Create a multi-source pipeline with DeepStream integration.
        
        Args:
            sources: List of video sources to include
            pipeline_id: Optional custom pipeline ID
            
        Returns:
            Pipeline ID for management
            
        Raises:
            PipelineError: If pipeline creation fails
        """
        if not sources:
            raise PipelineError("Cannot create pipeline with no sources")
        
        pipeline_id = pipeline_id or f"pipeline-{uuid.uuid4().hex[:8]}"
        
        with self._lock:
            if pipeline_id in self._pipelines:
                raise PipelineError(f"Pipeline {pipeline_id} already exists")
        
        self.logger.info(f"Creating multi-source pipeline {pipeline_id} with {len(sources)} sources")
        
        try:
            with performance_context(f"create_pipeline_{pipeline_id}"):
                # Create pipeline
                pipeline = Gst.Pipeline.new(pipeline_id)
                
                # Create and configure stream muxer
                streammux = await self._create_stream_muxer(len(sources))
                pipeline.add(streammux)
                
                # Add sources to pipeline
                source_pads = []
                for i, source in enumerate(sources):
                    try:
                        source_bin, src_pad = await self._create_source_bin(source, i)
                        pipeline.add(source_bin)
                        
                        # Connect to muxer
                        sink_pad = streammux.get_request_pad(f"sink_{i}")
                        if src_pad.link(sink_pad) != Gst.PadLinkReturn.OK:
                            raise PipelineError(f"Failed to link source {source.id} to muxer")
                        
                        # Track pad connection
                        pad_info = SourcePadInfo(
                            source_id=source.id,
                            pad=src_pad,
                            sink_pad=sink_pad,
                            connected=True
                        )
                        source_pads.append(pad_info)
                        self._source_pads[source.id] = pad_info
                        self._active_sources.add(source.id)
                        
                        self.logger.debug(f"Connected source {source.id} to muxer pad {i}")
                    
                    except Exception as e:
                        self.logger.error(f"Failed to add source {source.id}: {e}")
                        # Continue with other sources if possible
                        continue
                
                # Add detection and output elements
                await self._add_detection_elements(pipeline, streammux)
                await self._add_output_elements(pipeline)
                
                # Set up bus message handling
                bus_watch_id = await self._setup_bus_handling(pipeline, pipeline_id)
                
                # Create pipeline info
                pipeline_info = PipelineInfo(
                    pipeline_id=pipeline_id,
                    pipeline=pipeline,
                    sources=sources,
                    bus_watch_id=bus_watch_id
                )
                
                # Store pipeline
                with self._lock:
                    self._pipelines[pipeline_id] = pipeline_info
                    self._fps_tracking[pipeline_id] = []
                    self._performance_metrics[pipeline_id] = {}
                
                log_pipeline_event(
                    self.logger,
                    "pipeline_created",
                    pipeline_id=pipeline_id,
                    source_count=len(sources),
                    sources=[s.id for s in sources]
                )
                
                return pipeline_id
        
        except Exception as e:
            error = handle_error(e, context={'pipeline_id': pipeline_id, 'sources': [s.id for s in sources]})
            self.logger.error(f"Failed to create pipeline {pipeline_id}: {error}")
            
            # Clean up partial pipeline
            await self._cleanup_failed_pipeline(pipeline_id)
            raise error
    
    async def _create_stream_muxer(self, num_sources: int) -> Gst.Element:
        """Create and configure nvstreammux element."""
        try:
            streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
            if not streammux:
                raise PipelineError("Failed to create nvstreammux element")
            
            # Configure muxer properties
            config = self.config.pipeline
            streammux.set_property('width', config.width)
            streammux.set_property('height', config.height)
            streammux.set_property('batch-size', min(num_sources, config.batch_size))
            streammux.set_property('batched-push-timeout', config.batch_timeout_us)
            streammux.set_property('gpu-id', config.gpu_id)
            
            # Set memory type based on configuration
            if hasattr(streammux.props, 'nvbuf-memory-type'):
                memory_type_mapping = {
                    'device': 0,    # NVBUF_MEM_CUDA_DEVICE
                    'unified': 1,   # NVBUF_MEM_CUDA_UNIFIED
                    'pinned': 2     # NVBUF_MEM_CUDA_PINNED
                }
                memory_type = memory_type_mapping.get(config.memory_type, 0)
                streammux.set_property('nvbuf-memory-type', memory_type)
            
            self.logger.debug(f"Created nvstreammux with batch-size {min(num_sources, config.batch_size)}")
            return streammux
        
        except Exception as e:
            raise PipelineError(f"Failed to create stream muxer: {e}", original_exception=e)
    
    async def _create_source_bin(self, source: SourceConfig, index: int) -> Tuple[Gst.Bin, Gst.Pad]:
        """Create a source bin for the given source configuration."""
        bin_name = f"source-bin-{index}"
        source_bin = Gst.Bin.new(bin_name)
        
        try:
            # Create source element based on type
            if source.type.value == "file":
                src_element = Gst.ElementFactory.make("filesrc", f"file-src-{index}")
                if not src_element:
                    raise SourceError(f"Failed to create filesrc for {source.id}")
                
                src_element.set_property("location", source.uri.replace("file://", ""))
                
                # Add demuxer for video files
                demux = Gst.ElementFactory.make("qtdemux", f"demux-{index}")
                if not demux:
                    demux = Gst.ElementFactory.make("matroskademux", f"demux-{index}")
                
                if demux:
                    source_bin.add(src_element, demux)
                    src_element.link(demux)
                    
                    # Connect demuxer pad-added signal
                    demux.connect("pad-added", self._on_demux_pad_added, source_bin, index)
                else:
                    source_bin.add(src_element)
            
            elif source.type.value == "rtsp":
                src_element = Gst.ElementFactory.make("rtspsrc", f"rtsp-src-{index}")
                if not src_element:
                    raise SourceError(f"Failed to create rtspsrc for {source.id}")
                
                src_element.set_property("location", source.uri)
                src_element.set_property("latency", source.parameters.get("latency", 200))
                
                # Set timeout if specified
                if source.timeout:
                    src_element.set_property("timeout", source.timeout * 1000000)  # microseconds
                
                source_bin.add(src_element)
                
                # Connect pad-added signal for RTSP
                src_element.connect("pad-added", self._on_rtsp_pad_added, source_bin, index)
            
            elif source.type.value == "webcam":
                src_element = Gst.ElementFactory.make("v4l2src", f"v4l2-src-{index}")
                if not src_element:
                    raise SourceError(f"Failed to create v4l2src for {source.id}")
                
                # Set device (URI should be device ID or path)
                if source.uri.startswith("/dev/"):
                    src_element.set_property("device", source.uri)
                else:
                    src_element.set_property("device", f"/dev/video{source.uri}")
                
                source_bin.add(src_element)
            
            elif source.type.value == "test":
                src_element = Gst.ElementFactory.make("videotestsrc", f"test-src-{index}")
                if not src_element:
                    raise SourceError(f"Failed to create videotestsrc for {source.id}")
                
                # Configure test pattern
                pattern = source.parameters.get("pattern", "smpte")
                pattern_map = {
                    "smpte": 0, "snow": 1, "black": 2, "white": 3,
                    "red": 4, "green": 5, "blue": 6, "checkers-1": 7,
                    "checkers-2": 8, "checkers-4": 9, "checkers-8": 10,
                    "circular": 11, "blink": 12, "smpte75": 13
                }
                src_element.set_property("pattern", pattern_map.get(pattern, 0))
                
                # Set framerate
                framerate = source.parameters.get("framerate", "30/1")
                caps_filter = Gst.ElementFactory.make("capsfilter", f"caps-{index}")
                caps_filter.set_property("caps", 
                    Gst.Caps.from_string(f"video/x-raw,framerate={framerate}"))
                
                source_bin.add(src_element, caps_filter)
                src_element.link(caps_filter)
            
            else:
                raise SourceError(f"Unsupported source type: {source.type.value}")
            
            # Add common elements for format conversion
            await self._add_conversion_elements(source_bin, index)
            
            # Create ghost pad
            ghost_pad = await self._create_ghost_pad(source_bin, index)
            
            self.logger.debug(f"Created source bin for {source.id} (type: {source.type.value})")
            return source_bin, ghost_pad
        
        except Exception as e:
            raise SourceError(
                f"Failed to create source bin for {source.id}: {e}",
                source_uri=source.uri,
                source_type=source.type.value,
                original_exception=e
            )
    
    async def _add_conversion_elements(self, source_bin: Gst.Bin, index: int):
        """Add format conversion elements to source bin."""
        try:
            # Video convert for format compatibility
            videoconvert = Gst.ElementFactory.make("videoconvert", f"convert-{index}")
            if not videoconvert:
                raise PipelineError(f"Failed to create videoconvert for source {index}")
            
            # Video scale for resolution compatibility
            videoscale = Gst.ElementFactory.make("videoscale", f"scale-{index}")
            if not videoscale:
                raise PipelineError(f"Failed to create videoscale for source {index}")
            
            # Caps filter for format specification
            caps_filter = Gst.ElementFactory.make("capsfilter", f"caps-filter-{index}")
            if not caps_filter:
                raise PipelineError(f"Failed to create capsfilter for source {index}")
            
            # Set caps for consistency
            caps_string = f"video/x-raw(memory:NVMM),format=NV12,width={self.config.pipeline.width},height={self.config.pipeline.height}"
            caps_filter.set_property("caps", Gst.Caps.from_string(caps_string))
            
            source_bin.add(videoconvert, videoscale, caps_filter)
            
            # Link conversion elements
            videoconvert.link(videoscale)
            videoscale.link(caps_filter)
            
            self.logger.debug(f"Added conversion elements for source {index}")
        
        except Exception as e:
            raise PipelineError(f"Failed to add conversion elements: {e}", original_exception=e)
    
    async def _create_ghost_pad(self, source_bin: Gst.Bin, index: int) -> Gst.Pad:
        """Create ghost pad for source bin output."""
        try:
            # Find the last element in the bin
            caps_filter = source_bin.get_by_name(f"caps-filter-{index}")
            if not caps_filter:
                raise PipelineError(f"Could not find caps filter for ghost pad creation")
            
            src_pad = caps_filter.get_static_pad("src")
            if not src_pad:
                raise PipelineError(f"Could not get src pad from caps filter")
            
            ghost_pad = Gst.GhostPad.new("src", src_pad)
            if not ghost_pad:
                raise PipelineError(f"Failed to create ghost pad for source {index}")
            
            source_bin.add_pad(ghost_pad)
            
            self.logger.debug(f"Created ghost pad for source {index}")
            return ghost_pad
        
        except Exception as e:
            raise PipelineError(f"Failed to create ghost pad: {e}", original_exception=e)
    
    def _on_demux_pad_added(self, demux: Gst.Element, pad: Gst.Pad, source_bin: Gst.Bin, index: int):
        """Handle demuxer pad-added signal."""
        try:
            caps = pad.get_current_caps()
            if not caps:
                return
            
            structure = caps.get_structure(0)
            if structure and structure.get_name().startswith("video"):
                # This is a video pad - connect to conversion elements
                videoconvert = source_bin.get_by_name(f"convert-{index}")
                if videoconvert:
                    sink_pad = videoconvert.get_static_pad("sink")
                    if sink_pad and not sink_pad.is_linked():
                        pad.link(sink_pad)
                        self.logger.debug(f"Linked demux video pad for source {index}")
        
        except Exception as e:
            self.logger.error(f"Error in demux pad-added callback: {e}")
    
    def _on_rtsp_pad_added(self, rtspsrc: Gst.Element, pad: Gst.Pad, source_bin: Gst.Bin, index: int):
        """Handle RTSP source pad-added signal."""
        try:
            caps = pad.get_current_caps()
            if not caps:
                return
            
            structure = caps.get_structure(0)
            if structure and structure.get_name().startswith("application/x-rtp"):
                # Create depayloader
                depay = Gst.ElementFactory.make("rtph264depay", f"depay-{index}")
                if not depay:
                    depay = Gst.ElementFactory.make("rtph265depay", f"depay-{index}")
                
                if depay:
                    source_bin.add(depay)
                    depay.sync_state_with_parent()
                    
                    # Link RTSP pad to depayloader
                    sink_pad = depay.get_static_pad("sink")
                    if sink_pad:
                        pad.link(sink_pad)
                        
                        # Link depayloader to conversion elements
                        videoconvert = source_bin.get_by_name(f"convert-{index}")
                        if videoconvert:
                            depay.link(videoconvert)
                            self.logger.debug(f"Linked RTSP pad for source {index}")
        
        except Exception as e:
            self.logger.error(f"Error in RTSP pad-added callback: {e}")
    
    async def _add_detection_elements(self, pipeline: Gst.Pipeline, streammux: Gst.Element):
        """Add DeepStream detection elements to pipeline."""
        try:
            # Create nvinfer element for detection
            nvinfer = Gst.ElementFactory.make("nvinfer", "primary-inference")
            if not nvinfer:
                raise DeepStreamError("Failed to create nvinfer element")
            
            # Configure nvinfer properties
            detection_config = self.config.detection
            nvinfer.set_property("config-file-path", "config/yolo_config.txt")  # Default config
            nvinfer.set_property("batch-size", self.config.pipeline.batch_size)
            nvinfer.set_property("unique-id", 1)
            nvinfer.set_property("gpu-id", self.config.pipeline.gpu_id)
            
            # Create nvtracker for object tracking if enabled
            tracker = None
            if detection_config.enable_tracking:
                tracker = Gst.ElementFactory.make("nvtracker", "tracker")
                if tracker:
                    tracker.set_property("tracker-width", 640)
                    tracker.set_property("tracker-height", 384)
                    tracker.set_property("gpu-id", self.config.pipeline.gpu_id)
                    tracker.set_property("ll-lib-file", "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so")
                    tracker.set_property("ll-config-file", "config/tracker_config.yml")
            
            # Create nvdsosd for visualization
            nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
            if nvosd:
                nvosd.set_property("process-mode", 0)  # CPU mode
                nvosd.set_property("display-text", True)
                nvosd.set_property("display-bbox", True)
                nvosd.set_property("display-clock", True)
                nvosd.set_property("gpu-id", self.config.pipeline.gpu_id)
            
            # Add elements to pipeline
            pipeline.add(nvinfer)
            if tracker:
                pipeline.add(tracker)
            if nvosd:
                pipeline.add(nvosd)
            
            # Link elements
            streammux.link(nvinfer)
            
            last_element = nvinfer
            if tracker:
                nvinfer.link(tracker)
                last_element = tracker
            
            if nvosd:
                last_element.link(nvosd)
                last_element = nvosd
            
            # Add probe for detection processing
            if last_element:
                sink_pad = last_element.get_static_pad("sink")
                if sink_pad:
                    probe_id = sink_pad.add_probe(
                        Gst.PadProbeType.BUFFER,
                        self._detection_probe_callback
                    )
                    self.logger.debug(f"Added detection probe with ID {probe_id}")
            
            self.logger.info("Added detection elements to pipeline")
        
        except Exception as e:
            raise DeepStreamError(f"Failed to add detection elements: {e}", original_exception=e)
    
    async def _add_output_elements(self, pipeline: Gst.Pipeline):
        """Add output elements to pipeline."""
        try:
            # Create nvegltransform for display preparation
            transform = Gst.ElementFactory.make("nvegltransform", "transform")
            if not transform:
                self.logger.warning("nvegltransform not available, using identity")
                transform = Gst.ElementFactory.make("identity", "identity")
            
            # Create sink element
            sink = Gst.ElementFactory.make("nveglglessink", "display-sink")
            if not sink:
                # Fallback to fakesink for headless operation
                sink = Gst.ElementFactory.make("fakesink", "fake-sink")
                sink.set_property("sync", False)
                sink.set_property("async", False)
                self.logger.info("Using fakesink for headless operation")
            else:
                sink.set_property("sync", False)
                sink.set_property("async", False)
            
            # Add and link output elements
            pipeline.add(transform, sink)
            transform.link(sink)
            
            # Find the last detection element to link from
            nvosd = pipeline.get_by_name("onscreendisplay")
            tracker = pipeline.get_by_name("tracker")
            nvinfer = pipeline.get_by_name("primary-inference")
            
            last_element = nvosd or tracker or nvinfer
            if last_element:
                last_element.link(transform)
            
            self.logger.info("Added output elements to pipeline")
        
        except Exception as e:
            self.logger.error(f"Failed to add output elements: {e}")
            # Don't raise - output elements are not critical for detection
    
    def _detection_probe_callback(self, pad: Gst.Pad, info: Gst.PadProbeInfo) -> Gst.PadProbeReturn:
        """Probe callback for processing detection metadata."""
        try:
            # Get buffer from probe info
            gst_buffer = info.get_buffer()
            if not gst_buffer:
                return Gst.PadProbeReturn.OK
            
            # Extract DeepStream metadata
            batch_meta = self._deepstream_api.get_batch_meta(gst_buffer)
            if not batch_meta:
                return Gst.PadProbeReturn.OK
            
            # Process detection metadata asynchronously
            asyncio.create_task(self._process_detection_metadata(batch_meta))
            
            return Gst.PadProbeReturn.OK
        
        except Exception as e:
            self.logger.error(f"Error in detection probe callback: {e}")
            return Gst.PadProbeReturn.OK
    
    async def _process_detection_metadata(self, batch_meta):
        """Process DeepStream detection metadata."""
        try:
            detections = []
            processing_start = time.perf_counter()
            
            # Extract detections from batch metadata
            # This is a simplified version - full implementation would iterate through
            # frame metadata and object metadata using DeepStream API
            
            # Create detection result
            result = DetectionResult(
                detections=detections,
                processing_time_ms=(time.perf_counter() - processing_start) * 1000
            )
            
            # Notify detection callbacks
            for callback in self._detection_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(result)
                    else:
                        callback(result)
                except Exception as e:
                    self.logger.error(f"Error in detection callback: {e}")
        
        except Exception as e:
            self.logger.error(f"Error processing detection metadata: {e}")
    
    async def _setup_bus_handling(self, pipeline: Gst.Pipeline, pipeline_id: str) -> int:
        """Set up GStreamer bus message handling."""
        try:
            bus = pipeline.get_bus()
            bus.add_signal_watch()
            
            # Connect message signal
            bus.connect("message", self._on_bus_message, pipeline_id)
            
            self.logger.debug(f"Set up bus handling for pipeline {pipeline_id}")
            return 1  # Placeholder watch ID
        
        except Exception as e:
            raise PipelineError(f"Failed to set up bus handling: {e}", original_exception=e)
    
    def _on_bus_message(self, bus: Gst.Bus, message: Gst.Message, pipeline_id: str) -> bool:
        """Handle GStreamer bus messages."""
        try:
            message_type = message.type
            source_name = message.src.get_name() if message.src else "unknown"
            
            if message_type == Gst.MessageType.ERROR:
                err, debug = message.parse_error()
                self.logger.error(f"Pipeline {pipeline_id} error from {source_name}: {err}")
                
                # Queue error for async handling
                self._event_queue.put_nowait({
                    'type': 'error',
                    'pipeline_id': pipeline_id,
                    'source': source_name,
                    'error': str(err),
                    'debug': debug
                })
            
            elif message_type == Gst.MessageType.WARNING:
                warn, debug = message.parse_warning()
                self.logger.warning(f"Pipeline {pipeline_id} warning from {source_name}: {warn}")
            
            elif message_type == Gst.MessageType.STATE_CHANGED:
                old_state, new_state, pending = message.parse_state_changed()
                if message.src == self._pipelines.get(pipeline_id, {}).get('pipeline'):
                    self._update_pipeline_state(pipeline_id, new_state)
            
            elif message_type == Gst.MessageType.EOS:
                self.logger.info(f"End of stream from {source_name} in pipeline {pipeline_id}")
                
                # Queue EOS for async handling
                self._event_queue.put_nowait({
                    'type': 'eos',
                    'pipeline_id': pipeline_id,
                    'source': source_name
                })
            
            elif message_type == Gst.MessageType.BUFFERING:
                percent = message.parse_buffering()
                if percent < 100:
                    self.logger.debug(f"Buffering {percent}% in pipeline {pipeline_id}")
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error handling bus message: {e}")
            return True
    
    def _update_pipeline_state(self, pipeline_id: str, gst_state: Gst.State):
        """Update pipeline state tracking."""
        try:
            with self._lock:
                if pipeline_id not in self._pipelines:
                    return
                
                pipeline_info = self._pipelines[pipeline_id]
                
                # Map GStreamer state to our state enum
                state_mapping = {
                    Gst.State.NULL: PipelineState.NULL,
                    Gst.State.READY: PipelineState.READY,
                    Gst.State.PAUSED: PipelineState.PAUSED,
                    Gst.State.PLAYING: PipelineState.PLAYING
                }
                
                new_state = state_mapping.get(gst_state, PipelineState.ERROR)
                old_state = pipeline_info.state
                pipeline_info.state = new_state
                pipeline_info.last_activity = time.time()
                
                if old_state != new_state:
                    log_pipeline_event(
                        self.logger,
                        "state_changed",
                        pipeline_id=pipeline_id,
                        old_state=old_state.value,
                        new_state=new_state.value
                    )
                    
                    # Notify state change callbacks
                    for callback in self._state_change_callbacks:
                        try:
                            callback(pipeline_id, old_state, new_state)
                        except Exception as e:
                            self.logger.error(f"Error in state change callback: {e}")
        
        except Exception as e:
            self.logger.error(f"Error updating pipeline state: {e}")
    
    async def start_pipeline(self, pipeline_id: str, timeout: float = 30.0) -> bool:
        """
        Start a pipeline with timeout and error handling.
        
        Args:
            pipeline_id: ID of pipeline to start
            timeout: Maximum time to wait for pipeline to start
            
        Returns:
            True if pipeline started successfully
        """
        with self._lock:
            if pipeline_id not in self._pipelines:
                raise PipelineError(f"Pipeline {pipeline_id} not found")
            
            pipeline_info = self._pipelines[pipeline_id]
        
        self.logger.info(f"Starting pipeline {pipeline_id}")
        
        try:
            with performance_context(f"start_pipeline_{pipeline_id}"):
                # Set pipeline to PLAYING state
                ret = pipeline_info.pipeline.set_state(Gst.State.PLAYING)
                
                if ret == Gst.StateChangeReturn.FAILURE:
                    raise PipelineError(f"Failed to start pipeline {pipeline_id}")
                
                elif ret == Gst.StateChangeReturn.ASYNC:
                    # Wait for state change to complete
                    start_time = time.time()
                    while time.time() - start_time < timeout:
                        ret, current_state, pending_state = pipeline_info.pipeline.get_state(Gst.SECOND)
                        
                        if ret == Gst.StateChangeReturn.SUCCESS and current_state == Gst.State.PLAYING:
                            break
                        elif ret == Gst.StateChangeReturn.FAILURE:
                            raise PipelineError(f"Pipeline {pipeline_id} failed to start")
                        
                        await asyncio.sleep(0.1)
                    else:
                        raise PipelineError(f"Pipeline {pipeline_id} start timeout after {timeout}s")
                
                # Update pipeline state
                self._update_pipeline_state(pipeline_id, Gst.State.PLAYING)
                
                log_pipeline_event(
                    self.logger,
                    "pipeline_started",
                    pipeline_id=pipeline_id,
                    source_count=len(pipeline_info.sources)
                )
                
                return True
        
        except Exception as e:
            error = handle_error(e, context={'pipeline_id': pipeline_id})
            self.logger.error(f"Failed to start pipeline {pipeline_id}: {error}")
            
            # Try to recover pipeline state
            await self._handle_pipeline_error(pipeline_id, error)
            raise error
    
    async def stop_pipeline(self, pipeline_id: str, timeout: float = 10.0) -> bool:
        """
        Stop a pipeline gracefully.
        
        Args:
            pipeline_id: ID of pipeline to stop
            timeout: Maximum time to wait for pipeline to stop
            
        Returns:
            True if pipeline stopped successfully
        """
        with self._lock:
            if pipeline_id not in self._pipelines:
                self.logger.warning(f"Pipeline {pipeline_id} not found for stopping")
                return True
            
            pipeline_info = self._pipelines[pipeline_id]
        
        self.logger.info(f"Stopping pipeline {pipeline_id}")
        
        try:
            # Set pipeline to NULL state
            ret = pipeline_info.pipeline.set_state(Gst.State.NULL)
            
            if ret == Gst.StateChangeReturn.ASYNC:
                # Wait for state change to complete
                start_time = time.time()
                while time.time() - start_time < timeout:
                    ret, current_state, pending_state = pipeline_info.pipeline.get_state(Gst.SECOND)
                    
                    if ret == Gst.StateChangeReturn.SUCCESS and current_state == Gst.State.NULL:
                        break
                    elif ret == Gst.StateChangeReturn.FAILURE:
                        self.logger.warning(f"Pipeline {pipeline_id} failed to stop cleanly")
                        break
                    
                    await asyncio.sleep(0.1)
            
            # Update pipeline state
            self._update_pipeline_state(pipeline_id, Gst.State.NULL)
            
            log_pipeline_event(
                self.logger,
                "pipeline_stopped",
                pipeline_id=pipeline_id
            )
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error stopping pipeline {pipeline_id}: {e}")
            return False
    
    async def remove_pipeline(self, pipeline_id: str) -> bool:
        """
        Remove a pipeline completely.
        
        Args:
            pipeline_id: ID of pipeline to remove
            
        Returns:
            True if pipeline removed successfully
        """
        self.logger.info(f"Removing pipeline {pipeline_id}")
        
        try:
            # Stop pipeline first
            await self.stop_pipeline(pipeline_id)
            
            with self._lock:
                if pipeline_id not in self._pipelines:
                    return True
                
                pipeline_info = self._pipelines[pipeline_id]
                
                # Clean up bus handling
                if pipeline_info.bus_watch_id:
                    # Remove bus watch (implementation depends on GStreamer version)
                    pass
                
                # Remove source pads
                sources_to_remove = [s.id for s in pipeline_info.sources]
                for source_id in sources_to_remove:
                    self._source_pads.pop(source_id, None)
                    self._active_sources.discard(source_id)
                
                # Remove pipeline
                del self._pipelines[pipeline_id]
                self._fps_tracking.pop(pipeline_id, None)
                self._performance_metrics.pop(pipeline_id, None)
            
            log_pipeline_event(
                self.logger,
                "pipeline_removed",
                pipeline_id=pipeline_id
            )
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error removing pipeline {pipeline_id}: {e}")
            return False
    
    async def _handle_pipeline_error(self, pipeline_id: str, error: Exception):
        """Handle pipeline errors with recovery attempts."""
        try:
            with self._lock:
                if pipeline_id not in self._pipelines:
                    return
                
                pipeline_info = self._pipelines[pipeline_id]
                pipeline_info.error_count += 1
                pipeline_info.state = PipelineState.ERROR
            
            self.logger.error(f"Handling pipeline error for {pipeline_id}: {error}")
            
            # Notify error callbacks
            for callback in self._error_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(pipeline_id, error)
                    else:
                        callback(pipeline_id, error)
                except Exception as e:
                    self.logger.error(f"Error in error callback: {e}")
            
            # Attempt recovery if error is recoverable
            if hasattr(error, 'recoverable') and error.recoverable:
                if recovery_strategy.should_retry(error, pipeline_id):
                    delay = recovery_strategy.get_retry_delay(error, pipeline_id)
                    recovery_strategy.record_retry(error, pipeline_id)
                    
                    self.logger.info(f"Attempting pipeline recovery for {pipeline_id} in {delay:.2f}s")
                    await asyncio.sleep(delay)
                    
                    # Try to restart pipeline
                    try:
                        await self.stop_pipeline(pipeline_id)
                        await self.start_pipeline(pipeline_id)
                        recovery_strategy.reset_retry_count(error, pipeline_id)
                        
                        log_pipeline_event(
                            self.logger,
                            "pipeline_recovered",
                            pipeline_id=pipeline_id,
                            error_count=pipeline_info.error_count
                        )
                    except Exception as recovery_error:
                        self.logger.error(f"Pipeline recovery failed for {pipeline_id}: {recovery_error}")
        
        except Exception as e:
            self.logger.error(f"Error in pipeline error handler: {e}")
    
    async def _cleanup_failed_pipeline(self, pipeline_id: str):
        """Clean up resources from a failed pipeline creation."""
        try:
            with self._lock:
                # Remove from active sources
                for source_id in list(self._active_sources):
                    if source_id in self._source_pads:
                        pad_info = self._source_pads[source_id]
                        # Clean up pad connections
                        self._source_pads.pop(source_id, None)
                        self._active_sources.discard(source_id)
                
                # Remove pipeline if it exists
                if pipeline_id in self._pipelines:
                    del self._pipelines[pipeline_id]
                
                # Clean up tracking data
                self._fps_tracking.pop(pipeline_id, None)
                self._performance_metrics.pop(pipeline_id, None)
            
            self.logger.debug(f"Cleaned up failed pipeline {pipeline_id}")
        
        except Exception as e:
            self.logger.error(f"Error cleaning up failed pipeline: {e}")
    
    # Public API methods for callbacks and management
    
    def add_detection_callback(self, callback: Callable):
        """Add callback for detection events."""
        self._detection_callbacks.append(callback)
    
    def add_state_change_callback(self, callback: Callable):
        """Add callback for pipeline state changes."""
        self._state_change_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable):
        """Add callback for pipeline errors."""
        self._error_callbacks.append(callback)
    
    def get_pipeline_info(self, pipeline_id: str) -> Optional[PipelineInfo]:
        """Get information about a specific pipeline."""
        with self._lock:
            return self._pipelines.get(pipeline_id)
    
    def get_all_pipelines(self) -> Dict[str, PipelineInfo]:
        """Get information about all pipelines."""
        with self._lock:
            return self._pipelines.copy()
    
    def get_active_sources(self) -> Set[str]:
        """Get set of currently active source IDs."""
        with self._lock:
            return self._active_sources.copy()
    
    async def stop(self):
        """Stop all pipelines and clean up resources."""
        self.logger.info("Stopping PipelineManager...")
        
        try:
            # Stop all pipelines
            pipeline_ids = list(self._pipelines.keys())
            for pipeline_id in pipeline_ids:
                await self.remove_pipeline(pipeline_id)
            
            # Clear all callbacks
            self._detection_callbacks.clear()
            self._state_change_callbacks.clear()
            self._error_callbacks.clear()
            
            self.logger.info("PipelineManager stopped successfully")
        
        except Exception as e:
            self.logger.error(f"Error stopping PipelineManager: {e}")


# Global pipeline manager instance
_pipeline_manager: Optional[PipelineManager] = None


def get_pipeline_manager() -> PipelineManager:
    """Get global pipeline manager instance."""
    global _pipeline_manager
    if _pipeline_manager is None:
        _pipeline_manager = PipelineManager()
    return _pipeline_manager