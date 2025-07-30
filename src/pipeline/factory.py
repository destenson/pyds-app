"""
Pipeline factory for creating optimized GStreamer pipelines.

This module provides factory patterns for different source configurations,
template-based pipeline construction, and batch processing optimization.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import gi

# CRITICAL: Initialize GStreamer before imports
gi.require_version('Gst', '1.0')
from gi.repository import Gst

from ..config import AppConfig, SourceConfig, SourceType, PipelineConfig
from ..utils.errors import PipelineError, handle_error
from ..utils.logging import get_logger, performance_context, log_pipeline_event
from ..utils.deepstream import get_deepstream_info
from .manager import PipelineManager, PipelineInfo
from .sources import VideoSourceManager, SourceFactory
from .elements import DeepStreamElementsManager, ElementInfo, ElementType


class PipelineTemplate(Enum):
    """Pre-defined pipeline templates."""
    BASIC_DETECTION = "basic_detection"
    MULTI_SOURCE_DETECTION = "multi_source_detection"
    ANALYTICS_PIPELINE = "analytics_pipeline"
    STREAMING_PIPELINE = "streaming_pipeline"
    RECORDING_PIPELINE = "recording_pipeline"
    CUSTOM = "custom"


class OptimizationProfile(Enum):
    """Pipeline optimization profiles."""
    REAL_TIME = "real_time"           # Low latency, high performance
    HIGH_THROUGHPUT = "high_throughput"  # Maximum fps, higher latency OK
    BALANCED = "balanced"             # Balance between latency and throughput
    LOW_POWER = "low_power"          # Minimize power consumption
    HIGH_QUALITY = "high_quality"    # Maximum quality, performance secondary


@dataclass
class PipelineTemplate:
    """Pipeline template configuration."""
    name: str
    description: str
    supported_source_types: List[SourceType]
    required_elements: List[ElementType]
    optimization_profile: OptimizationProfile
    max_sources: int = 1
    supports_analytics: bool = False
    supports_tracking: bool = False
    supports_recording: bool = False
    template_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineBlueprint:
    """Complete pipeline blueprint with all configuration."""
    template: PipelineTemplate
    sources: List[SourceConfig]
    elements: List[ElementInfo]
    optimization_settings: Dict[str, Any]
    pipeline_config: PipelineConfig
    custom_properties: Dict[str, Any] = field(default_factory=dict)


class PipelineBuilder(ABC):
    """Abstract base class for pipeline builders."""
    
    def __init__(self, config: AppConfig):
        """
        Initialize pipeline builder.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self.elements_manager = DeepStreamElementsManager(config)
        self.source_manager = VideoSourceManager()
        
    @abstractmethod
    async def build_pipeline(self, sources: List[SourceConfig], **kwargs) -> PipelineBlueprint:
        """
        Build pipeline blueprint.
        
        Args:
            sources: List of video sources
            **kwargs: Additional configuration
            
        Returns:
            Complete pipeline blueprint
        """
        pass
    
    @abstractmethod
    def get_template(self) -> PipelineTemplate:
        """Get pipeline template information."""
        pass
    
    def validate_sources(self, sources: List[SourceConfig]) -> bool:
        """
        Validate sources against template requirements.
        
        Args:
            sources: List of sources to validate
            
        Returns:
            True if sources are compatible
        """
        template = self.get_template()
        
        # Check source count
        if len(sources) > template.max_sources:
            raise PipelineError(
                f"Too many sources for template {template.name}: "
                f"{len(sources)} > {template.max_sources}"
            )
        
        # Check source types
        for source in sources:
            if source.type not in template.supported_source_types:
                raise PipelineError(
                    f"Source type {source.type.value} not supported by template {template.name}"
                )
        
        return True
    
    def apply_optimization_profile(self, elements: List[ElementInfo], profile: OptimizationProfile):
        """Apply optimization profile to elements."""
        try:
            if profile == OptimizationProfile.REAL_TIME:
                self._apply_real_time_optimizations(elements)
            elif profile == OptimizationProfile.HIGH_THROUGHPUT:
                self._apply_high_throughput_optimizations(elements)
            elif profile == OptimizationProfile.BALANCED:
                self._apply_balanced_optimizations(elements)
            elif profile == OptimizationProfile.LOW_POWER:
                self._apply_low_power_optimizations(elements)
            elif profile == OptimizationProfile.HIGH_QUALITY:
                self._apply_high_quality_optimizations(elements)
            
            self.logger.info(f"Applied {profile.value} optimization profile to {len(elements)} elements")
        
        except Exception as e:
            self.logger.error(f"Error applying optimization profile: {e}")
    
    def _apply_real_time_optimizations(self, elements: List[ElementInfo]):
        """Apply real-time optimizations."""
        for element_info in elements:
            element = element_info.element
            element_type = element_info.element_type
            
            if element_type == ElementType.STREAMMUX:
                # Minimize batching for low latency
                element.set_property('batch-size', 1)
                element.set_property('batched-push-timeout', 1000)  # 1ms
                element.set_property('live-source', True)
            
            elif element_type == ElementType.INFERENCE:
                # Process every frame for real-time
                element.set_property('interval', 0)
                element.set_property('batch-size', 1)
            
            elif element_type == ElementType.TRACKER:
                # Optimize tracker for speed
                element.set_property('tracker-width', 320)
                element.set_property('tracker-height', 240)
    
    def _apply_high_throughput_optimizations(self, elements: List[ElementInfo]):
        """Apply high throughput optimizations."""
        for element_info in elements:
            element = element_info.element
            element_type = element_info.element_type
            
            if element_type == ElementType.STREAMMUX:
                # Maximize batching
                element.set_property('batch-size', self.config.pipeline.batch_size)
                element.set_property('batched-push-timeout', 40000)  # 40ms
            
            elif element_type == ElementType.INFERENCE:
                # Use full batch processing
                element.set_property('batch-size', self.config.pipeline.batch_size)
                element.set_property('interval', 1)  # Skip frames if needed
    
    def _apply_balanced_optimizations(self, elements: List[ElementInfo]):
        """Apply balanced optimizations."""
        batch_size = min(4, self.config.pipeline.batch_size)
        
        for element_info in elements:
            element = element_info.element
            element_type = element_info.element_type
            
            if element_type == ElementType.STREAMMUX:
                element.set_property('batch-size', batch_size)
                element.set_property('batched-push-timeout', 20000)  # 20ms
            
            elif element_type == ElementType.INFERENCE:
                element.set_property('batch-size', batch_size)
                element.set_property('interval', 0)
    
    def _apply_low_power_optimizations(self, elements: List[ElementInfo]):
        """Apply low power optimizations."""
        for element_info in elements:
            element = element_info.element
            element_type = element_info.element_type
            
            if element_type == ElementType.STREAMMUX:
                # Use smaller batch sizes
                element.set_property('batch-size', 1)
                element.set_property('width', 640)
                element.set_property('height', 480)  # Lower resolution
            
            elif element_type == ElementType.INFERENCE:
                element.set_property('interval', 3)  # Process every 3rd frame
            
            elif element_type == ElementType.TRACKER:
                # Use smaller tracker resolution
                element.set_property('tracker-width', 320)
                element.set_property('tracker-height', 240)
    
    def _apply_high_quality_optimizations(self, elements: List[ElementInfo]):
        """Apply high quality optimizations."""
        for element_info in elements:
            element = element_info.element
            element_type = element_info.element_type
            
            if element_type == ElementType.STREAMMUX:
                # Use full resolution
                element.set_property('width', self.config.pipeline.width)
                element.set_property('height', self.config.pipeline.height)
                element.set_property('interpolation-method', 1)  # Bilinear
            
            elif element_type == ElementType.INFERENCE:
                # Process every frame at full quality
                element.set_property('interval', 0)
                element.set_property('batch-size', 1)  # Individual processing
            
            elif element_type == ElementType.TRACKER:
                # Use higher tracker resolution
                element.set_property('tracker-width', 960)
                element.set_property('tracker-height', 544)


class BasicDetectionBuilder(PipelineBuilder):
    """Builder for basic detection pipelines."""
    
    def get_template(self) -> PipelineTemplate:
        """Get basic detection template."""
        return PipelineTemplate(
            name="Basic Detection",
            description="Single-source detection with tracking and visualization",
            supported_source_types=[SourceType.FILE, SourceType.WEBCAM, SourceType.RTSP, SourceType.TEST],
            required_elements=[ElementType.STREAMMUX, ElementType.INFERENCE, ElementType.OSD],
            optimization_profile=OptimizationProfile.BALANCED,
            max_sources=1,
            supports_analytics=False,
            supports_tracking=True,
            supports_recording=False
        )
    
    async def build_pipeline(self, sources: List[SourceConfig], **kwargs) -> PipelineBlueprint:
        """Build basic detection pipeline."""
        if len(sources) != 1:
            raise PipelineError("Basic detection template supports exactly 1 source")
        
        self.validate_sources(sources)
        
        try:
            # Create detection elements
            elements = self.elements_manager.create_detection_pipeline_elements("basic_detection")
            
            # Apply optimization profile
            profile = kwargs.get('optimization_profile', OptimizationProfile.BALANCED)
            self.apply_optimization_profile(elements, profile)
            
            # Create blueprint
            blueprint = PipelineBlueprint(
                template=self.get_template(),
                sources=sources,
                elements=elements,
                optimization_settings={'profile': profile.value},
                pipeline_config=self.config.pipeline,
                custom_properties=kwargs
            )
            
            self.logger.info(f"Built basic detection pipeline for source {sources[0].id}")
            return blueprint
        
        except Exception as e:
            raise PipelineError(f"Failed to build basic detection pipeline: {e}", original_exception=e)


class MultiSourceDetectionBuilder(PipelineBuilder):
    """Builder for multi-source detection pipelines."""
    
    def get_template(self) -> PipelineTemplate:
        """Get multi-source detection template."""
        return PipelineTemplate(
            name="Multi-Source Detection",
            description="Multiple source detection with batch processing",
            supported_source_types=[SourceType.FILE, SourceType.WEBCAM, SourceType.RTSP, SourceType.TEST],
            required_elements=[ElementType.STREAMMUX, ElementType.INFERENCE, ElementType.OSD],
            optimization_profile=OptimizationProfile.HIGH_THROUGHPUT,
            max_sources=16,
            supports_analytics=True,
            supports_tracking=True,
            supports_recording=True
        )
    
    async def build_pipeline(self, sources: List[SourceConfig], **kwargs) -> PipelineBlueprint:
        """Build multi-source detection pipeline."""
        if len(sources) == 0:
            raise PipelineError("Multi-source detection requires at least 1 source")
        
        if len(sources) > self.config.max_concurrent_sources:
            raise PipelineError(
                f"Too many sources: {len(sources)} > {self.config.max_concurrent_sources}"
            )
        
        self.validate_sources(sources)
        
        try:
            # Create detection elements optimized for multiple sources
            elements = self.elements_manager.create_detection_pipeline_elements("multi_source_detection")
            
            # Configure stream muxer for multiple sources
            streammux = next((e for e in elements if e.element_type == ElementType.STREAMMUX), None)
            if streammux:
                streammux.element.set_property('batch-size', min(len(sources), self.config.pipeline.batch_size))
                streammux.element.set_property('live-source', True)
                streammux.element.set_property('batched-push-timeout', 40000)  # 40ms for batching
            
            # Apply optimization profile
            profile = kwargs.get('optimization_profile', OptimizationProfile.HIGH_THROUGHPUT)
            self.apply_optimization_profile(elements, profile)
            
            # Add analytics elements if requested
            if kwargs.get('enable_analytics', False):
                analytics_elements = self.elements_manager.create_analytics_pipeline_elements("analytics")
                elements.extend(analytics_elements)
            
            # Create blueprint
            blueprint = PipelineBlueprint(
                template=self.get_template(),
                sources=sources,
                elements=elements,
                optimization_settings={
                    'profile': profile.value,
                    'batch_optimized': True,
                    'source_count': len(sources)
                },
                pipeline_config=self.config.pipeline,
                custom_properties=kwargs
            )
            
            self.logger.info(f"Built multi-source detection pipeline for {len(sources)} sources")
            return blueprint
        
        except Exception as e:
            raise PipelineError(f"Failed to build multi-source detection pipeline: {e}", original_exception=e)


class AnalyticsPipelineBuilder(PipelineBuilder):
    """Builder for analytics-focused pipelines."""
    
    def get_template(self) -> PipelineTemplate:
        """Get analytics pipeline template."""
        return PipelineTemplate(
            name="Analytics Pipeline",
            description="Advanced analytics with message generation",
            supported_source_types=[SourceType.FILE, SourceType.WEBCAM, SourceType.RTSP],
            required_elements=[
                ElementType.STREAMMUX, ElementType.INFERENCE, ElementType.TRACKER,
                ElementType.ANALYTICS, ElementType.MSGCONV, ElementType.MSGBROKER
            ],
            optimization_profile=OptimizationProfile.BALANCED,
            max_sources=8,
            supports_analytics=True,
            supports_tracking=True,
            supports_recording=True
        )
    
    async def build_pipeline(self, sources: List[SourceConfig], **kwargs) -> PipelineBlueprint:
        """Build analytics pipeline."""
        self.validate_sources(sources)
        
        try:
            # Create detection elements
            detection_elements = self.elements_manager.create_detection_pipeline_elements("detection")
            
            # Create analytics elements
            analytics_elements = self.elements_manager.create_analytics_pipeline_elements("analytics")
            
            # Combine all elements
            elements = detection_elements + analytics_elements
            
            # Apply optimization profile
            profile = kwargs.get('optimization_profile', OptimizationProfile.BALANCED)
            self.apply_optimization_profile(elements, profile)
            
            # Configure analytics-specific settings
            analytics_element = next((e for e in elements if e.element_type == ElementType.ANALYTICS), None)
            if analytics_element:
                # Configure ROI and analytics zones
                analytics_config = kwargs.get('analytics_config', {})
                if analytics_config:
                    analytics_element.element.set_property('config-file', analytics_config.get('config_file', ''))
            
            # Create blueprint
            blueprint = PipelineBlueprint(
                template=self.get_template(),
                sources=sources,
                elements=elements,
                optimization_settings={
                    'profile': profile.value,
                    'analytics_enabled': True,
                    'message_generation': True
                },
                pipeline_config=self.config.pipeline,
                custom_properties=kwargs
            )
            
            self.logger.info(f"Built analytics pipeline for {len(sources)} sources")
            return blueprint
        
        except Exception as e:
            raise PipelineError(f"Failed to build analytics pipeline: {e}", original_exception=e)


class StreamingPipelineBuilder(PipelineBuilder):
    """Builder for streaming/RTMP output pipelines."""
    
    def get_template(self) -> PipelineTemplate:
        """Get streaming pipeline template."""
        return PipelineTemplate(
            name="Streaming Pipeline",
            description="Detection with RTMP/WebRTC streaming output",
            supported_source_types=[SourceType.FILE, SourceType.WEBCAM, SourceType.RTSP, SourceType.TEST],
            required_elements=[
                ElementType.STREAMMUX, ElementType.INFERENCE, ElementType.OSD,
                ElementType.ENCODER, ElementType.CONVERTER
            ],
            optimization_profile=OptimizationProfile.REAL_TIME,
            max_sources=4,
            supports_analytics=False,
            supports_tracking=True,
            supports_recording=False
        )
    
    async def build_pipeline(self, sources: List[SourceConfig], **kwargs) -> PipelineBlueprint:
        """Build streaming pipeline."""
        self.validate_sources(sources)
        
        try:
            # Create detection elements
            elements = self.elements_manager.create_detection_pipeline_elements("streaming")
            
            # Add streaming-specific elements
            encoder = self.elements_manager._factory.create_element(
                ElementType.ENCODER,
                "h264encoder",
                bitrate=kwargs.get('bitrate', 2000000),
                profile=0,  # Baseline profile for compatibility
                **{'insert-sps-pps': True}
            )
            elements.append(encoder)
            
            # Apply real-time optimization profile
            profile = OptimizationProfile.REAL_TIME
            self.apply_optimization_profile(elements, profile)
            
            # Create blueprint
            blueprint = PipelineBlueprint(
                template=self.get_template(),
                sources=sources,
                elements=elements,
                optimization_settings={
                    'profile': profile.value,
                    'streaming_optimized': True,
                    'low_latency': True
                },
                pipeline_config=self.config.pipeline,
                custom_properties=kwargs
            )
            
            self.logger.info(f"Built streaming pipeline for {len(sources)} sources")
            return blueprint
        
        except Exception as e:
            raise PipelineError(f"Failed to build streaming pipeline: {e}", original_exception=e)


class PipelineFactory:
    """
    Factory for creating optimized GStreamer pipelines.
    
    Provides template-based pipeline construction with optimization profiles
    and supports custom pipeline modifications.
    """
    
    def __init__(self, config: AppConfig):
        """
        Initialize pipeline factory.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.pipeline_manager = PipelineManager(config)
        
        # Register pipeline builders
        self._builders: Dict[PipelineTemplate, PipelineBuilder] = {
            PipelineTemplate.BASIC_DETECTION: BasicDetectionBuilder(config),
            PipelineTemplate.MULTI_SOURCE_DETECTION: MultiSourceDetectionBuilder(config),
            PipelineTemplate.ANALYTICS_PIPELINE: AnalyticsPipelineBuilder(config),
            PipelineTemplate.STREAMING_PIPELINE: StreamingPipelineBuilder(config)
        }
        
        # Template registry
        self._templates: Dict[str, PipelineTemplate] = {}
        self._initialize_templates()
        
        self.logger.info("PipelineFactory initialized with {} templates".format(len(self._builders)))
    
    def _initialize_templates(self):
        """Initialize pipeline template registry."""
        for template_enum, builder in self._builders.items():
            template = builder.get_template()
            self._templates[template.name] = template_enum
    
    def get_available_templates(self) -> List[str]:
        """Get list of available pipeline templates."""
        return list(self._templates.keys())
    
    def get_template_info(self, template_name: str) -> Optional[PipelineTemplate]:
        """Get information about a pipeline template."""
        template_enum = self._templates.get(template_name)
        if template_enum and template_enum in self._builders:
            return self._builders[template_enum].get_template()
        return None
    
    def recommend_template(self, sources: List[SourceConfig], requirements: Dict[str, Any]) -> str:
        """
        Recommend optimal pipeline template based on sources and requirements.
        
        Args:
            sources: List of video sources
            requirements: Requirements dictionary
            
        Returns:
            Recommended template name
        """
        try:
            # Analyze source characteristics
            source_count = len(sources)
            source_types = set(source.type for source in sources)
            
            # Analyze requirements
            needs_analytics = requirements.get('analytics', False)
            needs_streaming = requirements.get('streaming', False)
            needs_recording = requirements.get('recording', False)
            performance_priority = requirements.get('performance', 'balanced')
            
            # Decision logic
            if needs_streaming and source_count <= 4:
                return "Streaming Pipeline"
            elif needs_analytics:
                return "Analytics Pipeline"
            elif source_count == 1:
                return "Basic Detection"
            elif source_count > 1:
                return "Multi-Source Detection"
            else:
                return "Basic Detection"  # Fallback
        
        except Exception as e:
            self.logger.warning(f"Error recommending template: {e}")
            return "Basic Detection"  # Safe fallback
    
    async def create_pipeline_from_template(
        self,
        template_name: str,
        sources: List[SourceConfig],
        pipeline_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Create pipeline from template.
        
        Args:
            template_name: Name of pipeline template
            sources: List of video sources
            pipeline_id: Optional custom pipeline ID
            **kwargs: Additional configuration
            
        Returns:
            Pipeline ID
            
        Raises:
            PipelineError: If pipeline creation fails
        """
        template_enum = self._templates.get(template_name)
        if not template_enum or template_enum not in self._builders:
            raise PipelineError(f"Unknown pipeline template: {template_name}")
        
        builder = self._builders[template_enum]
        
        try:
            with performance_context(f"create_pipeline_{template_name}"):
                # Build pipeline blueprint
                blueprint = await builder.build_pipeline(sources, **kwargs)
                
                # Create pipeline using blueprint
                return await self._create_pipeline_from_blueprint(blueprint, pipeline_id)
        
        except Exception as e:
            raise PipelineError(f"Failed to create pipeline from template {template_name}: {e}", original_exception=e)
    
    async def _create_pipeline_from_blueprint(
        self,
        blueprint: PipelineBlueprint,
        pipeline_id: Optional[str] = None
    ) -> str:
        """Create pipeline from blueprint."""
        try:
            # Create pipeline using manager
            created_pipeline_id = await self.pipeline_manager.create_multi_source_pipeline(
                blueprint.sources,
                pipeline_id
            )
            
            # Log pipeline creation
            log_pipeline_event(
                self.logger,
                "pipeline_created_from_template",
                pipeline_id=created_pipeline_id,
                template=blueprint.template.name,
                source_count=len(blueprint.sources),
                element_count=len(blueprint.elements),
                optimization_profile=blueprint.optimization_settings.get('profile', 'unknown')
            )
            
            return created_pipeline_id
        
        except Exception as e:
            raise PipelineError(f"Failed to create pipeline from blueprint: {e}", original_exception=e)
    
    async def create_optimized_pipeline(
        self,
        sources: List[SourceConfig],
        optimization_profile: OptimizationProfile = OptimizationProfile.BALANCED,
        pipeline_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Create optimized pipeline with automatic template selection.
        
        Args:
            sources: List of video sources
            optimization_profile: Optimization profile to apply
            pipeline_id: Optional custom pipeline ID
            **kwargs: Additional requirements
            
        Returns:
            Pipeline ID
        """
        try:
            # Recommend template based on sources and requirements
            requirements = {
                'performance': optimization_profile.value,
                **kwargs
            }
            template_name = self.recommend_template(sources, requirements)
            
            self.logger.info(f"Recommended template {template_name} for {len(sources)} sources")
            
            # Create pipeline from recommended template
            return await self.create_pipeline_from_template(
                template_name,
                sources,
                pipeline_id,
                optimization_profile=optimization_profile,
                **kwargs
            )
        
        except Exception as e:
            raise PipelineError(f"Failed to create optimized pipeline: {e}", original_exception=e)
    
    def register_custom_builder(self, template_name: str, builder: PipelineBuilder):
        """Register custom pipeline builder."""
        try:
            template_enum = PipelineTemplate.CUSTOM
            self._builders[template_enum] = builder
            self._templates[template_name] = template_enum
            
            self.logger.info(f"Registered custom pipeline builder: {template_name}")
        
        except Exception as e:
            self.logger.error(f"Failed to register custom builder: {e}")
    
    def get_pipeline_manager(self) -> PipelineManager:
        """Get underlying pipeline manager."""
        return self.pipeline_manager
    
    async def clone_pipeline(
        self,
        source_pipeline_id: str,
        new_sources: List[SourceConfig],
        new_pipeline_id: Optional[str] = None
    ) -> str:
        """
        Clone existing pipeline with new sources.
        
        Args:
            source_pipeline_id: ID of pipeline to clone
            new_sources: New sources for cloned pipeline
            new_pipeline_id: Optional ID for new pipeline
            
        Returns:
            New pipeline ID
        """
        try:
            # Get source pipeline info
            source_info = self.pipeline_manager.get_pipeline_info(source_pipeline_id)
            if not source_info:
                raise PipelineError(f"Source pipeline {source_pipeline_id} not found")
            
            # For now, create a new multi-source pipeline
            # In a full implementation, this would copy the exact configuration
            return await self.pipeline_manager.create_multi_source_pipeline(
                new_sources,
                new_pipeline_id
            )
        
        except Exception as e:
            raise PipelineError(f"Failed to clone pipeline: {e}", original_exception=e)
    
    def get_optimization_profiles(self) -> List[str]:
        """Get available optimization profiles."""
        return [profile.value for profile in OptimizationProfile]
    
    def validate_template_compatibility(
        self,
        template_name: str,
        sources: List[SourceConfig]
    ) -> Tuple[bool, List[str]]:
        """
        Validate template compatibility with sources.
        
        Args:
            template_name: Template name to validate
            sources: Sources to validate against
            
        Returns:
            Tuple of (is_compatible, issues)
        """
        template_enum = self._templates.get(template_name)
        if not template_enum or template_enum not in self._builders:
            return False, [f"Unknown template: {template_name}"]
        
        builder = self._builders[template_enum]
        template = builder.get_template()
        issues = []
        
        try:
            # Check source count
            if len(sources) > template.max_sources:
                issues.append(f"Too many sources: {len(sources)} > {template.max_sources}")
            
            # Check source types
            for source in sources:
                if source.type not in template.supported_source_types:
                    issues.append(f"Unsupported source type: {source.type.value}")
            
            # Check DeepStream capabilities
            deepstream_info = get_deepstream_info()
            missing_capabilities = []
            
            for element_type in template.required_elements:
                # This is a simplified check - in practice would verify actual plugin availability
                if element_type == ElementType.ANALYTICS and not deepstream_info.capabilities.get('analytics', True):
                    missing_capabilities.append("analytics")
            
            if missing_capabilities:
                issues.append(f"Missing DeepStream capabilities: {missing_capabilities}")
            
            return len(issues) == 0, issues
        
        except Exception as e:
            return False, [f"Validation error: {e}"]
    
    async def stop(self):
        """Stop pipeline factory and clean up resources."""
        try:
            self.logger.info("Stopping PipelineFactory...")
            
            # Stop pipeline manager
            await self.pipeline_manager.stop()
            
            # Clean up builders
            for builder in self._builders.values():
                if hasattr(builder, 'cleanup'):
                    try:
                        await builder.cleanup()
                    except Exception as e:
                        self.logger.error(f"Error cleaning up builder: {e}")
            
            self.logger.info("PipelineFactory stopped successfully")
        
        except Exception as e:
            self.logger.error(f"Error stopping PipelineFactory: {e}")


# Global factory instance
_global_factory: Optional[PipelineFactory] = None


def get_pipeline_factory(config: Optional[AppConfig] = None) -> PipelineFactory:
    """Get global pipeline factory instance."""
    global _global_factory
    if _global_factory is None and config is not None:
        _global_factory = PipelineFactory(config)
    return _global_factory