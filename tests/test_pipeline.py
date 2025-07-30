"""
Tests for GStreamer pipeline management.

Tests pipeline lifecycle, multi-source management, state transitions, and error recovery.
"""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, patch, call
from datetime import datetime
import time

from src.pipeline.manager import (
    PipelineManager, PipelineState, PipelineInfo, SourcePadInfo
)
from src.pipeline.sources import (
    VideoSourceManager, SourceStatus, SourceHealth, SourceMetrics
)
from src.pipeline.elements import (
    DeepStreamElementsManager, ElementType, MemoryType, ElementInfo
)
from src.pipeline.factory import (
    PipelineFactory, PipelineTemplate, OptimizationProfile
)
from src.config import AppConfig, SourceConfig, SourceType
from src.utils.errors import PipelineError, SourceError


class TestPipelineManager:
    """Test pipeline lifecycle management."""
    
    @pytest.mark.asyncio
    async def test_create_pipeline(self, test_config, test_source, mock_gstreamer):
        """Test creating a basic pipeline."""
        manager = PipelineManager(test_config)
        
        # Create pipeline
        pipeline_id = await manager.create_multi_source_pipeline([test_source])
        
        assert pipeline_id is not None
        assert pipeline_id in manager._pipelines
        
        pipeline_info = manager._pipelines[pipeline_id]
        assert pipeline_info.state == PipelineState.NULL
        assert len(pipeline_info.sources) == 1
    
    @pytest.mark.asyncio
    async def test_pipeline_state_transitions(self, test_config, test_source, mock_gstreamer):
        """Test pipeline state transitions."""
        manager = PipelineManager(test_config)
        pipeline_id = await manager.create_multi_source_pipeline([test_source])
        
        # Mock successful state changes
        mock_gstreamer.StateChangeReturn.SUCCESS = 1
        manager._pipelines[pipeline_id].pipeline.set_state = Mock(return_value=1)
        
        # Start pipeline
        await manager.start_pipeline(pipeline_id)
        assert manager._pipelines[pipeline_id].state == PipelineState.PLAYING
        
        # Pause pipeline
        await manager.pause_pipeline(pipeline_id)
        assert manager._pipelines[pipeline_id].state == PipelineState.PAUSED
        
        # Stop pipeline
        await manager.stop_pipeline(pipeline_id)
        assert manager._pipelines[pipeline_id].state == PipelineState.NULL
    
    @pytest.mark.asyncio
    async def test_multi_source_pipeline(self, test_config, mock_gstreamer):
        """Test pipeline with multiple sources."""
        manager = PipelineManager(test_config)
        
        # Create multiple sources
        sources = [
            SourceConfig(id=f"source_{i}", name=f"Source {i}", 
                        type=SourceType.TEST, uri="videotestsrc")
            for i in range(3)
        ]
        
        pipeline_id = await manager.create_multi_source_pipeline(sources)
        
        assert pipeline_id is not None
        pipeline_info = manager._pipelines[pipeline_id]
        assert len(pipeline_info.sources) == 3
    
    @pytest.mark.asyncio
    async def test_pipeline_error_handling(self, test_config, test_source, mock_gstreamer):
        """Test pipeline error handling."""
        manager = PipelineManager(test_config)
        pipeline_id = await manager.create_multi_source_pipeline([test_source])
        
        # Mock state change failure
        mock_gstreamer.StateChangeReturn.FAILURE = 3
        manager._pipelines[pipeline_id].pipeline.set_state = Mock(return_value=3)
        
        # Should raise error on failed state change
        with pytest.raises(PipelineError):
            await manager.start_pipeline(pipeline_id)
        
        assert manager._pipelines[pipeline_id].state == PipelineState.ERROR
    
    @pytest.mark.asyncio
    async def test_dynamic_source_management(self, test_config, mock_gstreamer):
        """Test adding/removing sources dynamically."""
        manager = PipelineManager(test_config)
        
        # Start with one source
        source1 = SourceConfig(id="source_1", name="Source 1", 
                              type=SourceType.TEST, uri="videotestsrc")
        pipeline_id = await manager.create_multi_source_pipeline([source1])
        
        # Add another source
        source2 = SourceConfig(id="source_2", name="Source 2",
                              type=SourceType.TEST, uri="videotestsrc")
        await manager.add_source(pipeline_id, source2)
        
        assert len(manager._pipelines[pipeline_id].sources) == 2
        
        # Remove source
        await manager.remove_source(pipeline_id, "source_1")
        assert len(manager._pipelines[pipeline_id].sources) == 1
    
    def test_pipeline_metrics(self, test_config, test_source):
        """Test pipeline metrics collection."""
        manager = PipelineManager(test_config)
        
        # Create mock pipeline info
        pipeline_info = PipelineInfo(
            pipeline_id="test_pipeline",
            pipeline=MagicMock(),
            sources=[test_source],
            fps=30.0,
            frame_count=1000,
            memory_usage_mb=256.5
        )
        
        manager._pipelines["test_pipeline"] = pipeline_info
        
        # Get pipeline state
        states = manager.get_pipeline_states()
        assert len(states) == 1
        assert states[0]['pipeline_id'] == "test_pipeline"
        assert states[0]['fps'] == 30.0
        assert states[0]['frame_count'] == 1000


class TestVideoSourceManager:
    """Test video source management."""
    
    def test_register_source(self, test_source):
        """Test registering a video source."""
        manager = VideoSourceManager()
        
        # Register source
        manager.register_source(test_source)
        
        assert test_source.id in manager._sources
        state = manager._sources[test_source.id]
        assert state.config == test_source
        assert state.status == SourceStatus.UNKNOWN
    
    @pytest.mark.asyncio
    async def test_validate_rtsp_source(self):
        """Test RTSP source validation."""
        manager = VideoSourceManager()
        
        # Mock successful RTSP connection check
        with patch('socket.socket') as mock_socket:
            mock_conn = MagicMock()
            mock_socket.return_value = mock_conn
            mock_conn.connect_ex.return_value = 0  # Success
            
            rtsp_source = SourceConfig(
                id="rtsp_1",
                name="RTSP Camera",
                type=SourceType.RTSP,
                uri="rtsp://192.168.1.100:554/stream"
            )
            
            available = await manager.validate_source(rtsp_source)
            assert available is True
    
    @pytest.mark.asyncio
    async def test_validate_file_source(self, temp_dir):
        """Test file source validation."""
        manager = VideoSourceManager()
        
        # Create test file
        test_file = temp_dir / "test_video.mp4"
        test_file.write_text("dummy video content")
        
        file_source = SourceConfig(
            id="file_1",
            name="Test Video",
            type=SourceType.FILE,
            uri=str(test_file)
        )
        
        available = await manager.validate_source(file_source)
        assert available is True
        
        # Test non-existent file
        bad_source = SourceConfig(
            id="file_2",
            name="Missing Video",
            type=SourceType.FILE,
            uri="/nonexistent/video.mp4"
        )
        
        available = await manager.validate_source(bad_source)
        assert available is False
    
    def test_source_health_monitoring(self, test_source):
        """Test source health monitoring."""
        manager = VideoSourceManager()
        manager.register_source(test_source)
        
        # Update source metrics
        manager.update_source_metrics(
            test_source.id,
            frames_received=1000,
            fps=30.0,
            bytes_received=1024*1024
        )
        
        health = manager.get_source_health(test_source.id)
        assert health == SourceHealth.HEALTHY
        
        # Simulate unhealthy source (no frames)
        manager._sources[test_source.id].metrics.last_frame_time = time.time() - 60
        health = manager.get_source_health(test_source.id)
        assert health == SourceHealth.UNHEALTHY
    
    @pytest.mark.asyncio
    async def test_source_reconnection(self, test_source):
        """Test automatic source reconnection."""
        manager = VideoSourceManager()
        manager.register_source(test_source)
        
        # Simulate connection failure
        manager._sources[test_source.id].status = SourceStatus.ERROR
        manager._sources[test_source.id].retry_count = 0
        
        # Mock successful reconnection
        with patch.object(manager, 'validate_source', return_value=True):
            reconnected = await manager.reconnect_source(test_source.id)
            assert reconnected is True
            assert manager._sources[test_source.id].status == SourceStatus.AVAILABLE
    
    def test_source_metrics_aggregation(self):
        """Test aggregating metrics from multiple sources."""
        manager = VideoSourceManager()
        
        # Register multiple sources
        for i in range(3):
            source = SourceConfig(
                id=f"source_{i}",
                name=f"Source {i}",
                type=SourceType.TEST,
                uri="videotestsrc"
            )
            manager.register_source(source)
            
            # Update metrics
            manager.update_source_metrics(
                source.id,
                frames_received=1000 * (i + 1),
                fps=30.0,
                bytes_received=1024 * 1024 * (i + 1)
            )
        
        # Get aggregated metrics
        total_metrics = manager.get_aggregated_metrics()
        assert total_metrics['total_frames'] == 6000  # 1000 + 2000 + 3000
        assert total_metrics['average_fps'] == 30.0
        assert total_metrics['total_sources'] == 3


class TestDeepStreamElements:
    """Test DeepStream element management."""
    
    def test_create_streammux_element(self, test_config, mock_gstreamer):
        """Test creating nvstreammux element."""
        manager = DeepStreamElementsManager(test_config)
        
        # Create streammux
        element_info = manager.create_streammux(
            batch_size=4,
            width=1920,
            height=1080
        )
        
        assert element_info is not None
        assert element_info.element_type == ElementType.STREAMMUX
        mock_gstreamer.ElementFactory.make.assert_called_with("nvstreammux", "stream-muxer")
    
    def test_create_inference_element(self, test_config, mock_gstreamer):
        """Test creating nvinfer element."""
        manager = DeepStreamElementsManager(test_config)
        
        # Create inference element
        element_info = manager.create_inference_element(
            config_file="/path/to/config.txt",
            batch_size=4
        )
        
        assert element_info is not None
        assert element_info.element_type == ElementType.INFERENCE
        mock_gstreamer.ElementFactory.make.assert_called_with("nvinfer", "primary-inference")
    
    def test_version_compatibility(self, test_config, mock_deepstream_info):
        """Test DeepStream version compatibility."""
        manager = DeepStreamElementsManager(test_config)
        
        # Check plugin availability
        available = manager.is_plugin_available("nvstreammux")
        assert available is True  # Mocked as available
        
        # Get version-specific properties
        props = manager.get_version_specific_properties(
            ElementType.STREAMMUX,
            mock_deepstream_info.version
        )
        assert isinstance(props, dict)
    
    def test_memory_type_configuration(self, test_config, mock_gstreamer):
        """Test GPU memory type configuration."""
        manager = DeepStreamElementsManager(test_config)
        
        # Create element with specific memory type
        element_info = manager.create_converter(
            memory_type=MemoryType.UNIFIED
        )
        
        assert element_info is not None
        assert element_info.memory_type == MemoryType.UNIFIED


class TestPipelineFactory:
    """Test pipeline factory patterns."""
    
    @pytest.mark.asyncio
    async def test_basic_detection_pipeline(self, test_config, test_source, mock_gstreamer):
        """Test creating basic detection pipeline."""
        factory = PipelineFactory(test_config)
        
        # Build pipeline
        blueprint = await factory.build_pipeline(
            sources=[test_source],
            template=PipelineTemplate.BASIC_DETECTION
        )
        
        assert blueprint is not None
        assert blueprint.template.name == PipelineTemplate.BASIC_DETECTION.value
        assert len(blueprint.sources) == 1
    
    @pytest.mark.asyncio
    async def test_multi_source_optimization(self, test_config, mock_gstreamer):
        """Test multi-source pipeline optimization."""
        factory = PipelineFactory(test_config)
        
        # Create multiple sources
        sources = [
            SourceConfig(id=f"s{i}", name=f"S{i}", type=SourceType.TEST, uri="test")
            for i in range(4)
        ]
        
        # Build optimized pipeline
        blueprint = await factory.build_pipeline(
            sources=sources,
            template=PipelineTemplate.MULTI_SOURCE_DETECTION,
            optimization_profile=OptimizationProfile.HIGH_THROUGHPUT
        )
        
        assert blueprint is not None
        assert blueprint.optimization_settings['profile'] == OptimizationProfile.HIGH_THROUGHPUT
        assert blueprint.pipeline_config.batch_size >= len(sources)
    
    def test_pipeline_templates(self, test_config):
        """Test available pipeline templates."""
        factory = PipelineFactory(test_config)
        
        templates = factory.get_available_templates()
        assert len(templates) > 0
        
        # Check template properties
        for template in templates:
            assert hasattr(template, 'name')
            assert hasattr(template, 'supported_source_types')
            assert hasattr(template, 'required_elements')
    
    def test_custom_pipeline_builder(self, test_config, test_source):
        """Test custom pipeline builder."""
        factory = PipelineFactory(test_config)
        
        # Register custom builder
        def custom_builder(sources, config):
            return {
                'elements': ['custom_element'],
                'properties': {'custom': True}
            }
        
        factory.register_custom_builder('custom_pipeline', custom_builder)
        
        # Use custom builder
        result = factory.build_custom_pipeline('custom_pipeline', [test_source])
        assert result['properties']['custom'] is True