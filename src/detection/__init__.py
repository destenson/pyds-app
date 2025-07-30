"""
Detection engine package for the DeepStream inference system.

This package provides pattern detection capabilities with extensible strategies,
custom model integration, and comprehensive result processing.
"""

from .models import (
    VideoDetection,
    DetectionMetadata,
    BoundingBox,
    DetectionResult,
    DetectionStrategy,
    ModelInfo,
    DetectionStatistics
)

__all__ = [
    'VideoDetection',
    'DetectionMetadata', 
    'BoundingBox',
    'DetectionResult',
    'DetectionStrategy',
    'ModelInfo',
    'DetectionStatistics'
]