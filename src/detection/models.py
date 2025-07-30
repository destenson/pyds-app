"""
Data models for the detection engine.

This module provides comprehensive data models for detection results, metadata,
bounding boxes, and detection statistics with validation and serialization.
"""

import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Tuple, NamedTuple
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field, validator, root_validator


class DetectionClass(str, Enum):
    """Common object detection classes."""
    PERSON = "person"
    VEHICLE = "vehicle"
    CAR = "car"
    TRUCK = "truck"
    BUS = "bus"
    MOTORCYCLE = "motorcycle"
    BICYCLE = "bicycle"
    ANIMAL = "animal"
    OBJECT = "object"
    FACE = "face"
    LICENSE_PLATE = "license_plate"
    UNKNOWN = "unknown"


class ConfidenceLevel(str, Enum):
    """Confidence level categories."""
    VERY_LOW = "very_low"    # 0.0 - 0.3
    LOW = "low"              # 0.3 - 0.5  
    MEDIUM = "medium"        # 0.5 - 0.7
    HIGH = "high"            # 0.7 - 0.9
    VERY_HIGH = "very_high"  # 0.9 - 1.0


@dataclass(frozen=True)
class BoundingBox:
    """
    Normalized bounding box coordinates.
    
    All coordinates are normalized to [0, 1] range relative to frame dimensions.
    """
    x: float  # Left edge (normalized)
    y: float  # Top edge (normalized) 
    width: float   # Width (normalized)
    height: float  # Height (normalized)
    
    def __post_init__(self):
        """Validate bounding box coordinates."""
        if not (0.0 <= self.x <= 1.0):
            raise ValueError(f"x coordinate must be in [0, 1], got {self.x}")
        if not (0.0 <= self.y <= 1.0):
            raise ValueError(f"y coordinate must be in [0, 1], got {self.y}")
        if not (0.0 < self.width <= 1.0):
            raise ValueError(f"width must be in (0, 1], got {self.width}")
        if not (0.0 < self.height <= 1.0):
            raise ValueError(f"height must be in (0, 1], got {self.height}")
        if self.x + self.width > 1.0:
            raise ValueError(f"x + width ({self.x + self.width}) exceeds 1.0")
        if self.y + self.height > 1.0:
            raise ValueError(f"y + height ({self.y + self.height}) exceeds 1.0")
    
    @property
    def center_x(self) -> float:
        """Get center x coordinate."""
        return self.x + self.width / 2
    
    @property
    def center_y(self) -> float:
        """Get center y coordinate."""
        return self.y + self.height / 2
    
    @property
    def area(self) -> float:
        """Get bounding box area."""
        return self.width * self.height
    
    @property
    def aspect_ratio(self) -> float:
        """Get aspect ratio (width / height)."""
        return self.width / self.height
    
    def to_absolute(self, frame_width: int, frame_height: int) -> 'AbsoluteBoundingBox':
        """Convert to absolute pixel coordinates."""
        return AbsoluteBoundingBox(
            x=int(self.x * frame_width),
            y=int(self.y * frame_height),
            width=int(self.width * frame_width),
            height=int(self.height * frame_height),
            frame_width=frame_width,
            frame_height=frame_height
        )
    
    def iou(self, other: 'BoundingBox') -> float:
        """Calculate Intersection over Union with another bounding box."""
        # Calculate intersection coordinates
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.width, other.x + other.width)
        y2 = min(self.y + self.height, other.y + other.height)
        
        # Check if there's intersection
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        # Calculate areas
        intersection_area = (x2 - x1) * (y2 - y1)
        union_area = self.area + other.area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'x': self.x,
            'y': self.y,
            'width': self.width,
            'height': self.height
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'BoundingBox':
        """Create from dictionary."""
        return cls(
            x=data['x'],
            y=data['y'],
            width=data['width'],
            height=data['height']
        )
    
    @classmethod
    def from_xyxy(cls, x1: float, y1: float, x2: float, y2: float) -> 'BoundingBox':
        """Create from (x1, y1, x2, y2) coordinates."""
        return cls(
            x=x1,
            y=y1,
            width=x2 - x1,
            height=y2 - y1
        )


@dataclass(frozen=True)
class AbsoluteBoundingBox:
    """Absolute bounding box in pixel coordinates."""
    x: int           # Left edge in pixels
    y: int           # Top edge in pixels
    width: int       # Width in pixels
    height: int      # Height in pixels
    frame_width: int # Frame width for validation
    frame_height: int # Frame height for validation
    
    def __post_init__(self):
        """Validate absolute coordinates."""
        if self.x < 0 or self.x >= self.frame_width:
            raise ValueError(f"x coordinate {self.x} out of bounds [0, {self.frame_width})")
        if self.y < 0 or self.y >= self.frame_height:
            raise ValueError(f"y coordinate {self.y} out of bounds [0, {self.frame_height})")
        if self.width <= 0 or self.x + self.width > self.frame_width:
            raise ValueError(f"Invalid width {self.width} for x={self.x}, frame_width={self.frame_width}")
        if self.height <= 0 or self.y + self.height > self.frame_height:
            raise ValueError(f"Invalid height {self.height} for y={self.y}, frame_height={self.frame_height}")
    
    def to_normalized(self) -> BoundingBox:
        """Convert to normalized coordinates."""
        return BoundingBox(
            x=self.x / self.frame_width,
            y=self.y / self.frame_height,
            width=self.width / self.frame_width,
            height=self.height / self.frame_height
        )


@dataclass
class DetectionMetadata:
    """Extended metadata for detections."""
    object_id: Optional[int] = None          # Tracking ID
    class_name: Optional[str] = None         # Human-readable class name
    class_id: Optional[int] = None           # Numeric class ID
    attributes: Dict[str, Any] = field(default_factory=dict)  # Custom attributes
    
    # DeepStream specific metadata
    tracker_confidence: Optional[float] = None   # Tracker confidence
    age: Optional[int] = None                    # Object age in frames
    direction: Optional[Tuple[float, float]] = None  # Movement direction (vx, vy)
    
    # Additional detection info
    model_name: Optional[str] = None             # Model that made the detection
    inference_time_ms: Optional[float] = None    # Inference time
    preprocessing_time_ms: Optional[float] = None # Preprocessing time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DetectionMetadata':
        """Create from dictionary."""
        return cls(**data)


class VideoDetection(BaseModel):
    """
    Represents a detected pattern in a video frame with comprehensive metadata.
    
    This is the core detection result model used throughout the system.
    """
    
    # Core detection information
    detection_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique detection ID")
    pattern_name: str = Field(description="Name of detected pattern/class")
    confidence: float = Field(ge=0.0, le=1.0, description="Detection confidence score")
    bounding_box: BoundingBox = Field(description="Normalized bounding box coordinates")
    
    # Temporal information
    timestamp: datetime = Field(default_factory=datetime.now, description="Detection timestamp")
    frame_number: int = Field(ge=0, description="Frame number in video sequence")
    
    # Source information
    source_id: str = Field(description="Video source identifier")
    source_name: Optional[str] = Field(default=None, description="Human-readable source name")
    
    # Extended metadata
    metadata: DetectionMetadata = Field(default_factory=DetectionMetadata, description="Extended detection metadata")
    
    # Processing information
    processing_latency_ms: Optional[float] = Field(default=None, ge=0, description="Processing latency in milliseconds")
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
            BoundingBox: lambda bb: bb.to_dict()
        }
    
    @validator('pattern_name')
    def validate_pattern_name(cls, v):
        """Validate pattern name is not empty."""
        if not v or not v.strip():
            raise ValueError("Pattern name cannot be empty")
        return v.strip()
    
    @validator('source_id')
    def validate_source_id(cls, v):
        """Validate source ID is not empty."""
        if not v or not v.strip():
            raise ValueError("Source ID cannot be empty")
        return v.strip()
    
    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Get confidence level category."""
        if self.confidence < 0.3:
            return ConfidenceLevel.VERY_LOW
        elif self.confidence < 0.5:
            return ConfidenceLevel.LOW
        elif self.confidence < 0.7:
            return ConfidenceLevel.MEDIUM
        elif self.confidence < 0.9:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH
    
    @property
    def age_seconds(self) -> float:
        """Get detection age in seconds."""
        return (datetime.now() - self.timestamp).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization."""
        result = self.dict()
        result['timestamp'] = self.timestamp.isoformat()
        result['bounding_box'] = self.bounding_box.to_dict()
        result['metadata'] = self.metadata.to_dict()
        result['confidence_level'] = self.confidence_level.value
        result['age_seconds'] = self.age_seconds
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VideoDetection':
        """Create from dictionary."""
        # Handle nested objects
        if 'bounding_box' in data and isinstance(data['bounding_box'], dict):
            data['bounding_box'] = BoundingBox.from_dict(data['bounding_box'])
        if 'metadata' in data and isinstance(data['metadata'], dict):
            data['metadata'] = DetectionMetadata.from_dict(data['metadata'])
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'VideoDetection':
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


@dataclass
class DetectionResult:
    """
    Result of detection processing for a single frame or batch.
    """
    detections: List[VideoDetection] = field(default_factory=list)
    frame_number: int = 0
    source_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    processing_time_ms: float = 0.0
    
    # Statistics
    total_objects: int = field(init=False)
    high_confidence_objects: int = field(init=False)
    classes_detected: List[str] = field(init=False)
    
    def __post_init__(self):
        """Calculate statistics after initialization."""
        self.total_objects = len(self.detections)
        self.high_confidence_objects = sum(1 for d in self.detections if d.confidence >= 0.7)
        self.classes_detected = list(set(d.pattern_name for d in self.detections))
    
    @property
    def average_confidence(self) -> float:
        """Get average confidence of all detections."""
        if not self.detections:
            return 0.0
        return sum(d.confidence for d in self.detections) / len(self.detections)
    
    @property
    def max_confidence(self) -> float:
        """Get maximum confidence of all detections."""
        if not self.detections:
            return 0.0
        return max(d.confidence for d in self.detections)
    
    def filter_by_confidence(self, min_confidence: float) -> 'DetectionResult':
        """Create new result with detections above confidence threshold."""
        filtered_detections = [d for d in self.detections if d.confidence >= min_confidence]
        return DetectionResult(
            detections=filtered_detections,
            frame_number=self.frame_number,
            source_id=self.source_id,
            timestamp=self.timestamp,
            processing_time_ms=self.processing_time_ms
        )
    
    def filter_by_class(self, class_names: List[str]) -> 'DetectionResult':
        """Create new result with detections of specified classes."""
        filtered_detections = [d for d in self.detections if d.pattern_name in class_names]
        return DetectionResult(
            detections=filtered_detections,
            frame_number=self.frame_number,
            source_id=self.source_id,
            timestamp=self.timestamp,
            processing_time_ms=self.processing_time_ms
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'detections': [d.to_dict() for d in self.detections],
            'frame_number': self.frame_number,
            'source_id': self.source_id,
            'timestamp': self.timestamp.isoformat(),
            'processing_time_ms': self.processing_time_ms,
            'total_objects': self.total_objects,
            'high_confidence_objects': self.high_confidence_objects,
            'classes_detected': self.classes_detected,
            'average_confidence': self.average_confidence,
            'max_confidence': self.max_confidence
        }


@dataclass
class ModelInfo:
    """Information about a detection model."""
    name: str
    version: str
    path: Optional[Path] = None
    
    # Model specifications
    input_shape: Tuple[int, int, int] = (3, 640, 640)  # (C, H, W)
    output_classes: List[str] = field(default_factory=list)
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.5
    
    # Performance characteristics
    average_inference_time_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    
    # Metadata
    description: Optional[str] = None
    author: Optional[str] = None
    license: Optional[str] = None
    created_date: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        if self.path:
            result['path'] = str(self.path)
        if self.created_date:
            result['created_date'] = self.created_date.isoformat()
        return result


class DetectionStrategy(ABC):
    """
    Abstract base class for detection strategies.
    
    All detection strategies must implement this interface to be compatible
    with the detection engine.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize detection strategy.
        
        Args:
            name: Strategy name
            config: Strategy configuration
        """
        self.name = name
        self.config = config
        self.enabled = config.get('enabled', True)
        self.model_info: Optional[ModelInfo] = None
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the detection strategy.
        
        Returns:
            True if initialization successful
        """
        pass
    
    @abstractmethod
    async def detect(self, frame_data: Any, source_id: str, frame_number: int) -> DetectionResult:
        """
        Perform detection on frame data.
        
        Args:
            frame_data: Frame data (format depends on strategy)
            source_id: Source identifier
            frame_number: Frame number
            
        Returns:
            Detection results
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up strategy resources."""
        pass
    
    def should_process(self, detection: VideoDetection) -> bool:
        """
        Determine if this strategy should process the given detection.
        
        Args:
            detection: Detection to evaluate
            
        Returns:
            True if strategy should process the detection
        """
        return self.enabled
    
    def get_info(self) -> Dict[str, Any]:
        """Get strategy information."""
        return {
            'name': self.name,
            'enabled': self.enabled,
            'config': self.config,
            'model_info': self.model_info.to_dict() if self.model_info else None
        }


@dataclass
class DetectionStatistics:
    """Statistics for detection performance tracking."""
    
    # Counters
    total_detections: int = 0
    detections_by_class: Dict[str, int] = field(default_factory=dict)
    detections_by_confidence: Dict[str, int] = field(default_factory=dict)
    
    # Timing statistics
    total_processing_time_ms: float = 0.0
    average_processing_time_ms: float = 0.0
    min_processing_time_ms: float = float('inf')
    max_processing_time_ms: float = 0.0
    
    # Frame statistics
    frames_processed: int = 0
    frames_with_detections: int = 0
    
    # Source statistics
    detections_by_source: Dict[str, int] = field(default_factory=dict)
    
    # Time tracking
    start_time: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)
    
    def update(self, result: DetectionResult):
        """Update statistics with new detection result."""
        self.frames_processed += 1
        self.last_update = datetime.now()
        
        if result.detections:
            self.frames_with_detections += 1
            
        # Update detection counts
        for detection in result.detections:
            self.total_detections += 1
            
            # By class
            class_name = detection.pattern_name
            self.detections_by_class[class_name] = self.detections_by_class.get(class_name, 0) + 1
            
            # By confidence level
            confidence_level = detection.confidence_level.value
            self.detections_by_confidence[confidence_level] = self.detections_by_confidence.get(confidence_level, 0) + 1
            
            # By source
            source_id = detection.source_id
            self.detections_by_source[source_id] = self.detections_by_source.get(source_id, 0) + 1
        
        # Update timing statistics
        processing_time = result.processing_time_ms
        if processing_time > 0:
            self.total_processing_time_ms += processing_time
            self.average_processing_time_ms = self.total_processing_time_ms / self.frames_processed
            self.min_processing_time_ms = min(self.min_processing_time_ms, processing_time)
            self.max_processing_time_ms = max(self.max_processing_time_ms, processing_time)
    
    @property
    def detection_rate(self) -> float:
        """Get detection rate (detections per frame)."""
        if self.frames_processed == 0:
            return 0.0
        return self.total_detections / self.frames_processed
    
    @property
    def frame_detection_rate(self) -> float:
        """Get frame detection rate (frames with detections / total frames)."""
        if self.frames_processed == 0:
            return 0.0
        return self.frames_with_detections / self.frames_processed
    
    @property
    def processing_fps(self) -> float:
        """Get processing FPS based on processing time."""
        if self.average_processing_time_ms <= 0:
            return 0.0
        return 1000.0 / self.average_processing_time_ms
    
    @property
    def runtime_seconds(self) -> float:
        """Get total runtime in seconds."""
        return (self.last_update - self.start_time).total_seconds()
    
    def reset(self):
        """Reset all statistics."""
        self.total_detections = 0
        self.detections_by_class.clear()
        self.detections_by_confidence.clear()
        self.total_processing_time_ms = 0.0
        self.average_processing_time_ms = 0.0
        self.min_processing_time_ms = float('inf')
        self.max_processing_time_ms = 0.0
        self.frames_processed = 0
        self.frames_with_detections = 0
        self.detections_by_source.clear()
        self.start_time = datetime.now()
        self.last_update = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_detections': self.total_detections,
            'detections_by_class': self.detections_by_class,
            'detections_by_confidence': self.detections_by_confidence,
            'total_processing_time_ms': self.total_processing_time_ms,
            'average_processing_time_ms': self.average_processing_time_ms,
            'min_processing_time_ms': self.min_processing_time_ms if self.min_processing_time_ms != float('inf') else 0.0,
            'max_processing_time_ms': self.max_processing_time_ms,
            'frames_processed': self.frames_processed,
            'frames_with_detections': self.frames_with_detections,
            'detections_by_source': self.detections_by_source,
            'detection_rate': self.detection_rate,
            'frame_detection_rate': self.frame_detection_rate,
            'processing_fps': self.processing_fps,
            'runtime_seconds': self.runtime_seconds,
            'start_time': self.start_time.isoformat(),
            'last_update': self.last_update.isoformat()
        }


# Utility functions for common operations
def create_test_detection(
    pattern_name: str = "person",
    confidence: float = 0.85,
    bbox: Optional[BoundingBox] = None,
    source_id: str = "test_source",
    frame_number: int = 1
) -> VideoDetection:
    """Create a test detection for development and testing."""
    if bbox is None:
        bbox = BoundingBox(x=0.1, y=0.1, width=0.2, height=0.3)
    
    return VideoDetection(
        pattern_name=pattern_name,
        confidence=confidence,
        bounding_box=bbox,
        source_id=source_id,
        frame_number=frame_number
    )


def merge_detection_results(results: List[DetectionResult]) -> DetectionResult:
    """Merge multiple detection results into a single result."""
    if not results:
        return DetectionResult()
    
    # Use first result as base
    merged = DetectionResult(
        detections=[],
        frame_number=results[0].frame_number,
        source_id=results[0].source_id,
        timestamp=results[0].timestamp,
        processing_time_ms=sum(r.processing_time_ms for r in results)
    )
    
    # Merge all detections
    for result in results:
        merged.detections.extend(result.detections)
    
    return merged


def filter_overlapping_detections(
    detections: List[VideoDetection],
    iou_threshold: float = 0.5
) -> List[VideoDetection]:
    """Filter overlapping detections using Non-Maximum Suppression."""
    if not detections:
        return []
    
    # Sort by confidence (highest first)
    sorted_detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
    filtered = []
    
    for detection in sorted_detections:
        # Check if this detection overlaps significantly with any kept detection
        keep = True
        for kept_detection in filtered:
            if detection.bounding_box.iou(kept_detection.bounding_box) > iou_threshold:
                keep = False
                break
        
        if keep:
            filtered.append(detection)
    
    return filtered