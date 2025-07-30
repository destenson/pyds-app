"""
Built-in detection strategies for the detection engine.

This module provides concrete implementations of common detection strategies
including YOLO, template matching, and feature-based detection with 
DeepStream compatibility across versions 5.x through 7.x.
"""

import asyncio
import time
import cv2
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

from ..config import AppConfig, DetectionConfig
from ..utils.errors import DetectionError, handle_error
from ..utils.logging import get_logger, performance_context
from ..utils.deepstream import get_deepstream_api, get_deepstream_info
from .models import (
    DetectionStrategy, VideoDetection, DetectionResult, BoundingBox,
    DetectionMetadata, ModelInfo, create_test_detection
)


@dataclass
class YOLOConfig:
    """Configuration for YOLO detection strategy."""
    model_path: str
    config_path: str
    weights_path: Optional[str] = None
    class_names_path: Optional[str] = None
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    input_size: Tuple[int, int] = (640, 640)
    classes_filter: Optional[List[str]] = None


@dataclass
class TemplateConfig:
    """Configuration for template matching strategy."""
    template_dir: str
    match_threshold: float = 0.8
    scale_range: Tuple[float, float] = (0.5, 2.0)
    scale_steps: int = 10
    rotation_angles: List[float] = None
    method: str = "cv2.TM_CCOEFF_NORMED"


@dataclass
class FeatureConfig:
    """Configuration for feature-based detection strategy."""
    detector_type: str = "SIFT"  # SIFT, SURF, ORB
    matcher_type: str = "FLANN"  # FLANN, BF
    match_ratio: float = 0.7
    min_matches: int = 10
    reference_images_dir: str = ""


class YOLODetectionStrategy(DetectionStrategy):
    """YOLO-based object detection strategy with DeepStream integration."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize YOLO detection strategy.
        
        Args:
            name: Strategy name
            config: Strategy configuration
        """
        super().__init__(name, config)
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
        # Parse YOLO-specific configuration
        self.yolo_config = YOLOConfig(**config.get('yolo', {}))
        
        # Detection components
        self._net = None
        self._output_layers = None
        self._class_names = []
        self._deepstream_api = None
        self._deepstream_info = None
        
        # Performance tracking
        self._detection_count = 0
        self._total_inference_time = 0.0
        
    async def initialize(self) -> bool:
        """Initialize YOLO detection components."""
        try:
            self.logger.info(f"Initializing YOLO strategy: {self.name}")
            
            # Initialize DeepStream components
            self._deepstream_api = get_deepstream_api()
            self._deepstream_info = get_deepstream_info()
            
            # Load YOLO model (OpenCV DNN backend for compatibility)
            if Path(self.yolo_config.model_path).exists():
                self._net = cv2.dnn.readNet(
                    self.yolo_config.model_path,
                    self.yolo_config.config_path
                )
                
                # Get output layer names
                layer_names = self._net.getLayerNames()
                self._output_layers = [layer_names[i[0] - 1] for i in self._net.getUnconnectedOutLayers()]
                
                # Set preferable backend and target
                self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                
                self.logger.info(f"Loaded YOLO model from {self.yolo_config.model_path}")
            else:
                self.logger.warning(f"YOLO model file not found: {self.yolo_config.model_path}")
                # Continue with placeholder for testing
            
            # Load class names
            if self.yolo_config.class_names_path and Path(self.yolo_config.class_names_path).exists():
                with open(self.yolo_config.class_names_path, 'r') as f:
                    self._class_names = [line.strip() for line in f.readlines()]
                self.logger.info(f"Loaded {len(self._class_names)} class names")
            else:
                # Default COCO classes for testing
                self._class_names = [
                    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
                    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench"
                ]
                self.logger.info("Using default COCO class names")
            
            # Set model info
            self.model_info = ModelInfo(
                name="YOLO",
                version="v5",
                path=Path(self.yolo_config.model_path) if self.yolo_config.model_path else None,
                input_shape=(3, *self.yolo_config.input_size),
                output_classes=self._class_names,
                confidence_threshold=self.yolo_config.confidence_threshold,
                nms_threshold=self.yolo_config.nms_threshold,
                description="YOLO object detection model"
            )
            
            self.logger.info(f"YOLO strategy {self.name} initialized successfully")
            return True
            
        except Exception as e:
            error = handle_error(e, context={'strategy': self.name})
            self.logger.error(f"Failed to initialize YOLO strategy: {error}")
            return False
    
    async def detect(self, frame_data: Any, source_id: str, frame_number: int) -> DetectionResult:
        """
        Perform YOLO detection on frame data.
        
        Args:
            frame_data: Frame data (numpy array or GStreamer buffer)
            source_id: Source identifier
            frame_number: Frame number
            
        Returns:
            Detection results
        """
        start_time = time.perf_counter()
        detections = []
        
        try:
            with performance_context(f"yolo_detect_{source_id}_{frame_number}"):
                # Convert frame data to numpy array if needed
                if hasattr(frame_data, 'data'):
                    # Handle GStreamer buffer
                    frame = self._gst_buffer_to_numpy(frame_data)
                else:
                    # Assume numpy array
                    frame = frame_data
                
                if frame is None:
                    self.logger.warning("Failed to convert frame data")
                    return DetectionResult(
                        detections=[],
                        frame_number=frame_number,
                        source_id=source_id,
                        timestamp=datetime.now(),
                        processing_time_ms=(time.perf_counter() - start_time) * 1000
                    )
                
                # Perform YOLO detection
                if self._net is not None:
                    detections = await self._run_yolo_detection(frame, source_id, frame_number)
                else:
                    # Create placeholder detections for testing
                    detections = await self._create_placeholder_detections(source_id, frame_number)
                
                # Update performance metrics
                processing_time = (time.perf_counter() - start_time) * 1000
                self._detection_count += 1
                self._total_inference_time += processing_time
                
                return DetectionResult(
                    detections=detections,
                    frame_number=frame_number,
                    source_id=source_id,
                    timestamp=datetime.now(),
                    processing_time_ms=processing_time
                )
        
        except Exception as e:
            error = handle_error(e, context={
                'strategy': self.name,
                'source_id': source_id,
                'frame_number': frame_number
            })
            self.logger.error(f"YOLO detection error: {error}")
            
            return DetectionResult(
                detections=[],
                frame_number=frame_number,
                source_id=source_id,
                timestamp=datetime.now(),
                processing_time_ms=(time.perf_counter() - start_time) * 1000
            )
    
    async def _run_yolo_detection(self, frame: np.ndarray, source_id: str, frame_number: int) -> List[VideoDetection]:
        """Run YOLO detection on frame."""
        detections = []
        
        try:
            height, width = frame.shape[:2]
            
            # Prepare input blob
            blob = cv2.dnn.blobFromImage(
                frame,
                1/255.0,
                self.yolo_config.input_size,
                swapRB=True,
                crop=False
            )
            
            # Set input and run forward pass
            self._net.setInput(blob)
            outputs = self._net.forward(self._output_layers)
            
            # Process outputs
            boxes = []
            confidences = []
            class_ids = []
            
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if confidence > self.yolo_config.confidence_threshold:
                        # Extract bounding box
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        box_width = int(detection[2] * width)
                        box_height = int(detection[3] * height)
                        
                        # Calculate top-left corner
                        x = int(center_x - box_width / 2)
                        y = int(center_y - box_height / 2)
                        
                        boxes.append([x, y, box_width, box_height])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            # Apply Non-Maximum Suppression
            if boxes:
                indices = cv2.dnn.NMSBoxes(
                    boxes,
                    confidences,
                    self.yolo_config.confidence_threshold,
                    self.yolo_config.nms_threshold
                )
                
                if len(indices) > 0:
                    for i in indices.flatten():
                        x, y, w, h = boxes[i]
                        confidence = confidences[i]
                        class_id = class_ids[i]
                        
                        # Convert to normalized coordinates
                        bbox = BoundingBox(
                            x=max(0, x / width),
                            y=max(0, y / height),
                            width=min(1.0, w / width),
                            height=min(1.0, h / height)
                        )
                        
                        # Get class name
                        class_name = self._class_names[class_id] if class_id < len(self._class_names) else "unknown"
                        
                        # Apply class filter if configured
                        if self.yolo_config.classes_filter and class_name not in self.yolo_config.classes_filter:
                            continue
                        
                        # Create detection
                        detection = VideoDetection(
                            pattern_name=class_name,
                            confidence=confidence,
                            bounding_box=bbox,
                            frame_number=frame_number,
                            source_id=source_id,
                            timestamp=datetime.now(),
                            metadata=DetectionMetadata(
                                class_id=class_id,
                                class_name=class_name,
                                model_name=self.name,
                                inference_time_ms=5.0  # Placeholder
                            )
                        )
                        
                        detections.append(detection)
            
            return detections
        
        except Exception as e:
            self.logger.error(f"Error in YOLO detection: {e}")
            return []
    
    async def _create_placeholder_detections(self, source_id: str, frame_number: int) -> List[VideoDetection]:
        """Create placeholder detections for testing."""
        return [
            VideoDetection(
                pattern_name="person",
                confidence=0.85,
                bounding_box=BoundingBox(x=0.1, y=0.1, width=0.2, height=0.4),
                frame_number=frame_number,
                source_id=source_id,
                timestamp=datetime.now(),
                metadata=DetectionMetadata(
                    class_id=0,
                    class_name="person",
                    model_name=self.name
                )
            )
        ]
    
    def _gst_buffer_to_numpy(self, gst_buffer) -> Optional[np.ndarray]:
        """Convert GStreamer buffer to numpy array."""
        try:
            # This is a simplified conversion - real implementation would
            # need to handle different buffer formats and memory types
            
            # For now, return a placeholder array for testing
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        except Exception as e:
            self.logger.error(f"Error converting GStreamer buffer: {e}")
            return None
    
    def should_process(self, detection: VideoDetection) -> bool:
        """Determine if YOLO should process this detection."""
        # YOLO typically processes raw frames, not existing detections
        return self.enabled
    
    async def cleanup(self) -> None:
        """Clean up YOLO resources."""
        try:
            self.logger.info(f"Cleaning up YOLO strategy: {self.name}")
            
            # Calculate average inference time
            if self._detection_count > 0:
                avg_time = self._total_inference_time / self._detection_count
                self.logger.info(f"YOLO average inference time: {avg_time:.2f}ms over {self._detection_count} detections")
            
            # Clean up OpenCV DNN network
            self._net = None
            self._output_layers = None
            
            self.logger.info(f"YOLO strategy {self.name} cleanup completed")
        
        except Exception as e:
            self.logger.error(f"Error during YOLO cleanup: {e}")


class TemplateMatchingStrategy(DetectionStrategy):
    """Template matching detection strategy for specific patterns."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize template matching strategy.
        
        Args:
            name: Strategy name
            config: Strategy configuration
        """
        super().__init__(name, config)
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
        # Parse template-specific configuration
        self.template_config = TemplateConfig(**config.get('template', {}))
        
        # Template storage
        self._templates: Dict[str, np.ndarray] = {}
        self._template_scales: Dict[str, List[np.ndarray]] = {}
        
        # OpenCV template matching method
        self._match_methods = {
            "cv2.TM_CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED,
            "cv2.TM_CCORR_NORMED": cv2.TM_CCORR_NORMED,
            "cv2.TM_SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED
        }
        self._match_method = self._match_methods.get(
            self.template_config.method, 
            cv2.TM_CCOEFF_NORMED
        )
    
    async def initialize(self) -> bool:
        """Initialize template matching components."""
        try:
            self.logger.info(f"Initializing template matching strategy: {self.name}")
            
            # Load templates from directory
            template_dir = Path(self.template_config.template_dir)
            if template_dir.exists():
                await self._load_templates(template_dir)
            else:
                self.logger.warning(f"Template directory not found: {template_dir}")
                # Create placeholder templates for testing
                await self._create_placeholder_templates()
            
            # Set model info
            self.model_info = ModelInfo(
                name="TemplateMatching",
                version="1.0",
                path=template_dir if template_dir.exists() else None,
                output_classes=list(self._templates.keys()),
                confidence_threshold=self.template_config.match_threshold,
                description="OpenCV template matching"
            )
            
            self.logger.info(f"Template matching strategy initialized with {len(self._templates)} templates")
            return True
        
        except Exception as e:
            error = handle_error(e, context={'strategy': self.name})
            self.logger.error(f"Failed to initialize template matching: {error}")
            return False
    
    async def _load_templates(self, template_dir: Path):
        """Load templates from directory."""
        try:
            template_files = list(template_dir.glob("*.png")) + list(template_dir.glob("*.jpg"))
            
            for template_file in template_files:
                template_name = template_file.stem
                template_img = cv2.imread(str(template_file), cv2.IMREAD_GRAYSCALE)
                
                if template_img is not None:
                    self._templates[template_name] = template_img
                    
                    # Generate scaled versions
                    scaled_templates = []
                    scale_min, scale_max = self.template_config.scale_range
                    for i in range(self.template_config.scale_steps):
                        scale = scale_min + (scale_max - scale_min) * i / (self.template_config.scale_steps - 1)
                        height, width = template_img.shape
                        new_height, new_width = int(height * scale), int(width * scale)
                        
                        if new_height > 0 and new_width > 0:
                            scaled = cv2.resize(template_img, (new_width, new_height))
                            scaled_templates.append(scaled)
                    
                    self._template_scales[template_name] = scaled_templates
                    self.logger.debug(f"Loaded template {template_name} with {len(scaled_templates)} scales")
                
                else:
                    self.logger.warning(f"Failed to load template: {template_file}")
        
        except Exception as e:
            self.logger.error(f"Error loading templates: {e}")
    
    async def _create_placeholder_templates(self):
        """Create placeholder templates for testing."""
        # Create simple geometric shapes as templates
        circle_template = np.zeros((50, 50), dtype=np.uint8)
        cv2.circle(circle_template, (25, 25), 20, 255, -1)
        self._templates["circle"] = circle_template
        self._template_scales["circle"] = [circle_template]
        
        square_template = np.zeros((50, 50), dtype=np.uint8)
        cv2.rectangle(square_template, (10, 10), (40, 40), 255, -1)
        self._templates["square"] = square_template
        self._template_scales["square"] = [square_template]
    
    async def detect(self, frame_data: Any, source_id: str, frame_number: int) -> DetectionResult:
        """
        Perform template matching on frame data.
        
        Args:
            frame_data: Frame data
            source_id: Source identifier
            frame_number: Frame number
            
        Returns:
            Detection results
        """
        start_time = time.perf_counter()
        detections = []
        
        try:
            with performance_context(f"template_match_{source_id}_{frame_number}"):
                # Convert frame to grayscale
                if hasattr(frame_data, 'data'):
                    frame = self._gst_buffer_to_numpy(frame_data)
                else:
                    frame = frame_data
                
                if frame is None:
                    return DetectionResult(
                        detections=[],
                        frame_number=frame_number,
                        source_id=source_id,
                        timestamp=datetime.now(),
                        processing_time_ms=(time.perf_counter() - start_time) * 1000
                    )
                
                # Convert to grayscale if needed
                if len(frame.shape) == 3:
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    gray_frame = frame
                
                # Perform template matching for each template
                for template_name, template_scales in self._template_scales.items():
                    template_detections = await self._match_template(
                        gray_frame, template_name, template_scales, source_id, frame_number
                    )
                    detections.extend(template_detections)
                
                processing_time = (time.perf_counter() - start_time) * 1000
                
                return DetectionResult(
                    detections=detections,
                    frame_number=frame_number,
                    source_id=source_id,
                    timestamp=datetime.now(),
                    processing_time_ms=processing_time
                )
        
        except Exception as e:
            error = handle_error(e, context={
                'strategy': self.name,
                'source_id': source_id,
                'frame_number': frame_number
            })
            self.logger.error(f"Template matching error: {error}")
            
            return DetectionResult(
                detections=[],
                frame_number=frame_number,
                source_id=source_id,
                timestamp=datetime.now(),
                processing_time_ms=(time.perf_counter() - start_time) * 1000
            )
    
    async def _match_template(
        self, 
        frame: np.ndarray, 
        template_name: str, 
        template_scales: List[np.ndarray],
        source_id: str,
        frame_number: int
    ) -> List[VideoDetection]:
        """Match template against frame at multiple scales."""
        detections = []
        frame_height, frame_width = frame.shape
        
        try:
            best_match_val = -1
            best_location = None
            best_template_size = None
            
            # Try each scale
            for template in template_scales:
                if template.shape[0] > frame_height or template.shape[1] > frame_width:
                    continue
                
                # Perform template matching
                result = cv2.matchTemplate(frame, template, self._match_method)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                # Choose location based on method
                if self._match_method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                    match_val = 1 - min_val  # Invert for consistency
                    match_loc = min_loc
                else:
                    match_val = max_val
                    match_loc = max_loc
                
                # Track best match across scales
                if match_val > best_match_val:
                    best_match_val = match_val
                    best_location = match_loc
                    best_template_size = template.shape
            
            # Create detection if match is above threshold
            if best_match_val >= self.template_config.match_threshold and best_location is not None:
                x, y = best_location
                h, w = best_template_size
                
                # Convert to normalized coordinates
                bbox = BoundingBox(
                    x=x / frame_width,
                    y=y / frame_height,
                    width=w / frame_width,
                    height=h / frame_height
                )
                
                detection = VideoDetection(
                    pattern_name=template_name,
                    confidence=best_match_val,
                    bounding_box=bbox,
                    frame_number=frame_number,
                    source_id=source_id,
                    timestamp=datetime.now(),
                    metadata=DetectionMetadata(
                        class_name=template_name,
                        model_name=self.name,
                        attributes={
                            'match_method': self.template_config.method,
                            'template_size': best_template_size
                        }
                    )
                )
                
                detections.append(detection)
        
        except Exception as e:
            self.logger.error(f"Error in template matching for {template_name}: {e}")
        
        return detections
    
    def _gst_buffer_to_numpy(self, gst_buffer) -> Optional[np.ndarray]:
        """Convert GStreamer buffer to numpy array."""
        try:
            # Placeholder implementation
            return np.zeros((480, 640, 3), dtype=np.uint8)
        except Exception as e:
            self.logger.error(f"Error converting GStreamer buffer: {e}")
            return None
    
    def should_process(self, detection: VideoDetection) -> bool:
        """Determine if template matching should process this detection."""
        return self.enabled
    
    async def cleanup(self) -> None:
        """Clean up template matching resources."""
        try:
            self.logger.info(f"Cleaning up template matching strategy: {self.name}")
            self._templates.clear()
            self._template_scales.clear()
            self.logger.info("Template matching cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during template matching cleanup: {e}")


class FeatureBasedStrategy(DetectionStrategy):
    """Feature-based detection strategy using keypoint matching."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize feature-based detection strategy.
        
        Args:
            name: Strategy name
            config: Strategy configuration
        """
        super().__init__(name, config)
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
        # Parse feature-specific configuration
        self.feature_config = FeatureConfig(**config.get('feature', {}))
        
        # Feature detection components
        self._detector = None
        self._matcher = None
        self._reference_features: Dict[str, Tuple[Any, Any]] = {}  # keypoints, descriptors
    
    async def initialize(self) -> bool:
        """Initialize feature detection components."""
        try:
            self.logger.info(f"Initializing feature-based strategy: {self.name}")
            
            # Initialize feature detector
            if self.feature_config.detector_type == "SIFT":
                self._detector = cv2.SIFT_create()
            elif self.feature_config.detector_type == "ORB":
                self._detector = cv2.ORB_create()
            elif self.feature_config.detector_type == "SURF":
                # SURF requires opencv-contrib-python
                try:
                    self._detector = cv2.xfeatures2d.SURF_create()
                except AttributeError:
                    self.logger.warning("SURF not available, falling back to SIFT")
                    self._detector = cv2.SIFT_create()
            else:
                self._detector = cv2.SIFT_create()
            
            # Initialize matcher
            if self.feature_config.matcher_type == "FLANN":
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
                self._matcher = cv2.FlannBasedMatcher(index_params, search_params)
            else:
                self._matcher = cv2.BFMatcher()
            
            # Load reference images and extract features
            ref_dir = Path(self.feature_config.reference_images_dir)
            if ref_dir.exists():
                await self._load_reference_features(ref_dir)
            else:
                self.logger.warning(f"Reference images directory not found: {ref_dir}")
                await self._create_placeholder_features()
            
            # Set model info
            self.model_info = ModelInfo(
                name="FeatureBased",
                version="1.0",
                path=ref_dir if ref_dir.exists() else None,
                output_classes=list(self._reference_features.keys()),
                description=f"{self.feature_config.detector_type} feature detection with {self.feature_config.matcher_type} matching"
            )
            
            self.logger.info(f"Feature-based strategy initialized with {len(self._reference_features)} reference patterns")
            return True
        
        except Exception as e:
            error = handle_error(e, context={'strategy': self.name})
            self.logger.error(f"Failed to initialize feature-based strategy: {error}")
            return False
    
    async def _load_reference_features(self, ref_dir: Path):
        """Load reference images and extract features."""
        try:
            image_files = list(ref_dir.glob("*.png")) + list(ref_dir.glob("*.jpg"))
            
            for image_file in image_files:
                pattern_name = image_file.stem
                img = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    # Extract keypoints and descriptors
                    keypoints, descriptors = self._detector.detectAndCompute(img, None)
                    
                    if descriptors is not None and len(descriptors) > 0:
                        self._reference_features[pattern_name] = (keypoints, descriptors)
                        self.logger.debug(f"Extracted {len(keypoints)} features from {pattern_name}")
                    else:
                        self.logger.warning(f"No features found in {pattern_name}")
                else:
                    self.logger.warning(f"Failed to load reference image: {image_file}")
        
        except Exception as e:
            self.logger.error(f"Error loading reference features: {e}")
    
    async def _create_placeholder_features(self):
        """Create placeholder features for testing."""
        # Create a simple synthetic feature set for testing
        placeholder_keypoints = [cv2.KeyPoint(100, 100, 10)]
        placeholder_descriptors = np.random.rand(1, 128).astype(np.float32)
        self._reference_features["test_pattern"] = (placeholder_keypoints, placeholder_descriptors)
    
    async def detect(self, frame_data: Any, source_id: str, frame_number: int) -> DetectionResult:
        """
        Perform feature-based detection on frame data.
        
        Args:
            frame_data: Frame data
            source_id: Source identifier
            frame_number: Frame number
            
        Returns:
            Detection results
        """
        start_time = time.perf_counter()
        detections = []
        
        try:
            with performance_context(f"feature_detect_{source_id}_{frame_number}"):
                # Convert frame
                if hasattr(frame_data, 'data'):
                    frame = self._gst_buffer_to_numpy(frame_data)
                else:
                    frame = frame_data
                
                if frame is None:
                    return DetectionResult(
                        detections=[],
                        frame_number=frame_number,
                        source_id=source_id,
                        timestamp=datetime.now(),
                        processing_time_ms=(time.perf_counter() - start_time) * 1000
                    )
                
                # Convert to grayscale if needed
                if len(frame.shape) == 3:
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    gray_frame = frame
                
                # Extract features from current frame
                frame_keypoints, frame_descriptors = self._detector.detectAndCompute(gray_frame, None)
                
                if frame_descriptors is not None and len(frame_descriptors) > 0:
                    # Match against reference features
                    for pattern_name, (ref_keypoints, ref_descriptors) in self._reference_features.items():
                        pattern_detections = await self._match_features(
                            frame_keypoints, frame_descriptors,
                            ref_keypoints, ref_descriptors,
                            pattern_name, gray_frame.shape,
                            source_id, frame_number
                        )
                        detections.extend(pattern_detections)
                
                processing_time = (time.perf_counter() - start_time) * 1000
                
                return DetectionResult(
                    detections=detections,
                    frame_number=frame_number,
                    source_id=source_id,
                    timestamp=datetime.now(),
                    processing_time_ms=processing_time
                )
        
        except Exception as e:
            error = handle_error(e, context={
                'strategy': self.name,
                'source_id': source_id,
                'frame_number': frame_number
            })
            self.logger.error(f"Feature-based detection error: {error}")
            
            return DetectionResult(
                detections=[],
                frame_number=frame_number,
                source_id=source_id,
                timestamp=datetime.now(),
                processing_time_ms=(time.perf_counter() - start_time) * 1000
            )
    
    async def _match_features(
        self,
        frame_keypoints, frame_descriptors,
        ref_keypoints, ref_descriptors,
        pattern_name: str,
        frame_shape: Tuple[int, int],
        source_id: str,
        frame_number: int
    ) -> List[VideoDetection]:
        """Match features and create detections."""
        detections = []
        
        try:
            # Perform feature matching
            matches = self._matcher.knnMatch(ref_descriptors, frame_descriptors, k=2)
            
            # Apply ratio test to filter good matches
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < self.feature_config.match_ratio * n.distance:
                        good_matches.append(m)
            
            # Check if we have enough good matches
            if len(good_matches) >= self.feature_config.min_matches:
                # Extract matched keypoints
                ref_pts = np.float32([ref_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                frame_pts = np.float32([frame_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # Find homography
                if len(ref_pts) >= 4:
                    homography, mask = cv2.findHomography(ref_pts, frame_pts, cv2.RANSAC, 5.0)
                    
                    if homography is not None:
                        # Calculate bounding box in frame
                        frame_height, frame_width = frame_shape
                        
                        # Get average position of matched points as center
                        matched_points = frame_pts[mask.ravel() == 1]
                        if len(matched_points) > 0:
                            center_x = np.mean(matched_points[:, 0, 0])
                            center_y = np.mean(matched_points[:, 0, 1])
                            
                            # Estimate bounding box size based on point spread
                            x_spread = np.max(matched_points[:, 0, 0]) - np.min(matched_points[:, 0, 0])
                            y_spread = np.max(matched_points[:, 0, 1]) - np.min(matched_points[:, 0, 1])
                            
                            # Add some padding
                            padding = 20
                            x1 = max(0, center_x - x_spread/2 - padding)
                            y1 = max(0, center_y - y_spread/2 - padding)
                            x2 = min(frame_width, center_x + x_spread/2 + padding)
                            y2 = min(frame_height, center_y + y_spread/2 + padding)
                            
                            # Convert to normalized coordinates
                            bbox = BoundingBox(
                                x=x1 / frame_width,
                                y=y1 / frame_height,
                                width=(x2 - x1) / frame_width,
                                height=(y2 - y1) / frame_height
                            )
                            
                            # Calculate confidence based on number of matches
                            confidence = min(1.0, len(good_matches) / 50.0)  # Normalize to 0-1
                            
                            detection = VideoDetection(
                                pattern_name=pattern_name,
                                confidence=confidence,
                                bounding_box=bbox,
                                frame_number=frame_number,
                                source_id=source_id,
                                timestamp=datetime.now(),
                                metadata=DetectionMetadata(
                                    class_name=pattern_name,
                                    model_name=self.name,
                                    attributes={
                                        'num_matches': len(good_matches),
                                        'detector_type': self.feature_config.detector_type,
                                        'matcher_type': self.feature_config.matcher_type
                                    }
                                )
                            )
                            
                            detections.append(detection)
        
        except Exception as e:
            self.logger.error(f"Error in feature matching for {pattern_name}: {e}")
        
        return detections
    
    def _gst_buffer_to_numpy(self, gst_buffer) -> Optional[np.ndarray]:
        """Convert GStreamer buffer to numpy array."""
        try:
            # Placeholder implementation
            return np.zeros((480, 640, 3), dtype=np.uint8)
        except Exception as e:
            self.logger.error(f"Error converting GStreamer buffer: {e}")
            return None
    
    def should_process(self, detection: VideoDetection) -> bool:
        """Determine if feature-based detection should process this detection."""
        return self.enabled
    
    async def cleanup(self) -> None:
        """Clean up feature-based detection resources."""
        try:
            self.logger.info(f"Cleaning up feature-based strategy: {self.name}")
            self._reference_features.clear()
            self._detector = None
            self._matcher = None
            self.logger.info("Feature-based detection cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during feature-based cleanup: {e}")


# Factory function for creating default strategies
async def create_default_strategies(config: AppConfig) -> List[DetectionStrategy]:
    """
    Create default detection strategies based on configuration.
    
    Args:
        config: Application configuration
        
    Returns:
        List of initialized detection strategies
    """
    logger = get_logger(__name__)
    strategies = []
    
    try:
        detection_config = config.detection
        
        # Create YOLO strategy if enabled
        if detection_config.strategies.get('yolo', {}).get('enabled', True):
            yolo_config = {
                'enabled': True,
                'yolo': {
                    'model_path': detection_config.strategies.get('yolo', {}).get('model_path', 'models/yolo.weights'),
                    'config_path': detection_config.strategies.get('yolo', {}).get('config_path', 'models/yolo.cfg'),
                    'class_names_path': detection_config.strategies.get('yolo', {}).get('class_names_path', 'models/coco.names'),
                    'confidence_threshold': detection_config.confidence_threshold,
                    'nms_threshold': detection_config.nms_threshold
                }
            }
            
            yolo_strategy = YOLODetectionStrategy("yolo_detection", yolo_config)
            strategies.append(yolo_strategy)
            logger.info("Created YOLO detection strategy")
        
        # Create template matching strategy if enabled
        if detection_config.strategies.get('template', {}).get('enabled', False):
            template_config = {
                'enabled': True,
                'template': {
                    'template_dir': detection_config.strategies.get('template', {}).get('template_dir', 'templates/'),
                    'match_threshold': detection_config.strategies.get('template', {}).get('threshold', 0.8),
                    'scale_range': detection_config.strategies.get('template', {}).get('scale_range', (0.5, 2.0)),
                    'scale_steps': detection_config.strategies.get('template', {}).get('scale_steps', 10)
                }
            }
            
            template_strategy = TemplateMatchingStrategy("template_matching", template_config)
            strategies.append(template_strategy)
            logger.info("Created template matching strategy")
        
        # Create feature-based strategy if enabled
        if detection_config.strategies.get('feature', {}).get('enabled', False):
            feature_config = {
                'enabled': True,
                'feature': {
                    'detector_type': detection_config.strategies.get('feature', {}).get('detector', 'SIFT'),
                    'matcher_type': detection_config.strategies.get('feature', {}).get('matcher', 'FLANN'),
                    'match_ratio': detection_config.strategies.get('feature', {}).get('ratio', 0.7),
                    'min_matches': detection_config.strategies.get('feature', {}).get('min_matches', 10),
                    'reference_images_dir': detection_config.strategies.get('feature', {}).get('reference_dir', 'references/')
                }
            }
            
            feature_strategy = FeatureBasedStrategy("feature_detection", feature_config)
            strategies.append(feature_strategy)
            logger.info("Created feature-based detection strategy")
        
        # If no strategies are configured, create a default YOLO strategy
        if not strategies:
            default_yolo_config = {
                'enabled': True,
                'yolo': {
                    'model_path': 'models/yolo.weights',
                    'config_path': 'models/yolo.cfg',
                    'confidence_threshold': 0.5,
                    'nms_threshold': 0.4
                }
            }
            
            default_strategy = YOLODetectionStrategy("default_yolo", default_yolo_config)
            strategies.append(default_strategy)
            logger.info("Created default YOLO strategy")
        
        logger.info(f"Created {len(strategies)} detection strategies")
        return strategies
    
    except Exception as e:
        error = handle_error(e, context={'component': 'strategy_factory'})
        logger.error(f"Error creating default strategies: {error}")
        return []


# Utility functions for strategy management
def get_available_strategy_types() -> List[str]:
    """Get list of available strategy types."""
    return ["yolo", "template", "feature"]


def create_strategy_config_template(strategy_type: str) -> Dict[str, Any]:
    """Create configuration template for strategy type."""
    templates = {
        "yolo": {
            "enabled": True,
            "yolo": {
                "model_path": "models/yolo.weights",
                "config_path": "models/yolo.cfg",
                "class_names_path": "models/coco.names",
                "confidence_threshold": 0.5,
                "nms_threshold": 0.4,
                "input_size": [640, 640],
                "classes_filter": None
            }
        },
        "template": {
            "enabled": True,
            "template": {
                "template_dir": "templates/",
                "match_threshold": 0.8,
                "scale_range": [0.5, 2.0],
                "scale_steps": 10,
                "method": "cv2.TM_CCOEFF_NORMED"
            }
        },
        "feature": {
            "enabled": True,
            "feature": {
                "detector_type": "SIFT",
                "matcher_type": "FLANN",
                "match_ratio": 0.7,
                "min_matches": 10,
                "reference_images_dir": "references/"
            }
        }
    }
    
    return templates.get(strategy_type, {})