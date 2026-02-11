"""
Object Detector Module
----------------------
YOLOv8-based object detection for navigation assistance.

Features:
- Synchronous detection (called from main thread)
- Model loaded at initialization with warmup (fail fast)
- Returns all detections (filtering done by safety_manager)
- Latency tracking for telemetry
- Input validation and error handling
- Visualization helper
"""

import time
import yaml
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import cv2

try:
    from ultralytics import YOLO
except ImportError:
    print("[ObjectDetector] ERROR: ultralytics not installed. Run: pip install ultralytics")
    raise


@dataclass
class Detection:
    """
    Single object detection result.
    Zone and distance are NOT included - those are safety concepts.
    """
    class_name: str               # Object class (e.g., 'person', 'chair')
    confidence: float             # Detection confidence (0.0 - 1.0)
    bbox: Tuple[int, int, int, int]  # Bounding box (x1, y1, x2, y2)
    bbox_center: Tuple[int, int]  # Center point (x, y)
    bbox_area: int                # Area in pixels
    bbox_height: int              # Height in pixels (for distance estimation)
    bbox_width: int               # Width in pixels
    
    def to_dict(self) -> dict:
        """Convert to dict for telemetry logging."""
        return {
            'class': self.class_name,
            'confidence': round(self.confidence, 3),
            'bbox': list(self.bbox),
            'bbox_center': list(self.bbox_center),
            'bbox_area': self.bbox_area,
            'bbox_height': self.bbox_height,
            'bbox_width': self.bbox_width
        }


@dataclass
class DetectionResult:
    """
    Complete detection result for a single frame.
    """
    frame_id: int                 # Frame this detection belongs to
    detections: List[Detection]   # List of detected objects
    inference_time_ms: float      # YOLO inference time
    timestamp: float              # When detection was performed
    
    @property
    def count(self) -> int:
        """Number of objects detected."""
        return len(self.detections)
    
    def to_dict(self) -> dict:
        """Convert to dict for telemetry logging."""
        return {
            'frame_id': self.frame_id,
            'detection_count': self.count,
            'inference_time_ms': round(self.inference_time_ms, 2),
            'timestamp': self.timestamp,
            'detections': [d.to_dict() for d in self.detections]
        }


class ObjectDetector:
    """
    YOLOv8-based object detector.
    
    Usage:
        detector = ObjectDetector(config_path="config/settings.yaml")
        result = detector.detect(frame, frame_id=1)
        for detection in result.detections:
            print(f"{detection.class_name}: {detection.confidence:.2f}")
    """
    
    # Performance threshold for warnings (ms)
    SLOW_INFERENCE_THRESHOLD_MS = 200
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """
        Initialize detector and load YOLO model.
        Model is loaded immediately with warmup (fail fast if missing).
        """
        # Load configuration
        self._load_config(config_path)
        
        # Load model at init
        self._load_model()
        
        # Statistics
        self._total_detections: int = 0
        self._total_frames: int = 0
        self._total_inference_time: float = 0.0
        self._max_inference_time: float = 0.0
        self._slow_inference_count: int = 0
    
    def _load_config(self, config_path: str) -> None:
        """Load YOLO settings from YAML config."""
        # Defaults
        self._model_path: str = "yolov8n.pt"
        self._confidence_threshold: float = 0.5
        self._iou_threshold: float = 0.45
        self._device: str = "cpu"
        self._max_detections: int = 10
        self._img_size: int = 640
        self._allowed_classes: List[str] = []  # Empty = all classes
        
        # Load from file if exists
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                yolo_cfg = config.get('yolo', {})
                
                self._model_path = yolo_cfg.get('model_path', "yolov8n.pt")
                self._confidence_threshold = yolo_cfg.get('confidence_threshold', 0.5)
                self._iou_threshold = yolo_cfg.get('iou_threshold', 0.45)
                self._device = yolo_cfg.get('device', "cpu")
                self._max_detections = yolo_cfg.get('max_detections', 10)
                self._img_size = yolo_cfg.get('img_size', 640)
                self._allowed_classes = yolo_cfg.get('allowed_classes', [])
    
    def _load_model(self) -> None:
        """Load YOLO model with warmup. Called once at initialization."""
        print(f"[ObjectDetector] Loading YOLOv8 model: {self._model_path}...")
        start_time = time.time()
        
        try:
            self._model = YOLO(self._model_path)
            
            # Get class names from model
            self._class_names = self._model.names
            
            load_time = (time.time() - start_time) * 1000
            print(f"[ObjectDetector] Model loaded ({load_time:.0f}ms)")
            
            # Warmup inference (first inference is always slow)
            print(f"[ObjectDetector] Running warmup inference...")
            warmup_start = time.time()
            
            # Create dummy image matching expected input size
            dummy_frame = np.zeros((self._img_size, self._img_size, 3), dtype=np.uint8)
            _ = self._model(
                dummy_frame,
                conf=self._confidence_threshold,
                device=self._device,
                verbose=False
            )
            
            warmup_time = (time.time() - warmup_start) * 1000
            print(f"[ObjectDetector] Warmup complete ({warmup_time:.0f}ms)")
            
            print(f"[ObjectDetector] Device: {self._device}")
            print(f"[ObjectDetector] Classes available: {len(self._class_names)}")
            print(f"[ObjectDetector] Confidence threshold: {self._confidence_threshold}")
            
        except Exception as e:
            print(f"[ObjectDetector] ERROR: Failed to load model: {e}")
            raise RuntimeError(f"Failed to load YOLO model: {e}")
    
    def detect(self, frame: np.ndarray, frame_id: int = 0) -> DetectionResult:
        """
        Run object detection on a frame.
        
        Args:
            frame: BGR image as numpy array
            frame_id: Frame identifier for tracking
            
        Returns:
            DetectionResult containing all detections (empty if error)
        """
        start_time = time.time()
        
        # Input validation
        if frame is None or frame.size == 0:
            print(f"[ObjectDetector] WARNING: Invalid frame {frame_id}")
            return DetectionResult(
                frame_id=frame_id,
                detections=[],
                inference_time_ms=0.0,
                timestamp=time.time()
            )
        
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            print(f"[ObjectDetector] WARNING: Frame {frame_id} has wrong shape: {frame.shape}")
            return DetectionResult(
                frame_id=frame_id,
                detections=[],
                inference_time_ms=0.0,
                timestamp=time.time()
            )
        
        try:
            # Run inference
            results = self._model(
                frame,
                conf=self._confidence_threshold,
                iou=self._iou_threshold,
                device=self._device,
                imgsz=self._img_size,
                verbose=False,
                max_det=self._max_detections
            )
            
            inference_time_ms = (time.time() - start_time) * 1000
            
            # Parse results
            detections: List[Detection] = []
            
            for result in results:
                boxes = result.boxes
                
                if boxes is None or len(boxes) == 0:
                    continue
                
                for i in range(len(boxes)):
                    # Get bounding box
                    box = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Get confidence and class
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    class_name = self._class_names[cls_id]
                    
                    # Apply allowed_classes filter if specified
                    if self._allowed_classes and class_name not in self._allowed_classes:
                        continue
                    
                    # Calculate derived values
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    bbox_center = (x1 + bbox_width // 2, y1 + bbox_height // 2)
                    bbox_area = bbox_width * bbox_height
                    
                    detection = Detection(
                        class_name=class_name,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                        bbox_center=bbox_center,
                        bbox_area=bbox_area,
                        bbox_height=bbox_height,
                        bbox_width=bbox_width
                    )
                    detections.append(detection)
            
            # Sort by confidence (highest first)
            detections.sort(key=lambda d: d.confidence, reverse=True)
            
            # Update statistics
            self._total_frames += 1
            self._total_detections += len(detections)
            self._total_inference_time += inference_time_ms
            self._max_inference_time = max(self._max_inference_time, inference_time_ms)
            
            # Performance warning
            if inference_time_ms > self.SLOW_INFERENCE_THRESHOLD_MS:
                self._slow_inference_count += 1
                print(f"[ObjectDetector] ⚠️  Slow inference: {inference_time_ms:.0f}ms "
                      f"(frame {frame_id})")
            
            return DetectionResult(
                frame_id=frame_id,
                detections=detections,
                inference_time_ms=inference_time_ms,
                timestamp=time.time()
            )
        
        except Exception as e:
            # If inference fails, return empty result
            inference_time_ms = (time.time() - start_time) * 1000
            print(f"[ObjectDetector] ERROR during inference: {e}")
            
            return DetectionResult(
                frame_id=frame_id,
                detections=[],
                inference_time_ms=inference_time_ms,
                timestamp=time.time()
            )
    
    def visualize(self, frame: np.ndarray, result: DetectionResult,
                  show_confidence: bool = True) -> np.ndarray:
        """
        Draw detections on frame for visualization.
        
        Args:
            frame: Original frame (BGR)
            result: Detection result
            show_confidence: Whether to show confidence scores
            
        Returns:
            Frame copy with drawn detections
        """
        display_frame = frame.copy()
        
        for det in result.detections:
            x1, y1, x2, y2 = det.bbox
            
            # Draw bounding box
            color = (0, 255, 0)  # Green
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            if show_confidence:
                label = f"{det.class_name} {det.confidence:.2f}"
            else:
                label = det.class_name
                
            cv2.putText(
                display_frame, label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2
            )
            
            # Draw center point
            cv2.circle(display_frame, det.bbox_center, 5, (0, 0, 255), -1)
        
        return display_frame
    
    def get_stats(self) -> dict:
        """Get detector statistics for monitoring."""
        avg_inference_time = (
            self._total_inference_time / max(1, self._total_frames)
        )
        avg_detections = (
            self._total_detections / max(1, self._total_frames)
        )
        
        return {
            'total_frames_processed': self._total_frames,
            'total_detections': self._total_detections,
            'avg_detections_per_frame': round(avg_detections, 2),
            'avg_inference_time_ms': round(avg_inference_time, 2),
            'max_inference_time_ms': round(self._max_inference_time, 2),
            'slow_inference_count': self._slow_inference_count
        }
    
    @property
    def class_names(self) -> dict:
        """Get available class names from model."""
        return self._class_names
    
    @property
    def confidence_threshold(self) -> float:
        """Get current confidence threshold."""
        return self._confidence_threshold


# ============================================
# Testing / Demo
# ============================================
if __name__ == "__main__":
    """Test object detector module standalone."""
    from camera_capture import CameraCapture
    
    print("=" * 50)
    print("Object Detector Module Test")
    print("=" * 50)
    
    # Initialize detector
    detector = ObjectDetector(config_path="config/settings.yaml")
    
    # Use camera for testing
    with CameraCapture(config_path="config/settings.yaml") as camera:
        print("\nPress 'q' to quit\n")
        
        frame_count = 0
        
        while True:
            # Get frame
            frame_packet = camera.get_frame(timeout=1.0)
            
            if frame_packet is None:
                print("No frame received")
                continue
            
            frame_count += 1
            
            # Skip frames based on frame_skip setting
            if frame_count % camera.frame_skip != 0:
                # Still show frame but don't run detection
                cv2.imshow("Object Detector Test", frame_packet.data)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            # Run detection
            result = detector.detect(frame_packet.data, frame_id=frame_packet.frame_id)
            
            # Visualize detections
            display_frame = detector.visualize(frame_packet.data, result)
            
            # Draw stats overlay
            stats_text = f"Detections: {result.count} | Inference: {result.inference_time_ms:.1f}ms"
            cv2.putText(
                display_frame, stats_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2
            )
            
            # Show frame
            cv2.imshow("Object Detector Test", display_frame)
            
            # Print to console
            if result.count > 0:
                objects = [f"{d.class_name}({d.confidence:.2f})" for d in result.detections[:3]]
                print(f"Frame {result.frame_id}: {', '.join(objects)}")
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Print final stats
        stats = detector.get_stats()
        print(f"\n[ObjectDetector] Final Statistics:")
        print(f"  Frames processed: {stats['total_frames_processed']}")
        print(f"  Total detections: {stats['total_detections']}")
        print(f"  Avg detections/frame: {stats['avg_detections_per_frame']}")
        print(f"  Avg inference time: {stats['avg_inference_time_ms']:.1f}ms")
        print(f"  Max inference time: {stats['max_inference_time_ms']:.1f}ms")
        print(f"  Slow inference count: {stats['slow_inference_count']}")
    
    cv2.destroyAllWindows()
    print("\nTest complete!")
