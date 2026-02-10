"""
Depth Estimator Module
----------------------
Monocular depth estimation using MiDAS for accurate distance measurement.

Features:
- MiDAS v3.1 (DPT-Large or Small models)
- Real-time depth map generation
- Depth-to-distance calibration
- Per-object distance extraction
- Confidence scoring
- Optional stereo depth (future)

Upgrade from: Simple bounding box height estimation
Enhancement: Pixel-level depth understanding
"""

import time
import yaml
import os
import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

try:
    import torch
    from torchvision import transforms
except ImportError:
    print("[DepthEstimator] ERROR: torch not installed. Run: pip install torch torchvision")
    raise


@dataclass
class DepthResult:
    """
    Depth estimation result for a frame.
    """
    frame_id: int
    depth_map: np.ndarray          # Normalized depth map (0-255)
    depth_map_raw: np.ndarray      # Raw depth values
    inference_time_ms: float
    timestamp: float
    calibration_scale: float = 1.0
    
    def get_depth_at_point(self, x: int, y: int) -> float:
        """Get depth value at specific pixel."""
        if 0 <= y < self.depth_map_raw.shape[0] and 0 <= x < self.depth_map_raw.shape[1]:
            return float(self.depth_map_raw[y, x] * self.calibration_scale)
        return 0.0
    
    def get_depth_in_bbox(self, bbox: Tuple[int, int, int, int],
                          method: str = "median") -> float:
        """
        Get representative depth value within bounding box.
        
        Args:
            bbox: (x1, y1, x2, y2)
            method: 'median', 'mean', 'min', 'center'
        """
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(self.depth_map_raw.shape[1], x2)
        y2 = min(self.depth_map_raw.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        roi = self.depth_map_raw[y1:y2, x1:x2]
        
        if method == "median":
            depth = float(np.median(roi))
        elif method == "mean":
            depth = float(np.mean(roi))
        elif method == "min":
            depth = float(np.min(roi))
        elif method == "center":
            cy, cx = roi.shape[0] // 2, roi.shape[1] // 2
            depth = float(roi[cy, cx])
        else:
            depth = float(np.median(roi))
        
        return depth * self.calibration_scale
    
    def to_dict(self) -> dict:
        """Convert to dict for telemetry."""
        return {
            'frame_id': self.frame_id,
            'inference_time_ms': round(self.inference_time_ms, 2),
            'timestamp': self.timestamp,
            'depth_range': [float(self.depth_map_raw.min()), 
                           float(self.depth_map_raw.max())],
            'calibration_scale': self.calibration_scale
        }


class DepthEstimator:
    """
    Monocular depth estimation using MiDAS.
    
    Usage:
        estimator = DepthEstimator(config_path="config/settings.yaml")
        depth_result = estimator.estimate(frame, frame_id=1)
        
        # Get distance for specific object
        distance = depth_result.get_depth_in_bbox(detection.bbox)
    """
    
    # Model options
    MODELS = {
        'midas_small': 'MiDAS_small',      # Fastest, least accurate
        'dpt_hybrid': 'DPT_Hybrid',        # Balanced
        'dpt_large': 'DPT_Large'           # Best, slowest
    }
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize depth estimator with configuration."""
        self._load_config(config_path)
        
        # Model (loaded in _init_model)
        self._model = None
        self._transform = None
        self._device = None
        
        # Calibration
        self._calibration_scale = 1.0
        self._calibration_offset = 0.0
        
        # Statistics
        self._total_inferences = 0
        self._total_inference_time = 0.0
        self._max_inference_time = 0.0
        
        # Load model at initialization
        self._init_model()
    
    def _load_config(self, config_path: str) -> None:
        """Load depth estimation settings from YAML config."""
        # Defaults
        self._enabled = True
        self._model_name = 'dpt_hybrid'
        self._use_gpu = torch.cuda.is_available()
        self._depth_method = 'median'  # For bbox depth extraction
        self._max_depth_m = 10.0
        self._min_depth_m = 0.3
        
        # Calibration defaults
        self._auto_calibrate = True
        self._calibration_distance_m = 2.0
        self._calibration_object_class = 'person'
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
                
                depth_cfg = config.get('depth', {})
                self._enabled = depth_cfg.get('enabled', True)
                self._model_name = depth_cfg.get('model', 'dpt_hybrid')
                self._use_gpu = depth_cfg.get('use_gpu', torch.cuda.is_available())
                self._depth_method = depth_cfg.get('bbox_method', 'median')
                self._max_depth_m = depth_cfg.get('max_depth_m', 10.0)
                self._min_depth_m = depth_cfg.get('min_depth_m', 0.3)
                
                calib_cfg = depth_cfg.get('calibration', {})
                self._auto_calibrate = calib_cfg.get('auto_calibrate', True)
                self._calibration_distance_m = calib_cfg.get('reference_distance_m', 2.0)
                self._calibration_object_class = calib_cfg.get('reference_object', 'person')
        
        print(f"[DepthEstimator] Initialized")
        print(f"[DepthEstimator] Model: {self._model_name}")
        print(f"[DepthEstimator] Device: {'GPU' if self._use_gpu else 'CPU'}")
    
    def _init_model(self) -> None:
        """Initialize MiDAS model."""
        if not self._enabled:
            print("[DepthEstimator] Depth estimation disabled")
            return
        
        print(f"[DepthEstimator] Loading MiDAS model: {self._model_name}...")
        start_time = time.time()
        
        try:
            # Set device
            self._device = torch.device('cuda' if self._use_gpu else 'cpu')
            
            # Load MiDAS model from torch hub
            model_type = self.MODELS.get(self._model_name, 'DPT_Hybrid')
            self._model = torch.hub.load('intel-isl/MiDAS', model_type, 
                                         pretrained=True, trust_repo=True)
            self._model.to(self._device)
            self._model.eval()
            
            # Load transforms
            midas_transforms = torch.hub.load('intel-isl/MiDAS', 'transforms', 
                                             trust_repo=True)
            
            if 'DPT' in model_type:
                self._transform = midas_transforms.dpt_transform
            else:
                self._transform = midas_transforms.small_transform
            
            load_time = (time.time() - start_time) * 1000
            print(f"[DepthEstimator] Model loaded ({load_time:.0f}ms)")
            
            # Warmup
            print("[DepthEstimator] Running warmup inference...")
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            _ = self.estimate(dummy_frame, frame_id=0)
            print("[DepthEstimator] Warmup complete")
            
        except Exception as e:
            print(f"[DepthEstimator] ERROR loading model: {e}")
            print("[DepthEstimator] Falling back to bbox-based distance estimation")
            self._enabled = False
    
    def estimate(self, frame: np.ndarray, frame_id: int = 0) -> Optional[DepthResult]:
        """
        Estimate depth map for frame.
        
        Args:
            frame: BGR image (OpenCV format)
            frame_id: Frame identifier
            
        Returns:
            DepthResult or None if error
        """
        if not self._enabled or self._model is None:
            return None
        
        start_time = time.time()
        
        try:
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Prepare input
            input_batch = self._transform(img_rgb).to(self._device)
            
            # Inference
            with torch.no_grad():
                prediction = self._model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img_rgb.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            
            # Convert to numpy
            depth_raw = prediction.cpu().numpy()
            
            # Normalize for visualization (0-255)
            depth_normalized = cv2.normalize(depth_raw, None, 0, 255, 
                                            cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # Apply calibration scale
            # MiDAS outputs inverse depth, we convert to approximate distance
            depth_raw = self._inverse_depth_to_distance(depth_raw)
            
            inference_time_ms = (time.time() - start_time) * 1000
            
            # Update statistics
            self._total_inferences += 1
            self._total_inference_time += inference_time_ms
            self._max_inference_time = max(self._max_inference_time, inference_time_ms)
            
            return DepthResult(
                frame_id=frame_id,
                depth_map=depth_normalized,
                depth_map_raw=depth_raw,
                inference_time_ms=inference_time_ms,
                timestamp=time.time(),
                calibration_scale=self._calibration_scale
            )
            
        except Exception as e:
            print(f"[DepthEstimator] ERROR during inference: {e}")
            return None
    
    def _inverse_depth_to_distance(self, inverse_depth: np.ndarray) -> np.ndarray:
        """
        Convert MiDAS inverse depth to approximate distance.
        
        MiDAS outputs relative inverse depth. We convert to meters using calibration.
        """
        # Avoid division by zero
        inverse_depth = np.maximum(inverse_depth, 1e-6)
        
        # Convert inverse depth to distance (rough approximation)
        # This needs calibration for accuracy
        distance = 1.0 / inverse_depth
        
        # Normalize to reasonable range
        distance = np.clip(distance, self._min_depth_m, self._max_depth_m)
        
        return distance
    
    def calibrate(self, frame: np.ndarray, bbox: Tuple[int, int, int, int],
                  true_distance_m: float) -> float:
        """
        Calibrate depth estimation using known object distance.
        
        Args:
            frame: Frame containing calibration object
            bbox: Bounding box of calibration object
            true_distance_m: Actual measured distance to object
            
        Returns:
            Calibration scale factor
        """
        depth_result = self.estimate(frame)
        
        if depth_result is None:
            print("[DepthEstimator] Calibration failed: no depth result")
            return 1.0
        
        estimated_depth = depth_result.get_depth_in_bbox(bbox, method=self._depth_method)
        
        if estimated_depth > 0:
            self._calibration_scale = true_distance_m / estimated_depth
            self._calibration_offset = 0.0
            
            print(f"[DepthEstimator] Calibration complete:")
            print(f"  Estimated: {estimated_depth:.2f}m")
            print(f"  Actual: {true_distance_m:.2f}m")
            print(f"  Scale: {self._calibration_scale:.3f}")
            
            return self._calibration_scale
        
        print("[DepthEstimator] Calibration failed: invalid depth")
        return 1.0
    
    def visualize_depth(self, depth_result: DepthResult, 
                       colormap: int = cv2.COLORMAP_MAGMA) -> np.ndarray:
        """
        Create colored depth visualization.
        
        Args:
            depth_result: DepthResult from estimate()
            colormap: OpenCV colormap for visualization
            
        Returns:
            Colored depth map (BGR)
        """
        depth_colored = cv2.applyColorMap(depth_result.depth_map, colormap)
        return depth_colored
    
    def get_stats(self) -> dict:
        """Get depth estimator statistics."""
        avg_time = (self._total_inference_time / max(1, self._total_inferences))
        
        return {
            'total_inferences': self._total_inferences,
            'avg_inference_time_ms': round(avg_time, 2),
            'max_inference_time_ms': round(self._max_inference_time, 2),
            'enabled': self._enabled,
            'model': self._model_name,
            'device': 'GPU' if self._use_gpu else 'CPU',
            'calibration_scale': round(self._calibration_scale, 3)
        }
    
    @property
    def is_enabled(self) -> bool:
        """Check if depth estimation is enabled and working."""
        return self._enabled and self._model is not None


# ============================================
# Testing / Demo
# ============================================
if __name__ == "__main__":
    """Test depth estimator module standalone."""
    import sys
    
    print("=" * 50)
    print("Depth Estimator Module Test")
    print("=" * 50)
    
    # Initialize modules
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'settings.yaml')
    estimator = DepthEstimator(config_path=config_path)
    
    if not estimator.is_enabled:
        print("\nDepth estimation not available. Exiting.")
        sys.exit(0)
    
    # Test with synthetic frame
    print("\nTesting with synthetic frame...")
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = estimator.estimate(test_frame, frame_id=1)
    
    if result:
        print(f"  Inference time: {result.inference_time_ms:.0f}ms")
        print(f"  Depth range: {result.depth_map_raw.min():.2f} - {result.depth_map_raw.max():.2f}")
        print(f"  Center depth: {result.get_depth_at_point(320, 240):.2f}m")
        
        # Test bbox depth
        bbox_depth = result.get_depth_in_bbox((200, 200, 400, 400))
        print(f"  Bbox depth (median): {bbox_depth:.2f}m")
    
    # Print stats
    stats = estimator.get_stats()
    print(f"\n[DepthEstimator] Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nTest complete!")
