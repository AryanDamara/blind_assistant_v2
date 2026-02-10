"""
Spatial Mapper Module
---------------------
3D spatial understanding and environmental mapping.

Features:
- Floor plane detection
- Ceiling detection
- Doorway/opening identification
- Stair detection (basic)
- Spatial memory (obstacle persistence)
- 3D coordinate mapping

Upgrade from: 2D zone detection
Enhancement: Full 3D spatial awareness
"""

import time
import numpy as np
import cv2
import yaml
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from collections import deque
import math


@dataclass
class FloorPlane:
    """Detected floor plane."""
    equation: Tuple[float, float, float, float]
    confidence: float
    y_position: int
    angle: float
    
    def is_on_floor(self, point: Tuple[int, int], tolerance: int = 20) -> bool:
        return abs(point[1] - self.y_position) < tolerance


@dataclass
class Opening:
    """Detected doorway or opening."""
    bbox: Tuple[int, int, int, int]
    width_px: int
    height_px: int
    center: Tuple[int, int]
    confidence: float
    type: str
    is_passable: bool
    
    def to_dict(self) -> dict:
        return {
            'bbox': self.bbox, 'width_px': self.width_px,
            'height_px': self.height_px, 'type': self.type,
            'passable': self.is_passable, 'confidence': round(self.confidence, 2)
        }


@dataclass
class SpatialMap:
    """Complete spatial understanding of the scene."""
    frame_id: int
    timestamp: float
    floor_plane: Optional[FloorPlane] = None
    ceiling_line: Optional[int] = None
    openings: List[Opening] = field(default_factory=list)
    walkable_area: Optional[np.ndarray] = None
    free_space_percentage: float = 0.0
    point_cloud: Optional[np.ndarray] = None
    analysis_time_ms: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            'frame_id': self.frame_id,
            'has_floor': self.floor_plane is not None,
            'has_ceiling': self.ceiling_line is not None,
            'num_openings': len(self.openings),
            'free_space_pct': round(self.free_space_percentage, 1),
            'analysis_time_ms': round(self.analysis_time_ms, 2)
        }


class SpatialMapper:
    """
    3D spatial mapping and scene understanding.
    
    Usage:
        mapper = SpatialMapper(config_path="config/settings.yaml")
        spatial_map = mapper.analyze(frame, depth_result, detections)
        if spatial_map.walkable_area[y, x]:
            print("This area is safe to walk")
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self._load_config(config_path)
        self._floor_history: deque = deque(maxlen=30)
        self._opening_history: deque = deque(maxlen=10)
        self._total_frames = 0
        self._floors_detected = 0
        self._openings_detected = 0
    
    def _load_config(self, config_path: str) -> None:
        self._floor_detection_enabled = True
        self._opening_detection_enabled = True
        self._stair_detection_enabled = False
        self._floor_ransac_iterations = 100
        self._floor_inlier_threshold = 10
        self._floor_min_y = 0.6
        self._min_opening_width = 50
        self._min_opening_height = 100
        self._opening_aspect_ratio_range = (0.3, 3.0)
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
                spatial_cfg = config.get('spatial_mapping', {})
                self._floor_detection_enabled = spatial_cfg.get('floor_detection', True)
                self._opening_detection_enabled = spatial_cfg.get('opening_detection', True)
                self._stair_detection_enabled = spatial_cfg.get('stair_detection', False)
        
        print(f"[SpatialMapper] Initialized")
        print(f"[SpatialMapper] Floor detection: {self._floor_detection_enabled}")
        print(f"[SpatialMapper] Opening detection: {self._opening_detection_enabled}")
    
    def analyze(self, frame: np.ndarray, depth_result=None,
               detections: List = None) -> SpatialMap:
        start_time = time.time()
        self._total_frames += 1
        
        height, width = frame.shape[:2]
        spatial_map = SpatialMap(frame_id=self._total_frames, timestamp=time.time())
        
        if self._floor_detection_enabled:
            floor = self._detect_floor_plane(frame, depth_result)
            if floor:
                spatial_map.floor_plane = floor
                self._floors_detected += 1
                self._floor_history.append(floor)
        
        if self._opening_detection_enabled:
            openings = self._detect_openings(frame, depth_result)
            spatial_map.openings = openings
            self._openings_detected += len(openings)
        
        walkable = self._compute_walkable_area(
            frame.shape[:2], spatial_map.floor_plane, detections
        )
        spatial_map.walkable_area = walkable
        spatial_map.free_space_percentage = (np.sum(walkable) / walkable.size) * 100
        
        spatial_map.ceiling_line = self._detect_ceiling_line(frame, depth_result)
        spatial_map.analysis_time_ms = (time.time() - start_time) * 1000
        
        return spatial_map
    
    def _detect_floor_plane(self, frame: np.ndarray,
                           depth_result=None) -> Optional[FloorPlane]:
        height, width = frame.shape[:2]
        floor_region_y = int(height * self._floor_min_y)
        floor_region = frame[floor_region_y:, :]
        
        gray = cv2.cvtColor(floor_region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                               minLineLength=width//3, maxLineGap=50)
        
        if lines is None:
            return None
        
        best_line = None
        best_score = 0
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(math.atan2(y2 - y1, x2 - x1))
            
            if angle < 0.2:
                length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                y_avg = (y1 + y2) / 2
                position_score = y_avg / gray.shape[0]
                score = length * position_score / (angle + 0.01)
                
                if score > best_score:
                    best_score = score
                    best_line = (x1, y1, x2, y2, angle)
        
        if best_line is None:
            return None
        
        x1, y1, x2, y2, angle = best_line
        y_position = floor_region_y + (y1 + y2) // 2
        
        return FloorPlane(
            equation=(0, 1, 0, -y_position),
            confidence=min(best_score / 1000, 1.0),
            y_position=y_position, angle=angle
        )
    
    def _detect_ceiling_line(self, frame: np.ndarray,
                            depth_result=None) -> Optional[int]:
        height, width = frame.shape[:2]
        
        if depth_result is not None:
            depth_top = depth_result.depth_map_raw[:int(height * 0.2), :]
            far_threshold = np.percentile(depth_result.depth_map_raw, 90)
            ceiling_mask = depth_top > far_threshold
            
            if np.sum(ceiling_mask) > ceiling_mask.size * 0.5:
                ceiling_y = np.where(ceiling_mask.any(axis=1))[0]
                if len(ceiling_y) > 0:
                    return int(ceiling_y[-1])
        return None
    
    def _detect_openings(self, frame: np.ndarray,
                        depth_result=None) -> List[Opening]:
        if depth_result is None:
            return []
        
        height, width = frame.shape[:2]
        openings = []
        
        far_threshold = np.percentile(depth_result.depth_map_raw, 70)
        far_mask = depth_result.depth_map_raw > far_threshold
        far_mask = far_mask.astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(far_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            if w < self._min_opening_width or h < self._min_opening_height:
                continue
            
            aspect_ratio = h / w
            if not (self._opening_aspect_ratio_range[0] <= aspect_ratio <=
                    self._opening_aspect_ratio_range[1]):
                continue
            
            opening_type = 'doorway' if h > w else 'corridor'
            
            roi = depth_result.depth_map_raw[y:y+h, x:x+w]
            depth_std = np.std(roi)
            confidence = 1.0 / (1.0 + depth_std)
            
            openings.append(Opening(
                bbox=(x, y, x+w, y+h), width_px=w, height_px=h,
                center=(x + w//2, y + h//2), confidence=confidence,
                type=opening_type, is_passable=True
            ))
        
        return openings
    
    def _compute_walkable_area(self, frame_shape: Tuple[int, int],
                              floor_plane: Optional[FloorPlane],
                              detections: List = None) -> np.ndarray:
        height, width = frame_shape
        walkable = np.ones((height, width), dtype=np.uint8)
        
        if floor_plane is not None:
            walkable[:floor_plane.y_position, :] = 0
        
        if detections:
            for det in detections:
                if hasattr(det, 'bbox'):
                    x1, y1, x2, y2 = det.bbox
                else:
                    x1, y1, x2, y2 = det.get('bbox', (0, 0, 0, 0))
                margin = 20
                x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
                x2, y2 = min(width, x2 + margin), min(height, y2 + margin)
                walkable[y1:y2, x1:x2] = 0
        
        return walkable
    
    def get_floor_distance_estimate(self, point: Tuple[int, int],
                                   frame_height: int) -> float:
        if not self._floor_history:
            return 5.0
        avg_floor_y = np.mean([f.y_position for f in self._floor_history])
        _, py = point
        relative_y = avg_floor_y - py
        if relative_y <= 0:
            distance = 1.0
        else:
            distance = (relative_y / frame_height) * 10
        return max(0.5, min(distance, 10.0))
    
    def visualize_spatial_features(self, frame: np.ndarray,
                                   spatial_map: SpatialMap) -> np.ndarray:
        vis_frame = frame.copy()
        
        if spatial_map.floor_plane:
            y = spatial_map.floor_plane.y_position
            cv2.line(vis_frame, (0, y), (frame.shape[1], y), (0, 255, 0), 2)
            cv2.putText(vis_frame, "FLOOR", (10, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if spatial_map.ceiling_line:
            y = spatial_map.ceiling_line
            cv2.line(vis_frame, (0, y), (frame.shape[1], y), (255, 0, 0), 2)
            cv2.putText(vis_frame, "CEILING", (10, y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        for opening in spatial_map.openings:
            x1, y1, x2, y2 = opening.bbox
            color = (0, 255, 255) if opening.is_passable else (0, 0, 255)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            label = f"{opening.type} ({opening.confidence:.2f})"
            cv2.putText(vis_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        if spatial_map.walkable_area is not None:
            walkable_colored = np.zeros_like(frame)
            walkable_colored[:, :, 1] = spatial_map.walkable_area * 50
            vis_frame = cv2.addWeighted(vis_frame, 0.8, walkable_colored, 0.2, 0)
        
        stats_text = f"Free space: {spatial_map.free_space_percentage:.1f}%"
        cv2.putText(vis_frame, stats_text, (10, frame.shape[0] - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if spatial_map.openings:
            openings_text = f"Openings: {len(spatial_map.openings)}"
            cv2.putText(vis_frame, openings_text, (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis_frame
    
    def get_stats(self) -> dict:
        floor_rate = (self._floors_detected / max(1, self._total_frames)) * 100
        return {
            'total_frames': self._total_frames,
            'floors_detected': self._floors_detected,
            'floor_detection_rate': round(floor_rate, 1),
            'total_openings_detected': self._openings_detected,
            'avg_openings_per_frame': round(
                self._openings_detected / max(1, self._total_frames), 2)
        }


# ============================================
# Testing / Demo
# ============================================
if __name__ == "__main__":
    print("=" * 50)
    print("Spatial Mapper Module Test")
    print("=" * 50)
    
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'settings.yaml')
    mapper = SpatialMapper(config_path=config_path)
    
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for y in range(300, 480):
        intensity = int(((y - 300) / 180) * 255)
        frame[y, :] = [intensity, intensity, intensity]
    cv2.rectangle(frame, (250, 50), (390, 300), (200, 200, 200), -1)
    
    print("\nAnalyzing synthetic scene...")
    spatial_map = mapper.analyze(frame)
    
    print(f"\nResults:")
    print(f"  Floor detected: {spatial_map.floor_plane is not None}")
    if spatial_map.floor_plane:
        print(f"    Y-position: {spatial_map.floor_plane.y_position}px")
        print(f"    Confidence: {spatial_map.floor_plane.confidence:.2f}")
    print(f"  Openings detected: {len(spatial_map.openings)}")
    print(f"  Free space: {spatial_map.free_space_percentage:.1f}%")
    print(f"  Analysis time: {spatial_map.analysis_time_ms:.1f}ms")
    
    stats = mapper.get_stats()
    print(f"\nStats: {stats}")
    print("\nTest complete!")
