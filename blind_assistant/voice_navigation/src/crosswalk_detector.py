"""
Crosswalk / Zebra Crossing Detector Module
-------------------------------------------
Detects pedestrian crosswalks using line pattern recognition.

Features:
- Canny edge detection + Hough line transform
- Parallel line grouping to identify zebra stripe patterns
- Temporal consistency tracking (multi-frame validation)
- Context validation (brightness, orientation checks)
- Improved distance estimation via line-point sampling
- Orientation and position classification

Usage:
    detector = CrosswalkDetector(config_path="config/settings.yaml")
    result = detector.detect(rgb_frame, depth_map)

    if result.detected:
        audio.speak(result.get_announcement(), priority=4)
"""

import time
import yaml
import os
import numpy as np
import cv2
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


# ============================================================
# Data Class
# ============================================================

@dataclass
class CrosswalkAlert:
    """Alert for detected crosswalk."""
    detected: bool
    crosswalk_type: str = 'unknown'  # 'zebra', 'painted', 'unmarked'
    position: str = 'unknown'        # 'ahead', 'left', 'right'
    distance_m: float = 0.0
    orientation: float = 0.0         # Degrees
    width_m: float = 0.0
    has_traffic_island: bool = False
    confidence: float = 0.0
    temporal_consistency: int = 0    # Number of consecutive frames detected

    analysis_time_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def get_announcement(self) -> str:
        if not self.detected:
            return ""
        if self.distance_m > 0 and self.distance_m < 2.0:
            return f"Crosswalk directly {self.position} - {int(self.distance_m * 100)} centimeters"
        elif self.distance_m >= 2.0:
            return f"Crosswalk {self.position} at {self.distance_m:.1f} meters"
        return f"Crosswalk {self.position}"

    def to_dict(self) -> dict:
        return {
            'detected': self.detected,
            'crosswalk_type': self.crosswalk_type,
            'position': self.position,
            'distance_m': round(self.distance_m, 2),
            'orientation': round(self.orientation, 1),
            'confidence': round(self.confidence, 3),
            'temporal_consistency': self.temporal_consistency,
            'analysis_time_ms': round(self.analysis_time_ms, 2),
        }


# ============================================================
# Crosswalk Detector
# ============================================================

class CrosswalkDetector:
    """
    Detects pedestrian crosswalks via parallel stripe detection.

    Pipeline:
    1. Convert to greyscale, focus on lower half (ground level)
    2. Canny edge detection
    3. Hough line transform
    4. Group lines by angle → find parallel sets
    5. Context validation (brightness, orientation)
    6. Temporal consistency filtering
    7. Largest parallel set with ≥4 lines = crosswalk
    """

    def __init__(self, config_path: str = "config/settings.yaml"):
        self._enabled = True
        self._min_confidence: float = 0.6
        self._min_parallel_lines: int = 4
        self._max_line_angle_var: float = 15.0  # degrees
        self._cooldown_sec: float = 3.0
        self._last_alert_time: float = 0.0

        # Temporal tracking
        self._detection_history: deque = deque(maxlen=5)
        self._min_temporal_consistency: int = 2
        self._use_temporal_tracking: bool = True

        # Context validation
        self._validate_context: bool = True

        # Stats
        self._total_analyses: int = 0
        self._total_detections: int = 0

        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    cfg = yaml.safe_load(f) or {}
                cc = cfg.get('crosswalk_detection', {})
                self._enabled = cc.get('enabled', True)
                self._min_confidence = cc.get('min_confidence', 0.6)
                self._min_parallel_lines = cc.get('min_parallel_lines', 4)
                self._cooldown_sec = cc.get('cooldown_sec', 3.0)
                self._use_temporal_tracking = cc.get('use_temporal_tracking', True)
                self._min_temporal_consistency = cc.get('min_temporal_consistency', 2)
                self._validate_context = cc.get('validate_context', True)
            except Exception as e:
                print(f"[CrosswalkDetector] WARNING: Config error: {e}")

        print(f"[CrosswalkDetector] Initialized (enabled={self._enabled})")

    # ---------------------------------------------------------------- detect
    def detect(self, rgb_frame: np.ndarray,
               depth_map: Optional[np.ndarray] = None) -> CrosswalkAlert:
        """Detect crosswalk in frame."""
        start = time.time()
        self._total_analyses += 1

        if not self._enabled:
            return CrosswalkAlert(detected=False,
                                 analysis_time_ms=(time.time() - start) * 1000)
        if time.time() - self._last_alert_time < self._cooldown_sec:
            return CrosswalkAlert(detected=False,
                                 analysis_time_ms=(time.time() - start) * 1000)
        if rgb_frame is None or rgb_frame.size == 0:
            return CrosswalkAlert(detected=False,
                                 analysis_time_ms=(time.time() - start) * 1000)

        try:
            gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            # Focus on lower half (ground level)
            roi = gray[h // 2:, :]
            edges = cv2.Canny(roi, 50, 150, apertureSize=3)

            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100,
                                    minLineLength=50, maxLineGap=10)
            if lines is None or len(lines) < self._min_parallel_lines:
                self._detection_history.append(False)
                return CrosswalkAlert(detected=False,
                                     analysis_time_ms=(time.time() - start) * 1000)

            groups = self._find_parallel_lines(lines)
            if not groups:
                self._detection_history.append(False)
                return CrosswalkAlert(detected=False,
                                     analysis_time_ms=(time.time() - start) * 1000)

            best = max(groups, key=len)
            if len(best) < self._min_parallel_lines:
                self._detection_history.append(False)
                return CrosswalkAlert(detected=False,
                                     analysis_time_ms=(time.time() - start) * 1000)

            # Context validation: check brightness and orientation
            if self._validate_context:
                if not self._validate_crosswalk_context(best, rgb_frame):
                    self._detection_history.append(False)
                    return CrosswalkAlert(detected=False,
                                         analysis_time_ms=(time.time() - start) * 1000)

            orientation = self._calculate_orientation(best)
            position = self._calculate_position(best, w)
            distance = self._estimate_distance(depth_map, best) if depth_map is not None else 0.0
            confidence = min(1.0, len(best) / 10.0)

            # Temporal consistency filtering
            self._detection_history.append(True)
            recent_detections = sum(self._detection_history)

            if self._use_temporal_tracking and recent_detections < self._min_temporal_consistency:
                # Not enough consecutive frames — suppress but record
                return CrosswalkAlert(
                    detected=False, confidence=confidence * 0.5,
                    temporal_consistency=recent_detections,
                    analysis_time_ms=(time.time() - start) * 1000,
                )

            result = CrosswalkAlert(
                detected=True, crosswalk_type='zebra', position=position,
                distance_m=distance, orientation=orientation, confidence=confidence,
                temporal_consistency=recent_detections,
                analysis_time_ms=(time.time() - start) * 1000,
            )
            self._total_detections += 1
            self._last_alert_time = time.time()
            return result

        except Exception as e:
            print(f"[CrosswalkDetector] ERROR: {e}")
            return CrosswalkAlert(detected=False,
                                 analysis_time_ms=(time.time() - start) * 1000)

    # -------------------------------------------------------- helpers
    def _find_parallel_lines(self, lines) -> List[List]:
        """Group lines by similar angle."""
        groups: List[List] = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

            found = False
            for g in groups:
                if abs(angle - g[0]) < self._max_line_angle_var:
                    g[1].append(line)
                    found = True
                    break
            if not found:
                groups.append([angle, [line]])

        return [g[1] for g in groups if len(g[1]) >= self._min_parallel_lines]

    def _calculate_orientation(self, lines) -> float:
        """Calculate average orientation of crosswalk lines (safe from empty)."""
        if not lines:
            return 0.0
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angles.append(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        return float(np.mean(angles)) if angles else 0.0

    def _calculate_position(self, lines, frame_width: int) -> str:
        """Classify crosswalk position (safe from empty)."""
        if not lines:
            return 'unknown'
        xs = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            xs.extend([x1, x2])
        if not xs:
            return 'unknown'
        avg = np.mean(xs)
        if avg < frame_width * 0.3:
            return 'left'
        if avg > frame_width * 0.7:
            return 'right'
        return 'ahead'

    def _estimate_distance(self, depth_map, lines) -> float:
        """Estimate distance to nearest stripe edge via multi-point sampling."""
        if depth_map is None or not lines:
            return 0.0
        try:
            min_distances = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Sample 5 points along each line
                for i in range(5):
                    t = i / max(1, 4)
                    x = int(x1 + t * (x2 - x1))
                    y = int(y1 + t * (y2 - y1))
                    if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
                        depth = depth_map[y, x]
                        if depth > 0:
                            min_distances.append(depth)

            if min_distances:
                # 10th percentile: close to minimum but robust to outliers
                distance = float(np.percentile(min_distances, 10))
                return max(1.0, min(distance, 20.0))
            return 0.0
        except Exception as e:
            print(f"[CrosswalkDetector] Distance estimation error: {e}")
            return 0.0

    def _validate_crosswalk_context(self, lines, rgb_frame) -> bool:
        """Validate that detected lines are a real crosswalk (not fence, shirt, etc.)."""
        if not lines or rgb_frame is None or rgb_frame.size == 0:
            return False
        try:
            # 1. Check orientation — zebra stripes should be roughly horizontal
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                angles.append(angle)
            if not angles:
                return False
            avg_angle = np.mean(angles)
            # Perpendicular stripes: ~0° or ~180° (horizontal in image)
            if not (abs(avg_angle) < 30 or abs(180 - abs(avg_angle)) < 30):
                return False

            # 2. Check brightness — crosswalk paint is white / bright
            gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            roi = gray[h // 2:, :]

            brightness_samples = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                y1c = max(0, min(roi.shape[0] - 1, y1))
                x1c = max(0, min(roi.shape[1] - 1, x1))
                x2c = max(0, min(roi.shape[1] - 1, x2))
                if x2c > x1c:
                    region = roi[y1c:min(y1c + 5, roi.shape[0]), x1c:x2c]
                    if region.size > 0:
                        brightness_samples.append(np.mean(region))

            if brightness_samples:
                avg_brightness = np.mean(brightness_samples)
                if avg_brightness < 120:  # Too dark to be white paint
                    return False

            return True
        except Exception:
            return True  # Accept on error to avoid suppressing real detections

    def get_stats(self) -> dict:
        return {
            'total_analyses': self._total_analyses,
            'total_detections': self._total_detections,
            'detection_rate': round(self._total_detections / max(1, self._total_analyses), 4),
        }

    @property
    def is_enabled(self) -> bool:
        return self._enabled


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Crosswalk Detector — Self Test")
    print("=" * 60)

    det = CrosswalkDetector()
    h, w = 480, 640
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # Simulate zebra stripes in lower half
    for i in range(6):
        y = 300 + i * 20
        cv2.line(frame, (100, y), (540, y), (255, 255, 255), 3)

    r = det.detect(frame)
    print(f"  Detected={r.detected}  Type={r.crosswalk_type}  Pos={r.position}")
    if r.detected:
        print(f"  Announcement: {r.get_announcement()}")
    print(f"[Stats] {det.get_stats()}")
    print("Done!")
