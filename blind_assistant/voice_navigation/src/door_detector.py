"""
Door Detector Module
---------------------
Detects doors using edge detection, rectangle fitting, and glass door
frame analysis.

Features:
- Glass door detection (CRITICAL SAFETY) — transparent doors via frame edges
- Rectangle contour detection with aspect-ratio constraints
- Open/closed state from interior variance
- Improved handle detection using edge analysis
- Distance estimation from depth map

Usage:
    detector = DoorDetector(config_path="config/settings.yaml")
    result = detector.detect(rgb_frame, depth_map)

    if result.detected:
        audio.speak(result.get_announcement(), priority=2)
"""

import time
import yaml
import os
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional, Tuple


# ============================================================
# Data Class
# ============================================================

@dataclass
class DoorAlert:
    """Alert for detected door."""
    detected: bool
    door_type: str = 'unknown'     # 'standard', 'glass', 'automatic', 'revolving', 'double'
    state: str = 'unknown'         # 'open', 'closed', 'opening'
    position: str = 'unknown'      # 'ahead', 'left', 'right'
    width_m: float = 0.0
    distance_m: float = 0.0
    has_handle: bool = False
    handle_side: str = 'unknown'   # 'left', 'right', 'both'
    confidence: float = 0.0

    analysis_time_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def get_announcement(self) -> str:
        if not self.detected:
            return ""

        parts = []
        if self.door_type == 'glass':
            parts.append("CAUTION - Glass door")
        elif self.door_type == 'automatic':
            parts.append("Automatic door")
        else:
            parts.append("Door")
        parts.append(self.state)
        parts.append(self.position)

        if self.distance_m > 0:
            parts.append(f"{self.distance_m:.1f} meters")

        if self.state == 'closed' and self.has_handle:
            parts.append(f"handle on {self.handle_side}")

        return ", ".join(parts)

    def to_dict(self) -> dict:
        return {
            'detected': self.detected,
            'door_type': self.door_type,
            'state': self.state,
            'position': self.position,
            'width_m': round(self.width_m, 2),
            'distance_m': round(self.distance_m, 2),
            'has_handle': self.has_handle,
            'handle_side': self.handle_side,
            'confidence': round(self.confidence, 3),
            'analysis_time_ms': round(self.analysis_time_ms, 2),
        }


# ============================================================
# Door Detector
# ============================================================

class DoorDetector:
    """
    Detects doors via edge/contour analysis and glass-frame detection.

    Pipeline:
    1. Try glass door detection FIRST (most dangerous)
    2. Canny edge detection for standard doors
    3. Find contours, approximate to polygons
    4. Keep 4-vertex polygons (rectangles)
    5. Filter by size (15-50 % frame width) and aspect ratio (>1.5)
    6. Determine open/closed from interior pixel variance
    7. Improved handle detection via edge analysis
    """

    def __init__(self, config_path: str = "config/settings.yaml"):
        self._enabled: bool = True
        self._min_width_ratio: float = 0.15
        self._max_width_ratio: float = 0.50
        self._min_aspect_ratio: float = 1.5
        self._cooldown_sec: float = 3.0
        self._last_alert_time: float = 0.0

        # Glass door detection
        self._detect_glass_doors: bool = True
        self._detect_automatic: bool = True
        self._improved_handle_detection: bool = True

        # Stats
        self._total_analyses: int = 0
        self._total_detections: int = 0

        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    cfg = yaml.safe_load(f) or {}
                dc = cfg.get('door_detection', {})
                self._enabled = dc.get('enabled', True)
                self._cooldown_sec = dc.get('cooldown_sec', 3.0)
                self._detect_glass_doors = dc.get('detect_glass_doors', True)
                self._detect_automatic = dc.get('detect_automatic', True)
                self._improved_handle_detection = dc.get('improved_handle_detection', True)
            except Exception as e:
                print(f"[DoorDetector] WARNING: Config error: {e}")

        print(f"[DoorDetector] Initialized (enabled={self._enabled}, "
              f"glass={self._detect_glass_doors})")

    # ---------------------------------------------------------------- detect
    def detect(self, rgb_frame: np.ndarray,
               depth_map: Optional[np.ndarray] = None) -> DoorAlert:
        """Detect door in frame.  Tries glass doors first (safety-critical)."""
        start = time.time()
        self._total_analyses += 1

        if not self._enabled:
            return DoorAlert(detected=False,
                             analysis_time_ms=(time.time() - start) * 1000)
        if time.time() - self._last_alert_time < self._cooldown_sec:
            return DoorAlert(detected=False,
                             analysis_time_ms=(time.time() - start) * 1000)
        if rgb_frame is None or rgb_frame.size == 0:
            return DoorAlert(detected=False,
                             analysis_time_ms=(time.time() - start) * 1000)

        try:
            # --- Glass door detection FIRST (most dangerous) ---
            if self._detect_glass_doors:
                glass_result = self._detect_glass_door(rgb_frame, depth_map)
                if glass_result.detected:
                    glass_result.analysis_time_ms = (time.time() - start) * 1000
                    self._total_detections += 1
                    self._last_alert_time = time.time()
                    return glass_result

            # --- Standard door detection ---
            gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
                eps = 0.02 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, eps, True)

                if len(approx) == 4:
                    x, y, bw, bh = cv2.boundingRect(approx)
                    w_ratio = bw / w
                    a_ratio = bh / bw if bw > 0 else 0

                    if (self._min_width_ratio < w_ratio < self._max_width_ratio
                            and a_ratio > self._min_aspect_ratio):

                        # Position
                        cx = x + bw / 2
                        if cx < w * 0.3:
                            position = 'left'
                        elif cx > w * 0.7:
                            position = 'right'
                        else:
                            position = 'ahead'

                        # Distance
                        distance = 0.0
                        if depth_map is not None:
                            try:
                                distance = float(np.median(
                                    depth_map[y:y + bh, x:x + bw]))
                            except Exception:
                                pass

                        # Open / closed
                        interior = gray[y:y + bh, x:x + bw]
                        var = float(np.var(interior))
                        state = 'open' if var < 500 else 'closed'

                        # Handle detection
                        if self._improved_handle_detection:
                            has_handle, handle_side = self._detect_door_handle(
                                gray, (x, y, bw, bh))
                        else:
                            has_handle = state == 'closed'
                            handle_side = ('right'
                                           if (x + bw / 2) < w / 2 else 'left')

                        result = DoorAlert(
                            detected=True, door_type='standard',
                            state=state, position=position,
                            width_m=w_ratio * 3.0, distance_m=distance,
                            has_handle=has_handle, handle_side=handle_side,
                            confidence=0.7,
                            analysis_time_ms=(time.time() - start) * 1000,
                        )
                        self._total_detections += 1
                        self._last_alert_time = time.time()
                        return result

            return DoorAlert(detected=False,
                             analysis_time_ms=(time.time() - start) * 1000)

        except Exception as e:
            print(f"[DoorDetector] ERROR: {e}")
            return DoorAlert(detected=False,
                             analysis_time_ms=(time.time() - start) * 1000)

    # ------------------------------------------------------- glass door
    def _detect_glass_door(self, rgb_frame: np.ndarray,
                           depth_map: Optional[np.ndarray]) -> DoorAlert:
        """Detect transparent glass doors using vertical frame edges.

        Glass doors are the MOST DANGEROUS obstacle for visually impaired
        users — transparent, minimal edges.  Detection relies on finding
        parallel vertical lines (the metal / wood door frame) with
        consistent depth between them.
        """
        try:
            gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100,
                                    minLineLength=h // 3, maxLineGap=20)
            if lines is None:
                return DoorAlert(detected=False)

            # Filter for vertical lines (80°-100°)
            vertical_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if 80 < angle < 100:
                    vertical_lines.append(line)
            if len(vertical_lines) < 2:
                return DoorAlert(detected=False)

            # Look for a pair of verticals forming a door frame
            for i, line1 in enumerate(vertical_lines):
                for line2 in vertical_lines[i + 1:]:
                    x1_avg = (line1[0][0] + line1[0][2]) / 2
                    x2_avg = (line2[0][0] + line2[0][2]) / 2
                    width_px = abs(x2_avg - x1_avg)

                    # Typical door width in pixels at viewing-distance
                    if 80 < width_px < int(w * 0.5):
                        # If depth available, verify minimal depth variation
                        # (glass is flat)
                        if depth_map is not None:
                            try:
                                x_min = int(min(x1_avg, x2_avg))
                                x_max = int(max(x1_avg, x2_avg))
                                y_mid = h // 2
                                door_region = depth_map[
                                    max(0, y_mid - 50):min(h, y_mid + 50),
                                    max(0, x_min):min(w, x_max)
                                ]
                                if door_region.size > 0:
                                    depth_var = float(np.var(door_region))
                                    if depth_var > 0.05:
                                        continue  # Too variable, not flat glass
                            except Exception:
                                pass

                        # Position
                        center_x = (x1_avg + x2_avg) / 2
                        if center_x < w * 0.3:
                            position = 'left'
                        elif center_x > w * 0.7:
                            position = 'right'
                        else:
                            position = 'ahead'

                        # Distance
                        distance = 0.0
                        if depth_map is not None:
                            try:
                                x_min = int(min(x1_avg, x2_avg))
                                x_max = int(max(x1_avg, x2_avg))
                                y_mid = h // 2
                                region = depth_map[
                                    max(0, y_mid - 50):min(h, y_mid + 50),
                                    max(0, x_min):min(w, x_max)
                                ]
                                if region.size > 0:
                                    distance = float(np.median(region))
                            except Exception:
                                pass

                        return DoorAlert(
                            detected=True, door_type='glass',
                            state='closed', position=position,
                            width_m=width_px / w * 3.0,
                            distance_m=distance, confidence=0.75,
                        )

            return DoorAlert(detected=False)

        except Exception as e:
            print(f"[DoorDetector] Glass door detection error: {e}")
            return DoorAlert(detected=False)

    # ------------------------------------------------------- handle
    def _detect_door_handle(self, gray: np.ndarray,
                            bbox: Tuple) -> Tuple[bool, str]:
        """Detect door handle and which side it's on using edge analysis."""
        x, y, bw, bh = bbox

        # Handles are typically at 40-60 % of door height (waist level)
        handle_y1 = y + int(bh * 0.4)
        handle_y2 = y + int(bh * 0.6)

        # Check left and right quarter of the door
        left_region = gray[handle_y1:handle_y2, x:x + bw // 4]
        right_region = gray[handle_y1:handle_y2, x + 3 * bw // 4:x + bw]

        def _has_handle_features(region: np.ndarray) -> bool:
            if region.size == 0:
                return False
            edges = cv2.Canny(region, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 20,
                                    minLineLength=15, maxLineGap=5)
            if lines is None:
                return False
            horizontal = sum(
                1 for l in lines
                if abs(np.arctan2(l[0][3] - l[0][1],
                                  l[0][2] - l[0][0]) * 180 / np.pi) < 30
            )
            return horizontal >= 2

        left_has = _has_handle_features(left_region)
        right_has = _has_handle_features(right_region)

        if left_has and not right_has:
            return (True, 'left')
        elif right_has and not left_has:
            return (True, 'right')
        elif left_has and right_has:
            return (True, 'both')
        return (False, 'unknown')

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
    print("Door Detector — Self Test")
    print("=" * 60)

    det = DoorDetector()
    h, w = 480, 640
    frame = np.ones((h, w, 3), dtype=np.uint8) * 200
    # Draw a rectangle resembling a door
    cv2.rectangle(frame, (200, 50), (400, 450), (80, 50, 30), 3)

    r = det.detect(frame)
    print(f"  Detected={r.detected}  Type={r.door_type}  State={r.state}")
    if r.detected:
        print(f"  Announcement: {r.get_announcement()}")

    print(f"[Stats] {det.get_stats()}")
    print("Done!")
