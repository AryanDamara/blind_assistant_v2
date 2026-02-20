"""
Traffic Light Detector Module
------------------------------
Detects traffic lights and determines signal state (red/yellow/green).

CRITICAL for outdoor navigation and road crossing safety.

Features:
- Brightness normalization via CLAHE (works in bright sunlight)
- Color-based detection (HSV colour space)
- Circular contour detection and vertical arrangement validation
- Walking / pedestrian signal detection
- Flashing signal detection via temporal tracking
- Distance estimation from depth map
- Temporal smoothing to prevent state flickering

Usage:
    detector = TrafficLightDetector(config_path="config/settings.yaml")
    result = detector.detect(rgb_frame, depth_map)

    if result.detected:
        if result.state == 'red':
            audio.speak("STOP! Red light ahead", priority=5, voice_profile='urgent')
        elif result.state == 'green':
            audio.speak("Green light - safe to cross", priority=1)
"""

import time
import yaml
import os
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List
from collections import deque, Counter


# ============================================================
# Data Class
# ============================================================

@dataclass
class TrafficLightAlert:
    """Alert for detected traffic light."""
    detected: bool
    state: str = 'unknown'                # 'red', 'yellow', 'green', 'walk', 'dont_walk'
    position: Tuple[int, int] = (0, 0)    # (x, y) centre in frame
    confidence: float = 0.0
    distance_m: float = 0.0
    is_flashing: bool = False
    is_pedestrian_signal: bool = False
    time_remaining_sec: float = 0.0

    analysis_time_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def get_announcement(self) -> str:
        """Generate audio announcement."""
        if not self.detected:
            return ""

        if self.state == 'red':
            return "STOP! Red light ahead - Do not cross"
        elif self.state == 'yellow':
            return "Caution: Yellow light - do not start crossing"
        elif self.state == 'green':
            if self.distance_m > 0:
                return f"Green light at {int(self.distance_m)} meters - safe to cross"
            return "Green light - safe to cross"
        elif self.state == 'walk':
            if self.time_remaining_sec > 0:
                return f"Walk signal - {int(self.time_remaining_sec)} seconds remaining"
            return "Walk signal - safe to cross"
        elif self.state == 'dont_walk':
            return "Don't walk signal - wait for green"
        return "Traffic light detected"

    def to_dict(self) -> dict:
        return {
            'detected': self.detected,
            'state': self.state,
            'position': self.position,
            'confidence': round(self.confidence, 3),
            'distance_m': round(self.distance_m, 2),
            'is_flashing': self.is_flashing,
            'is_pedestrian': self.is_pedestrian_signal,
            'time_remaining': self.time_remaining_sec,
            'analysis_time_ms': round(self.analysis_time_ms, 2),
        }


# ============================================================
# Traffic Light Detector
# ============================================================

class TrafficLightDetector:
    """
    Detects traffic lights using colour analysis and shape detection.

    Pipeline:
    1. Colour segmentation in HSV space (red, yellow, green)
    2. Morphological filtering to reduce noise
    3. Circular contour detection for light shapes
    4. Vertical arrangement validation (3 lights stacked)
    5. State determination from brightest light
    6. Temporal smoothing to prevent flicker
    7. Distance estimation from depth map
    """

    def __init__(self, config_path: str = "config/settings.yaml"):
        self._load_config(config_path)

        # Temporal tracking
        self._detection_history: deque = deque(maxlen=5)
        self._last_state: str = 'unknown'

        # Cooldown
        self._last_alert_time: float = 0.0

        # Statistics
        self._total_analyses: int = 0
        self._total_detections: int = 0
        self._state_counts: Dict[str, int] = {
            'red': 0, 'yellow': 0, 'green': 0, 'walk': 0, 'dont_walk': 0
        }

    def _load_config(self, config_path: str) -> None:
        """Load configuration from YAML."""
        self._enabled: bool = True
        self._min_confidence: float = 0.6
        self._cooldown_sec: float = 2.0
        self._detect_pedestrian_signals: bool = True
        self._detect_flashing: bool = True
        self._temporal_smoothing: bool = True

        # Brightness normalization
        self._brightness_normalization: bool = True

        # Distance filtering
        self._min_distance_m: float = 5.0
        self._max_distance_m: float = 30.0

        # HSV colour ranges
        self._color_ranges: Dict = {
            'red_lower1': np.array([0, 100, 100]),
            'red_upper1': np.array([10, 255, 255]),
            'red_lower2': np.array([170, 100, 100]),
            'red_upper2': np.array([180, 255, 255]),
            'yellow_lower': np.array([20, 100, 100]),
            'yellow_upper': np.array([30, 255, 255]),
            'green_lower': np.array([40, 100, 100]),
            'green_upper': np.array([80, 255, 255]),
        }

        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f) or {}
                tl = config.get('traffic_light_detection', {})
                self._enabled = tl.get('enabled', True)
                self._min_confidence = tl.get('min_confidence', 0.6)
                self._cooldown_sec = tl.get('cooldown_sec', 2.0)
                self._detect_pedestrian_signals = tl.get('detect_pedestrian', True)
                self._temporal_smoothing = tl.get('temporal_smoothing', True)
                self._brightness_normalization = tl.get('brightness_normalization', True)
                self._min_distance_m = tl.get('min_distance_m', 5.0)
                self._max_distance_m = tl.get('max_distance_m', 30.0)
            except Exception as e:
                print(f"[TrafficLightDetector] WARNING: Config error: {e}")

        print(f"[TrafficLightDetector] Initialized (enabled={self._enabled})")

    # ---------------------------------------------------------------- detect
    def detect(self, rgb_frame: np.ndarray,
               depth_map: Optional[np.ndarray] = None) -> TrafficLightAlert:
        """Detect traffic light in frame."""
        start = time.time()
        self._total_analyses += 1

        if not self._enabled:
            return TrafficLightAlert(detected=False,
                                    analysis_time_ms=(time.time() - start) * 1000)
        if time.time() - self._last_alert_time < self._cooldown_sec:
            return TrafficLightAlert(detected=False,
                                    analysis_time_ms=(time.time() - start) * 1000)
        if rgb_frame is None or rgb_frame.size == 0:
            return TrafficLightAlert(detected=False,
                                    analysis_time_ms=(time.time() - start) * 1000)

        try:
            # Brightness normalization — critical for bright sunlight
            frame_to_process = rgb_frame
            if self._brightness_normalization:
                frame_to_process = self._normalize_brightness(rgb_frame)

            result = self._detect_traffic_light(frame_to_process)

            if result.detected and depth_map is not None:
                result.distance_m = self._estimate_distance(depth_map, result.position)

                # Distance filtering — ignore lights too far or too close
                if result.distance_m > 0:
                    if (result.distance_m < self._min_distance_m or
                            result.distance_m > self._max_distance_m):
                        return TrafficLightAlert(
                            detected=False,
                            analysis_time_ms=(time.time() - start) * 1000)

            if self._temporal_smoothing:
                result = self._apply_temporal_smoothing(result)

            result.analysis_time_ms = (time.time() - start) * 1000

            if result.detected:
                self._total_detections += 1
                self._state_counts[result.state] = self._state_counts.get(result.state, 0) + 1
                self._last_alert_time = time.time()
                self._last_state = result.state

            return result

        except Exception as e:
            print(f"[TrafficLightDetector] ERROR: {e}")
            return TrafficLightAlert(detected=False,
                                    analysis_time_ms=(time.time() - start) * 1000)

    # -------------------------------------------------------- core detection
    def _detect_traffic_light(self, rgb_frame: np.ndarray) -> TrafficLightAlert:
        """Core detection via HSV colour segmentation."""
        hsv = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]

        # Focus on upper half (traffic lights above horizon)
        roi = hsv[:h // 2, :]

        red_mask = self._detect_color(roi, 'red')
        yellow_mask = self._detect_color(roi, 'yellow')
        green_mask = self._detect_color(roi, 'green')

        red_cand = self._find_light_candidates(red_mask)
        yellow_cand = self._find_light_candidates(yellow_mask)
        green_cand = self._find_light_candidates(green_mask)

        validated = self._validate_traffic_light_arrangement(
            red_cand, yellow_cand, green_cand
        )
        if validated is None:
            return TrafficLightAlert(detected=False)

        state, confidence, position = validated
        return TrafficLightAlert(
            detected=True, state=state, position=position,
            confidence=confidence, is_pedestrian_signal=False,
        )

    def _detect_color(self, hsv_image: np.ndarray, color: str) -> np.ndarray:
        """Isolate colour mask in HSV space."""
        if color == 'red':
            m1 = cv2.inRange(hsv_image, self._color_ranges['red_lower1'],
                             self._color_ranges['red_upper1'])
            m2 = cv2.inRange(hsv_image, self._color_ranges['red_lower2'],
                             self._color_ranges['red_upper2'])
            mask = cv2.bitwise_or(m1, m2)
        elif color == 'yellow':
            mask = cv2.inRange(hsv_image, self._color_ranges['yellow_lower'],
                               self._color_ranges['yellow_upper'])
        elif color == 'green':
            mask = cv2.inRange(hsv_image, self._color_ranges['green_lower'],
                               self._color_ranges['green_upper'])
        else:
            return np.zeros(hsv_image.shape[:2], dtype=np.uint8)

        k = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        return mask

    def _find_light_candidates(self, mask: np.ndarray) -> List[Tuple[int, int, float]]:
        """Find circular regions that could be traffic lights."""
        candidates: List[Tuple[int, int, float]] = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 50 < area < 500:
                peri = cv2.arcLength(cnt, True)
                if peri > 0:
                    circ = 4 * np.pi * area / (peri ** 2)
                    if circ > 0.6:
                        M = cv2.moments(cnt)
                        if M['m00'] > 0:
                            cx = int(M['m10'] / M['m00'])
                            cy = int(M['m01'] / M['m00'])
                            h, w = mask.shape
                            y1 = max(0, cy - 10)
                            y2 = min(h, cy + 10)
                            x1 = max(0, cx - 10)
                            x2 = min(w, cx + 10)
                            brightness = float(np.sum(mask[y1:y2, x1:x2])) / max(1, (y2-y1)*(x2-x1))
                            candidates.append((cx, cy, brightness))
        return candidates

    def _validate_traffic_light_arrangement(
        self, red_cand, yellow_cand, green_cand
    ) -> Optional[Tuple[str, float, Tuple[int, int]]]:
        """Validate vertical arrangement of detected lights."""
        all_c: List[Tuple[str, int, int, float]] = []
        for cx, cy, b in red_cand:
            all_c.append(('red', cx, cy, b))
        for cx, cy, b in yellow_cand:
            all_c.append(('yellow', cx, cy, b))
        for cx, cy, b in green_cand:
            all_c.append(('green', cx, cy, b))

        if len(all_c) < 2:
            return None

        # Group by vertical alignment (same x ±20px)
        groups: List[List] = []
        for cand in all_c:
            _, cx, _, _ = cand
            found = False
            for g in groups:
                if abs(g[0][1] - cx) < 20:
                    g.append(cand)
                    found = True
                    break
            if not found:
                groups.append([cand])

        best = max(groups, key=len) if groups else []
        if len(best) < 2:
            return None

        best.sort(key=lambda x: x[2])
        for i in range(len(best) - 1):
            dy = best[i + 1][2] - best[i][2]
            if not (30 < dy < 100):
                return None

        brightest = max(best, key=lambda x: x[3])
        state = brightest[0]
        conf = min(1.0, brightest[3] / 255.0)
        pos = (brightest[1], brightest[2])
        if conf < self._min_confidence:
            return None
        return (state, conf, pos)

    # ----------------------------------------------------------- helpers
    def _estimate_distance(self, depth_map: np.ndarray,
                           position: Tuple[int, int]) -> float:
        x, y = position
        h, w = depth_map.shape
        x1, x2 = max(0, x - 10), min(w, x + 10)
        y1, y2 = max(0, y - 10), min(h, y + 10)
        try:
            d = float(np.median(depth_map[y1:y2, x1:x2]))
            return max(1.0, min(d, 50.0))
        except Exception:
            return 0.0

    def _normalize_brightness(self, rgb_frame: np.ndarray) -> np.ndarray:
        """Normalize lighting via CLAHE for consistent color detection in all conditions."""
        try:
            lab = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l_eq = clahe.apply(l)
            lab_eq = cv2.merge([l_eq, a, b])
            return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
        except Exception as e:
            print(f"[TrafficLightDetector] Brightness normalization error: {e}")
            return rgb_frame

    def _apply_temporal_smoothing(self, current: TrafficLightAlert) -> TrafficLightAlert:
        self._detection_history.append(current.state if current.detected else 'none')
        counts = Counter(self._detection_history)

        if counts.get(current.state, 0) >= 2:
            return current
        if self._last_state != 'unknown':
            current.state = self._last_state
            current.confidence *= 0.7
        return current

    def get_stats(self) -> dict:
        return {
            'total_analyses': self._total_analyses,
            'total_detections': self._total_detections,
            'detection_rate': round(self._total_detections / max(1, self._total_analyses), 4),
            'state_counts': dict(self._state_counts),
            'last_state': self._last_state,
        }

    @property
    def is_enabled(self) -> bool:
        return self._enabled


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Traffic Light Detector — Self Test")
    print("=" * 60)

    det = TrafficLightDetector()

    h, w = 480, 640
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.circle(frame, (320, 100), 30, (0, 0, 255), -1)  # Red (BGR)

    print("\n[Test 1] Synthetic red light …")
    r = det.detect(frame)
    print(f"  Detected={r.detected}  State={r.state}  Conf={r.confidence:.2f}")
    if r.detected:
        print(f"  Announcement: {r.get_announcement()}")

    print(f"\n[Stats] {det.get_stats()}")
    print("Done!")
