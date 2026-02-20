"""
Curb Detector Module
---------------------
Detects curbs and step-downs using depth gradient analysis.

CRITICAL safety feature — curbs are a major trip/fall risk.

Features:
- Horizontal depth gradient analysis in the bottom 40 % of the frame
- Curb height estimation from depth differences
- Up/down direction detection (step-up vs drop-off)
- Shadow false-positive filtering (via RGB brightness check)
- Ramp detection filter (avoids triggering on gradual slopes)
- Temporal consistency tracking (multi-frame validation)
- Position classification (left / ahead / right)
- Close-distance urgent warnings

Usage:
    detector = CurbDetector(config_path="config/settings.yaml")
    result = detector.detect(depth_map, rgb_frame)

    if result.detected:
        priority = 5 if result.distance_m < 1.0 else 3
        audio.speak(result.get_announcement(), priority=priority)
"""

import time
import yaml
import os
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

# Optional OpenCV for shadow filtering
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


# ============================================================
# Data Class
# ============================================================

@dataclass
class CurbAlert:
    """Alert for detected curb / step."""
    detected: bool
    height_cm: float = 0.0
    position: str = 'unknown'         # 'ahead', 'left', 'right'
    curb_direction: str = 'unknown'   # 'up', 'down', 'unknown'
    distance_m: float = 0.0
    confidence: float = 0.0

    analysis_time_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def get_announcement(self) -> str:
        if not self.detected:
            return ""

        # Direction-specific labels
        if self.curb_direction == 'down':
            label = "DROP-OFF" if self.height_cm > 15 else "Step down"
        elif self.curb_direction == 'up':
            label = "CURB UP" if self.height_cm > 15 else "Step up"
        elif self.height_cm > 15:
            label = "HIGH CURB"
        elif self.height_cm > 10:
            label = "Curb"
        else:
            label = "Small curb"

        if self.distance_m > 0 and self.distance_m < 1.0:
            return (f"CAUTION! {label} {self.position} - "
                    f"VERY CLOSE at {int(self.distance_m * 100)}cm")
        elif self.distance_m >= 1.0:
            return (f"{label} {self.position} at {self.distance_m:.1f} meters, "
                    f"height {int(self.height_cm)}cm")
        return f"{label} {self.position}"

    def to_dict(self) -> dict:
        return {
            'detected': self.detected,
            'height_cm': round(self.height_cm, 1),
            'position': self.position,
            'curb_direction': self.curb_direction,
            'distance_m': round(self.distance_m, 2),
            'confidence': round(self.confidence, 3),
            'analysis_time_ms': round(self.analysis_time_ms, 2),
        }


# ============================================================
# Curb Detector
# ============================================================

class CurbDetector:
    """
    Detects curbs via sudden horizontal depth transitions at ground level.

    Pipeline:
    1. Extract bottom-40 % of depth map (ground plane)
    2. Compute column-wise depth differences
    3. Find rows with significant horizontal gradient
    4. Filter shadows (if RGB frame available)
    5. Filter ramps (gradual slopes)
    6. Determine up/down direction
    7. Estimate curb height from depth differential
    8. Temporal consistency filtering
    9. Classify position (left / centre / right)
    """

    def __init__(self, config_path: str = "config/settings.yaml"):
        self._enabled: bool = True
        self._min_gradient: float = 0.05
        self._min_curb_height_cm: float = 5.0
        self._cooldown_sec: float = 2.0
        self._last_alert_time: float = 0.0

        # Enhanced features
        self._filter_shadows: bool = True
        self._detect_direction: bool = True
        self._filter_ramps: bool = True
        self._use_temporal_tracking: bool = True
        self._min_temporal_consistency: int = 2

        # Temporal tracking
        self._detection_history: deque = deque(maxlen=5)

        # Stats
        self._total_analyses: int = 0
        self._total_detections: int = 0

        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    cfg = yaml.safe_load(f) or {}
                cc = cfg.get('curb_detection', {})
                self._enabled = cc.get('enabled', True)
                self._min_gradient = cc.get('min_gradient', 0.05)
                self._min_curb_height_cm = cc.get('min_curb_height_cm', 5.0)
                self._cooldown_sec = cc.get('cooldown_sec', 2.0)
                self._filter_shadows = cc.get('filter_shadows', True)
                self._detect_direction = cc.get('detect_direction', True)
                self._filter_ramps = cc.get('filter_ramps', True)
                self._use_temporal_tracking = cc.get('use_temporal_tracking', True)
                self._min_temporal_consistency = cc.get('min_temporal_consistency', 2)
            except Exception as e:
                print(f"[CurbDetector] WARNING: Config error: {e}")

        print(f"[CurbDetector] Initialized (enabled={self._enabled})")

    # ---------------------------------------------------------------- detect
    def detect(self, depth_map: np.ndarray,
               rgb_frame: Optional[np.ndarray] = None) -> CurbAlert:
        """Detect curb in depth map with optional RGB validation."""
        start = time.time()
        self._total_analyses += 1

        if not self._enabled or depth_map is None or depth_map.size == 0:
            return CurbAlert(detected=False,
                             analysis_time_ms=(time.time() - start) * 1000)
        if time.time() - self._last_alert_time < self._cooldown_sec:
            return CurbAlert(detected=False,
                             analysis_time_ms=(time.time() - start) * 1000)

        try:
            h, w = depth_map.shape[:2]
            if len(depth_map.shape) == 3:
                depth_map = depth_map[:, :, 0]

            # Bottom 40 % of frame
            roi = depth_map[int(h * 0.6):, :]
            if roi.size == 0 or roi.shape[1] < 3:
                return CurbAlert(detected=False,
                                 analysis_time_ms=(time.time() - start) * 1000)

            # Horizontal gradient
            dx = np.abs(np.diff(roi, axis=1))
            row_grads = np.mean(dx, axis=1)
            sig_rows = np.where(row_grads > self._min_gradient)[0]

            if len(sig_rows) == 0:
                self._detection_history.append(False)
                return CurbAlert(detected=False,
                                 analysis_time_ms=(time.time() - start) * 1000)

            peak = sig_rows[np.argmax(row_grads[sig_rows])]

            # --- Shadow filter ---
            if self._filter_shadows and rgb_frame is not None and _HAS_CV2:
                if self._is_shadow_not_curb(roi, rgb_frame, peak):
                    self._detection_history.append(False)
                    return CurbAlert(detected=False,
                                     analysis_time_ms=(time.time() - start) * 1000)

            # --- Ramp filter ---
            if self._filter_ramps:
                if self._is_ramp_not_curb(roi, peak):
                    self._detection_history.append(False)
                    return CurbAlert(detected=False,
                                     analysis_time_ms=(time.time() - start) * 1000)

            # Height from depth difference
            row_depth = roi[peak, :]
            depth_diff = float(np.max(row_depth) - np.min(row_depth))
            height_cm = depth_diff * 100

            if height_cm < self._min_curb_height_cm:
                self._detection_history.append(False)
                return CurbAlert(detected=False,
                                 analysis_time_ms=(time.time() - start) * 1000)

            # --- Direction detection ---
            curb_direction = 'unknown'
            if self._detect_direction:
                curb_direction = self._determine_curb_direction(roi, peak)

            # Position
            end_row = min(peak + 5, dx.shape[0])
            col_grads = np.mean(dx[peak:end_row, :], axis=0)
            max_col = int(np.argmax(col_grads))

            if max_col < w * 0.3:
                position = 'left'
            elif max_col > w * 0.7:
                position = 'right'
            else:
                position = 'ahead'

            distance_m = float(np.median(roi[peak, :]))
            confidence = min(1.0, row_grads[peak] / 0.2)

            # --- Temporal consistency ---
            self._detection_history.append(True)
            if self._use_temporal_tracking:
                recent_detections = sum(self._detection_history)
                if recent_detections < self._min_temporal_consistency:
                    return CurbAlert(
                        detected=False, confidence=confidence * 0.6,
                        analysis_time_ms=(time.time() - start) * 1000,
                    )

            result = CurbAlert(
                detected=True, height_cm=height_cm, position=position,
                curb_direction=curb_direction,
                distance_m=distance_m, confidence=confidence,
                analysis_time_ms=(time.time() - start) * 1000,
            )
            self._total_detections += 1
            self._last_alert_time = time.time()
            return result

        except Exception as e:
            print(f"[CurbDetector] ERROR: {e}")
            return CurbAlert(detected=False,
                             analysis_time_ms=(time.time() - start) * 1000)

    # -------------------------------------------------------- shadow filter
    def _is_shadow_not_curb(self, roi: np.ndarray,
                            rgb_frame: np.ndarray,
                            curb_row: int) -> bool:
        """Distinguish shadow from real curb using depth change + brightness."""
        if curb_row < 2 or curb_row >= roi.shape[0] - 2:
            return False

        # Real curb: significant depth change across the edge
        depth_before = np.mean(roi[max(0, curb_row - 2):curb_row, :])
        depth_after = np.mean(roi[curb_row + 1:min(roi.shape[0], curb_row + 3), :])
        depth_change = abs(depth_after - depth_before)

        if depth_change < 0.03:  # < 3 cm actual depth change → probably shadow
            return True

        # Check RGB brightness at the edge row
        try:
            gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
            fh, _ = gray.shape
            roi_start = int(fh * 0.6)
            actual_row = roi_start + curb_row

            if 0 <= actual_row < gray.shape[0]:
                row_brightness = float(np.mean(gray[actual_row, :]))
                if row_brightness < 80:  # Very dark → shadow edge
                    return True
        except Exception:
            pass

        return False

    # -------------------------------------------------------- ramp filter
    def _is_ramp_not_curb(self, roi: np.ndarray, peak_row: int) -> bool:
        """Check if detected edge is a ramp (gradual slope), not a curb."""
        if peak_row < 10 or peak_row >= roi.shape[0] - 10:
            return False
        try:
            profile_before = []
            profile_after = []

            for i in range(-10, 0):
                if peak_row + i >= 0:
                    profile_before.append(np.median(roi[peak_row + i, :]))
            for i in range(1, 11):
                if peak_row + i < roi.shape[0]:
                    profile_after.append(np.median(roi[peak_row + i, :]))

            if not profile_before or not profile_after:
                return False

            grad_before = np.diff(profile_before)
            grad_after = np.diff(profile_after)

            if len(grad_before) > 0 and len(grad_after) > 0:
                # Ramp: low std (consistent gradient) on both sides
                if np.std(grad_before) < 0.02 and np.std(grad_after) < 0.02:
                    return True

            return False
        except Exception:
            return False

    # -------------------------------------------------------- direction
    def _determine_curb_direction(self, roi: np.ndarray, peak_row: int) -> str:
        """Determine if curb goes up or down (step-up vs drop-off)."""
        if peak_row < 2 or peak_row >= roi.shape[0] - 2:
            return 'unknown'
        try:
            depth_before = np.median(roi[max(0, peak_row - 2):peak_row, :])
            depth_after = np.median(roi[peak_row + 1:min(roi.shape[0], peak_row + 3), :])
            diff = depth_after - depth_before

            if diff > 0.05:   # Depth increases → step down / drop-off
                return 'down'
            elif diff < -0.05:  # Depth decreases → step up
                return 'up'
            return 'unknown'
        except Exception:
            return 'unknown'

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
    print("Curb Detector — Self Test")
    print("=" * 60)

    det = CurbDetector()

    h, w = 100, 160
    dm = np.ones((h, w), dtype=np.float32) * 2.0
    # Create curb: sudden horizontal depth change in bottom area
    dm[70:, 80:] = 2.3

    r = det.detect(dm)
    print(f"  Detected={r.detected}  Height={r.height_cm:.1f}cm  Pos={r.position}")
    print(f"  Direction={r.curb_direction}")
    if r.detected:
        print(f"  Distance={r.distance_m:.2f}m  Confidence={r.confidence:.2f}")
        print(f"  Announcement: {r.get_announcement()}")

    print(f"[Stats] {det.get_stats()}")
    print("Done!")
