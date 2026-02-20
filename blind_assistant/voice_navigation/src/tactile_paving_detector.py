"""
Tactile Paving Detector Module
-------------------------------
Detects tactile paving (textured ground surface indicators) for
visually impaired navigation.

Types:
- Warning (bumps): Stop before road crossing
- Directional (lines): Follow direction
- Platform edge: Danger — edge ahead

Features:
- Block-wise variance analysis on bottom 30% of frame
- Gradient-based type classification (warning vs directional)
- FFT periodic pattern validation (filters gravel, grass)
- Direction angle extraction for directional paving
"""

import time
import yaml
import os
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class TactilePavingAlert:
    """Alert for tactile paving detection."""
    detected: bool
    paving_type: str = 'unknown'      # 'warning', 'directional', 'platform_edge'
    position: str = 'unknown'
    distance_m: float = 0.0
    confidence: float = 0.0
    direction_angle: float = 0.0      # 0-360° for directional paving

    analysis_time_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def get_announcement(self) -> str:
        if not self.detected:
            return ""
        if self.paving_type == 'warning':
            return f"Warning tactile paving {self.position} - stop before crossing"
        elif self.paving_type == 'directional':
            # Convert angle to human-friendly direction
            angle = self.direction_angle
            if 45 < angle < 135:
                dir_str = "to your right"
            elif 135 < angle < 225:
                dir_str = "behind you"
            elif 225 < angle < 315:
                dir_str = "to your left"
            else:
                dir_str = "straight ahead"
            return f"Directional paving - follow the grooves {dir_str}"
        elif self.paving_type == 'platform_edge':
            return "Platform edge ahead - CAUTION!"
        return f"Tactile paving {self.position}"

    def to_dict(self) -> dict:
        return {
            'detected': self.detected, 'paving_type': self.paving_type,
            'position': self.position, 'distance_m': round(self.distance_m, 2),
            'confidence': round(self.confidence, 3),
            'direction_angle': round(self.direction_angle, 1),
            'analysis_time_ms': round(self.analysis_time_ms, 2),
        }


class TactilePavingDetector:
    """
    Detects tactile paving via texture (variance) analysis with pattern
    validation and type classification.

    Pipeline:
    1. Extract bottom-30 % ROI (ground level)
    2. Block-wise variance calculation
    3. If variance exceeds threshold → candidate
    4. Validate periodic pattern via FFT
    5. Classify type via gradient orientation histogram
    """

    def __init__(self, config_path: str = "config/settings.yaml"):
        self._enabled = True
        self._cooldown_sec = 5.0
        self._variance_threshold = 300.0
        self._last_alert_time = 0.0
        self._total_analyses = 0
        self._total_detections = 0

        # Enhanced features
        self._classify_type: bool = True
        self._validate_pattern: bool = True
        self._extract_direction: bool = True

        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    cfg = yaml.safe_load(f) or {}
                tc = cfg.get('tactile_paving_detection', {})
                self._enabled = tc.get('enabled', True)
                self._cooldown_sec = tc.get('cooldown_sec', 5.0)
                self._variance_threshold = tc.get('variance_threshold', 300.0)
                self._classify_type = tc.get('classify_type', True)
                self._validate_pattern = tc.get('validate_pattern', True)
                self._extract_direction = tc.get('extract_direction', True)
            except Exception as e:
                print(f"[TactilePavingDetector] WARNING: Config error: {e}")
        print(f"[TactilePavingDetector] Initialized (enabled={self._enabled})")

    def detect(self, rgb_frame: np.ndarray) -> TactilePavingAlert:
        start = time.time()
        self._total_analyses += 1

        if not self._enabled or rgb_frame is None or rgb_frame.size == 0:
            return TactilePavingAlert(detected=False,
                                     analysis_time_ms=(time.time() - start) * 1000)
        if time.time() - self._last_alert_time < self._cooldown_sec:
            return TactilePavingAlert(detected=False,
                                     analysis_time_ms=(time.time() - start) * 1000)

        try:
            gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            roi = gray[int(h * 0.7):, :]

            # --- Block-wise variance ---
            block_size = 20
            variances = []
            for y in range(0, roi.shape[0], block_size):
                for x in range(0, roi.shape[1], block_size):
                    block = roi[y:y + block_size, x:x + block_size]
                    if block.size > 0:
                        variances.append(float(np.var(block)))

            if not variances:
                return TactilePavingAlert(detected=False,
                                         analysis_time_ms=(time.time() - start) * 1000)

            avg_var = float(np.mean(variances))

            if avg_var > self._variance_threshold:
                # --- Pattern validation (FFT) ---
                if self._validate_pattern:
                    if not self._validate_tactile_pattern(roi):
                        return TactilePavingAlert(
                            detected=False,
                            analysis_time_ms=(time.time() - start) * 1000)

                # --- Type classification ---
                paving_type = 'warning'
                direction_angle = 0.0
                if self._classify_type:
                    paving_type, direction_angle = self._classify_tactile_type(roi)

                confidence = min(1.0, avg_var / 500.0)
                result = TactilePavingAlert(
                    detected=True, paving_type=paving_type, position='ahead',
                    confidence=confidence, direction_angle=direction_angle,
                    analysis_time_ms=(time.time() - start) * 1000,
                )
                self._total_detections += 1
                self._last_alert_time = time.time()
                return result

            return TactilePavingAlert(detected=False,
                                     analysis_time_ms=(time.time() - start) * 1000)

        except Exception as e:
            print(f"[TactilePavingDetector] ERROR: {e}")
            return TactilePavingAlert(detected=False,
                                     analysis_time_ms=(time.time() - start) * 1000)

    # -------------------------------------------------------- type classification
    def _classify_tactile_type(self, roi_gray: np.ndarray) -> Tuple[str, float]:
        """Classify tactile paving type via gradient orientation histogram.

        Warning paving (bumps): isotropic gradients → uniform histogram.
        Directional paving (grooves): anisotropic → concentrated histogram.

        Returns:
            (paving_type, direction_angle_degrees)
        """
        try:
            gx = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(roi_gray, cv2.CV_64F, 0, 1, ksize=3)
            angles = np.arctan2(gy, gx)

            hist, bin_edges = np.histogram(angles.flatten(), bins=36,
                                           range=(-np.pi, np.pi))
            total = np.sum(hist) + 1e-6
            hist_norm = hist.astype(float) / total

            max_bin = float(np.max(hist_norm))
            max_bin_idx = int(np.argmax(hist_norm))

            # High concentration in one direction → directional paving
            if max_bin > 0.15:
                # Grooves run perpendicular to dominant gradient
                dominant_angle = (bin_edges[max_bin_idx] +
                                  bin_edges[max_bin_idx + 1]) / 2
                direction_deg = ((dominant_angle * 180 / np.pi) + 90) % 360
                return ('directional', float(direction_deg))
            else:
                return ('warning', 0.0)

        except Exception as e:
            print(f"[TactilePavingDetector] Type classification error: {e}")
            return ('warning', 0.0)

    # -------------------------------------------------------- pattern validation
    def _validate_tactile_pattern(self, roi_gray: np.ndarray) -> bool:
        """Validate periodic bump/groove pattern via FFT to filter gravel/grass."""
        try:
            f_transform = np.fft.fft2(roi_gray.astype(float))
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.abs(f_shift)

            h, w = magnitude.shape
            center_y, center_x = h // 2, w // 2

            # Expected bump spacing ≈ 20-30 px at typical distance
            radius_min = max(1, h // 30)
            radius_max = max(radius_min + 1, h // 20)

            y, x = np.ogrid[:h, :w]
            distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            mask = ((distance >= radius_min) & (distance <= radius_max)).astype(float)

            freq_energy = float(np.sum(magnitude * mask))
            total_energy = float(np.sum(magnitude)) + 1e-6

            # Periodic pattern: meaningful energy in the expected frequency band
            if freq_energy / total_energy > 0.01:
                return True

            return False

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
    print("Tactile Paving Detector — Self Test")
    print("=" * 60)

    det = TactilePavingDetector()

    h, w = 480, 640
    frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

    r = det.detect(frame)
    print(f"  Detected={r.detected}  Type={r.paving_type}  Pos={r.position}")
    if r.detected:
        print(f"  Direction={r.direction_angle:.1f}°")
        print(f"  Announcement: {r.get_announcement()}")

    print(f"[Stats] {det.get_stats()}")
    print("Done!")
