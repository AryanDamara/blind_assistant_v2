"""
Enhanced Stair Detector Module
-------------------------------
Detects stairs using depth gradient analysis with 10 major enhancements.

Features:
1.  Temporal consistency — reduces false positives by requiring multi-frame agreement
2.  Edge detection fusion — validates depth patterns with Canny edge evidence
3.  Precise distance estimation — meters instead of "near/mid/far"
4.  Step height estimation — warns about dangerous non-standard steps
5.  Stair type classification — indoor / outdoor / escalator
6.  Landing detection — finds safe rest points between flights
7.  Safety score (0-1) — composite risk for priority-based announcements
8.  Handrail inference — left / right / both / none
9.  Lighting assessment — extra caution in dim / dark conditions
10. Enhanced announcements — minimal / standard / detailed verbosity

Safety Critical: Stairs are the #1 fall risk for visually impaired navigation.

Usage:
    detector = StairDetector(config_path="config/settings.yaml")
    result = detector.detect(depth_map, rgb_frame=frame)

    if result.detected:
        audio.speak(result.get_announcement('detailed'), priority=5)
"""

import time
import yaml
import os
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple

# Optional OpenCV for edge fusion and lighting
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


# ============================================================
# Data Classes
# ============================================================

@dataclass
class StairAlert:
    """Alert generated when stairs are detected."""
    detected: bool
    direction: str = 'unknown'           # 'ascending', 'descending', 'unknown'
    num_steps: int = 0
    distance_region: str = 'unknown'     # 'near', 'mid', 'far' (legacy)
    confidence: float = 0.0              # 0.0 – 1.0
    analysis_time_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    # --- Enhanced fields ---
    estimated_distance_m: float = 0.0    # Precise distance in metres
    step_height_cm: float = 0.0          # Average step height
    stair_type: str = 'unknown'          # 'indoor', 'outdoor', 'escalator', 'unknown'
    has_landing: bool = False            # Flat rest point detected
    safety_score: float = 1.0            # 0 (very dangerous) → 1 (safe)
    handrail_likely: str = 'unknown'     # 'left', 'right', 'both', 'none', 'unknown'
    lighting_condition: str = 'bright'   # 'bright', 'dim', 'dark'
    temporal_consistency: int = 0        # Consecutive frames with detection
    regularity_score: float = 0.0        # Spacing regularity 0-1

    # ---- Announcements ----

    def get_announcement(self, verbosity: str = 'standard') -> str:
        """Generate human-readable announcement.

        Args:
            verbosity: 'minimal', 'standard', or 'detailed'
        """
        if not self.detected:
            return ""

        if verbosity == 'minimal':
            return f"Stairs {self.direction} ahead"

        # --- standard ---
        parts = ["CAUTION! Stairs"]
        if self.direction != 'unknown':
            parts.append(self.direction)
        parts.append("ahead")

        if self.estimated_distance_m > 0:
            if self.estimated_distance_m < 1.0:
                parts.append(f"VERY CLOSE at {int(self.estimated_distance_m * 100)}cm")
            else:
                parts.append(f"at {self.estimated_distance_m:.1f} meters")
        elif self.distance_region == 'near':
            parts.append("very close")

        if self.num_steps > 0:
            parts.append(f"approximately {self.num_steps} steps")

        if verbosity == 'standard':
            return ", ".join(parts) + "!"

        # --- detailed ---
        if self.step_height_cm > 0:
            if self.step_height_cm > 22:
                parts.append(f"WARNING tall steps {int(self.step_height_cm)}cm")
            elif self.step_height_cm < 10:
                parts.append(f"WARNING shallow steps {int(self.step_height_cm)}cm")

        if self.stair_type != 'unknown':
            parts.append(f"{self.stair_type} stairs")

        if self.has_landing:
            parts.append("landing available midway")

        if self.handrail_likely not in ('unknown', 'none'):
            side = self.handrail_likely
            parts.append(f"handrail on your {side}")

        if self.lighting_condition in ('dim', 'dark'):
            parts.append(f"CAUTION {self.lighting_condition} lighting")

        if self.safety_score < 0.5:
            parts.insert(0, "EXTREME CAUTION REQUIRED!")

        return ", ".join(parts) + "!"

    def to_dict(self) -> dict:
        """Convert to dict for telemetry."""
        return {
            'detected': self.detected,
            'direction': self.direction,
            'num_steps': self.num_steps,
            'distance_region': self.distance_region,
            'confidence': round(self.confidence, 3),
            'estimated_distance_m': round(self.estimated_distance_m, 2),
            'step_height_cm': round(self.step_height_cm, 1),
            'stair_type': self.stair_type,
            'has_landing': self.has_landing,
            'safety_score': round(self.safety_score, 2),
            'handrail_likely': self.handrail_likely,
            'lighting_condition': self.lighting_condition,
            'temporal_consistency': self.temporal_consistency,
            'regularity_score': round(self.regularity_score, 3),
            'analysis_time_ms': round(self.analysis_time_ms, 2),
        }


# ============================================================
# Enhanced Stair Detector
# ============================================================

class StairDetector:
    """
    Detects stairs using depth gradient analysis with multi-modal validation.

    Core algorithm:
        1. Extract bottom-70% ROI from depth map
        2. Compute row-to-row depth differences (np.diff)
        3. Find rows where many pixels show large vertical change
        4. Group nearby rows into edge clusters
        5. Check regular spacing (stair rhythm)
        6. Determine direction from gradient sign

    Enhanced features (toggleable via config):
        - Temporal consistency filter
        - Canny edge fusion
        - Precise distance from depth values
        - Step height estimation
        - Stair type classification
        - Landing detection
        - Safety score
        - Handrail inference
        - Lighting assessment
    """

    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize enhanced stair detector."""
        self._load_config(config_path)

        # Cooldown
        self._last_alert_time: float = 0.0

        # Temporal tracking
        self._detection_history: deque = deque(maxlen=10)

        # Statistics
        self._total_analyses: int = 0
        self._total_detections: int = 0

    # ------------------------------------------------------------------ config
    def _load_config(self, config_path: str) -> None:
        """Load stair detection settings from YAML config."""
        # --- defaults (core) ---
        self._enabled: bool = True
        self._gradient_threshold: float = 0.08
        self._min_stair_edges: int = 3
        self._min_row_coverage: float = 0.25
        self._max_step_spacing_px: int = 80
        self._min_step_spacing_px: int = 8
        self._cooldown_sec: float = 5.0
        self._analyze_region: Tuple[float, float] = (0.3, 1.0)

        # --- defaults (enhancements) ---
        self._use_edge_fusion: bool = True
        self._use_temporal_tracking: bool = True
        self._min_temporal_consistency: int = 2
        self._detect_landings: bool = True
        self._estimate_step_height_flag: bool = True
        self._estimate_distance_flag: bool = True
        self._infer_handrail_flag: bool = True
        self._min_safe_distance_m: float = 1.5
        self._max_safe_step_height_cm: float = 22.0
        self._min_safety_score: float = 0.5

        # Load from file
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f) or {}

                sc = config.get('stair_detection', {})
                self._enabled = sc.get('enabled', self._enabled)
                self._gradient_threshold = sc.get('gradient_threshold', self._gradient_threshold)
                self._min_stair_edges = sc.get('min_stair_edges', self._min_stair_edges)
                self._min_row_coverage = sc.get('min_row_coverage', self._min_row_coverage)
                self._max_step_spacing_px = sc.get('max_step_spacing_px', self._max_step_spacing_px)
                self._min_step_spacing_px = sc.get('min_step_spacing_px', self._min_step_spacing_px)
                self._cooldown_sec = sc.get('cooldown_sec', self._cooldown_sec)

                region = sc.get('analyze_region', list(self._analyze_region))
                if isinstance(region, list) and len(region) == 2:
                    self._analyze_region = tuple(region)

                # Enhancement flags
                self._use_edge_fusion = sc.get('use_edge_fusion', self._use_edge_fusion)
                self._use_temporal_tracking = sc.get('use_temporal_tracking', self._use_temporal_tracking)
                self._min_temporal_consistency = sc.get('min_temporal_consistency', self._min_temporal_consistency)
                self._detect_landings = sc.get('detect_landings', self._detect_landings)
                self._estimate_step_height_flag = sc.get('estimate_step_height', self._estimate_step_height_flag)
                self._estimate_distance_flag = sc.get('estimate_distance', self._estimate_distance_flag)
                self._infer_handrail_flag = sc.get('infer_handrail', self._infer_handrail_flag)
                self._min_safe_distance_m = sc.get('min_safe_distance_m', self._min_safe_distance_m)
                self._max_safe_step_height_cm = sc.get('max_safe_step_height_cm', self._max_safe_step_height_cm)

            except Exception as e:
                print(f"[StairDetector] WARNING: Config load error: {e}, using defaults")

        print(f"[StairDetector] Initialized (enabled={self._enabled})")
        print(f"[StairDetector] Gradient threshold: {self._gradient_threshold}")
        print(f"[StairDetector] Min stair edges: {self._min_stair_edges}")

    # ---------------------------------------------------------------- detect()
    def detect(self, depth_map: np.ndarray,
               rgb_frame: Optional[np.ndarray] = None) -> StairAlert:
        """
        Detect stairs in a depth map with optional RGB validation.

        Args:
            depth_map: 2-D numpy array of depth values (metres or relative).
            rgb_frame: Optional BGR image for edge-fusion and lighting checks.

        Returns:
            StairAlert with full detection results.
        """
        start_time = time.time()
        self._total_analyses += 1

        # --- guards ---
        if not self._enabled:
            return StairAlert(detected=False,
                              analysis_time_ms=(time.time() - start_time) * 1000)

        if time.time() - self._last_alert_time < self._cooldown_sec:
            return StairAlert(detected=False,
                              analysis_time_ms=(time.time() - start_time) * 1000)

        if depth_map is None or depth_map.size == 0:
            return StairAlert(detected=False,
                              analysis_time_ms=(time.time() - start_time) * 1000)

        if len(depth_map.shape) == 3:
            depth_map = depth_map[:, :, 0]
        elif len(depth_map.shape) != 2:
            return StairAlert(detected=False,
                              analysis_time_ms=(time.time() - start_time) * 1000)

        try:
            # --- core gradient detection ---
            result = self._analyze_depth_gradient(depth_map)

            # --- enhancements (only if core detected) ---
            if result.detected:
                # Distance estimation
                if self._estimate_distance_flag:
                    result.estimated_distance_m = self._estimate_distance(depth_map)

                # Step height
                if self._estimate_step_height_flag:
                    result.step_height_cm = self._estimate_step_height(depth_map)

                # Stair type
                result.stair_type = self._classify_stair_type(result)

                # Landing
                if self._detect_landings:
                    result.has_landing = self._detect_landing(depth_map)

                # Handrail
                if self._infer_handrail_flag:
                    result.handrail_likely = self._infer_handrail(depth_map)

                # Edge fusion (boosts / reduces confidence)
                if self._use_edge_fusion and rgb_frame is not None and _HAS_CV2:
                    edge_conf = self._analyze_edges(rgb_frame)
                    result.confidence = result.confidence * 0.7 + edge_conf * 0.3

                # Lighting
                if rgb_frame is not None and _HAS_CV2:
                    result.lighting_condition = self._assess_lighting(rgb_frame)

                # Safety score (depends on several enhanced fields)
                result.safety_score = self._calculate_safety_score(result)

            # --- temporal consistency ---
            if self._use_temporal_tracking:
                self._detection_history.append(result.detected)
                recent = sum(self._detection_history)
                result.temporal_consistency = recent

                if result.detected and recent < self._min_temporal_consistency:
                    result.detected = False
                    result.confidence *= 0.5

            result.analysis_time_ms = (time.time() - start_time) * 1000

            if result.detected:
                self._total_detections += 1
                self._last_alert_time = time.time()

            return result

        except Exception as e:
            print(f"[StairDetector] ERROR: {e}")
            return StairAlert(detected=False,
                              analysis_time_ms=(time.time() - start_time) * 1000)

    # ====================================================================
    # CORE ALGORITHM
    # ====================================================================

    def _analyze_depth_gradient(self, depth_map: np.ndarray) -> StairAlert:
        """Core stair detection via depth row-difference analysis."""
        h, w = depth_map.shape

        # 1. Extract ROI
        y_start = int(h * self._analyze_region[0])
        y_end = int(h * self._analyze_region[1])
        roi = depth_map[y_start:y_end, :]

        if roi.size == 0 or roi.shape[0] < 10:
            return StairAlert(detected=False)

        # Normalise to 0-1
        roi_min, roi_max = roi.min(), roi.max()
        if roi_max - roi_min < 1e-6:
            return StairAlert(detected=False)

        roi_norm = (roi - roi_min) / (roi_max - roi_min)

        # 2. Row-to-row differences
        dy = np.diff(roi_norm, axis=0)
        abs_dy = np.abs(dy)

        # 3. Find rows with wide horizontal gradient bands
        significant = abs_dy > self._gradient_threshold
        row_coverage = np.mean(significant, axis=1)
        stair_edge_rows = np.where(row_coverage > self._min_row_coverage)[0]

        if len(stair_edge_rows) < self._min_stair_edges:
            return StairAlert(detected=False)

        # 4. Merge nearby rows into edge groups
        edge_groups = self._merge_nearby_rows(stair_edge_rows, min_gap=3)
        if len(edge_groups) < self._min_stair_edges:
            return StairAlert(detected=False)

        edge_centers = [int(np.mean(g)) for g in edge_groups]

        # 5. Check spacing regularity
        spacings = np.diff(edge_centers)
        if len(spacings) == 0:
            return StairAlert(detected=False)

        valid = spacings[(spacings >= self._min_step_spacing_px) &
                         (spacings <= self._max_step_spacing_px)]
        if len(valid) < 2:
            return StairAlert(detected=False)

        mean_sp = np.mean(valid)
        std_sp = np.std(valid)
        if mean_sp < 1e-6:
            return StairAlert(detected=False)

        regularity = 1.0 - min(std_sp / mean_sp, 1.0)
        if regularity < 0.3:
            return StairAlert(detected=False)

        # 6. Direction
        signs = []
        for c in edge_centers:
            if c < dy.shape[0]:
                signs.append(float(np.mean(dy[c, :])))
        avg_grad = np.mean(signs) if signs else 0.0

        if avg_grad > 0.01:
            direction = 'descending'
        elif avg_grad < -0.01:
            direction = 'ascending'
        else:
            direction = 'unknown'

        num_steps = len(valid) + 1
        confidence = min(1.0,
                         0.3 * min(num_steps / 5, 1.0) +
                         0.4 * regularity +
                         0.3 * float(np.mean(row_coverage[stair_edge_rows])))

        # Legacy distance region
        avg_y = np.mean(edge_centers) / roi.shape[0]
        if avg_y > 0.7:
            distance_region = 'near'
        elif avg_y > 0.4:
            distance_region = 'mid'
        else:
            distance_region = 'far'

        return StairAlert(
            detected=True,
            direction=direction,
            num_steps=num_steps,
            distance_region=distance_region,
            confidence=round(confidence, 3),
            regularity_score=round(regularity, 3),
        )

    # ====================================================================
    # ENHANCEMENT METHODS
    # ====================================================================

    def _estimate_distance(self, depth_map: np.ndarray) -> float:
        """Estimate distance to first step in metres from raw depth."""
        h, w = depth_map.shape
        y_start = int(h * self._analyze_region[0])
        roi = depth_map[y_start:, :]

        dy = np.abs(np.diff(roi, axis=0))
        row_means = np.mean(dy, axis=1)
        high_rows = np.where(row_means > 0.05)[0]

        if len(high_rows) > 0:
            first = high_rows[0]
            dist = float(np.median(roi[first, :]))
            return max(0.3, min(dist, 10.0))
        return 0.0

    def _estimate_step_height(self, depth_map: np.ndarray) -> float:
        """Estimate average step height in centimetres."""
        dy = np.diff(depth_map, axis=0)
        row_means = np.mean(np.abs(dy), axis=1)
        sig = np.where(row_means > 0.05)[0]

        diffs = []
        for i in range(len(sig) - 1):
            if sig[i + 1] - sig[i] < 30:
                d = abs(float(np.median(depth_map[sig[i + 1]])) -
                        float(np.median(depth_map[sig[i]])))
                diffs.append(d)

        if diffs:
            avg = np.mean(diffs)
            cm = avg * 100 * 0.7  # rough projection
            return max(5.0, min(cm, 35.0))
        return 0.0

    def _classify_stair_type(self, alert: StairAlert) -> str:
        """Classify as indoor / outdoor / escalator."""
        if alert.regularity_score > 0.85 and 16 <= alert.step_height_cm <= 20:
            return 'indoor'
        if alert.regularity_score > 0.95 and 16 <= alert.step_height_cm <= 20:
            return 'escalator'
        if 10 <= alert.step_height_cm <= 16:
            return 'outdoor'
        if alert.regularity_score > 0.6:
            return 'indoor'
        return 'unknown'

    def _detect_landing(self, depth_map: np.ndarray) -> bool:
        """Detect flat landing between stair flights."""
        dy = np.abs(np.diff(depth_map, axis=0))
        row_grads = np.mean(dy, axis=1)
        flat = row_grads < 0.02

        run, max_run = 0, 0
        for f in flat:
            if f:
                run += 1
            else:
                max_run = max(max_run, run)
                run = 0
        max_run = max(max_run, run)
        return max_run >= 10

    def _infer_handrail(self, depth_map: np.ndarray) -> str:
        """Infer handrail side from stair width ratio."""
        dy = np.abs(np.diff(depth_map, axis=0))
        stair_mask = dy > 0.05
        col_sums = np.sum(stair_mask, axis=0)
        sig_cols = np.where(col_sums > 3)[0]

        if len(sig_cols) == 0:
            return 'unknown'

        width_ratio = (sig_cols[-1] - sig_cols[0]) / depth_map.shape[1]

        if width_ratio > 0.7:
            return 'both'
        if width_ratio < 0.5:
            w3 = depth_map.shape[1] // 3
            left_var = float(np.var(depth_map[:, :w3]))
            right_var = float(np.var(depth_map[:, 2 * w3:]))
            return 'right' if left_var > right_var else 'left'
        return 'unknown'

    def _analyze_edges(self, rgb_frame: np.ndarray) -> float:
        """Validate depth detections with Canny edge evidence."""
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        h, w = edges.shape
        y_start = int(h * self._analyze_region[0])
        roi = edges[y_start:, :]

        # Look for horizontal edge lines
        kernel = np.ones((1, 30), np.uint8)
        horiz = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)

        row_sums = np.sum(horiz, axis=1)
        sig = np.where(row_sums > w * 0.3 * 255)[0]

        if len(sig) >= 3:
            spacings = np.diff(sig)
            if len(spacings) > 0:
                mean_s = np.mean(spacings)
                if mean_s > 0:
                    reg = 1.0 - min(float(np.std(spacings)) / mean_s, 1.0)
                    if reg > 0.4:
                        return 0.8
        return 0.0

    def _assess_lighting(self, rgb_frame: np.ndarray) -> str:
        """Assess lighting condition from frame brightness."""
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
        avg = float(np.mean(gray))

        if avg < 40:
            return 'dark'
        if avg < 100:
            return 'dim'
        return 'bright'

    def _calculate_safety_score(self, alert: StairAlert) -> float:
        """Calculate composite safety score 0-1 (lower = more dangerous)."""
        score = 1.0

        # Distance
        if alert.estimated_distance_m > 0:
            if alert.estimated_distance_m < 0.5:
                score *= 0.3
            elif alert.estimated_distance_m < 1.0:
                score *= 0.5
            elif alert.estimated_distance_m < self._min_safe_distance_m:
                score *= 0.7

        # Direction
        if alert.direction == 'descending':
            score *= 0.8

        # Step height
        if alert.step_height_cm > 0:
            if alert.step_height_cm > self._max_safe_step_height_cm:
                score *= 0.7
            elif alert.step_height_cm < 10:
                score *= 0.7

        # Regularity
        if alert.regularity_score < 0.5:
            score *= 0.8

        # Lighting
        if alert.lighting_condition == 'dark':
            score *= 0.4
        elif alert.lighting_condition == 'dim':
            score *= 0.7

        # Landing bonus
        if alert.has_landing:
            score = min(1.0, score * 1.1)

        return max(0.0, min(1.0, score))

    # ====================================================================
    # HELPERS
    # ====================================================================

    @staticmethod
    def _merge_nearby_rows(rows: np.ndarray, min_gap: int = 3) -> list:
        """Merge consecutive row indices into groups."""
        if len(rows) == 0:
            return []
        groups = [[rows[0]]]
        for i in range(1, len(rows)):
            if rows[i] - rows[i - 1] <= min_gap:
                groups[-1].append(rows[i])
            else:
                groups.append([rows[i]])
        return groups

    def get_stats(self) -> dict:
        """Get stair detector statistics."""
        return {
            'total_analyses': self._total_analyses,
            'total_detections': self._total_detections,
            'detection_rate': round(
                self._total_detections / max(1, self._total_analyses), 4
            ),
        }

    @property
    def is_enabled(self) -> bool:
        return self._enabled


# ============================================================
# Demo / Self-Test
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Enhanced Stair Detector — Self Test")
    print("=" * 60)

    detector = StairDetector()

    # --- Test 1: Descending stairs ---
    print("\n[Test 1] Synthetic descending stairs …")
    h, w = 100, 160
    dm = np.zeros((h, w), dtype=np.float32)
    for i in range(6):
        y_s = 30 + i * 12
        y_e = y_s + 12
        if y_e <= h:
            dm[y_s:y_e, :] = 2.0 + i * 0.3

    r = detector.detect(dm)
    print(f"  Detected (frame 1, temporal): {r.detected}")
    # Second frame for temporal consistency
    detector._last_alert_time = 0
    r = detector.detect(dm)
    print(f"  Detected (frame 2): {r.detected}")
    if r.detected:
        print(f"  Direction: {r.direction}")
        print(f"  Steps: {r.num_steps}")
        print(f"  Confidence: {r.confidence}")
        print(f"  Distance: {r.estimated_distance_m:.1f}m")
        print(f"  Step height: {r.step_height_cm:.1f}cm")
        print(f"  Stair type: {r.stair_type}")
        print(f"  Landing: {r.has_landing}")
        print(f"  Safety: {r.safety_score:.2f}")
        print(f"  Handrail: {r.handrail_likely}")
        print(f"  Announcement: {r.get_announcement('detailed')}")
    print(f"  Time: {r.analysis_time_ms:.2f}ms")

    # --- Test 2: Flat surface ---
    print("\n[Test 2] Flat surface (should NOT detect) …")
    detector._last_alert_time = 0
    detector._detection_history.clear()
    flat = np.ones((h, w), dtype=np.float32) * 3.0
    flat += np.random.normal(0, 0.01, flat.shape).astype(np.float32)
    r = detector.detect(flat)
    print(f"  Detected: {r.detected} (expected: False)")

    print(f"\n[Stats] {detector.get_stats()}")
    print("\nDone!")
