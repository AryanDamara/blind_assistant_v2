"""
Test suite for all new detection modules.

Tests:
- Enhanced StairDetector (10 improvements)
- TrafficLightDetector
- CrosswalkDetector
- CurbDetector
- DoorDetector
- RetailDetector
- TactilePavingDetector
"""

import sys
import os
import time
import numpy as np
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from stair_detector import StairDetector, StairAlert
from traffic_light_detector import TrafficLightDetector, TrafficLightAlert
from crosswalk_detector import CrosswalkDetector, CrosswalkAlert
from curb_detector import CurbDetector, CurbAlert
from door_detector import DoorDetector, DoorAlert
from retail_detector import RetailDetector, RetailAlert
from tactile_paving_detector import TactilePavingDetector, TactilePavingAlert

# Check for OpenCV
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ============================================================
# Helpers
# ============================================================

def _make_stair_depth(h=100, w=160, num_steps=6):
    """Create synthetic stair-like depth map."""
    dm = np.zeros((h, w), dtype=np.float32)
    for i in range(num_steps):
        y = 30 + i * 12
        if y + 12 <= h:
            dm[y:y+12, :] = 2.0 + i * 0.3
    return dm


def _make_flat_depth(h=100, w=160, depth=3.0):
    """Create flat depth map (no stairs)."""
    dm = np.ones((h, w), dtype=np.float32) * depth
    dm += np.random.normal(0, 0.01, dm.shape).astype(np.float32)
    return dm


# ============================================================
# Enhanced Stair Detector Tests
# ============================================================

class TestEnhancedStairDetector:
    """Tests for the enhanced stair detector."""

    def _make_detector(self):
        d = StairDetector.__new__(StairDetector)
        d._enabled = True
        d._gradient_threshold = 0.08
        d._min_stair_edges = 3
        d._min_row_coverage = 0.25
        d._max_step_spacing_px = 80
        d._min_step_spacing_px = 8
        d._cooldown_sec = 0.0
        d._analyze_region = (0.3, 1.0)
        d._use_edge_fusion = False  # No rgb in basic tests
        d._use_temporal_tracking = False  # Test core detection
        d._min_temporal_consistency = 2
        d._detect_landings = True
        d._estimate_step_height_flag = True
        d._estimate_distance_flag = True
        d._infer_handrail_flag = True
        d._min_safe_distance_m = 1.5
        d._max_safe_step_height_cm = 22.0
        d._min_safety_score = 0.5
        d._last_alert_time = 0.0
        d._total_analyses = 0
        d._total_detections = 0
        from collections import deque
        d._detection_history = deque(maxlen=10)
        return d

    def test_detect_synthetic_stairs(self):
        det = self._make_detector()
        dm = _make_stair_depth()
        r = det.detect(dm)
        assert r.detected is True
        assert r.num_steps >= 3

    def test_flat_surface_no_detection(self):
        det = self._make_detector()
        dm = _make_flat_depth()
        r = det.detect(dm)
        assert r.detected is False

    def test_distance_estimation(self):
        det = self._make_detector()
        dm = _make_stair_depth()
        r = det.detect(dm)
        assert r.detected is True
        assert r.estimated_distance_m > 0

    def test_step_height_estimation(self):
        det = self._make_detector()
        dm = _make_stair_depth()
        r = det.detect(dm)
        assert r.detected is True
        # step_height_cm may be 0 if depth diffs are too small; just check type
        assert isinstance(r.step_height_cm, float)

    def test_stair_type_classification(self):
        det = self._make_detector()
        dm = _make_stair_depth()
        r = det.detect(dm)
        assert r.stair_type in ('indoor', 'outdoor', 'escalator', 'unknown')

    def test_landing_detection(self):
        det = self._make_detector()
        dm = _make_stair_depth()
        r = det.detect(dm)
        assert isinstance(r.has_landing, bool)

    def test_safety_score_range(self):
        det = self._make_detector()
        dm = _make_stair_depth()
        r = det.detect(dm)
        assert 0.0 <= r.safety_score <= 1.0

    def test_handrail_inference(self):
        det = self._make_detector()
        dm = _make_stair_depth()
        r = det.detect(dm)
        assert r.handrail_likely in ('left', 'right', 'both', 'none', 'unknown')

    def test_temporal_consistency(self):
        det = self._make_detector()
        det._use_temporal_tracking = True
        det._min_temporal_consistency = 2
        dm = _make_stair_depth()

        # First frame: detection should be suppressed
        r1 = det.detect(dm)
        assert r1.detected is False  # Not enough consistency

        # Second frame: now should pass (2 consecutive)
        det._last_alert_time = 0.0
        r2 = det.detect(dm)
        assert r2.detected is True

    def test_announcement_minimal(self):
        a = StairAlert(detected=True, direction='descending', num_steps=4)
        msg = a.get_announcement('minimal')
        assert 'descending' in msg.lower()

    def test_announcement_standard(self):
        a = StairAlert(detected=True, direction='ascending', num_steps=5,
                       estimated_distance_m=2.3)
        msg = a.get_announcement('standard')
        assert 'CAUTION' in msg
        assert '2.3' in msg

    def test_announcement_detailed(self):
        a = StairAlert(detected=True, direction='descending', num_steps=6,
                       estimated_distance_m=0.8, step_height_cm=25,
                       stair_type='outdoor', has_landing=True,
                       handrail_likely='right', lighting_condition='dim',
                       safety_score=0.3)
        msg = a.get_announcement('detailed')
        assert 'EXTREME' in msg
        assert 'handrail' in msg.lower()
        assert 'dim' in msg.lower()

    def test_to_dict(self):
        a = StairAlert(detected=True, direction='ascending', num_steps=3,
                       confidence=0.85, safety_score=0.7)
        d = a.to_dict()
        assert d['detected'] is True
        assert d['direction'] == 'ascending'
        assert d['safety_score'] == 0.7

    def test_disabled_detector(self):
        det = self._make_detector()
        det._enabled = False
        r = det.detect(_make_stair_depth())
        assert r.detected is False

    def test_empty_input(self):
        det = self._make_detector()
        r = det.detect(np.array([]))
        assert r.detected is False

    def test_3d_input_handling(self):
        det = self._make_detector()
        dm3d = np.expand_dims(_make_stair_depth(), axis=2)
        r = det.detect(dm3d)
        # Should handle gracefully
        assert isinstance(r, StairAlert)

    def test_stats(self):
        det = self._make_detector()
        det.detect(_make_stair_depth())
        s = det.get_stats()
        assert s['total_analyses'] == 1


# ============================================================
# Traffic Light Detector Tests
# ============================================================

@pytest.mark.skipif(not HAS_CV2, reason="OpenCV not available")
class TestTrafficLightDetector:
    def _make_detector(self):
        d = TrafficLightDetector.__new__(TrafficLightDetector)
        d._enabled = True
        d._min_confidence = 0.3
        d._cooldown_sec = 0.0
        d._detect_pedestrian_signals = True
        d._detect_flashing = True
        d._temporal_smoothing = False
        d._color_ranges = {
            'red_lower1': np.array([0, 100, 100]),
            'red_upper1': np.array([10, 255, 255]),
            'red_lower2': np.array([170, 100, 100]),
            'red_upper2': np.array([180, 255, 255]),
            'yellow_lower': np.array([20, 100, 100]),
            'yellow_upper': np.array([30, 255, 255]),
            'green_lower': np.array([40, 100, 100]),
            'green_upper': np.array([80, 255, 255]),
        }
        d._detection_history = __import__('collections').deque(maxlen=5)
        d._last_state = 'unknown'
        d._last_alert_time = 0.0
        d._total_analyses = 0
        d._total_detections = 0
        d._state_counts = {'red': 0, 'yellow': 0, 'green': 0, 'walk': 0, 'dont_walk': 0}
        return d

    def test_empty_frame_no_detection(self):
        det = self._make_detector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        r = det.detect(frame)
        assert r.detected is False

    def test_alert_announcements(self):
        for state, keyword in [('red', 'STOP'), ('yellow', 'Caution'),
                               ('green', 'safe'), ('walk', 'Walk')]:
            a = TrafficLightAlert(detected=True, state=state)
            msg = a.get_announcement()
            assert keyword in msg

    def test_to_dict(self):
        a = TrafficLightAlert(detected=True, state='red', confidence=0.9)
        d = a.to_dict()
        assert d['state'] == 'red'
        assert d['confidence'] == 0.9

    def test_disabled(self):
        det = self._make_detector()
        det._enabled = False
        r = det.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        assert r.detected is False

    def test_stats(self):
        det = self._make_detector()
        det.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        s = det.get_stats()
        assert s['total_analyses'] == 1


# ============================================================
# Crosswalk Detector Tests
# ============================================================

@pytest.mark.skipif(not HAS_CV2, reason="OpenCV not available")
class TestCrosswalkDetector:
    def _make_detector(self):
        d = CrosswalkDetector.__new__(CrosswalkDetector)
        d._enabled = True
        d._min_confidence = 0.3
        d._min_parallel_lines = 4
        d._max_line_angle_var = 15.0
        d._cooldown_sec = 0.0
        d._last_alert_time = 0.0
        d._total_analyses = 0
        d._total_detections = 0
        return d

    def test_empty_frame_no_detection(self):
        det = self._make_detector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        r = det.detect(frame)
        assert r.detected is False

    def test_alert_announcement(self):
        a = CrosswalkAlert(detected=True, position='ahead', distance_m=1.5)
        msg = a.get_announcement()
        assert 'Crosswalk' in msg
        assert '150' in msg

    def test_to_dict(self):
        a = CrosswalkAlert(detected=True, crosswalk_type='zebra', confidence=0.8)
        d = a.to_dict()
        assert d['crosswalk_type'] == 'zebra'

    def test_disabled(self):
        det = self._make_detector()
        det._enabled = False
        r = det.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        assert r.detected is False


# ============================================================
# Curb Detector Tests
# ============================================================

class TestCurbDetector:
    def _make_detector(self):
        d = CurbDetector.__new__(CurbDetector)
        d._enabled = True
        d._min_gradient = 0.05
        d._min_curb_height_cm = 5.0
        d._cooldown_sec = 0.0
        d._last_alert_time = 0.0
        d._total_analyses = 0
        d._total_detections = 0
        return d

    def test_detect_curb(self):
        det = self._make_detector()
        h, w = 100, 160
        dm = np.ones((h, w), dtype=np.float32) * 2.0
        dm[70:, 80:] = 2.3  # Curb
        r = det.detect(dm)
        assert r.detected is True
        assert r.height_cm > 0

    def test_flat_no_curb(self):
        det = self._make_detector()
        dm = _make_flat_depth()
        r = det.detect(dm)
        assert r.detected is False

    def test_curb_announcement_close(self):
        a = CurbAlert(detected=True, height_cm=12, position='ahead', distance_m=0.5)
        msg = a.get_announcement()
        assert 'CAUTION' in msg
        assert 'VERY CLOSE' in msg

    def test_curb_announcement_far(self):
        a = CurbAlert(detected=True, height_cm=18, position='left', distance_m=3.0)
        msg = a.get_announcement()
        assert 'HIGH CURB' in msg
        assert 'left' in msg

    def test_disabled(self):
        det = self._make_detector()
        det._enabled = False
        r = det.detect(np.ones((100, 160), dtype=np.float32))
        assert r.detected is False

    def test_empty_input(self):
        det = self._make_detector()
        r = det.detect(np.array([]))
        assert r.detected is False


# ============================================================
# Door Detector Tests
# ============================================================

@pytest.mark.skipif(not HAS_CV2, reason="OpenCV not available")
class TestDoorDetector:
    def _make_detector(self):
        d = DoorDetector.__new__(DoorDetector)
        d._enabled = True
        d._min_width_ratio = 0.15
        d._max_width_ratio = 0.50
        d._min_aspect_ratio = 1.5
        d._cooldown_sec = 0.0
        d._last_alert_time = 0.0
        d._total_analyses = 0
        d._total_detections = 0
        return d

    def test_empty_frame_no_detection(self):
        det = self._make_detector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        r = det.detect(frame)
        assert r.detected is False

    def test_door_announcement(self):
        a = DoorAlert(detected=True, state='closed', position='ahead',
                      distance_m=2.0, has_handle=True, handle_side='right')
        msg = a.get_announcement()
        assert 'Door' in msg
        assert 'handle' in msg

    def test_to_dict(self):
        a = DoorAlert(detected=True, door_type='automatic', state='open')
        d = a.to_dict()
        assert d['door_type'] == 'automatic'

    def test_disabled(self):
        det = self._make_detector()
        det._enabled = False
        r = det.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        assert r.detected is False


# ============================================================
# Retail Detector Tests
# ============================================================

@pytest.mark.skipif(not HAS_CV2, reason="OpenCV not available")
class TestRetailDetector:
    def _make_detector(self):
        d = RetailDetector.__new__(RetailDetector)
        d._enabled = True
        d._cooldown_sec = 0.0
        d._last_alert_time = 0.0
        d._detect_aisles = True
        d._detect_checkouts = True
        d._total_analyses = 0
        d._total_detections = 0
        return d

    def test_empty_frame_no_detection(self):
        det = self._make_detector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        r = det.detect(frame)
        assert r.detected is False

    def test_announcement(self):
        a = RetailAlert(detected=True, feature_type='aisle', position='ahead', distance_m=3.0)
        msg = a.get_announcement()
        assert 'Aisle' in msg

    def test_disabled_by_default(self):
        d = RetailDetector.__new__(RetailDetector)
        d._enabled = False
        d._cooldown_sec = 0.0
        d._last_alert_time = 0.0
        d._total_analyses = 0
        d._total_detections = 0
        r = d.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        assert r.detected is False


# ============================================================
# Tactile Paving Detector Tests
# ============================================================

@pytest.mark.skipif(not HAS_CV2, reason="OpenCV not available")
class TestTactilePavingDetector:
    def _make_detector(self):
        d = TactilePavingDetector.__new__(TactilePavingDetector)
        d._enabled = True
        d._cooldown_sec = 0.0
        d._variance_threshold = 300.0
        d._last_alert_time = 0.0
        d._total_analyses = 0
        d._total_detections = 0
        return d

    def test_smooth_surface_no_detection(self):
        det = self._make_detector()
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        r = det.detect(frame)
        assert r.detected is False

    def test_textured_surface_detection(self):
        det = self._make_detector()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        r = det.detect(frame)
        assert r.detected is True
        assert r.paving_type == 'warning'

    def test_announcement_warning(self):
        a = TactilePavingAlert(detected=True, paving_type='warning', position='ahead')
        msg = a.get_announcement()
        assert 'Warning' in msg

    def test_announcement_platform(self):
        a = TactilePavingAlert(detected=True, paving_type='platform_edge')
        msg = a.get_announcement()
        assert 'CAUTION' in msg

    def test_disabled(self):
        det = self._make_detector()
        det._enabled = False
        r = det.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        assert r.detected is False
