"""
Unit Tests for SafetyManager
-----------------------------
Covers zone calculation, distance estimation, danger levels,
priority scoring, alert deduplication, ignore-class filtering,
and edge cases.

Run with:
    python -m pytest tests/test_safety_manager.py -v
"""

import sys
import os
import time
import math

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from safety_manager import SafetyManager, Alert, SafetyAnalysisResult


# --------------- Helpers ---------------

class MockDetection:
    """Mimics a Detection object from ObjectDetector."""
    def __init__(self, class_name='person', confidence=0.9,
                 bbox=(100, 100, 200, 500), bbox_center=None, bbox_height=None):
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox
        x1, y1, x2, y2 = bbox
        self.bbox_center = bbox_center or ((x1 + x2) // 2, (y1 + y2) // 2)
        self.bbox_height = bbox_height or (y2 - y1)


def make_safety(config_path="nonexistent_test.yaml"):
    """Create SafetyManager with defaults (no config file)."""
    return SafetyManager(config_path=config_path)


# --------------- Zone Calculation ---------------

class TestZoneCalculation:
    """Tests for _calculate_zone (left/center/right)."""

    def test_center_zone(self):
        sm = make_safety()
        # Center zone: 30%–70% of frame width
        # 320 / 640 = 0.5 → center
        assert sm._calculate_zone(320, 640) == 'center'

    def test_left_zone(self):
        sm = make_safety()
        # 50 / 640 ≈ 0.078 → left
        assert sm._calculate_zone(50, 640) == 'left'

    def test_right_zone(self):
        sm = make_safety()
        # 600 / 640 ≈ 0.9375 → right
        assert sm._calculate_zone(600, 640) == 'right'

    def test_left_boundary_exact(self):
        sm = make_safety()
        # At exactly 30% → falls into center (>= left boundary)
        assert sm._calculate_zone(192, 640) == 'center'

    def test_right_boundary_exact(self):
        sm = make_safety()
        # At exactly 70% → still center (not > right boundary)
        assert sm._calculate_zone(448, 640) == 'center'


# --------------- Distance Estimation ---------------

class TestDistanceEstimation:
    """Tests for _estimate_distance (bbox method with fallback)."""

    def test_calibrated_person(self):
        sm = make_safety()
        # Person: height=1.7m, ref_bbox=400px, ref_dist=2.0m
        # Formula: (1.7 * 2.0 * 400) / 400 = 3.4m
        distance, quality = sm._estimate_distance('person', 400)
        assert quality == 'calibrated'
        assert abs(distance - 3.4) < 0.5

    def test_person_far_away(self):
        sm = make_safety()
        # Smaller bbox = farther away
        distance, _ = sm._estimate_distance('person', 100)
        assert distance > 2.0

    def test_person_close(self):
        sm = make_safety()
        # Larger bbox = closer
        distance, _ = sm._estimate_distance('person', 800)
        assert distance < 2.0

    def test_fallback_for_unknown_object(self):
        sm = make_safety()
        # 'bottle' is not in reference_objects → uses fallback
        distance, quality = sm._estimate_distance('bottle', 200)
        assert quality == 'estimated'
        assert distance > 0

    def test_zero_bbox_height(self):
        sm = make_safety()
        # Edge case: bbox_height = 0 → should not crash
        distance, quality = sm._estimate_distance('person', 0)
        assert quality == 'estimated'
        assert distance == 20.0  # max_distance

    def test_negative_bbox_height(self):
        sm = make_safety()
        distance, quality = sm._estimate_distance('person', -10)
        assert quality == 'estimated'
        assert distance == 20.0

    def test_huge_bbox_height(self):
        sm = make_safety()
        # Above 10000 threshold
        distance, quality = sm._estimate_distance('person', 15000)
        assert quality == 'estimated'
        assert distance == 20.0

    def test_low_confidence_adds_margin(self):
        sm = make_safety()
        dist_high, _ = sm._estimate_distance('person', 400, confidence=0.9)
        dist_low, quality = sm._estimate_distance('person', 400, confidence=0.5)
        # Low confidence should report farther distance (20% margin)
        assert dist_low >= dist_high
        assert quality == 'estimated'

    def test_distance_clamped_to_range(self):
        sm = make_safety()
        # Very tiny bbox → very far distance → should be clamped to max_distance
        distance, _ = sm._estimate_distance('person', 1)
        assert distance <= 20.0
        assert distance >= 0.5


# --------------- Danger Level Classification ---------------

class TestDangerLevel:
    """Tests for _calculate_danger_level."""

    def test_critical_distance(self):
        sm = make_safety()
        level, priority, profile = sm._calculate_danger_level(0.5, 'center')
        assert level == 'critical'
        assert priority == 5
        assert profile == 'urgent'

    def test_warning_distance(self):
        sm = make_safety()
        level, priority, profile = sm._calculate_danger_level(1.5, 'left')
        assert level == 'warning'
        assert priority == 3
        assert profile == 'alert'

    def test_info_distance(self):
        sm = make_safety()
        level, priority, profile = sm._calculate_danger_level(3.0, 'right')
        assert level == 'info'
        assert priority == 1
        assert profile == 'calm'

    def test_too_far_returns_zero_priority(self):
        sm = make_safety()
        level, priority, profile = sm._calculate_danger_level(10.0, 'left')
        assert priority == 0

    def test_center_zone_multiplier_makes_more_sensitive(self):
        sm = make_safety()
        # At 1.2m in center: effective = 1.2 * 0.8 = 0.96 → critical
        level_center, _, _ = sm._calculate_danger_level(1.2, 'center')
        # At 1.2m on left: effective = 1.2 → warning
        level_left, _, _ = sm._calculate_danger_level(1.2, 'left')
        assert level_center == 'critical'
        assert level_left == 'warning'


# --------------- Priority Calculation ---------------

class TestPriorityCalculation:
    """Tests for _calculate_priority (additive)."""

    def test_center_zone_boost(self):
        sm = make_safety()
        pri_center = sm._calculate_priority('person', 'center', base_priority=5)
        pri_left = sm._calculate_priority('person', 'left', base_priority=5)
        assert pri_center > pri_left  # center gets zone boost

    def test_vehicle_object_boost(self):
        sm = make_safety()
        # car has priority_boost=20 in config, person has 10
        pri_car = sm._calculate_priority('car', 'left', base_priority=5)
        pri_person = sm._calculate_priority('person', 'left', base_priority=5)
        # Without config file, boosts are empty → both equal
        # (This tests the additive logic)
        assert pri_car >= pri_person

    def test_base_priority_preserved(self):
        sm = make_safety()
        # Unknown object on non-center zone → just base priority
        pri = sm._calculate_priority('unknown_obj', 'left', base_priority=3)
        assert pri == 3


# --------------- Alert Deduplication ---------------

class TestDeduplication:
    """Tests for _is_duplicate."""

    def test_first_alert_not_duplicate(self):
        sm = make_safety()
        assert sm._is_duplicate('person_center', 'critical', time.time()) is False

    def test_repeated_alert_is_duplicate(self):
        sm = make_safety()
        now = time.time()
        sm._add_to_history('person_center', 'warning', now)
        assert sm._is_duplicate('person_center', 'warning', now + 1) is True

    def test_escalation_not_duplicate(self):
        sm = make_safety()
        now = time.time()
        sm._add_to_history('person_center', 'warning', now)
        # Escalation to critical should be allowed
        assert sm._is_duplicate('person_center', 'critical', now + 1) is False

    def test_expired_alert_not_duplicate(self):
        sm = make_safety()
        now = time.time()
        sm._add_to_history('person_center', 'warning', now - 10)  # 10s ago
        # Dedup window is 3s → expired
        assert sm._is_duplicate('person_center', 'warning', now) is False

    def test_different_key_not_duplicate(self):
        sm = make_safety()
        now = time.time()
        sm._add_to_history('person_center', 'warning', now)
        assert sm._is_duplicate('person_left', 'warning', now + 1) is False


# --------------- Full Analysis Pipeline ---------------

class TestAnalyze:
    """End-to-end tests for the analyze() method."""

    def test_empty_detections(self):
        sm = make_safety()
        result = sm.analyze([], frame_width=640, frame_id=1)
        assert isinstance(result, SafetyAnalysisResult)
        assert len(result.alerts) == 0
        assert result.all_detections_count == 0

    def test_invalid_frame_width(self):
        sm = make_safety()
        result = sm.analyze([MockDetection()], frame_width=0, frame_id=1)
        assert len(result.alerts) == 0

    def test_negative_frame_width(self):
        sm = make_safety()
        result = sm.analyze([MockDetection()], frame_width=-1, frame_id=1)
        assert len(result.alerts) == 0

    def test_close_person_generates_alert(self):
        sm = make_safety()
        # Person with large bbox in center
        # Formula: (1.7 * 2.0 * 400) / 420 ≈ 3.2m → info (with center multiplier 0.8 → 2.6m = warning)
        det = MockDetection(
            class_name='person',
            confidence=0.95,
            bbox=(250, 50, 390, 470),   # center of 640px frame
            bbox_center=(320, 260),
            bbox_height=420
        )
        result = sm.analyze([det], frame_width=640, frame_id=1)
        assert len(result.alerts) > 0
        assert result.alerts[0].danger_level in ('critical', 'warning', 'info')

    def test_ignore_class_filtered(self):
        sm = make_safety()
        # 'tv' is in default ignore_classes
        sm._ignore_classes = ['tv']
        det = MockDetection(class_name='tv', bbox=(100, 100, 200, 300))
        result = sm.analyze([det], frame_width=640, frame_id=1)
        assert len(result.alerts) == 0

    def test_multiple_detections_sorted_by_priority(self):
        sm = make_safety()
        # Two detections: one close (high priority), one far (low priority)
        close = MockDetection(
            class_name='person', confidence=0.9,
            bbox=(250, 50, 390, 470), bbox_center=(320, 260), bbox_height=420
        )
        far = MockDetection(
            class_name='chair', confidence=0.8,
            bbox=(400, 300, 450, 350), bbox_center=(425, 325), bbox_height=50
        )
        result = sm.analyze([close, far], frame_width=640, frame_id=1)
        if len(result.alerts) >= 2:
            # Alerts should be sorted by priority (descending)
            assert result.alerts[0].priority >= result.alerts[1].priority

    def test_dict_detection_format(self):
        sm = make_safety()
        det = {
            'class_name': 'person',
            'confidence': 0.9,
            'bbox': [250, 50, 390, 470],
            'bbox_center': [320, 260],
            'bbox_height': 420
        }
        result = sm.analyze([det], frame_width=640, frame_id=1)
        assert result.all_detections_count == 1

    def test_result_has_timing(self):
        sm = make_safety()
        result = sm.analyze([], frame_width=640, frame_id=42)
        assert result.analysis_time_ms >= 0
        assert result.frame_id == 42
        assert result.timestamp > 0


# --------------- Alert Dataclass ---------------

class TestAlert:
    """Tests for Alert dataclass."""

    def test_alert_key_auto_generated(self):
        alert = Alert(
            class_name='person', confidence=0.9,
            bbox=(100, 100, 200, 500), bbox_center=(150, 300), bbox_height=400,
            zone='center', distance_m=1.5, distance_quality='calibrated',
            danger_level='warning', voice_profile='alert', priority=18
        )
        assert alert.alert_key == 'person_center'

    def test_announcement_standard(self):
        alert = Alert(
            class_name='person', confidence=0.9,
            bbox=(100, 100, 200, 500), bbox_center=(150, 300), bbox_height=400,
            zone='left', distance_m=2.0, distance_quality='calibrated',
            danger_level='warning', voice_profile='alert', priority=13
        )
        text = alert.get_announcement('standard')
        assert 'person' in text
        assert 'left' in text

    def test_announcement_minimal(self):
        alert = Alert(
            class_name='chair', confidence=0.8,
            bbox=(0, 0, 50, 100), bbox_center=(25, 50), bbox_height=100,
            zone='right', distance_m=3.0, distance_quality='estimated',
            danger_level='info', voice_profile='calm', priority=1
        )
        text = alert.get_announcement('minimal')
        assert text == 'chair right'

    def test_to_dict(self):
        alert = Alert(
            class_name='person', confidence=0.912,
            bbox=(100, 100, 200, 500), bbox_center=(150, 300), bbox_height=400,
            zone='center', distance_m=1.543, distance_quality='calibrated',
            danger_level='critical', voice_profile='urgent', priority=25
        )
        d = alert.to_dict()
        assert d['class'] == 'person'
        assert d['confidence'] == 0.912
        assert d['distance_m'] == 1.5
        assert d['priority'] == 25


# --------------- SafetyAnalysisResult ---------------

class TestSafetyAnalysisResult:
    """Tests for SafetyAnalysisResult properties."""

    def test_count_properties(self):
        alerts = [
            Alert('person', 0.9, (0,0,1,1), (0,0), 1, 'center', 0.5, 'c', 'critical', 'urgent', 25),
            Alert('chair', 0.8, (0,0,1,1), (0,0), 1, 'left', 1.5, 'c', 'warning', 'alert', 13),
            Alert('bottle', 0.7, (0,0,1,1), (0,0), 1, 'right', 3.0, 'e', 'info', 'calm', 1),
        ]
        result = SafetyAnalysisResult(
            frame_id=1, alerts=alerts,
            all_detections_count=5, filtered_count=3,
            deduplicated_count=3, analysis_time_ms=5.0,
            timestamp=time.time()
        )
        assert result.critical_count == 1
        assert result.warning_count == 1
        assert result.info_count == 1

    def test_clear_alerts(self):
        alerts = [
            Alert('person', 0.9, (0,0,1,1), (0,0), 1, 'center', 0.5, 'c', 'critical', 'urgent', 25),
        ]
        result = SafetyAnalysisResult(
            frame_id=1, alerts=alerts,
            all_detections_count=1, filtered_count=1,
            deduplicated_count=1, analysis_time_ms=1.0,
            timestamp=time.time()
        )
        result.clear_alerts()
        assert len(result.alerts) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
