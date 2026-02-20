"""
Integration Tests
-----------------
Tests the interaction between modules:
- Detection → SafetyManager → Alert pipeline
- Config validation across modules
- Stair detector integration

Runs without camera, model files, or TTS engine.

Run with:
    python -m pytest tests/test_integration.py -v
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from safety_manager import SafetyManager, Alert
from audio_feedback import SpeechItem


# --------------- Helpers ---------------

class MockDetection:
    """Mimics ObjectDetector Detection output."""
    def __init__(self, class_name='person', confidence=0.9,
                 bbox=(200, 50, 440, 470), bbox_center=None, bbox_height=None):
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox
        x1, y1, x2, y2 = bbox
        self.bbox_center = bbox_center or ((x1 + x2) // 2, (y1 + y2) // 2)
        self.bbox_height = bbox_height or (y2 - y1)


# --------------- Detection → Safety → Audio Pipeline ---------------

class TestDetectionToAlertPipeline:
    """End-to-end: detections → safety analysis → speech items."""

    def test_close_person_produces_urgent_speech_item(self):
        """Person at <1m should generate a critical alert suitable for urgent TTS."""
        sm = SafetyManager(config_path="nonexistent.yaml")
        
        det = MockDetection(
            class_name='person', confidence=0.95,
            bbox=(250, 20, 390, 460),
            bbox_center=(320, 240),
            bbox_height=440
        )
        
        result = sm.analyze([det], frame_width=640, frame_id=1)
        assert len(result.alerts) > 0
        
        alert = result.alerts[0]
        # Convert alert to SpeechItem as AudioFeedback would
        text = alert.get_announcement('standard')
        item = SpeechItem(
            text=text,
            priority=alert.priority,
            voice_profile=alert.voice_profile,
            timestamp=time.time()
        )

        assert 'person' in item.text.lower()
        assert item.priority > 0
        assert item.voice_profile in ('urgent', 'alert', 'calm')

    def test_far_object_no_alert(self):
        """Object beyond 5m should not generate alerts."""
        sm = SafetyManager(config_path="nonexistent.yaml")
        
        det = MockDetection(
            class_name='chair', confidence=0.8,
            bbox=(300, 400, 320, 410),  # tiny bbox = far away
            bbox_center=(310, 405),
            bbox_height=10
        )
        
        result = sm.analyze([det], frame_width=640, frame_id=1)
        # Should either have no alerts or only info-level
        critical = [a for a in result.alerts if a.danger_level == 'critical']
        assert len(critical) == 0

    def test_multiple_objects_priority_sorted(self):
        """Multiple detections should produce priority-sorted alerts."""
        sm = SafetyManager(config_path="nonexistent.yaml")
        
        dets = [
            MockDetection('person', 0.9, (250, 20, 390, 460), (320, 240), 440),  # close, center
            MockDetection('chair', 0.7, (50, 300, 100, 380), (75, 340), 80),      # far, left
        ]
        
        result = sm.analyze(dets, frame_width=640, frame_id=1)
        if len(result.alerts) >= 2:
            assert result.alerts[0].priority >= result.alerts[1].priority


# --------------- Config Consistency ---------------

class TestConfigConsistency:
    """Verify that default configs are consistent across modules."""

    def test_safety_voice_profiles_match_audio(self):
        """SafetyManager voice profiles should be valid AudioFeedback profiles."""
        sm = SafetyManager(config_path="nonexistent.yaml")
        af_profiles = {'urgent', 'alert', 'calm'}
        
        for level, cfg in sm._danger_levels.items():
            assert cfg['voice_profile'] in af_profiles, \
                f"Danger level '{level}' has unknown voice_profile '{cfg['voice_profile']}'"

    def test_priority_ordering_consistent(self):
        """Critical priority > warning priority > info priority."""
        sm = SafetyManager(config_path="nonexistent.yaml")
        assert sm._danger_levels['critical']['priority'] > sm._danger_levels['warning']['priority']
        assert sm._danger_levels['warning']['priority'] > sm._danger_levels['info']['priority']

    def test_distance_thresholds_ordered(self):
        """Critical < warning < info distance thresholds."""
        sm = SafetyManager(config_path="nonexistent.yaml")
        assert sm._danger_levels['critical']['distance_m'] < sm._danger_levels['warning']['distance_m']
        assert sm._danger_levels['warning']['distance_m'] < sm._danger_levels['info']['distance_m']


# --------------- Stair Detector Integration ---------------

class TestStairDetectorIntegration:
    """Test stair detector integration with safety pipeline."""

    def test_stair_alert_announcement(self):
        """Stair alert should produce valid announcement text."""
        try:
            from stair_detector import StairDetector, StairAlert
        except ImportError:
            pytest.skip("stair_detector not available")
        
        alert = StairAlert(
            detected=True,
            direction='descending',
            num_steps=5,
            distance_region='near',
            confidence=0.85
        )
        
        text = alert.get_announcement()
        assert 'stairs' in text.lower() or 'STAIRS' in text
        assert 'descending' in text.lower()
        assert len(text) > 0

    def test_stair_alert_to_speech_item(self):
        """Stair alert should convert to a valid SpeechItem."""
        try:
            from stair_detector import StairAlert
        except ImportError:
            pytest.skip("stair_detector not available")
        
        alert = StairAlert(
            detected=True,
            direction='ascending',
            num_steps=3,
            distance_region='mid',
            confidence=0.75
        )
        
        item = SpeechItem(
            text=alert.get_announcement(),
            priority=25,  # Critical priority for stairs
            voice_profile='urgent',
            timestamp=time.time()
        )
        
        assert item.priority == 25
        assert item.voice_profile == 'urgent'
        assert 'stairs' in item.text.lower() or 'STAIRS' in item.text

    def test_no_stair_no_announcement(self):
        """No stairs detected = empty announcement."""
        try:
            from stair_detector import StairAlert
        except ImportError:
            pytest.skip("stair_detector not available")
        
        alert = StairAlert(detected=False)
        assert alert.get_announcement() == ""

    def test_stair_detector_synthetic_depth(self):
        """StairDetector should detect stairs in a synthetic depth map."""
        try:
            import numpy as np
            from stair_detector import StairDetector
        except ImportError:
            pytest.skip("stair_detector or numpy not available")
        
        sd = StairDetector(config_path="nonexistent.yaml")
        
        # Create synthetic stair depth map
        # 6 steps of 12px each starting at row 30
        # ROI starts at row 30 (30% of 100), so steps are fully in analysis region
        h, w = 100, 160
        depth_map = np.zeros((h, w), dtype=np.float32)
        for i in range(6):
            y_start = 30 + i * 12
            y_end = y_start + 12
            if y_end <= h:
                depth_map[y_start:y_end, :] = 2.0 + i * 0.3
        
        result = sd.detect(depth_map)
        assert result.detected
        assert result.num_steps >= 3


# --------------- Dict-format Detection Integration ---------------

class TestDictFormatDetection:
    """Test that dict-format detections work through the full pipeline."""

    def test_dict_detection_to_alert(self):
        sm = SafetyManager(config_path="nonexistent.yaml")
        
        det = {
            'class_name': 'person',
            'confidence': 0.9,
            'bbox': [250, 50, 390, 470],
            'bbox_center': [320, 260],
            'bbox_height': 420
        }
        
        result = sm.analyze([det], frame_width=640, frame_id=1)
        assert result.all_detections_count == 1
        assert len(result.alerts) > 0

    def test_dict_with_class_key(self):
        """Some detectors use 'class' instead of 'class_name'."""
        sm = SafetyManager(config_path="nonexistent.yaml")
        
        det = {
            'class': 'person',
            'confidence': 0.85,
            'bbox': [250, 50, 390, 470],
            'bbox_center': [320, 260],
            'bbox_height': 420
        }
        
        result = sm.analyze([det], frame_width=640, frame_id=1)
        assert result.all_detections_count == 1


# --------------- Result Telemetry ---------------

class TestResultTelemetry:
    """Test that results serialize properly for telemetry."""

    def test_result_to_dict(self):
        sm = SafetyManager(config_path="nonexistent.yaml")
        det = MockDetection('person', 0.9,
                          (250, 50, 390, 470), (320, 260), 420)
        
        result = sm.analyze([det], frame_width=640, frame_id=1)
        d = result.to_dict()
        
        assert 'frame_id' in d
        assert 'alerts' in d
        assert isinstance(d['alerts'], list)
        assert 'analysis_time_ms' in d

    def test_alert_to_dict_serializable(self):
        """Alert.to_dict() should produce JSON-serializable data."""
        import json
        
        sm = SafetyManager(config_path="nonexistent.yaml")
        det = MockDetection('person', 0.9,
                          (250, 50, 390, 470), (320, 260), 420)
        
        result = sm.analyze([det], frame_width=640, frame_id=1)
        if result.alerts:
            json_str = json.dumps(result.alerts[0].to_dict())
            assert len(json_str) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
