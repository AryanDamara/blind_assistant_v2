"""
Shopping / Retail Detector Module
----------------------------------
Detects retail environment features for indoor navigation.

Features:
- Aisle detection (parallel vertical lines from shelving)
- Checkout counter detection (horizontal edges at waist height)
"""

import time
import yaml
import os
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RetailAlert:
    """Alert for retail environment features."""
    detected: bool
    feature_type: str = 'unknown'
    position: str = 'unknown'
    width_m: float = 0.0
    distance_m: float = 0.0
    confidence: float = 0.0
    analysis_time_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def get_announcement(self) -> str:
        if not self.detected:
            return ""
        if self.distance_m > 0:
            return f"{self.feature_type.title()} {self.position} at {self.distance_m:.1f} meters"
        return f"{self.feature_type.title()} {self.position}"

    def to_dict(self) -> dict:
        return {
            'detected': self.detected, 'feature_type': self.feature_type,
            'position': self.position, 'distance_m': round(self.distance_m, 2),
            'confidence': round(self.confidence, 3),
            'analysis_time_ms': round(self.analysis_time_ms, 2),
        }


class RetailDetector:
    """Detects retail features: aisles and checkout counters."""

    def __init__(self, config_path: str = "config/settings.yaml"):
        self._enabled = False  # Off by default
        self._cooldown_sec = 5.0
        self._last_alert_time = 0.0
        self._detect_aisles = True
        self._detect_checkouts = True
        self._total_analyses = 0
        self._total_detections = 0

        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    cfg = yaml.safe_load(f) or {}
                rc = cfg.get('retail_detection', {})
                self._enabled = rc.get('enabled', False)
                self._detect_aisles = rc.get('detect_aisles', True)
                self._detect_checkouts = rc.get('detect_checkouts', True)
                self._cooldown_sec = rc.get('cooldown_sec', 5.0)
            except Exception as e:
                print(f"[RetailDetector] WARNING: Config error: {e}")
        print(f"[RetailDetector] Initialized (enabled={self._enabled})")

    def detect(self, rgb_frame: np.ndarray, depth_map: Optional[np.ndarray] = None) -> RetailAlert:
        start = time.time()
        self._total_analyses += 1
        if not self._enabled or rgb_frame is None or rgb_frame.size == 0:
            return RetailAlert(detected=False, analysis_time_ms=(time.time()-start)*1000)
        if time.time() - self._last_alert_time < self._cooldown_sec:
            return RetailAlert(detected=False, analysis_time_ms=(time.time()-start)*1000)
        try:
            if self._detect_aisles:
                a = self._detect_aisle(rgb_frame)
                if a.detected:
                    a.analysis_time_ms = (time.time()-start)*1000
                    self._total_detections += 1
                    self._last_alert_time = time.time()
                    return a
            if self._detect_checkouts:
                c = self._detect_checkout(rgb_frame)
                if c.detected:
                    c.analysis_time_ms = (time.time()-start)*1000
                    self._total_detections += 1
                    self._last_alert_time = time.time()
                    return c
            return RetailAlert(detected=False, analysis_time_ms=(time.time()-start)*1000)
        except Exception as e:
            print(f"[RetailDetector] ERROR: {e}")
            return RetailAlert(detected=False, analysis_time_ms=(time.time()-start)*1000)

    def _detect_aisle(self, rgb_frame) -> RetailAlert:
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=20)
        if lines is None or len(lines) < 4:
            return RetailAlert(detected=False)
        vert = [l for l in lines if 80 < abs(np.arctan2(l[0][3]-l[0][1], l[0][2]-l[0][0])*180/np.pi) < 100]
        if len(vert) >= 4:
            return RetailAlert(detected=True, feature_type='aisle', position='ahead', confidence=0.7)
        return RetailAlert(detected=False)

    def _detect_checkout(self, rgb_frame) -> RetailAlert:
        h, w = rgb_frame.shape[:2]
        roi = rgb_frame[int(h*0.4):int(h*0.6), :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=200, maxLineGap=20)
        if lines is not None and len(lines) > 0:
            return RetailAlert(detected=True, feature_type='checkout', position='ahead', confidence=0.6)
        return RetailAlert(detected=False)

    def get_stats(self) -> dict:
        return {'total_analyses': self._total_analyses, 'total_detections': self._total_detections,
                'detection_rate': round(self._total_detections / max(1, self._total_analyses), 4)}

    @property
    def is_enabled(self) -> bool:
        return self._enabled
