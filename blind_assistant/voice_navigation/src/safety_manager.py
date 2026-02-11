"""
Safety Manager Module
---------------------
Analyzes detections for navigation safety.

Features:
- Zone detection (left/center/right)
- Distance estimation (bounding box method with fallback)
- Danger level classification (critical/warning/info)
- Voice profile assignment for audio feedback
- Priority calculation (additive: base + object boost + zone boost)
- Alert deduplication (class + zone within time window)
- Filters out ignore_classes
"""

import time
import yaml
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import deque


@dataclass
class Alert:
    """
    Safety alert for a detected object.
    Includes zone, distance, voice profile, and priority for audio output.
    """
    class_name: str               # Object class (e.g., 'person', 'chair')
    confidence: float             # Detection confidence
    bbox: Tuple[int, int, int, int]  # Bounding box (x1, y1, x2, y2)
    bbox_center: Tuple[int, int]  # Center point (x, y)
    bbox_height: int              # Height in pixels
    
    # Safety analysis results
    zone: str                     # 'left', 'center', 'right'
    distance_m: float             # Estimated distance in meters
    distance_quality: str         # 'calibrated' or 'estimated'
    danger_level: str             # 'critical', 'warning', 'info'
    voice_profile: str            # 'urgent', 'alert', 'calm' (for audio_feedback)
    priority: int                 # Final priority (higher = more urgent)
    
    # For deduplication
    alert_key: str = field(default="")  # class_zone key
    
    def __post_init__(self):
        """Generate alert key after initialization."""
        if not self.alert_key:
            self.alert_key = f"{self.class_name}_{self.zone}"
    
    def to_dict(self) -> dict:
        """Convert to dict for telemetry logging."""
        return {
            'class': self.class_name,
            'confidence': round(self.confidence, 3),
            'bbox': list(self.bbox),
            'zone': self.zone,
            'distance_m': round(self.distance_m, 1),
            'distance_quality': self.distance_quality,
            'danger_level': self.danger_level,
            'voice_profile': self.voice_profile,
            'priority': self.priority,
            'alert_key': self.alert_key
        }
    
    def get_announcement(self, verbosity: str = "standard") -> str:
        """
        Generate human-readable announcement text.
        
        Args:
            verbosity: 'minimal', 'standard', or 'detailed'
        """
        if verbosity == "minimal":
            return f"{self.class_name} {self.zone}"
        
        elif verbosity == "standard":
            if self.distance_quality == "estimated":
                return f"{self.class_name} approximately {self.distance_m:.1f} meters on your {self.zone}"
            else:
                return f"{self.class_name} {self.distance_m:.1f} meters on your {self.zone}"
        
        else:  # detailed
            confidence_pct = int(self.confidence * 100)
            if self.distance_quality == "estimated":
                return (f"{self.class_name} detected on your {self.zone}, "
                        f"approximately {self.distance_m:.1f} meters away, "
                        f"{confidence_pct}% confidence")
            else:
                return (f"{self.class_name} on your {self.zone} at {self.distance_m:.1f} meters, "
                        f"{confidence_pct}% confidence")


@dataclass
class SafetyAnalysisResult:
    """
    Complete safety analysis for a frame.
    """
    frame_id: int
    alerts: List[Alert]           # Filtered and prioritized alerts
    all_detections_count: int     # Before filtering
    filtered_count: int           # After ignore_classes filter
    deduplicated_count: int       # After deduplication
    analysis_time_ms: float
    timestamp: float
    
    @property
    def critical_count(self) -> int:
        return sum(1 for a in self.alerts if a.danger_level == 'critical')
    
    @property
    def warning_count(self) -> int:
        return sum(1 for a in self.alerts if a.danger_level == 'warning')
    
    @property
    def info_count(self) -> int:
        return sum(1 for a in self.alerts if a.danger_level == 'info')
    
    def to_dict(self) -> dict:
        """Convert to dict for telemetry logging."""
        return {
            'frame_id': self.frame_id,
            'alert_count': len(self.alerts),
            'all_detections': self.all_detections_count,
            'filtered': self.filtered_count,
            'deduplicated': self.deduplicated_count,
            'critical_count': self.critical_count,
            'warning_count': self.warning_count,
            'info_count': self.info_count,
            'analysis_time_ms': round(self.analysis_time_ms, 2),
            'alerts': [a.to_dict() for a in self.alerts]
        }


class SafetyManager:
    """
    Analyzes object detections for navigation safety.
    
    Usage:
        safety = SafetyManager(config_path="config/settings.yaml")
        result = safety.analyze(detections, frame_width, frame_id)
        
        for alert in result.alerts:
            print(f"{alert.class_name}: {alert.danger_level} at {alert.distance_m}m")
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize safety manager with configuration."""
        self._load_config(config_path)
        
        # Alert history for deduplication
        self._alert_history: deque = deque(maxlen=50)
        
        # Statistics
        self._total_alerts: int = 0
        self._total_frames: int = 0
        self._deduplicated_alerts: int = 0
    
    def _load_config(self, config_path: str) -> None:
        """Load safety settings from YAML config."""
        # Defaults
        self._ignore_classes: List[str] = []
        self._danger_levels: Dict = {
            'critical': {'distance_m': 1.0, 'priority': 5, 'voice_profile': 'urgent'},
            'warning': {'distance_m': 2.5, 'priority': 3, 'voice_profile': 'alert'},
            'info': {'distance_m': 5.0, 'priority': 1, 'voice_profile': 'calm'}
        }
        self._zones: Dict = {
            'center': [0.3, 0.7],
            'left': [0.0, 0.3],
            'right': [0.7, 1.0]
        }
        self._dedup_window_sec: float = 3.0
        self._priority_boosts: Dict = {}
        self._zone_boost_center: int = 15
        self._center_zone_multiplier: float = 0.8
        
        # Distance estimation defaults
        self._reference_objects: Dict = {
            'person': {'height_m': 1.7, 'ref_bbox_height': 400, 'ref_distance_m': 2.0}
        }
        self._fallback_enabled: bool = True
        self._fallback_height_m: float = 1.0
        self._fallback_ref_bbox: int = 200
        self._fallback_ref_distance: float = 2.0
        self._min_distance: float = 0.5  # Minimum realistic distance
        self._max_distance: float = 20.0  # Maximum tracked distance
        
        # Load from file if exists
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
                # Safety section
                safety_cfg = config.get('safety', {})
                self._ignore_classes = safety_cfg.get('ignore_classes', [])
                self._dedup_window_sec = safety_cfg.get('deduplication_window_sec', 3.0)
                self._priority_boosts = safety_cfg.get('priority_boost', {})
                self._center_zone_multiplier = safety_cfg.get('center_zone_multiplier', 0.8)
                
                # Danger levels with voice_profile
                danger_cfg = safety_cfg.get('danger_levels', {})
                if danger_cfg:
                    for level in ['critical', 'warning', 'info']:
                        if level in danger_cfg:
                            self._danger_levels[level] = {
                                'distance_m': danger_cfg[level].get('distance_m', self._danger_levels[level]['distance_m']),
                                'priority': danger_cfg[level].get('priority', self._danger_levels[level]['priority']),
                                'voice_profile': danger_cfg[level].get('voice_profile', self._danger_levels[level]['voice_profile'])
                            }
                
                # Zones
                zones_cfg = safety_cfg.get('zones', {})
                if zones_cfg:
                    self._zones = zones_cfg
                
                # Distance estimation
                distance_cfg = config.get('distance', {})
                ref_objects = distance_cfg.get('reference_objects', {})
                if ref_objects:
                    self._reference_objects = ref_objects
                
                fallback_cfg = distance_cfg.get('fallback', {})
                if fallback_cfg:
                    self._fallback_enabled = fallback_cfg.get('enabled', True)
                    self._fallback_height_m = fallback_cfg.get('estimated_height_m', 1.0)
                    self._fallback_ref_bbox = fallback_cfg.get('ref_bbox_height', 200)
                    self._fallback_ref_distance = fallback_cfg.get('ref_distance_m', 2.0)
        
        print(f"[SafetyManager] Initialized")
        print(f"[SafetyManager] Ignore classes: {self._ignore_classes}")
        print(f"[SafetyManager] Dedup window: {self._dedup_window_sec}s")
        print(f"[SafetyManager] Center zone multiplier: {self._center_zone_multiplier}")
        print(f"[SafetyManager] Reference objects: {list(self._reference_objects.keys())}")
    
    def _calculate_zone(self, center_x: int, frame_width: int) -> str:
        """
        Determine which zone an object is in based on horizontal position.
        
        Args:
            center_x: X coordinate of object center
            frame_width: Width of frame in pixels
            
        Returns:
            'left', 'center', or 'right'
        """
        relative_x = center_x / frame_width
        
        if relative_x < self._zones['center'][0]:
            return 'left'
        elif relative_x > self._zones['center'][1]:
            return 'right'
        else:
            return 'center'
    
    def _estimate_distance(self, class_name: str, bbox_height: int) -> Tuple[float, str]:
        """
        Estimate distance using bounding box height (FIXED).
        
        Formula: distance = (real_height × ref_distance × ref_bbox_height) / current_bbox_height
        
        FIX: Added comprehensive input validation to prevent division by zero.
        
        Args:
            class_name: Object class for calibration lookup
            bbox_height: Current bounding box height in pixels
            
        Returns:
            (distance_m, quality) where quality is 'calibrated' or 'estimated'
        """
        # FIX: Validate input range
        if bbox_height <= 0 or bbox_height > 10000:
            return (self._max_distance, 'estimated')
        
        # Check for calibrated reference
        if class_name in self._reference_objects:
            ref = self._reference_objects[class_name]
            
            # FIX: Validate reference values before division
            if (ref.get('height_m', 0) <= 0 or 
                ref.get('ref_bbox_height', 0) <= 0 or
                ref.get('ref_distance_m', 0) <= 0):
                print(f"[SafetyManager] WARNING: Invalid reference for {class_name}")
                # Fall through to fallback
            else:
                # Safe calculation
                numerator = (ref['height_m'] * 
                            ref['ref_distance_m'] * 
                            ref['ref_bbox_height'])
                distance = numerator / bbox_height
                
                return (
                    round(max(self._min_distance, min(distance, self._max_distance)), 1),
                    'calibrated'
                )
        
        # Use fallback estimation
        if self._fallback_enabled:
            # FIX: Validate fallback values before division
            if (self._fallback_height_m > 0 and 
                self._fallback_ref_bbox > 0 and
                self._fallback_ref_distance > 0):
                
                numerator = (self._fallback_height_m * 
                            self._fallback_ref_distance * 
                            self._fallback_ref_bbox)
                distance = numerator / bbox_height
                
                return (
                    round(max(self._min_distance, min(distance, self._max_distance)), 1),
                    'estimated'
                )
        
        # Ultimate fallback
        return (self._max_distance, 'estimated')
    
    def _calculate_danger_level(self, distance_m: float, zone: str) -> Tuple[str, int, str]:
        """
        Determine danger level based on distance and zone.
        
        Args:
            distance_m: Estimated distance in meters
            zone: 'left', 'center', or 'right'
            
        Returns:
            (danger_level, base_priority, voice_profile)
        """
        # Center zone is more critical - reduce effective distance threshold
        effective_distance = distance_m
        if zone == 'center':
            effective_distance = distance_m * self._center_zone_multiplier
        
        if effective_distance < self._danger_levels['critical']['distance_m']:
            return (
                'critical',
                self._danger_levels['critical']['priority'],
                self._danger_levels['critical']['voice_profile']
            )
        elif effective_distance < self._danger_levels['warning']['distance_m']:
            return (
                'warning',
                self._danger_levels['warning']['priority'],
                self._danger_levels['warning']['voice_profile']
            )
        elif effective_distance < self._danger_levels['info']['distance_m']:
            return (
                'info',
                self._danger_levels['info']['priority'],
                self._danger_levels['info']['voice_profile']
            )
        else:
            return ('info', 0, 'calm')  # Too far, lowest priority
    
    def _calculate_priority(self, class_name: str, zone: str, base_priority: int) -> int:
        """
        Calculate final priority (additive).
        
        Priority = base_priority + object_boost + zone_boost
        
        Args:
            class_name: Object class for boost lookup
            zone: 'left', 'center', 'right'
            base_priority: From danger level
            
        Returns:
            Final priority (higher = more urgent)
        """
        priority = base_priority
        
        # Add object-specific boost
        object_boost = self._priority_boosts.get(class_name, 0)
        priority += object_boost
        
        # Add zone boost (center is more critical)
        if zone == 'center':
            priority += self._zone_boost_center
        
        return priority
    
    def _is_duplicate(self, alert_key: str, danger_level: str, current_time: float) -> bool:
        """
        Check if this alert was recently announced.
        
        Allows escalation (danger level increase) even within dedup window.
        
        Args:
            alert_key: class_zone combination
            danger_level: Current danger level
            current_time: Current timestamp
            
        Returns:
            True if duplicate (should skip), False if new or escalated
        """
        danger_order = {'info': 1, 'warning': 2, 'critical': 3}
        
        for prev in self._alert_history:
            if prev['key'] == alert_key:
                time_diff = current_time - prev['time']
                
                if time_diff < self._dedup_window_sec:
                    # Within window - check for escalation
                    if danger_order.get(danger_level, 0) > danger_order.get(prev['danger_level'], 0):
                        return False  # Escalation - not duplicate
                    return True  # Same or lower level - duplicate
        
        return False  # Not found in history
    
    def _add_to_history(self, alert_key: str, danger_level: str, current_time: float) -> None:
        """Add/update alert in history for deduplication."""
        # Update existing entry if found (more efficient than rebuilding)
        for alert in self._alert_history:
            if alert['key'] == alert_key:
                alert['danger_level'] = danger_level
                alert['time'] = current_time
                return
        
        # Not found - add new entry
        self._alert_history.append({
            'key': alert_key,
            'danger_level': danger_level,
            'time': current_time
        })
    
    def analyze(self, detections: List, frame_width: int, frame_id: int = 0) -> SafetyAnalysisResult:
        """
        Analyze detections for safety alerts.
        
        Args:
            detections: List of Detection objects from ObjectDetector
            frame_width: Frame width in pixels (for zone calculation)
            frame_id: Frame identifier
            
        Returns:
            SafetyAnalysisResult with filtered, prioritized alerts
        """
        start_time = time.time()
        current_time = time.time()
        
        # Validate frame width
        if frame_width <= 0:
            print(f"[SafetyManager] ERROR: Invalid frame_width: {frame_width}")
            return SafetyAnalysisResult(
                frame_id=frame_id,
                alerts=[],
                all_detections_count=0,
                filtered_count=0,
                deduplicated_count=0,
                analysis_time_ms=0.0,
                timestamp=current_time
            )
        
        all_count = len(detections)
        alerts: List[Alert] = []
        filtered_count = 0
        dedup_count = 0
        
        for det in detections:
            # Get detection attributes (handle both Detection objects and dicts)
            if hasattr(det, 'class_name'):
                class_name = det.class_name
                confidence = det.confidence
                bbox = det.bbox
                bbox_center = det.bbox_center
                bbox_height = det.bbox_height
            else:
                # Dict format - safer extraction
                class_name = det.get('class_name') or det.get('class', 'unknown')
                confidence = det.get('confidence', 0.0)
                bbox = tuple(det.get('bbox', [0, 0, 0, 0]))
                bbox_center = tuple(det.get('bbox_center', [0, 0]))
                bbox_height = det.get('bbox_height', 0)
                
                # Skip if critical fields missing
                if class_name == 'unknown' or bbox_height == 0:
                    continue
            
            # Filter ignore_classes
            if class_name in self._ignore_classes:
                continue
            
            filtered_count += 1
            
            # Calculate zone
            zone = self._calculate_zone(bbox_center[0], frame_width)
            
            # Estimate distance
            distance_m, distance_quality = self._estimate_distance(class_name, bbox_height)
            
            # Calculate danger level and voice profile
            danger_level, base_priority, voice_profile = self._calculate_danger_level(distance_m, zone)
            
            # Skip if too far (no danger)
            if base_priority == 0:
                continue
            
            # Calculate final priority
            priority = self._calculate_priority(class_name, zone, base_priority)
            
            # Create alert
            alert = Alert(
                class_name=class_name,
                confidence=confidence,
                bbox=bbox,
                bbox_center=bbox_center,
                bbox_height=bbox_height,
                zone=zone,
                distance_m=distance_m,
                distance_quality=distance_quality,
                danger_level=danger_level,
                voice_profile=voice_profile,
                priority=priority
            )
            
            # Check deduplication
            if self._is_duplicate(alert.alert_key, danger_level, current_time):
                self._deduplicated_alerts += 1
                dedup_count += 1
                continue
            
            # Add to history and alerts list
            self._add_to_history(alert.alert_key, danger_level, current_time)
            alerts.append(alert)
        
        # Sort by priority (highest first)
        alerts.sort(key=lambda a: a.priority, reverse=True)
        
        # Update statistics
        self._total_frames += 1
        self._total_alerts += len(alerts)
        
        analysis_time_ms = (time.time() - start_time) * 1000
        
        return SafetyAnalysisResult(
            frame_id=frame_id,
            alerts=alerts,
            all_detections_count=all_count,
            filtered_count=filtered_count,
            deduplicated_count=dedup_count,
            analysis_time_ms=analysis_time_ms,
            timestamp=current_time
        )
    
    def get_stats(self) -> dict:
        """Get safety manager statistics."""
        return {
            'total_frames': self._total_frames,
            'total_alerts': self._total_alerts,
            'deduplicated_alerts': self._deduplicated_alerts,
            'avg_alerts_per_frame': round(self._total_alerts / max(1, self._total_frames), 2)
        }
    
    def clear_history(self) -> None:
        """Clear alert history (useful for testing)."""
        self._alert_history.clear()


# ============================================
# Testing / Demo
# ============================================
if __name__ == "__main__":
    """Test safety manager module standalone."""
    from object_detector import ObjectDetector
    from camera_capture import CameraCapture
    import cv2
    
    print("=" * 50)
    print("Safety Manager Module Test")
    print("=" * 50)
    
    # Initialize modules
    detector = ObjectDetector(config_path="config/settings.yaml")
    safety = SafetyManager(config_path="config/settings.yaml")
    
    DANGER_COLORS = {
        'critical': (0, 0, 255),  # Red
        'warning': (0, 165, 255), # Orange
        'info': (0, 255, 0)       # Green
    }
    
    with CameraCapture(config_path="config/settings.yaml") as camera:
        print("\nPress 'q' to quit\n")
        
        frame_count = 0
        
        while True:
            frame_packet = camera.get_frame(timeout=1.0)
            
            if frame_packet is None:
                continue
            
            frame_count += 1
            
            # Skip frames
            if frame_count % camera.frame_skip != 0:
                cv2.imshow("Safety Manager Test", frame_packet.data)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            # Detect objects
            detection_result = detector.detect(frame_packet.data, frame_id=frame_packet.frame_id)
            
            # Analyze safety
            safety_result = safety.analyze(
                detection_result.detections,
                frame_width=frame_packet.width,
                frame_id=frame_packet.frame_id
            )
            
            # Draw visualization
            display_frame = frame_packet.data.copy()
            
            # Draw zone lines
            left_line = int(frame_packet.width * 0.3)
            right_line = int(frame_packet.width * 0.7)
            cv2.line(display_frame, (left_line, 0), (left_line, frame_packet.height), (100, 100, 100), 1)
            cv2.line(display_frame, (right_line, 0), (right_line, frame_packet.height), (100, 100, 100), 1)
            
            # Draw alerts
            for alert in safety_result.alerts:
                x1, y1, x2, y2 = alert.bbox
                color = DANGER_COLORS.get(alert.danger_level, (255, 255, 255))
                
                # Draw box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with voice profile
                label = f"{alert.class_name} {alert.distance_m}m [{alert.zone}] {alert.voice_profile}"
                cv2.putText(display_frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw stats
            stats_text = (f"Alerts: {len(safety_result.alerts)} "
                         f"(C:{safety_result.critical_count} W:{safety_result.warning_count} I:{safety_result.info_count})")
            cv2.putText(display_frame, stats_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow("Safety Manager Test", display_frame)
            
            # Print alerts
            if safety_result.alerts:
                for alert in safety_result.alerts[:3]:
                    print(f"  [{alert.danger_level.upper()}] [{alert.voice_profile}] {alert.get_announcement()}")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Print final stats
        stats = safety.get_stats()
        print(f"\n[SafetyManager] Final Statistics:")
        print(f"  Total frames: {stats['total_frames']}")
        print(f"  Total alerts: {stats['total_alerts']}")
        print(f"  Deduplicated: {stats['deduplicated_alerts']}")
        print(f"  Avg alerts/frame: {stats['avg_alerts_per_frame']}")
    
    cv2.destroyAllWindows()
    print("\nTest complete!")
