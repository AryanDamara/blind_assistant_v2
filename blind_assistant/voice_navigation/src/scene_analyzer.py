"""
Scene Analyzer Module
---------------------
Centroid-based object tracking and scene understanding.

Features:
- Object tracking with persistent track IDs
- 5-second position history per tracked object
- Movement detection (stationary vs moving)
- Spatial grouping of nearby objects
- Scene summary generation

Thread: Runs synchronously in main loop (after SafetyManager)
"""

import time
import math
import yaml
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import deque

from object_detector import Detection


@dataclass
class TrackedObject:
    """
    A tracked object with position history.
    """
    track_id: int                     # Unique tracking ID
    class_name: str                   # Object class (e.g., 'person')
    confidence: float                 # Latest detection confidence
    bbox: Tuple[int, int, int, int]   # Latest bounding box (x1, y1, x2, y2)
    centroid: Tuple[int, int]         # Current centroid (x, y)
    
    # Position history: list of (x, y, timestamp)
    positions: deque = field(default_factory=lambda: deque(maxlen=50))
    
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    
    # Movement analysis
    is_moving: bool = False
    velocity: Tuple[float, float] = (0.0, 0.0)  # Pixels per second (vx, vy)
    total_displacement: float = 0.0              # Total distance moved
    
    # Zone and distance info (enriched from SafetyManager)
    zone: str = "center"
    distance_m: float = 0.0  # Estimated distance from SafetyManager alerts
    
    def __post_init__(self):
        """Initialize position history with current centroid."""
        if len(self.positions) == 0 and self.centroid:
            self.positions.append((self.centroid[0], self.centroid[1], time.time()))
    
    def update(self, detection: Detection, timestamp: float) -> None:
        """Update tracked object with new detection."""
        self.class_name = detection.class_name
        self.confidence = detection.confidence
        self.bbox = detection.bbox
        
        old_centroid = self.centroid
        self.centroid = detection.bbox_center
        self.last_seen = timestamp
        
        # Add to position history
        self.positions.append((self.centroid[0], self.centroid[1], timestamp))
        
        # Calculate displacement from previous position
        if old_centroid:
            dx = self.centroid[0] - old_centroid[0]
            dy = self.centroid[1] - old_centroid[1]
            displacement = math.sqrt(dx*dx + dy*dy)
            self.total_displacement += displacement
    
    def calculate_velocity(self) -> Tuple[float, float]:
        """Calculate velocity from position history."""
        if len(self.positions) < 2:
            return (0.0, 0.0)
        
        # Use first and last positions in history
        first = self.positions[0]
        last = self.positions[-1]
        
        time_diff = last[2] - first[2]
        if time_diff < 0.1:  # Less than 100ms
            return (0.0, 0.0)
        
        vx = (last[0] - first[0]) / time_diff
        vy = (last[1] - first[1]) / time_diff
        
        self.velocity = (vx, vy)
        return self.velocity
    
    def get_speed(self) -> float:
        """Get speed in pixels per second."""
        vx, vy = self.velocity
        return math.sqrt(vx*vx + vy*vy)
    
    def get_age(self) -> float:
        """Get time since first detection."""
        return time.time() - self.first_seen
    
    def get_time_since_seen(self) -> float:
        """Get time since last detection."""
        return time.time() - self.last_seen
    
    def to_dict(self) -> dict:
        """Convert to dict for telemetry/logging."""
        return {
            'track_id': self.track_id,
            'class': self.class_name,
            'confidence': round(self.confidence, 3),
            'centroid': list(self.centroid),
            'is_moving': self.is_moving,
            'velocity': [round(v, 1) for v in self.velocity],
            'speed': round(self.get_speed(), 1),
            'age_sec': round(self.get_age(), 1),
            'zone': self.zone
        }


@dataclass
class ObjectGroup:
    """
    A spatial group of nearby objects.
    """
    group_id: int
    objects: List[TrackedObject]
    center: Tuple[int, int]            # Group centroid
    zone: str                          # Dominant zone
    
    @property
    def count(self) -> int:
        return len(self.objects)
    
    @property
    def class_names(self) -> List[str]:
        return [obj.class_name for obj in self.objects]
    
    def get_summary(self) -> str:
        """Generate human-readable group summary."""
        class_counts = {}
        for obj in self.objects:
            class_counts[obj.class_name] = class_counts.get(obj.class_name, 0) + 1
        
        parts = [f"{count} {name}{'s' if count > 1 else ''}" 
                 for name, count in class_counts.items()]
        
        return f"{', '.join(parts)} on your {self.zone}"


@dataclass
class SceneAnalysis:
    """
    Complete scene analysis result.
    """
    frame_id: int
    timestamp: float
    
    # Tracked objects
    tracked_objects: List[TrackedObject]
    active_tracks: int
    lost_tracks: int
    new_tracks: int
    
    # Grouping
    object_groups: List[ObjectGroup]
    
    # Movement
    moving_objects: List[TrackedObject]
    stationary_objects: List[TrackedObject]
    
    # Scene summary
    scene_summary: str
    analysis_time_ms: float
    
    @property
    def total_objects(self) -> int:
        return len(self.tracked_objects)
    
    def to_dict(self) -> dict:
        """Convert to dict for telemetry/logging."""
        return {
            'frame_id': self.frame_id,
            'total_objects': self.total_objects,
            'active_tracks': self.active_tracks,
            'moving_count': len(self.moving_objects),
            'stationary_count': len(self.stationary_objects),
            'group_count': len(self.object_groups),
            'scene_summary': self.scene_summary,
            'analysis_time_ms': round(self.analysis_time_ms, 2)
        }


class SceneAnalyzer:
    """
    Centroid-based object tracking and scene understanding.
    
    Usage:
        analyzer = SceneAnalyzer(config_path="config/settings.yaml")
        
        # In main loop:
        scene = analyzer.analyze(detections, frame_id)
        
        for obj in scene.tracked_objects:
            print(f"{obj.class_name} (ID {obj.track_id}): moving={obj.is_moving}")
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize scene analyzer with configuration."""
        self._load_config(config_path)
        
        # Active tracks: track_id -> TrackedObject
        self._tracks: Dict[int, TrackedObject] = {}
        self._next_track_id: int = 1
        
        # Statistics
        self._total_frames: int = 0
        self._total_new_tracks: int = 0
        self._total_lost_tracks: int = 0
    
    def _load_config(self, config_path: str) -> None:
        """Load scene analysis settings from YAML config."""
        # Defaults
        self._tracking_enabled: bool = True
        self._history_seconds: float = 5.0
        self._history_frames: int = 50
        self._min_movement_threshold_px: int = 10
        self._grouping_enabled: bool = True
        self._grouping_distance_px: int = 100
        self._track_timeout_sec: float = 1.0  # Lose track after 1 second unseen
        self._max_match_distance_px: int = 150  # Max distance for centroid matching
        
        # Zone boundaries (from safety config)
        self._zones = {
            'center': [0.3, 0.7],
            'left': [0.0, 0.3],
            'right': [0.7, 1.0]
        }
        self._frame_width: int = 640  # Will be updated dynamically
        
        # Load from file if exists
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
                scene_cfg = config.get('scene', {})
                self._tracking_enabled = scene_cfg.get('tracking_enabled', True)
                self._history_seconds = scene_cfg.get('history_seconds', 5.0)
                self._history_frames = scene_cfg.get('history_frames', 50)
                self._min_movement_threshold_px = scene_cfg.get('min_movement_threshold_px', 10)
                self._grouping_enabled = scene_cfg.get('grouping_enabled', True)
                self._grouping_distance_px = scene_cfg.get('grouping_distance_px', 100)
                
                # Get zone config from safety section
                safety_cfg = config.get('safety', {})
                zones_cfg = safety_cfg.get('zones', {})
                if zones_cfg:
                    self._zones = zones_cfg
        
        print(f"[SceneAnalyzer] Initialized")
        print(f"[SceneAnalyzer] Tracking enabled: {self._tracking_enabled}")
        print(f"[SceneAnalyzer] History: {self._history_seconds}s ({self._history_frames} frames)")
        print(f"[SceneAnalyzer] Movement threshold: {self._min_movement_threshold_px}px")
        print(f"[SceneAnalyzer] Grouping enabled: {self._grouping_enabled}")
    
    def _calculate_zone(self, center_x: int) -> str:
        """Determine zone from x coordinate."""
        if self._frame_width <= 0:
            return 'center'
        
        relative_x = center_x / self._frame_width
        
        if relative_x < self._zones['center'][0]:
            return 'left'
        elif relative_x > self._zones['center'][1]:
            return 'right'
        else:
            return 'center'
    
    def _centroid_distance(self, c1: Tuple[int, int], c2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two centroids."""
        dx = c1[0] - c2[0]
        dy = c1[1] - c2[1]
        return math.sqrt(dx*dx + dy*dy)
    
    def _match_detections_to_tracks(self, detections: List[Detection]) -> Dict[int, Detection]:
        """
        Match new detections to existing tracks using centroid distance.
        Returns: {track_id: detection} for matched detections
        """
        if not self._tracks or not detections:
            return {}
        
        matches: Dict[int, Detection] = {}
        used_detections: set = set()
        
        # For each track, find closest matching detection of same class
        for track_id, track in self._tracks.items():
            best_match: Optional[Detection] = None
            best_distance: float = float('inf')
            best_idx: int = -1
            
            for idx, det in enumerate(detections):
                if idx in used_detections:
                    continue
                
                # Must be same class
                if det.class_name != track.class_name:
                    continue
                
                distance = self._centroid_distance(track.centroid, det.bbox_center)
                
                if distance < best_distance and distance < self._max_match_distance_px:
                    best_match = det
                    best_distance = distance
                    best_idx = idx
            
            if best_match is not None:
                matches[track_id] = best_match
                used_detections.add(best_idx)
        
        return matches
    
    def _create_track(self, detection: Detection, timestamp: float) -> TrackedObject:
        """Create a new track from a detection."""
        track = TrackedObject(
            track_id=self._next_track_id,
            class_name=detection.class_name,
            confidence=detection.confidence,
            bbox=detection.bbox,
            centroid=detection.bbox_center,
            zone=self._calculate_zone(detection.bbox_center[0])
        )
        
        self._next_track_id += 1
        self._total_new_tracks += 1
        
        return track
    
    def _update_movement(self, track: TrackedObject) -> None:
        """Update movement status based on position history."""
        track.calculate_velocity()
        speed = track.get_speed()
        
        # Object is moving if speed exceeds threshold
        track.is_moving = speed > self._min_movement_threshold_px
    
    def _enrich_tracks_with_alerts(self, alerts: List) -> None:
        """Enrich tracked objects with zone/distance data from SafetyManager alerts."""
        for track in self._tracks.values():
            # Find matching alert for this track
            for alert in alerts:
                # Get alert properties (handle both dict and object)
                if isinstance(alert, dict):
                    alert_class = alert.get('class_name', alert.get('class', ''))
                    alert_zone = alert.get('zone', '')
                    alert_distance = alert.get('distance_m', 0)
                    alert_bbox = alert.get('bbox', (0, 0, 0, 0))
                    # Calculate alert center
                    if alert_bbox and len(alert_bbox) == 4:
                        alert_center = ((alert_bbox[0] + alert_bbox[2]) // 2,
                                       (alert_bbox[1] + alert_bbox[3]) // 2)
                    else:
                        alert_center = (0, 0)
                else:
                    alert_class = getattr(alert, 'class_name', '')
                    alert_zone = getattr(alert, 'zone', '')
                    alert_distance = getattr(alert, 'distance_m', 0)
                    alert_center = getattr(alert, 'bbox_center', (0, 0))
                
                # Match by class and proximity
                if (alert_class == track.class_name and 
                    self._centroid_distance(track.centroid, alert_center) < 50):
                    track.zone = alert_zone
                    track.distance_m = alert_distance
                    break
    
    def _group_objects(self, tracks: List[TrackedObject]) -> List[ObjectGroup]:
        """Group nearby objects using simple clustering."""
        if not tracks or not self._grouping_enabled:
            return []
        
        groups: List[ObjectGroup] = []
        used: set = set()
        group_id = 1
        
        for i, track in enumerate(tracks):
            if i in used:
                continue
            
            # Start new group with this track
            group_members = [track]
            used.add(i)
            
            # Find nearby tracks
            for j, other in enumerate(tracks):
                if j in used:
                    continue
                
                distance = self._centroid_distance(track.centroid, other.centroid)
                if distance < self._grouping_distance_px:
                    group_members.append(other)
                    used.add(j)
            
            # Only create groups with 2+ members
            if len(group_members) >= 2:
                # Calculate group center
                cx = sum(m.centroid[0] for m in group_members) // len(group_members)
                cy = sum(m.centroid[1] for m in group_members) // len(group_members)
                
                # Determine dominant zone
                zones = [m.zone for m in group_members]
                dominant_zone = max(set(zones), key=zones.count)
                
                group = ObjectGroup(
                    group_id=group_id,
                    objects=group_members,
                    center=(cx, cy),
                    zone=dominant_zone
                )
                groups.append(group)
                group_id += 1
        
        return groups
    
    def _generate_scene_summary(self, tracks: List[TrackedObject], 
                                 groups: List[ObjectGroup]) -> str:
        """Generate human-readable scene summary with distance info."""
        if not tracks:
            return "Path ahead is clear, no objects detected"
        
        parts = []
        
        # Prioritize closest objects in center zone with distance
        center_tracks = [t for t in tracks if t.zone == 'center']
        if center_tracks:
            # Find closest center object
            closest = min(center_tracks, key=lambda t: t.distance_m if t.distance_m > 0 else 999)
            if closest.distance_m > 0:
                parts.append(f"{closest.class_name} directly ahead at {closest.distance_m:.1f} meters")
            else:
                parts.append(f"{closest.class_name} directly ahead")
        else:
            parts.append("Path ahead is clear")
        
        # Count objects by zone
        zone_counts = {'left': 0, 'center': 0, 'right': 0}
        class_counts = {}
        moving_count = 0
        
        for track in tracks:
            zone_counts[track.zone] = zone_counts.get(track.zone, 0) + 1
            class_counts[track.class_name] = class_counts.get(track.class_name, 0) + 1
            if track.is_moving:
                moving_count += 1
        
        # Object counts by class (excluding first mentioned)
        class_parts = [f"{count} {name}{'s' if count > 1 else ''}" 
                       for name, count in sorted(class_counts.items(), key=lambda x: -x[1])]
        if len(class_parts) > 1:
            parts.append(f"Total: {', '.join(class_parts[:3])}")
        
        # Left/right zone info
        side_parts = []
        if zone_counts['left'] > 0:
            parts.append(f"{zone_counts['left']} on left")
        if zone_counts['right'] > 0:
            parts.append(f"{zone_counts['right']} on right")
        
        # Movement info
        if moving_count > 0:
            parts.append(f"{moving_count} moving")
        
        return ". ".join(parts) if parts else "Scene analyzed"
    
    def analyze(self, detections: List[Detection], frame_id: int = 0,
                frame_width: int = 640, alerts: List = None) -> SceneAnalysis:
        """
        Analyze scene with object tracking and grouping.
        
        Args:
            detections: List of Detection objects from ObjectDetector
            frame_id: Frame identifier
            frame_width: Frame width for zone calculation
            alerts: Optional list of Alert objects from SafetyManager for zone/distance enrichment
            
        Returns:
            SceneAnalysis with tracked objects, groups, and summary
        """
        start_time = time.time()
        current_time = time.time()
        
        self._frame_width = frame_width
        self._total_frames += 1
        
        new_tracks_count = 0
        lost_tracks_count = 0
        
        if self._tracking_enabled and detections:
            # Match detections to existing tracks
            matches = self._match_detections_to_tracks(detections)
            
            # Update matched tracks
            matched_det_indices = set()
            for track_id, det in matches.items():
                self._tracks[track_id].update(det, current_time)
                self._tracks[track_id].zone = self._calculate_zone(det.bbox_center[0])
                
                # Find detection index for tracking
                for idx, d in enumerate(detections):
                    if d.bbox_center == det.bbox_center and d.class_name == det.class_name:
                        matched_det_indices.add(idx)
                        break
            
            # Create new tracks for unmatched detections
            for idx, det in enumerate(detections):
                if idx not in matched_det_indices:
                    track = self._create_track(det, current_time)
                    self._tracks[track.track_id] = track
                    new_tracks_count += 1
            
            # Remove stale tracks
            stale_track_ids = []
            for track_id, track in self._tracks.items():
                if track.get_time_since_seen() > self._track_timeout_sec:
                    stale_track_ids.append(track_id)
            
            for track_id in stale_track_ids:
                del self._tracks[track_id]
                lost_tracks_count += 1
                self._total_lost_tracks += 1
            
            # Update movement for all active tracks
            for track in self._tracks.values():
                self._update_movement(track)
            
            # Enrich tracks with alert data (zone/distance from SafetyManager)
            if alerts:
                self._enrich_tracks_with_alerts(alerts)
        
        # Get active track list
        active_tracks = list(self._tracks.values())
        
        # Categorize by movement
        moving = [t for t in active_tracks if t.is_moving]
        stationary = [t for t in active_tracks if not t.is_moving]
        
        # Group objects
        groups = self._group_objects(active_tracks)
        
        # Generate scene summary
        summary = self._generate_scene_summary(active_tracks, groups)
        
        analysis_time_ms = (time.time() - start_time) * 1000
        
        return SceneAnalysis(
            frame_id=frame_id,
            timestamp=current_time,
            tracked_objects=active_tracks,
            active_tracks=len(active_tracks),
            lost_tracks=lost_tracks_count,
            new_tracks=new_tracks_count,
            object_groups=groups,
            moving_objects=moving,
            stationary_objects=stationary,
            scene_summary=summary,
            analysis_time_ms=analysis_time_ms
        )
    
    def get_object_by_class(self, class_name: str) -> List[TrackedObject]:
        """Get all tracked objects of a specific class."""
        return [t for t in self._tracks.values() if t.class_name == class_name]
    
    def get_objects_in_zone(self, zone: str) -> List[TrackedObject]:
        """Get all tracked objects in a specific zone."""
        return [t for t in self._tracks.values() if t.zone == zone]
    
    def get_moving_objects(self) -> List[TrackedObject]:
        """Get all currently moving objects."""
        return [t for t in self._tracks.values() if t.is_moving]
    
    def get_scene_description(self) -> str:
        """Get a detailed scene description for LLM context."""
        if not self._tracks:
            return "The scene is clear with no objects detected."
        
        lines = []
        
        # Overview
        moving_count = len(self.get_moving_objects())
        lines.append(f"There are {len(self._tracks)} objects being tracked.")
        
        if moving_count > 0:
            lines.append(f"{moving_count} of them are moving.")
        
        # Per-zone breakdown
        for zone in ['center', 'left', 'right']:
            zone_objects = self.get_objects_in_zone(zone)
            if zone_objects:
                class_list = [f"{o.class_name}" for o in zone_objects[:3]]
                lines.append(f"On the {zone}: {', '.join(class_list)}")
        
        return " ".join(lines)
    
    def get_stats(self) -> dict:
        """Get scene analyzer statistics."""
        return {
            'total_frames': self._total_frames,
            'active_tracks': len(self._tracks),
            'total_new_tracks': self._total_new_tracks,
            'total_lost_tracks': self._total_lost_tracks,
            'next_track_id': self._next_track_id
        }
    
    def clear_tracks(self) -> None:
        """Clear all tracks (useful for testing)."""
        self._tracks.clear()
        self._next_track_id = 1


# ============================================
# Testing / Demo
# ============================================
if __name__ == "__main__":
    """Test scene analyzer module standalone."""
    from object_detector import ObjectDetector
    from camera_capture import CameraCapture
    import cv2
    
    print("=" * 50)
    print("Scene Analyzer Module Test")
    print("=" * 50)
    
    # Initialize modules
    detector = ObjectDetector(config_path="config/settings.yaml")
    analyzer = SceneAnalyzer(config_path="config/settings.yaml")
    
    # Colors for visualization
    TRACK_COLORS = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128),
        (128, 128, 0), (128, 0, 128), (0, 128, 128)
    ]
    
    with CameraCapture(config_path="config/settings.yaml") as camera:
        print("\nPress 'q' to quit\n")
        
        frame_count = 0
        
        while True:
            frame_packet = camera.get_frame(timeout=1.0)
            
            if frame_packet is None:
                continue
            
            frame_count += 1
            
            # Skip frames for performance
            if frame_count % camera.frame_skip != 0:
                cv2.imshow("Scene Analyzer Test", frame_packet.data)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            # Run detection
            detection_result = detector.detect(frame_packet.data, frame_id=frame_packet.frame_id)
            
            # Run scene analysis
            scene = analyzer.analyze(
                detection_result.detections,
                frame_id=frame_packet.frame_id,
                frame_width=frame_packet.width
            )
            
            # Visualization
            display_frame = frame_packet.data.copy()
            
            # Draw zone lines
            left_line = int(frame_packet.width * 0.3)
            right_line = int(frame_packet.width * 0.7)
            cv2.line(display_frame, (left_line, 0), (left_line, frame_packet.height), 
                    (100, 100, 100), 1)
            cv2.line(display_frame, (right_line, 0), (right_line, frame_packet.height), 
                    (100, 100, 100), 1)
            
            # Draw tracked objects
            for obj in scene.tracked_objects:
                color = TRACK_COLORS[obj.track_id % len(TRACK_COLORS)]
                x1, y1, x2, y2 = obj.bbox
                
                # Draw bounding box
                thickness = 3 if obj.is_moving else 2
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
                
                # Draw track ID and movement status
                status = "M" if obj.is_moving else "S"
                label = f"#{obj.track_id} {obj.class_name} [{status}]"
                cv2.putText(display_frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw centroid
                cv2.circle(display_frame, obj.centroid, 5, color, -1)
                
                # Draw velocity vector if moving
                if obj.is_moving:
                    vx, vy = obj.velocity
                    end_x = int(obj.centroid[0] + vx * 0.2)
                    end_y = int(obj.centroid[1] + vy * 0.2)
                    cv2.arrowedLine(display_frame, obj.centroid, (end_x, end_y), 
                                   (0, 255, 255), 2)
            
            # Draw groups (as connecting lines)
            for group in scene.object_groups:
                if len(group.objects) >= 2:
                    for i in range(len(group.objects) - 1):
                        c1 = group.objects[i].centroid
                        c2 = group.objects[i + 1].centroid
                        cv2.line(display_frame, c1, c2, (200, 200, 200), 1)
            
            # Draw stats overlay
            stats_text = (f"Tracks: {scene.active_tracks} | "
                         f"Moving: {len(scene.moving_objects)} | "
                         f"Groups: {len(scene.object_groups)}")
            cv2.putText(display_frame, stats_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw scene summary
            cv2.putText(display_frame, scene.scene_summary[:60], (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow("Scene Analyzer Test", display_frame)
            
            # Print scene info periodically
            if frame_count % 30 == 0:
                print(f"\n[Frame {frame_packet.frame_id}]")
                print(f"  {scene.scene_summary}")
                print(f"  Active tracks: {scene.active_tracks}, New: {scene.new_tracks}, Lost: {scene.lost_tracks}")
                
                for obj in scene.tracked_objects[:3]:
                    print(f"    #{obj.track_id} {obj.class_name}: zone={obj.zone}, "
                          f"moving={obj.is_moving}, speed={obj.get_speed():.1f}px/s")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Print final stats
        stats = analyzer.get_stats()
        print(f"\n[SceneAnalyzer] Final Statistics:")
        print(f"  Total frames: {stats['total_frames']}")
        print(f"  Active tracks: {stats['active_tracks']}")
        print(f"  Total new tracks: {stats['total_new_tracks']}")
        print(f"  Total lost tracks: {stats['total_lost_tracks']}")
    
    cv2.destroyAllWindows()
    print("\nTest complete!")
