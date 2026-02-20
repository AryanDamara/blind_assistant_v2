#!/usr/bin/env python3
"""
Voice Navigation System - Main Orchestrator (POLISHED)
------------------------------------------------------
Real-time navigation assistance for visually impaired users.

Fixes Applied:
- Memory leak (frame cleanup + gc)
- Frame validation
- Adaptive frame skipping

Polish Applied:
- Named constants, deque for latency, cached FPS
- Zone overlay caching, scene cleanup, display frame cleanup
- cv2 error handling, module init validation
- Frame drop stats, auto crash recovery

Pipeline: Camera → YOLO Detection → Safety Analysis → Scene Analysis → Audio Feedback
         Voice Input → AI Assistant → Audio Response

Usage:
    python src/main.py
    
Press 'q' to quit (when video window is shown)
Press Ctrl+C to quit (in any mode)
Press SPACE to speak a query (when voice input enabled)
"""

import os
import sys
import time
import signal
import math
import traceback
import yaml
import cv2
import gc
import numpy as np
from pathlib import Path
from typing import Optional
from collections import deque

# Resolve project root relative to this script (src/ -> voice_navigation/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from camera_capture import CameraCapture
from object_detector import ObjectDetector
from safety_manager import SafetyManager
from audio_feedback import AudioFeedback
from scene_analyzer import SceneAnalyzer
from ai_assistant import AIAssistant
from conversation_handler import ConversationHandler
from telemetry import TelemetryLogger, LatencyMetrics

# Navigation detectors
from stair_detector import StairDetector
from traffic_light_detector import TrafficLightDetector
from crosswalk_detector import CrosswalkDetector
from curb_detector import CurbDetector
from door_detector import DoorDetector
from retail_detector import RetailDetector
from tactile_paving_detector import TactilePavingDetector


class NavigationSystem:
    """
    Main orchestrator for the voice navigation system.
    
    Coordinates all modules:
    - CameraCapture: Video input
    - ObjectDetector: YOLO detection
    - SafetyManager: Distance/zone/danger analysis
    - SceneAnalyzer: Object tracking and scene understanding (Phase 2)
    - AIAssistant: LLM-powered responses (Phase 2)
    - ConversationHandler: Voice input processing (Phase 2)
    - AudioFeedback: Text-to-speech output
    """
    
    # Named constants (Issue 1.9)
    LATENCY_PRINT_INTERVAL = 30       # Print latency every N frames
    GC_INTERVAL_FRAMES = 1000         # Garbage collect every N frames
    LATENCY_DECREASE_THRESHOLD = 0.7  # Decrease skip if latency < 70% target
    MAX_FRAME_SKIP = 10               # Maximum adaptive frame skip
    MIN_FRAME_SKIP = 1                # Minimum frame skip
    LATENCY_WINDOW_SIZE = 10          # Rolling window for latency history
    FPS_CACHE_INTERVAL = 30           # Cache FPS calculation every N frames
    
    def __init__(self, config_path: str = None):
        """Initialize the navigation system."""
        # Use Path-based resolution so it works from any working directory
        if config_path is None:
            self.config_path = str(BASE_DIR / "config" / "settings.yaml")
        else:
            self.config_path = config_path
        self._load_config()
        
        # Modules (initialized in start())
        self.camera: Optional[CameraCapture] = None
        self.detector: Optional[ObjectDetector] = None
        self.safety: Optional[SafetyManager] = None
        self.audio: Optional[AudioFeedback] = None
        
        # Phase 2 modules
        self.scene_analyzer: Optional[SceneAnalyzer] = None
        self.ai_assistant: Optional[AIAssistant] = None
        self.conversation: Optional[ConversationHandler] = None
        
        # Phase 3 modules
        self.telemetry: Optional[TelemetryLogger] = None
        
        # Navigation detectors
        self.stair_detector: Optional[StairDetector] = None
        self.traffic_light_detector: Optional[TrafficLightDetector] = None
        self.crosswalk_detector: Optional[CrosswalkDetector] = None
        self.curb_detector: Optional[CurbDetector] = None
        self.door_detector: Optional[DoorDetector] = None
        self.retail_detector: Optional[RetailDetector] = None
        self.tactile_paving_detector: Optional[TactilePavingDetector] = None
        
        # Latest scene analysis (for voice queries)
        self._latest_scene = None
        
        # State
        self._running = False
        self._frame_count = 0
        self._start_time = 0.0
        self._error_count = 0
        
        # Statistics
        self._total_frames = 0
        self._total_detections = 0
        self._total_alerts = 0
        self._frames_skipped = 0  # Issue 1.12: Track frame drops
        
        # Adaptive frame skipping
        self._adaptive_skip = True
        self._current_skip = 3
        self._target_latency_ms = 800
        self._latency_history = deque(maxlen=self.LATENCY_WINDOW_SIZE)  # Issue 1.5: deque
        
        # Garbage collection tracking
        self._gc_interval = self.GC_INTERVAL_FRAMES
        self._last_gc_frame = 0
        
        # Issue 1.6: Cached FPS calculation
        self._cached_elapsed = 0.0
        self._cached_fps = 0.0
        
        # Issue 1.14: Cached zone overlay
        self._zone_overlay = None
        self._zone_overlay_size = (0, 0)
        
        # Register signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _load_config(self) -> None:
        """Load configuration from YAML."""
        self.config = {}
        
        if not os.path.exists(self.config_path):
            print(f"[Main] WARNING: config not found: {self.config_path} — using defaults")
            return
        
        with open(self.config_path, 'r') as f:
            # safe_load returns None for empty files -> coalesce to {}
            self.config = yaml.safe_load(f) or {}
        
        # Debug settings
        debug_cfg = self.config.get('debug', {})
        self._show_video = debug_cfg.get('show_video', True)
        self._show_bboxes = debug_cfg.get('show_bboxes', True)
        self._show_zones = debug_cfg.get('show_zones', True)
        self._show_fps = debug_cfg.get('show_fps', True)
        self._print_latency = debug_cfg.get('print_latency', True)
        self._print_detections = debug_cfg.get('print_detections', True)
        self._print_alerts = debug_cfg.get('print_alerts', False)
        
        # System settings
        system_cfg = self.config.get('system', {})
        self._quit_key = ord(system_cfg.get('quit_key', 'q'))
        self._max_errors = system_cfg.get('max_consecutive_errors', 10)
        self._shutdown_timeout = system_cfg.get('shutdown_timeout_sec', 5.0)
        
        # Performance settings
        perf_cfg = self.config.get('performance', {})
        self._adaptive_skip = perf_cfg.get('adaptive_skip', True)
        self._target_latency_ms = perf_cfg.get('target_latency_ms', 800)
    
    def _signal_handler(self, sig, frame) -> None:
        """Handle Ctrl+C and other signals gracefully."""
        print("\n[Main] Shutdown signal received...")
        self._running = False
    
    # FIX: Added frame validation method
    def _validate_frame(self, frame) -> bool:
        """
        Validate frame integrity.
        
        Returns:
            bool: True if frame is valid
        """
        if frame is None:
            return False
        
        if frame.size == 0:
            return False
        
        # Check dimensions
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            return False
        
        # Check minimum size
        if frame.shape[0] < 100 or frame.shape[1] < 100:
            return False
        
        # Check if frame is all zeros (camera failure)
        if frame.max() == 0:
            return False
        
        return True
    
    def _adjust_frame_skip(self, latency_ms: float) -> None:
        """
        Adjust frame skip based on recent latency.
        Uses deque with maxlen for efficient rolling window.
        """
        if not self._adaptive_skip:
            return
        
        # deque auto-truncates (Issue 1.5)
        self._latency_history.append(latency_ms)
        
        if len(self._latency_history) == 0:
            return
        
        avg_latency = sum(self._latency_history) / len(self._latency_history)
        
        if avg_latency > self._target_latency_ms:
            self._current_skip = min(self._current_skip + 1, self.MAX_FRAME_SKIP)
            if self._print_latency:
                print(f"[Main] ⚠️ Latency high ({avg_latency:.0f}ms), skip → {self._current_skip}")
        
        elif avg_latency < self._target_latency_ms * self.LATENCY_DECREASE_THRESHOLD:
            self._current_skip = max(self._current_skip - 1, self.MIN_FRAME_SKIP)
            if self._print_latency:
                print(f"[Main] ✓ Latency good ({avg_latency:.0f}ms), skip → {self._current_skip}")
    
    def start(self) -> bool:
        """
        Initialize and start all modules.
        
        Returns:
            bool: True if started successfully
        """
        print("=" * 60)
        print("Voice Navigation System - Starting")
        print("=" * 60)
        
        # Issue 1.7: Validate config structure before init
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config_check = yaml.safe_load(f) or {}
                required_sections = ['camera', 'yolo', 'safety', 'audio']
                missing = [s for s in required_sections if s not in config_check]
                if missing:
                    print(f"[Main] WARNING: Config missing sections: {missing} — using defaults")
            except yaml.YAMLError as e:
                print(f"[Main] WARNING: Config parse error: {e} — using defaults")
        
        try:
            # Initialize modules
            print("\n[Main] Initializing Phase 1 modules...")
            
            self.camera = CameraCapture(config_path=self.config_path)
            self.detector = ObjectDetector(config_path=self.config_path)
            self.safety = SafetyManager(config_path=self.config_path)
            self.audio = AudioFeedback(config_path=self.config_path)
            
            # Initialize Phase 2 modules
            print("\n[Main] Initializing Phase 2 modules...")
            
            self.scene_analyzer = SceneAnalyzer(config_path=self.config_path)
            self.ai_assistant = AIAssistant(config_path=self.config_path)
            self.conversation = ConversationHandler(
                config_path=self.config_path,
                ai_assistant=self.ai_assistant,
                audio_feedback=self.audio,
                scene_provider=self._get_scene_description
            )
            
            # Start modules
            print("\n[Main] Starting modules...")
            
            if not self.camera.start():
                print("[Main] ERROR: Failed to start camera")
                return False
            
            if not self.audio.start():
                print("[Main] WARNING: Audio not started, continuing without audio")
                self.audio = None  # Set to None for easy null checks
            
            # Start conversation handler (voice input)
            if self.conversation.start():
                print("[Main] Voice input enabled (Press SPACE to speak)")
            else:
                print("[Main] WARNING: Voice input not started")
                self.conversation = None
            
            # Start telemetry (Phase 3)
            self.telemetry = TelemetryLogger(config_path=self.config_path)
            if self.telemetry.start():
                print("[Main] Telemetry logging enabled")
            else:
                print("[Main] WARNING: Telemetry not started")
                self.telemetry = None
            
            # Initialize navigation detectors
            print("\n[Main] Initializing navigation detectors...")
            detector_classes = [
                ('stair_detector', StairDetector),
                ('traffic_light_detector', TrafficLightDetector),
                ('crosswalk_detector', CrosswalkDetector),
                ('curb_detector', CurbDetector),
                ('door_detector', DoorDetector),
                ('retail_detector', RetailDetector),
                ('tactile_paving_detector', TactilePavingDetector),
            ]
            for attr_name, cls in detector_classes:
                try:
                    det = cls(config_path=self.config_path)
                    setattr(self, attr_name, det)
                except Exception as e:
                    print(f"[Main] WARNING: {cls.__name__} init failed: {e}")
                    setattr(self, attr_name, None)
            
            self._running = True
            self._start_time = time.time()
            
            print("\n" + "=" * 60)
            if self._show_video:
                print("System ready! Press 'q' to quit, SPACE to speak.")
            else:
                print("System ready! Running in headless mode.")
                print("Press Ctrl+C to quit, SPACE to speak.")
            print("=" * 60 + "\n")
            
            return True
            
        except Exception as e:
            print(f"[Main] ERROR during startup: {e}")
            self.stop()
            return False
    
    def stop(self) -> None:
        """Stop all modules and cleanup."""
        print("\n[Main] Stopping system...")
        self._running = False
        
        # Stop modules (in reverse order)
        if self.telemetry is not None:
            stats = self.telemetry.stop()
            print(f"[Main] Session stats: avg latency {stats.avg_latency_ms:.1f}ms, p99 {stats.p99_latency_ms:.1f}ms")
        
        if self.conversation is not None:
            self.conversation.stop()
        
        if self.audio is not None:
            self.audio.stop()
        
        if self.camera is not None:
            self.camera.stop()
        
        # Close video window
        if self._show_video:
            cv2.destroyAllWindows()
        
        # FIX: Force final garbage collection
        gc.collect()
        
        # Print final statistics
        self._print_final_stats()
        
        print("[Main] System stopped")
    
    def _get_scene_description(self) -> str:
        """Get detailed scene description for voice queries."""
        if self._latest_scene is None:
            return "No scene data available yet"
        
        # Use the full scene description method if available
        if self.scene_analyzer and hasattr(self.scene_analyzer, 'get_scene_description'):
            return self.scene_analyzer.get_scene_description()
        else:
            return self._latest_scene.scene_summary
    
    def run(self) -> None:
        """Main processing loop."""
        if not self._running:
            if not self.start():
                return
        
        # FIX: Use adaptive frame skip
        self._current_skip = self.camera.frame_skip
        
        while self._running:
            try:
                # Get frame from camera
                frame_packet = self.camera.get_frame(timeout=1.0)
                
                if frame_packet is None:
                    self._error_count += 1
                    if self._error_count >= self._max_errors:
                        print(f"[Main] Too many consecutive errors ({self._max_errors}), stopping")
                        break
                    continue
                
                # FIX: Validate frame before processing
                if not self._validate_frame(frame_packet.data):
                    print(f"[Main] WARNING: Invalid frame {frame_packet.frame_id}, skipping")
                    del frame_packet  # Release memory
                    continue
                
                self._error_count = 0
                self._frame_count += 1
                self._total_frames += 1
                
                # Adaptive frame skip
                if self._frame_count % self._current_skip != 0:
                    self._frames_skipped += 1  # Issue 1.12
                    
                    if self._show_video:
                        try:
                            cv2.imshow('Navigation System', frame_packet.data)
                            if cv2.waitKey(1) & 0xFF == self._quit_key:
                                print("[Main] Quit key pressed")
                                break
                        except cv2.error:
                            print("[Main] Display error, switching to headless")
                            self._show_video = False
                    
                    del frame_packet.data
                    del frame_packet
                    continue
                
                # Run detection
                loop_start = time.time()
                detection_result = self.detector.detect(
                    frame_packet.data,
                    frame_id=frame_packet.frame_id
                )
                detection_time = (time.time() - loop_start) * 1000
                
                self._total_detections += detection_result.count
                
                # Run safety analysis
                safety_start = time.time()
                safety_result = self.safety.analyze(
                    detection_result.detections,
                    frame_width=frame_packet.width,
                    frame_id=frame_packet.frame_id
                )
                safety_time = (time.time() - safety_start) * 1000
                
                self._total_alerts += len(safety_result.alerts)
                
                # Run scene analysis (Phase 2)
                scene_time = 0.0
                if self.scene_analyzer is not None:
                    scene_start = time.time()
                    self._latest_scene = self.scene_analyzer.analyze(
                        detection_result.detections,
                        frame_id=frame_packet.frame_id,
                        frame_width=frame_packet.width,
                        alerts=safety_result.alerts
                    )
                    scene_time = (time.time() - scene_start) * 1000
                
                # ---- Navigation detectors ----
                nav_det_start = time.time()
                nav_alerts = []
                frame_data = frame_packet.data
                
                # RGB-based detectors
                for det_attr, det_name in [
                    ('traffic_light_detector', 'Traffic Light'),
                    ('crosswalk_detector', 'Crosswalk'),
                    ('door_detector', 'Door'),
                    ('retail_detector', 'Retail'),
                    ('tactile_paving_detector', 'Tactile Paving'),
                ]:
                    det = getattr(self, det_attr, None)
                    if det is not None and det.is_enabled:
                        try:
                            result = det.detect(frame_data)
                            if result.detected:
                                nav_alerts.append(result)
                        except Exception as e:
                            if self._print_latency:
                                print(f"[Main] {det_name} error: {e}")
                
                # Depth-based detectors (stair, curb) — skipped when no depth map
                # These are wired up for future depth estimation integration
                
                nav_det_time = (time.time() - nav_det_start) * 1000
                
                # Announce navigation detector alerts
                if nav_alerts and self.audio:
                    for nav_alert in nav_alerts:
                        announcement = nav_alert.get_announcement()
                        if announcement:
                            self.audio.speak(announcement, priority=3)
                
                # Log nav detector alerts to telemetry
                if nav_alerts and self.telemetry:
                    for nav_alert in nav_alerts:
                        self.telemetry.log_alert(nav_alert)
                
                # Log telemetry (Phase 3) — Issue 1.4: inline metrics
                if self.telemetry is not None:
                    self.telemetry.log_latency(LatencyMetrics(
                        frame_id=frame_packet.frame_id,
                        timestamp=time.time(),
                        camera_ms=frame_packet.get_age(),
                        detection_ms=detection_time,
                        safety_ms=safety_time,
                        scene_ms=scene_time,
                        detection_count=detection_result.count,
                        alert_count=len(safety_result.alerts)
                    ))
                    
                    for alert in safety_result.alerts:
                        self.telemetry.log_alert(alert)
                
                # Announce alerts
                if safety_result.alerts and self.audio:
                    self.audio.announce_alerts(safety_result.alerts)
                    
                    if self._print_alerts:
                        for alert in safety_result.alerts[:3]:
                            print(f"  🔊 [{alert.danger_level.upper()}] {alert.get_announcement()}")
                
                # Calculate total latency
                total_time = (time.time() - loop_start) * 1000
                
                # FIX: Adjust frame skip based on performance
                self._adjust_frame_skip(total_time)
                
                # Print latency info — Issue 1.6: cache FPS
                if self._frame_count % self.FPS_CACHE_INTERVAL == 0:
                    self._cached_elapsed = time.time() - self._start_time
                    self._cached_fps = (self._total_frames / self._cached_elapsed
                                       if self._cached_elapsed > 0 else 0)
                
                if self._print_latency and self._frame_count % self.LATENCY_PRINT_INTERVAL == 0:
                    print(f"[Latency] Detection: {detection_time:.0f}ms | "
                          f"Safety: {safety_time:.0f}ms | "
                          f"Total: {total_time:.0f}ms | "
                          f"FPS: {self._cached_fps:.1f} | "
                          f"Skip: {self._current_skip}")
                
                # Print detections
                if self._print_detections and detection_result.count > 0:
                    objects = [f"{d.class_name}({d.confidence:.2f})" 
                              for d in detection_result.detections[:5]]
                    print(f"[Detected] {', '.join(objects)}")
                
                # Update display if enabled
                if self._show_video:
                    display_frame = self._draw_overlay(
                        frame_packet.data,
                        detection_result,
                        safety_result
                    )
                    
                    # Issue 1.3: Wrap cv2 calls in try/except
                    try:
                        cv2.imshow('Navigation System', display_frame)
                        del display_frame  # Issue 1.2: Release immediately
                        
                        key = cv2.waitKey(1) & 0xFF
                        if key == self._quit_key:
                            print("[Main] Quit key pressed")
                            break
                        elif key == ord(' '):
                            if self.conversation and self.conversation.is_running:
                                print("[Main] Voice input triggered via keyboard fallback...")
                    except cv2.error as e:
                        print(f"[Main] Display error: {e}, switching to headless")
                        self._show_video = False
                        try:
                            cv2.destroyAllWindows()
                        except:
                            pass
                
                # Issue 1.1: Clear scene tracking data (keeps summary only)
                if self._latest_scene is not None and hasattr(self._latest_scene, 'tracked_objects'):
                    self._latest_scene.tracked_objects.clear()
                
                # Release frame memory
                del frame_packet.data
                del detection_result
                del safety_result
                del frame_packet
                
                # Periodic garbage collection
                if self._frame_count - self._last_gc_frame >= self._gc_interval:
                    gc.collect()
                    self._last_gc_frame = self._frame_count
                
            except Exception as e:
                print(f"[Main] ERROR in main loop: {e}")
                traceback.print_exc()
                
                # Log to telemetry if available
                if self.telemetry:
                    self.telemetry.log_error(str(e), traceback.format_exc())
                
                self._error_count += 1
                if self._error_count >= self._max_errors:
                    print(f"[Main] Too many errors ({self._max_errors}), stopping")
                    break
        
        self.stop()
    
    def _get_zone_overlay(self, width: int, height: int) -> np.ndarray:
        """Get cached zone overlay (Issue 1.14)."""
        if self._zone_overlay is None or self._zone_overlay_size != (width, height):
            overlay = np.zeros((height, width, 3), dtype=np.uint8)
            
            left_line = int(width * 0.3)
            right_line = int(width * 0.7)
            cv2.line(overlay, (left_line, 0), (left_line, height), (100, 100, 100), 1)
            cv2.line(overlay, (right_line, 0), (right_line, height), (100, 100, 100), 1)
            
            cv2.putText(overlay, "LEFT", (10, height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            cv2.putText(overlay, "CENTER", (width//2 - 30, height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            cv2.putText(overlay, "RIGHT", (width - 60, height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            
            self._zone_overlay = overlay
            self._zone_overlay_size = (width, height)
        
        return self._zone_overlay
    
    def _draw_overlay(self, frame, detection_result, safety_result) -> any:
        """Draw debug overlay on frame."""
        display = frame.copy()
        height, width = display.shape[:2]
        
        # Issue 1.14: Use cached zone overlay
        if self._show_zones:
            zone_overlay = self._get_zone_overlay(width, height)
            mask = zone_overlay > 0
            display[mask] = zone_overlay[mask]
        
        # Draw bounding boxes with danger colors
        if self._show_bboxes:
            danger_colors = {
                'critical': (0, 0, 255),
                'warning': (0, 165, 255),
                'info': (0, 255, 0)
            }
            
            for det in detection_result.detections:
                x1, y1, x2, y2 = det.bbox
                cv2.rectangle(display, (x1, y1), (x2, y2), (128, 128, 128), 1)
                label = f"{det.class_name} {det.confidence:.2f}"
                cv2.putText(display, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
            
            for alert in safety_result.alerts:
                x1, y1, x2, y2 = alert.bbox
                color = danger_colors.get(alert.danger_level, (0, 255, 0))
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                
                label = f"{alert.class_name} {alert.distance_m}m [{alert.danger_level}]"
                cv2.putText(display, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Issue 1.6: Use cached FPS
        if self._show_fps:
            fps_text = f"FPS: {self._cached_fps:.1f} | Frame: {self._frame_count} | Skip: {self._current_skip}"
            cv2.putText(display, fps_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        alert_text = (f"Alerts: {len(safety_result.alerts)} "
                     f"(C:{safety_result.critical_count} "
                     f"W:{safety_result.warning_count} "
                     f"I:{safety_result.info_count})")
        cv2.putText(display, alert_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return display
    
    def _print_final_stats(self) -> None:
        """Print final session statistics."""
        elapsed = time.time() - self._start_time if self._start_time > 0 else 0
        
        print("\n" + "=" * 60)
        print("Session Statistics")
        print("=" * 60)
        print(f"  Duration: {elapsed:.1f} seconds")
        print(f"  Total frames: {self._total_frames}")
        print(f"  Average FPS: {self._total_frames / elapsed if elapsed > 0 else 0:.1f}")
        print(f"  Total detections: {self._total_detections}")
        print(f"  Total alerts: {self._total_alerts}")
        
        # Issue 1.12: Frame drop stats
        if self._total_frames > 0:
            skip_rate = (self._frames_skipped / 
                        (self._total_frames + self._frames_skipped) * 100)
            print(f"  Frames skipped: {self._frames_skipped} ({skip_rate:.1f}%)")
        
        if self.detector:
            det_stats = self.detector.get_stats()
            print(f"  Avg inference time: {det_stats['avg_inference_time_ms']:.1f}ms")
        
        if self.safety:
            safety_stats = self.safety.get_stats()
            print(f"  Deduplicated alerts: {safety_stats['deduplicated_alerts']}")
        
        if self.audio:
            audio_stats = self.audio.get_stats()
            print(f"  Announcements: {audio_stats['total_announcements']}")
            print(f"  Interrupted: {audio_stats['interrupted_count']}")
        
        # Navigation detector stats
        det_names = [
            ('stair_detector', 'Stair'),
            ('traffic_light_detector', 'Traffic Light'),
            ('crosswalk_detector', 'Crosswalk'),
            ('curb_detector', 'Curb'),
            ('door_detector', 'Door'),
            ('retail_detector', 'Retail'),
            ('tactile_paving_detector', 'Tactile Paving'),
        ]
        active_dets = []
        for attr, label in det_names:
            det = getattr(self, attr, None)
            if det is not None:
                stats = det.get_stats()
                if stats.get('total_detections', 0) > 0:
                    active_dets.append(f"{label}: {stats['total_detections']}")
        if active_dets:
            print(f"  Nav detections: {', '.join(active_dets)}")
        
        print("=" * 60)
    
    def run_with_restart(self, max_restarts: int = 3) -> None:
        """Run with automatic restart on crash (Issue 1.16)."""
        restart_count = 0
        
        while restart_count < max_restarts:
            try:
                self.run()
                break  # Normal exit
            except Exception as e:
                restart_count += 1
                print(f"\n[Main] CRASH: {e}")
                print(f"[Main] Restart {restart_count}/{max_restarts} in 5 seconds...")
                
                try:
                    self.stop()
                except:
                    pass
                
                time.sleep(5)
                
                if restart_count >= max_restarts:
                    print("[Main] Max restarts reached, giving up")
                    raise


def main():
    """Entry point with argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Voice Navigation System for Visually Impaired",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/main.py                  # Normal mode
  python src/main.py --demo           # Demo mode (sample video, no mic)
  python src/main.py --config my.yaml # Custom config
        """
    )
    parser.add_argument('--demo', action='store_true',
                       help='Demo mode: uses sample video, disables voice input')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--validate', action='store_true',
                       help='Validate config and exit')
    
    args = parser.parse_args()
    
    # Determine config path using Path-based resolution
    if args.config:
        config_path = str(Path(args.config).resolve())
    else:
        config_path = str(BASE_DIR / "config" / "settings.yaml")
    
    # Check if config exists
    if not os.path.exists(config_path):
        print(f"[Main] ERROR: Config file not found: {config_path}")
        sys.exit(1)
    
    print(f"[Main] Using config: {config_path}")
    
    # Demo mode: create temp config with video source
    if args.demo:
        print("[Main] Running in DEMO mode")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
        
        # Override settings for demo
        demo_video = str(BASE_DIR / "data" / "demo_video.mp4")
        if not os.path.exists(demo_video):
            print(f"[Main] Demo video not found: {demo_video}")
            print("[Main] Using camera with voice input disabled")
        else:
            config['camera']['source'] = demo_video
        
        config['voice_input'] = config.get('voice_input', {})
        config['voice_input']['enabled'] = False
        config['debug'] = config.get('debug', {})
        config['debug']['show_video'] = True
        
        # Save temp config
        demo_config_path = str(BASE_DIR / "config" / "_demo_settings.yaml")
        with open(demo_config_path, 'w') as f:
            yaml.dump(config, f)
        config_path = demo_config_path
    
    # Validate config mode
    if args.validate:
        print("[Main] Validating configuration...")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
        
        errors = []
        required = ['camera', 'yolo', 'safety', 'audio']
        for section in required:
            if section not in config:
                errors.append(f"Missing section: {section}")
        
        # Check thresholds
        conf = config.get('yolo', {}).get('confidence_threshold', 0)
        if not 0 < conf < 1:
            errors.append(f"Invalid confidence_threshold: {conf}")
        
        if errors:
            print("Validation FAILED:")
            for e in errors:
                print(f"  ✗ {e}")
            sys.exit(1)
        else:
            print("✓ Configuration valid")
            sys.exit(0)
    
    # Create and run system
    system = NavigationSystem(config_path=config_path)
    system.run()


if __name__ == "__main__":
    main()

