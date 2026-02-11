#!/usr/bin/env python3
"""
Voice Navigation System - Main Orchestrator (FIXED)
--------------------------------------------
Real-time navigation assistance for visually impaired users.

Fixed Issues:
- Memory leak in main loop (frame cleanup + gc)
- Missing frame validation
- Adaptive frame skipping

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
import yaml
import cv2
import gc  # FIX: Added for garbage collection
from pathlib import Path
from typing import Optional

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
        
        # FIX: Adaptive frame skipping
        self._adaptive_skip = True
        self._current_skip = 3  # Start with default
        self._target_latency_ms = 800
        self._latency_history = []
        
        # FIX: Garbage collection tracking
        self._gc_interval = 1000  # Collect every 1000 frames
        self._last_gc_frame = 0
        
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
    
    # FIX: Added adaptive frame skip adjustment
    def _adjust_frame_skip(self, latency_ms: float) -> None:
        """
        Adjust frame skip based on recent latency.
        
        Args:
            latency_ms: Latest processing latency
        """
        if not self._adaptive_skip:
            return
        
        # Track latency history (last 10 frames)
        self._latency_history.append(latency_ms)
        if len(self._latency_history) > 10:
            self._latency_history.pop(0)
        
        # Calculate average latency
        avg_latency = sum(self._latency_history) / len(self._latency_history)
        
        # Adjust skip based on performance
        if avg_latency > self._target_latency_ms:
            # Too slow - skip more frames
            self._current_skip = min(self._current_skip + 1, 10)
            if self._print_latency:
                print(f"[Main] ⚠️ Latency high ({avg_latency:.0f}ms), increasing skip to {self._current_skip}")
        
        elif avg_latency < self._target_latency_ms * 0.7:
            # Fast enough - can process more frames
            self._current_skip = max(self._current_skip - 1, 1)
            if self._print_latency:
                print(f"[Main] ✓ Latency good ({avg_latency:.0f}ms), decreasing skip to {self._current_skip}")
    
    def start(self) -> bool:
        """
        Initialize and start all modules.
        
        Returns:
            bool: True if started successfully
        """
        print("=" * 60)
        print("Voice Navigation System - Starting")
        print("=" * 60)
        
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
                
                # FIX: Use adaptive frame skip
                if self._frame_count % self._current_skip != 0:
                    # Still update display if enabled
                    if self._show_video:
                        cv2.imshow('Navigation System', frame_packet.data)
                        if cv2.waitKey(1) & 0xFF == self._quit_key:
                            print("[Main] Quit key pressed")
                            break
                    
                    # FIX: Release frame memory before skipping
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
                
                # Log telemetry (Phase 3)
                if self.telemetry is not None:
                    metrics = LatencyMetrics(
                        frame_id=frame_packet.frame_id,
                        timestamp=time.time(),
                        camera_ms=frame_packet.get_age(),
                        detection_ms=detection_time,
                        safety_ms=safety_time,
                        scene_ms=scene_time,
                        detection_count=detection_result.count,
                        alert_count=len(safety_result.alerts)
                    )
                    self.telemetry.log_latency(metrics)
                    
                    # Log alerts
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
                
                # Print latency info
                if self._print_latency and self._frame_count % 30 == 0:
                    elapsed = time.time() - self._start_time
                    actual_fps = self._total_frames / elapsed if elapsed > 0 else 0
                    print(f"[Latency] Detection: {detection_time:.0f}ms | "
                          f"Safety: {safety_time:.0f}ms | "
                          f"Total: {total_time:.0f}ms | "
                          f"FPS: {actual_fps:.1f} | "
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
                    cv2.imshow('Navigation System', display_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == self._quit_key:
                        print("[Main] Quit key pressed")
                        break
                    # Keyboard fallback for PTT (space bar)
                    elif key == ord(' '):
                        if self.conversation and self.conversation.is_running:
                            print("[Main] Voice input triggered via keyboard fallback...")
                
                # FIX: CRITICAL - Release frame memory
                del frame_packet.data
                del detection_result
                del safety_result
                del frame_packet
                
                # FIX: Periodic garbage collection
                if self._frame_count - self._last_gc_frame >= self._gc_interval:
                    gc.collect()
                    self._last_gc_frame = self._frame_count
                
            except Exception as e:
                print(f"[Main] ERROR in main loop: {e}")
                import traceback
                traceback.print_exc()
                
                self._error_count += 1
                if self._error_count >= self._max_errors:
                    print(f"[Main] Too many errors, stopping")
                    break
        
        self.stop()
    
    def _draw_overlay(self, frame, detection_result, safety_result) -> any:
        """Draw debug overlay on frame."""
        display = frame.copy()
        height, width = display.shape[:2]
        
        # Draw zone lines
        if self._show_zones:
            left_line = int(width * 0.3)
            right_line = int(width * 0.7)
            cv2.line(display, (left_line, 0), (left_line, height), (100, 100, 100), 1)
            cv2.line(display, (right_line, 0), (right_line, height), (100, 100, 100), 1)
            
            # Zone labels
            cv2.putText(display, "LEFT", (10, height - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            cv2.putText(display, "CENTER", (width//2 - 30, height - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            cv2.putText(display, "RIGHT", (width - 60, height - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        # Draw bounding boxes with danger colors
        if self._show_bboxes:
            danger_colors = {
                'critical': (0, 0, 255),   # Red
                'warning': (0, 165, 255),  # Orange
                'info': (0, 255, 0)        # Green
            }
            
            # Draw all detections in gray first (thin boxes)
            for det in detection_result.detections:
                x1, y1, x2, y2 = det.bbox
                cv2.rectangle(display, (x1, y1), (x2, y2), (128, 128, 128), 1)
                label = f"{det.class_name} {det.confidence:.2f}"
                cv2.putText(display, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
            
            # Overdraw alerts with danger colors (thicker boxes)
            for alert in safety_result.alerts:
                x1, y1, x2, y2 = alert.bbox
                color = danger_colors.get(alert.danger_level, (0, 255, 0))
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                
                label = f"{alert.class_name} {alert.distance_m}m [{alert.danger_level}]"
                cv2.putText(display, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw FPS
        if self._show_fps:
            elapsed = time.time() - self._start_time
            fps = self._total_frames / elapsed if elapsed > 0 else 0
            fps_text = f"FPS: {fps:.1f} | Frame: {self._frame_count} | Skip: {self._current_skip}"
            cv2.putText(display, fps_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw alert summary
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
        
        print("=" * 60)


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

