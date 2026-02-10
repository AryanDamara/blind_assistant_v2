"""
Camera Capture Module
---------------------
Handles video capture from webcam or video file.
Runs as a background thread, pushing frames to an internal queue.

Features:
- Supports webcam (int) or video file (str path)
- Internal queue with configurable buffer size
- Timestamps on each frame with latency tracking
- Auto-restart on failure
- Warmup frame handling
- Statistics tracking
- Configurable video looping
"""

import cv2
import time
import queue
import threading
import yaml
import os
from dataclasses import dataclass
from typing import Optional, Union
import numpy as np


@dataclass
class Frame:
    """
    Frame container with complete metadata for latency tracking.
    Aligned with telemetry logging format.
    """
    data: np.ndarray              # The actual frame (BGR format)
    frame_id: int                 # Sequential frame counter
    capture_time: float           # When frame was captured (time.time())
    queue_time: float             # When frame was added to queue
    width: int
    height: int
    source: str                   # Camera source identifier
    
    def get_queue_latency(self) -> float:
        """Time spent waiting in queue (milliseconds)."""
        return (time.time() - self.queue_time) * 1000
    
    def get_age(self) -> float:
        """Total age since capture (milliseconds)."""
        return (time.time() - self.capture_time) * 1000
    
    def to_dict(self) -> dict:
        """Convert to dict for telemetry logging."""
        return {
            'frame_id': self.frame_id,
            'capture_time': self.capture_time,
            'queue_time': self.queue_time,
            'width': self.width,
            'height': self.height,
            'source': self.source,
            'age_ms': self.get_age(),
            'queue_latency_ms': self.get_queue_latency()
        }


class CameraCapture:
    """
    Thread-safe camera capture with internal queue.
    
    Usage:
        camera = CameraCapture(config_path="config/settings.yaml")
        camera.start()
        
        while running:
            frame = camera.get_frame(timeout=1.0)
            if frame:
                process(frame.data)
        
        camera.stop()
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize camera capture with configuration."""
        # Load configuration
        self._load_config(config_path)
        
        # Internal state
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=self._buffer_size)
        self._frame_counter: int = 0
        
        # Thread control
        self._capture_thread: Optional[threading.Thread] = None
        self._running: bool = False
        self._stop_event: threading.Event = threading.Event()
        
        # Error tracking
        self._consecutive_errors: int = 0
        self._last_error: Optional[str] = None
        
        # Statistics tracking
        self._frames_captured: int = 0
        self._frames_dropped: int = 0
        self._total_queue_latency: float = 0.0
        self._max_queue_latency: float = 0.0
        
    def _load_config(self, config_path: str) -> None:
        """Load camera settings from YAML config."""
        # Defaults
        self._source: Union[int, str] = 0
        self._width: int = 640
        self._height: int = 480
        self._fps: int = 30
        self._buffer_size: int = 2
        self._frame_skip: int = 3
        self._auto_restart: bool = True
        self._restart_delay: float = 2.0
        self._warmup_frames: int = 10
        self._loop_video: bool = True
        
        # Load from file if exists
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                cam_cfg = config.get('camera', {})
                
                self._source = cam_cfg.get('source', 0)
                self._width = cam_cfg.get('width', 640)
                self._height = cam_cfg.get('height', 480)
                self._fps = cam_cfg.get('fps', 30)
                self._buffer_size = cam_cfg.get('buffer_size', 2)
                self._frame_skip = cam_cfg.get('frame_skip', 3)
                self._auto_restart = cam_cfg.get('auto_restart', True)
                self._restart_delay = cam_cfg.get('restart_delay_sec', 2.0)
                self._warmup_frames = cam_cfg.get('warmup_frames', 10)
                self._loop_video = cam_cfg.get('loop_video', True)
                self._backend = cam_cfg.get('backend', None)  # e.g., 'avfoundation' for macOS
    
    def _validate_source(self) -> bool:
        """Validate source exists before opening."""
        if isinstance(self._source, str):
            if not os.path.exists(self._source):
                self._last_error = f"Video file not found: {self._source}"
                print(f"[CameraCapture] ERROR: {self._last_error}")
                return False
        return True
    
    def _init_capture(self) -> bool:
        """Initialize the video capture device."""
        try:
            # Validate source first
            if not self._validate_source():
                return False
            
            # Release existing capture if any
            if self._cap is not None:
                self._cap.release()
            
            # Open capture source with optional backend
            if hasattr(self, '_backend') and self._backend:
                backend_map = {
                    'avfoundation': cv2.CAP_AVFOUNDATION,
                    'v4l2': cv2.CAP_V4L2,
                    'dshow': cv2.CAP_DSHOW,
                    'gstreamer': cv2.CAP_GSTREAMER,
                }
                backend = backend_map.get(self._backend.lower())
                if backend is not None:
                    print(f"[CameraCapture] Using backend: {self._backend}")
                    self._cap = cv2.VideoCapture(self._source, backend)
                else:
                    print(f"[CameraCapture] Unknown backend '{self._backend}', using default")
                    self._cap = cv2.VideoCapture(self._source)
            else:
                self._cap = cv2.VideoCapture(self._source)
            
            if not self._cap.isOpened():
                self._last_error = f"Failed to open camera source: {self._source}"
                print(f"[CameraCapture] ERROR: {self._last_error}")
                return False
            
            # Configure camera properties (only for webcam, not video files)
            if isinstance(self._source, int):
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
                self._cap.set(cv2.CAP_PROP_FPS, self._fps)
                
                # Reduce buffer to minimize latency
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Perform warmup (skip first N frames for camera exposure adjustment)
            print(f"[CameraCapture] Warming up ({self._warmup_frames} frames)...")
            for i in range(self._warmup_frames):
                ret, _ = self._cap.read()
                if not ret:
                    print(f"[CameraCapture] WARNING: Warmup frame {i} failed")
            
            # Get actual resolution (may differ from requested)
            actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self._cap.get(cv2.CAP_PROP_FPS))
            
            print(f"[CameraCapture] Initialized: {actual_width}x{actual_height} @ {actual_fps} FPS")
            print(f"[CameraCapture] Source: {self._source}")
            
            self._consecutive_errors = 0
            return True
            
        except Exception as e:
            self._last_error = str(e)
            print(f"[CameraCapture] ERROR initializing: {e}")
            return False
    
    def _capture_loop(self) -> None:
        """Main capture loop running in background thread."""
        print("[CameraCapture] Capture thread started")
        
        try:
            while not self._stop_event.is_set():
                # Check if capture is valid
                if self._cap is None or not self._cap.isOpened():
                    if self._auto_restart:
                        print(f"[CameraCapture] Attempting restart in {self._restart_delay}s...")
                        time.sleep(self._restart_delay)
                        if not self._init_capture():
                            self._consecutive_errors += 1
                            continue
                    else:
                        print("[CameraCapture] Camera not available, stopping")
                        break
                
                # Read frame
                ret, frame = self._cap.read()
                capture_timestamp = time.time()
                
                if not ret or frame is None:
                    self._consecutive_errors += 1
                    self._last_error = "Failed to read frame"
                    
                    # Check if video file ended
                    if isinstance(self._source, str):
                        print("[CameraCapture] Video file ended")
                        if self._loop_video:
                            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            print("[CameraCapture] Looping video...")
                            continue
                        else:
                            print("[CameraCapture] Stopping (loop disabled)")
                            break
                    
                    if self._consecutive_errors > 10:
                        print("[CameraCapture] Too many consecutive errors")
                        if self._auto_restart:
                            self._cap.release()
                            self._cap = None
                    continue
                
                # Reset error counter on success
                self._consecutive_errors = 0
                self._frames_captured += 1
                
                # Create frame packet with both timestamps
                queue_timestamp = time.time()
                packet = Frame(
                    data=frame,
                    frame_id=self._frames_captured,
                    capture_time=capture_timestamp,
                    queue_time=queue_timestamp,
                    width=frame.shape[1],
                    height=frame.shape[0],
                    source=str(self._source)
                )
                
                # Add to queue (non-blocking, drop old frames if full)
                if self._frame_queue.full():
                    try:
                        dropped = self._frame_queue.get_nowait()
                        self._frames_dropped += 1
                        
                        # Warn if dropped frame was stale (>500ms old)
                        dropped_latency = dropped.get_age()
                        if dropped_latency > 500:
                            print(f"[CameraCapture] ⚠️  Dropped stale frame {dropped.frame_id} "
                                  f"(age: {dropped_latency:.0f}ms)")
                    except queue.Empty:
                        pass
                
                try:
                    self._frame_queue.put_nowait(packet)
                except queue.Full:
                    pass
        
        except Exception as e:
            print(f"[CameraCapture] FATAL ERROR in capture loop: {e}")
            self._last_error = str(e)
        
        finally:
            # Ensure cleanup happens
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            print("[CameraCapture] Capture thread stopped")
    
    def start(self) -> bool:
        """
        Start the camera capture thread.
        
        Returns:
            bool: True if started successfully
        """
        if self._running:
            print("[CameraCapture] Already running")
            return True
        
        # Print configuration summary
        print(f"""
[CameraCapture] Configuration:
  Source: {self._source}
  Resolution: {self._width}x{self._height}
  FPS: {self._fps}
  Buffer size: {self._buffer_size}
  Frame skip: {self._frame_skip}
  Auto-restart: {self._auto_restart}
  Loop video: {self._loop_video}
        """)
        
        # Initialize capture
        if not self._init_capture():
            return False
        
        # Start capture thread
        self._stop_event.clear()
        self._running = True
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="CameraCaptureThread",
            daemon=True
        )
        self._capture_thread.start()
        
        return True
    
    def stop(self) -> None:
        """Stop the camera capture thread and print statistics."""
        if not self._running:
            return
        
        print("[CameraCapture] Stopping...")
        self._stop_event.set()
        self._running = False
        
        # Wait for thread to finish
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=5.0)
            self._capture_thread = None
        
        # Clear queue
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break
        
        # Print statistics
        stats = self.get_stats()
        print(f"""
[CameraCapture] Session Statistics:
  Frames captured: {stats['frames_captured']}
  Frames dropped: {stats['frames_dropped']} ({stats['drop_rate']:.1%})
  Avg queue latency: {stats['avg_queue_latency_ms']:.1f}ms
  Max queue latency: {stats['max_queue_latency_ms']:.1f}ms
        """)
        
        print("[CameraCapture] Stopped")
    
    def get_frame(self, timeout: float = 1.0) -> Optional[Frame]:
        """
        Get the next frame from the queue.
        
        Args:
            timeout: Max seconds to wait for a frame
            
        Returns:
            Frame or None if no frame available
        """
        try:
            frame = self._frame_queue.get(timeout=timeout)
            
            # Track queue latency statistics
            queue_latency = frame.get_queue_latency()
            self._total_queue_latency += queue_latency
            self._max_queue_latency = max(self._max_queue_latency, queue_latency)
            
            return frame
        except queue.Empty:
            return None
    
    def get_latest_frame(self) -> Optional[Frame]:
        """
        Get the most recent frame, discarding any older ones.
        
        Returns:
            Frame or None if no frame available
        """
        latest = None
        discarded = 0
        
        # Get all available frames, keep only the last one
        while not self._frame_queue.empty():
            try:
                frame = self._frame_queue.get_nowait()
                if latest is not None:
                    discarded += 1
                latest = frame
            except queue.Empty:
                break
        
        if discarded > 0:
            print(f"[CameraCapture] Discarded {discarded} old frames")
        
        # Track latency if we got a frame
        if latest is not None:
            queue_latency = latest.get_queue_latency()
            self._total_queue_latency += queue_latency
            self._max_queue_latency = max(self._max_queue_latency, queue_latency)
        
        return latest
    
    def get_stats(self) -> dict:
        """Get camera statistics for monitoring."""
        avg_queue_latency = (
            self._total_queue_latency / max(1, self._frames_captured)
        )
        
        return {
            'frames_captured': self._frames_captured,
            'frames_dropped': self._frames_dropped,
            'drop_rate': self._frames_dropped / max(1, self._frames_captured),
            'avg_queue_latency_ms': avg_queue_latency,
            'max_queue_latency_ms': self._max_queue_latency,
            'queue_size': self.queue_size
        }
    
    @property
    def is_running(self) -> bool:
        """Check if capture is running."""
        return self._running
    
    @property
    def frame_skip(self) -> int:
        """Get configured frame skip value (for main loop to use)."""
        return self._frame_skip
    
    @property
    def last_error(self) -> Optional[str]:
        """Get the last error message."""
        return self._last_error
    
    @property
    def queue_size(self) -> int:
        """Get current number of frames in queue."""
        return self._frame_queue.qsize()
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


# ============================================
# Testing / Demo
# ============================================
if __name__ == "__main__":
    """Test camera capture module standalone."""
    
    print("=" * 50)
    print("Camera Capture Module Test")
    print("=" * 50)
    
    # Use context manager for automatic cleanup
    with CameraCapture(config_path="config/settings.yaml") as camera:
        print(f"\nFrame skip setting: {camera.frame_skip}")
        print("Press 'q' to quit\n")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            # Get frame
            frame_packet = camera.get_frame(timeout=1.0)
            
            if frame_packet is None:
                print("No frame received")
                continue
            
            frame_count += 1
            
            # Calculate FPS
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Get frame age
            age_ms = frame_packet.get_age()
            
            # Display frame info
            cv2.putText(
                frame_packet.data,
                f"Frame: {frame_packet.frame_id} | FPS: {fps:.1f} | Age: {age_ms:.0f}ms",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            cv2.putText(
                frame_packet.data,
                f"Queue: {camera.queue_size} | Dropped: {camera.get_stats()['frames_dropped']}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            
            # Show frame
            cv2.imshow("Camera Test", frame_packet.data)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cv2.destroyAllWindows()
    print("\nTest complete!")