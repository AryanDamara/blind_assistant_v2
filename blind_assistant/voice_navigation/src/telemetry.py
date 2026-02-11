"""
Telemetry Module
----------------
Centralized metrics collection, session management, and performance monitoring.

Features:
- Session management with timestamped directories
- JSON logging for detections, alerts, latency, LLM responses
- CSV export for analysis
- Statistics calculation (rolling avg, min/max, percentiles)
- Anomaly detection (latency spikes, errors)
- Memory usage tracking

Thread: Main thread (synchronous logging with async file writes)
"""

import os
import json
import time
import csv
import yaml
import threading
import queue
import sys
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from collections import deque
import statistics


@dataclass
class LatencyMetrics:
    """Metrics for a single frame processing cycle."""
    frame_id: int
    timestamp: float
    
    # Component latencies (milliseconds)
    camera_ms: float = 0.0
    detection_ms: float = 0.0
    safety_ms: float = 0.0
    scene_ms: float = 0.0
    audio_ms: float = 0.0
    llm_ms: float = 0.0
    
    # Computed total
    total_ms: float = 0.0
    
    # Additional context
    detection_count: int = 0
    alert_count: int = 0
    
    def calculate_total(self) -> float:
        """Calculate total latency from components."""
        self.total_ms = (self.camera_ms + self.detection_ms + 
                        self.safety_ms + self.scene_ms + self.audio_ms)
        return self.total_ms
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class SessionStats:
    """Aggregated statistics for a telemetry session."""
    session_id: str
    start_time: float
    end_time: float = 0.0
    
    # Frame statistics
    total_frames: int = 0
    dropped_frames: int = 0
    
    # Latency statistics (milliseconds)
    avg_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    
    # Component averages
    avg_camera_ms: float = 0.0
    avg_detection_ms: float = 0.0
    avg_safety_ms: float = 0.0
    avg_scene_ms: float = 0.0
    avg_audio_ms: float = 0.0
    
    # Detection statistics
    total_detections: int = 0
    total_alerts: int = 0
    alerts_by_level: Dict[str, int] = field(default_factory=dict)
    
    # Anomalies detected
    latency_spikes: int = 0
    errors: int = 0
    
    # Resource usage
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class SessionManager:
    """Manages telemetry session lifecycle and directories."""
    
    def __init__(self, base_log_dir: str = "data/logs"):
        self._base_log_dir = base_log_dir
        self._current_session: Optional[str] = None
        self._session_dir: Optional[str] = None
        self._start_time: float = 0.0
        
        # Ensure base directory exists
        os.makedirs(base_log_dir, exist_ok=True)
    
    def create_session(self) -> str:
        """Create a new session with timestamped directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._current_session = f"session_{timestamp}"
        self._session_dir = os.path.join(self._base_log_dir, self._current_session)
        self._start_time = time.time()
        
        os.makedirs(self._session_dir, exist_ok=True)
        
        # Create session metadata file
        metadata = {
            'session_id': self._current_session,
            'start_time': datetime.now().isoformat(),
            'start_timestamp': self._start_time,
            'platform': sys.platform,
            'python_version': sys.version
        }
        
        with open(os.path.join(self._session_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"[Telemetry] Session created: {self._current_session}")
        return self._current_session
    
    def get_current_session(self) -> Optional[str]:
        """Get current session ID."""
        return self._current_session
    
    def get_session_dir(self) -> Optional[str]:
        """Get current session directory path."""
        return self._session_dir
    
    def get_session_duration(self) -> float:
        """Get session duration in seconds."""
        if self._start_time == 0:
            return 0.0
        return time.time() - self._start_time
    
    def close_session(self, stats: SessionStats) -> None:
        """Close current session and write final statistics."""
        if self._session_dir is None:
            return
        
        stats.end_time = time.time()
        
        # Write final statistics
        stats_path = os.path.join(self._session_dir, 'session_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats.to_dict(), f, indent=2)
        
        # Update metadata
        metadata_path = os.path.join(self._session_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            metadata['end_time'] = datetime.now().isoformat()
            metadata['duration_seconds'] = self.get_session_duration()
            metadata['total_frames'] = stats.total_frames
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        print(f"[Telemetry] Session closed: {self._current_session}")
        print(f"[Telemetry] Duration: {self.get_session_duration():.1f}s, Frames: {stats.total_frames}")
        
        self._current_session = None
        self._session_dir = None


class TelemetryLogger:
    """
    Centralized telemetry logging with async file writes.
    
    Usage:
        logger = TelemetryLogger(config_path="config/settings.yaml")
        logger.start()
        
        # In main loop:
        metrics = LatencyMetrics(frame_id=1, timestamp=time.time())
        metrics.detection_ms = 50.0
        logger.log_latency(metrics)
        logger.log_alert(alert)
        
        # On shutdown:
        stats = logger.get_stats()
        logger.stop()
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self._load_config(config_path)
        
        self._session = SessionManager(self._log_directory)
        
        # Async write queue
        self._write_queue: queue.Queue = queue.Queue(maxsize=1000)
        self._write_thread: Optional[threading.Thread] = None
        self._running: bool = False
        
        # File handles
        self._latency_file: Optional[Any] = None
        self._events_file: Optional[Any] = None
        self._csv_writer: Optional[csv.DictWriter] = None
        
        # Statistics tracking
        self._latency_history: deque = deque(maxlen=1000)
        self._component_history: Dict[str, deque] = {
            'camera': deque(maxlen=500),
            'detection': deque(maxlen=500),
            'safety': deque(maxlen=500),
            'scene': deque(maxlen=500),
            'audio': deque(maxlen=500),
            'llm': deque(maxlen=500)
        }
        
        # Counters
        self._total_frames: int = 0
        self._total_detections: int = 0
        self._total_alerts: int = 0
        self._alerts_by_level: Dict[str, int] = {}
        self._error_count: int = 0
        self._latency_spikes: int = 0
        
        # Memory tracking
        self._memory_samples: deque = deque(maxlen=100)
        self._peak_memory_mb: float = 0.0
    
    def _load_config(self, config_path: str) -> None:
        """Load telemetry configuration from YAML."""
        # Defaults
        self._enabled: bool = True
        self._log_directory: str = "data/logs"
        self._log_level: str = "INFO"
        self._log_detections: bool = True
        self._log_latency: bool = True
        self._log_alerts: bool = True
        self._log_llm: bool = True
        self._warning_threshold_ms: float = 1000.0
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
                telem_cfg = config.get('telemetry', {})
                self._enabled = telem_cfg.get('enabled', True)
                self._log_directory = telem_cfg.get('log_directory', 'data/logs')
                self._log_level = telem_cfg.get('log_level', 'INFO')
                self._log_detections = telem_cfg.get('log_detections', True)
                self._log_latency = telem_cfg.get('log_latency', True)
                self._log_alerts = telem_cfg.get('log_alerts', True)
                self._log_llm = telem_cfg.get('log_llm_responses', True)
                self._warning_threshold_ms = telem_cfg.get(
                    'performance_warning_threshold_ms', 1000.0)
    
    def start(self) -> bool:
        """Start telemetry logging session."""
        if not self._enabled:
            print("[Telemetry] Disabled in config")
            return False
        
        # Create new session
        session_id = self._session.create_session()
        session_dir = self._session.get_session_dir()
        
        if session_dir is None:
            return False
        
        # Open log files
        try:
            # JSON lines file for latency metrics
            latency_path = os.path.join(session_dir, 'latency.jsonl')
            self._latency_file = open(latency_path, 'w')
            
            # JSON lines file for events (alerts, detections, etc.)
            events_path = os.path.join(session_dir, 'events.jsonl')
            self._events_file = open(events_path, 'w')
            
            # CSV file for easy analysis
            csv_path = os.path.join(session_dir, 'latency.csv')
            csv_file = open(csv_path, 'w', newline='')
            self._csv_writer = csv.DictWriter(csv_file, fieldnames=[
                'frame_id', 'timestamp', 'camera_ms', 'detection_ms',
                'safety_ms', 'scene_ms', 'audio_ms', 'llm_ms', 'total_ms',
                'detection_count', 'alert_count'
            ])
            self._csv_writer.writeheader()
            self._csv_file = csv_file
            
        except Exception as e:
            print(f"[Telemetry] ERROR opening log files: {e}")
            return False
        
        # Start async write thread
        self._running = True
        self._write_thread = threading.Thread(
            target=self._write_loop,
            name="TelemetryWriteThread",
            daemon=True
        )
        self._write_thread.start()
        
        print(f"[Telemetry] Started logging to {session_dir}")
        return True
    
    def stop(self) -> SessionStats:
        """Stop telemetry logging and return session statistics."""
        self._running = False
        
        # Wait for write queue to drain
        if self._write_thread is not None:
            self._write_thread.join(timeout=5.0)
        
        # Calculate final statistics
        stats = self._calculate_stats()
        
        # Close session
        self._session.close_session(stats)
        
        # Close file handles
        if self._latency_file:
            self._latency_file.close()
        if self._events_file:
            self._events_file.close()
        if hasattr(self, '_csv_file'):
            self._csv_file.close()
        
        return stats
    
    def _write_loop(self) -> None:
        """Background thread for async file writes."""
        while self._running or not self._write_queue.empty():
            try:
                item = self._write_queue.get(timeout=0.1)
                
                file_type = item.get('type')
                data = item.get('data')
                
                if file_type == 'latency' and self._latency_file:
                    self._latency_file.write(json.dumps(data) + '\n')
                    self._latency_file.flush()
                    
                    # Also write to CSV
                    if self._csv_writer:
                        self._csv_writer.writerow(data)
                        self._csv_file.flush()
                
                elif file_type == 'event' and self._events_file:
                    self._events_file.write(json.dumps(data) + '\n')
                    self._events_file.flush()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Telemetry] Write error: {e}")
    
    def log_latency(self, metrics: LatencyMetrics) -> None:
        """Log latency metrics for a frame."""
        if not self._enabled or not self._log_latency:
            return
        
        metrics.calculate_total()
        
        # Update statistics
        self._total_frames += 1
        self._latency_history.append(metrics.total_ms)
        
        self._component_history['camera'].append(metrics.camera_ms)
        self._component_history['detection'].append(metrics.detection_ms)
        self._component_history['safety'].append(metrics.safety_ms)
        self._component_history['scene'].append(metrics.scene_ms)
        self._component_history['audio'].append(metrics.audio_ms)
        if metrics.llm_ms > 0:
            self._component_history['llm'].append(metrics.llm_ms)
        
        self._total_detections += metrics.detection_count
        self._total_alerts += metrics.alert_count
        
        # Check for latency spike
        if metrics.total_ms > self._warning_threshold_ms:
            self._latency_spikes += 1
            if self._log_level == "DEBUG":
                print(f"[Telemetry] ⚠️ Latency spike: {metrics.total_ms:.0f}ms")
        
        # Queue for async write
        try:
            self._write_queue.put_nowait({
                'type': 'latency',
                'data': metrics.to_dict()
            })
        except queue.Full:
            pass  # Drop if queue full
    
    def log_detection(self, frame_id: int, detections: List[dict]) -> None:
        """Log detection results."""
        if not self._enabled or not self._log_detections:
            return
        
        event = {
            'type': 'detection',
            'timestamp': time.time(),
            'frame_id': frame_id,
            'count': len(detections),
            'objects': [d.get('class_name', 'unknown') for d in detections[:10]]
        }
        
        try:
            self._write_queue.put_nowait({'type': 'event', 'data': event})
        except queue.Full:
            pass
    
    def log_alert(self, alert: Any) -> None:
        """Log a safety alert."""
        if not self._enabled or not self._log_alerts:
            return
        
        # Handle both dict and Alert objects
        if hasattr(alert, 'to_dict'):
            alert_data = alert.to_dict()
        elif isinstance(alert, dict):
            alert_data = alert
        else:
            alert_data = {
                'class_name': getattr(alert, 'class_name', 'unknown'),
                'danger_level': getattr(alert, 'danger_level', 'unknown'),
                'zone': getattr(alert, 'zone', 'unknown'),
                'distance_m': getattr(alert, 'distance_m', 0)
            }
        
        # Track by level
        level = alert_data.get('danger_level', 'info')
        self._alerts_by_level[level] = self._alerts_by_level.get(level, 0) + 1
        
        event = {
            'type': 'alert',
            'timestamp': time.time(),
            **alert_data
        }
        
        try:
            self._write_queue.put_nowait({'type': 'event', 'data': event})
        except queue.Full:
            pass
    
    def log_llm_response(self, query: str, response: str, 
                         latency_ms: float, source: str = 'llm') -> None:
        """Log LLM query and response."""
        if not self._enabled or not self._log_llm:
            return
        
        event = {
            'type': 'llm_response',
            'timestamp': time.time(),
            'query': query[:200],  # Truncate long queries
            'response': response[:500],  # Truncate long responses
            'latency_ms': latency_ms,
            'source': source
        }
        
        try:
            self._write_queue.put_nowait({'type': 'event', 'data': event})
        except queue.Full:
            pass
    
    def log_error(self, error_type: str, message: str, 
                  traceback_str: str = None) -> None:
        """Log an error event."""
        self._error_count += 1
        
        event = {
            'type': 'error',
            'timestamp': time.time(),
            'error_type': error_type,
            'message': message,
            'traceback': traceback_str
        }
        
        try:
            self._write_queue.put_nowait({'type': 'event', 'data': event})
        except queue.Full:
            pass
        
        print(f"[Telemetry] ERROR logged: {error_type}: {message}")
    
    def record_memory(self, memory_mb: float) -> None:
        """Record current memory usage."""
        self._memory_samples.append(memory_mb)
        self._peak_memory_mb = max(self._peak_memory_mb, memory_mb)
    
    def _calculate_stats(self) -> SessionStats:
        """Calculate session statistics from collected data."""
        stats = SessionStats(
            session_id=self._session.get_current_session() or "unknown",
            start_time=self._session._start_time
        )
        
        stats.total_frames = self._total_frames
        stats.total_detections = self._total_detections
        stats.total_alerts = self._total_alerts
        stats.alerts_by_level = self._alerts_by_level.copy()
        stats.latency_spikes = self._latency_spikes
        stats.errors = self._error_count
        
        # Calculate latency statistics
        if self._latency_history:
            latencies = list(self._latency_history)
            stats.avg_latency_ms = statistics.mean(latencies)
            stats.min_latency_ms = min(latencies)
            stats.max_latency_ms = max(latencies)
            
            sorted_latencies = sorted(latencies)
            n = len(sorted_latencies)
            stats.p50_latency_ms = sorted_latencies[int(n * 0.50)]
            stats.p95_latency_ms = sorted_latencies[int(n * 0.95)]
            stats.p99_latency_ms = sorted_latencies[int(n * 0.99)]
        
        # Component averages
        for name, history in self._component_history.items():
            if history:
                avg = statistics.mean(history)
                setattr(stats, f'avg_{name}_ms', avg)
        
        # Memory stats
        if self._memory_samples:
            stats.peak_memory_mb = self._peak_memory_mb
            stats.avg_memory_mb = statistics.mean(self._memory_samples)
        
        return stats
    
    def get_stats(self) -> SessionStats:
        """Get current session statistics."""
        return self._calculate_stats()
    
    def get_current_fps(self) -> float:
        """Get current effective FPS based on latency."""
        if not self._latency_history:
            return 0.0
        
        # Use recent latencies
        recent = list(self._latency_history)[-30:]
        avg_ms = statistics.mean(recent) if recent else 1000
        return 1000.0 / max(avg_ms, 1)
    
    def detect_anomalies(self) -> List[str]:
        """Detect performance anomalies."""
        anomalies = []
        
        if not self._latency_history:
            return anomalies
        
        avg = statistics.mean(self._latency_history)
        
        # Check for consistent high latency
        if avg > self._warning_threshold_ms:
            anomalies.append(f"High average latency: {avg:.0f}ms")
        
        # Check for frequent spikes
        spike_rate = self._latency_spikes / max(1, self._total_frames)
        if spike_rate > 0.1:  # More than 10% spikes
            anomalies.append(f"Frequent latency spikes: {spike_rate:.1%}")
        
        # Check for errors
        if self._error_count > 0:
            anomalies.append(f"Errors recorded: {self._error_count}")
        
        # Check memory trend (if enough samples)
        if len(self._memory_samples) > 10:
            first_half = list(self._memory_samples)[:len(self._memory_samples)//2]
            second_half = list(self._memory_samples)[len(self._memory_samples)//2:]
            
            if statistics.mean(second_half) > statistics.mean(first_half) * 1.5:
                anomalies.append("Possible memory leak detected")
        
        return anomalies
    
    def export_summary(self, output_path: str = None) -> str:
        """Export session summary to JSON file."""
        stats = self.get_stats()
        
        if output_path is None:
            session_dir = self._session.get_session_dir()
            if session_dir:
                output_path = os.path.join(session_dir, 'summary.json')
            else:
                output_path = 'telemetry_summary.json'
        
        summary = {
            'session': stats.to_dict(),
            'anomalies': self.detect_anomalies(),
            'export_time': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"[Telemetry] Summary exported to {output_path}")
        return output_path
    
    @property
    def is_running(self) -> bool:
        """Check if telemetry is active."""
        return self._running


# ============================================
# Standalone Test
# ============================================
if __name__ == "__main__":
    """Test telemetry module standalone."""
    import random
    
    print("=" * 50)
    print("Telemetry Module Test")
    print("=" * 50)
    
    logger = TelemetryLogger(config_path="config/settings.yaml")
    
    if not logger.start():
        print("Failed to start telemetry")
        exit(1)
    
    # Simulate 100 frames
    print("\nSimulating 100 frames...")
    for i in range(100):
        metrics = LatencyMetrics(
            frame_id=i + 1,
            timestamp=time.time(),
            camera_ms=random.uniform(5, 15),
            detection_ms=random.uniform(30, 80),
            safety_ms=random.uniform(2, 8),
            scene_ms=random.uniform(5, 15),
            audio_ms=random.uniform(1, 5),
            detection_count=random.randint(0, 5),
            alert_count=random.randint(0, 2)
        )
        
        # Occasional spike
        if random.random() < 0.05:
            metrics.detection_ms = random.uniform(200, 500)
        
        logger.log_latency(metrics)
        
        # Simulate occasional alert
        if random.random() < 0.1:
            logger.log_alert({
                'class_name': 'person',
                'danger_level': random.choice(['critical', 'warning', 'caution']),
                'zone': random.choice(['left', 'center', 'right']),
                'distance_m': random.uniform(1, 4)
            })
        
        time.sleep(0.01)  # Simulate some processing time
    
    # Check for anomalies
    anomalies = logger.detect_anomalies()
    if anomalies:
        print(f"\nAnomalies detected:")
        for a in anomalies:
            print(f"  ⚠️ {a}")
    
    # Export summary
    logger.export_summary()
    
    # Get final stats
    stats = logger.stop()
    
    print(f"\n" + "=" * 50)
    print("Session Statistics:")
    print("=" * 50)
    print(f"Total frames: {stats.total_frames}")
    print(f"Average latency: {stats.avg_latency_ms:.1f}ms")
    print(f"P95 latency: {stats.p95_latency_ms:.1f}ms")
    print(f"P99 latency: {stats.p99_latency_ms:.1f}ms")
    print(f"Latency spikes: {stats.latency_spikes}")
    print(f"Total alerts: {stats.total_alerts}")
    print(f"Alerts by level: {stats.alerts_by_level}")
    
    print("\nTest complete!")
