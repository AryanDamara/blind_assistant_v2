#!/usr/bin/env python3
"""
Utility Functions
-----------------
Shared helpers used across all navigation system modules.

Features:
- Configuration loading with defaults
- FPS calculator with rolling window
- Timer context manager for profiling
- Frame resizing with aspect ratio preservation
- System info collection
- Logging formatter with colors
- Distance/math helpers
"""

import os
import sys
import time
import math
import logging
import platform
from contextlib import contextmanager
from collections import deque
from typing import Dict, Optional, Tuple, Any

import yaml
import numpy as np


# ============================================
# CONFIGURATION
# ============================================

def load_config(config_path: str = "config/settings.yaml") -> Dict:
    """
    Load YAML configuration with fallback defaults.
    
    Args:
        config_path: Path to settings.yaml
        
    Returns:
        Configuration dictionary
    """
    defaults = {
        'camera': {
            'source': 0,
            'width': 640,
            'height': 480,
            'fps': 30,
            'buffer_size': 2,
            'frame_skip': 3,
            'auto_restart': True,
            'restart_delay_sec': 2.0,
            'warmup_frames': 10,
            'loop_video': True,
        },
        'yolo': {
            'model_path': 'models/yolo/yolov8n.pt',
            'confidence_threshold': 0.5,
            'iou_threshold': 0.45,
            'device': 'cpu',
            'max_detections': 10,
            'img_size': 640,
            'allowed_classes': [],
        },
        'safety': {
            'ignore_classes': [],
            'deduplication_window_sec': 3.0,
            'center_zone_multiplier': 0.8,
        },
        'audio': {
            'enabled': True,
            'engine': 'pyttsx3',
            'verbosity': 'standard',
            'interrupt_on_critical': True,
            'max_alerts_per_cycle': 3,
            'voice_index': 0,
        },
        'voice_input': {
            'enabled': True,
            'activation_mode': 'push_to_talk',
            'activation_key': 'space',
            'timeout_sec': 5.0,
            'phrase_time_limit': 10.0,
        },
        'llm': {
            'enabled': True,
            'model': 'llama3.2:3b',
            'temperature': 0.7,
            'max_tokens': 100,
            'timeout_sec': 2.0,
        },
        'telemetry': {
            'enabled': True,
            'log_directory': 'data/logs',
            'log_level': 'INFO',
        },
        'debug': {
            'show_video': True,
            'show_bboxes': True,
            'show_zones': True,
            'show_fps': True,
            'print_latency': True,
            'print_detections': True,
        },
    }
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
            
            if user_config:
                # Deep merge user config over defaults
                config = _deep_merge(defaults, user_config)
                return config
        except Exception as e:
            print(f"[Utils] WARNING: Failed to load config: {e}, using defaults")
    else:
        print(f"[Utils] WARNING: Config not found at {config_path}, using defaults")
    
    return defaults


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Deep merge two dictionaries. Override values take precedence.
    
    Args:
        base: Default values
        override: User values (take precedence)
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


# ============================================
# FPS CALCULATOR
# ============================================

class FPSCalculator:
    """
    Rolling FPS calculator with statistics.
    
    Usage:
        fps_calc = FPSCalculator(window_size=30)
        
        while running:
            fps_calc.tick()
            print(f"FPS: {fps_calc.fps:.1f}")
    """
    
    def __init__(self, window_size: int = 30):
        """
        Args:
            window_size: Number of frames to average over
        """
        self._window_size = window_size
        self._timestamps: deque = deque(maxlen=window_size + 1)
        self._frame_count: int = 0
        self._start_time: float = time.time()
    
    def tick(self) -> None:
        """Record a frame timestamp."""
        self._timestamps.append(time.time())
        self._frame_count += 1
    
    @property
    def fps(self) -> float:
        """Current FPS (rolling window average)."""
        if len(self._timestamps) < 2:
            return 0.0
        
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        
        return (len(self._timestamps) - 1) / elapsed
    
    @property
    def avg_fps(self) -> float:
        """Overall average FPS since start."""
        elapsed = time.time() - self._start_time
        if elapsed <= 0:
            return 0.0
        return self._frame_count / elapsed
    
    @property
    def frame_time_ms(self) -> float:
        """Average time per frame in ms."""
        current_fps = self.fps
        if current_fps <= 0:
            return 0.0
        return 1000.0 / current_fps
    
    def reset(self) -> None:
        """Reset all counters."""
        self._timestamps.clear()
        self._frame_count = 0
        self._start_time = time.time()


# ============================================
# TIMER / PROFILER
# ============================================

@contextmanager
def timer(label: str = "", print_result: bool = True):
    """
    Context manager for timing code blocks.
    
    Usage:
        with timer("Detection"):
            result = detector.detect(frame)
        # Prints: [Timer] Detection: 45.2ms
        
        with timer("Processing", print_result=False) as t:
            do_work()
        elapsed_ms = t.elapsed_ms
    """
    class TimerResult:
        def __init__(self):
            self.start_time = time.time()
            self.end_time = None
            self.elapsed_ms = 0.0
        
        def stop(self):
            self.end_time = time.time()
            self.elapsed_ms = (self.end_time - self.start_time) * 1000
    
    result = TimerResult()
    try:
        yield result
    finally:
        result.stop()
        if print_result and label:
            print(f"[Timer] {label}: {result.elapsed_ms:.1f}ms")


class LatencyTracker:
    """
    Track latency statistics over time.
    
    Usage:
        tracker = LatencyTracker(name="detection")
        
        tracker.record(45.2)
        tracker.record(52.1)
        
        print(tracker.summary())
    """
    
    def __init__(self, name: str = "", max_samples: int = 1000):
        self.name = name
        self._samples: deque = deque(maxlen=max_samples)
    
    def record(self, latency_ms: float) -> None:
        """Record a latency measurement."""
        self._samples.append(latency_ms)
    
    @property
    def count(self) -> int:
        return len(self._samples)
    
    @property
    def mean(self) -> float:
        if not self._samples:
            return 0.0
        return sum(self._samples) / len(self._samples)
    
    @property
    def min(self) -> float:
        return min(self._samples) if self._samples else 0.0
    
    @property
    def max(self) -> float:
        return max(self._samples) if self._samples else 0.0
    
    @property
    def std(self) -> float:
        if len(self._samples) < 2:
            return 0.0
        mean = self.mean
        variance = sum((x - mean) ** 2 for x in self._samples) / len(self._samples)
        return math.sqrt(variance)
    
    def percentile(self, p: float) -> float:
        """Get p-th percentile (0-100)."""
        if not self._samples:
            return 0.0
        sorted_samples = sorted(self._samples)
        idx = int(len(sorted_samples) * p / 100)
        idx = min(idx, len(sorted_samples) - 1)
        return sorted_samples[idx]
    
    def summary(self) -> Dict[str, float]:
        """Get summary statistics."""
        return {
            'name': self.name,
            'count': self.count,
            'mean_ms': round(self.mean, 1),
            'min_ms': round(self.min, 1),
            'max_ms': round(self.max, 1),
            'std_ms': round(self.std, 1),
            'p50_ms': round(self.percentile(50), 1),
            'p95_ms': round(self.percentile(95), 1),
            'p99_ms': round(self.percentile(99), 1),
        }
    
    def reset(self) -> None:
        """Clear all samples."""
        self._samples.clear()


# ============================================
# FRAME UTILITIES
# ============================================

def resize_frame(frame: np.ndarray, max_width: int = 640,
                 max_height: int = 480) -> np.ndarray:
    """
    Resize frame while preserving aspect ratio.
    
    Args:
        frame: Input frame (numpy array)
        max_width: Maximum width
        max_height: Maximum height
        
    Returns:
        Resized frame
    """
    import cv2
    
    h, w = frame.shape[:2]
    
    if w <= max_width and h <= max_height:
        return frame
    
    # Calculate scale factor
    scale = min(max_width / w, max_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def validate_frame(frame: np.ndarray) -> bool:
    """
    Validate that a frame is usable for detection.
    
    Args:
        frame: Input frame
        
    Returns:
        True if frame is valid
    """
    if frame is None:
        return False
    
    if not isinstance(frame, np.ndarray):
        return False
    
    if frame.ndim < 2:
        return False
    
    if frame.shape[0] < 10 or frame.shape[1] < 10:
        return False
    
    return True


# ============================================
# DISTANCE HELPERS
# ============================================

def estimate_distance_bbox(bbox_height_px: float,
                           focal_length: float,
                           real_height_m: float) -> float:
    """
    Estimate distance using bounding box height and pinhole camera model.
    
    Formula: distance = (focal_length × real_height) / bbox_height
    
    Args:
        bbox_height_px: Height of bounding box in pixels
        focal_length: Camera focal length (calibrated)
        real_height_m: Real-world height of object in meters
        
    Returns:
        Estimated distance in meters
    """
    if bbox_height_px <= 0:
        return float('inf')
    
    return (focal_length * real_height_m) / bbox_height_px


def calculate_focal_length(bbox_height_px: float,
                           known_distance_m: float,
                           real_height_m: float) -> float:
    """
    Calculate focal length from a known measurement.
    
    Formula: focal_length = (bbox_height × distance) / real_height
    
    Args:
        bbox_height_px: Height of bounding box at known distance
        known_distance_m: Known distance to object
        real_height_m: Real-world height of object
        
    Returns:
        Calculated focal length in pixels
    """
    if real_height_m <= 0:
        return 0.0
    
    return (bbox_height_px * known_distance_m) / real_height_m


def bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """
    Calculate the center point of a bounding box.
    
    Args:
        bbox: (x1, y1, x2, y2) bounding box
        
    Returns:
        (cx, cy) center coordinates
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def bbox_area(bbox: Tuple[int, int, int, int]) -> int:
    """Calculate area of a bounding box."""
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_iou(bbox1: Tuple[int, int, int, int],
             bbox2: Tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union between two bounding boxes.
    
    Args:
        bbox1, bbox2: (x1, y1, x2, y2) bounding boxes
        
    Returns:
        IoU value [0.0, 1.0]
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = bbox_area(bbox1)
    area2 = bbox_area(bbox2)
    
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union


def euclidean_distance(p1: Tuple[float, float],
                       p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two 2D points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# ============================================
# SYSTEM INFO
# ============================================

def get_system_info() -> Dict[str, Any]:
    """
    Collect system information for diagnostics.
    
    Returns:
        Dictionary with system details
    """
    info = {
        'platform': platform.system(),
        'platform_version': platform.version(),
        'python_version': platform.python_version(),
        'processor': platform.processor(),
        'machine': platform.machine(),
    }
    
    # Check GPU
    try:
        import torch
        info['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info['gpu_name'] = torch.cuda.get_device_name(0)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            info['mps_available'] = True
    except ImportError:
        info['cuda_available'] = False
    
    # Memory
    try:
        import psutil
        mem = psutil.virtual_memory()
        info['total_memory_gb'] = round(mem.total / (1024**3), 1)
        info['available_memory_gb'] = round(mem.available / (1024**3), 1)
    except ImportError:
        pass
    
    return info


# ============================================
# LOGGING
# ============================================

class ColorFormatter(logging.Formatter):
    """Colored log formatter for terminal output."""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[41m',   # Red background
    }
    RESET = '\033[0m'
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logger(name: str, level: str = "INFO",
                 log_file: str = None) -> logging.Logger:
    """
    Create a configured logger with colored console output.
    
    Args:
        name: Logger name
        level: Log level string
        log_file: Optional file path for file logging
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Console handler with colors
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            ColorFormatter('[%(levelname)s] %(name)s: %(message)s')
        )
        logger.addHandler(console_handler)
    
    # Optional file handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        )
        logger.addHandler(file_handler)
    
    return logger


# ============================================
# MISC HELPERS
# ============================================

def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max."""
    return max(min_val, min(max_val, value))


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def safe_divide(numerator: float, denominator: float,
                default: float = 0.0) -> float:
    """Safe division with default for zero denominator."""
    if denominator == 0:
        return default
    return numerator / denominator
