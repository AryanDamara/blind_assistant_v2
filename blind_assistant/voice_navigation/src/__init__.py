"""
Voice Navigation System for Visually Impaired
==============================================

A real-time navigation assistant using computer vision, object detection,
and voice feedback to help visually impaired individuals navigate safely.

Architecture:
    Thread 1: Camera capture (background)
    Thread 2: Object detection (main loop)
    Thread 3: Audio feedback (background)
    Thread 4: Voice input (background, optional)

Pipeline:
    Camera → Detection → Safety Analysis → Scene Understanding →
    → LLM Enhancement → Audio Output

Modules:
    camera_capture      - Video capture and frame management
    object_detector     - YOLOv8 object detection
    safety_manager      - Danger assessment and alert generation
    scene_analyzer      - Object tracking and scene understanding
    audio_feedback      - Text-to-speech output with priority queue
    conversation_handler - Voice input and query processing
    ai_assistant        - LLM integration for context-aware responses
    telemetry           - Performance logging and metrics
    utils               - Shared utility functions

Requirements:
    Python: 3.8+
    OpenCV: 4.8.0+
    PyTorch: 2.0+ (for YOLOv8, auto-installed)
    Ollama: Latest (for LLM features)
"""

__version__ = "0.2.0"
__author__ = "Blind Assistant Team"
__python_requires__ = ">=3.8"
__license__ = "MIT"

# Lazy imports for faster startup
import sys
import warnings

# Version check
if sys.version_info < (3, 8):
    raise RuntimeError(
        f"Python 3.8+ required. You have {sys.version_info.major}.{sys.version_info.minor}"
    )

# Import core modules with error handling
_import_errors = []

try:
    from .camera_capture import CameraCapture
except ImportError as e:
    _import_errors.append(f"camera_capture: {e}")
    CameraCapture = None

try:
    from .object_detector import ObjectDetector
except ImportError as e:
    _import_errors.append(f"object_detector: {e}")
    ObjectDetector = None

try:
    from .safety_manager import SafetyManager
except ImportError as e:
    _import_errors.append(f"safety_manager: {e}")
    SafetyManager = None

try:
    from .scene_analyzer import SceneAnalyzer
except ImportError as e:
    _import_errors.append(f"scene_analyzer: {e}")
    SceneAnalyzer = None

try:
    from .audio_feedback import AudioFeedback
except ImportError as e:
    _import_errors.append(f"audio_feedback: {e}")
    AudioFeedback = None

try:
    from .ai_assistant import AIAssistant
except ImportError as e:
    _import_errors.append(f"ai_assistant: {e}")
    AIAssistant = None

try:
    from .conversation_handler import ConversationHandler
except ImportError as e:
    _import_errors.append(f"conversation_handler: {e}")
    ConversationHandler = None

try:
    from .telemetry import TelemetryLogger
except ImportError as e:
    _import_errors.append(f"telemetry: {e}")
    TelemetryLogger = None

try:
    from . import utils
except ImportError as e:
    _import_errors.append(f"utils: {e}")
    utils = None

# Report any import failures
if _import_errors:
    warnings.warn(
        "Some modules failed to import. System may have limited functionality:\n" +
        "\n".join(f"  - {err}" for err in _import_errors)
    )

__all__ = [
    "CameraCapture",
    "ObjectDetector",
    "SafetyManager",
    "SceneAnalyzer",
    "AudioFeedback",
    "AIAssistant",
    "TelemetryLogger",
    "ConversationHandler",
    "utils",
]


# Helpful startup check function
def check_dependencies():
    """
    Check if all required dependencies are available.

    Returns:
        dict: Status of each dependency
    """
    status = {
        'opencv': False,
        'ultralytics': False,
        'pyttsx3': False,
        'speech_recognition': False,
        'ollama': False,
        'yaml': False,
    }

    try:
        import cv2
        status['opencv'] = True
    except ImportError:
        pass

    try:
        from ultralytics import YOLO
        status['ultralytics'] = True
    except ImportError:
        pass

    try:
        import pyttsx3
        status['pyttsx3'] = True
    except ImportError:
        pass

    try:
        import speech_recognition
        status['speech_recognition'] = True
    except ImportError:
        pass

    try:
        import requests
        # Try pinging Ollama
        resp = requests.get("http://localhost:11434/api/tags", timeout=1)
        status['ollama'] = resp.status_code == 200
    except Exception:
        pass

    try:
        import yaml
        status['yaml'] = True
    except ImportError:
        pass

    return status


# Quick setup validator
def validate_setup():
    """
    Validate that the system is properly configured.

    Returns:
        tuple: (is_valid, list of issues)
    """
    import os

    issues = []
    deps = check_dependencies()

    # Critical dependencies
    if not deps['opencv']:
        issues.append("OpenCV not installed: pip install opencv-python")
    if not deps['ultralytics']:
        issues.append("Ultralytics not installed: pip install ultralytics")
    if not deps['yaml']:
        issues.append("PyYAML not installed: pip install pyyaml")

    # Optional but recommended
    if not deps['pyttsx3']:
        issues.append("WARNING: pyttsx3 not installed (no audio feedback)")
    if not deps['ollama']:
        issues.append("WARNING: Ollama not running (no LLM features)")

    # Check config file
    if not os.path.exists('config/settings.yaml'):
        issues.append("Config file missing: config/settings.yaml")

    # Check model file
    if not os.path.exists('models/yolo/yolov8n.pt'):
        issues.append("YOLO model missing (will auto-download on first run)")

    return (len([i for i in issues if not i.startswith('WARNING')]) == 0, issues)


__all__.extend(['check_dependencies', 'validate_setup'])
