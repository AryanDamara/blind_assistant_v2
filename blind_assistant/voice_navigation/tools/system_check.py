#!/usr/bin/env python3
"""
System Check Tool
-----------------
Pre-flight validation for the navigation system.

Checks:
- Python version and dependencies
- Camera availability
- Microphone access
- Audio output
- YOLO model files
- Ollama availability
- Configuration validity

Usage:
    python tools/system_check.py
"""

import os
import sys
import time
import importlib

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class SystemChecker:
    """Pre-flight system validation."""
    
    def __init__(self):
        self.results = {}
        self.errors = []
        self.warnings = []
    
    def check_all(self) -> bool:
        """Run all system checks."""
        print("\n" + "=" * 60)
        print("NAVIGATION SYSTEM PRE-FLIGHT CHECK")
        print("=" * 60)
        
        checks = [
            ("Python Version", self.check_python),
            ("Core Dependencies", self.check_dependencies),
            ("PyAudio", self.check_pyaudio),
            ("Configuration File", self.check_config),
            ("YOLO Model", self.check_yolo),
            ("Camera Indices", self.check_camera_indices),
            ("Microphone", self.check_microphone),
            ("Audio Output", self.check_audio),
            ("Ollama (Optional)", self.check_ollama),
            ("GPU (Optional)", self.check_gpu),
            ("Disk Space", self.check_disk),
        ]
        
        for name, check_func in checks:
            print(f"\n[{name}]")
            try:
                result = check_func()
                self.results[name] = result
            except Exception as e:
                self.results[name] = False
                self.errors.append(f"{name}: {e}")
                print(f"  ✗ ERROR: {e}")
        
        return self._print_summary()
    
    def check_python(self) -> bool:
        """Check Python version."""
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"
        
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print(f"  ✗ Python {version_str} is too old (need 3.8+)")
            self.errors.append("Python version too old")
            return False
        
        print(f"  ✓ Python {version_str}")
        return True
    
    def check_dependencies(self) -> bool:
        """Check required packages."""
        required = {
            'cv2': 'opencv-python',
            'numpy': 'numpy',
            'yaml': 'pyyaml',
            'ultralytics': 'ultralytics',
            'pyttsx3': 'pyttsx3',
            'speech_recognition': 'SpeechRecognition',
        }
        
        optional = {
            'pynput': 'pynput',
            'requests': 'requests',
            'psutil': 'psutil',
        }
        
        all_ok = True
        
        for module, package in required.items():
            try:
                importlib.import_module(module)
                print(f"  ✓ {package}")
            except ImportError:
                print(f"  ✗ {package} NOT FOUND - pip install {package}")
                self.errors.append(f"Missing package: {package}")
                all_ok = False
        
        for module, package in optional.items():
            try:
                importlib.import_module(module)
                print(f"  ✓ {package} (optional)")
            except ImportError:
                print(f"  ⚠ {package} not installed (optional)")
                self.warnings.append(f"Optional package missing: {package}")
        
        return all_ok
    
    def check_config(self) -> bool:
        """Check configuration file exists and is valid."""
        config_path = "config/settings.yaml"
        
        if not os.path.exists(config_path):
            print(f"  ✗ Config file not found: {config_path}")
            self.errors.append("Config file missing")
            return False
        
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check required sections
            required_sections = ['camera', 'yolo', 'safety', 'audio']
            missing = [s for s in required_sections if s not in config]
            
            if missing:
                print(f"  ⚠ Missing config sections: {missing}")
                self.warnings.append(f"Config sections missing: {missing}")
            
            print(f"  ✓ Configuration valid")
            return True
            
        except Exception as e:
            print(f"  ✗ Config parse error: {e}")
            self.errors.append(f"Config error: {e}")
            return False
    
    def check_yolo(self) -> bool:
        """Check YOLO model availability."""
        model_paths = [
            "models/yolov8n.pt",
            os.path.expanduser("~/.cache/ultralytics/yolov8n.pt"),
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                print(f"  ✓ YOLO model found: {path} ({size_mb:.1f}MB)")
                return True
        
        # Try to load via ultralytics (will download if needed)
        try:
            from ultralytics import YOLO
            print("  ⚠ YOLO model not cached - will download on first run")
            self.warnings.append("YOLO model will be downloaded on first run")
            return True
        except Exception as e:
            print(f"  ✗ YOLO not available: {e}")
            self.errors.append("YOLO model unavailable")
            return False
    
    def check_pyaudio(self) -> bool:
        """Check PyAudio installation (common issue)."""
        try:
            import pyaudio
            p = pyaudio.PyAudio()
            device_count = p.get_device_count()
            p.terminate()
            
            print(f"  ✓ PyAudio working ({device_count} audio devices)")
            return True
        except ImportError:
            print("  ⚠ PyAudio not installed")
            print("     macOS: brew install portaudio && pip install pyaudio")
            print("     Linux: sudo apt-get install portaudio19-dev python3-pyaudio")
            self.warnings.append("PyAudio not installed - microphone may not work")
            return True  # Not critical
        except Exception as e:
            print(f"  ⚠ PyAudio error: {e}")
            self.warnings.append(f"PyAudio: {e}")
            return True
    
    def check_camera_indices(self) -> bool:
        """Check all available camera indices, not just 0."""
        try:
            import cv2
            available_cameras = []
            
            for i in range(5):  # Check first 5 indices
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        available_cameras.append(i)
                    cap.release()
            
            if not available_cameras:
                print("  ✗ No working cameras found (checked indices 0-4)")
                self.errors.append("No cameras available")
                return False
            
            print(f"  ✓ Available cameras: {available_cameras}")
            if 0 not in available_cameras:
                print(f"  ⚠ Camera 0 not available - update config to use: {available_cameras[0]}")
                self.warnings.append(f"Update camera.source to {available_cameras[0]} in config")
            
            return True
            
        except Exception as e:
            print(f"  ✗ Camera check failed: {e}")
            self.errors.append(f"Camera: {e}")
            return False
    
    def check_gpu(self) -> bool:
        """Check for GPU availability (optional but improves performance)."""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                print(f"  ✓ GPU available: {gpu_name}")
                print(f"    Consider using device='cuda' in YOLO config")
                return True
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                print(f"  ✓ Apple MPS available (Metal acceleration)")
                return True
            else:
                print("  ⚠ No GPU detected (CPU mode - inference will be slower)")
                self.warnings.append("No GPU - using CPU for inference")
                return True
        except ImportError:
            print("  ⚠ PyTorch not installed (GPU check skipped)")
            return True
        except Exception as e:
            print(f"  ⚠ GPU check failed: {e}")
            return True
    
    def check_camera(self) -> bool:
        """Check camera availability (legacy, replaced by check_camera_indices)."""
        return self.check_camera_indices()
    
    def check_microphone(self) -> bool:
        """Check microphone availability."""
        try:
            import speech_recognition as sr
            
            recognizer = sr.Recognizer()
            
            # List available microphones
            mics = sr.Microphone.list_microphone_names()
            if not mics:
                print("  ⚠ No microphones detected")
                self.warnings.append("No microphones found")
                return True  # Not critical
            
            print(f"  ✓ Microphones available: {len(mics)}")
            print(f"    Default: {mics[0][:40]}...")
            
            return True
            
        except Exception as e:
            print(f"  ⚠ Microphone check failed: {e}")
            self.warnings.append(f"Microphone: {e}")
            return True  # Not critical
    
    def check_audio(self) -> bool:
        """Check audio output."""
        try:
            import pyttsx3
            
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            
            if voices:
                print(f"  ✓ TTS engine initialized ({len(voices)} voices)")
            else:
                print("  ⚠ TTS engine has no voices")
                self.warnings.append("No TTS voices found")
            
            # Don't actually speak during check
            engine.stop()
            return True
            
        except Exception as e:
            print(f"  ⚠ Audio warning: {e}")
            self.warnings.append(f"Audio: {e}")
            return True  # Not critical for basic operation
    
    def check_ollama(self) -> bool:
        """Check Ollama availability (optional)."""
        try:
            import requests
            
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            
            if response.status_code == 200:
                models = response.json().get('models', [])
                if models:
                    print(f"  ✓ Ollama running with {len(models)} model(s)")
                    for m in models[:3]:
                        print(f"    - {m.get('name', 'unknown')}")
                else:
                    print("  ⚠ Ollama running but no models installed")
                    self.warnings.append("Ollama has no models")
                return True
            
        except requests.ConnectionError:
            print("  ⚠ Ollama not running (optional for AI responses)")
            self.warnings.append("Ollama not available")
        except Exception as e:
            print(f"  ⚠ Ollama check failed: {e}")
            self.warnings.append(f"Ollama: {e}")
        
        return True  # Optional component
    
    def check_disk(self) -> bool:
        """Check available disk space."""
        try:
            import shutil
            
            total, used, free = shutil.disk_usage(".")
            free_gb = free / (1024**3)
            
            if free_gb < 1:
                print(f"  ⚠ Low disk space: {free_gb:.1f}GB free")
                self.warnings.append("Low disk space")
            else:
                print(f"  ✓ Disk space: {free_gb:.1f}GB free")
            
            return True
            
        except Exception as e:
            print(f"  ⚠ Disk check failed: {e}")
            return True
    
    def _print_summary(self) -> bool:
        """Print summary and return overall status."""
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for v in self.results.values() if v)
        total = len(self.results)
        
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for e in self.errors:
                print(f"   • {e}")
        
        if self.warnings:
            print(f"\n⚠ WARNINGS ({len(self.warnings)}):")
            for w in self.warnings:
                print(f"   • {w}")
        
        all_passed = len(self.errors) == 0
        
        if all_passed:
            print(f"\n✅ All {total} checks passed!")
            print("System is ready to run: python src/main.py")
        else:
            print(f"\n❌ {total - passed} of {total} checks failed")
            print("Please fix errors before running the system")
        
        print("=" * 60 + "\n")
        
        return all_passed


def main():
    """Run system check."""
    # Change to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    checker = SystemChecker()
    success = checker.check_all()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
