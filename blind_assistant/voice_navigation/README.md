# Voice Navigation Module

Core navigation engine for the Blind Assistant system.

## Directory Structure

```
voice_navigation/
├── config/
│   ├── settings.yaml              # Main configuration
│   ├── settings.example.yaml      # Configuration reference
│   └── prompts/
│       ├── navigation.yaml        # LLM navigation prompt templates
│       └── safety.yaml            # LLM safety prompt templates
├── src/
│   ├── __init__.py                # Package exports
│   ├── main.py                    # System orchestrator
│   ├── camera_capture.py          # Video capture (threaded)
│   ├── object_detector.py         # YOLOv8 detection
│   ├── safety_manager.py          # Danger analysis & alerts
│   ├── scene_analyzer.py          # Object tracking & grouping
│   ├── audio_feedback.py          # TTS with priority queue
│   ├── conversation_handler.py    # Voice input & intent detection
│   ├── ai_assistant.py            # Ollama LLM integration
│   ├── telemetry.py               # Metrics & session logging
│   ├── calibration_tool.py        # Distance calibration wizard
│   └── utils.py                   # Shared utilities
├── models/
│   └── yolo/
│       └── yolov8n.pt             # YOLOv8 nano model
├── tests/
│   ├── test_integration.py        # Integration tests (pytest)
│   ├── benchmark.py               # Performance benchmarks
│   └── user_testing_protocol.md   # User acceptance testing guide
├── tools/
│   └── system_check.py            # Pre-flight system validation
├── docs/
│   ├── user_guide.md              # End-user documentation
│   ├── calibration_guide.md       # Calibration instructions
│   └── privacy_policy.md          # Privacy policy
├── data/
│   ├── logs/                      # Telemetry output (auto-created)
│   └── test_scenarios/            # Test video files
└── requirements.txt               # Python dependencies
```

## Pipeline

```
┌──────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────────┐
│  Camera   │───▶│  YOLOv8      │───▶│  Safety       │───▶│  Audio       │
│  Capture  │    │  Detector    │    │  Manager      │    │  Feedback    │
└──────────┘    └──────────────┘    └───────────────┘    └──────────────┘
                       │                    │
                       ▼                    ▼
                ┌──────────────┐    ┌───────────────┐
                │  Scene       │    │  Telemetry    │
                │  Analyzer    │    │  Logger       │
                └──────────────┘    └───────────────┘
                       │
                       ▼
                ┌──────────────┐    ┌───────────────┐
                │  AI          │◀───│  Conversation │◀── Voice Input
                │  Assistant   │    │  Handler      │
                └──────────────┘    └───────────────┘
```

## Configuration Reference

See `config/settings.yaml` for all options. Key sections:

| Section | Controls |
|---------|----------|
| `camera` | Source, resolution, frame skip, auto-restart |
| `yolo` | Model path, confidence, device (cpu/cuda) |
| `safety` | Danger thresholds, zones, ignore list, deduplication |
| `distance` | Calibration data per object class |
| `scene` | Tracking history, movement threshold, grouping |
| `llm` | Ollama model, temperature, caching |
| `audio` | Voice profiles, verbosity, interrupt mode |
| `voice_input` | Push-to-talk key, timeout, mic settings |
| `telemetry` | Log directory, enabled metrics |
| `debug` | Visual overlays, console output |
| `system` | Shutdown timeout, error recovery |

## Running

```bash
# Activate virtual environment
source venv/bin/activate

# Run navigation system
python src/main.py

# Run with specific config
python src/main.py --config config/settings.yaml
```

## Quick Start

1. **Install dependencies**:
   ```bash
   cd voice_navigation
   python -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Check system readiness**:
   ```bash
   python tools/system_check.py
   ```

3. **Run with defaults**:
   ```bash
   python src/main.py
   ```
   Press `q` to quit. Press `Space` for push-to-talk voice queries.

4. **Demo mode** (no camera, uses test video):
   Edit `config/settings.yaml`:
   ```yaml
   debug:
     mock_camera: true
     mock_video_path: "data/test_scenarios/crowded_hallway.mp4"
   ```

See [docs/QUICKSTART.md](docs/QUICKSTART.md) for a detailed beginner walkthrough.

## Testing

```bash
source venv/bin/activate
pip install pytest  # if not installed

# Unit tests
python -m pytest tests/test_safety_manager.py -v
python -m pytest tests/test_audio_feedback.py -v

# Integration tests
python -m pytest tests/test_integration.py -v

# All tests
python -m pytest tests/ -v

# Performance benchmark
python tests/benchmark.py
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `No camera found` | Check camera index in `settings.yaml` → `camera.source` |
| `pyttsx3 not working` | macOS: `pip install pyobjc-framework-Cocoa`. Linux: `sudo apt install espeak` |
| `YOLO model not found` | Run `python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"` to download |
| `Ollama not running` | Start Ollama: `ollama serve`, then `ollama pull llama3.2:3b` |
| `High latency (>1s)` | Increase `camera.frame_skip` or set `yolo.device: cuda` if GPU available |
| `No audio output` | Check `audio.enabled: true` and test: `python -c "import pyttsx3; e=pyttsx3.init(); e.say('test'); e.runAndWait()"` |
| `Speech recognition fails` | Set `voice_input.energy_threshold` lower (e.g., 2000) for quiet environments |

