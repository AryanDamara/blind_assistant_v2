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
