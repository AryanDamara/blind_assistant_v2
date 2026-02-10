# 🦮 Blind Assistant — Voice Navigation System

A real-time AI-powered navigation assistant for visually impaired individuals. Uses computer vision, object detection, and voice feedback to provide safe navigation guidance.

---

## Features

- **Real-time Object Detection** — YOLOv8 identifies obstacles, people, vehicles, and more
- **Safety Alerts** — Priority-based danger classification (critical/warning/info) with distance estimation
- **Voice Feedback** — Natural text-to-speech with urgency-based voice profiles
- **Voice Commands** — Push-to-talk speech recognition for querying the scene
- **AI Assistant** — Ollama LLM integration for context-aware scene descriptions
- **Object Tracking** — Centroid tracking with movement detection across frames
- **Telemetry** — Performance logging, latency tracking, and session analytics
- **Calibration** — Interactive distance calibration wizard

---

## Architecture

```
Camera → YOLOv8 Detection → Safety Analysis → Audio Alerts
                                    ↓
                           Scene Analysis → AI Assistant → Voice Response
                                    ↓
                              Telemetry Logger
```

### Modules

| Module | Responsibility |
|--------|---------------|
| `camera_capture.py` | Video capture with threading and frame queue |
| `object_detector.py` | YOLOv8 model loading and inference |
| `safety_manager.py` | Zone classification, distance estimation, danger levels |
| `scene_analyzer.py` | Object tracking, movement detection, spatial grouping |
| `audio_feedback.py` | Text-to-speech with priority queue |
| `conversation_handler.py` | Push-to-talk voice input and intent detection |
| `ai_assistant.py` | LLM integration via Ollama with caching |
| `telemetry.py` | Session metrics, logging, and performance analysis |
| `calibration_tool.py` | Interactive distance calibration wizard |
| `utils.py` | Shared utilities: config, FPS, timing, math helpers |
| `main.py` | System orchestrator and main processing loop |

---

## Quick Start

### 1. Install Dependencies

```bash
cd blind_assistant/voice_navigation
python -m venv venv
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### 2. Install Ollama (Optional — for AI responses)

```bash
# macOS
brew install ollama
ollama pull llama3.2:3b
```

### 3. System Check

```bash
python tools/system_check.py
```

### 4. Run

```bash
python src/main.py
```

### 5. Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `Space` | Push-to-talk (hold to speak) |

---

## Configuration

All settings in `config/settings.yaml`. Key options:

```yaml
camera:
  source: 0              # Camera index or video file path
  frame_skip: 3          # Process every Nth frame

yolo:
  confidence_threshold: 0.5
  device: "cpu"          # or "cuda" for GPU

audio:
  verbosity: "standard"  # minimal | standard | detailed

debug:
  show_video: true       # Display camera feed with overlays
```

See `config/settings.example.yaml` for full reference.

---

## Calibration

For accurate distance estimation:

```bash
python src/calibration_tool.py
```

See [Calibration Guide](voice_navigation/docs/calibration_guide.md) for details.

---

## Testing

```bash
# Run integration tests
python -m pytest tests/test_integration.py -v

# Run benchmarks
python tests/benchmark.py --duration 30

# System pre-flight check
python tools/system_check.py
```

---

## Documentation

- [User Guide](voice_navigation/docs/user_guide.md) — End-user instructions
- [Calibration Guide](voice_navigation/docs/calibration_guide.md) — Distance calibration
- [Privacy Policy](voice_navigation/docs/privacy_policy.md) — Data handling
- [Testing Protocol](voice_navigation/tests/user_testing_protocol.md) — User testing procedures

---

## Requirements

- Python 3.8+
- Webcam (720p recommended)
- Microphone (for voice commands)
- macOS / Linux / Windows
- Optional: Ollama for AI assistant, GPU for faster detection

---

## License

This project is for educational and assistive purposes.
