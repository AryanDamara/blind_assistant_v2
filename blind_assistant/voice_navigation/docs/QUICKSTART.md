# Quick Start Guide

Get the Voice Navigation System running in under 5 minutes.

## Prerequisites

- **Python 3.9+**
- **macOS / Linux** (Windows partially supported)
- **Webcam** (built-in or USB)
- **Ollama** (for AI guidance — optional)

## Installation

```bash
# 1. Navigate to the project
cd voice_navigation

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python tools/system_check.py
```

## First Run

```bash
python src/main.py
```

**Expected output:**
```
[Camera] Initialized (source=0, 640x480)
[ObjectDetector] YOLOv8n loaded (80 classes)
[SafetyManager] Initialized
[AudioFeedback] TTS engine 'nsss' initialized successfully
[NavigationSystem] All modules ready. Starting...
```

A video window will open showing the camera feed with detection overlays.

## Controls

| Key | Action |
|-----|--------|
| `q` | Quit the system |
| `Space` | Push-to-talk (ask a question) |

## What You'll See

- **Green boxes**: Detected objects
- **Colored zones**: Left / Center / Right navigation zones
- **FPS counter**: Top-left corner

## What You'll Hear

- **"Person 2 meters on your center"** — Standard navigation alert
- **"CAUTION! Car critical, on your left"** — Urgent vehicle warning
- Voice changes speed and volume based on danger level

## Demo Mode (No Camera)

If you don't have a camera connected or want to test with a video file:

1. Edit `config/settings.yaml`:
   ```yaml
   debug:
     mock_camera: true
     mock_video_path: "data/test_scenarios/crowded_hallway.mp4"
   ```

2. Run normally: `python src/main.py`

## Optional: AI Assistant

For intelligent scene descriptions and navigation advice:

```bash
# Install Ollama (macOS)
brew install ollama

# Start the Ollama server
ollama serve

# Pull the model (one-time, ~2GB download)
ollama pull llama3.2:3b
```

The system will automatically use the AI assistant when Ollama is running.

## Next Steps

- Read [calibration_guide.md](calibration_guide.md) for distance accuracy tuning
- Read [user_guide.md](user_guide.md) for full feature documentation
- Check `config/settings.yaml` to customize detection, audio, and safety settings
