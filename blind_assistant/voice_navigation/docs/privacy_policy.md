# Privacy Policy

**Voice Navigation System for Visually Impaired**  
**Version:** 0.2.0  
**Last Updated:** February 10, 2026  
**Effective Date:** February 10, 2026

---

## Executive Summary

**TL;DR:** This system processes everything **locally on your device**. No data leaves your computer except optional speech-to-text via Google API (which you can disable). No tracking. No cloud storage. No analytics.

---

## 1. Data Collection & Processing

### 1.1 What We Process (All Locally)

| Data Type | Purpose | Where It's Processed | Retention |
|-----------|---------|---------------------|-----------|
| **Camera frames** | Real-time object detection | Your device (CPU/GPU) | **0 seconds** - discarded immediately |
| **Voice commands** | Speech-to-text conversion | Your device + Google API* | **0 seconds** - not recorded |
| **Object detections** | Safety alerts generation | Your device | **0 seconds** (or logged if telemetry enabled) |
| **System metrics** | Performance optimization | Your device | Optional local logs |

\* *Google Speech API is optional - see Section 2.2 for offline alternative*

### 1.2 What We Absolutely Do NOT Collect

✅ **We never:**
- Save photos or videos (unless you manually enable debug recording)
- Record or store voice audio
- Collect personal information (name, email, etc.)
- Track your location or movements
- Send data to any external servers (except Google Speech API if enabled)
- Use cookies, analytics, or tracking pixels
- Share data with third parties
- Use your data for advertising or profiling

---

## 2. External Services (Opt-In/Opt-Out)

### 2.1 Google Speech Recognition (Optional)

**What it does:** Converts your voice to text when you ask questions

**Data sent to Google:**
- Audio snippet (5-10 seconds) of your voice command only
- No identifying information attached
- Transmitted over HTTPS

**You can disable this:**
```yaml
# config/settings.yaml
voice_input:
  enabled: false  # Turns off voice commands entirely
```

**Or use offline alternative (advanced):**
```bash
# Install Vosk for 100% offline speech recognition
pip install vosk
# Download model: https://alphacephei.com/vosk/models
```

**Google's Privacy Policy:** https://policies.google.com/privacy

### 2.2 Ollama LLM (100% Local)

**What it does:** Provides intelligent navigation guidance

**Data flow:**
- Runs on `localhost:11434` (your computer only)
- No internet connection required after model download
- No telemetry sent to Ollama servers

**First-time setup downloads model (~2GB):**
```bash
ollama pull llama3.2:3b
```

---

## 3. Telemetry & Logging (Opt-In)

### 3.1 What Gets Logged (If Enabled)

When `telemetry.enabled: true` in config:

```json
{
  "timestamp": "2026-02-10T14:30:00Z",
  "frame_id": 12345,
  "detections": [
    {"class": "person", "confidence": 0.92, "zone": "center"}
  ],
  "latency_ms": 45.2,
  "alerts_generated": 2
}
```

**No images, no audio, no personal data.**

### 3.2 Where Logs Are Stored

```
data/logs/
├── session_20260210_143000.json
├── performance_metrics.csv
└── error_log.txt
```

All files stay on your device in plaintext (you can read/delete them anytime).

### 3.3 How to Disable Logging

**Option 1 - via config:**
```yaml
telemetry:
  enabled: false
```

**Option 2 - delete all logs:**
```bash
rm -rf data/logs/
```

---

## 4. Camera & Microphone Access

### 4.1 Camera

- **Used for:** Real-time object detection (person, car, chair, etc.)
- **Processing:** Every frame analyzed by YOLO model, then **immediately discarded**
- **Storage:** None (unless debug recording manually enabled)
- **Access time:** Only while application is running

**To verify no recording is happening:**
```bash
# Check disk writes while running
sudo fs_usage | grep -i camera  # macOS
sudo inotifywait -m data/       # Linux
```

### 4.2 Microphone

- **Used for:** Push-to-talk voice commands (spacebar activation)
- **Processing:** Audio sent to speech recognizer, converted to text, **audio discarded**
- **Storage:** None (audio is never written to disk)
- **Access time:** Only when spacebar is pressed

---

## 5. Data Retention Policy

| Data Type | Retention Period |
|-----------|-----------------|
| Camera frames | 0 seconds (processed & discarded) |
| Voice audio | 0 seconds (never stored) |
| Telemetry logs | Until manually deleted |
| Model weights (YOLO) | Permanent (stored in `models/` folder) |
| Configuration | Permanent (stored in `config/settings.yaml`) |

---

## 6. Third-Party Components

### 6.1 Network Access Summary

| Component | Network Required? | What Gets Sent? |
|-----------|-------------------|-----------------|
| **YOLOv8** | First run only (model download) | Nothing (downloads from ultralytics.com) |
| **Ollama** | First run only (model download) | Nothing externally (localhost only after setup) |
| **Google Speech API** | Per voice command (if enabled) | Audio snippet for transcription |
| **pyttsx3** | Never | Nothing |
| **OpenCV** | Never | Nothing |

### 6.2 Model Downloads (One-Time)

First run downloads models over HTTPS:

```bash
# YOLOv8-nano (~6 MB)
https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

# Ollama Llama 3.2 (~2 GB)
https://registry.ollama.ai/library/llama3.2:3b
```

After download, everything runs offline (except Google Speech if enabled).

---

## 7. User Rights (GDPR/CCPA Compliance)

Even though this is a local application, we respect data rights:

### 7.1 Right to Access
All your data is in plaintext formats you can read:
- Logs: `data/logs/*.json`
- Config: `config/settings.yaml`
- Models: `models/` (PyTorch .pt files)

### 7.2 Right to Deletion
```bash
# Delete all telemetry
rm -rf data/logs/

# Delete all models
rm -rf models/

# Complete uninstall
pip uninstall -y ultralytics opencv-python pyttsx3
rm -rf ~/.cache/ultralytics  # Model cache
```

### 7.3 Right to Portability
All data is in open formats:
- JSON (logs)
- YAML (config)
- CSV (metrics)
- PyTorch .pt (models)

You can export and use them elsewhere.

### 7.4 Right to Object
You can disable any data processing via `config/settings.yaml`.

---

## 8. Security Measures

### 8.1 Technical Safeguards

- ✅ All processing runs in sandboxed Python process
- ✅ No network server exposed (except Ollama on localhost)
- ✅ File permissions set to user-only (`chmod 600` on logs)
- ✅ No password storage (no authentication needed)
- ✅ Dependencies from trusted sources (PyPI, GitHub)

### 8.2 Recommended Practices

```bash
# Run in virtual environment (isolation)
python -m venv venv
source venv/bin/activate

# Set restrictive file permissions
chmod 700 data/logs/
chmod 600 config/settings.yaml

# Regular updates
pip install --upgrade ultralytics opencv-python

# Monitor processes
ps aux | grep python  # Check no unexpected processes
```

---

## 9. Children's Privacy (COPPA Compliance)

This system:
- Does **not** collect personal information from anyone
- Does **not** have user accounts or age verification
- Does **not** target children
- Does **not** knowingly process data from users under 13

**Parental Note:** This is a computer vision tool. If used by a minor, no personal data is collected, but parents should supervise system usage.

---

## 10. International Users

### 10.1 GDPR (European Union)

- **Legal Basis:** Legitimate interest (assistive technology)
- **Data Controller:** End user (you control all data)
- **Data Processor:** N/A (no external processing)
- **Data Protection Officer:** Not required (local processing only)
- **Cross-border transfers:** None (except optional Google Speech API - see Section 2.1)

### 10.2 CCPA (California)

- **Do Not Sell:** We don't sell data (we don't collect personal data)
- **Opt-Out:** All features can be disabled via config
- **Data Disclosure:** All data visible in `data/logs/`

---

## 11. Audit & Transparency

### 11.1 How to Audit This System

```bash
# 1. Check network connections while running
sudo lsof -i -P | grep python

# 2. Monitor file system writes
sudo fs_usage | grep python

# 3. Inspect telemetry logs
cat data/logs/session_*.json | jq .

# 4. Review source code
# All code is open and readable in src/ folder
```

### 11.2 Expected Network Activity

**Normal:**
- `localhost:11434` - Ollama LLM server (local only)
- `speech.googleapis.com:443` - Google Speech API (if voice enabled)

**Not normal (report if seen):**
- Any other external connections
- Unexpected file uploads
- Unknown processes

---

## 12. Changes to This Policy

### 12.1 Notification

- Updates will be documented in `CHANGELOG.md`
- Major changes will require config file migration
- "Last Updated" date at top reflects revision

### 12.2 Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.2.0 | Feb 10, 2026 | Added LLM integration, voice input |
| 0.1.0 | Jan 15, 2026 | Initial release |

---

## 13. Contact & Reporting

### 13.1 Privacy Questions
Email: privacy@blindassistant.example.com  
Response time: 48 hours

### 13.2 Security Issues
Email: security@blindassistant.example.com  
PGP Key: [available on request]  
Response time: 24 hours

### 13.3 Bug Reports
GitHub Issues: https://github.com/yourorg/blind-assistant/issues

---

## 14. Legal Disclaimers

### 14.1 Medical Device Classification
This is **not** a medical device. It is an assistive technology tool. Always use with caution and additional mobility aids.

### 14.2 Warranty
Provided "AS IS" without warranty. See LICENSE file.

### 14.3 Liability
Not liable for accidents or injuries. User assumes all risk.

---

## Appendix A: Data Minimization Principles

We follow these principles:

1. **Collection Limitation:** Collect only what's needed for functionality
2. **Purpose Limitation:** Use data only for navigation assistance
3. **Data Minimization:** Process minimum data required
4. **Accuracy:** Object detection ~85-95% accurate (YOLO limitations)
5. **Storage Limitation:** Delete data immediately after use
6. **Integrity & Confidentiality:** Local processing only
7. **Accountability:** All code is auditable and open-source

---

**By using this system, you acknowledge you have read and understand this privacy policy.**

For the most current version, see: https://github.com/yourorg/blind-assistant/blob/main/PRIVACY.md
