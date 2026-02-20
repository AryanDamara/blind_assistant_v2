# Distance Calibration Guide

## Overview

The navigation system estimates distances using bounding box sizes. Accurate distance estimation requires calibration for your specific camera setup.

## When to Calibrate

- First-time setup
- After changing camera
- After adjusting camera height/angle
- If distance estimates seem inaccurate

## Quick Calibration

### Step 1: Prepare Environment

1. Clear space: minimum 4 meters ahead
2. Mark distances on floor: 1m, 2m, 3m, 4m
3. Have a person stand as the reference object

### Step 2: Run Calibration Wizard

```bash
cd blind_assistant/voice_navigation
source venv/bin/activate
python src/calibration_tool.py
```

### Step 3: Follow Prompts

The wizard will ask you to:
1. Place person at 1 meter → Press ENTER
2. Place person at 2 meters → Press ENTER
3. Place person at 3 meters → Press ENTER
4. Continue for each distance point

### Step 4: Validate

The wizard validates accuracy by testing at different distances.
**Target: < 20% error at all distances**

### Step 5: Save

Save calibration to `config/settings.yaml`.

---

## Manual Calibration

If the wizard doesn't work, calibrate manually:

### Measure Reference Values

1. Place a person at exactly **2.0 meters**
2. Run detection and note bounding box height in pixels:
   ```bash
   python src/object_detector.py  # Shows bbox dimensions
   ```
3. Record the average bbox height

### Calculate Focal Length

```
focal_length = (bbox_height × distance) / actual_height
focal_length = (400px × 2.0m) / 1.7m = 470
```

### Update Config

Edit `config/settings.yaml`:

```yaml
safety:
  distance_estimation:
    method: bbox_height
    focal_length: 470  # Your calculated value
    reference_heights:
      person: 1.70
      chair: 0.45
      dog: 0.50
```

---

## Troubleshooting

### "Distance estimates too short"
- Increase `focal_length` value
- Check camera is level (not looking down)

### "Distance estimates too long"
- Decrease `focal_length` value
- Ensure person is fully visible in frame

### "Inconsistent readings"
- Improve lighting conditions
- Increase detection confidence threshold
- Recalibrate with more samples

### "No detections during calibration"
- Ensure person is clearly visible
- Check camera is working: `python src/camera_capture.py`
- Lower confidence threshold temporarily

---

## Camera Setup Tips

### Optimal Position
- Height: chest level (1.2-1.4m)
- Angle: horizontal, not tilted
- Field of view: 60-90 degrees

### Mounting Options
- GoPro chest mount (recommended)
- Lanyard mount
- Hat/cap mount (less stable)

### Cable Management
- Route cables under clothing
- Use wireless camera if available
- Secure connections with tape

---

## Verification Test

After calibration, verify accuracy:

```bash
# Run benchmark with distance validation
python tests/benchmark.py --duration 30
```

Check the report for distance estimation accuracy in different scenarios.
