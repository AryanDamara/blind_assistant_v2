# User Testing Protocol

## Overview
This protocol outlines the testing procedure for evaluating the voice navigation system with visually impaired users.

## Pre-Test Requirements

### Equipment Checklist
- [ ] Laptop/computer with webcam (720p minimum)
- [ ] External microphone (recommended)
- [ ] Earbuds or bone conduction headphones
- [ ] Backup audio output device
- [ ] Consent forms
- [ ] Session recording software (optional)

### Environment Setup
- [ ] Clear indoor test space (minimum 5m x 5m)
- [ ] Controlled lighting conditions
- [ ] Test obstacles: chairs, tables, standing person
- [ ] Emergency stop procedure confirmed

### Software Verification
```bash
# Run system check before each test
python tools/system_check.py

# Verify audio output
python src/audio_feedback.py

# Test camera
python src/camera_capture.py
```

---

## Participant Information

### Inclusion Criteria
- Age 18+
- Visual impairment (low vision to complete blindness)
- Able to provide informed consent
- Basic familiarity with smartphone/voice assistants

### Exclusion Criteria
- Mobility impairments affecting walking
- Hearing impairments
- Cognitive impairments affecting task comprehension

---

## Test Scenarios

### Scenario 1: Empty Room (Baseline)
**Duration:** 2 minutes
**Setup:** Clear path, no obstacles
**Tasks:**
1. Walk forward 3 meters
2. Ask "Is the path clear?"
3. Turn left and proceed

**Success Criteria:**
- [ ] User reports path is clear
- [ ] No false positives

---

### Scenario 2: Static Obstacle
**Duration:** 3 minutes
**Setup:** Chair placed 2m ahead center
**Tasks:**
1. Start walking forward
2. Verify alert for chair
3. Navigate around obstacle

**Success Criteria:**
- [ ] Alert within 2.5m of obstacle
- [ ] Correct zone identification (center)
- [ ] User successfully navigates around

---

### Scenario 3: Moving Person
**Duration:** 3 minutes
**Setup:** Assistant walks across path
**Tasks:**
1. User walks forward
2. System detects approaching person
3. User responds to alert

**Success Criteria:**
- [ ] Moving person detected
- [ ] Alert priority appropriate
- [ ] User stops or adjusts path

---

### Scenario 4: Complex Scene
**Duration:** 5 minutes
**Setup:** Multiple obstacles, narrow passage
**Tasks:**
1. Navigate from point A to B
2. Ask "What's around me?"
3. Find the doorway

**Success Criteria:**
- [ ] Multiple objects detected
- [ ] Scene description accurate
- [ ] User reaches destination

---

### Scenario 5: Voice Interaction
**Duration:** 3 minutes
**Setup:** Standard room
**Tasks:**
1. Ask "What's ahead?"
2. Ask "Is it safe to proceed?"
3. Ask "Find the chair"

**Success Criteria:**
- [ ] Speech recognized correctly
- [ ] Relevant responses generated
- [ ] Response time < 3 seconds

---

## Evaluation Metrics

### Quantitative Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Detection accuracy | > 85% | Compare alerts to ground truth |
| False positive rate | < 15% | Count incorrect alerts |
| Alert latency | < 500ms | System telemetry |
| Speech recognition | > 90% | Transcription accuracy |
| Task completion rate | > 80% | Scenarios completed successfully |

### Qualitative Metrics

**Post-Test Questionnaire (1-5 scale):**

1. How safe did you feel using the system?
2. How clear were the audio alerts?
3. How helpful were the scene descriptions?
4. How natural was the voice interaction?
5. Would you use this system daily?

**Open-Ended Questions:**
- What worked well?
- What was confusing or frustrating?
- What features would you add?
- Any safety concerns?

---

## Session Recording

### Data Collected
- System telemetry (latency, detections, alerts)
- Audio recordings (with consent)
- Observer notes
- Questionnaire responses

### Privacy Considerations
- No video recording of participant faces
- Audio recordings stored encrypted
- Data anonymized for analysis
- Retention period: 90 days

---

## Safety Protocol

### During Testing
- Spotter present at all times
- Clear stop command: "STOP" or "EMERGENCY"
- Maximum test duration: 30 minutes
- Break every 10 minutes

### Incident Reporting
If participant makes contact with obstacle:
1. Stop test immediately
2. Check for injury
3. Document incident
4. Review system logs
5. Determine if system failure

---

## Post-Test Procedure

1. Complete questionnaire
2. Debrief interview (5 minutes)
3. Export session telemetry
4. Compensate participant
5. Document observations

### Data Analysis
```bash
# Export session summary
python -c "
from src.telemetry import TelemetryLogger
logger = TelemetryLogger()
logger.export_summary('data/logs/session_xxx/summary.json')
"
```

---

## Sample Consent Form

```
INFORMED CONSENT FOR USER TESTING

Project: Voice Navigation System for Visually Impaired
Principal Investigator: [Name]
Date: [Date]

Purpose: Evaluate the effectiveness and safety of an AI-powered
navigation assistance system.

Procedures: You will wear headphones and use voice commands to
navigate a controlled indoor environment with obstacles.

Risks: Minimal risk of bumping into soft obstacles. A spotter
will be present at all times.

Benefits: Opportunity to shape assistive technology development.

Confidentiality: All data will be anonymized. Audio recordings
(with permission) stored securely for 90 days.

Voluntary: Participation is voluntary. You may stop at any time.

[ ] I agree to participate
[ ] I agree to audio recording

Signature: _________________ Date: _________
```

---

## Test Schedule Template

| Date | Participant | Scenarios | Observer | Notes |
|------|-------------|-----------|----------|-------|
| | | | | |
| | | | | |
| | | | | |
