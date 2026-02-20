# Voice Navigation System - User Guide

## For Blind Users

This guide explains how to use the voice navigation system for daily assistance.

---

## Getting Started

### 1. Put on the Device

1. **Earbuds/Headphones**: Use bone conduction headphones (recommended) or earbuds
2. **Camera**: Position at chest level, facing forward
3. **Microphone**: Near your mouth for voice commands

### 2. Starting the System

Your assistant will:
1. Turn on the device
2. Run the system: `python src/main.py`
3. Wait for "Navigation system ready" voice announcement

---

## Voice Commands

### Quick Commands

Say these single words to get instant responses:

| Command | What it does |
|---------|--------------|
| **"clear"** | Check if path ahead is clear |
| **"ahead"** | What's directly in front of you |
| **"help"** | Get navigation assistance |
| **"status"** | Everything around you right now |
| **"danger"** | Any nearby hazards |
| **"repeat"** | Repeat the last message |

### Full Questions

You can also ask complete questions:

- "What's in front of me?"
- "Is it safe to walk forward?"
- "Where is the door?"
- "Describe my surroundings"
- "How far is the person?"

---

## Understanding Alerts

The system uses different sounds for different dangers:

### Critical Alert (STOP!)
- **Sound**: High pitch, fast beeping
- **Meaning**: Object less than 1 meter away
- **Action**: Stop immediately

### Warning Alert
- **Sound**: Medium pitch, regular beeping  
- **Meaning**: Object 1-2.5 meters away
- **Action**: Slow down, prepare to stop

### Awareness Alert
- **Sound**: Low pitch, slow beeping
- **Meaning**: Object 2.5-5 meters away
- **Action**: Be aware, continue with caution

---

## Direction Words

When the system describes locations:

| Word | Meaning |
|------|---------|
| **"ahead"** / **"center"** | Directly in front of you |
| **"left"** | To your left side |
| **"right"** | To your right side |
| **"close"** | Less than 2 meters away |
| **"far"** | More than 3 meters away |

---

## Example Scenarios

### Walking in a Hallway
1. System: "Path ahead is clear"
2. You walk forward
3. System: "Person ahead at 3 meters"
4. You continue, person passes
5. System: "Path clear"

### Approaching an Obstacle
1. System: *(awareness beep)* "Chair on your left"
2. You shift slightly right
3. System: "Path ahead is clear"
4. You continue safely

### Entering a Room
1. You say: "What's around me?"
2. System: "You are entering a room. Table ahead at 2 meters. Two chairs on your left. Door behind you."

---

## Tips for Best Results

### Speaking
- Speak clearly, at normal volume
- Wait for the beep before speaking
- Keep commands short and simple

### Movement
- Walk at a steady pace
- Don't move too fast (system needs time)
- Stop when you hear critical alerts

### Environment
- Works best indoors
- Good lighting improves detection
- Avoid very crowded spaces initially

---

## Troubleshooting

### "System doesn't hear me"
- Check microphone is positioned near mouth
- Speak louder and clearer
- Wait for the listening beep

### "Too many alerts"
- Reduce sensitivity in settings
- Move to less crowded area
- Ask assistant to adjust thresholds

### "System seems slow"
- This is normal; detection takes ~500ms
- Trust the alerts, even if delayed slightly

### "Wrong directions"
- Ensure camera is pointing forward
- Ask assistant to recalibrate
- Check camera isn't tilted

---

## Daily Checklist

Before each use:
- [ ] Charge devices (camera, headphones)
- [ ] Check camera lens is clean
- [ ] Test with "status" command
- [ ] Verify you can hear alerts clearly

---

## Emergency

If something isn't working:
1. Say **"help"** for assistance
2. Stop moving if unsure
3. Use your cane as backup
4. Contact your assistant

---

## Summary of Commands

| To check... | Say... |
|-------------|--------|
| Path ahead | "clear" or "ahead" |
| Full surroundings | "status" |
| Specific object | "where is the [object]?" |
| Repeat info | "repeat" |
| Get help | "help" |

---

## Frequently Asked Questions

### General

**Q: How long does the battery last?**  
A: 4-6 hours typical usage. Laptop: 4-5 hours, portable setup: 6-8 hours.

**Q: Can I use this outdoors?**  
A: Limited outdoor support. Works best in covered areas. Struggles with bright sunlight.

**Q: What if I lose internet?**  
A: System works completely offline. All processing is local. AI responses may be limited.

### Issues

**Q: "Calibration required" error**  
A: Contact your assistant to run: `python src/calibration_tool.py`

**Q: False alerts when nobody is there**  
A: Ask assistant to increase confidence threshold or improve lighting.

**Q: Voice commands not recognized**  
A: Check microphone position, speak louder, ensure it's not covered.

---

## Maintenance Schedule

### Daily
- [ ] Clean camera lens
- [ ] Check battery levels
- [ ] Test with "status" command

### Weekly
- [ ] Check cable connections
- [ ] Review for errors

### Monthly
- [ ] Recalibrate distance estimation
- [ ] Check for software updates
- [ ] Backup configuration

---

## Contact Your Assistant If:
- System crashes frequently
- Distance estimates consistently wrong
- Voice recognition stops working
- New unusual errors appear

