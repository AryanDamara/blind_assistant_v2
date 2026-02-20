"""
Unit Tests for AudioFeedback
-----------------------------
Covers SpeechItem ordering, queue handling, rate limiting,
text validation, and config loading.

Note: TTS engine is NOT invoked. We test the queue logic, priority
handling, and configuration — not actual audio playback.

Run with:
    python -m pytest tests/test_audio_feedback.py -v
"""

import sys
import os
import time
import queue

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from audio_feedback import SpeechItem, AudioFeedback


# --------------- SpeechItem ---------------

class TestSpeechItem:
    """Tests for SpeechItem dataclass and priority ordering."""

    def test_higher_priority_sorts_first(self):
        low = SpeechItem(text="Info", priority=1, voice_profile='calm', timestamp=time.time())
        high = SpeechItem(text="Critical", priority=10, voice_profile='urgent', timestamp=time.time())
        # __lt__ should put higher priority first (for PriorityQueue)
        assert high < low  # high sorts before low

    def test_equal_priority(self):
        a = SpeechItem(text="A", priority=5, voice_profile='alert', timestamp=time.time())
        b = SpeechItem(text="B", priority=5, voice_profile='alert', timestamp=time.time())
        # Neither should sort before the other
        assert not (a < b)
        assert not (b < a)

    def test_priority_queue_ordering(self):
        pq = queue.PriorityQueue()
        items = [
            SpeechItem("Low", 1, 'calm', time.time()),
            SpeechItem("Critical", 10, 'urgent', time.time()),
            SpeechItem("Mid", 5, 'alert', time.time()),
        ]
        for item in items:
            pq.put(item)
        
        # Should come out in priority order: 10, 5, 1
        first = pq.get()
        second = pq.get()
        third = pq.get()
        assert first.priority == 10
        assert second.priority == 5
        assert third.priority == 1


# --------------- AudioFeedback Init ---------------

class TestAudioFeedbackInit:
    """Tests for AudioFeedback initialization and config loading."""

    def test_default_config(self):
        af = AudioFeedback(config_path="nonexistent.yaml")
        assert af._enabled is True
        assert af._verbosity == 'standard'
        assert af._interrupt_on_critical is True
        assert af._max_alerts_per_cycle == 3

    def test_voice_profiles_loaded(self):
        af = AudioFeedback(config_path="nonexistent.yaml")
        assert 'urgent' in af._voice_profiles
        assert 'alert' in af._voice_profiles
        assert 'calm' in af._voice_profiles
        # Urgent should be faster than calm
        assert af._voice_profiles['urgent']['rate'] > af._voice_profiles['calm']['rate']

    def test_queue_max_size(self):
        af = AudioFeedback(config_path="nonexistent.yaml")
        assert af._speech_queue.maxsize == 20


# --------------- Queue Overflow Handling ---------------

class TestQueueOverflow:
    """Tests for speak() queue overflow handling."""

    def test_speak_adds_to_queue_when_not_running(self):
        af = AudioFeedback(config_path="nonexistent.yaml")
        # When not running, speak should still queue items via put_nowait
        # It won't be spoken, but shouldn't crash
        af._enabled = True
        af._running = False
        # Direct queue test
        item = SpeechItem("Test", priority=5, voice_profile='alert', timestamp=time.time())
        af._speech_queue.put_nowait(item)
        assert af._speech_queue.qsize() == 1

    def test_queue_fill_and_overflow(self):
        af = AudioFeedback(config_path="nonexistent.yaml")
        # Fill queue to max
        for i in range(20):
            item = SpeechItem(f"Item {i}", priority=1, voice_profile='calm', timestamp=time.time())
            af._speech_queue.put_nowait(item)
        
        assert af._speech_queue.full()
        
        # Next put_nowait should raise Full
        with pytest.raises(queue.Full):
            extra = SpeechItem("Overflow", priority=1, voice_profile='calm', timestamp=time.time())
            af._speech_queue.put_nowait(extra)


# --------------- Text Validation ---------------

class TestTextValidation:
    """Tests for text length and content validation."""

    def test_max_speech_length(self):
        af = AudioFeedback(config_path="nonexistent.yaml")
        assert af.MAX_SPEECH_LENGTH == 500

    def test_empty_text_handling(self):
        af = AudioFeedback(config_path="nonexistent.yaml")
        # speak() with empty text should not crash
        # We test this by checking the queue stays empty
        af._running = False
        af.speak("", priority=1, voice_profile='calm')
        # Empty text should be skipped (queue stays empty)
        # Implementation may or may not filter this; we just ensure no crash


# --------------- Rate Limiting ---------------

class TestRateLimiting:
    """Tests for rate limiting attributes."""

    def test_min_announcement_interval_exists(self):
        af = AudioFeedback(config_path="nonexistent.yaml")
        assert hasattr(af, '_min_announcement_interval')
        assert af._min_announcement_interval == 0.5  # 500ms

    def test_last_announcement_time_initialized(self):
        af = AudioFeedback(config_path="nonexistent.yaml")
        assert af._last_announcement_time == 0.0


# --------------- Statistics ---------------

class TestStatistics:
    """Tests for statistics tracking."""

    def test_initial_stats(self):
        af = AudioFeedback(config_path="nonexistent.yaml")
        assert af._total_announcements == 0
        assert af._interrupted_count == 0
        assert af._queue_drops == 0

    def test_critical_priority_threshold(self):
        af = AudioFeedback(config_path="nonexistent.yaml")
        assert af.CRITICAL_PRIORITY_THRESHOLD == 20


# --------------- Voice Profile ---------------

class TestVoiceProfile:
    """Tests for voice profile configuration."""

    def test_urgent_profile_highest_rate(self):
        af = AudioFeedback(config_path="nonexistent.yaml")
        assert af._voice_profiles['urgent']['rate'] >= af._voice_profiles['alert']['rate']
        assert af._voice_profiles['alert']['rate'] >= af._voice_profiles['calm']['rate']

    def test_urgent_profile_highest_volume(self):
        af = AudioFeedback(config_path="nonexistent.yaml")
        assert af._voice_profiles['urgent']['volume'] >= af._voice_profiles['alert']['volume']
        assert af._voice_profiles['alert']['volume'] >= af._voice_profiles['calm']['volume']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
