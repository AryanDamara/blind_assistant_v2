"""
Conversation Handler Module
---------------------------
Push-to-talk voice input and query processing.

Features:
- Push-to-talk listener (spacebar activation)
- Speech recognition using Google Web API
- Query intent detection
- Context-aware responses using scene data
- Integration with AIAssistant for LLM responses
- Routes responses through AudioFeedback

Dependencies:
- speech_recognition (pip install SpeechRecognition)
- pyaudio (pip install pyaudio)
- pynput (pip install pynput) for keyboard monitoring

Thread: Runs in background thread (Thread 4 in pipeline)
"""

import time
import threading
import queue
import yaml
import os
from dataclasses import dataclass
from typing import Optional, Callable, List, Dict
from enum import Enum

try:
    import speech_recognition as sr
except ImportError:
    print("[ConversationHandler] ERROR: speech_recognition not installed.")
    print("  Run: pip install SpeechRecognition pyaudio")
    raise

try:
    from pynput import keyboard
except ImportError:
    print("[ConversationHandler] WARNING: pynput not installed. Keyboard PTT disabled.")
    print("  Run: pip install pynput")
    keyboard = None


class QueryIntent(Enum):
    """Types of user queries we can handle."""
    DESCRIBE_SCENE = "describe"      # "What's around me?"
    FIND_OBJECT = "find"             # "Where is the door?"
    NAVIGATION_HELP = "navigate"     # "Is it safe to walk?"
    GENERAL_QUERY = "general"        # Other questions
    SYSTEM_COMMAND = "system"        # "Stop", "Pause", etc.
    UNKNOWN = "unknown"


@dataclass
class QueryResult:
    """Result of processing a voice query."""
    original_text: str
    intent: QueryIntent
    response: str
    source: str              # 'llm', 'fallback', or 'cache'
    processing_time_ms: float
    confidence: float = 1.0
    
    def to_dict(self) -> dict:
        return {
            'query': self.original_text,
            'intent': self.intent.value,
            'response': self.response,
            'source': self.source,
            'processing_time_ms': round(self.processing_time_ms, 2)
        }


class ConversationHandler:
    """
    Push-to-talk voice input handler with LLM integration.
    
    Usage:
        handler = ConversationHandler(
            config_path="config/settings.yaml",
            ai_assistant=assistant,     # AIAssistant instance
            audio_feedback=audio,       # AudioFeedback instance
            scene_provider=get_scene    # Callable that returns current scene
        )
        
        handler.start()
        # User presses space and speaks
        # Handler processes query and speaks response
        handler.stop()
    """
    
    # Intent detection keywords
    INTENT_KEYWORDS = {
        QueryIntent.DESCRIBE_SCENE: [
            'around', 'see', 'describe', 'what is', 'what do', 'tell me about',
            'surroundings', 'environment', 'front of me', 'near me'
        ],
        QueryIntent.FIND_OBJECT: [
            'where', 'find', 'locate', 'is there', 'any', 'looking for',
            'can you see', 'do you see'
        ],
        QueryIntent.NAVIGATION_HELP: [
            'safe', 'walk', 'go', 'navigate', 'path', 'clear', 'danger',
            'obstacle', 'move', 'proceed', 'forward', 'direction'
        ],
        QueryIntent.SYSTEM_COMMAND: [
            'stop', 'pause', 'quit', 'exit', 'be quiet', 'shut up',
            'thank you', 'thanks', 'goodbye', 'bye'
        ]
    }
    
    # Voice command shortcuts for quick queries
    QUICK_COMMANDS = {
        'repeat': 'What did you just say?',
        'clear': 'Is the path clear ahead?',
        'ahead': "What's directly ahead of me?",
        'help': 'Help me navigate safely',
        'status': "What's around me right now?",
        'danger': 'Are there any dangers nearby?'
    }
    
    def __init__(self, config_path: str = "config/settings.yaml",
                 ai_assistant=None, audio_feedback=None,
                 scene_provider: Callable = None):
        """
        Initialize conversation handler.
        
        Args:
            config_path: Path to settings.yaml
            ai_assistant: AIAssistant instance for LLM queries
            audio_feedback: AudioFeedback instance for speaking responses
            scene_provider: Callable that returns current scene description
        """
        self._load_config(config_path)
        
        # External dependencies
        self._ai_assistant = ai_assistant
        self._audio_feedback = audio_feedback
        self._scene_provider = scene_provider or (lambda: "No scene data available")
        
        # Speech recognition
        self._recognizer = sr.Recognizer()
        self._microphone: Optional[sr.Microphone] = None
        
        # Threading
        self._listener_thread: Optional[threading.Thread] = None
        self._running = False
        self._stop_event = threading.Event()
        self._query_queue: queue.Queue = queue.Queue(maxsize=5)
        
        # PTT state
        self._ptt_active = False
        self._keyboard_listener = None
        
        # Statistics
        self._total_queries = 0
        self._successful_recognitions = 0
        self._failed_recognitions = 0
    
    def _load_config(self, config_path: str) -> None:
        """Load voice input settings from YAML config."""
        # Defaults
        self._enabled: bool = True
        self._activation_mode: str = "push_to_talk"
        self._activation_key: str = "space"
        self._timeout_sec: float = 5.0
        self._mic_index: Optional[int] = None
        self._energy_threshold: int = 4000
        self._phrase_time_limit: float = 10.0
        
        # Load from file
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
                voice_cfg = config.get('voice_input', {})
                self._enabled = voice_cfg.get('enabled', False)
                self._activation_mode = voice_cfg.get('activation_mode', 'push_to_talk')
                self._activation_key = voice_cfg.get('activation_key', 'space')
                self._timeout_sec = voice_cfg.get('timeout_sec', 5.0)
                self._mic_index = voice_cfg.get('mic_index')
                self._energy_threshold = voice_cfg.get('energy_threshold', 4000)
        
        print(f"[ConversationHandler] Initialized")
        print(f"[ConversationHandler] Enabled: {self._enabled}")
        print(f"[ConversationHandler] Mode: {self._activation_mode}")
        print(f"[ConversationHandler] Activation key: {self._activation_key}")
    
    def _init_microphone(self) -> bool:
        """Initialize microphone for speech recognition."""
        try:
            if self._mic_index is not None:
                self._microphone = sr.Microphone(device_index=self._mic_index)
            else:
                self._microphone = sr.Microphone()
            
            # Adjust for ambient noise
            print("[ConversationHandler] Calibrating microphone...")
            with self._microphone as source:
                self._recognizer.adjust_for_ambient_noise(source, duration=1)
                self._recognizer.energy_threshold = self._energy_threshold
            
            print(f"[ConversationHandler] Microphone ready (threshold: {self._recognizer.energy_threshold})")
            return True
            
        except Exception as e:
            print(f"[ConversationHandler] ERROR initializing microphone: {e}")
            print("  Make sure you have a microphone connected and pyaudio installed.")
            return False
    
    def _init_keyboard_listener(self) -> bool:
        """Initialize keyboard listener for PTT."""
        if keyboard is None:
            print("[ConversationHandler] Keyboard listener not available")
            return False
        
        try:
            def on_press(key):
                try:
                    if hasattr(key, 'name') and key.name == self._activation_key:
                        if not self._ptt_active:
                            self._ptt_active = True
                            self._on_ptt_start()
                except AttributeError:
                    pass
            
            def on_release(key):
                try:
                    if hasattr(key, 'name') and key.name == self._activation_key:
                        if self._ptt_active:
                            self._ptt_active = False
                            self._on_ptt_end()
                except AttributeError:
                    pass
            
            self._keyboard_listener = keyboard.Listener(
                on_press=on_press,
                on_release=on_release
            )
            self._keyboard_listener.start()
            print(f"[ConversationHandler] PTT keyboard listener started")
            return True
            
        except Exception as e:
            print(f"[ConversationHandler] ERROR starting keyboard listener: {e}")
            return False
    
    def _on_ptt_start(self) -> None:
        """Called when PTT key is pressed."""
        print("[ConversationHandler] 🎤 Listening...")
        if self._audio_feedback:
            # Give audio feedback that we're listening
            self._audio_feedback.speak("Listening", priority=1, voice_profile="calm")
    
    def _on_ptt_end(self) -> None:
        """Called when PTT key is released."""
        print("[ConversationHandler] 🎤 Processing...")
        # The actual recording happens in the listener thread
    
    def _listener_loop(self) -> None:
        """Main listener loop running in background thread."""
        print("[ConversationHandler] Listener thread started")
        
        while not self._stop_event.is_set():
            try:
                # Wait for PTT to be activated
                if self._activation_mode == "push_to_talk" and not self._ptt_active:
                    time.sleep(0.1)
                    continue
                
                # Listen for speech
                text = self._listen_for_speech()
                
                if text:
                    self._successful_recognitions += 1
                    self._total_queries += 1
                    
                    # Process the query
                    result = self.process_query(text)
                    
                    # Speak the response
                    if self._audio_feedback and result.response:
                        # Choose voice profile based on intent
                        profile = "calm"
                        if result.intent == QueryIntent.NAVIGATION_HELP:
                            profile = "alert"
                        elif result.intent == QueryIntent.SYSTEM_COMMAND:
                            profile = "calm"
                        
                        self._audio_feedback.speak(
                            result.response,
                            priority=5,
                            voice_profile=profile
                        )
                    
                    print(f"[ConversationHandler] Query: {text}")
                    print(f"[ConversationHandler] Response: {result.response}")
                else:
                    self._failed_recognitions += 1
                
                # Reset PTT
                self._ptt_active = False
                
            except Exception as e:
                print(f"[ConversationHandler] ERROR in listener loop: {e}")
                time.sleep(0.5)
        
        print("[ConversationHandler] Listener thread stopped")
    
    def _listen_for_speech(self) -> Optional[str]:
        """Listen and transcribe speech."""
        if self._microphone is None:
            return None
        
        try:
            with self._microphone as source:
                print("[ConversationHandler] Listening...")
                
                audio = self._recognizer.listen(
                    source,
                    timeout=self._timeout_sec,
                    phrase_time_limit=self._phrase_time_limit
                )
                
                # Recognize using Google Web API
                print("[ConversationHandler] Recognizing...")
                text = self._recognizer.recognize_google(audio)
                return text.strip()
                
        except sr.WaitTimeoutError:
            print("[ConversationHandler] No speech detected (timeout)")
            return None
        except sr.UnknownValueError:
            print("[ConversationHandler] Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"[ConversationHandler] Speech recognition error: {e}")
            return None
        except Exception as e:
            print(f"[ConversationHandler] ERROR during recognition: {e}")
            return None
    
    def _detect_intent(self, query: str) -> QueryIntent:
        """Detect the intent of a user query."""
        query_lower = query.lower()
        
        # Score each intent
        scores = {intent: 0 for intent in QueryIntent}
        
        for intent, keywords in self.INTENT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query_lower:
                    scores[intent] += 1
        
        # Find highest scoring intent
        max_score = max(scores.values())
        
        if max_score == 0:
            return QueryIntent.GENERAL_QUERY
        
        for intent, score in scores.items():
            if score == max_score:
                return intent
        
        return QueryIntent.GENERAL_QUERY
    
    def _detect_quick_command(self, query: str) -> Optional[str]:
        """Detect and expand quick voice commands."""
        query_lower = query.lower().strip()
        
        for trigger, expansion in self.QUICK_COMMANDS.items():
            if query_lower == trigger or query_lower.startswith(trigger + ' '):
                return expansion
        return None
    
    def process_query(self, query: str) -> QueryResult:
        """
        Process a voice query and generate response.
        
        Args:
            query: Transcribed speech text
            
        Returns:
            QueryResult with response
        """
        start_time = time.time()
        
        # Check for quick command shortcuts
        expanded = self._detect_quick_command(query)
        if expanded:
            print(f"[ConversationHandler] Quick command: '{query}' -> '{expanded}'")
            query = expanded
        
        # Detect intent
        intent = self._detect_intent(query)
        
        # Get current scene context
        scene_context = self._scene_provider()
        
        # Handle system commands locally
        if intent == QueryIntent.SYSTEM_COMMAND:
            response = self._handle_system_command(query)
            return QueryResult(
                original_text=query,
                intent=intent,
                response=response,
                source='local',
                processing_time_ms=(time.time() - start_time) * 1000
            )
        
        # Use AI Assistant if available
        if self._ai_assistant:
            if intent == QueryIntent.DESCRIBE_SCENE:
                llm_response = self._ai_assistant.summarize_scene(scene_context)
            else:
                llm_response = self._ai_assistant.answer_query(query, scene_context)
            
            return QueryResult(
                original_text=query,
                intent=intent,
                response=llm_response.text,
                source=llm_response.source,
                processing_time_ms=(time.time() - start_time) * 1000
            )
        
        # Fallback without AI Assistant
        response = self._generate_fallback_response(query, intent, scene_context)
        return QueryResult(
            original_text=query,
            intent=intent,
            response=response,
            source='fallback',
            processing_time_ms=(time.time() - start_time) * 1000
        )
    
    def _handle_system_command(self, query: str) -> str:
        """Handle system commands like stop, pause, etc."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['stop', 'pause', 'quiet']):
            return "Okay, I'll be quiet."
        
        if any(word in query_lower for word in ['thank', 'thanks']):
            return "You're welcome! Let me know if you need anything else."
        
        if any(word in query_lower for word in ['goodbye', 'bye']):
            return "Goodbye! Stay safe."
        
        return "Command received."
    
    def _generate_fallback_response(self, query: str, intent: QueryIntent, 
                                     scene_context: str) -> str:
        """Generate response without AI Assistant."""
        if intent == QueryIntent.DESCRIBE_SCENE:
            return f"Based on my detection: {scene_context}"
        
        if intent == QueryIntent.FIND_OBJECT:
            return f"Looking at the scene: {scene_context}"
        
        if intent == QueryIntent.NAVIGATION_HELP:
            return f"For navigation: {scene_context}. Please proceed with caution."
        
        return f"I heard: {query}. Current scene: {scene_context}"
    
    def start(self) -> bool:
        """
        Start the conversation handler.
        
        Returns:
            bool: True if started successfully
        """
        if self._running:
            print("[ConversationHandler] Already running")
            return True
        
        if not self._enabled:
            print("[ConversationHandler] Voice input disabled in config")
            return False
        
        # Initialize microphone
        if not self._init_microphone():
            print("[ConversationHandler] Failed to initialize microphone")
            return False
        
        # Initialize keyboard listener for PTT
        if self._activation_mode == "push_to_talk":
            if not self._init_keyboard_listener():
                print("[ConversationHandler] Warning: PTT keyboard listener not available")
        
        # Start listener thread
        self._stop_event.clear()
        self._running = True
        self._listener_thread = threading.Thread(
            target=self._listener_loop,
            name="ConversationHandlerThread",
            daemon=True
        )
        self._listener_thread.start()
        
        print(f"[ConversationHandler] Started (Press {self._activation_key} to speak)")
        return True
    
    def stop(self) -> None:
        """Stop the conversation handler."""
        if not self._running:
            return
        
        print("[ConversationHandler] Stopping...")
        
        self._stop_event.set()
        self._running = False
        
        # Stop keyboard listener
        if self._keyboard_listener is not None:
            self._keyboard_listener.stop()
            self._keyboard_listener = None
        
        # Wait for listener thread
        if self._listener_thread is not None:
            self._listener_thread.join(timeout=2.0)
            self._listener_thread = None
        
        # Print statistics
        stats = self.get_stats()
        print(f"""
[ConversationHandler] Session Statistics:
  Total queries: {stats['total_queries']}
  Successful recognitions: {stats['successful_recognitions']}
  Failed recognitions: {stats['failed_recognitions']}
        """)
        
        print("[ConversationHandler] Stopped")
    
    def get_stats(self) -> dict:
        """Get conversation handler statistics."""
        total = self._successful_recognitions + self._failed_recognitions
        success_rate = (self._successful_recognitions / total * 100) if total > 0 else 0
        
        return {
            'total_queries': self._total_queries,
            'successful_recognitions': self._successful_recognitions,
            'failed_recognitions': self._failed_recognitions,
            'recognition_success_rate': round(success_rate, 1),
            'is_running': self._running,
            'ptt_active': self._ptt_active
        }
    
    def set_scene_provider(self, provider: Callable) -> None:
        """Set the scene context provider callback."""
        self._scene_provider = provider
    
    def set_ai_assistant(self, assistant) -> None:
        """Set the AI assistant instance."""
        self._ai_assistant = assistant
    
    def set_audio_feedback(self, audio) -> None:
        """Set the audio feedback instance."""
        self._audio_feedback = audio
    
    @property
    def is_running(self) -> bool:
        """Check if handler is running."""
        return self._running
    
    @property
    def is_listening(self) -> bool:
        """Check if currently listening (PTT active)."""
        return self._ptt_active
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


# ============================================
# Testing / Demo
# ============================================
if __name__ == "__main__":
    """Test conversation handler module standalone."""
    
    print("=" * 50)
    print("Conversation Handler Module Test")
    print("=" * 50)
    
    # Test intent detection
    print("\n--- Testing Intent Detection ---")
    handler = ConversationHandler(config_path="config/settings.yaml")
    
    test_queries = [
        "What's around me?",
        "Is there a person nearby?",
        "Where is the door?",
        "Is it safe to walk forward?",
        "Help me navigate",
        "Stop talking",
        "Thank you",
        "What time is it?"
    ]
    
    for query in test_queries:
        intent = handler._detect_intent(query)
        print(f"  '{query}' -> {intent.value}")
    
    # Test query processing (without real microphone)
    print("\n--- Testing Query Processing ---")
    
    # Mock scene provider
    def mock_scene():
        return "2 persons in center, 1 chair on left, path ahead is partially blocked"
    
    handler.set_scene_provider(mock_scene)
    
    # Process sample queries
    for query in test_queries[:5]:
        result = handler.process_query(query)
        print(f"\nQ: {query}")
        print(f"Intent: {result.intent.value}")
        print(f"Response ({result.source}): {result.response[:80]}...")
        print(f"Time: {result.processing_time_ms:.1f}ms")
    
    # Test with AI Assistant if available
    print("\n--- Testing with AI Assistant ---")
    try:
        from ai_assistant import AIAssistant
        assistant = AIAssistant(config_path="config/settings.yaml")
        handler.set_ai_assistant(assistant)
        
        result = handler.process_query("What's in front of me?")
        print(f"Q: What's in front of me?")
        print(f"Response ({result.source}): {result.response}")
    except ImportError:
        print("AI Assistant not available for testing")
    
    # Test PTT (requires keyboard interaction)
    print("\n--- Testing Push-to-Talk ---")
    print("Note: Full PTT test requires microphone and keyboard interaction.")
    print("To test PTT:")
    print("  1. Enable voice_input in config/settings.yaml")
    print("  2. Run this module")
    print("  3. Press and hold SPACE, speak, then release")
    
    # Check if we should do interactive test
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        print("\nStarting interactive test...")
        print("Press SPACE to speak. Press Ctrl+C to exit.\n")
        
        handler._enabled = True  # Force enable for testing
        
        try:
            with handler:
                while True:
                    time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nInteractive test ended.")
    else:
        print("\nRun with --interactive flag for live microphone test")
    
    # Print stats
    print("\n--- Statistics ---")
    stats = handler.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nTest complete!")
