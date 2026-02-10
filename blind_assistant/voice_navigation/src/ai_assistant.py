"""
AI Assistant Module
-------------------
LLM integration using Ollama for context-aware navigation assistance.

Features:
- Ollama HTTP API integration for local LLM inference
- Navigation-focused prompt engineering
- LRU cache with TTL for response caching
- Rule-based fallback when LLM unavailable/slow
- Scene summarization and query answering

Dependencies:
- requests (for Ollama API)
- Ollama running locally with llama3.2:3b model
"""

import time
import json
import hashlib
import yaml
import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from functools import lru_cache
from collections import OrderedDict
import threading

try:
    import requests
except ImportError:
    print("[AIAssistant] ERROR: requests not installed. Run: pip install requests")
    raise


@dataclass
class LLMResponse:
    """Response from LLM or fallback."""
    text: str
    source: str           # 'llm', 'cache', or 'fallback'
    latency_ms: float
    model: str
    cached: bool = False
    
    def to_dict(self) -> dict:
        return {
            'text': self.text,
            'source': self.source,
            'latency_ms': round(self.latency_ms, 2),
            'model': self.model,
            'cached': self.cached
        }


class TTLCache:
    """
    Simple LRU cache with time-to-live expiration.
    Thread-safe implementation.
    """
    
    def __init__(self, max_entries: int = 100, ttl_seconds: int = 300):
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[str, float] = {}
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        
        # Statistics
        self._hits = 0
        self._misses = 0
    
    def _make_key(self, *args) -> str:
        """Create cache key from arguments."""
        key_data = json.dumps(args, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[str]:
        """Get value from cache if not expired."""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            # Check TTL
            if time.time() - self._timestamps[key] > self._ttl_seconds:
                # Expired
                del self._cache[key]
                del self._timestamps[key]
                self._misses += 1
                return None
            
            # Move to end (LRU)
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
    
    def set(self, key: str, value: str) -> None:
        """Set value in cache."""
        with self._lock:
            # Remove oldest if at capacity
            while len(self._cache) >= self._max_entries:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                del self._timestamps[oldest_key]
            
            self._cache[key] = value
            self._timestamps[key] = time.time()
    
    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        return {
            'size': len(self._cache),
            'max_entries': self._max_entries,
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': round(hit_rate, 1)
        }


class AIAssistant:
    """
    LLM-powered assistant for navigation guidance.
    
    Usage:
        assistant = AIAssistant(config_path="config/settings.yaml")
        
        # Summarize scene
        summary = assistant.summarize_scene(scene_description)
        
        # Answer user query
        response = assistant.answer_query("What's in front of me?", scene_context)
    """
    
    # Navigation-focused system prompt
    SYSTEM_PROMPT = """You are a helpful navigation assistant for a visually impaired person. 
Your job is to describe their surroundings clearly and concisely to help them navigate safely.

Key guidelines:
- Be brief and clear - they're listening, not reading
- Prioritize safety information (obstacles, moving objects)
- Use clock positions or left/center/right for directions
- Mention distances when relevant
- Don't use visual descriptions like "blue" or "bright"
- Focus on what matters for navigation and safety

Current scene context will be provided. Answer questions naturally and helpfully."""

    # Rule-based fallback templates
    FALLBACK_TEMPLATES = {
        'scene_summary': "I can see {object_count} objects. {details}",
        'navigation': "Based on the current view: {advice}",
        'find_object': "Looking for {object}. {result}",
        'general': "Based on what I can detect: {info}",
        'error': "I'm having trouble processing that request. The scene shows {basic_info}."
    }
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize AI assistant with configuration."""
        self._load_config(config_path)
        
        # Initialize cache
        self._cache = TTLCache(
            max_entries=self._cache_max_entries,
            ttl_seconds=self._cache_ttl_seconds
        )
        
        # Connection state
        self._llm_available = False
        self._last_connection_check = 0
        self._connection_check_interval = 30  # Recheck every 30 seconds
        
        # Statistics
        self._total_queries = 0
        self._llm_responses = 0
        self._fallback_responses = 0
        self._cache_responses = 0
        self._total_latency = 0.0
        
        # Check initial connection
        self._check_ollama_connection()
    
    def _load_config(self, config_path: str) -> None:
        """Load LLM settings from YAML config."""
        # Defaults
        self._enabled: bool = True
        self._model: str = "llama3.2:3b"
        self._temperature: float = 0.7
        self._max_tokens: int = 100
        self._timeout_sec: float = 2.0
        self._ollama_url: str = "http://localhost:11434/api/generate"
        
        # Cache settings
        self._cache_enabled: bool = True
        self._cache_max_entries: int = 100
        self._cache_ttl_seconds: int = 300
        
        # Invoke conditions
        self._min_objects_for_llm: int = 2
        self._complex_scene_threshold: bool = True
        self._always_llm_for_queries: bool = True
        
        # Load from file if exists
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
                llm_cfg = config.get('llm', {})
                self._enabled = llm_cfg.get('enabled', True)
                self._model = llm_cfg.get('model', "llama3.2:3b")
                self._temperature = llm_cfg.get('temperature', 0.7)
                self._max_tokens = llm_cfg.get('max_tokens', 100)
                self._timeout_sec = llm_cfg.get('timeout_sec', 2.0)
                
                cache_cfg = llm_cfg.get('cache', {})
                self._cache_enabled = cache_cfg.get('enabled', True)
                self._cache_max_entries = cache_cfg.get('max_entries', 100)
                self._cache_ttl_seconds = cache_cfg.get('ttl_seconds', 300)
                
                invoke_cfg = llm_cfg.get('invoke_conditions', {})
                self._min_objects_for_llm = invoke_cfg.get('min_objects', 2)
                self._complex_scene_threshold = invoke_cfg.get('complex_scene', True)
                self._always_llm_for_queries = invoke_cfg.get('on_user_query', True)
        
        print(f"[AIAssistant] Initialized")
        print(f"[AIAssistant] LLM enabled: {self._enabled}")
        print(f"[AIAssistant] Model: {self._model}")
        print(f"[AIAssistant] Timeout: {self._timeout_sec}s")
        print(f"[AIAssistant] Cache: {'enabled' if self._cache_enabled else 'disabled'}")
    
    def _check_ollama_connection(self) -> bool:
        """Check if Ollama is available."""
        try:
            # Try to reach Ollama API
            response = requests.get(
                "http://localhost:11434/api/tags",
                timeout=2.0
            )
            if response.status_code == 200:
                self._llm_available = True
                models = response.json().get('models', [])
                model_names = [m.get('name', '') for m in models]
                print(f"[AIAssistant] Ollama connected. Models: {model_names[:3]}")
                return True
        except Exception as e:
            print(f"[AIAssistant] Ollama not available: {e}")
        
        self._llm_available = False
        print("[AIAssistant] Using rule-based fallback mode")
        return False
    
    def _should_check_connection(self) -> bool:
        """Check if we should retry Ollama connection."""
        return time.time() - self._last_connection_check > self._connection_check_interval
    
    def _call_ollama(self, prompt: str, system_prompt: str = None) -> Optional[str]:
        """
        Call Ollama API with the given prompt.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt override
            
        Returns:
            Response text or None on failure
        """
        if not self._enabled:
            return None
        
        # Periodic connection check
        if not self._llm_available and self._should_check_connection():
            self._last_connection_check = time.time()
            self._check_ollama_connection()
        
        if not self._llm_available:
            return None
        
        try:
            payload = {
                "model": self._model,
                "prompt": prompt,
                "system": system_prompt or self.SYSTEM_PROMPT,
                "stream": False,
                "options": {
                    "temperature": self._temperature,
                    "num_predict": self._max_tokens
                }
            }
            
            response = requests.post(
                self._ollama_url,
                json=payload,
                timeout=self._timeout_sec
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                print(f"[AIAssistant] Ollama error: {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"[AIAssistant] Ollama timeout ({self._timeout_sec}s)")
            return None
        except Exception as e:
            print(f"[AIAssistant] Ollama error: {e}")
            self._llm_available = False
            return None
    
    def _generate_fallback_response(self, query_type: str, context: Dict[str, Any]) -> str:
        """Generate rule-based fallback response."""
        template = self.FALLBACK_TEMPLATES.get(query_type, self.FALLBACK_TEMPLATES['general'])
        
        try:
            return template.format(**context)
        except KeyError:
            return template.format(info=str(context))
    
    def summarize_scene(self, scene_description: str, object_count: int = 0) -> LLMResponse:
        """
        Summarize the current scene for the user.
        
        Args:
            scene_description: Text description of scene from SceneAnalyzer
            object_count: Number of objects detected
            
        Returns:
            LLMResponse with scene summary
        """
        start_time = time.time()
        self._total_queries += 1
        
        # Check cache
        if self._cache_enabled:
            cache_key = self._cache._make_key('scene', scene_description)
            cached = self._cache.get(cache_key)
            if cached:
                self._cache_responses += 1
                return LLMResponse(
                    text=cached,
                    source='cache',
                    latency_ms=(time.time() - start_time) * 1000,
                    model=self._model,
                    cached=True
                )
        
        # Try LLM
        prompt = f"""Briefly describe this scene for navigation:

{scene_description}

Keep your response under 30 words."""

        llm_response = self._call_ollama(prompt)
        
        if llm_response:
            self._llm_responses += 1
            latency = (time.time() - start_time) * 1000
            self._total_latency += latency
            
            # Cache the response
            if self._cache_enabled:
                self._cache.set(cache_key, llm_response)
            
            return LLMResponse(
                text=llm_response,
                source='llm',
                latency_ms=latency,
                model=self._model
            )
        
        # Fallback
        self._fallback_responses += 1
        fallback_text = self._generate_fallback_response('scene_summary', {
            'object_count': object_count,
            'details': scene_description
        })
        
        return LLMResponse(
            text=fallback_text,
            source='fallback',
            latency_ms=(time.time() - start_time) * 1000,
            model='rule-based'
        )
    
    def answer_query(self, query: str, scene_context: str) -> LLMResponse:
        """
        Answer a user's voice query with scene context.
        
        Args:
            query: User's question
            scene_context: Current scene description
            
        Returns:
            LLMResponse with answer
        """
        start_time = time.time()
        self._total_queries += 1
        
        # Build prompt with context
        prompt = f"""Current scene: {scene_context}

User asks: "{query}"

Answer briefly and helpfully for navigation. Keep response under 40 words."""

        # Check cache for similar queries
        if self._cache_enabled:
            cache_key = self._cache._make_key('query', query, scene_context[:100])
            cached = self._cache.get(cache_key)
            if cached:
                self._cache_responses += 1
                return LLMResponse(
                    text=cached,
                    source='cache',
                    latency_ms=(time.time() - start_time) * 1000,
                    model=self._model,
                    cached=True
                )
        
        # Try LLM
        llm_response = self._call_ollama(prompt)
        
        if llm_response:
            self._llm_responses += 1
            latency = (time.time() - start_time) * 1000
            self._total_latency += latency
            
            if self._cache_enabled:
                self._cache.set(cache_key, llm_response)
            
            return LLMResponse(
                text=llm_response,
                source='llm',
                latency_ms=latency,
                model=self._model
            )
        
        # Fallback - try to answer common queries
        self._fallback_responses += 1
        fallback_text = self._answer_query_fallback(query, scene_context)
        
        return LLMResponse(
            text=fallback_text,
            source='fallback',
            latency_ms=(time.time() - start_time) * 1000,
            model='rule-based'
        )
    
    def _answer_query_fallback(self, query: str, scene_context: str) -> str:
        """Generate fallback answer for common query types."""
        query_lower = query.lower()
        
        # What's around / describe scene
        if any(word in query_lower for word in ['around', 'see', 'describe', 'what']):
            return f"Based on my detection: {scene_context}"
        
        # Is there / find
        if any(word in query_lower for word in ['is there', 'find', 'where', 'any']):
            # Try to extract object name
            for obj in ['person', 'chair', 'car', 'door', 'table', 'dog', 'cat']:
                if obj in query_lower:
                    if obj in scene_context.lower():
                        return f"Yes, I detect a {obj} in the scene. {scene_context}"
                    else:
                        return f"I don't currently detect a {obj}. {scene_context}"
            return f"Looking at the scene: {scene_context}"
        
        # Safe / danger
        if any(word in query_lower for word in ['safe', 'danger', 'obstacle', 'clear']):
            if 'center' in scene_context.lower():
                return "Be cautious. There are objects detected ahead. " + scene_context
            else:
                return "The path ahead appears relatively clear. " + scene_context
        
        # Help / how
        if any(word in query_lower for word in ['help', 'how', 'can you']):
            return ("I can describe your surroundings, find objects, and help you navigate. "
                   "Just ask me what's around you or if there are any obstacles.")
        
        # Default
        return f"Based on what I detect: {scene_context}"
    
    def get_navigation_advice(self, alerts: List[Dict]) -> LLMResponse:
        """
        Get navigation advice based on safety alerts.
        
        Args:
            alerts: List of alert dicts from SafetyManager
            
        Returns:
            LLMResponse with navigation advice
        """
        start_time = time.time()
        self._total_queries += 1
        
        if not alerts:
            return LLMResponse(
                text="The path appears clear. You may proceed safely.",
                source='fallback',
                latency_ms=(time.time() - start_time) * 1000,
                model='rule-based'
            )
        
        # Build alert description
        alert_desc = []
        for alert in alerts[:5]:
            if isinstance(alert, dict):
                class_name = alert.get('class_name', alert.get('class', 'object'))
                zone = alert.get('zone', 'ahead')
                distance = alert.get('distance_m', 0)
                danger = alert.get('danger_level', 'info')
            else:
                class_name = alert.class_name
                zone = alert.zone
                distance = alert.distance_m
                danger = alert.danger_level
            
            alert_desc.append(f"{class_name} at {distance}m on {zone} ({danger})")
        
        alert_text = "; ".join(alert_desc)
        
        # Check cache
        if self._cache_enabled:
            cache_key = self._cache._make_key('nav', alert_text)
            cached = self._cache.get(cache_key)
            if cached:
                self._cache_responses += 1
                return LLMResponse(
                    text=cached,
                    source='cache',
                    latency_ms=(time.time() - start_time) * 1000,
                    model=self._model,
                    cached=True
                )
        
        prompt = f"""Current obstacles: {alert_text}

Give brief navigation advice (under 25 words). Focus on immediate safety."""

        llm_response = self._call_ollama(prompt)
        
        if llm_response:
            self._llm_responses += 1
            if self._cache_enabled:
                self._cache.set(cache_key, llm_response)
            return LLMResponse(
                text=llm_response,
                source='llm',
                latency_ms=(time.time() - start_time) * 1000,
                model=self._model
            )
        
        # Fallback
        self._fallback_responses += 1
        critical = [a for a in alerts if (a.get('danger_level') if isinstance(a, dict) else a.danger_level) == 'critical']
        
        if critical:
            fallback = f"Caution! {alert_desc[0]}. Please slow down and proceed carefully."
        else:
            fallback = f"Be aware: {alert_desc[0]}. Path requires attention."
        
        return LLMResponse(
            text=fallback,
            source='fallback',
            latency_ms=(time.time() - start_time) * 1000,
            model='rule-based'
        )
    
    def get_stats(self) -> dict:
        """Get AI assistant statistics."""
        avg_latency = self._total_latency / max(1, self._llm_responses)
        cache_stats = self._cache.get_stats() if self._cache_enabled else {}
        
        return {
            'total_queries': self._total_queries,
            'llm_responses': self._llm_responses,
            'fallback_responses': self._fallback_responses,
            'cache_responses': self._cache_responses,
            'avg_llm_latency_ms': round(avg_latency, 2),
            'llm_available': self._llm_available,
            'cache': cache_stats
        }
    
    @property
    def is_llm_available(self) -> bool:
        """Check if LLM is currently available."""
        return self._llm_available


# ============================================
# Testing / Demo
# ============================================
if __name__ == "__main__":
    """Test AI assistant module standalone."""
    
    print("=" * 50)
    print("AI Assistant Module Test")
    print("=" * 50)
    
    assistant = AIAssistant(config_path="config/settings.yaml")
    
    # Test scene summary
    print("\n--- Testing Scene Summary ---")
    scene = "There are 3 objects being tracked. 2 persons on the center, 1 chair on the left."
    response = assistant.summarize_scene(scene, object_count=3)
    print(f"Scene: {scene}")
    print(f"Response ({response.source}): {response.text}")
    print(f"Latency: {response.latency_ms:.1f}ms")
    
    # Test cache
    print("\n--- Testing Cache ---")
    response2 = assistant.summarize_scene(scene, object_count=3)
    print(f"Cached response ({response2.source}): {response2.text}")
    print(f"Latency: {response2.latency_ms:.1f}ms")
    
    # Test query answering
    print("\n--- Testing Query Answering ---")
    queries = [
        "What's around me?",
        "Is there a person nearby?",
        "Is it safe to walk forward?",
        "Where is the door?",
        "Help me navigate"
    ]
    
    for query in queries:
        response = assistant.answer_query(query, scene)
        print(f"\nQ: {query}")
        print(f"A ({response.source}): {response.text}")
    
    # Test navigation advice
    print("\n--- Testing Navigation Advice ---")
    mock_alerts = [
        {'class_name': 'person', 'zone': 'center', 'distance_m': 1.5, 'danger_level': 'warning'},
        {'class_name': 'chair', 'zone': 'left', 'distance_m': 0.8, 'danger_level': 'critical'}
    ]
    response = assistant.get_navigation_advice(mock_alerts)
    print(f"Alerts: {mock_alerts}")
    print(f"Advice ({response.source}): {response.text}")
    
    # Print stats
    print("\n--- Statistics ---")
    stats = assistant.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nTest complete!")
