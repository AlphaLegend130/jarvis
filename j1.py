import importlib
import importlib.util
import json
import math
import difflib
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from collections import deque
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus

import cv2
import numpy as np
import psutil


def optional_import(module_name: str):
    if importlib.util.find_spec(module_name) is None:
        return None
    return importlib.import_module(module_name)


vosk = optional_import("vosk")
sd = optional_import("sounddevice")
pyttsx3 = optional_import("pyttsx3")
mp = optional_import("mediapipe")
requests = optional_import("requests")
bs4 = optional_import("bs4")
soundfile = optional_import("soundfile")

VOSK_AVAILABLE = vosk is not None
AUDIO_AVAILABLE = sd is not None
TTS_AVAILABLE = pyttsx3 is not None
MEDIAPIPE_AVAILABLE = mp is not None and hasattr(mp, "solutions")
WEB_AVAILABLE = requests is not None and bs4 is not None
PIPER_AUDIO_AVAILABLE = soundfile is not None and sd is not None
BeautifulSoup = bs4.BeautifulSoup if bs4 else None


def mediapipe_install_guidance():
    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    base = f"[VISION] MediaPipe hand tracking is unavailable on Python {py}."
    install = " Try: python -m pip install --upgrade pip setuptools wheel && python -m pip install mediapipe"
    fallback = " If installation still fails, create a Python 3.10 virtualenv and install mediapipe there."
    if sys.version_info >= (3, 12):
        return base + " MediaPipe wheels are often unavailable for 3.12+ on many platforms." + fallback
    if sys.version_info >= (3, 11):
        return base + install + fallback
    return base + install


class Config:
    TARGET_FPS = 30
    CAMERA_WIDTH = 320
    CAMERA_HEIGHT = 240

    WAKE_WORD = "jarvis"
    WAKE_WORD_ALTERNATIVES = ["jervis", "jarviz", "jarviss", "jarv", "jarvus"]
    VOSK_MODEL_PATH = "vosk-model-small-en-us-0.15"
    SAMPLE_RATE = 16000

    UI_WIDTH = 1280
    UI_HEIGHT = 720

    HOLO_ENABLED = True
    HOLO_PARTICLES = 0
    HOLO_SCAN_SPEED = 2.0
    HOLO_MAX_OBJECTS = 12

    EVENT_COOLDOWN_SECONDS = 300.0  # Only announce face detection every 5 minutes
    GESTURE_COOLDOWN_SECONDS = 2.0
    VOICE_COMMAND_TIMEOUT_SECONDS = 6.0
    CURSOR_SMOOTHING = 0.35
    FOCUS_SCAN_INTERVAL = 3
    SEARCH_DISPLAY_SECONDS = 3
    MAX_NOTES = 20
    MAX_COMMAND_HISTORY = 25
    DATA_DIR = "."
    NOTES_FILE = "jarvis.json"
    PROFILE_FILE = "profile.json"
    REMINDERS_FILE = "reminders.json"
    HOLOGRAMS_FILE = "holograms.json"
    FACE_DATA_FILE = "face_data.json"
    NIGHTVISION_DEFAULT = False
    SHUTDOWN_FLAG = False
    
    # Voice activation tuning
    WAKE_SIMILARITY_THRESHOLD = 0.7  # Lowered from 0.78 for easier detection
    PARTIAL_RESULT_INTERVAL = 0.3  # Check partial results every 300ms


class VoiceOverlayState:
    def __init__(self):
        self.lock = threading.Lock()
        self.wake_active = False
        self.last_partial = ""
        self.last_final = ""
        self.history = deque(maxlen=3)
        self.speaking_active = False
        self.speaking_level = 0.0

    def set_wake(self, active):
        with self.lock:
            self.wake_active = active

    def set_partial(self, text):
        with self.lock:
            self.last_partial = text

    def push_final(self, text):
        with self.lock:
            self.last_final = text
            self.last_partial = ""
            if text:
                self.history.appendleft(text)

    def set_speaking(self, active, level=0.0):
        with self.lock:
            self.speaking_active = active
            self.speaking_level = level

    def snapshot(self):
        with self.lock:
            return {
                "wake_active": self.wake_active,
                "last_partial": self.last_partial,
                "last_final": self.last_final,
                "history": list(self.history),
                "speaking_active": self.speaking_active,
                "speaking_level": self.speaking_level,
            }


class HolographicEffects:
    def __init__(self):
        self.particles = []
        self.time_offset = 0
        self.active_search = False
        self.search_text = ""
        self.search_results = []
        self.result_display_time = 0

        self.holograms = []
        self.current_stroke = []
        self.selected_hologram = None
        self.last_cursor = None
        self.draw_mode = False
        self.mode_history = deque(maxlen=6)

    def update(self):
        self.time_offset += 0.05
        return

    def update_hand_interaction(self, hand_state, width, height):
        if not hand_state.get("active"):
            self._finish_stroke()
            self.last_cursor = None
            self.selected_hologram = None
            self.mode_history.clear()
            return

        if hand_state.get("mode", "none") == "move" and hand_state.get("palm_x") is not None:
            cursor = (int(hand_state["palm_x"] * width), int(hand_state["palm_y"] * height))
        else:
            cursor = (int(hand_state["x"] * width), int(hand_state["y"] * height))
        mode = hand_state.get("mode", "none")
        self.mode_history.append(mode)
        stable_mode = max(set(self.mode_history), key=self.mode_history.count) if self.mode_history else mode

        if stable_mode == "erase":
            self.erase_near(cursor)
            self.draw_mode = False
        elif stable_mode == "draw":
            self.draw_mode = True
            if not self.current_stroke:
                self.current_stroke = [cursor]
            else:
                prev = self.current_stroke[-1]
                if abs(cursor[0] - prev[0]) + abs(cursor[1] - prev[1]) > 1:
                    self.current_stroke.append(cursor)
        else:
            if self.draw_mode:
                self._finish_stroke()
            self.draw_mode = False

        if stable_mode == "move":
            if self.selected_hologram is None:
                self.selected_hologram = self._nearest_hologram(cursor)
                self.last_cursor = cursor
            elif self.last_cursor is not None:
                dx, dy = cursor[0] - self.last_cursor[0], cursor[1] - self.last_cursor[1]
                self._move_selected(dx, dy)
                self.last_cursor = cursor
        else:
            self.selected_hologram = None
            self.last_cursor = cursor

    def _finish_stroke(self):
        if len(self.current_stroke) > 8:
            points = self.current_stroke.copy()
            if len(points) > 20:
                first, last = points[0], points[-1]
                if math.hypot(first[0] - last[0], first[1] - last[1]) < 25:
                    points.append(first)
            self.holograms.append({"points": points, "life": 1.0})
            if len(self.holograms) > Config.HOLO_MAX_OBJECTS:
                self.holograms = self.holograms[-Config.HOLO_MAX_OBJECTS:]
        self.current_stroke = []

    def _nearest_hologram(self, cursor):
        if not self.holograms:
            return None
        best_idx, best_dist = None, float("inf")
        for idx, holo in enumerate(self.holograms):
            pts = np.array(holo["points"], dtype=np.int32)
            cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
            d = math.hypot(cursor[0] - cx, cursor[1] - cy)
            if d < best_dist and d < 180:
                best_idx, best_dist = idx, d
        return best_idx

    def _move_selected(self, dx, dy):
        if self.selected_hologram is None:
            return
        points = self.holograms[self.selected_hologram]["points"]
        self.holograms[self.selected_hologram]["points"] = [(x + dx, y + dy) for x, y in points]

    def draw_particles(self, overlay):
        for p in self.particles:
            color = (255, int(200 * p["life"]), 0)
            cv2.circle(overlay, (int(p["x"]), int(p["y"])), 2, color, -1)

    def draw_scan_lines(self, overlay, h, w):
        for i in range(3):
            offset = (self.time_offset * Config.HOLO_SCAN_SPEED * 100 + i * h / 3) % h
            cv2.line(overlay, (0, int(offset)), (w, int(offset)), (255, 255, 0), 1)

    def draw_hand_holograms(self, overlay):
        pulse = int((math.sin(self.time_offset * 3) + 1) * 55)
        for idx, holo in enumerate(self.holograms):
            pts = np.array(holo["points"], dtype=np.int32)
            core_color = (255, min(255, 220 + pulse // 2), 120)
            glow_color = (255, 255, 180)
            thick = 6 if idx == self.selected_hologram else 4
            cv2.polylines(overlay, [pts], False, glow_color, thick + 3)
            cv2.polylines(overlay, [pts], False, core_color, thick)
            if len(pts) > 2:
                hull = cv2.convexHull(pts)
                cv2.polylines(overlay, [hull], True, (200, 190 + pulse // 2, 80), 2)

        if len(self.current_stroke) > 1:
            pts = np.array(self.current_stroke, dtype=np.int32)
            cv2.polylines(overlay, [pts], False, (255, 255, 220), 7)
            cv2.polylines(overlay, [pts], False, (255, 255, 150), 4)

    def draw_hexagon(self, overlay, center, radius, color, thickness=2):
        points = np.array(
            [
                [int(center[0] + radius * math.cos(math.pi / 3 * i)), int(center[1] + radius * math.sin(math.pi / 3 * i))]
                for i in range(6)
            ],
            np.int32,
        )
        cv2.polylines(overlay, [points], True, color, thickness)

    def draw_search_display(self, overlay, h, w):
        return

    def set_search(self, text, results=None):
        self.active_search = True
        self.search_text = text
        self.search_results = results or []
        self.result_display_time = time.time()

    def clear_search(self):
        if self.active_search and time.time() - self.result_display_time > Config.SEARCH_DISPLAY_SECONDS:
            self.active_search = False
            self.search_text = ""
            self.search_results = []

    def erase_near(self, cursor, radius=36):
        if not self.holograms:
            return 0
        kept = []
        removed = 0
        r2 = radius * radius
        for holo in self.holograms:
            pts = [pt for pt in holo.get("points", []) if (pt[0] - cursor[0]) ** 2 + (pt[1] - cursor[1]) ** 2 > r2]
            if len(pts) >= 3:
                holo["points"] = pts
                kept.append(holo)
            else:
                removed += 1
        self.holograms = kept
        return removed

    def clear_holograms(self):
        self.holograms = []
        self.current_stroke = []
        self.selected_hologram = None


class DataStore:
    def __init__(self):
        self.base = Path(__file__).resolve().parent / Config.DATA_DIR
        self.base.mkdir(parents=True, exist_ok=True)
        self.notes_path = self.base / Config.NOTES_FILE
        self.profile_path = self.base / Config.PROFILE_FILE
        self.reminders_path = self.base / Config.REMINDERS_FILE
        self.holograms_path = self.base / Config.HOLOGRAMS_FILE
        self.face_data_path = self.base / Config.FACE_DATA_FILE

    def _read_json(self, path, default):
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default

    def _write_json(self, path, payload):
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load_notes(self):
        data = self._read_json(self.notes_path, [])
        return data if isinstance(data, list) else []

    def save_notes(self, notes):
        self._write_json(self.notes_path, list(notes))

    def load_profile(self):
        data = self._read_json(self.profile_path, {})
        return data if isinstance(data, dict) else {}

    def save_profile(self, profile):
        self._write_json(self.profile_path, profile)

    def load_reminders(self):
        data = self._read_json(self.reminders_path, [])
        return data if isinstance(data, list) else []

    def save_reminders(self, reminders):
        self._write_json(self.reminders_path, reminders)

    def load_holograms(self):
        data = self._read_json(self.holograms_path, [])
        return data if isinstance(data, list) else []

    def save_holograms(self, holograms):
        self._write_json(self.holograms_path, holograms)

    def load_face_data(self):
        data = self._read_json(self.face_data_path, {})
        return data if isinstance(data, dict) else {}

    def save_face_data(self, payload):
        self._write_json(self.face_data_path, payload)


class IntentMatcher:
    def __init__(self):
        self.intents = {
            "greeting": ["hello", "hi", "hey", "good morning", "good evening"],
            "time": ["time", "what time", "current time", "clock"],
            "date": ["date", "what date", "today", "day"],
            "search": ["search", "look up", "find", "google"],
            "wikipedia": ["wikipedia", "wiki", "tell me about"],
            "youtube": ["youtube", "video", "play video"],
            "music": ["play music", "music", "song"],
            "open": ["open", "launch", "start"],
            "status": ["status", "how are you", "systems"],
            "vision": ["see", "look", "detect", "camera", "vision"],
            "exit": ["exit", "quit", "goodbye", "bye"],
            "note": ["note", "remember", "take note"],
            "help": ["help", "what can you do", "commands"],
            "hologram": ["hologram", "overlay", "hud"],
            "history": ["history", "last commands", "recent commands"],
            "timer": ["timer", "countdown", "set timer"],
            "reminder": ["remind", "reminder", "remember at"],
            "weather": ["weather", "temperature", "forecast"],
            "math": ["calculate", "math", "what is"],
            "profile": ["my name", "call me", "who am i"],
        }
        self.vocab = sorted({word for phrases in self.intents.values() for phrase in phrases for word in phrase.lower().split()})

    def _vectorize(self, text):
        words = text.lower().split()
        vec = np.array([words.count(word) for word in self.vocab], dtype=float)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def match(self, text):
        text_vec = self._vectorize(text)
        best_intent, best_score = None, 0.0
        for intent, phrases in self.intents.items():
            for phrase in phrases:
                score = float(np.dot(text_vec, self._vectorize(phrase)))
                if score > best_score:
                    best_intent, best_score = intent, score
        return best_intent if best_score > 0.3 else "unknown"


class VoiceEngine:
    def __init__(self, command_queue, tts_queue, voice_overlay_state):
        self.command_queue = command_queue
        self.tts_queue = tts_queue
        self.voice_overlay_state = voice_overlay_state
        self.is_running = False
        self.listening = False
        self.listening_since = 0.0
        self.model = None
        self.recognizer = None
        self.lock = threading.Lock()
        self.last_partial_check = 0.0

        if not (VOSK_AVAILABLE and AUDIO_AVAILABLE):
            print("[VOICE] ❌ Voice unavailable (missing vosk/sounddevice)")
            print("[VOICE] Install with: pip install vosk sounddevice")
            return
        if not os.path.exists(Config.VOSK_MODEL_PATH):
            print(f"[VOICE] ❌ Missing Vosk model: {Config.VOSK_MODEL_PATH}")
            print("[VOICE] Download from: https://alphacephei.com/vosk/models")
            return
        
        print("[VOICE] 🎤 Initializing voice recognition...")
        self.model = vosk.Model(Config.VOSK_MODEL_PATH)
        self.recognizer = vosk.KaldiRecognizer(self.model, Config.SAMPLE_RATE)
        self.recognizer.SetWords(True)  # Enable word-level timestamps
        print(f"[VOICE] ✅ Voice engine ready. Wake word: '{Config.WAKE_WORD}'")

    def start(self):
        if self.recognizer is None:
            print("[VOICE] ⚠️  Cannot start - recognizer not initialized")
            return
        self.is_running = True
        threading.Thread(target=self._listen_loop, daemon=True).start()
        print("[VOICE] 🎧 Listening for wake word...")

    def stop(self):
        self.is_running = False

    def _wake_match_span(self, text):
        """
        Enhanced wake word detection with better fuzzy matching
        """
        lower = text.lower().strip()
        
        # Don't spam console with everything heard
        # Only log when we detect something important
        
        # Split into words for better matching
        words = re.findall(r"[a-zA-Z]+", lower)
        
        for i, word in enumerate(words):
            # Exact match
            if word == Config.WAKE_WORD:
                print(f"[VOICE] ✅ Wake word detected")
                # Find position in original text
                match = re.search(r'\b' + re.escape(word) + r'\b', lower)
                if match:
                    return match.start(), match.end()
            
            # Check alternatives
            if word in Config.WAKE_WORD_ALTERNATIVES:
                print(f"[VOICE] ✅ Wake word detected (variant)")
                match = re.search(r'\b' + re.escape(word) + r'\b', lower)
                if match:
                    return match.start(), match.end()
            
            # Fuzzy match - check if word starts with "jarv"
            if word.startswith("jarv") and len(word) >= 4:
                print(f"[VOICE] ✅ Wake word detected")
                match = re.search(r'\b' + re.escape(word) + r'\b', lower)
                if match:
                    return match.start(), match.end()
            
            # Similarity match
            ratio = difflib.SequenceMatcher(None, word, Config.WAKE_WORD).ratio()
            if ratio >= Config.WAKE_SIMILARITY_THRESHOLD:
                print(f"[VOICE] ✅ Wake word detected")
                match = re.search(r'\b' + re.escape(word) + r'\b', lower)
                if match:
                    return match.start(), match.end()
        
        return None

    def _listen_loop(self):
        try:
            with sd.RawInputStream(
                samplerate=Config.SAMPLE_RATE, 
                blocksize=8000, 
                dtype="int16", 
                channels=1
            ) as stream:
                while self.is_running:
                    try:
                        data, overflowed = stream.read(4000)
                        
                        with self.lock:
                            if self.recognizer.AcceptWaveform(bytes(data)):
                                result = json.loads(self.recognizer.Result())
                                text = result.get("text", "").strip()
                                
                                if not text:
                                    continue
                                
                                wake_span = self._wake_match_span(text)
                                
                                if wake_span is not None:
                                    self.voice_overlay_state.push_final(text)
                                    wake_end = wake_span[1]
                                    remainder = text[wake_end:].strip(" ,.:;!?")
                                    
                                    if remainder:
                                        # Wake word + command in one utterance
                                        self.listening = False
                                        self.voice_overlay_state.set_wake(False)
                                        self.command_queue.put(("voice_command", remainder))
                                    else:
                                        # Just wake word - start listening
                                        self.listening = True
                                        self.listening_since = time.time()
                                        self.voice_overlay_state.set_wake(True)
                                        self.tts_queue.put("Yes, sir?")
                                    continue
                                
                                # If we're already listening, this is the command
                                if self.listening:
                                    self.listening = False
                                    self.voice_overlay_state.set_wake(False)
                                    self.voice_overlay_state.push_final(text)
                                    self.command_queue.put(("voice_command", text))
                            else:
                                # Partial result - check more frequently
                                now = time.time()
                                if now - self.last_partial_check >= Config.PARTIAL_RESULT_INTERVAL:
                                    self.last_partial_check = now
                                    
                                    partial_result = json.loads(self.recognizer.PartialResult())
                                    partial = partial_result.get("partial", "").strip()
                                    
                                    if partial:
                                        # Check if wake word is in partial
                                        if self._wake_match_span(partial) is not None and not self.listening:
                                            self.listening = True
                                            self.listening_since = time.time()
                                            self.voice_overlay_state.set_wake(True)
                                        
                                        if self.listening:
                                            self.voice_overlay_state.set_partial(partial)
                            
                            # Timeout check
                            if self.listening and (time.time() - self.listening_since) > Config.VOICE_COMMAND_TIMEOUT_SECONDS:
                                self.listening = False
                                self.voice_overlay_state.set_wake(False)
                                self.voice_overlay_state.set_partial("")
                    
                    except Exception as e:
                        print(f"[VOICE] ❌ Error: {e}")
                        time.sleep(0.1)
        except Exception as e:
            print(f"[VOICE] ❌ Failed to start audio stream: {e}")
            print("[VOICE] Check your microphone permissions and device")


class TTSEngine:
    def __init__(self, tts_queue, voice_overlay_state):
        self.tts_queue = tts_queue
        self.voice_overlay_state = voice_overlay_state
        self.is_running = False
        self.use_piper = False
        self.piper_path = None
        self.model_path = None
        self.engine = None
        
        # Try to detect Piper TTS first (higher quality)
        self._detect_piper()
        
        # Fall back to pyttsx3 if Piper not available
        if not self.use_piper and TTS_AVAILABLE:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty("rate", 175)
                self.engine.setProperty("volume", 0.9)
                print("[TTS] ✅ Text-to-speech engine initialized (pyttsx3)")
            except Exception as e:
                print(f"[TTS] ❌ Failed to initialize TTS: {e}")
        elif not self.use_piper:
            print("[TTS] ❌ TTS unavailable (missing both Piper and pyttsx3)")
            print("[TTS] Install pyttsx3 with: pip install pyttsx3")
            print("[TTS] Or install Piper from: https://github.com/rhasspy/piper")
    
    def _detect_piper(self):
        """Detect and configure Piper TTS if available"""
        # Common Piper installation paths
        possible_paths = [
            # Windows paths
            r"C:\Users\Admin\Desktop\jarvis\piper_windows_amd64\piper\piper.exe",
            r"C:\piper\piper.exe",
            os.path.expanduser(r"~\Desktop\jarvis\piper_windows_amd64\piper\piper.exe"),
            # Linux/Mac paths
            "/usr/local/bin/piper",
            "/usr/bin/piper",
            os.path.expanduser("~/piper/piper"),
        ]
        
        # Common model paths
        model_paths = [
            r"C:\Users\Admin\Desktop\jarvis\piper_windows_amd64\piper\en_US-lessac-medium.onnx",
            r"C:\piper\en_US-lessac-medium.onnx",
            os.path.expanduser(r"~\Desktop\jarvis\piper_windows_amd64\piper\en_US-lessac-medium.onnx"),
            "/usr/local/share/piper/en_US-lessac-medium.onnx",
            os.path.expanduser("~/piper/en_US-lessac-medium.onnx"),
        ]
        
        # Check if soundfile and sounddevice are available
        try:
            import soundfile as sf
            import sounddevice as sd
        except ImportError:
            print("[TTS] ⚠️  Piper requires soundfile and sounddevice")
            print("[TTS] Install with: pip install soundfile sounddevice")
            return
        
        # Find Piper executable
        for path in possible_paths:
            if os.path.exists(path):
                self.piper_path = path
                break
        
        # Find model
        for path in model_paths:
            if os.path.exists(path):
                self.model_path = path
                break
        
        if self.piper_path and self.model_path:
            self.use_piper = True
            print(f"[TTS] ✅ Piper TTS initialized (high quality)")
            print(f"[TTS] 📂 Executable: {self.piper_path}")
            print(f"[TTS] 📂 Model: {self.model_path}")
        elif self.piper_path:
            print(f"[TTS] ⚠️  Piper found but model missing")
            print(f"[TTS] Download model from: https://github.com/rhasspy/piper/releases")
        elif self.model_path:
            print(f"[TTS] ⚠️  Piper model found but executable missing")

    def start(self):
        if not self.use_piper and self.engine is None:
            print("[TTS] ⚠️  Cannot start - no TTS engine available")
            return
        self.is_running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self.is_running = False

    def _speak_piper(self, text):
        """Speak using Piper TTS"""
        import soundfile as sf
        import sounddevice as sd
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            wav_path = tmp.name
        
        try:
            cmd = [
                self.piper_path,
                "-m", self.model_path,
                "-f", wav_path
            ]
            
            # Hide the console window on Windows
            startupinfo = None
            if os.name == 'nt':  # Windows
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
            
            result = subprocess.run(
                cmd, 
                input=text.encode("utf-8"),
                capture_output=True,
                timeout=10,
                startupinfo=startupinfo,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            
            if result.returncode != 0:
                return False
            
            # Play the audio
            data, samplerate = sf.read(wav_path)
            sd.play(data, samplerate)
            sd.wait()
            
            return True
        except subprocess.TimeoutExpired:
            return False
        except Exception as e:
            return False
        finally:
            # Clean up temp file
            try:
                if os.path.exists(wav_path):
                    os.remove(wav_path)
            except:
                pass
    
    def _speak_pyttsx3(self, text):
        """Speak using pyttsx3"""
        try:
            self.engine.say(text)
            self.engine.runAndWait()
            return True
        except Exception as e:
            print(f"[TTS] ❌ Speech error: {e}")
            return False

    def _loop(self):
        while self.is_running:
            try:
                text = self.tts_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            
            # Only log the message, don't spam with emoji
            print(f"[JARVIS] {text}")
            level = min(1.0, 0.35 + len(text) / 220.0)
            self.voice_overlay_state.set_speaking(True, level)
            
            # Use Piper if available, otherwise fall back to pyttsx3
            if self.use_piper:
                success = self._speak_piper(text)
                if not success and self.engine:
                    # Fall back to pyttsx3 if Piper fails
                    self._speak_pyttsx3(text)
            elif self.engine:
                self._speak_pyttsx3(text)
            
            self.voice_overlay_state.set_speaking(False, 0.0)


class VisionEngine:
    def __init__(self, vision_queue):
        self.vision_queue = vision_queue
        self.is_running = False
        self.frame = None
        self.frame_lock = threading.Lock()
        self.face_detected = False
        self.hand_detected = False
        self.gesture_enabled = False
        self.fps_deque = deque(maxlen=30)
        self.last_frame_time = time.time()
        self.last_event_time = {"face": 0.0, "hand": 0.0}
        self.last_gesture_time = {}
        self.smoothed_cursor = None
        self.gesture_history = deque(maxlen=7)
        self.mode_history = deque(maxlen=6)
        self.frame_counter = 0
        self.focus_request = None
        self.focus_object = None
        self.smoothed_palm = None
        self.hand_state = {
            "active": False, 
            "x": 0.5, 
            "y": 0.5, 
            "palm_x": 0.5, 
            "palm_y": 0.5, 
            "mode": "none", 
            "gesture": "none", 
            "pinch_idx": 1.0, 
            "pinch_mid": 1.0, 
            "bbox": None
        }
        self.regression_window = deque(maxlen=6)
        self.eye_state = {"left": (0.5, 0.5), "right": (0.5, 0.5), "movement": "steady"}
        self.face_data = {}

        self.use_mediapipe = MEDIAPIPE_AVAILABLE
        self.mp_face = None
        self.mp_hands = None
        self.face_mesh = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        
        if self.use_mediapipe:
            print("[VISION] ✅ MediaPipe + OpenCV tracking enabled")
            self.mp_face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
            self.mp_hands = mp.solutions.hands.Hands(
                min_detection_confidence=0.5, 
                min_tracking_confidence=0.5, 
                max_num_hands=2
            )
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1, 
                refine_landmarks=True, 
                min_detection_confidence=0.5, 
                min_tracking_confidence=0.5
            )
        else:
            print(mediapipe_install_guidance())
            print("[VISION] ⚠️  MediaPipe unavailable; OpenCV face tracking active")
            print("[VISION] ⚠️  Hand gestures require MediaPipe")

    def start(self):
        self.is_running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self.is_running = False

    def get_frame(self):
        with self.frame_lock:
            return self.frame.copy() if self.frame is not None else None

    def _emit_event(self, event_name):
        now = time.time()
        if now - self.last_event_time.get(event_name, 0) >= Config.EVENT_COOLDOWN_SECONDS:
            self.vision_queue.put(("detection", event_name))
            self.last_event_time[event_name] = now

    def _emit_gesture(self, gesture_name):
        now = time.time()
        last = self.last_gesture_time.get(gesture_name, 0.0)
        if now - last >= Config.GESTURE_COOLDOWN_SECONDS:
            self.vision_queue.put(("gesture", gesture_name))
            self.last_gesture_time[gesture_name] = now
            return True
        return False

    def _classify_color(self, hsv_roi):
        h = float(np.mean(hsv_roi[:, :, 0]))
        s = float(np.mean(hsv_roi[:, :, 1]))
        v = float(np.mean(hsv_roi[:, :, 2]))
        if v < 50:
            return "dark"
        if s < 35:
            return "neutral"
        if h < 10 or h >= 160:
            return "red"
        if h < 25:
            return "orange"
        if h < 35:
            return "yellow"
        if h < 85:
            return "green"
        if h < 130:
            return "blue"
        return "purple"

    def _classify_shape(self, roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 120)
        edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return "object", 0.35
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < 100:
            return "small object", 0.3
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        v = len(approx)
        if v <= 3:
            return "triangle", 0.55
        if v == 4:
            return "box", 0.65
        if v < 8:
            return "polygon", 0.55
        return "round object", 0.6

    def _scan_focus_object(self, frame, center):
        h, w = frame.shape[:2]
        cx, cy = int(center[0]), int(center[1])
        half = 48
        x1, y1 = max(0, cx - half), max(0, cy - half)
        x2, y2 = min(w, cx + half), min(h, cy + half)
        if x2 - x1 < 20 or y2 - y1 < 20:
            return None
        roi = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        color = self._classify_color(hsv)
        shape, conf = self._classify_shape(roi)
        return {
            "bbox": (x1, y1, x2, y2),
            "label": f"{color} {shape}",
            "confidence": conf,
            "updated": time.time(),
        }

    def _update_focus_object(self, frame):
        self.frame_counter += 1
        if self.focus_request is not None:
            result = self._scan_focus_object(frame, self.focus_request)
            if result is not None:
                self.focus_object = result
                self.vision_queue.put(("focus", result["label"]))
            self.focus_request = None
        elif self.focus_object is not None and self.frame_counter % Config.FOCUS_SCAN_INTERVAL == 0:
            x1, y1, x2, y2 = self.focus_object["bbox"]
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            result = self._scan_focus_object(frame, center)
            if result is not None:
                self.focus_object = result

    def _draw_focus_overlay(self, frame):
        if not self.focus_object:
            return
        x1, y1, x2, y2 = self.focus_object["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 0), 2)
        txt = f"TARGET: {self.focus_object['label']} ({self.focus_object['confidence']:.2f})"
        cv2.putText(frame, txt, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 180, 0), 1)

    def _linear_regression_predict(self, x, y):
        self.regression_window.append((x, y))
        if len(self.regression_window) < 4:
            return x, y
        t = np.arange(len(self.regression_window), dtype=float)
        xs = np.array([p[0] for p in self.regression_window], dtype=float)
        ys = np.array([p[1] for p in self.regression_window], dtype=float)
        px = np.polyfit(t, xs, 1)
        py = np.polyfit(t, ys, 1)
        t_next = float(len(self.regression_window))
        pred_x = float(np.clip(px[0] * t_next + px[1], 0.0, 1.0))
        pred_y = float(np.clip(py[0] * t_next + py[1], 0.0, 1.0))
        return pred_x, pred_y

    def _is_open_palm(self, lmk):
        wrist = lmk[0]
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        up = 0
        for tip_i, pip_i in zip(tips, pips):
            tip = lmk[tip_i]
            pip = lmk[pip_i]
            if math.hypot(tip.x - wrist.x, tip.y - wrist.y) > math.hypot(pip.x - wrist.x, pip.y - wrist.y) * 1.10:
                up += 1
        return up == 4

    def _update_face_eye_tracking(self, frame, rgb):
        if not self.face_mesh:
            return
        
        mesh_res = self.face_mesh.process(rgb)
        if not (mesh_res and mesh_res.multi_face_landmarks):
            return
        
        lmk = mesh_res.multi_face_landmarks[0].landmark
        left_ids = [33, 133, 159, 145]
        right_ids = [362, 263, 386, 374]
        left = np.mean(np.array([[lmk[i].x, lmk[i].y] for i in left_ids], dtype=float), axis=0)
        right = np.mean(np.array([[lmk[i].x, lmk[i].y] for i in right_ids], dtype=float), axis=0)
        prev_l = self.eye_state.get("left", tuple(left))
        movement = "steady"
        if left[0] - prev_l[0] > 0.01:
            movement = "right"
        elif left[0] - prev_l[0] < -0.01:
            movement = "left"
        self.eye_state = {
            "left": (float(left[0]), float(left[1])), 
            "right": (float(right[0]), float(right[1])), 
            "movement": movement
        }
        h, w = frame.shape[:2]
        lx, ly = int(left[0] * w), int(left[1] * h)
        rx, ry = int(right[0] * w), int(right[1] * h)
        cv2.circle(frame, (lx, ly), 4, (220, 255, 120), -1)
        cv2.circle(frame, (rx, ry), 4, (220, 255, 120), -1)
        cv2.putText(frame, f"EYES: {movement.upper()}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 120), 1)
        self.face_data = {"eyes": self.eye_state, "updated": time.time()}

    def _detect_faces_opencv(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        ) if self.face_cascade is not None else []

    def _process_mediapipe(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.mp_face.process(rgb) if self.mp_face else None
        cv_faces = self._detect_faces_opencv(frame)
        hands = self.mp_hands.process(rgb) if self.mp_hands else None
        
        mp_face_detected = bool(faces and faces.detections)
        cv_face_detected = len(cv_faces) > 0
        self.face_detected = mp_face_detected or cv_face_detected
        
        all_hands = hands.multi_hand_landmarks if (hands and hands.multi_hand_landmarks) else []
        self.hand_detected = bool(all_hands)

        if self.face_detected and self.gesture_enabled:
            self._emit_event("face")

        for x, y, w, h in cv_faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 180), 1)

        self._update_face_eye_tracking(frame, rgb)

        if len(all_hands) >= 2 and all(self._is_open_palm(h.landmark) for h in all_hands[:2]):
            if not self.gesture_enabled:
                self.gesture_enabled = True
                self.vision_queue.put(("trigger", "two_hands"))

        if self.hand_detected and self.gesture_enabled:
            self._emit_event("hand")
            lmk = all_hands[0].landmark
            idx_tip = lmk[8]
            thumb_tip = lmk[4]
            mid_tip = lmk[12]
            pinch_idx = math.hypot(idx_tip.x - thumb_tip.x, idx_tip.y - thumb_tip.y)
            pinch_mid = math.hypot(mid_tip.x - thumb_tip.x, mid_tip.y - thumb_tip.y)

            idx_mcp = lmk[5]
            pinky_mcp = lmk[17]
            hand_scale = max(0.03, math.hypot(idx_mcp.x - pinky_mcp.x, idx_mcp.y - pinky_mcp.y))
            draw_threshold = 0.55 * hand_scale
            move_threshold = 0.60 * hand_scale

            mode = "none"
            if pinch_idx < draw_threshold:
                mode = "draw"
            elif pinch_mid < move_threshold:
                mode = "move"

            all_pts = np.array([[lm.x, lm.y] for lm in lmk], dtype=float)
            palm_ids = [0, 5, 9, 13, 17]
            palm_center = np.mean(all_pts[palm_ids], axis=0)
            draw_point = 0.72 * np.array([idx_tip.x, idx_tip.y]) + 0.28 * palm_center

            raw_x, raw_y = self._linear_regression_predict(float(draw_point[0]), float(draw_point[1]))
            if self.smoothed_cursor is None:
                self.smoothed_cursor = (raw_x, raw_y)
            else:
                a = Config.CURSOR_SMOOTHING
                self.smoothed_cursor = (
                    a * raw_x + (1 - a) * self.smoothed_cursor[0],
                    a * raw_y + (1 - a) * self.smoothed_cursor[1],
                )

            raw_px, raw_py = float(palm_center[0]), float(palm_center[1])
            if self.smoothed_palm is None:
                self.smoothed_palm = (raw_px, raw_py)
            else:
                a2 = min(0.55, Config.CURSOR_SMOOTHING + 0.1)
                self.smoothed_palm = (
                    a2 * raw_px + (1 - a2) * self.smoothed_palm[0],
                    a2 * raw_py + (1 - a2) * self.smoothed_palm[1],
                )

            wrist = lmk[0]
            def extended(tip_i, pip_i):
                tip = lmk[tip_i]
                pip = lmk[pip_i]
                d_tip = math.hypot(tip.x - wrist.x, tip.y - wrist.y)
                d_pip = math.hypot(pip.x - wrist.x, pip.y - wrist.y)
                return d_tip > d_pip * 1.12

            index_up = extended(8, 6)
            middle_up = extended(12, 10)
            ring_up = extended(16, 14)
            pinky_up = extended(20, 18)

            raw_mode = "none"
            if pinch_idx < draw_threshold * 0.90:
                raw_mode = "draw"
            elif pinch_mid < move_threshold * 0.92 and pinch_idx > draw_threshold * 1.15:
                raw_mode = "move"
            self.mode_history.append(raw_mode)
            mode = max(set(self.mode_history), key=self.mode_history.count) if self.mode_history else raw_mode

            raw_gesture = "none"
            if index_up and middle_up and ring_up and pinky_up:
                raw_gesture = "eraser"
            elif index_up and middle_up and not ring_up and not pinky_up:
                raw_gesture = "peace"
            elif not any([index_up, middle_up, ring_up, pinky_up]) and raw_mode == "none":
                raw_gesture = "fist"

            self.gesture_history.append(raw_gesture)
            gesture = max(set(self.gesture_history), key=self.gesture_history.count) if self.gesture_history else raw_gesture

            if gesture != "none":
                self._emit_gesture(gesture)

            min_xy = np.clip(np.min(all_pts, axis=0), 0.0, 1.0)
            max_xy = np.clip(np.max(all_pts, axis=0), 0.0, 1.0)
            bbox = (float(min_xy[0]), float(min_xy[1]), float(max_xy[0]), float(max_xy[1]))

            self.hand_state = {
                "active": True,
                "x": self.smoothed_cursor[0],
                "y": self.smoothed_cursor[1],
                "palm_x": self.smoothed_palm[0],
                "palm_y": self.smoothed_palm[1],
                "mode": "erase" if gesture == "eraser" else mode,
                "gesture": gesture,
                "pinch_idx": pinch_idx,
                "pinch_mid": pinch_mid,
                "bbox": bbox,
            }

            h, w = frame.shape[:2]
            cx, cy = int(self.smoothed_cursor[0] * w), int(self.smoothed_cursor[1] * h)
            
            if self.use_mediapipe:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    all_hands[0],
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style(),
                )
            cv2.circle(frame, (cx, cy), 6, (0, 255, 255), -1)
            cv2.putText(
                frame, 
                f"HAND MODE: {mode.upper()}  GESTURE: {gesture.upper()}", 
                (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 255, 255), 
                1
            )
            cv2.putText(
                frame, 
                f"pinchIT:{pinch_idx:.3f}/{draw_threshold:.3f} pinchTM:{pinch_mid:.3f}/{move_threshold:.3f}", 
                (10, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.43, 
                (130, 255, 255), 
                1
            )
        else:
            self.smoothed_cursor = None
            if not self.gesture_enabled:
                cv2.putText(
                    frame, 
                    "SHOW TWO OPEN HANDS TO START TRACKING", 
                    (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 170, 255), 
                    1
                )
            self.mode_history.clear()
            self.gesture_history.clear()
            self.smoothed_palm = None
            self.hand_state = {
                "active": False, 
                "x": 0.5, 
                "y": 0.5, 
                "palm_x": 0.5, 
                "palm_y": 0.5, 
                "mode": "none", 
                "gesture": "none", 
                "pinch_idx": 1.0, 
                "pinch_mid": 1.0, 
                "bbox": None
            }

        self._update_focus_object(frame)
        self._draw_focus_overlay(frame)
        return frame

    def _process_opencv(self, frame):
        faces = self._detect_faces_opencv(frame)
        self.face_detected = len(faces) > 0
        self.hand_detected = False
        self.gesture_enabled = False
        self.smoothed_palm = None
        self.hand_state = {
            "active": False, 
            "x": 0.5, 
            "y": 0.5, 
            "palm_x": 0.5, 
            "palm_y": 0.5, 
            "mode": "none", 
            "gesture": "none", 
            "pinch_idx": 1.0, 
            "pinch_mid": 1.0, 
            "bbox": None
        }
        if self.face_detected:
            self._emit_event("face")
            for x, y, w, h in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        self._update_focus_object(frame)
        self._draw_focus_overlay(frame)
        return frame

    def _loop(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[VISION] ❌ Failed to open camera")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
        
        while self.is_running and cap.isOpened():
            start = time.time()
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            
            frame = cv2.flip(frame, 1)
            frame = self._process_mediapipe(frame) if self.use_mediapipe else self._process_opencv(frame)
            
            with self.frame_lock:
                self.frame = frame
            
            dt = time.time() - self.last_frame_time
            if dt > 0:
                self.fps_deque.append(1.0 / dt)
            self.last_frame_time = time.time()
            time.sleep(max(0, 1 / Config.TARGET_FPS - (time.time() - start)))
        
        cap.release()

    def get_fps(self):
        return sum(self.fps_deque) / len(self.fps_deque) if self.fps_deque else 0.0


class CommandEngine:
    def __init__(self, command_queue, tts_queue, vision_queue, holo_effects):
        self.command_queue = command_queue
        self.tts_queue = tts_queue
        self.vision_queue = vision_queue
        self.holo_effects = holo_effects
        self.intent_matcher = IntentMatcher()
        self.is_running = False
        self.last_feedback = {}
        self.store = DataStore()
        self.notes = deque(self.store.load_notes(), maxlen=Config.MAX_NOTES)
        self.command_history = deque(maxlen=Config.MAX_COMMAND_HISTORY)
        self.last_topics = deque(maxlen=5)
        self.profile = self.store.load_profile()
        self.reminders = self.store.load_reminders()

    def start(self):
        self.is_running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self.is_running = False

    def _persist_notes(self):
        self.store.save_notes(list(self.notes))

    def _persist_profile(self):
        self.store.save_profile(self.profile)

    def _persist_reminders(self):
        self.store.save_reminders(self.reminders)

    def _parse_seconds(self, text):
        lower = text.lower()
        m = re.search(r"(\d+)\s*(second|sec|s)\b", lower)
        if m:
            return int(m.group(1))
        m = re.search(r"(\d+)\s*(minute|min|m)\b", lower)
        if m:
            return int(m.group(1)) * 60
        m = re.search(r"(\d+)\s*(hour|hr|h)\b", lower)
        if m:
            return int(m.group(1)) * 3600
        m = re.search(r"(\d+)", lower)
        if m:
            return int(m.group(1))
        return None

    def _check_reminders(self):
        if not self.reminders:
            return
        now = time.time()
        remaining = []
        for r in self.reminders:
            if r.get("ts", 0) <= now:
                self.tts_queue.put(f"Reminder: {r.get('text', 'Task')}")
                self.holo_effects.set_search("REMINDER", [r.get("text", "Task")])
            else:
                remaining.append(r)
        if len(remaining) != len(self.reminders):
            self.reminders = remaining
            self._persist_reminders()

    def _safe_eval(self, expr):
        if not re.fullmatch(r"[0-9\s\+\-\*\/\(\)\.%]+", expr):
            return None
        try:
            return eval(expr, {"__builtins__": {}}, {})
        except Exception:
            return None

    def _weather_lookup(self, location=""):
        if not WEB_AVAILABLE:
            self.tts_queue.put("Weather lookup needs web access.")
            return
        loc = quote_plus(location.strip() or "")
        url = f"https://wttr.in/{loc}?format=j1"
        try:
            r = requests.get(url, timeout=5)
            data = r.json()
            cc = data.get("current_condition", [{}])[0]
            temp = cc.get("temp_C", "?")
            desc = cc.get("weatherDesc", [{"value": "unknown"}])[0].get("value", "unknown")
            text = f"Weather {location or 'here'}: {temp}°C and {desc}."
            self.holo_effects.set_search("WEATHER", [text])
            self.tts_queue.put(text)
        except Exception:
            self.tts_queue.put("Unable to fetch weather right now.")

    def _set_timer(self, text):
        secs = self._parse_seconds(text)
        if not secs or secs <= 0:
            self.tts_queue.put("Please specify a timer duration.")
            return
        item = {"text": f"Timer for {secs} seconds", "ts": time.time() + secs}
        self.reminders.append(item)
        self._persist_reminders()
        self.tts_queue.put(f"Timer set for {secs} seconds.")

    def _set_profile(self, text):
        m = re.search(r"call me\s+([a-zA-Z0-9 _-]{2,30})", text, flags=re.IGNORECASE)
        if not m:
            self.tts_queue.put("Tell me like: call me Alex.")
            return
        name = m.group(1).strip()
        self.profile["name"] = name
        self._persist_profile()
        self.tts_queue.put(f"Got it. I'll call you {name}.")

    def _get_user_name(self):
        return self.profile.get("name", "sir")

    def _remember_command(self, text):
        cleaned = text.strip()
        if cleaned:
            self.command_history.appendleft(cleaned)

    def _show_help_overlay(self):
        tips = [
            "Say: Jarvis what time is it",
            "Say: note buy groceries",
            "Say: show notes / clear notes",
            "Gesture eraser: local erase",
            "Gesture fist: clear holograms",
        ]
        self.holo_effects.set_search("JARVIS COMMANDS", tips)

    def _set_note(self, text):
        body = re.sub(r"^(note|remember|take note)\s+", "", text, flags=re.IGNORECASE).strip()
        if not body:
            self.tts_queue.put("Please tell me what to note.")
            return
        self.notes.appendleft(body)
        self._persist_notes()
        self.holo_effects.set_search("NOTE SAVED", [body])
        self.tts_queue.put("Noted.")

    def _show_notes(self):
        if not self.notes:
            self.tts_queue.put("No saved notes.")
            return
        top = list(self.notes)[:3]
        self.holo_effects.set_search("YOUR NOTES", top)
        self.tts_queue.put(f"You have {len(self.notes)} saved notes.")

    def _show_history(self):
        if not self.command_history:
            self.tts_queue.put("No recent commands yet.")
            return
        top = list(self.command_history)[:3]
        self.holo_effects.set_search("RECENT COMMANDS", top)
        self.tts_queue.put("Showing recent commands.")

    def _loop(self):
        while self.is_running:
            self._drain_queues()
            self._check_reminders()

    def _drain_queues(self):
        try:
            cmd_type, cmd_data = self.command_queue.get(timeout=0.1)
            if cmd_type == "voice_command":
                self._process_voice_command(cmd_data)
        except queue.Empty:
            pass
        
        try:
            vis_type, vis_data = self.vision_queue.get(timeout=0.05)
            if vis_type == "detection":
                # Removed annoying presence announcements
                pass
            elif vis_type == "gesture":
                self._process_gesture(vis_data)
            elif vis_type == "trigger":
                if vis_data == "two_hands" and self._can_speak("two_hands", 5.0):
                    self.tts_queue.put("Hand tracking active.")
            elif vis_type == "focus":
                if self._can_speak("focus", 2.5):
                    self.tts_queue.put(f"Target identified: {vis_data}.")
        except queue.Empty:
            pass

    def _can_speak(self, key, cooldown=2.0):
        now = time.time()
        last = self.last_feedback.get(key, 0.0)
        if now - last >= cooldown:
            self.last_feedback[key] = now
            return True
        return False

    def _process_gesture(self, gesture):
        if gesture == "fist":
            self.holo_effects.clear_holograms()
            self.holo_effects.set_search("Gestures", ["Fist: holograms cleared"])
            if self._can_speak("fist", 3.0):
                self.tts_queue.put("Gesture recognized. Clearing holograms.")
        elif gesture == "peace":
            cpu = psutil.cpu_percent()
            ram = psutil.virtual_memory().percent
            if self._can_speak("peace", 3.0):
                self.tts_queue.put(f"System status. CPU {cpu:.0f} percent. Memory {ram:.0f} percent.")
        elif gesture == "eraser":
            if self._can_speak("eraser", 2.5):
                self.tts_queue.put("Eraser gesture active.")

    def _is_conversational(self, lower):
        starters = (
            "how are you", "what are you doing", "tell me", "can you", "could you",
            "do you", "thanks", "thank you", "who are you", "i feel", "i am ",
        )
        return lower.startswith(starters) or lower in ("hello", "hi", "hey")

    def _chat_response(self, text):
        lower = text.lower().strip()
        if "how are you" in lower:
            return "I'm doing well. I'm here with you and ready to help with anything you need."
        if "who are you" in lower:
            return "I'm Jarvis, your local AI assistant. I can chat, search the web, and control the HUD."
        if "thank" in lower:
            return "You're welcome. Happy to help."
        if lower.startswith("i feel") or lower.startswith("i am "):
            return "Thanks for sharing that. Want me to help you with something practical right now?"
        if "what can you do" in lower or "help" in lower:
            return "I can chat naturally, run web searches, manage notes, and use hand gestures for HUD controls."
        return "Got it. Tell me what you'd like to do, and I'll help step by step."

    def _process_voice_command(self, text):
        lower = text.lower().strip()
        self._remember_command(text)
        intent = self.intent_matcher.match(text)

        if intent == "greeting":
            self.tts_queue.put(f"Hey {self._get_user_name()}, good to hear you. How can I help right now?")
        elif intent == "help" or "help" in lower:
            self._show_help_overlay()
            self.tts_queue.put("Displaying available commands.")
        elif intent == "note" or lower.startswith("note ") or lower.startswith("remember ") or lower.startswith("take note"):
            self._set_note(text)
        elif "show notes" in lower or "list notes" in lower:
            self._show_notes()
        elif "clear notes" in lower:
            self.notes.clear()
            self._persist_notes()
            self.holo_effects.set_search("NOTES", ["All notes cleared"])
            self.tts_queue.put("Notes cleared.")
        elif intent == "history" or "recent commands" in lower:
            self._show_history()
        elif intent == "profile" or "call me" in lower:
            if "call me" in lower:
                self._set_profile(text)
            else:
                self.tts_queue.put(f"You are {self._get_user_name()}.")
        elif intent == "timer" or "set timer" in lower:
            self._set_timer(text)
        elif intent == "reminder" or lower.startswith("remind me"):
            secs = self._parse_seconds(text)
            msg = re.sub(r"^remind me", "", text, flags=re.IGNORECASE).strip()
            if secs and msg:
                self.reminders.append({"text": msg, "ts": time.time() + secs})
                self._persist_reminders()
                self.tts_queue.put("Reminder saved.")
            else:
                self.tts_queue.put("Say: remind me in 10 minutes to stretch.")
        elif intent == "weather" or "weather" in lower:
            loc = re.sub(r"^.*weather( in)?", "", text, flags=re.IGNORECASE).strip()
            self._weather_lookup(loc)
        elif intent == "math" or lower.startswith("calculate") or lower.startswith("what is"):
            expr = re.sub(r"^(calculate|what is)\s+", "", text, flags=re.IGNORECASE).strip()
            ans = self._safe_eval(expr)
            if ans is None:
                self.tts_queue.put("I couldn't calculate that expression.")
            else:
                self.tts_queue.put(f"The answer is {ans}.")
        elif intent == "hologram" or "toggle hologram" in lower or "toggle hud" in lower:
            Config.HOLO_ENABLED = not Config.HOLO_ENABLED
            state = "enabled" if Config.HOLO_ENABLED else "disabled"
            self.tts_queue.put(f"Holographic overlay {state}.")
        elif intent == "time":
            self.tts_queue.put(f"The time is {datetime.now().strftime('%I:%M %p')}, sir.")
        elif intent == "date":
            self.tts_queue.put(f"Today is {datetime.now().strftime('%B %d, %Y')}, sir.")
        elif intent == "status":
            self.tts_queue.put(f"CPU at {psutil.cpu_percent():.0f} percent, memory at {psutil.virtual_memory().percent:.0f} percent.")
        elif intent == "exit":
            self.tts_queue.put("Shutting down, sir.")
            Config.SHUTDOWN_FLAG = True
        elif "wikipedia" in lower or intent == "wikipedia":
            query = re.sub(r"(wikipedia|wiki|tell me about)\s+", "", text, flags=re.IGNORECASE).strip()
            if query:
                self._wikipedia_search(query)
        elif "youtube" in lower or intent == "youtube":
            query = re.sub(r"(youtube|video|play video)\s+", "", text, flags=re.IGNORECASE).strip()
            webbrowser.open(f"https://www.youtube.com/results?search_query={quote_plus(query or text)}")
            self.tts_queue.put("Opening YouTube.")
        elif "play music" in lower or intent == "music":
            webbrowser.open("https://music.youtube.com")
            self.tts_queue.put("Opening music player.")
        elif intent == "open":
            self._open_application(lower)
        elif intent == "vision":
            self.tts_queue.put("Vision systems are active.")
        elif self._is_conversational(lower):
            self.tts_queue.put(self._chat_response(text))
        else:
            query = re.sub(r"^(search for|search|look up|find|google)\s+", "", text, flags=re.IGNORECASE).strip()
            if len(query) > 2:
                self.last_topics.appendleft(query)
                self._web_search(query)

    def _open_application(self, lower_text):
        if "calculator" in lower_text:
            cmd = "calc.exe" if os.name == "nt" else "gnome-calculator"
            subprocess.Popen(cmd, shell=(os.name != "nt"))
            self.tts_queue.put("Opening calculator.")
        elif "notepad" in lower_text or "editor" in lower_text:
            cmd = "notepad.exe" if os.name == "nt" else "gedit"
            subprocess.Popen(cmd, shell=(os.name != "nt"))
            self.tts_queue.put("Opening text editor.")
        else:
            webbrowser.open("https://www.google.com")
            self.tts_queue.put("Opening browser.")

    def _web_search(self, query):
        if not WEB_AVAILABLE:
            self.tts_queue.put("Web capabilities are offline.")
            return
        self.holo_effects.set_search(query)
        url = "https://www.google.com/search"
        try:
            response = requests.get(
                url,
                params={"q": query, "hl": "en"},
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                timeout=6,
            )
            soup = BeautifulSoup(response.text, "html.parser")

            titles = [h3.get_text(" ", strip=True) for h3 in soup.find_all("h3") if h3.get_text(strip=True)]
            snippets = [
                d.get_text(" ", strip=True)
                for d in soup.select("div.BNeawe.s3v9rd.AP7Wnd, div.VwiC3b")
                if d.get_text(strip=True)
            ]

            merged = []
            for i, title in enumerate(titles[:5]):
                snippet = snippets[i] if i < len(snippets) else ""
                line = f"{title} — {snippet[:120]}" if snippet else title
                merged.append(line)

            self.holo_effects.set_search(query, merged)
            if merged:
                self.tts_queue.put(f"I found this: {merged[0][:140]}")
            else:
                self.tts_queue.put("I couldn't extract useful results. Try rephrasing your query.")
        except Exception:
            self.tts_queue.put("Search failed. Connection may be unavailable.")

    def _wikipedia_search(self, query):
        if not WEB_AVAILABLE:
            self.tts_queue.put("Web capabilities are offline.")
            return
        self.holo_effects.set_search(f"Wikipedia: {query}")
        try:
            url = f"https://en.wikipedia.org/wiki/{quote_plus(query).replace('+', '_')}"
            response = requests.get(url, timeout=5)
            soup = BeautifulSoup(response.text, "html.parser")
            for p in soup.find_all("p"):
                txt = p.get_text().strip()
                if len(txt) > 100:
                    summary = ". ".join(txt.split(".")[:2]).strip() + "."
                    self.holo_effects.set_search(f"Wikipedia: {query}", [summary])
                    self.tts_queue.put(summary)
                    return
            self.tts_queue.put("No information found.")
        except Exception:
            self.tts_queue.put("Wikipedia access failed.")


class JarvisUI:
    def __init__(self, vision_engine, holo_effects, voice_overlay_state):
        self.vision_engine = vision_engine
        self.holo_effects = holo_effects
        self.voice_overlay_state = voice_overlay_state
        self.is_running = False
        self.show_commands = False
        self.nightvision_enabled = Config.NIGHTVISION_DEFAULT
        self.startup_started_at = time.time()
        self.startup_effect_seconds = 4.0

    def start(self):
        self.is_running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self.is_running = False

    def _draw_rounded_rect(self, image, pt1, pt2, color, radius=12, thickness=-1):
        x1, y1 = pt1
        x2, y2 = pt2
        cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
        cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
        for cx, cy in ((x1 + radius, y1 + radius), (x2 - radius, y1 + radius), (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)):
            cv2.circle(image, (cx, cy), radius, color, thickness)

    def _draw_voice_box(self, overlay, h, w):
        state = self.voice_overlay_state.snapshot()
        box_w, box_h = 350, 96
        x1, y1 = 18, h - box_h - 20
        x2, y2 = x1 + box_w, y1 + box_h
        panel = overlay.copy()
        self._draw_rounded_rect(panel, (x1, y1), (x2, y2), (20, 20, 30), radius=14, thickness=-1)
        cv2.addWeighted(panel, 0.65, overlay, 0.35, 0, overlay)
        self._draw_rounded_rect(overlay, (x1, y1), (x2, y2), (255, 255, 0), radius=14, thickness=2)

        wake_text = "WAKE: LISTENING" if state["wake_active"] else "WAKE: STANDBY"
        wake_color = (0, 255, 255) if state["wake_active"] else (130, 160, 160)
        cv2.putText(overlay, wake_text, (x1 + 12, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, wake_color, 1)

        transcript = state["last_partial"] or state["last_final"] or "(say 'jarvis' to activate)"
        transcript = transcript[:40] + "..." if len(transcript) > 40 else transcript
        cv2.putText(overlay, transcript, (x1 + 12, y1 + 53), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (220, 220, 220), 1)

        hand = self.vision_engine.hand_state
        cv2.putText(overlay, f"mode:{hand.get('mode', 'none').upper()} sign:{hand.get('gesture', 'none').upper()}", (x1 + 12, y1 + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 240, 240), 1)
        nv = "ON" if self.nightvision_enabled else "OFF"
        cv2.putText(overlay, f"N-nightvision:{nv}", (x1 + 205, y1 + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (130, 230, 130), 1)

    def _apply_nightvision(self, display):
        if not self.nightvision_enabled:
            return display
        gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
        eq = cv2.equalizeHist(gray)
        green = cv2.applyColorMap(eq, cv2.COLORMAP_SUMMER)
        boosted = cv2.convertScaleAbs(green, alpha=1.15, beta=6)
        return cv2.addWeighted(boosted, 0.78, display, 0.22, 0)

    def _apply_startup_lidar_effect(self, frame):
        elapsed = time.time() - self.startup_started_at
        if elapsed >= self.startup_effect_seconds:
            return frame

        progress = max(0.0, min(1.0, elapsed / self.startup_effect_seconds))
        h, w = frame.shape[:2]

        cam_alpha = np.clip((progress - 0.42) / 0.58, 0.0, 1.0)
        base = cv2.convertScaleAbs(frame, alpha=cam_alpha, beta=0)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 45, 130)
        edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > 140]
        contours.sort(key=cv2.contourArea, reverse=True)

        lidar = np.zeros_like(frame)
        rng = np.random.default_rng(1337)
        sweep_y = int(progress * h)

        max_objs = min(45, len(contours))
        for i, c in enumerate(contours[:max_objs]):
            reveal = np.clip((progress - i * 0.012) * 2.4, 0.0, 1.0)
            if reveal <= 0:
                continue

            obj_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(obj_mask, [c], -1, 255, thickness=cv2.FILLED)
            ys, xs = np.where(obj_mask > 0)
            if len(xs) == 0:
                continue

            scanned = ys <= sweep_y
            xs, ys = xs[scanned], ys[scanned]
            if len(xs) == 0:
                continue

            sample_count = max(12, int(len(xs) * (0.006 + 0.022 * reveal)))
            pick = rng.choice(len(xs), size=min(sample_count, len(xs)), replace=False)
            px = xs[pick]
            py = ys[pick]

            depth = 1.0 - (py.astype(np.float32) / max(1, h - 1))
            intens = np.clip((0.35 + 0.65 * depth) * (0.35 + 0.65 * reveal), 0.0, 1.0)

            for x, y, a in zip(px, py, intens):
                color = (int(30 + 55 * a), int(120 + 120 * a), int(210 + 45 * a))
                r = 1 if a < 0.72 else 2
                cv2.circle(lidar, (int(x), int(y)), r, color, -1)

        dot_alpha = 0.95 * (1.0 - 0.18 * cam_alpha)
        composed = cv2.addWeighted(base, 1.0, lidar, dot_alpha, 0)
        cv2.line(composed, (0, sweep_y), (w, sweep_y), (170, 255, 255), 2)
        return composed

    def _draw_voice_globe(self, overlay, h, w):
        state = self.voice_overlay_state.snapshot()
        cx, cy = w - 120, 130
        t = time.time()
        base = 95
        level = state.get("speaking_level", 0.0)
        pulse = 8 * math.sin(t * 4.0) if state.get("speaking_active") else 3 * math.sin(t * 2.0)
        r = int(base + pulse + level * 20)
        for i in range(3):
            rr = r + i * 14
            alpha = max(60, 150 - i * 35)
            color = (255, alpha, 40)
            cv2.circle(overlay, (cx, cy), rr, color, 1)
        for a in np.linspace(0, 2 * math.pi, 120):
            wobble = 6 * math.sin(6 * a + t * 2.5)
            x = int(cx + (r + wobble) * math.cos(a))
            y = int(cy + (r + wobble) * math.sin(a))
            cv2.circle(overlay, (x, y), 1, (255, 220, 80), -1)
        cv2.circle(overlay, (cx, cy), 22, (255, 150, 40), 1)

    def _draw_commands_panel(self, overlay, h, w):
        if not self.show_commands:
            return
        x1, y1 = 24, 24
        x2, y2 = 420, 250
        panel = overlay.copy()
        self._draw_rounded_rect(panel, (x1, y1), (x2, y2), (18, 22, 30), radius=12, thickness=-1)
        cv2.addWeighted(panel, 0.68, overlay, 0.32, 0, overlay)
        self._draw_rounded_rect(overlay, (x1, y1), (x2, y2), (255, 255, 0), radius=12, thickness=2)
        lines = [
            "VOICE COMMANDS (press I to hide)",
            "- Jarvis what's the time",
            "- search latest AI news",
            "- note buy milk",
            "- show notes / clear notes",
            "- set timer 2 minutes",
            "- remind me in 10 minutes to stretch",
            "- weather in London",
            "- calculate (12+8)*3",
            "GESTURES: draw pinch | move pinch | eraser four-finger | fist clear",
        ]
        y = y1 + 24
        for i, line in enumerate(lines):
            scale = 0.42 if i else 0.48
            color = (255, 240, 100) if i == 0 else (200, 220, 220)
            cv2.putText(overlay, line, (x1 + 10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)
            y += 21

    def _draw_hud(self, frame):
        display = cv2.resize(frame, (Config.UI_WIDTH, Config.UI_HEIGHT))
        display = self._apply_nightvision(display)
        display = self._apply_startup_lidar_effect(display)
        overlay = display.copy()
        h, w = display.shape[:2]

        if Config.HOLO_ENABLED:
            self.holo_effects.update()
            self.holo_effects.update_hand_interaction(self.vision_engine.hand_state, w, h)
            self.holo_effects.clear_search()
            self.holo_effects.draw_hand_holograms(overlay)

        status_box = overlay.copy()
        self._draw_rounded_rect(status_box, (w - 350, 20), (w - 20, 170), (20, 20, 30), radius=12, thickness=-1)
        cv2.addWeighted(status_box, 0.62, overlay, 0.38, 0, overlay)
        self._draw_rounded_rect(overlay, (w - 350, 20), (w - 20, 170), (255, 255, 0), radius=12, thickness=2)

        cv2.putText(overlay, "J.A.R.V.I.S. HUD", (w - 330, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 0), 2)
        cv2.putText(overlay, f"FPS: {self.vision_engine.get_fps():.1f}", (w - 330, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (0, 255, 180), 1)
        cv2.putText(overlay, f"CPU: {psutil.cpu_percent():.1f}%", (w - 330, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (0, 255, 180), 1)
        cv2.putText(overlay, f"RAM: {psutil.virtual_memory().percent:.1f}%", (w - 330, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (0, 255, 180), 1)
        if self.vision_engine.use_mediapipe:
            cv2.putText(overlay, "Hands: ONLINE", (w - 330, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 180), 1)
        else:
            cv2.putText(overlay, "Hands: OFFLINE", (w - 330, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 120, 255), 1)
        if self.vision_engine.focus_object:
            focus_txt = f"Target: {self.vision_engine.focus_object['label']}"
            cv2.putText(overlay, focus_txt[:36], (w - 330, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 180, 0), 1)
        cv2.putText(overlay, datetime.now().strftime("%H:%M:%S"), (w - 165, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 255, 0), 2)

        self._draw_voice_globe(overlay, h, w)
        self._draw_voice_box(overlay, h, w)
        self._draw_commands_panel(overlay, h, w)
        return cv2.addWeighted(overlay, 0.95, display, 0.05, 0)

    def _loop(self):
        while self.is_running:
            frame = self.vision_engine.get_frame()
            if frame is None:
                frame = np.zeros((Config.CAMERA_HEIGHT, Config.CAMERA_WIDTH, 3), dtype=np.uint8)
            cv2.imshow("J.A.R.V.I.S.", self._draw_hud(frame))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("i"):
                self.show_commands = not self.show_commands
            if key == ord("n"):
                self.nightvision_enabled = not self.nightvision_enabled
            if key == ord("s"):
                DataStore().save_holograms(self.holo_effects.holograms)
            if key in (ord("q"), 27):
                Config.SHUTDOWN_FLAG = True
                break
            time.sleep(0.01)
        cv2.destroyAllWindows()


class Jarvis:
    def __init__(self):
        print("\n" + "="*60)
        print("J.A.R.V.I.S. - Just A Rather Very Intelligent System")
        print("="*60)
        
        self.command_queue = queue.Queue()
        self.tts_queue = queue.Queue()
        self.vision_queue = queue.Queue()
        self.voice_overlay_state = VoiceOverlayState()

        self.holo_effects = HolographicEffects()
        self.vision_engine = VisionEngine(self.vision_queue)
        self.voice_engine = VoiceEngine(self.command_queue, self.tts_queue, self.voice_overlay_state)
        self.tts_engine = TTSEngine(self.tts_queue, self.voice_overlay_state)
        self.command_engine = CommandEngine(self.command_queue, self.tts_queue, self.vision_queue, self.holo_effects)
        self.ui = JarvisUI(self.vision_engine, self.holo_effects, self.voice_overlay_state)
        self.store = DataStore()
        self.holo_effects.holograms = self.store.load_holograms()
        
        print("\n[INIT] System initialization complete")
        print(f"[INIT] Voice recognition: {'✅ Active' if VOSK_AVAILABLE and AUDIO_AVAILABLE else '❌ Offline'}")
        
        # Check TTS status (Piper or pyttsx3)
        tts_status = "❌ Offline"
        if self.tts_engine.use_piper:
            tts_status = "✅ Active (Piper - High Quality)"
        elif TTS_AVAILABLE:
            tts_status = "✅ Active (pyttsx3)"
        print(f"[INIT] Text-to-speech: {tts_status}")
        
        print(f"[INIT] Computer vision: {'✅ Active' if MEDIAPIPE_AVAILABLE else '⚠️  Limited (no hand tracking)'}")
        print(f"[INIT] Web access: {'✅ Active' if WEB_AVAILABLE else '❌ Offline'}")
        print("\n[TIPS] Say '{}' to activate voice commands".format(Config.WAKE_WORD.upper()))
        print("[TIPS] Press 'I' to show/hide command list")
        print("[TIPS] Press 'N' to toggle night vision")
        print("[TIPS] Press 'Q' or ESC to quit")
        print("="*60 + "\n")

    def start(self):
        print("[JARVIS] 🚀 Starting all subsystems...")
        self.vision_engine.start()
        time.sleep(0.5)
        self.voice_engine.start()
        self.tts_engine.start()
        self.command_engine.start()
        self.ui.start()
        self.tts_queue.put("Good evening, sir. All systems online.")
        
        while not Config.SHUTDOWN_FLAG:
            time.sleep(0.5)
        
        self.shutdown()

    def shutdown(self):
        Config.SHUTDOWN_FLAG = True
        self.store.save_holograms(self.holo_effects.holograms)
        self.store.save_face_data(self.vision_engine.face_data)
        self.voice_engine.stop()
        self.vision_engine.stop()
        self.tts_engine.stop()
        self.command_engine.stop()
        self.ui.stop()
        print("[JARVIS] All systems offline. Goodbye, sir.")


if __name__ == "__main__":
    try:
        Jarvis().start()
    except KeyboardInterrupt:
        print("\n[JARVIS] Interrupted by user")
    except Exception as e:
        print(f"\n[JARVIS] ❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[JARVIS] Shutdown complete.")
        sys.exit(0)
