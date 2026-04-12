"""Central Jarvis engine — always-on continuous conversation."""

from __future__ import annotations

import re
import threading
import time
from typing import Callable, Optional

import numpy as np

from . import config
from .audio import AudioEngine
from .llm import LLMEngine
from .memory import MemoryStore
from .monitor import ProactiveMonitor
from .stt import STTEngine
from .tools import ToolExecutor
from .tts import TTSEngine
from .vision import VisionEngine


class JarvisEngine:
    """Always-on brain. Listens continuously, responds naturally."""

    def __init__(
        self,
        on_status: Optional[Callable[[str], None]] = None,
        on_user_text: Optional[Callable[[str], None]] = None,
        on_assistant_token: Optional[Callable[[str], None]] = None,
        on_assistant_done: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        on_listening_state: Optional[Callable[[str], None]] = None,
        on_audio_level: Optional[Callable[[float], None]] = None,
        on_speaking: Optional[Callable[[bool], None]] = None,
        on_posture: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.on_status = on_status or (lambda s: print(f"[Status] {s}"))
        self.on_user_text = on_user_text or (lambda t: print(f"[You] {t}"))
        self.on_assistant_token = on_assistant_token or (lambda t: print(t, end="", flush=True))
        self.on_assistant_done = on_assistant_done or (lambda t: print())
        self.on_error = on_error or (lambda e: print(f"[Error] {e}"))
        self.on_listening_state = on_listening_state or (lambda s: None)
        self.on_audio_level = on_audio_level or (lambda l: None)
        self.on_speaking = on_speaking or (lambda b: None)
        self.on_posture = on_posture or (lambda s: None)

        self._processing = False
        self._last_proactive_message = 0.0
        self._last_posture_alert = 0.0

        self.on_status("Initializing J.A.R.V.I.S. …")

        self.memory = MemoryStore()

        # STT
        try:
            self.stt = STTEngine()
        except Exception as e:
            self.on_error(f"STT failed: {e}")
            self.stt = None

        # TTS
        try:
            self.tts = TTSEngine()
            self.tts.on_speak_start = self._on_tts_start
            self.tts.on_speak_end = self._on_tts_end
        except Exception as e:
            self.on_error(f"TTS failed: {e}")
            self.tts = None

        # LLM
        try:
            self.llm = LLMEngine()
        except Exception as e:
            self.on_error(f"LLM failed: {e}")
            self.llm = None

        # Vision
        try:
            self.vision = VisionEngine()
            # Wire posture callback — rate-limited in engine, not in vision
            self.vision._on_posture_change = self._on_vision_posture
            # Wire identify callback — fires when pointing gesture held
            self.vision._on_identify = self._on_vision_identify
            self.vision.start()
        except Exception as e:
            self.on_error(f"Vision failed: {e}")
            self.vision = None

        # Audio (always-on)
        self.audio = AudioEngine(
            on_speech_ready=self._on_speech_ready,
            on_level_update=self.on_audio_level,
            on_listening_state=self.on_listening_state,
        )

        self.monitor = ProactiveMonitor(
            project_root=config.PROJECT_ROOT,
            on_alert=self._on_proactive_alert,
        )
        if config.PROACTIVE_MONITOR_ENABLED:
            self.monitor.start()

        self.on_status("Online — speak anytime, sir")

    def start(self) -> None:
        self.audio.start()

    def stop(self) -> None:
        self.audio.stop()
        if self.vision:
            self.vision.stop()
        if self.tts:
            self.tts.stop()
        if self.monitor:
            self.monitor.stop()

    def _on_tts_start(self) -> None:
        self.audio.pause()
        self.on_speaking(True)

    def _on_tts_end(self) -> None:
        self.audio.resume()
        self.on_speaking(False)
        self.on_listening_state("idle")

    def _on_speech_ready(self, audio: np.ndarray) -> None:
        if self._processing:
            return
        threading.Thread(target=self._process_speech, args=(audio,), daemon=True).start()

    def _process_speech(self, audio: np.ndarray) -> None:
        if self.stt is None:
            self.on_error("STT not available")
            return

        self._processing = True
        try:
            self.on_status("Transcribing …")
            text = self.stt.transcribe(audio)

            if not text:
                self.on_status("Online — speak anytime, sir")
                self._processing = False
                self.on_listening_state("idle")
                return

            self.process_text(text)
        except Exception as e:
            self.on_error(f"Speech error: {e}")
        finally:
            self._processing = False

    def process_text(self, text: str) -> None:
        """Process user input → respond with text + voice."""
        self._processing = True
        self.on_user_text(text)
        self.memory.update_preferences_from_text(text)

        # ── Quick commands ──────────────────────────────────────
        quick = ToolExecutor.quick_match(text)
        if quick:
            self.on_assistant_token(quick)
            self.on_assistant_done(quick)
            if self.tts:
                threading.Thread(target=self.tts.speak, args=(quick,), daemon=True).start()
            self.on_status("Online — speak anytime, sir")
            self._processing = False
            return

        low = text.strip().lower()
        if "how many fingers" in low or "count my fingers" in low:
            if self.vision is None or not self.vision.is_available:
                reply = "I cannot see your hand right now, sir. Camera is unavailable."
            else:
                count = self.vision.get_finger_count()
                reply = (
                    f"I can see approximately {count} finger{'s' if count != 1 else ''}, sir."
                    if count is not None
                    else "I cannot confidently detect your fingers yet. Hold your hand in better light."
                )
            self.on_assistant_token(reply)
            self.on_assistant_done(reply)
            if self.tts:
                threading.Thread(target=self.tts.speak, args=(reply,), daemon=True).start()
            self.on_status("Online — speak anytime, sir")
            self._processing = False
            return

        if "what have i drawn" in low or "what did i draw" in low or "analyze drawing" in low:
            if self.vision is None or not self.vision.is_available:
                reply = "I cannot analyze your air drawing because the camera is offline."
            else:
                reply = self.vision.summarize_airdraw()
            self.on_assistant_token(reply)
            self.on_assistant_done(reply)
            if self.tts:
                threading.Thread(target=self.tts.speak, args=(reply,), daemon=True).start()
            self.on_status("Online — speak anytime, sir")
            self._processing = False
            return

        # ── Identify object ──────────────────────────────────────────
        if any(p in low for p in ("identify this", "what is this", "identify object", "scan this")):
            if self.vision is None or not self.vision.is_available:
                reply = "Camera is not available for identification, sir."
            else:
                self.on_status("Identifying object …")
                reply = self.vision.identify_object()
            self.on_assistant_token(reply)
            self.on_assistant_done(reply)
            if self.tts:
                threading.Thread(target=self.tts.speak, args=(reply,), daemon=True).start()
            self.on_status("Online — speak anytime, sir")
            self._processing = False
            return

        if "enable air draw" in low or "start air draw" in low:
            if self.vision is not None:
                self.vision.enable_airdraw(True)
            reply = "Air drawing enabled, sir. Raise one finger to draw in mid-air."
            self.on_assistant_token(reply)
            self.on_assistant_done(reply)
            if self.tts:
                threading.Thread(target=self.tts.speak, args=(reply,), daemon=True).start()
            self.on_status("Online — speak anytime, sir")
            self._processing = False
            return

        if "disable air draw" in low or "stop air draw" in low:
            if self.vision is not None:
                self.vision.enable_airdraw(False)
            reply = "Air drawing disabled."
            self.on_assistant_token(reply)
            self.on_assistant_done(reply)
            if self.tts:
                threading.Thread(target=self.tts.speak, args=(reply,), daemon=True).start()
            self.on_status("Online — speak anytime, sir")
            self._processing = False
            return

        if "clear drawing" in low or "clear air draw" in low:
            if self.vision is not None:
                self.vision.clear_airdraw()
            reply = "Air drawing cleared, sir."
            self.on_assistant_token(reply)
            self.on_assistant_done(reply)
            if self.tts:
                threading.Thread(target=self.tts.speak, args=(reply,), daemon=True).start()
            self.on_status("Online — speak anytime, sir")
            self._processing = False
            return

        # ── LLM ─────────────────────────────────────────────────
        if self.llm is None:
            self.on_error("Ollama not running. Start it with: ollama serve")
            self._processing = False
            return

        self.on_status("Thinking …")

        # Vision context — only include if there's something INTERESTING
        vision_context = None
        if self.vision and self.vision.is_available:
            ctx = self.vision.get_detection_context()
            if ctx:  # Only if non-empty (boring stuff is filtered out)
                vision_context = ctx

        preference_context = self.memory.get_preferences_text()

        # Stream tokens to screen
        full_response = ""
        try:
            for token in self.llm.stream_chat(
                text,
                vision_context=vision_context,
                preference_context=preference_context,
            ):
                full_response += token
                self.on_assistant_token(token)
        except Exception as e:
            self.on_error(f"LLM error: {e}")
            self._processing = False
            return

        self.on_assistant_done(full_response)

        # ── Tool calls — match BOTH formats the LLM might use ──
        tool_calls = self._extract_tools(full_response)
        if tool_calls:
            for call in tool_calls:
                self.on_status(f"Executing: {call['function']}()")
                result = ToolExecutor.execute(call["function"], call["args"])
                self.on_assistant_token(f"\n\n📋 {result}")
                self.on_assistant_done(result)

        # ── Speak (remove tool markers) ─────────────────────────
        if self.tts:
            spoken_text = re.sub(r"\[TOOL(?:_CALL)?:\s*\w+\(.*?\)\]", "", full_response, flags=re.DOTALL).strip()
            if spoken_text:
                threading.Thread(target=self.tts.speak, args=(spoken_text,), daemon=True).start()

        self.on_status("Online — speak anytime, sir")
        self._processing = False

    def _on_proactive_alert(self, text: str) -> None:
        """Emit proactive assistant suggestions without interrupting active requests."""
        # Rate limit proactive chatter.
        now = time.time()
        if now - self._last_proactive_message < 20:
            return
        self._last_proactive_message = now

        self.on_assistant_token(f"\n\n[Proactive] {text}")
        self.on_assistant_done(text)
        if self.tts and not self._processing:
            threading.Thread(target=self.tts.speak, args=(text,), daemon=True).start()

    @staticmethod
    def _extract_tools(text: str) -> list[dict]:
        """Extract tool calls from LLM output — handles multiple formats.
        
        Matches:
          [TOOL: func(param="val")]
          [TOOL_CALL: func(param="val")]
        """
        # Try both formats
        patterns = [
            r'\[TOOL:\s*(\w+)\((.*?)\)\]',
            r'\[TOOL_CALL:\s*(\w+)\((.*?)\)\]',
        ]
        calls = []
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                func = match.group(1)
                args_str = match.group(2)
                args = {}
                for am in re.finditer(r'(\w+)\s*=\s*"((?:[^"\\]|\\.)*)"', args_str):
                    args[am.group(1)] = am.group(2)
                calls.append({"function": func, "args": args, "raw": match.group(0)})
        return calls

    # ── Posture alert (from vision thread) ───────────────────────────

    def _on_vision_posture(self, state: str) -> None:
        """Fired by VisionEngine when posture changes. Rate-limited here."""
        now = time.time()
        if state == "good":
            return
        if now - self._last_posture_alert < config.POSTURE_ALERT_COOLDOWN:
            return
        self._last_posture_alert = now

        # Forward to GUI
        self.on_posture(state)

        # Speak a brief reminder
        msgs = {
            "hunched": "Sir, I've noticed your shoulders have dropped. Sit up straight.",
            "tilted": "Sir, your posture is off — your shoulders are uneven.",
        }
        reminder = msgs.get(state, "Sir, please check your posture.")
        if self.tts and not self._processing:
            threading.Thread(target=self.tts.speak, args=(reminder,), daemon=True).start()

    # ── Identify gesture callback (from vision thread) ────────────────

    def _on_vision_identify(self, result: str) -> None:
        """Fired by VisionEngine when the pointing gesture triggers identification."""
        self.on_assistant_token(f"\n\n[Identify] {result}")
        self.on_assistant_done(result)
        if self.tts and not self._processing:
            threading.Thread(
                target=self.tts.speak,
                args=(f"Object identified: {result}",),
                daemon=True,
            ).start()

    def get_camera_frame(self):
        if self.vision:
            try:
                return self.vision.get_annotated_frame()
            except Exception:
                return None
        return None

    def describe_scene(self) -> str:
        if self.vision is None:
            return "Camera not available."
        self.on_status("Analyzing scene …")
        result = self.vision.describe_scene()
        self.on_status("Online — speak anytime, sir")
        return result

    def set_airdraw_enabled(self, enabled: bool) -> None:
        if self.vision is not None:
            self.vision.enable_airdraw(enabled)

    def clear_airdraw(self) -> None:
        if self.vision is not None:
            self.vision.clear_airdraw()

    def summarize_airdraw(self) -> str:
        if self.vision is None:
            return "Camera not available."
        return self.vision.summarize_airdraw()

    def get_finger_count(self) -> Optional[int]:
        if self.vision is None:
            return None
        return self.vision.get_finger_count()
