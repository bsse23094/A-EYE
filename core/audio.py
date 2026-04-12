"""Audio engine — ALWAYS-ON continuous listening with VAD. No buttons needed."""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

from . import config


class AudioEngine:
    """Always-on microphone with voice activity detection.
    
    Tony Stark doesn't press buttons — Jarvis always listens.
    Speech is automatically captured and routed to STT.
    """

    def __init__(
        self,
        on_speech_ready: Optional[Callable[[np.ndarray], None]] = None,
        on_level_update: Optional[Callable[[float], None]] = None,
        on_listening_state: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.on_speech_ready = on_speech_ready
        self.on_level_update = on_level_update
        self.on_listening_state = on_listening_state  # "idle" | "listening" | "processing"

        self._running = False
        self._stream: Optional[sd.InputStream] = None
        self._paused = False  # Pause during TTS playback to avoid feedback

        # Speech buffer
        self._speech_buffer: list[np.ndarray] = []
        self._is_speaking = False
        self._silence_start: float = 0.0
        self._speech_start: float = 0.0

        # Audio level history for waveform display
        self._level_history: deque[float] = deque(maxlen=100)

        print("[Audio] Always-on engine initialized")

    @property
    def is_active(self) -> bool:
        return self._is_speaking

    def start(self) -> None:
        """Start the always-on microphone stream."""
        if self._running:
            return

        self._running = True
        try:
            self._stream = sd.InputStream(
                channels=config.CHANNELS,
                samplerate=config.SAMPLE_RATE,
                dtype="float32",
                blocksize=config.AUDIO_BLOCK_SIZE,
                callback=self._audio_callback,
            )
            self._stream.start()
            print("[Audio] Always-on microphone active — just speak!")
        except Exception as e:
            print(f"[Audio] Failed to start microphone: {e}")
            self._running = False

    def stop(self) -> None:
        """Stop the microphone stream."""
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def pause(self) -> None:
        """Pause listening (during TTS playback to avoid echo)."""
        self._paused = True
        self._is_speaking = False
        self._speech_buffer.clear()

    def resume(self) -> None:
        """Resume listening after TTS finishes."""
        self._paused = False

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """Called by sounddevice for each audio block. Always running."""
        if not self._running:
            return

        audio = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        rms = float(np.sqrt(np.mean(audio ** 2)))

        # Update waveform display
        self._level_history.append(rms)
        if self.on_level_update:
            try:
                self.on_level_update(rms)
            except Exception:
                pass

        # Skip processing if paused (Jarvis is talking)
        if self._paused:
            return

        now = time.time()
        self._process_speech(audio, rms, now)

    def _process_speech(self, audio: np.ndarray, rms: float, now: float) -> None:
        """Continuous VAD — automatically detects speech start/stop."""
        if rms > config.VAD_ENERGY_THRESHOLD:
            # Speech detected
            if not self._is_speaking:
                self._is_speaking = True
                self._speech_start = now
                if self.on_listening_state:
                    self.on_listening_state("listening")

            self._speech_buffer.append(audio.copy())
            self._silence_start = 0.0

        elif self._is_speaking:
            # Still append during short pauses (natural speech gaps)
            self._speech_buffer.append(audio.copy())

            if self._silence_start == 0.0:
                self._silence_start = now
            elif now - self._silence_start > config.VAD_SILENCE_TIMEOUT:
                # Silence timeout — finalize utterance
                speech_duration = now - self._speech_start

                if speech_duration >= config.VAD_MIN_SPEECH_DURATION and self._speech_buffer:
                    audio_data = np.concatenate(self._speech_buffer)

                    if self.on_listening_state:
                        self.on_listening_state("processing")

                    if self.on_speech_ready:
                        try:
                            self.on_speech_ready(audio_data)
                        except Exception as e:
                            print(f"[Audio] Speech callback error: {e}")

                # Reset for next utterance
                self._is_speaking = False
                self._speech_buffer.clear()
                self._silence_start = 0.0

    def get_level_history(self) -> list[float]:
        """Get recent audio level samples for visualization."""
        return list(self._level_history)
