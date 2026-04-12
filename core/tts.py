"""Text-to-Speech — smooth JARVIS voice, single continuous audio."""

from __future__ import annotations

import asyncio
import os
import queue
import re
import threading
import time
from typing import Callable, Optional

from . import config


class TTSEngine:
    """JARVIS voice — always English, deep British tone."""

    def __init__(self) -> None:
        self.backend = "none"
        self._stop_flag = False
        self._speaking = False
        self._play_lock = threading.Lock()
        self._stream_queue: queue.Queue[Optional[str]] = queue.Queue()
        self._stream_thread: Optional[threading.Thread] = None
        self._stream_buffer = ""
        self._stream_mode = False
        self._stream_lock = threading.Lock()

        self.on_speak_start: Optional[Callable] = None
        self.on_speak_end: Optional[Callable] = None

        os.makedirs(config.TTS_TEMP_DIR, exist_ok=True)

        # Init pygame mixer once
        try:
            import pygame
            pygame.mixer.init(frequency=24000)
        except Exception as e:
            print(f"[TTS] Pygame mixer warning: {e}")

        try:
            import edge_tts  # noqa: F401
            self.backend = "edge-tts"
            print("[TTS] Backend: edge-tts (JARVIS voice)")
        except ImportError:
            try:
                import pyttsx3
                self._pyttsx = pyttsx3.init()
                self._pyttsx.setProperty("rate", 160)
                self.backend = "pyttsx3"
                print("[TTS] Backend: pyttsx3 (offline)")
            except Exception:
                pass

        if self.backend == "none":
            print("[TTS] WARNING: No TTS backend!")

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    def stop(self) -> None:
        self._stop_flag = True
        self._stream_mode = False
        try:
            self._stream_queue.put_nowait(None)
        except Exception:
            pass
        try:
            import pygame
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
        except Exception:
            pass

    def begin_stream(self) -> None:
        """Begin a streaming TTS session for incremental token playback."""
        if self.backend == "none":
            return

        with self._stream_lock:
            if self._stream_mode:
                return
            self._stream_mode = True
            self._stream_buffer = ""
            self._stop_flag = False
            self._speaking = True
            if self.on_speak_start:
                self.on_speak_start()
            self._stream_thread = threading.Thread(target=self._stream_worker, daemon=True)
            self._stream_thread.start()

    def push_stream_token(self, token: str) -> None:
        """Push streamed text tokens; queues sentence-like chunks for playback."""
        if not token or not self._stream_mode:
            return
        with self._stream_lock:
            self._stream_buffer += token
            chunks = self._extract_ready_chunks(self._stream_buffer)
            if chunks:
                consumed = "".join(chunks)
                self._stream_buffer = self._stream_buffer[len(consumed):]
                for chunk in chunks:
                    clean = chunk.strip()
                    if clean:
                        self._stream_queue.put(clean)

    def end_stream(self) -> None:
        """Finish streaming session and speak any trailing text."""
        if not self._stream_mode:
            return
        with self._stream_lock:
            tail = self._stream_buffer.strip()
            self._stream_buffer = ""
            if tail:
                self._stream_queue.put(tail)
            self._stream_queue.put(None)

    @staticmethod
    def _extract_ready_chunks(text: str) -> list[str]:
        chunks = []
        last_end = 0
        for m in re.finditer(r"[^.!?\n]+[.!?\n]", text):
            part = m.group(0)
            if len(part.strip()) >= 6:
                chunks.append(part)
                last_end = m.end()

        if not chunks and len(text) >= config.TTS_STREAMING_MIN_CHARS:
            split_at = text.rfind(" ")
            if split_at > 20:
                chunks.append(text[:split_at])

        return chunks

    def _stream_worker(self) -> None:
        try:
            while True:
                piece = self._stream_queue.get()
                if piece is None:
                    break
                if self._stop_flag:
                    continue
                if self.backend == "edge-tts":
                    self._speak_edge(piece)
                elif self.backend == "pyttsx3":
                    self._speak_pyttsx(piece)
        finally:
            self._stream_mode = False
            self._speaking = False
            if self.on_speak_end:
                self.on_speak_end()

    def speak(self, text: str) -> None:
        """Speak full text as one smooth audio. Blocking."""
        text = text.strip()
        if not text or self.backend == "none":
            return

        self._stop_flag = False
        self._speaking = True

        if self.on_speak_start:
            self.on_speak_start()

        try:
            if self.backend == "edge-tts":
                self._speak_edge(text)
            elif self.backend == "pyttsx3":
                self._speak_pyttsx(text)
        except Exception as e:
            print(f"[TTS] Error: {e}")
        finally:
            self._speaking = False
            if self.on_speak_end:
                self.on_speak_end()

    def _speak_edge(self, text: str) -> None:
        import edge_tts

        tmp = os.path.join(config.TTS_TEMP_DIR, f"j_{int(time.time()*1000)}.mp3")

        async def _gen():
            c = edge_tts.Communicate(text, voice=config.TTS_VOICE, rate=config.TTS_RATE, pitch=config.TTS_PITCH)
            await c.save(tmp)

        try:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(_gen())
            loop.close()
        except Exception as e:
            print(f"[TTS] Generation failed: {e}")
            return

        if self._stop_flag:
            self._rm(tmp)
            return

        self._play(tmp)

    def _play(self, path: str) -> None:
        with self._play_lock:
            try:
                import pygame
                if not pygame.mixer.get_init():
                    pygame.mixer.init(frequency=24000)
                pygame.mixer.music.load(path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    if self._stop_flag:
                        pygame.mixer.music.stop()
                        break
                    time.sleep(0.02)
            except Exception:
                try:
                    import soundfile as sf
                    import sounddevice as sd
                    data, sr = sf.read(path)
                    if not self._stop_flag:
                        sd.play(data, sr)
                        sd.wait()
                except Exception:
                    pass
            finally:
                self._rm(path)

    def _speak_pyttsx(self, text: str) -> None:
        try:
            import pyttsx3
            if not hasattr(self, "_pyttsx") or self._pyttsx is None:
                self._pyttsx = pyttsx3.init()
                self._pyttsx.setProperty("rate", 160)
            self._pyttsx.say(text)
            self._pyttsx.runAndWait()
        except Exception as e:
            print(f"[TTS] pyttsx3 error: {e}")

    @staticmethod
    def _rm(p: str) -> None:
        try:
            if os.path.exists(p):
                os.remove(p)
        except OSError:
            pass
