"""Speech-to-Text engine — understands English & Urdu."""

from __future__ import annotations

import numpy as np
from faster_whisper import WhisperModel

from . import config


class STTEngine:
    """Transcribes audio to text. Auto-detects language."""

    def __init__(self) -> None:
        print("[STT] Loading Whisper model …")
        self._model = WhisperModel(
            config.WHISPER_MODEL,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE,
        )
        print(f"[STT] Ready  ({config.WHISPER_MODEL}, {config.WHISPER_DEVICE})")

    # Common Whisper hallucinations to filter out completely
    _HALLUCINATIONS = {
        "thank you", "thanks for watching", "thanks for listening",
        "please subscribe", "like and subscribe", "see you next time",
        "bye", "bye bye", "goodbye", "you", "okay", "ok",
        "um", "uh", "hmm", "ah", "oh", "yeah", "yes", "no",
        ".", ",", "!", "?", "...", "…",
        "subtitles by", "transcribed by", "caption", "captions",
        "sub", "subs", "subtitle", "subtitles",
    }

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe float32 mono 16kHz audio → text string."""
        if audio.size == 0:
            return ""

        # Normalise to float32
        if audio.dtype != np.float32:
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            else:
                audio = audio.astype(np.float32)

        # Clip to valid range
        audio = np.clip(audio, -1.0, 1.0)

        segments, info = self._model.transcribe(
            audio,
            beam_size=5,
            best_of=5,
            temperature=0.0,
            vad_filter=True,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            vad_parameters=dict(
                min_silence_duration_ms=600,
                speech_pad_ms=300,
                threshold=0.45,
            ),
        )

        lang = info.language if info.language else "en"

        parts = []
        for seg in segments:
            text = seg.text.strip()
            # Skip segments with very low confidence
            if not text:
                continue
            # Skip if no_speech probability is high
            if getattr(seg, "no_speech_prob", 0.0) > 0.7:
                continue
            # Skip obvious hallucinations
            if text.lower().strip(".,!? ") in self._HALLUCINATIONS:
                continue
            parts.append(text)

        result = " ".join(parts).strip()

        # Reject if too short or pure punctuation/whitespace
        stripped = result.strip(".,!?… \t\u200b\u200c\u200d")
        if len(stripped) < 2:
            return ""

        # Reject common single-word hallucinations
        if stripped.lower() in self._HALLUCINATIONS:
            return ""

        print(f"[STT] [{lang}] {result[:80]}{'…' if len(result) > 80 else ''}")
        return result
