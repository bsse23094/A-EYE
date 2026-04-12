"""LLM engine — Ollama chat with streaming, conversation memory, and tool awareness."""

from __future__ import annotations

import json
import re
from typing import Generator, List, Optional, Dict

import httpx

from . import config


class LLMEngine:
    """Manages conversation with the local Ollama LLM."""

    def __init__(self) -> None:
        self.base_url = config.OLLAMA_URL.rstrip("/")
        self.history: List[Dict[str, str]] = []
        self.model_name = config.LLM_MODEL
        self._verify_connection()

    def _verify_connection(self) -> None:
        """Check Ollama is running."""
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{self.base_url}/api/tags")
                resp.raise_for_status()
                models = [m["name"] for m in resp.json().get("models", [])]
                print(f"[LLM] Ollama connected. Models: {', '.join(models[:5])}")

                # Resolve runtime model to something actually installed.
                if config.LLM_MODEL in models:
                    self.model_name = config.LLM_MODEL
                elif config.LLM_FALLBACK in models:
                    self.model_name = config.LLM_FALLBACK
                    print(f"[LLM] Using fallback model: {self.model_name}")
                else:
                    preferred_base = config.LLM_MODEL.split(":")[0]
                    by_base = next((m for m in models if m.split(":")[0] == preferred_base), None)
                    if by_base:
                        self.model_name = by_base
                        print(f"[LLM] Using closest installed model: {self.model_name}")
                    elif models:
                        self.model_name = models[0]
                        print(f"[LLM] Using first available model: {self.model_name}")
                    else:
                        print("[LLM] WARNING: No Ollama models installed.")
        except Exception as e:
            print(f"[LLM] WARNING: Cannot connect to Ollama at {self.base_url}: {e}")
            print("[LLM] Make sure Ollama is running: `ollama serve`")

    def stream_chat(
        self,
        user_text: str,
        vision_context: Optional[str] = None,
        preference_context: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Stream a response from the LLM. Yields tokens as they arrive."""
        prompt = user_text.strip()
        if vision_context:
            prompt += f"\n\n[camera: {vision_context}]"
        if preference_context:
            prompt += f"\n\n[user_preferences: {preference_context}]"

        self.history.append({"role": "user", "content": prompt})

        if len(self.history) > config.MAX_CONVERSATION_HISTORY:
            self.history = self.history[-config.MAX_CONVERSATION_HISTORY:]

        messages = [{"role": "system", "content": config.SYSTEM_PROMPT}]
        messages.extend(self.history)

        payload = {
            "model": self.model_name,
            "stream": True,
            "messages": messages,
            "options": {
                "temperature": 0.5,
                "top_p": 0.9,
                "num_predict": 220,
            },
        }

        full_response = ""
        try:
            with httpx.Client(timeout=180.0) as client:
                with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            token = data.get("message", {}).get("content", "")
                            if token:
                                full_response += token
                                yield token
                        except json.JSONDecodeError:
                            continue
        except httpx.ConnectError:
            error_msg = "I can't reach Ollama right now. Make sure it's running with `ollama serve`."
            yield error_msg
            full_response = error_msg
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response else "unknown"
            detail = ""
            try:
                detail = e.response.text if e.response else ""
            except Exception:
                detail = ""
            if status == 404 and "model" in detail.lower():
                error_msg = (
                    f"Model '{self.model_name}' is not installed in Ollama. "
                    "Run: ollama pull llama3.2:3b"
                )
            else:
                error_msg = f"Ollama HTTP {status}: {detail[:220]}"
            yield error_msg
            full_response = error_msg
        except Exception as e:
            error_msg = f"LLM error: {e}"
            yield error_msg
            full_response = error_msg

        # Add assistant response to history
        if full_response:
            self.history.append({"role": "assistant", "content": full_response})

    def describe_image(self, image_b64: str, prompt: str = "Describe what you see in detail.") -> str:
        """Use the vision model to describe a base64-encoded image."""
        payload = {
            "model": config.VISION_MODEL,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64],
                }
            ],
        }

        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
            return data.get("message", {}).get("content", "I can see something but couldn't describe it.")
        except Exception as e:
            return f"Vision model error: {e}"

    def clear_history(self) -> None:
        """Reset conversation memory."""
        self.history.clear()

    @staticmethod
    def extract_tool_calls(text: str) -> list[dict]:
        """Extract [TOOL_CALL: ...] patterns from LLM response."""
        pattern = r'\[TOOL_CALL:\s*(\w+)\((.*?)\)\]'
        calls = []
        for match in re.finditer(pattern, text):
            func_name = match.group(1)
            args_str = match.group(2)

            # Parse key="value" pairs
            args = {}
            for arg_match in re.finditer(r'(\w+)\s*=\s*"((?:[^"\\]|\\.)*)"', args_str):
                args[arg_match.group(1)] = arg_match.group(2)

            calls.append({"function": func_name, "args": args, "raw": match.group(0)})

        return calls
