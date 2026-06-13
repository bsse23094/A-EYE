"""Assistant core — services, turn execution, event hub. No UI here.

Every frontend (terminal, web, future GUI) is a thin renderer over the
same event stream:

    assistant.submit("do something", source="text")   -> turn_id
    q = assistant.subscribe()                          -> Queue of events

Events are plain dicts with a "type" key:
    user         {turn_id, text, source}
    token        {turn_id, text}
    thinking     {turn_id, text}            (reasoning-model thoughts)
    tool_start   {turn_id, name, args}
    tool_result  {turn_id, name, preview}
    done         {turn_id, text, model, tools_used}
    error        {turn_id, text}
    notify       {text}                      (watcher / system messages)
    voice        {state}                     (off|on|listening|...)

Turns are serialized through one worker thread, so voice, web, and
scheduled prompts never interleave mid-response.
"""

from __future__ import annotations

import itertools
import queue
import threading
from typing import Callable, Optional

from .agent import Agent, TurnUI
from .config import config
from .memory import Memory
from .models import ModelManager
from .providers import build_providers
from .scheduler import Scheduler
from .tools import ToolContext, build_registry
from .watcher import SystemWatcher


class Assistant:
    def __init__(self, voice: bool = False, resume: bool = False) -> None:
        self.memory = Memory()
        if not (resume and self.memory.resume_last_session()):
            self.memory.new_session()

        self.models = ModelManager(build_providers(config), config)
        self.models.discover_async()

        # Frontends may install an interactive confirm hook(prompt, source).
        self.confirm_hook: Optional[Callable[[str], bool]] = None
        self._current_source = "text"

        ctx = ToolContext(cfg=config, memory=self.memory, models=self.models,
                          confirm=self._confirm, notify=self.notify)
        self.registry = build_registry(ctx)
        self.agent = Agent(config, self.models, self.registry, self.memory)

        self.scheduler = Scheduler(
            self.memory, lambda prompt: self.submit(prompt, source="schedule"))
        ctx.scheduler = self.scheduler
        self.scheduler.start()

        self.watcher = SystemWatcher(config, self.notify)
        if config.watcher_enabled:
            self.watcher.start()

        self.voice = None
        self.speak_mode = config.speak_replies          # auto | on | off

        # Event hub
        self._subscribers: list[queue.Queue] = []
        self._sub_lock = threading.Lock()

        # Turn pipeline
        self._turn_ids = itertools.count(1)
        self._turn_queue: "queue.Queue[Optional[tuple]]" = queue.Queue()
        self._worker = threading.Thread(target=self._turn_worker, daemon=True,
                                        name="turn-worker")
        self._worker.start()

        if voice or config.voice_enabled:
            self.voice_on()

    # ── Event hub ────────────────────────────────────────────────

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=2000)
        with self._sub_lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._sub_lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def publish(self, event: dict) -> None:
        with self._sub_lock:
            subs = list(self._subscribers)
        for q in subs:
            try:
                q.put_nowait(event)
            except queue.Full:
                pass  # slow consumer loses events rather than blocking turns

    def notify(self, text: str) -> None:
        self.publish({"type": "notify", "text": text})
        if self.voice and self.voice.active and self.speak_mode != "off":
            self.voice.speak(text)

    # ── Turns ────────────────────────────────────────────────────

    def submit(self, text: str, source: str = "text") -> int:
        """Queue a user turn. Events stream to all subscribers."""
        turn_id = next(self._turn_ids)
        self._turn_queue.put((turn_id, text, source))
        return turn_id

    def _turn_worker(self) -> None:
        while True:
            item = self._turn_queue.get()
            if item is None:
                return
            turn_id, text, source = item
            try:
                self._run_turn(turn_id, text, source)
            except Exception as e:
                self.publish({"type": "error", "turn_id": turn_id,
                              "text": f"internal error: {e}"})

    def _run_turn(self, turn_id: int, text: str, source: str) -> None:
        self._current_source = source
        self.publish({"type": "user", "turn_id": turn_id,
                      "text": text, "source": source})

        speak = (self.speak_mode == "on"
                 or (self.speak_mode == "auto" and source == "voice"))
        speak = speak and self.voice is not None

        def on_token(tok: str) -> None:
            self.publish({"type": "token", "turn_id": turn_id, "text": tok})
            if speak:
                self.voice.speak_token(tok)

        def on_thinking(tok: str) -> None:
            self.publish({"type": "thinking", "turn_id": turn_id, "text": tok})

        def on_tool_start(name: str, args: dict) -> None:
            arg_str = ", ".join(f"{k}={str(v)[:50]!r}" for k, v in args.items())
            self.publish({"type": "tool_start", "turn_id": turn_id,
                          "name": name, "args": arg_str})

        def on_tool_result(name: str, output: str) -> None:
            preview = output.strip().replace("\n", " | ")[:200]
            self.publish({"type": "tool_result", "turn_id": turn_id,
                          "name": name, "preview": preview})

        ui = TurnUI(on_token=on_token, on_thinking=on_thinking,
                    on_tool_start=on_tool_start, on_tool_result=on_tool_result)
        result = self.agent.run_turn(text, ui)

        if speak:
            self.voice.flush_speech()
        if result.error and result.error != result.text:
            self.publish({"type": "error", "turn_id": turn_id, "text": result.error})
        self.publish({"type": "done", "turn_id": turn_id, "text": result.text,
                      "model": result.model, "tools_used": len(result.tool_log)})

    def _confirm(self, prompt: str) -> bool:
        if self.confirm_hook and self._current_source == "text":
            return self.confirm_hook(prompt)
        self.notify(f"Action needs confirmation and was skipped: {prompt}. "
                    "Run it from the terminal frontend, or relax the confirm_* "
                    "settings in config.json.")
        return False

    # ── Voice ────────────────────────────────────────────────────

    def voice_on(self) -> str:
        if self.voice and self.voice.active:
            return "Voice is already on."
        from .voice import VoiceIO
        if self.voice is None:
            self.voice = VoiceIO(
                config,
                on_utterance=lambda text: self.submit(text, source="voice"),
                on_state=lambda s: self.publish({"type": "voice", "state": s}))
        msg = self.voice.start()
        self.publish({"type": "voice",
                      "state": "on" if self.voice.active else "unavailable"})
        return msg

    def voice_off(self) -> str:
        if self.voice and self.voice.active:
            self.voice.stop()
            self.publish({"type": "voice", "state": "off"})
            return "Voice off."
        return "Voice was not on."

    @property
    def voice_active(self) -> bool:
        return bool(self.voice and self.voice.active)

    # ── Introspection for frontends ──────────────────────────────

    def status(self) -> dict:
        ready = self.models.ensure_ready(timeout=0.1)
        routes = {}
        if ready:
            for role in ("chat", "code", "vision"):
                picked = self.models.pick(role)
                routes[role] = picked.name if picked else None
        return {
            "models": len(self.models.models) if ready else None,
            "routes": routes,
            "errors": self.models.errors,
            "voice": self.voice_active,
            "speak": self.speak_mode,
            "tools": len(self.registry.names()),
            "session": self.memory.session_id,
            "facts": len(self.memory.all_facts()),
        }

    def new_session(self) -> int:
        return self.memory.new_session()

    def set_model_override(self, role: str, name: str) -> str:
        if name == "auto":
            config.model_overrides.pop(role, None)
        else:
            config.model_overrides[role] = name
        config.save()
        return f"route[{role}] -> {name}"

    def set_speak_mode(self, mode: str) -> str:
        if mode in ("on", "off", "auto"):
            self.speak_mode = mode
            config.set("speak_replies", mode)
        return f"Speak mode: {self.speak_mode}"

    # ── Lifecycle ────────────────────────────────────────────────

    def shutdown(self) -> None:
        self.agent.cancel.set()
        self._turn_queue.put(None)
        if self.voice:
            self.voice.stop()
        self.watcher.stop()
        self.scheduler.stop()
        for p in self.models.providers:
            p.close()
        self.memory.close()
