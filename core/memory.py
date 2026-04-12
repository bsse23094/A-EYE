"""Simple long-term memory store for user preferences across sessions."""

from __future__ import annotations

import json
import os
from typing import Any

from . import config


class MemoryStore:
    def __init__(self, path: str | None = None) -> None:
        self.path = path or config.MEMORY_FILE
        self._data: dict[str, Any] = {
            "preferences": {},
            "notes": [],
        }
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._data.update(data)
        except Exception:
            # Keep defaults if file is corrupted.
            pass

    def _save(self) -> None:
        folder = os.path.dirname(self.path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)

    def get_preferences_text(self) -> str:
        prefs = self._data.get("preferences", {})
        if not isinstance(prefs, dict) or not prefs:
            return ""
        items = [f"{k}: {v}" for k, v in prefs.items()]
        return "; ".join(items)

    def update_preferences_from_text(self, text: str) -> bool:
        """Heuristic extraction for durable user preferences."""
        low = text.lower()
        prefs = self._data.setdefault("preferences", {})
        changed = False

        if "minimal" in low and "dark" in low:
            if prefs.get("ui_style") != "minimal dark":
                prefs["ui_style"] = "minimal dark"
                changed = True

        if "british" in low and "voice" in low:
            if prefs.get("voice") != "british":
                prefs["voice"] = "british"
                changed = True

        if "call me" in low:
            marker = "call me"
            idx = low.find(marker)
            if idx != -1:
                name = text[idx + len(marker):].strip().strip(".!")
                if name and prefs.get("preferred_name") != name:
                    prefs["preferred_name"] = name
                    changed = True

        if changed:
            self._save()
        return changed
