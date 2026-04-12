"""Proactive background checks for crashes, lint failures, and command errors."""

from __future__ import annotations

import glob
import os
import subprocess
import threading
import time
from typing import Callable

from .tools import ToolExecutor


class ProactiveMonitor:
    """Watches local development environment and emits actionable alerts."""

    def __init__(self, project_root: str, on_alert: Callable[[str], None]) -> None:
        self.project_root = project_root
        self.on_alert = on_alert
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_failure_scan = time.time()
        self._last_lint_check = 0.0
        self._seen_alerts: set[str] = set()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def _loop(self) -> None:
        while self._running:
            try:
                self._check_recent_tool_failures()
                self._check_debug_logs()
                self._check_angular_lint()
            except Exception:
                # Never kill monitor thread on a transient failure.
                pass
            time.sleep(8)

    def _emit_once(self, key: str, text: str) -> None:
        if key in self._seen_alerts:
            return
        self._seen_alerts.add(key)
        self.on_alert(text)

    def _check_recent_tool_failures(self) -> None:
        failures = ToolExecutor.get_recent_failures(self._last_failure_scan)
        self._last_failure_scan = time.time()
        for f in failures:
            msg = (f.get("message") or "").strip()
            if not msg:
                continue
            head = msg.splitlines()[0][:180]
            key = f"tool:{f.get('kind')}:{head}"
            self._emit_once(
                key,
                (
                    "Proactive check: I detected a failed command and can help fix it. "
                    f"Issue: {head}"
                ),
            )

    def _check_debug_logs(self) -> None:
        patterns = [
            os.path.join(self.project_root, "**", "npm-debug.log*"),
            os.path.join(self.project_root, "**", "yarn-error.log*"),
            os.path.join(self.project_root, "**", "pnpm-debug.log*"),
        ]
        now = time.time()
        for pattern in patterns:
            for path in glob.glob(pattern, recursive=True):
                try:
                    mtime = os.path.getmtime(path)
                except OSError:
                    continue
                if now - mtime > 120:
                    continue
                key = f"log:{path}:{int(mtime)}"
                self._emit_once(
                    key,
                    (
                        "Proactive check: I found a fresh package-manager crash log. "
                        f"File: {path}"
                    ),
                )

    def _check_angular_lint(self) -> None:
        angular_json = os.path.join(self.project_root, "angular.json")
        package_json = os.path.join(self.project_root, "package.json")
        if not (os.path.exists(angular_json) and os.path.exists(package_json)):
            return

        now = time.time()
        if now - self._last_lint_check < 180:
            return
        self._last_lint_check = now

        try:
            result = subprocess.run(
                "npm run lint -- --quiet",
                shell=True,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=45,
            )
        except Exception:
            return

        if result.returncode == 0:
            return

        output = (result.stderr or "").strip() or (result.stdout or "").strip()
        if not output:
            return

        first_line = output.splitlines()[0][:200]
        key = f"lint:{first_line}"
        self._emit_once(
            key,
            (
                "Proactive check: Angular lint reported an issue. "
                f"First error: {first_line}"
            ),
        )
