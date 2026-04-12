"""Local tools — skill-based actions for docs, terminal, git, and system control."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import threading
import time
import webbrowser
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus
from urllib.request import urlopen, Request
from typing import Optional


class ToolExecutor:
    """Executes tools for JARVIS system integration."""

    _events: list[dict] = []
    _events_lock = threading.Lock()
    _max_events = 200

    @classmethod
    def _record_event(cls, kind: str, message: str, ok: bool) -> None:
        with cls._events_lock:
            cls._events.append(
                {
                    "ts": time.time(),
                    "kind": kind,
                    "message": message,
                    "ok": ok,
                }
            )
            if len(cls._events) > cls._max_events:
                cls._events = cls._events[-cls._max_events :]

    @classmethod
    def get_recent_failures(cls, since_ts: float) -> list[dict]:
        with cls._events_lock:
            return [
                e for e in cls._events
                if (not e.get("ok", True)) and e.get("ts", 0.0) >= since_ts
            ]

    @staticmethod
    def execute(func_name: str, args: dict) -> str:
        dispatch = {
            "open_app": ToolExecutor.open_app,
            "close_app": ToolExecutor.close_app,
            "run_command": ToolExecutor.run_command,
            "run_terminal": ToolExecutor.run_terminal,
            "read_file": ToolExecutor.read_file,
            "write_file": ToolExecutor.write_file,
            "list_directory": ToolExecutor.list_directory,
            "web_search": ToolExecutor.web_search,
            "search_docs": ToolExecutor.search_docs,
            "read_webpage": ToolExecutor.read_webpage,
            "get_news": ToolExecutor.get_news,
            "get_weather": ToolExecutor.get_weather,
            "take_screenshot": ToolExecutor.take_screenshot,
            "describe_screen": ToolExecutor.describe_screen,
            "set_volume": ToolExecutor.set_volume,
            "git_status": ToolExecutor.git_status,
            "git_commit": ToolExecutor.git_commit,
            "git_push": ToolExecutor.git_push,
            "git_prepare_pr": ToolExecutor.git_prepare_pr,
            "automation_type": ToolExecutor.automation_type,
            "automation_hotkey": ToolExecutor.automation_hotkey,
            "automation_click": ToolExecutor.automation_click,
            "dev_mode": ToolExecutor.dev_mode,
            "launch_environment": ToolExecutor.launch_environment,
        }
        handler = dispatch.get(func_name)
        if handler is None:
            return f"Unknown tool: {func_name}"
        try:
            return handler(**args)
        except Exception as e:
            return f"Tool '{func_name}' failed: {e}"

    # ── Web & Information ─────────────────────────────────────

    @staticmethod
    def get_news(limit: str = "5", **kw) -> str:
        try:
            n = int(limit)
        except (ValueError, TypeError):
            n = 5

        url = "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en"
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urlopen(req, timeout=15) as resp:
                xml = resp.read().decode("utf-8", errors="ignore")
            root = ET.fromstring(xml)
            items = root.findall("./channel/item/title")
            headlines = []
            for item in items[:n]:
                if item.text:
                    # Clean up "headline - Source" format
                    text = item.text.strip()
                    if " - " in text:
                        text = text.rsplit(" - ", 1)[0]
                    headlines.append(text)
            if not headlines:
                return "Couldn't find any headlines right now."
            return "Top headlines:\n" + "\n".join(f"{i+1}. {h}" for i, h in enumerate(headlines))
        except Exception as e:
            return f"News fetch failed: {e}"

    @staticmethod
    def web_search(query: str = "", **kw) -> str:
        """Proper web search using duckduckgo-search library."""
        if not query:
            return "No search query."

        # Try the proper library first
        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))

            if results:
                output = [f"Search results for '{query}':"]
                for r in results:
                    title = r.get("title", "")
                    body = r.get("body", "")[:200]
                    href = r.get("href", "")
                    output.append(f"\n• {title}")
                    if body:
                        output.append(f"  {body}")
                    if href:
                        output.append(f"  → {href}")
                return "\n".join(output)
        except ImportError:
            pass
        except Exception as e:
            print(f"[Tools] DuckDuckGo search error: {e}")

        # Fallback: open browser
        webbrowser.open(f"https://duckduckgo.com/?q={quote_plus(query)}")
        return f"Opened search for: {query}"

    @staticmethod
    def search_docs(query: str = "", source: str = "", **kw) -> str:
        """Search technical docs with source preference (mdn|stackoverflow|general)."""
        if not query:
            return "No search query."

        source_key = (source or "").strip().lower()
        scoped_query = query
        if source_key == "mdn":
            scoped_query = f"site:developer.mozilla.org {query}"
        elif source_key in {"stackoverflow", "so", "stack"}:
            scoped_query = f"site:stackoverflow.com {query}"
        elif source_key in {"angular", "ng"}:
            scoped_query = f"site:angular.dev OR site:material.angular.dev {query}"

        return ToolExecutor.web_search(query=scoped_query)

    @staticmethod
    def read_webpage(url: str = "", **kw) -> str:
        if not url:
            return "No URL provided."
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"

        req = Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        try:
            with urlopen(req, timeout=20) as resp:
                html = resp.read().decode("utf-8", errors="ignore")
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                    tag.decompose()
                text = soup.get_text(separator="\n", strip=True)
                lines = [l.strip() for l in text.split("\n") if l.strip()]
                content = "\n".join(lines[:60])
                if len(content) > 4000:
                    content = content[:4000] + "\n…[truncated]"
                return f"Content from {url}:\n{content}"
            except ImportError:
                import re
                text = re.sub(r"<[^>]+>", " ", html)
                text = re.sub(r"\s+", " ", text).strip()
                return f"Content from {url}:\n{text[:4000]}"
        except Exception as e:
            return f"Failed to read {url}: {e}"

    @staticmethod
    def get_weather(city: str = "auto", **kw) -> str:
        try:
            url = f"https://wttr.in/{quote_plus(city)}?format=%l:+%C+%t+%h+%w"
            req = Request(url, headers={"User-Agent": "curl/7.0"})
            with urlopen(req, timeout=10) as resp:
                return resp.read().decode("utf-8").strip()
        except Exception as e:
            return f"Weather failed: {e}"

    # ── System Control ────────────────────────────────────────

    @staticmethod
    def open_app(name: str = "", **kw) -> str:
        if not name:
            return "No app name."

        name_lower = name.strip().lower()
        app_map = {
            "notepad": ["notepad.exe"],
            "calculator": ["calc.exe"],
            "calc": ["calc.exe"],
            "paint": ["mspaint.exe"],
            "cmd": ["cmd.exe"],
            "terminal": ["wt.exe", "powershell.exe"],
            "powershell": ["powershell.exe"],
            "explorer": ["explorer.exe"],
            "file explorer": ["explorer.exe"],
            "task manager": ["taskmgr.exe"],
            "control panel": ["control.exe"],
            "word": ["winword.exe"],
            "excel": ["excel.exe"],
            "vscode": ["code.cmd", "code.exe"],
            "vs code": ["code.cmd", "code.exe"],
            "chrome": ["chrome.exe"],
            "edge": ["msedge.exe"],
            "firefox": ["firefox.exe"],
            "snipping tool": ["SnippingTool.exe"],
        }

        if name_lower == "settings":
            try:
                os.startfile("ms-settings:")  # type: ignore[attr-defined]
                ToolExecutor._record_event("open_app", "Opened settings", True)
                return "Opened settings."
            except Exception as e:
                ToolExecutor._record_event("open_app", f"Failed opening settings: {e}", False)
                return f"Failed to open settings: {e}"

        def _try_launch(candidates: list[str]) -> bool:
            for cand in candidates:
                resolved = shutil.which(cand) or cand
                try:
                    subprocess.Popen([resolved], shell=False)
                    return True
                except Exception:
                    try:
                        subprocess.Popen(f'start "" "{resolved}"', shell=True)
                        return True
                    except Exception:
                        continue
            return False

        candidates = app_map.get(name_lower)
        if candidates and _try_launch(candidates):
            ToolExecutor._record_event("open_app", f"Opened {name}", True)
            return f"Opened {name}."

        if name_lower in {"browser"}:
            webbrowser.open("https://www.google.com")
            ToolExecutor._record_event("open_app", "Opened browser", True)
            return "Opened browser."

        # Try running it directly
        try:
            resolved = shutil.which(name) or name
            subprocess.Popen([resolved], shell=False)
            ToolExecutor._record_event("open_app", f"Attempted opening {name}", True)
            return f"Attempting to open: {name}"
        except Exception as e1:
            try:
                subprocess.Popen(f'start "" "{name}"', shell=True)
                ToolExecutor._record_event("open_app", f"Attempted opening {name} via shell", True)
                return f"Attempting to open: {name}"
            except Exception as e2:
                ToolExecutor._record_event("open_app", f"Could not find {name}: {e2}", False)
                return f"Couldn't open '{name}'. Errors: {e1} | {e2}"

    @staticmethod
    def close_app(name: str = "", **kw) -> str:
        if not name:
            return "No app name."

        process_map = {
            "chrome": "chrome.exe",
            "edge": "msedge.exe",
            "firefox": "firefox.exe",
            "vscode": "Code.exe",
            "vs code": "Code.exe",
            "notepad": "notepad.exe",
            "calculator": "CalculatorApp.exe",
            "terminal": "WindowsTerminal.exe",
            "powershell": "powershell.exe",
        }
        target = process_map.get(name.strip().lower(), name.strip())
        if not target.lower().endswith(".exe"):
            target = f"{target}.exe"

        cmd = f'taskkill /IM "{target}" /F'
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                ToolExecutor._record_event("close_app", f"Closed {target}", True)
                return f"Closed {name}."
            stderr = (result.stderr or "").strip()
            msg = stderr or (result.stdout or "").strip() or "process not found"
            ToolExecutor._record_event("close_app", f"Close {target} failed: {msg}", False)
            return f"Could not close {name}: {msg}"
        except Exception as e:
            ToolExecutor._record_event("close_app", f"Close {target} error: {e}", False)
            return f"Failed to close {name}: {e}"

    @staticmethod
    def run_command(cmd: str = "", **kw) -> str:
        """Execute a system command and return output. No blocking."""
        if not cmd:
            return "No command provided."

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=os.path.expanduser("~"),
            )
            output = ""
            if result.stdout.strip():
                output = result.stdout.strip()
            if result.returncode != 0 and result.stderr.strip():
                if output:
                    output += "\n"
                output += f"[error]: {result.stderr.strip()}"
            if not output:
                output = f"Done (exit code {result.returncode})"
            if len(output) > 3000:
                output = output[:3000] + "\n…[truncated]"
            ToolExecutor._record_event("run_command", output, result.returncode == 0)
            return output
        except subprocess.TimeoutExpired:
            ToolExecutor._record_event("run_command", f"Timeout running: {cmd}", False)
            return f"Command timed out: {cmd}"
        except Exception as e:
            ToolExecutor._record_event("run_command", f"Failed running: {cmd} -> {e}", False)
            return f"Command failed: {e}"

    @staticmethod
    def run_terminal(cmd: str = "", cwd: str = "", timeout: str = "45", **kw) -> str:
        """Skill: RunTerminal — executes shell commands in a target directory."""
        if not cmd:
            return "No command provided."

        workdir = os.path.expanduser(cwd) if cwd else os.getcwd()
        if not os.path.isdir(workdir):
            return f"Invalid cwd: {workdir}"

        try:
            t = max(5, min(300, int(timeout)))
        except (TypeError, ValueError):
            t = 45

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=t,
            )
            out = (result.stdout or "").strip()
            err = (result.stderr or "").strip()
            combined = out
            if err:
                combined = f"{combined}\n{err}".strip()
            if not combined:
                combined = f"Done (exit code {result.returncode})"
            if len(combined) > 4000:
                combined = combined[:4000] + "\n…[truncated]"
            ToolExecutor._record_event("run_terminal", combined, result.returncode == 0)
            return combined
        except subprocess.TimeoutExpired:
            ToolExecutor._record_event("run_terminal", f"Timeout running: {cmd}", False)
            return f"Command timed out after {t}s: {cmd}"
        except Exception as e:
            ToolExecutor._record_event("run_terminal", f"Command failed: {e}", False)
            return f"Command failed: {e}"

    @staticmethod
    def read_file(path: str = "", **kw) -> str:
        if not path:
            return "No path."
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            return f"File not found: {path}"
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            if len(content) > 5000:
                content = content[:5000] + "\n…[truncated]"
            return f"Contents of {path}:\n{content}"
        except Exception as e:
            return f"Read failed: {e}"

    @staticmethod
    def write_file(path: str = "", content: str = "", **kw) -> str:
        if not path:
            return "No path."
        if not content:
            return "No content."
        path = os.path.expanduser(path)
        try:
            d = os.path.dirname(path)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Written to {path} ({len(content)} chars)"
        except Exception as e:
            return f"Write failed: {e}"

    @staticmethod
    def list_directory(path: str = ".", **kw) -> str:
        path = os.path.expanduser(path)
        if not os.path.isdir(path):
            return f"Not a directory: {path}"
        try:
            entries = os.listdir(path)
            if not entries:
                return f"{path} is empty."
            items = []
            for e in sorted(entries)[:40]:
                full = os.path.join(path, e)
                if os.path.isdir(full):
                    items.append(f"📁 {e}/")
                else:
                    sz = os.path.getsize(full)
                    s = f"{sz/1024/1024:.1f}MB" if sz > 1048576 else f"{sz/1024:.0f}KB" if sz > 1024 else f"{sz}B"
                    items.append(f"📄 {e} ({s})")
            return f"{path}:\n" + "\n".join(items)
        except Exception as e:
            return f"List failed: {e}"

    @staticmethod
    def take_screenshot(**kw) -> str:
        try:
            from PIL import ImageGrab
            import time
            fn = f"screenshot_{int(time.time())}.png"
            desk = os.path.join(os.path.expanduser("~"), "Desktop", fn)
            ImageGrab.grab().save(desk)
            return f"Screenshot saved: {desk}"
        except Exception as e:
            return f"Screenshot failed: {e}"

    @staticmethod
    def describe_screen(prompt: str = "Describe UI issues and layout alignment problems.", **kw) -> str:
        """Capture current screen and ask local vision model for analysis."""
        try:
            from PIL import ImageGrab
            import io
            import base64
            import httpx
            from . import config

            image = ImageGrab.grab()
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=85)
            image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

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
            with httpx.Client(timeout=90.0) as client:
                resp = client.post(f"{config.OLLAMA_URL}/api/chat", json=payload)
                resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "I captured the screen but could not analyze it.")
        except Exception as e:
            return f"Screen analysis failed: {e}"

    @staticmethod
    def set_volume(level: str = "50", **kw) -> str:
        try:
            v = max(0, min(100, int(level)))
            ps = f'$wshell = New-Object -ComObject wscript.shell; 1..50 | %{{ $wshell.SendKeys([char]174) }}; 1..{v//2} | %{{ $wshell.SendKeys([char]175) }}'
            subprocess.run(["powershell", "-Command", ps], capture_output=True, timeout=10)
            return f"Volume set to ~{v}%"
        except Exception as e:
            return f"Volume control failed: {e}"

    # ── GitOps skill ────────────────────────────────────────────

    @staticmethod
    def git_status(repo: str = ".", **kw) -> str:
        repo_path = os.path.expanduser(repo)
        if not os.path.isdir(repo_path):
            return f"Invalid repo path: {repo_path}"
        return ToolExecutor.run_terminal(cmd="git status --short --branch", cwd=repo_path, timeout="20")

    @staticmethod
    def git_commit(repo: str = ".", message: str = "Update", add_all: str = "true", **kw) -> str:
        repo_path = os.path.expanduser(repo)
        if not os.path.isdir(repo_path):
            return f"Invalid repo path: {repo_path}"

        if add_all.strip().lower() in {"true", "1", "yes"}:
            stage = ToolExecutor.run_terminal(cmd="git add -A", cwd=repo_path, timeout="20")
            if "fatal" in stage.lower() or "error" in stage.lower():
                return f"Staging failed: {stage}"

        safe_message = message.replace('"', "'")
        commit = ToolExecutor.run_terminal(
            cmd=f'git commit -m "{safe_message}"',
            cwd=repo_path,
            timeout="30",
        )
        return commit

    @staticmethod
    def git_push(repo: str = ".", remote: str = "origin", branch: str = "", **kw) -> str:
        repo_path = os.path.expanduser(repo)
        if not os.path.isdir(repo_path):
            return f"Invalid repo path: {repo_path}"
        if branch:
            cmd = f"git push {remote} {branch}"
        else:
            cmd = f"git push {remote}"
        return ToolExecutor.run_terminal(cmd=cmd, cwd=repo_path, timeout="120")

    @staticmethod
    def git_prepare_pr(repo: str = ".", base: str = "main", **kw) -> str:
        """Build a quick PR summary from recent commits and changed files."""
        repo_path = os.path.expanduser(repo)
        if not os.path.isdir(repo_path):
            return f"Invalid repo path: {repo_path}"

        head = ToolExecutor.run_terminal(cmd="git rev-parse --abbrev-ref HEAD", cwd=repo_path, timeout="10")
        files = ToolExecutor.run_terminal(cmd="git diff --name-only HEAD~1..HEAD", cwd=repo_path, timeout="10")
        commits = ToolExecutor.run_terminal(cmd="git log --oneline -5", cwd=repo_path, timeout="10")

        return (
            f"PR draft\n"
            f"Base: {base}\n"
            f"Head: {head.strip()}\n\n"
            f"Recent commits:\n{commits}\n\n"
            f"Changed files:\n{files}\n"
        )

    # ── UI automation skill (optional; requires pyautogui) ─────

    @staticmethod
    def automation_type(text: str = "", interval: str = "0.02", **kw) -> str:
        if not text:
            return "No text provided."
        try:
            import pyautogui
            pyautogui.write(text, interval=float(interval))
            return "Typed text."
        except Exception as e:
            return f"Automation typing failed: {e}"

    @staticmethod
    def automation_hotkey(keys: str = "", **kw) -> str:
        if not keys:
            return "No keys provided."
        parts = [k.strip() for k in keys.split("+") if k.strip()]
        if not parts:
            return "No keys provided."
        try:
            import pyautogui
            pyautogui.hotkey(*parts)
            return f"Pressed hotkey: {keys}"
        except Exception as e:
            return f"Automation hotkey failed: {e}"

    @staticmethod
    def automation_click(x: str = "", y: str = "", button: str = "left", **kw) -> str:
        try:
            import pyautogui
            if x and y:
                pyautogui.click(x=int(x), y=int(y), button=button)
            else:
                pyautogui.click(button=button)
            return "Clicked."
        except Exception as e:
            return f"Automation click failed: {e}"

    # ── Quick-match for common voice commands ─────────────────

    @staticmethod
    def quick_match(text: str) -> Optional[str]:
        """Match common voice commands without needing LLM. Returns result or None."""
        low = text.strip().lower()

        if any(k in low for k in ["news", "headlines", "what's happening"]):
            return ToolExecutor.get_news()

        if "weather" in low:
            import re
            m = re.search(r"weather\s+(?:in|at|for)\s+(.+)", low)
            city = m.group(1).strip() if m else "auto"
            return ToolExecutor.get_weather(city=city)

        if low.startswith("open "):
            target = text[5:].strip()
            if target:
                return ToolExecutor.open_app(name=target)

        if low.startswith("close "):
            target = text[6:].strip()
            if target:
                return ToolExecutor.close_app(name=target)

        if low.startswith("run "):
            cmd = text[4:].strip()
            if cmd:
                return ToolExecutor.run_terminal(cmd=cmd)

        if low.startswith("git status"):
            return ToolExecutor.git_status()

        for prefix in ["search for ", "search web for ", "google ", "look up "]:
            if low.startswith(prefix):
                q = text[len(prefix):].strip()
                if q:
                    return ToolExecutor.web_search(query=q)

        for prefix in ["search docs for ", "docs for "]:
            if low.startswith(prefix):
                q = text[len(prefix):].strip()
                if q:
                    return ToolExecutor.search_docs(query=q)

        # Dev mode shortcut
        for prefix in ["dev mode", "activate dev", "start dev", "open dev"]:
            if prefix in low:
                project = "default"
                for part in low.split():
                    if part not in ("dev", "mode", "activate", "start", "open", "environment", "for"):
                        project = part
                        break
                return ToolExecutor.dev_mode(project=project)

        return None

    # ── Dev Environment Dispatcher ────────────────────────────────

    @staticmethod
    def dev_mode(project: str = "default", **kw) -> str:
        """Open VS Code + optionally launch a dev server for a known project."""
        from . import config as _cfg
        import subprocess

        launched = []
        failed = []

        proj_cfg = _cfg.DEV_PROJECTS.get(project) or _cfg.DEV_PROJECTS.get("default")

        # Open editor
        try:
            editor_args = list(_cfg.DEV_EDITOR_CMD)
            if proj_cfg and proj_cfg.get("path"):
                editor_args.append(proj_cfg["path"])
            subprocess.Popen(editor_args, shell=False)
            launched.append("editor")
        except Exception as e:
            failed.append(f"editor ({e})")

        # Launch dev server if configured
        if proj_cfg and proj_cfg.get("server_cmd"):
            try:
                cwd = proj_cfg.get("path") or None
                # Use cmd /k so the terminal stays open
                subprocess.Popen(
                    ["cmd", "/k", proj_cfg["server_cmd"]],
                    cwd=cwd,
                    creationflags=subprocess.CREATE_NEW_CONSOLE,
                )
                launched.append("dev server")
            except Exception as e:
                failed.append(f"dev server ({e})")

        parts = []
        if launched:
            parts.append(f"Launched: {', '.join(launched)}")
        if failed:
            parts.append(f"Failed: {', '.join(failed)}")
        result = ". ".join(parts) or "Nothing configured under DEV_PROJECTS in config.py."
        ToolExecutor._record_event("dev_mode", result, not failed)
        return result

    @staticmethod
    def launch_environment(apps: str = "", cwd: str = "", **kw) -> str:
        """Launch a list of space-separated app names, optionally in a given cwd."""
        import subprocess
        app_list = [a.strip() for a in apps.split(",") if a.strip()]
        if not app_list:
            return "No apps specified."
        launched = []
        for app in app_list:
            try:
                kwargs = {"cwd": cwd} if cwd else {}
                subprocess.Popen(app.split(), **kwargs)
                launched.append(app)
            except Exception as e:
                launched.append(f"{app}(FAILED: {e})")
        return "Launched: " + ", ".join(launched)
