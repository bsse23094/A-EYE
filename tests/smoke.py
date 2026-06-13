"""Offline smoke test — no network, no models. Run: py tests/smoke.py"""
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
failures = []


def check(name, cond, detail=""):
    print(f"  {'ok ' if cond else 'FAIL'} {name}{' — ' + detail if detail and not cond else ''}")
    if not cond:
        failures.append(name)


# ── scheduler.parse_when ─────────────────────────────────────────
from core.scheduler import parse_when

due, rep = parse_when("in 10 minutes")
check("parse 'in 10 minutes'", due is not None and rep is None and 595 < due - time.time() < 605)
due, rep = parse_when("every 2 hours")
check("parse 'every 2 hours'", rep == 7200)
due, rep = parse_when("at 23:59")
check("parse 'at 23:59'", due is not None and rep is None)
due, rep = parse_when("whenever")
check("parse garbage -> None", due is None)

# ── memory ───────────────────────────────────────────────────────
from core.memory import Memory

with tempfile.TemporaryDirectory() as td:
    m = Memory(os.path.join(td, "t.db"))
    m.new_session()
    m.add_message("user", "hello")
    m.add_message("assistant", "hi there")
    msgs = m.recent_messages(10)
    check("messages persist", [x["role"] for x in msgs] == ["user", "assistant"])
    fid = m.remember("The user's favourite editor is VS Code", "prefs")
    m.remember("The dev board on the desk is an ESP32-S3", "hardware")
    hits = m.recall("which editor do I like?")
    check("recall ranks editor fact first", hits and "editor" in hits[0])
    check("forget works", m.forget(fid))
    sid = m.session_id
    check("resume finds last session", m.resume_last_session() == sid)
    m.close()

# ── agent fallback parser ────────────────────────────────────────
from types import SimpleNamespace
from core.agent import Agent

agent = Agent.__new__(Agent)
agent.registry = SimpleNamespace(names=lambda: ["read_file", "weather", "remember"])

calls, cleaned = agent._parse_fallback_calls(
    'Let me check.\n```json\n{"tool": "read_file", "args": {"path": "x.py"}}\n```')
check("fallback parses fenced JSON", calls and calls[0]["name"] == "read_file"
      and calls[0]["arguments"] == {"path": "x.py"})
calls, cleaned = agent._parse_fallback_calls('{"tool": "weather", "args": {"city": "Lahore"}}')
check("fallback parses bare JSON", calls and calls[0]["name"] == "weather")
calls, cleaned = agent._parse_fallback_calls(
    '{\n  "name": "remember",\n  "arguments": {"fact": "x", "topic": "y"}\n}')
check("pseudo native-format call parsed", calls and calls[0]["name"] == "remember"
      and calls[0]["arguments"]["fact"] == "x")
calls, cleaned = agent._parse_fallback_calls("Just a normal answer with {braces} in it.")
check("fallback ignores non-tool text", not calls)
calls, cleaned = agent._parse_fallback_calls('{"name": "ordinary json", "arguments": {}}')
check("unregistered name not coerced", not calls)

# ── tool registry + file tools ───────────────────────────────────
from core.tools import ToolContext, ToolRegistry
from core.tools import files as files_mod

ctx = ToolContext()
reg = ToolRegistry(ctx)
files_mod.register(reg)
check("specs generated", any(s["function"]["name"] == "edit_file" for s in reg.specs()))
check("docs generated", "grep_search" in reg.docs())

with tempfile.TemporaryDirectory() as td:
    p = os.path.join(td, "demo.txt")
    reg.execute("write_file", {"path": p, "content": "alpha\nbeta\ngamma\n"})
    out = reg.execute("read_file", {"path": p})
    check("write+read roundtrip", "beta" in out)
    out = reg.execute("edit_file", {"path": p, "find": "beta", "replace": "BETA"})
    check("edit_file applies", "1 replacement" in out and "BETA" in open(p).read())
    out = reg.execute("edit_file", {"path": p, "find": "nope", "replace": "x"})
    check("edit_file miss is graceful", "not found" in out)
    out = reg.execute("grep_search", {"pattern": "GAM", "root": td})
    check("grep_search no match msg", "No matches" in out)
    out = reg.execute("grep_search", {"pattern": "(?i)gam", "root": td})
    check("grep_search finds", "demo.txt" in out)
    out = reg.execute("glob_search", {"pattern": "*.txt", "root": td})
    check("glob_search finds", "demo.txt" in out)
    out = reg.execute("nonexistent_tool", {})
    check("unknown tool is graceful", "Unknown tool" in out)
    out = reg.execute("read_file", {"path": p, "start_line": "2", "end_line": "2"})
    check("string line args accepted", "beta" in out.lower() and "alpha" not in out)

# ── voice text cleanup ───────────────────────────────────────────
from core.voice import VoiceIO, _clean_for_speech

check("speech strips markdown",
      _clean_for_speech("**Bold** and `code` and [a link](http://x).") == "Bold and code and a link.")
check("speech drops fenced code",
      "print" not in _clean_for_speech("Here:\n```py\nprint(1)\n```\ndone."))

v = VoiceIO.__new__(VoiceIO)
v._in_code = False
check("code skip: outside fence", v._skip_code("plain text.") == "plain text.")
check("code skip: enters fence", v._skip_code("Look: ```py").strip() == "Look:" and v._in_code)
check("code skip: inside fence", v._skip_code("x = 1") == "")
check("code skip: exits fence", "after" in v._skip_code("``` after") and not v._in_code)

# ── persona ──────────────────────────────────────────────────────
from core.persona import build_system_prompt

sp = build_system_prompt(["User likes terse answers"], None)
check("persona includes facts", "terse answers" in sp)
check("persona omits protocol when native", "Tool protocol" not in sp)
sp = build_system_prompt([], "- read_file(path): read a file")
check("persona includes fallback docs", "Tool protocol" in sp and "read_file" in sp)

# ── models classification ────────────────────────────────────────
from core.models import ModelInfo, _parse_size_b, ModelManager

check("size parse 14.8B", _parse_size_b("14.8B") == 14.8)
mi = ModelInfo(name="qwen2.5-coder:14b", provider=None)
ModelManager._classify_by_name(mi)
check("coder classified", mi.has("code"))
mi = ModelInfo(name="gemma3:12b", provider=None)
ModelManager._classify_by_name(mi)
check("gemma3 classified vision", mi.has("vision"))

print()
if failures:
    print(f"{len(failures)} FAILURE(S): {failures}")
    sys.exit(1)
print("All smoke tests passed.")
