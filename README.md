# JARVIS · A-EYE

A **local-first AI operating-system assistant**. Talk to it or type to it; it
reads and edits files, runs commands, inspects repositories, watches system
health, remembers things long-term, schedules tasks, and orchestrates whatever
local models you happen to have — all on your machine.

```
you>  what's eating my RAM?
  > system_status(detail=True)
jarvis> Chrome, predictably — 41 of your 92 percent. The usual suspects follow.

you>  find the bug in core/scheduler.py and fix it
  > read_file(path='core/scheduler.py')
  > edit_file(path='core/scheduler.py', find=..., replace=...)
jarvis> The timer re-armed before persisting the next due time. Fixed; restart-safe now.
```

No cloud accounts. No camera gimmicks. A terminal, your models, your machine.

---

## How it works

```
jarvis.py            entry point: REPL / --server / --voice / --once / --check
core/
  assistant.py       the core service: turn queue + event hub (no UI)
  cli.py             terminal frontend — renders the event stream in a REPL
  server.py          web frontend — FastAPI + SSE + embedded chat/workspace UI
  agent.py           the loop: model -> tool calls -> results -> model ...
  models.py          discovers local models, classifies capabilities, routes
  providers.py       Ollama API + any OpenAI-compatible server (LM Studio, ...)
  tools/             the assistant's hands (see below)
  memory.py          SQLite: conversations, long-term facts, scheduled tasks
  scheduler.py       timed/repeating tasks (timer-driven, no polling)
  watcher.py         edge-triggered RAM/CPU/disk/battery alerts
  voice.py           VAD mic capture + faster-whisper STT + TTS (lazy-loaded)
  persona.py         compact JARVIS system prompt
  config.py          defaults + ~/.jarvis/config.json overlay
plugins/             drop a .py file here to add your own tools
tests/smoke.py       offline smoke tests (no network, no models needed)
```

Both frontends are thin renderers over the same event stream
(`assistant.submit()` → `token` / `thinking` / `tool_start` / `done` events),
so there is exactly one turn pipeline no matter how you talk to it. Reasoning
models' **thinking is streamed live** — dimmed in the terminal, a collapsible
block in the web UI (`"show_thinking": false` to disable).

### Model orchestration — no hardcoded models

At startup JARVIS asks every reachable provider what it serves (Ollama first,
plus any OpenAI-compatible endpoints in config), reads each model's
capabilities (tool calling, vision, thinking, size), and routes by role:

| role   | picked by                                          |
|--------|----------------------------------------------------|
| chat   | tool-capable general model inside `chat_size_range_b` |
| code   | largest code-tuned model (qwen-coder, deepseek-coder, ...) |
| vision | largest vision-capable model (gemma3, llava, ...)  |

Models that support **native tool calling** get JSON-schema tools; models
that don't are driven through a prompt-injected fallback protocol — detected
automatically at runtime. Loose `.gguf` files in configured directories are
reported and can be imported into Ollama on request. Pin any role manually
with `/model code qwen2.5-coder:14b`.

### Tools (the actual product)

| group   | tools |
|---------|-------|
| files   | `read_file` `write_file` `edit_file` `list_dir` `glob_search` `grep_search` |
| shell   | `run_command` `open_app` `close_app` |
| system  | `system_status` `screenshot` `describe_screen` `clipboard_get/set` `set_volume` `type_text` `press_keys` |
| web     | `web_search` `fetch_url` (text / links / CSS-selector scraping) `weather` `news_headlines` |
| dev     | `git` `github` `github_repos` `github_clone` (whitelisted subcommands) `repo_map` |
| memory  | `remember` `recall_memory` `forget` |
| tasks   | `schedule_task` `list_tasks` `cancel_task` |
| models  | `list_models` |
| email   | `email_unread` `email_read` `email_send` — only when configured |

The agent chains tool calls until the task is done (`max_tool_iterations`
caps runaway loops). Long-term facts are auto-recalled into context by
relevance on every turn.

---

## Install

```bash
git clone https://github.com/<you>/A-EYE.git && cd A-EYE
pip install -r requirements.txt     # text mode needs only httpx + psutil
ollama pull qwen3:8b                # or any models you like — discovery is automatic
python jarvis.py
```

> Windows note: if `python` resolves to the Microsoft Store stub, use
> `py jarvis.py` (or `start.bat`, which picks the right launcher).

| command | effect |
|---|---|
| `python jarvis.py` | text REPL — starts in well under a second |
| `python jarvis.py --server` | web UI (chat, model switcher, workspace editor) |
| `python jarvis.py --voice` | also start always-on voice (VAD, no wake word) |
| `python jarvis.py --resume` | continue the previous conversation |
| `python jarvis.py --once "summarize git log"` | one prompt, then exit |
| `python jarvis.py --check` | diagnose deps, providers, config |

### Web UI

`--server` (or `start.bat`) serves a local single-file web app at
`127.0.0.1:8765`: streaming chat with live **thinking** blocks, tool-call
traces, a stop button, model-role switcher, voice toggle, and a **workspace
panel** — browse any folder, open a file, edit and save it right in the
browser, or hit *ask jarvis* to pull the file into the conversation and let
the agent edit it with you. The session transcript survives page reloads.

### In-session commands

`/voice` `/speak on|off|auto` `/model [role] <name|auto>` `/models` `/tools`
`/memory` `/forget <id>` `/tasks` `/status` `/new` `/quit`

---

## Configuration

Everything lives in `~/.jarvis/config.json` (created on first run with
defaults). Notable keys:

```jsonc
{
  "ollama_url": "http://localhost:11434",
  "openai_endpoints": [{ "name": "lmstudio", "base_url": "http://localhost:1234/v1" }],
  "gguf_dirs": ["~/models", "~/.openclaw/models"],
  "model_overrides": { "code": "qwen2.5-coder:14b" },   // optional pins
  "chat_size_range_b": [3, 14],     // preferred chat-model size window
  "show_thinking": true,            // stream reasoning-model thoughts to the UI
  "confirm_shell_commands": false,  // set true to approve each command
  "voice_enabled": false,
  "tts_backend": "auto",            // edge-tts -> SAPI fallback; "sapi" = fully offline
  "email": { "imap_host": "...", "smtp_host": "...", "user": "...", "password": "app-password" }
}
```

Conversation history, long-term facts, and scheduled tasks persist in
`~/.jarvis/memory.db` (SQLite, WAL mode).

### Privacy notes

Everything is local except what you explicitly invoke: `web_search` /
`fetch_url` / `weather` / `news_headlines` touch the network, email tools
touch your mail server, and the default TTS voice (edge-tts) is a Microsoft
service — set `"tts_backend": "sapi"` for fully offline speech. STT
(faster-whisper) and all model inference are 100 % on-device.

---

## Extending

Drop a file in `plugins/`:

```python
def register(registry, ctx):
    @registry.register("my_tool", "One-line description",
                       {"arg": "string: what it is", "?opt": "integer: optional"})
    def my_tool(ctx, arg, opt=0):
        return "result string"
```

Extension points designed in: embedding-based memory recall (swap
`Memory.recall`), additional providers (subclass `BaseProvider`), more
roles in the router, richer schedulers — all without touching the agent loop.

---

Built by **Ahmed Ayyan** — Lahore, Pakistan.
