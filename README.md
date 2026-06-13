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
(`assistant.submit()` → `token` / `thinking` / `tool_start` / `file_edit` /
`done` events), so there is exactly one turn pipeline no matter how you talk
to it. Reasoning models' **thinking is streamed live** — dimmed in the
terminal, a collapsible block in the web UI (`"show_thinking": false` to
disable). Every file the agent writes is diffed and broadcast as a
`file_edit` event, which the web IDE renders side-by-side with the code.

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
| files   | `read_file` `read_pdf` `write_file` `edit_file` `list_dir` `glob_search` `grep_search` |
| shell   | `run_command` `open_app` `close_app` |
| system  | `system_status` `screenshot` `describe_screen` `read_image` (vision model on any image file) `clipboard_get/set` `set_volume` `type_text` `press_keys` |
| hardware| `hardware_report` (CPU/GPU/RAM/disk/temps/top processes) `gpu_status` (NVIDIA util/VRAM/temp/power) |
| web     | `web_search` `fetch_url` (text / links / CSS-selector scraping) `weather` `news_headlines` |
| dev     | `git` `github` `github_repos` `github_clone` (whitelisted subcommands) `repo_map` |
| memory  | `remember` `recall_memory` `forget` `profile_set` `profile_forget` `export_chat` |
| tasks   | `schedule_task` `list_tasks` `cancel_task` |
| models  | `list_models` |
| email   | `email_unread` `email_read` `email_search` `email_send` `schedule_email` `email_digest` (AI triage) `email_draft_reply` (AI draft, never auto-sends) — only when configured |

The agent chains tool calls until the task is done (`max_tool_iterations`
caps runaway loops).

### Memory — JARVIS actually learns who you are

Two layers, both persistent in SQLite:

- **profile** — key/value facts about *you* (name, preferences, projects).
  Injected into **every** system prompt, so every reply is personalized.
- **facts** — free-form knowledge, auto-recalled into context by relevance.

Filling them doesn't depend on the model remembering to call a tool: after
every turn an **auto-memory pass** re-reads your message with a tiny
extraction prompt (background thread, never blocks the conversation,
`"auto_memory": false` to disable) and saves anything durable. You'll see
`remembered: name = Ayyan` notes in chat as it learns. Inspect and edit
everything in the web UI's **memory** panel or with `/memory` in the
terminal; the model can also save explicitly via `profile_set` / `remember`.

`export_chat` saves the conversation as markdown (`~/.jarvis/chats` by
default; the web UI has an *export* button too).

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
traces, a stop button, voice toggle, an *export* button (chat → markdown
download), and an **IDE panel**:

- file tree + editor — browse any folder, open, edit, save in the browser
- **live agent diffs** — whenever JARVIS edits a file, the change appears
  in chat (`edit foo.py +12 -3`, clickable) and in the IDE's *changes*
  strip; selecting one shows the colored diff side-by-side with the code
- *ask jarvis* — pull the open file into the conversation and let the
  agent edit it with you

Plus **file attachments** — hit `+` (or drop a file on the composer) to
upload a PDF, image, or text file; JARVIS reads it with `read_pdf` /
`read_image` / `read_file` and answers about it.

**Premium left sidebar** — a slim icon rail with glowing JARVIS orb, clean
SVG icons grouped logically (tools, dashboards, session), and animated hover
tooltips. The active icon gets a blue glow with a left accent bar. Keeps the
UI clean while staying accessible.

**Floating windows for multitasking**:

- **hardware** — a draggable, resizable glassy window (backdrop-blur +
  layered shadows) showing a live dashboard (polls ~0.5 Hz while open): CPU/RAM/GPU
  gauges, per-core bars, sparklines, top processes, and temperatures. Hit
  **✨ AI recommendations** for the local model to read the snapshot and
  give tuning advice, or **🔥 roast my PC** for its honest opinion. CPU
  temperature from `psutil` or LibreHardwareMonitor WMI namespace. GPU via
  `nvidia-smi` if present.
- **mail** — Gmail unread list with per-message *read* / *reply* shortcuts,
  one-click **✨ AI digest** triage, and optional inbox rules (priority
  flags, smart summaries). (Shown when email is configured.)
- **tasks** — automations panel: scheduled prompts with add/cancel controls,
  equivalent to the terminal's `schedule_task` / `list_tasks` / `cancel_task`.

Keep all open at once for true multitasking — drag them around, resize from
the bottom-right corner, click one to bring it to front. Close via the × button
in the title bar.

**Slide-down drawers** (models, memory, past conversations, system info) stay
as quick toggles from the sidebar since they're transient settings rather than
always-open dashboards.

The session transcript and the change history survive page reloads.

### In-session commands

`/voice` `/speak on|off|auto` `/model [role] <name|auto>` `/models` `/tools`
`/memory` `/forget <id>` `/sessions` `/session <id>` `/tasks` `/status`
`/new` `/quit`

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
  "email": { "imap_host": "imap.gmail.com", "smtp_host": "smtp.gmail.com", "user": "you@gmail.com", "password": "app-password" },
  "email_check_minutes": 5,         // >0 = watch the inbox, notify on new mail
  "email_smart_notify": true,       // add a one-line AI summary to each new-mail alert
  "email_rules": [                  // tag/prioritize incoming mail (first match wins)
    { "name": "Boss", "from": "boss@", "priority": true }
  ]
}
```

With email configured you get Gmail automation: `email_search`, scheduled
sends (`schedule_email`, e.g. "send this at 09:00"), AI inbox triage
(`email_digest`), and AI-drafted replies (`email_draft_reply` — it drafts,
you send). With `email_check_minutes` set, a background inbox watcher
raises a notification the moment new mail lands; `email_smart_notify`
attaches a one-line AI summary, and `email_rules` tag or prioritize senders
(a ⚡ on anything matching a `priority` rule). Combine with `schedule_task`
for email-driven routines: *"every morning at 8, give me an AI digest of my
unread email and flag anything with a deadline."* — schedule it from the
**tasks** panel or just ask in chat.

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
