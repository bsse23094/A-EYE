"""Web frontend -- FastAPI server + embedded chat UI.

One process serves everything: the agent, voice, and a local web page.
The page is a thin renderer over GET /api/events (SSE) -- the same
event stream the terminal frontend consumes, so there is exactly one
turn pipeline regardless of frontend.

Endpoints:
    GET  /                  chat UI (self-contained HTML, no CDN)
    GET  /api/events        SSE event stream (tokens, thinking, tools, ...)
    POST /api/chat          {"text": ...} -> {"turn_id": ...}
    POST /api/stop          cancel the in-flight turn
    GET  /api/history       current session transcript (for page reloads)
    GET  /api/status        models/routes/voice/session snapshot
    POST /api/voice         {"on": true|false}
    POST /api/session/new   fresh conversation
    GET  /api/models        discovery summary (?refresh=1 to re-scan)
    POST /api/model         {"role": "chat", "name": "qwen3:8b"|"auto"}
    GET  /api/tools         list of registered tool names
    GET  /api/models/list   structured model list for the switcher UI
    GET  /api/workspace     directory listing  (?path=...)
    GET  /api/workspace/file   read a file     (?path=...)
    POST /api/workspace/file   save a file     {"path", "content"}
"""

from __future__ import annotations

import json
import os
import queue
import re
import socket
import threading
import webbrowser

from .assistant import Assistant
from .config import config

_TOOL_LOG_RX = re.compile(r"\n\[tools used: .*\]$", re.DOTALL)
_MAX_EDIT_BYTES = 2_000_000


def create_app(assistant: Assistant):
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, StreamingResponse

    app = FastAPI(title="JARVIS", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return PAGE

    @app.get("/api/events")
    def events():
        def stream():
            q = assistant.subscribe()
            try:
                yield f"data: {json.dumps({'type': 'hello'})}\n\n"
                while True:
                    try:
                        ev = q.get(timeout=15)
                        yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                    except queue.Empty:
                        yield ": keepalive\n\n"
            finally:
                assistant.unsubscribe(q)
        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                           "X-Accel-Buffering": "no"})

    @app.post("/api/chat")
    def chat(body: dict):
        text = (body.get("text") or "").strip()
        if not text:
            return {"error": "empty"}
        return {"turn_id": assistant.submit(text, source="text")}

    @app.post("/api/stop")
    def stop():
        assistant.agent.cancel.set()
        return {"ok": True}

    @app.get("/api/history")
    def history(limit: int = 60):
        msgs = []
        for m in assistant.memory.recent_messages(max(1, min(200, limit))):
            content = _TOOL_LOG_RX.sub("", m["content"])
            if content.strip():
                msgs.append({"role": m["role"], "content": content})
        return {"messages": msgs, "session": assistant.memory.session_id}

    @app.get("/api/status")
    def status():
        return assistant.status()

    @app.post("/api/voice")
    def voice(body: dict):
        msg = assistant.voice_on() if body.get("on") else assistant.voice_off()
        return {"message": msg, "active": assistant.voice_active}

    @app.post("/api/session/new")
    def new_session():
        return {"session": assistant.new_session()}

    @app.get("/api/models")
    def models(refresh: int = 0):
        if refresh:
            assistant.models.discover()
        assistant.models.ensure_ready()
        return {"summary": assistant.models.summary()}

    @app.post("/api/model")
    def set_model(body: dict):
        role = body.get("role", "chat")
        name = body.get("name", "auto")
        return {"message": assistant.set_model_override(role, name)}

    @app.get("/api/tools")
    def tools():
        return {"tools": assistant.registry.names()}

    # ── Workspace: browse and edit the codebase from the UI ──────

    @app.get("/api/workspace")
    def workspace(path: str = ""):
        root = os.path.abspath(os.path.expanduser(path) or os.getcwd())
        if not os.path.isdir(root):
            return {"error": f"Not a directory: {root}"}
        dirs, files = [], []
        try:
            for name in sorted(os.listdir(root), key=str.lower):
                full = os.path.join(root, name)
                if os.path.isdir(full):
                    dirs.append(name)
                else:
                    try:
                        files.append({"name": name, "size": os.path.getsize(full)})
                    except OSError:
                        files.append({"name": name, "size": 0})
        except PermissionError:
            return {"error": f"Permission denied: {root}"}
        parent = os.path.dirname(root)
        return {"path": root, "parent": parent if parent != root else None,
                "dirs": dirs[:300], "files": files[:300]}

    @app.get("/api/workspace/file")
    def workspace_read(path: str):
        full = os.path.abspath(os.path.expanduser(path))
        if not os.path.isfile(full):
            return {"error": f"Not a file: {full}"}
        if os.path.getsize(full) > _MAX_EDIT_BYTES:
            return {"error": "File too large to edit in the browser (>2 MB)."}
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as f:
                return {"path": full, "content": f.read()}
        except OSError as e:
            return {"error": str(e)}

    @app.post("/api/workspace/file")
    def workspace_write(body: dict):
        path = (body.get("path") or "").strip()
        if not path:
            return {"error": "No path."}
        full = os.path.abspath(os.path.expanduser(path))
        try:
            parent = os.path.dirname(full)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(full, "w", encoding="utf-8") as f:
                f.write(body.get("content") or "")
            return {"ok": True, "path": full}
        except OSError as e:
            return {"error": str(e)}

    @app.get("/api/models/list")
    def models_list():
        assistant.models.ensure_ready()
        items = []
        for m in sorted(assistant.models.models, key=lambda x: -x.size_b):
            items.append({
                "name": m.name,
                "size": f"{m.size_b:g}B" if m.size_b else "?",
                "caps": sorted(m.capabilities) or ["chat"],
                "provider": m.provider.name,
            })
        routes = {}
        for role in ("chat", "code", "vision"):
            picked = assistant.models.pick(role)
            routes[role] = picked.name if picked else None
        return {"models": items, "routes": routes}

    return app


def _free_port(host: str, start: int) -> int:
    for port in range(start, start + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    return start


def run(assistant: Assistant, host: str = "127.0.0.1",
        port: int = 0, open_browser: bool = True) -> None:
    import uvicorn

    port = _free_port(host, port or int(config.get("server_port", 8765)))
    url = f"http://{host}:{port}"
    app = create_app(assistant)

    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    print(f"  JARVIS is up at {url}  (Ctrl+C to stop)")

    try:
        uvicorn.run(app, host=host, port=port, log_level="warning")
    except KeyboardInterrupt:
        pass
    finally:
        assistant.shutdown()


# ---------------------------------------------------------------------------
# The UI
# ---------------------------------------------------------------------------

PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>JARVIS</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%}
body{
  background:#0c0c0c;color:#c8c8c8;
  font-family:Consolas,'Courier New',monospace;
  font-size:14px;line-height:1.6;
  display:flex;flex-direction:column;
}

::-webkit-scrollbar{width:5px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:#2a2a2a;border-radius:3px}

/* top bar */
.topbar{
  display:flex;align-items:center;gap:10px;
  padding:7px 16px;
  background:#111;border-bottom:1px solid #1e1e1e;
  flex-shrink:0;
}
.topbar .name{color:#e0e0e0;font-weight:bold;font-size:13px;letter-spacing:2px}
.topbar .sep{color:#333;margin:0 1px}
.topbar .status{color:#555;font-size:12px;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.topbar .status.ok{color:#5a5}
.topbar .status.err{color:#c44}
.topbar button{
  background:none;color:#666;border:1px solid #222;
  border-radius:4px;padding:3px 10px;cursor:pointer;
  font:inherit;font-size:11px;transition:all .15s;
}
.topbar button:hover{color:#ccc;border-color:#444}
.topbar button.on{color:#5bf;border-color:#5bf}
.topbar button.voice-listening{color:#4c4;border-color:#4c4;animation:pulse 1.5s infinite}
.topbar button.voice-speaking{color:#db4;border-color:#db4}
.topbar button.voice-loading{color:#888;border-color:#555}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}

/* drawers */
.drawer{
  background:#111;border-bottom:1px solid #1e1e1e;
  overflow:hidden;max-height:0;transition:max-height .3s ease;
  font-size:12px;color:#666;
}
.drawer.open{max-height:500px}
.drawer-inner{padding:10px 16px;display:flex;gap:24px;flex-wrap:wrap}
.drawer-inner .col{min-width:120px}
.drawer-inner .lbl{color:#555;font-size:10px;text-transform:uppercase;letter-spacing:1px;margin-bottom:2px}
.drawer-inner .val{color:#888}
.drawer-inner .val b{color:#aaa;font-weight:normal}

/* model switcher */
.model-panel{
  background:#111;border-bottom:1px solid #1e1e1e;
  overflow:hidden;max-height:0;transition:max-height .3s ease;
  font-size:12px;
}
.model-panel.open{max-height:600px}
.model-panel-inner{padding:12px 16px;max-height:400px;overflow-y:auto}
.model-panel-inner h3{color:#888;font-size:10px;text-transform:uppercase;letter-spacing:1px;margin:0 0 8px;font-weight:normal}
.model-row{
  display:flex;align-items:center;gap:8px;
  padding:5px 8px;border-radius:3px;cursor:pointer;
  transition:background .1s;
}
.model-row:hover{background:#1a1a1a}
.model-row .mname{color:#aaa;flex:1}
.model-row .msize{color:#555;font-size:11px;min-width:40px;text-align:right}
.model-row .mcaps{color:#444;font-size:10px}
.model-row .mtag{
  font-size:9px;padding:1px 5px;border-radius:2px;
  background:#1a2a1a;color:#5a5;text-transform:uppercase;letter-spacing:.5px;
}
.role-btns{display:flex;gap:4px;opacity:0;transition:opacity .15s}
.model-row:hover .role-btns{opacity:1}
.role-btns button{
  font-size:9px;padding:1px 6px;background:none;
  border:1px solid #333;color:#666;border-radius:2px;
  cursor:pointer;font:inherit;transition:all .1s;
}
.role-btns button:hover{color:#5bf;border-color:#5bf}
.role-btns button.active{color:#5a5;border-color:#5a5}

/* chat area */
#chat{flex:1;overflow-y:auto;padding:20px 0}
.inner{max-width:680px;margin:0 auto;padding:0 20px}

/* welcome */
.welcome{
  display:flex;flex-direction:column;align-items:center;
  justify-content:center;text-align:center;
  min-height:50vh;gap:16px;padding-top:8vh;
}
.welcome .mark{font-size:20px;letter-spacing:6px;color:#e0e0e0;font-weight:bold}
.welcome .sub{color:#555;font-size:13px;max-width:440px;line-height:1.7}
.welcome .caps{
  display:grid;grid-template-columns:1fr 1fr;gap:6px;
  max-width:420px;width:100%;margin-top:4px;
}
.welcome .cap{
  text-align:left;padding:10px 12px;
  border:1px solid #1a1a1a;border-radius:4px;
  cursor:pointer;transition:all .15s;
}
.welcome .cap:hover{border-color:#333;background:#141414}
.welcome .cap .cap-title{color:#999;font-size:12px;margin-bottom:2px}
.welcome .cap .cap-desc{color:#555;font-size:11px;line-height:1.4}

/* ai orb */
.orb-wrap{
  position:relative;width:100px;height:100px;margin-bottom:8px;
  animation:orb-float 4s ease-in-out infinite;
}
@keyframes orb-float{0%,100%{transform:translateY(0)}50%{transform:translateY(-6px)}}

.orb-svg{width:100px;height:100px;display:block}

/* rotating rings */
.ring-outer{animation:spin-cw 12s linear infinite;transform-origin:50px 50px}
.ring-mid{animation:spin-ccw 8s linear infinite;transform-origin:50px 50px}
.ring-inner{animation:spin-cw 5s linear infinite;transform-origin:50px 50px}
@keyframes spin-cw{to{transform:rotate(360deg)}}
@keyframes spin-ccw{to{transform:rotate(-360deg)}}

/* core pulse */
.core-glow{animation:core-pulse 2.5s ease-in-out infinite}
@keyframes core-pulse{
  0%,100%{opacity:.6;r:6}
  50%{opacity:1;r:8}
}
.core-flare{animation:flare 2.5s ease-in-out infinite}
@keyframes flare{
  0%,100%{opacity:.15;r:18}
  50%{opacity:.3;r:22}
}

/* particles orbiting */
.particle{animation:orbit 3s linear infinite;transform-origin:50px 50px}
.particle:nth-child(2){animation-delay:-1s;animation-duration:4s}
.particle:nth-child(3){animation-delay:-2s;animation-duration:3.5s}
@keyframes orbit{to{transform:rotate(360deg)}}

/* ground glow */
.orb-glow{
  position:absolute;bottom:-6px;left:50%;transform:translateX(-50%);
  width:60px;height:8px;border-radius:50%;
  background:radial-gradient(ellipse, rgba(80,180,255,.12) 0%, transparent 70%);
  animation:orb-float 4s ease-in-out infinite;
}

/* messages */
.msg{margin:0 0 16px}
.msg .tag{font-size:10px;color:#555;letter-spacing:1px;text-transform:uppercase;margin-bottom:4px}
.msg.user .bubble{
  background:#161616;border:1px solid #222;
  border-radius:6px;padding:8px 12px;
  display:inline-block;max-width:90%;color:#ccc;
}
.msg.bot .bubble{color:#c8c8c8}

/* markdown in bot */
.msg.bot .bubble pre{
  background:#0a0a0a;border:1px solid #1e1e1e;
  border-radius:4px;padding:10px 12px;overflow-x:auto;
  font-size:13px;margin:8px 0;line-height:1.5;
}
.msg.bot .bubble code{background:#151515;padding:1px 4px;border-radius:2px;font-size:13px;color:#8be}
.msg.bot .bubble pre code{background:none;padding:0;color:#c8c8c8}
.msg.bot .bubble b,.msg.bot .bubble strong{color:#e8e8e8;font-weight:bold}
.msg.bot .bubble a{color:#5bf;text-decoration:none}
.msg.bot .bubble a:hover{text-decoration:underline}
.msg.bot .bubble ul,.msg.bot .bubble ol{padding-left:18px;margin:4px 0}
.msg.bot .bubble li{margin:2px 0}
.msg.bot .bubble blockquote{border-left:2px solid #333;padding-left:10px;color:#888;margin:6px 0}
.msg.bot .bubble hr{border:none;border-top:1px solid #1e1e1e;margin:10px 0}
.msg.bot .bubble h1,.msg.bot .bubble h2,.msg.bot .bubble h3{color:#ddd;margin:8px 0 4px;font-size:15px}
.msg.bot .bubble table{border-collapse:collapse;width:100%;margin:8px 0;font-size:13px}
.msg.bot .bubble th{text-align:left;padding:6px 10px;border:1px solid #222;background:#151515;color:#aaa}
.msg.bot .bubble td{padding:6px 10px;border:1px solid #1e1e1e}

/* thinking block */
.think-block{
  margin:6px 0;border:1px solid #1a1a1a;border-radius:4px;
  background:#0a0a0a;font-size:12px;
}
.think-block summary{
  cursor:pointer;padding:5px 10px;color:#555;
  list-style:none;user-select:none;display:flex;align-items:center;gap:6px;
}
.think-block summary::-webkit-details-marker{display:none}
.think-block summary::before{content:"\25b6";font-size:8px;color:#444;transition:transform .15s}
.think-block[open] summary::before{transform:rotate(90deg)}
.think-block .think-body{
  padding:8px 12px;color:#666;border-top:1px solid #1a1a1a;
  line-height:1.5;white-space:pre-wrap;word-break:break-word;
}

/* tools */
.tool{
  font-size:12px;color:#555;
  border-left:2px solid #1e1e1e;
  margin:3px 0 3px 4px;padding-left:10px;
  white-space:pre-wrap;word-break:break-all;
}
.tool .fn{color:#5a8}
.tool .res{color:#555}

.meta{font-size:10px;color:#444;margin-top:4px}
.note{font-size:12px;color:#b90;margin:6px 0;padding:4px 0}
.err-banner{font-size:12px;color:#c44;margin:6px 0;padding:4px 0}

.typing::after{content:"_";animation:blink .8s step-start infinite;color:#5bf}
@keyframes blink{50%{opacity:0}}

.dots span{display:inline-block;width:4px;height:4px;background:#555;border-radius:50%;margin:0 2px;animation:dot 1s ease infinite}
.dots span:nth-child(2){animation-delay:.15s}
.dots span:nth-child(3){animation-delay:.3s}
@keyframes dot{0%,80%,100%{opacity:.2}40%{opacity:1}}

/* composer */
.composer{
  border-top:1px solid #1e1e1e;background:#111;
  padding:10px 16px;flex-shrink:0;
}
.compose-row{
  max-width:680px;margin:0 auto;
  display:flex;align-items:flex-end;gap:8px;
}
#box{
  flex:1;resize:none;
  background:#0c0c0c;color:#c8c8c8;
  border:1px solid #222;border-radius:4px;
  padding:8px 12px;font:inherit;outline:none;
  max-height:140px;line-height:1.5;
}
#box:focus{border-color:#333}
#box::placeholder{color:#444}
#sendbtn{
  background:#222;color:#888;border:none;
  border-radius:4px;padding:8px 14px;
  cursor:pointer;font:inherit;font-size:12px;
  transition:all .15s;flex-shrink:0;
}
#sendbtn:hover{background:#2a2a2a;color:#ccc}
#sendbtn:disabled{opacity:.3;cursor:default}
#sendbtn.stop{background:#3a1818;color:#c66}
#sendbtn.stop:hover{background:#4a1d1d;color:#e88}
.compose-hint{
  max-width:680px;margin:4px auto 0;
  font-size:10px;color:#333;text-align:center;
}

/* workspace panel */
.ws{
  position:fixed;top:0;right:0;bottom:0;width:520px;max-width:100%;
  background:#101010;border-left:1px solid #222;z-index:20;
  display:flex;flex-direction:column;
  transform:translateX(100%);transition:transform .25s ease;
}
.ws.open{transform:translateX(0)}
.ws-head{
  display:flex;align-items:center;gap:8px;padding:8px 12px;
  border-bottom:1px solid #1e1e1e;background:#111;flex-shrink:0;
}
.ws-head .ws-title{color:#888;font-size:11px;text-transform:uppercase;letter-spacing:1px}
#wspath{
  flex:1;background:#0c0c0c;color:#9a9;border:1px solid #222;
  border-radius:3px;padding:4px 8px;font:inherit;font-size:11px;outline:none;
}
#wspath:focus{border-color:#333}
.ws button{
  background:none;color:#666;border:1px solid #222;border-radius:3px;
  padding:3px 9px;cursor:pointer;font:inherit;font-size:11px;transition:all .15s;
}
.ws button:hover{color:#ccc;border-color:#444}
.ws-body{flex:1;display:flex;flex-direction:column;overflow:hidden}
.ws-list{flex:1;overflow-y:auto;padding:6px 0}
.ws-row{
  padding:3px 14px;cursor:pointer;font-size:12px;color:#888;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
}
.ws-row:hover{background:#181818;color:#ccc}
.ws-row.dir{color:#7ab}
.ws-row .fsize{color:#444;font-size:10px;margin-left:8px}
.ws-err{color:#c44;font-size:12px;padding:10px 14px}
.ws-edit{flex:1;display:flex;flex-direction:column;overflow:hidden}
.ws-file-bar{
  display:flex;align-items:center;gap:6px;padding:6px 12px;
  border-bottom:1px solid #1e1e1e;flex-shrink:0;
}
.ws-file-bar #wsfile{
  flex:1;color:#9a9;font-size:11px;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
}
.ws-file-bar .dirty{color:#db4}
#wstext{
  flex:1;background:#0c0c0c;color:#c8c8c8;border:none;outline:none;
  padding:10px 14px;font:inherit;font-size:13px;line-height:1.5;
  resize:none;white-space:pre;tab-size:4;
}

@media(max-width:600px){
  .welcome .caps{grid-template-columns:1fr}
  .topbar{gap:6px;padding:6px 10px}
  .topbar .status{display:none}
  .ws{width:100%}
}
</style>
</head>
<body>

<div class="topbar">
  <span class="name">JARVIS</span>
  <span class="sep">/</span>
  <span class="status" id="chip">starting&hellip;</span>
  <button id="wsbtn" title="browse and edit files">files</button>
  <button id="modelsbtn" title="switch models">models</button>
  <button id="infobtn" title="system info">info</button>
  <button id="micbtn" title="toggle voice">voice</button>
  <button id="newbtn" title="new session">new</button>
</div>

<div class="ws" id="ws">
  <div class="ws-head">
    <span class="ws-title">workspace</span>
    <input id="wspath" spellcheck="false" placeholder="path...">
    <button id="wsclose">close</button>
  </div>
  <div class="ws-body">
    <div class="ws-list" id="wslist"></div>
    <div class="ws-edit" id="wsedit" style="display:none">
      <div class="ws-file-bar">
        <span id="wsfile"></span>
        <button id="wsask" title="reference this file in chat">ask jarvis</button>
        <button id="wssave">save</button>
        <button id="wsback">back</button>
      </div>
      <textarea id="wstext" spellcheck="false"></textarea>
    </div>
  </div>
</div>

<div class="model-panel" id="modelpanel"><div class="model-panel-inner" id="modellist">loading...</div></div>
<div class="drawer" id="drawer"><div class="drawer-inner" id="dinfo"></div></div>

<div id="chat">
  <div class="inner" id="log">
    <div class="welcome" id="welcome">
      <div class="orb-wrap">
        <svg class="orb-svg" viewBox="0 0 100 100">
          <defs>
            <radialGradient id="cg" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stop-color="#5bf" stop-opacity=".4"/>
              <stop offset="100%" stop-color="#5bf" stop-opacity="0"/>
            </radialGradient>
          </defs>
          <!-- ambient flare -->
          <circle class="core-flare" cx="50" cy="50" r="20" fill="url(#cg)"/>
          <!-- outer ring -->
          <circle class="ring-outer" cx="50" cy="50" r="42" fill="none" stroke="rgba(80,180,255,.12)" stroke-width=".6" stroke-dasharray="8 12"/>
          <!-- mid ring -->
          <circle class="ring-mid" cx="50" cy="50" r="33" fill="none" stroke="rgba(80,180,255,.2)" stroke-width=".7" stroke-dasharray="5 10 2 10"/>
          <!-- inner ring -->
          <circle class="ring-inner" cx="50" cy="50" r="22" fill="none" stroke="rgba(100,200,255,.25)" stroke-width=".8" stroke-dasharray="3 7"/>
          <!-- orbiting particles -->
          <g class="particle"><circle cx="50" cy="8" r="1.2" fill="rgba(120,200,255,.5)"/></g>
          <g class="particle"><circle cx="92" cy="50" r="1" fill="rgba(120,200,255,.35)"/></g>
          <g class="particle"><circle cx="20" cy="80" r=".8" fill="rgba(120,200,255,.3)"/></g>
          <!-- core -->
          <circle class="core-glow" cx="50" cy="50" r="6" fill="rgba(140,220,255,.7)"/>
          <circle cx="50" cy="50" r="3" fill="#fff" opacity=".8"/>
        </svg>
        <div class="orb-glow"></div>
      </div>
      <div class="mark">JARVIS</div>
      <div class="sub">your local AI assistant. controls files, shell, git, github, system, web, memory, and schedules.</div>
      <div class="caps">
        <div class="cap" onclick="q('what processes are eating my RAM?')">
          <div class="cap-title">system</div>
          <div class="cap-desc">monitor RAM, CPU, battery, running processes</div>
        </div>
        <div class="cap" onclick="q('summarize recent git commits')">
          <div class="cap-title">git &amp; github</div>
          <div class="cap-desc">commit, push, pull, PRs, list repos</div>
        </div>
        <div class="cap" onclick="q('what\'s the weather?')">
          <div class="cap-title">web &amp; search</div>
          <div class="cap-desc">weather, news, web search, fetch URLs</div>
        </div>
        <div class="cap" onclick="q('list files in this directory')">
          <div class="cap-title">files &amp; code</div>
          <div class="cap-desc">read, write, edit files, run commands</div>
        </div>
      </div>
    </div>
  </div>
</div>

<div class="composer">
  <div class="compose-row">
    <textarea id="box" rows="1" placeholder="message jarvis&hellip;"></textarea>
    <button id="sendbtn">send</button>
  </div>
  <div class="compose-hint">enter to send &middot; shift+enter for newline</div>
</div>

<script>
var log=document.getElementById("log"),
    chat=document.getElementById("chat"),
    box=document.getElementById("box"),
    chip=document.getElementById("chip"),
    sendbtn=document.getElementById("sendbtn"),
    micbtn=document.getElementById("micbtn"),
    newbtn=document.getElementById("newbtn"),
    infobtn=document.getElementById("infobtn"),
    modelsbtn=document.getElementById("modelsbtn"),
    drawer=document.getElementById("drawer"),
    modelpanel=document.getElementById("modelpanel"),
    modellist=document.getElementById("modellist"),
    dinfo=document.getElementById("dinfo"),
    wel=document.getElementById("welcome"),
    cur=null,busy=false;

function esc(s){return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")}

function md(s){
  var h=esc(s);
  /* code blocks */
  h=h.replace(/```(\w*)\n?([\s\S]*?)```/g,function(_,l,c){return'<pre><code>'+c+'</code></pre>'});
  h=h.replace(/`([^`\n]+)`/g,'<code>$1</code>');
  /* headings */
  h=h.replace(/^#### (.+)$/gm,'<h4>$1</h4>');
  h=h.replace(/^### (.+)$/gm,'<h3>$1</h3>');
  h=h.replace(/^## (.+)$/gm,'<h2>$1</h2>');
  h=h.replace(/^# (.+)$/gm,'<h1>$1</h1>');
  /* bold/italic */
  h=h.replace(/\*\*\*(.+?)\*\*\*/g,'<b><em>$1</em></b>');
  h=h.replace(/\*\*(.+?)\*\*/g,'<b>$1</b>');
  h=h.replace(/(?<![\w*])\*([^*\n]+)\*(?![\w*])/g,'<em>$1</em>');
  /* blockquote */
  h=h.replace(/^&gt; (.+)$/gm,'<blockquote>$1</blockquote>');
  /* hr */
  h=h.replace(/^---$/gm,'<hr>');
  /* numbered list */
  h=h.replace(/(^\d+\. .+$(?:\n\d+\. .+$)*)/gm,function(block){
    var items=block.split('\n').map(function(l){return '<li>'+l.replace(/^\d+\.\s/,'')+'</li>'}).join('');
    return '<ol>'+items+'</ol>';
  });
  /* bullet list */
  h=h.replace(/(^[-*] .+$(?:\n[-*] .+$)*)/gm,function(block){
    var items=block.split('\n').map(function(l){return '<li>'+l.replace(/^[-*]\s/,'')+'</li>'}).join('');
    return '<ul>'+items+'</ul>';
  });
  /* tables */
  h=h.replace(/(^\|.+\|$(?:\n\|.+\|$)+)/gm,function(block){
    var rows=block.split('\n').filter(function(r){return r.trim()});
    if(rows.length<2)return block;
    var html='<table>';
    var hdr=rows[0].split('|').filter(function(c){return c.trim()!==''}).map(function(c){return c.trim()});
    html+='<tr>'+hdr.map(function(c){return '<th>'+c+'</th>'}).join('')+'</tr>';
    for(var i=2;i<rows.length;i++){
      var cells=rows[i].split('|').filter(function(c){return c.trim()!==''}).map(function(c){return c.trim()});
      html+='<tr>'+cells.map(function(c){return '<td>'+c+'</td>'}).join('')+'</tr>';
    }
    html+='</table>';return html;
  });
  /* links */
  h=h.replace(/\[([^\]]+)\]\(([^)]+)\)/g,'<a href="$2" target="_blank">$1</a>');
  /* paragraphs: convert double newlines */
  h=h.replace(/\n{2,}/g,'</p><p>');
  if(h.indexOf('</p><p>')!==-1)h='<p>'+h+'</p>';
  return h;
}

/* thinking blocks — streamed from the model via "thinking" events */
function thinkTok(t){
  if(!cur)return;
  if(!cur.thinkEl){
    var d=document.createElement("details");
    d.className="think-block";d.open=true;
    d.innerHTML='<summary>thinking...</summary><div class="think-body"></div>';
    cur.el.insertBefore(d,cur.body);
    cur.thinkEl=d;
  }
  cur.thinkEl.querySelector(".think-body").textContent+=t;
  scroll();
}

function closeThink(final_){
  if(cur&&cur.thinkEl){
    cur.thinkEl.open=false;
    cur.thinkEl.querySelector("summary").textContent="thought process";
    if(!final_)cur.thinkEl=null; /* next thinking phase gets a fresh block */
  }
}

function scroll(){requestAnimationFrame(function(){chat.scrollTop=chat.scrollHeight})}
function hide_wel(){if(wel&&wel.parentNode){wel.remove()}}

function addUser(text,source){
  hide_wel();
  var d=document.createElement("div");d.className="msg user";
  var tag=source==="voice"?"you (voice)":source==="schedule"?"scheduled":"you";
  d.innerHTML='<div class="tag">'+tag+'</div><div class="bubble"></div>';
  d.querySelector(".bubble").textContent=text;
  log.appendChild(d);scroll();
}

function startBot(tid){
  hide_wel();
  var d=document.createElement("div");d.className="msg bot";
  d.innerHTML='<div class="tag">jarvis</div><div class="bubble"><span class="dots"><span></span><span></span><span></span></span></div>';
  log.appendChild(d);
  var b=d.querySelector(".bubble");
  cur={el:d,body:b,tid:tid,txt:"",wait:true,thinkEl:null};
  scroll();
}

function need(tid){if(!cur||cur.tid!==tid)startBot(tid);return cur}

function tok(t){
  closeThink(true);
  if(cur.wait){cur.body.innerHTML="";cur.wait=false}
  cur.txt+=t;
  cur.body.innerHTML=md(cur.txt);
  cur.body.classList.add("typing");
  scroll();
}

function addTool(html){
  var d=document.createElement("div");d.className="tool";d.innerHTML=html;
  (cur?cur.el:log).appendChild(d);scroll();
}

function banner(cls,icon,txt){
  hide_wel();
  var d=document.createElement("div");d.className=cls;d.textContent=icon+" "+txt;
  (cur?cur.el:log).appendChild(d);scroll();
}

/* SSE */
var es=new EventSource("/api/events");
es.onmessage=function(m){
  var ev;try{ev=JSON.parse(m.data)}catch(e){return}
  switch(ev.type){
    case"hello":refreshStatus();break;
    case"user":
      addUser(ev.text,ev.source);startBot(ev.turn_id);setBusy(true);break;
    case"token":
      need(ev.turn_id);tok(ev.text);break;
    case"thinking":
      need(ev.turn_id);thinkTok(ev.text);break;
    case"tool_start":
      need(ev.turn_id);closeThink(false);
      addTool('<span class="fn">'+esc(ev.name)+'</span> '+esc(String(ev.args)));break;
    case"tool_result":
      addTool('<span class="res">  '+esc(String(ev.preview||""))+'</span>');break;
    case"done":
      if(cur&&cur.tid===ev.turn_id){
        closeThink(true);
        cur.body.classList.remove("typing");
        if(cur.wait){cur.body.innerHTML=ev.text?md(ev.text):"";cur.wait=false}
        else if(!cur.txt&&ev.text){cur.body.innerHTML=md(ev.text)}
        var mt=document.createElement("div");mt.className="meta";
        var p=[];
        if(ev.model)p.push(ev.model);
        if(ev.tools_used)p.push(ev.tools_used+" tool"+(ev.tools_used>1?"s":""));
        mt.textContent=p.join(" / ");
        cur.el.appendChild(mt);cur=null;
      }
      setBusy(false);refreshStatus();break;
    case"error":
      if(cur){closeThink(true);cur.body.classList.remove("typing");cur=null}
      banner("err-banner","!",ev.text);setBusy(false);break;
    case"notify":banner("note","*",ev.text);break;
    case"voice":voiceUI(ev.state);break;
  }
};
es.onerror=function(){chip.textContent="reconnecting...";chip.className="status err"};

/* status */
function refreshStatus(){
  fetch("/api/status").then(function(r){return r.json()}).then(function(s){
    var r=s.routes||{};
    if(s.models===null){chip.textContent="discovering models...";chip.className="status";return}
    if(s.errors&&s.errors.length&&!s.models){chip.textContent="! "+s.errors[0];chip.className="status err";return}
    chip.textContent=s.models+" models / "+s.tools+" tools / session #"+s.session;
    chip.className="status ok";
    dinfo.innerHTML=
      '<div class="col"><div class="lbl">routes</div>'+
        '<div class="val">chat <b>'+(r.chat||"-")+'</b></div>'+
        '<div class="val">code <b>'+(r.code||"-")+'</b></div>'+
        '<div class="val">vision <b>'+(r.vision||"-")+'</b></div></div>'+
      '<div class="col"><div class="lbl">stats</div>'+
        '<div class="val"><b>'+s.models+'</b> models</div>'+
        '<div class="val"><b>'+s.tools+'</b> tools</div>'+
        '<div class="val"><b>'+s.facts+'</b> facts</div></div>'+
      '<div class="col"><div class="lbl">voice</div>'+
        '<div class="val">'+(s.voice?"on":"off")+'</div>'+
        '<div class="lbl" style="margin-top:6px">speak</div>'+
        '<div class="val">'+s.speak+'</div></div>';
    if(!micbtn._t)voiceUI(s.voice?"on":"off");
  }).catch(function(){chip.textContent="unreachable";chip.className="status err"});
}
refreshStatus();setInterval(refreshStatus,15000);

/* voice UI */
function voiceUI(st){
  var on=st&&st!=="off"&&st!=="unavailable";
  micbtn.className="";
  if(st==="listening"){micbtn.className="voice-listening";micbtn.textContent="listening..."}
  else if(st==="speaking"){micbtn.className="voice-speaking";micbtn.textContent="speaking..."}
  else if(st==="loading"){micbtn.className="voice-loading";micbtn.textContent="loading..."}
  else if(st==="transcribing"){micbtn.className="voice-loading";micbtn.textContent="thinking..."}
  else if(on){micbtn.className="on";micbtn.textContent="voice on"}
  else{micbtn.textContent="voice"}
}

/* model switcher */
function loadModels(){
  fetch("/api/models/list").then(function(r){return r.json()}).then(function(data){
    var models=data.models||[];
    var routes=data.routes||{};
    if(!models.length){modellist.innerHTML="<span style='color:#555'>no models found</span>";return}
    var html="<h3>click a role button to assign a model</h3>";
    models.forEach(function(m){
      var tags=[];
      ["chat","code","vision"].forEach(function(role){
        if(routes[role]===m.name)tags.push('<span class="mtag">'+role+'</span>');
      });
      html+='<div class="model-row">';
      html+='<span class="mname">'+esc(m.name)+'</span>';
      html+=tags.join(" ");
      html+='<span class="mcaps">'+m.caps.join(", ")+'</span>';
      html+='<span class="msize">'+esc(m.size)+'</span>';
      html+='<span class="role-btns">';
      ["chat","code","vision"].forEach(function(role){
        var cls=routes[role]===m.name?" active":"";
        html+='<button class="'+cls+'" onclick="setRole(\''+role+'\',\''+m.name.replace(/'/g,"\\'")+'\')">'+role+'</button>';
      });
      html+='</span></div>';
    });
    modellist.innerHTML=html;
  }).catch(function(){modellist.innerHTML="<span style='color:#c44'>failed to load</span>"});
}

function setRole(role,name){
  fetch("/api/model",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({role:role,name:name})})
    .then(function(r){return r.json()}).then(function(r){
      banner("note","*",r.message||"model updated");
      loadModels();refreshStatus();
    }).catch(function(){});
}

/* send / stop */
function setBusy(b){
  busy=b;
  sendbtn.textContent=b?"stop":"send";
  sendbtn.className=b?"stop":"";
}
function send(){
  if(busy)return;
  var t=box.value.trim();if(!t)return;
  box.value="";box.style.height="auto";setBusy(true);
  fetch("/api/chat",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({text:t})})
    .catch(function(){setBusy(false)});
}
sendbtn.onclick=function(){
  if(busy){fetch("/api/stop",{method:"POST"}).catch(function(){})}
  else send();
};
box.addEventListener("keydown",function(e){if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();send()}});
box.addEventListener("input",function(){box.style.height="auto";box.style.height=Math.min(box.scrollHeight,140)+"px"});

/* quick send */
function q(t){box.value=t;send()}

/* voice toggle */
micbtn.onclick=function(){
  micbtn._t=true;var on=!micbtn.className.match(/on|listening|speaking/);
  fetch("/api/voice",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({on:on})})
    .then(function(r){return r.json()}).then(function(r){voiceUI(r.active?"on":"off");if(r.message)banner("note","*",r.message)})
    .catch(function(){});
};

/* new session */
newbtn.onclick=function(){
  fetch("/api/session/new",{method:"POST"}).then(function(){
    log.innerHTML="";cur=null;refreshStatus();
  }).catch(function(){});
};

/* drawer toggles */
infobtn.onclick=function(){
  drawer.classList.toggle("open");
  if(modelpanel.classList.contains("open"))modelpanel.classList.remove("open");
};
modelsbtn.onclick=function(){
  modelpanel.classList.toggle("open");
  if(drawer.classList.contains("open"))drawer.classList.remove("open");
  if(modelpanel.classList.contains("open"))loadModels();
};

/* history restore on reload */
fetch("/api/history").then(function(r){return r.json()}).then(function(d){
  var ms=d.messages||[];
  if(!ms.length)return;
  hide_wel();
  ms.forEach(function(m){
    if(m.role==="user"){addUser(m.content,"text")}
    else{
      var div=document.createElement("div");div.className="msg bot";
      div.innerHTML='<div class="tag">jarvis</div><div class="bubble"></div>';
      div.querySelector(".bubble").innerHTML=md(m.content);
      log.appendChild(div);
    }
  });
  scroll();
}).catch(function(){});

/* ── workspace: browse + edit files ─────────────────────────── */
var ws=document.getElementById("ws"),
    wsbtn=document.getElementById("wsbtn"),
    wsclose=document.getElementById("wsclose"),
    wspath=document.getElementById("wspath"),
    wslist=document.getElementById("wslist"),
    wsedit=document.getElementById("wsedit"),
    wsfile=document.getElementById("wsfile"),
    wstext=document.getElementById("wstext"),
    wssave=document.getElementById("wssave"),
    wsask=document.getElementById("wsask"),
    wsback=document.getElementById("wsback"),
    wsLoaded=false,wsDirty=false,wsCurFile="";

function fsize(n){return n>1048576?(n/1048576).toFixed(1)+"M":n>1024?Math.round(n/1024)+"K":n+"B"}

function wsRow(label,cls,onclick,size){
  var d=document.createElement("div");d.className="ws-row "+cls;
  d.textContent=label;
  if(size!==undefined){
    var s=document.createElement("span");s.className="fsize";s.textContent=fsize(size);
    d.appendChild(s);
  }
  d.onclick=onclick;
  return d;
}

function wsLoad(p){
  fetch("/api/workspace?path="+encodeURIComponent(p||"")).then(function(r){return r.json()}).then(function(d){
    if(d.error){wslist.innerHTML='<div class="ws-err"></div>';wslist.firstChild.textContent=d.error;return}
    wspath.value=d.path;
    var sep=d.path.indexOf("\\")>=0||d.path.indexOf(":")===1?"\\":"/";
    wslist.innerHTML="";
    if(d.parent)wslist.appendChild(wsRow("..","dir",function(){wsLoad(d.parent)}));
    d.dirs.forEach(function(n){
      wslist.appendChild(wsRow(n+"/","dir",function(){wsLoad(d.path+sep+n)}));
    });
    d.files.forEach(function(f){
      wslist.appendChild(wsRow(f.name,"file",function(){wsOpen(d.path+sep+f.name)},f.size));
    });
    wsedit.style.display="none";wslist.style.display="";
  }).catch(function(){});
}

function wsOpen(p){
  fetch("/api/workspace/file?path="+encodeURIComponent(p)).then(function(r){return r.json()}).then(function(d){
    if(d.error){banner("err-banner","!",d.error);return}
    wsCurFile=d.path;wsDirty=false;
    wsfile.textContent=d.path;wsfile.className="";
    wstext.value=d.content;
    wslist.style.display="none";wsedit.style.display="";
    wstext.focus();
  }).catch(function(){});
}

wstext.addEventListener("input",function(){
  if(!wsDirty){wsDirty=true;wsfile.className="dirty"}
});

wssave.onclick=function(){
  fetch("/api/workspace/file",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({path:wsCurFile,content:wstext.value})})
    .then(function(r){return r.json()}).then(function(d){
      if(d.error){banner("err-banner","!",d.error);return}
      wsDirty=false;wsfile.className="";
      banner("note","*","saved "+wsCurFile);
    }).catch(function(){});
};

wsask.onclick=function(){
  box.value="Regarding the file "+wsCurFile+": "+box.value;
  box.focus();
  box.setSelectionRange(box.value.length,box.value.length);
};

wsback.onclick=function(){
  if(wsDirty&&!confirm("Discard unsaved changes?"))return;
  wsedit.style.display="none";wslist.style.display="";
};

wspath.addEventListener("keydown",function(e){
  if(e.key==="Enter")wsLoad(wspath.value.trim());
});

wsbtn.onclick=function(){
  ws.classList.toggle("open");
  if(ws.classList.contains("open")&&!wsLoaded){wsLoaded=true;wsLoad("")}
};
wsclose.onclick=function(){ws.classList.remove("open")};

box.focus();
</script>
</body>
</html>
"""
