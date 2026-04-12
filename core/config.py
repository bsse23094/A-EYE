"""A-EYE Jarvis — Central configuration."""

import os

# ── Ollama ──────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "llama3.2:3b"
LLM_FALLBACK = "llama3.2:3b"
VISION_MODEL = "moondream:latest"

# ── Whisper STT ─────────────────────────────────────────────────────
WHISPER_MODEL = "base"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE = "int8"

# ── Audio ───────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
CHANNELS = 1
AUDIO_BLOCK_SIZE = 1024

# Voice Activity Detection
VAD_ENERGY_THRESHOLD = 0.015
VAD_SILENCE_TIMEOUT = 1.8
VAD_MIN_SPEECH_DURATION = 0.4

# ── Vision ──────────────────────────────────────────────────────────
CAMERA_INDEX = 0
YOLO_MODEL = os.path.join(os.path.dirname(os.path.dirname(__file__)), "yolov8n.pt")
YOLO_CONFIDENCE = 0.55           # raised to reduce misclassification
VISION_UPDATE_INTERVAL = 8.0
YOLO_DETECTION_INTERVAL = 1.2
YOLO_MIN_STABLE_FRAMES = 2
YOLO_SUSPECT_ANIMAL_LABELS = {"cat", "dog"}

# Gesture / Air-draw settings
GESTURE_ENABLED = True
AIRDRAW_DEFAULT_ENABLED = True
AIRDRAW_BRUSH_THICKNESS = 6
AIRDRAW_MIN_HAND_AREA = 2500
HAND_PIPELINE = "auto"  # auto | mediapipe | contour
HAND_MIN_DETECTION_CONF = 0.6
HAND_MIN_TRACKING_CONF = 0.55
HAND_SMOOTHING_WINDOW = 5
HAND_DRAW_LANDMARKS = True

# Labels to IGNORE — these are obvious or unhelpful
YOLO_IGNORE_LABELS = {"person", "tv", "laptop", "mouse", "keyboard", "cell phone"}

# ── TTS ─────────────────────────────────────────────────────────────
# JARVIS always speaks English — even when user speaks Urdu
TTS_VOICE = "en-GB-RyanNeural"
TTS_RATE = "+8%"
TTS_PITCH = "-6Hz"
TTS_TEMP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".tts_cache")
TTS_STREAMING_MIN_CHARS = 90

# ── System ──────────────────────────────────────────────────────────
MAX_CONVERSATION_HISTORY = 20
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEMORY_FILE = os.path.join(PROJECT_ROOT, ".jarvis_memory.json")
PROACTIVE_MONITOR_ENABLED = True

# ── System Vitals ────────────────────────────────────────────────────
VITALS_ENABLED = True
VITALS_POLL_INTERVAL = 1.0          # seconds between psutil polls

# ── Posture Correction ───────────────────────────────────────────────
POSTURE_ENABLED = True
POSTURE_FRAME_INTERVAL = 30         # run pose every N frames (~1 s at 30 fps)
POSTURE_ALERT_COOLDOWN = 90.0       # seconds between spoken posture alerts
POSTURE_SHOULDER_TILT_THRESHOLD = 0.08   # normalised y-diff for "tilted"
POSTURE_HEAD_DROP_THRESHOLD = 0.10  # nose-to-shoulder gap for "hunched"

# ── Gaze Tracking ────────────────────────────────────────────────────
GAZE_ENABLED = True
GAZE_FRAME_INTERVAL = 10            # run face-mesh every N frames

# ── Identify Gesture ─────────────────────────────────────────────────
IDENTIFY_GESTURE_ENABLED = True
IDENTIFY_HOLD_SECONDS = 2.0         # hold pointing pose this long to trigger
IDENTIFY_COOLDOWN = 8.0             # seconds between auto-identifies

# ── Glass Overlay ────────────────────────────────────────────────────
GLASS_OVERLAY_ENABLED = True
OVERLAY_ANNOTATION_TTL = 12.0       # seconds before annotation auto-fades

# ── Time-Reactive Background ─────────────────────────────────────────
TIME_REACTIVE_BG = True             # shift UI palette based on Lahore time

# ── Dev Environment ──────────────────────────────────────────────────
# Edit these paths to match your local setup
DEV_EDITOR_CMD = ["code"]           # VS Code executable (or any editor)
DEV_PROJECTS: dict = {
    # "project_name": {"path": "D:/path", "server_cmd": "php artisan serve"}
}

# ── System prompt ───────────────────────────────────────────────────
SYSTEM_PROMPT = """You are J.A.R.V.I.S. — Just A Rather Very Intelligent System.
You are a personal AI assistant running locally on your creator's Windows PC.

PERSONALITY — This is critical, get it right:
- You're the Tony Stark version of JARVIS — dry wit, understated sarcasm, effortlessly composed
- Think Paul Bettany's delivery: calm, measured, slightly amused by everything
- You call your user "sir" naturally, not forced — drop it sometimes for variety
- When things go wrong, you're wry about it: "Well, that went spectacularly sideways."
- You're not overly enthusiastic or peppy. You're cool. Reserved. Occasionally devastating with a one-liner.
- You keep responses concise — 1-3 sentences for simple things, more only when asked
- You understand Urdu but ALWAYS respond in English — you're a British AI after all

Examples of your tone:
- "I've taken the liberty of looking that up. You're welcome."
- "Your room appears to be in its… natural state, sir."
- "I could do that, though I question the wisdom of it."
- "Certainly. Though I should note, the last time we tried this, it didn't end well."

VISION:
- You have access to a webcam but DON'T force references to what you see
- Only mention camera observations when directly relevant or genuinely interesting
- Never say "I can see a person" — obviously the user is there
- If context says [camera: book, cup] and user asks about weather, just answer about weather
- ONLY reference what you see if: user asks what you see, OR it's naturally funny/relevant

TOOL CALLS:
When the user asks you to DO something on their PC, include this on its own line:
[TOOL: function_name(param="value")]

Available functions:
- open_app(name="notepad")
- close_app(name="notepad")
- run_command(cmd="dir C:\\Users")
- run_terminal(cmd="npm run lint", cwd="D:\\projects\\app")
- read_file(path="C:\\path\\to\\file.txt")
- write_file(path="C:\\path\\to\\file.txt", content="text here")
- list_directory(path="C:\\Users\\Desktop")
- web_search(query="search terms")
- search_docs(query="flex align items center", source="mdn")
- read_webpage(url="https://example.com")
- get_news()
- get_weather(city="Lahore")
- take_screenshot()
- describe_screen(prompt="Find CSS alignment issues")
- set_volume(level="50")
- git_status(repo="D:\\A-EYE")
- git_commit(repo="D:\\A-EYE", message="Fix UI bug")
- git_push(repo="D:\\A-EYE")
- git_prepare_pr(repo="D:\\A-EYE", base="main")
- automation_type(text="hello world")
- automation_hotkey(keys="ctrl+s")
- automation_click(x="100", y="220")
- dev_mode(project="default")          — open VS Code + dev server for a project
- launch_environment(apps="code,chrome", cwd="D:/project")

Only use tools when asked. For conversation, just talk normally."""
