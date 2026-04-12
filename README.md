# ⚙️ A-EYE · J.A.R.V.I.S.

```
    ___       _______  ____  __    ________
   /   |     / ____\ \/ /\ \/ /   / ____/ /
  / /| |    / __/   \  /  \  /   / __/ / /
 / ___ |   / /___   / /   / /   / /___/_/
/_/  |_|  /_____/  /_/   /_/   /_____(_)

J.A.R.V.I.S. — Just A Rather Very Intelligent System
Fully local. Entirely private. Unnervingly capable.
```

> *"At some point I'll stop being surprised. That point hasn't arrived yet."*

---

## What Is This?

**A-EYE** is a fully local, Iron Man-style AI assistant that runs entirely on your machine. No cloud. No subscriptions. No one watching.

Built with Python, PyQt6, and a carefully curated stack of on-device AI models, it delivers a holographic HUD experience with always-on voice listening, computer vision, gesture control, and a personality that would make Tony Stark nod approvingly.

---

## ✦ Feature Manifest

| System | Status | Notes |
|---|---|---|
| 🎙️ Always-on Voice (VAD) | **ONLINE** | Silero VAD — activates on speech, not keywords |
| 🧠 LLM Brain | **ONLINE** | Ollama · llama3.2:3b local inference |
| 🔊 Neural TTS | **ONLINE** | edge-tts · en-GB-RyanNeural (British JARVIS) |
| 👁️ Object Detection | **ONLINE** | YOLOv8n real-time scene awareness |
| 🌐 Scene Description | **ONLINE** | Moondream vision model via Ollama |
| 🖐️ Hand Tracking | **ONLINE** | MediaPipe Tasks + Solutions fallback |
| ✏️ Air Draw | **ONLINE** | Draw in mid-air with one finger |
| 🎯 Identify Gesture | **ONLINE** | Hold pointing pose 2s → object identified |
| 🦴 Posture Correction | **ONLINE** | MediaPipe Pose Lite — alerts when hunching |
| 👀 Gaze Tracking | **ONLINE** | Face Mesh iris → screen region detection |
| 📊 System Vitals | **ONLINE** | Live CPU / RAM / Disk I/O HUD panel |
| 🌆 Time-Reactive UI | **ONLINE** | Palette shifts with Lahore (PKT) time of day |
| 🖥️ Glass Overlay | **ONLINE** | Transparent always-on-top holographic annotations |
| 🎨 Holo Sketch Pad | **ONLINE** | Draw and ask JARVIS to interpret |
| 🔔 Proactive Monitor | **ONLINE** | Watches for system events, speaks unprompted |
| 🛠️ Dev Mode | **ONLINE** | Voice-launch VS Code + dev servers |

---

## 🏗️ Architecture

```
jarvis.py  (launcher)
│
├── core/engine.py       ← Central orchestrator — always-on conversation brain
├── core/gui.py          ← PyQt6 HUD (Arc Reactor, Waveform, Camera, Chat, Vitals)
├── core/vision.py       ← VisionEngine: YOLO · MediaPipe · Moondream · Air-Draw
├── core/stt.py          ← faster-whisper STT (beam=5, Silero VAD)
├── core/tts.py          ← edge-tts neural voice (en-GB-RyanNeural)
├── core/llm.py          ← Ollama streaming LLM (llama3.2:3b)
├── core/audio.py        ← AudioEngine: sounddevice capture + VAD
├── core/memory.py       ← Persistent preference memory (JSON)
├── core/tools.py        ← ToolExecutor: file ops, web, system, git, dev
├── core/monitor.py      ← ProactiveMonitor: watches files/system/time
├── core/vitals.py       ← SystemVitals: psutil CPU/RAM/Disk poller
└── core/config.py       ← All tunable parameters in one place
```

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| **UI** | PyQt6 — custom-painted Arc Reactor, waveforms, HUD panels |
| **LLM** | Ollama (llama3.2:3b) — fully local, streaming |
| **Vision LLM** | Moondream via Ollama — scene description + object ID |
| **Object Detection** | YOLOv8n (Ultralytics) |
| **Hand Tracking** | MediaPipe Tasks API (+ solutions fallback) |
| **Pose / Gaze** | MediaPipe Pose Lite + Face Landmarker |
| **Speech-to-Text** | faster-whisper (base model, int8, CPU) |
| **Text-to-Speech** | edge-tts (en-GB-RyanNeural, no network during playback) |
| **Audio** | sounddevice + Silero VAD |
| **System** | psutil — CPU, RAM, Disk I/O |

---

## ⚡ Installation

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- A webcam

### 1 · Clone

```bash
git clone https://github.com/BKPatt/A-EYE.git
cd A-EYE
```

### 2 · Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3 · Pull AI models

```bash
# LLM brain
ollama pull llama3.2:3b

# Vision model (for scene description + object identification)
ollama pull moondream
```

### 4 · Launch

```bash
python jarvis.py
```

> JARVIS will download MediaPipe model assets on first launch (~few MB). Subsequent launches are instant.

---

## 🎤 Voice Commands

JARVIS listens continuously — no wake word required. Just speak naturally.

| Say... | JARVIS does... |
|---|---|
| *"What do you see?"* | Describes the scene via Moondream |
| *"Identify this"* / *"What is this?"* | Crops and identifies the object in frame |
| *"How many fingers am I holding up?"* | Counts your fingers via hand tracking |
| *"Enable air draw"* | Start drawing in mid-air with index finger |
| *"Clear drawing"* | Wipes the air canvas |
| *"What did I draw?"* | JARVIS analyses your air drawing |
| *"Open notepad"* | Launches application |
| *"Take a screenshot"* | Captures screen |
| *"Run command dir C:\\"* | Executes terminal command |
| *"Search for X"* | Web search |
| *"What's the weather in Lahore?"* | Gets live weather |
| *"Git status"* | Checks the current repo |
| *"Dev mode"* | Opens VS Code + dev server |
| *Anything else* | Routed to the LLM — conversational response |

---

## 🖐️ Gesture Controls

| Gesture | Action |
|---|---|
| ☝️ 1 finger (hold 2s) | **Identify object** — JARVIS crops what you're pointing at and names it |
| ☝️ 1 finger (move) | **Air draw** — traces a line on the invisible canvas |
| ✌️ 2+ fingers | Pauses air drawing |
| Any non-gesture | Resets the air draw cursor |

> Tip: Use good lighting for best hand tracking accuracy.

---

## 🎨 HUD Layout

```
┌──────────────────────────────────────────────────────────────────────┐
│  A - E Y E  ·  J.A.R.V.I.S.    MARK IV · LOCAL · PRIVATE     ● ONLINE │
├─────────────────────────┬─────────────────────┬──────────────────────┤
│  ◈ VISUAL FEED          │                     │  ◈ COMMUNICATIONS    │
│  ┌───────────────────┐  │   ┌──────────────┐  │  ┌────────────────┐  │
│  │                   │  │   │  Arc Reactor │  │  │  Chat log      │  │
│  │  640 × 400 camera │  │   │  (state orb) │  │  │  SIR › ...     │  │
│  │  YOLO annotations │  │   └──────────────┘  │  │  JARVIS › ...  │  │
│  │  hand skeleton    │  │                     │  │                │  │
│  └───────────────────┘  │  ════ WAVEFORM ════  │  │                │  │
│  ◈ GESTURE CONTROL      │                     │  └────────────────┘  │
│  ◈ HOLO SKETCH          │  ◈ SYSTEM NERVES    │                      │
│                         │  CPU / RAM / Disk   │                      │
└─────────────────────────┴─────────────────────┴──────────────────────┘
│  Type command… or just speak                                    ▶    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Configuration

All settings live in [core/config.py](core/config.py). Notable knobs:

```python
# LLM
LLM_MODEL = "llama3.2:3b"        # swap to any Ollama model
VISION_MODEL = "moondream:latest"

# Whisper
WHISPER_MODEL = "base"            # base / small / medium
WHISPER_DEVICE = "cpu"

# Voice Activity Detection
VAD_ENERGY_THRESHOLD = 0.015
VAD_SILENCE_TIMEOUT = 1.8

# Gesture
IDENTIFY_HOLD_SECONDS = 2.0       # how long to point before triggering ID
HAND_SMOOTHING_WINDOW = 5

# Posture
POSTURE_ENABLED = True
POSTURE_ALERT_COOLDOWN = 90.0     # seconds between spoken posture reminders

# Dev projects (add your own)
DEV_PROJECTS = {
    "myapp": {"path": "D:/projects/myapp", "server_cmd": "npm run dev"}
}
```

---

## 🖥️ System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| **CPU** | Intel i3 / equivalent | i5+ for smoother vision |
| **RAM** | 6 GB | 8 GB+ |
| **GPU** | Not required | CUDA GPU for faster Whisper |
| **Webcam** | Any USB/built-in | 720p+ for better tracking |
| **OS** | Windows 10/11 | Windows 11 (tested) |
| **Python** | 3.10 | 3.11+ |

> Designed and tested on an **Intel i3** in Lahore, Pakistan. Performance optimised accordingly — YOLO runs async, vision models are frame-skipped, Whisper uses int8 quantisation.

---

## 📦 Requirements

```
PyQt6
opencv-python
numpy
faster-whisper
sounddevice
edge-tts
ultralytics
mediapipe
httpx
psutil
pyautogui
pygetwindow
requests
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 🗂️ Project Structure

```
A-EYE/
├── jarvis.py              ← Entry point
├── core/
│   ├── __init__.py
│   ├── config.py          ← All configuration
│   ├── engine.py          ← JarvisEngine orchestrator
│   ├── gui.py             ← Full PyQt6 HUD
│   ├── vision.py          ← VisionEngine (camera, YOLO, gestures)
│   ├── stt.py             ← Speech-to-text (faster-whisper)
│   ├── tts.py             ← Text-to-speech (edge-tts)
│   ├── llm.py             ← LLM interface (Ollama)
│   ├── audio.py           ← Audio capture + VAD
│   ├── memory.py          ← Persistent preferences
│   ├── monitor.py         ← Proactive background monitor
│   ├── tools.py           ← Tool executor (OS, web, git)
│   └── vitals.py          ← System vitals poller
├── .models/               ← Downloaded MediaPipe task models (auto)
├── .tts_cache/            ← Cached TTS audio (auto)
├── yolov8n.pt             ← YOLO weights (auto-downloaded)
└── requirements.txt
```

---

## 🔐 Privacy

Everything runs **100% locally**. No data leaves your machine.

- Speech transcription: faster-whisper (local)
- LLM inference: Ollama (local)
- Vision: YOLOv8 + MediaPipe (local)
- TTS synthesis: edge-tts (requires internet only for first synthesis per text; audio cached locally after)

> The only optional network calls are: weather, web search, and news — which you control by voice command.

---

## 🧠 Personality

JARVIS has opinions.

```
"I've taken the liberty of looking that up. You're welcome."
"Your room appears to be in its… natural state, sir."
"I could do that, though I question the wisdom of it."
"That went spectacularly sideways, sir. Shall I log it?"
"Certainly. Though I should note, the last time we tried this, it didn't end well."
```

He calls you **sir**. He understands Urdu. He responds in British English. He is, as Tony Stark intended, *effortlessly composed*.

---

## 🚀 Roadmap

- [ ] Wake-word activation option (Porcupine / Whisper keyword)
- [ ] Multi-screen Glass Overlay support
- [ ] Local image generation (Stable Diffusion integration)
- [ ] Emotion / facial expression recognition
- [ ] Voice cloning for personalised TTS

---

## 👨‍💻 Author

Built by **Ahmed Ayyan** — embedded electronics, AI/ML enthusiast, Laravel developer, and aspiring Tony Stark.  
Location: Lahore, Pakistan 🇵🇰  

---

<div align="center">

```
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
▓                                                  ▓
▓      J.A.R.V.I.S. SYSTEMS — ALL ONLINE           ▓
▓      READY WHEN YOU ARE, SIR.                    ▓
▓                                                  ▓
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
```

</div>
