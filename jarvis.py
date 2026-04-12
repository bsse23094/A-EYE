"""
A-EYE Jarvis — Local AI Assistant
===================================
Single command launcher. Run: python jarvis.py

Features:
  • Double-clap wake activation (like Tony Stark)
  • Continuous voice listening with VAD
  • LLM brain via Ollama (llama3.1:8b)
  • YOLO object detection + Moondream scene description
  • Web scraping, news, weather
  • System control (files, commands, apps)
  • Neural TTS via edge-tts
  • Sleek PyQt6 dark-theme GUI
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    print("=" * 55)
    print("  A-EYE  ·  Jarvis AI Assistant")
    print("  Local · Private · Powerful")
    print("=" * 55)
    print()

    # Import and create engine
    from core.engine import JarvisEngine
    from core.gui import create_and_run_gui

    engine = JarvisEngine()
    create_and_run_gui(engine)


if __name__ == "__main__":
    main()
