"""JARVIS HUD — Iron Man style holographic interface."""

from __future__ import annotations

import ctypes
import math
import sys
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import (
    QTimer, Qt, pyqtSignal, QObject, QRectF, QPointF,
)
from PyQt6.QtGui import (
    QImage, QPixmap, QFont, QColor, QPainter, QPen,
    QBrush, QRadialGradient, QLinearGradient, QConicalGradient,
    QPainterPath, QFontDatabase, QPalette, QAction, QIcon,
    QScreen,
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTextEdit, QLineEdit, QPushButton, QFrame,
    QSystemTrayIcon, QMenu, QGraphicsDropShadowEffect,
)

from .engine import JarvisEngine
from .vitals import SystemVitals, build_bg_stylesheet, get_time_period

_TZ_PKT = timezone(timedelta(hours=5))


# ── Signal bridge (thread-safe GUI updates) ──────────────────────

class SignalBridge(QObject):
    status_signal = pyqtSignal(str)
    user_text_signal = pyqtSignal(str)
    assistant_token_signal = pyqtSignal(str)
    assistant_done_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    listening_state_signal = pyqtSignal(str)
    audio_level_signal = pyqtSignal(float)
    speaking_signal = pyqtSignal(bool)
    posture_signal = pyqtSignal(str)      # "good" | "hunched" | "tilted"
    gaze_signal = pyqtSignal(str)         # "center" | "bottom_right" etc.
    overlay_signal = pyqtSignal(dict)     # annotation command


# ── System Vitals Widget ─────────────────────────────────────────────

class SystemVitalsWidget(QWidget):
    """Live monospace HUD panel for CPU, RAM, Disk I/O, and Lahore time."""

    def __init__(self, vitals: SystemVitals, parent=None):
        super().__init__(parent)
        self._vitals = vitals
        self._gaze_active = False
        self._phase = 0.0
        self._snap: dict = {}

        self.setFixedHeight(200)
        self.setMinimumWidth(280)

        # Refresh timer — 1 s
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(1000)

        # Subtle pulse timer
        self._pulse_timer = QTimer(self)
        self._pulse_timer.timeout.connect(self._pulse)
        self._pulse_timer.start(50)

    def set_gaze_active(self, active: bool) -> None:
        self._gaze_active = active
        self.update()

    def _refresh(self) -> None:
        self._snap = self._vitals.get()
        self.update()

    def _pulse(self) -> None:
        self._phase += 0.06
        if self._phase > 2 * math.pi:
            self._phase -= 2 * math.pi
        self.update()

    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # Background
        bg_alpha = 210 if self._gaze_active else 160
        painter.fillRect(self.rect(), QColor(4, 10, 22, bg_alpha))

        glow_strength = 0.5 + 0.5 * math.sin(self._phase)
        border_alpha = 200 if self._gaze_active else int(60 + 40 * glow_strength)
        painter.setPen(QPen(QColor(30, 140, 255, border_alpha), 1))
        painter.drawRect(0, 0, w - 1, h - 1)

        s = self._snap
        if not s:
            painter.setPen(QColor(60, 100, 160, 140))
            painter.setFont(QFont("Consolas", 8))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "VITALS LOADING…")
            painter.end()
            return

        font = QFont("Consolas", 8)
        painter.setFont(font)

        # Lahore time — prominent
        time_alpha = 220 if self._gaze_active else 180
        painter.setPen(QColor(80, 200, 255, time_alpha))
        painter.setFont(QFont("Consolas", 13, QFont.Weight.Bold))
        painter.drawText(8, 22, s.get("lahore_time", "--:--:--"))

        period_map = {
            "dawn": "PKT · DAWN", "morning": "PKT · MORNING",
            "afternoon": "PKT · AFTERNOON", "evening": "PKT · EVENING",
            "dusk": "PKT · DUSK", "night": "PKT · NIGHT",
        }
        period = get_time_period(s.get("lahore_hour", 12))
        painter.setFont(QFont("Consolas", 7))
        painter.setPen(QColor(40, 120, 200, 130))
        painter.drawText(8, 34, period_map.get(period, "PKT"))

        painter.setFont(font)
        y = 52

        def draw_bar(label: str, value: float, max_val: float,
                     color: QColor, suffix: str = "%") -> None:
            nonlocal y
            pct = min(1.0, value / max(0.001, max_val))
            bar_w = w - 90
            bar_h = 6

            # Label + value
            painter.setPen(QColor(100, 160, 220, 180))
            painter.drawText(8, y, label)
            val_text = f"{value:.1f}{suffix}"
            painter.setPen(QColor(180, 220, 255, 200))
            painter.drawText(w - 70, y, val_text)

            # Bar background
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(20, 40, 70, 120))
            painter.drawRoundedRect(8, y + 3, bar_w, bar_h, 3, 3)

            # Bar fill
            fill_alpha = 220 if self._gaze_active else 180
            painter.setBrush(QColor(color.red(), color.green(), color.blue(), fill_alpha))
            fill_w = max(2, int(bar_w * pct))
            painter.drawRoundedRect(8, y + 3, fill_w, bar_h, 3, 3)

            y += 20

        draw_bar("CPU ", s.get("cpu_pct", 0), 100, QColor(30, 200, 255))

        temp = s.get("cpu_temp")
        temp_str = f"{temp:.0f}°C" if temp is not None else "N/A"
        painter.setFont(QFont("Consolas", 7))
        painter.setPen(QColor(60, 140, 200, 150))
        painter.drawText(w - 70, y - 15, temp_str)
        painter.setFont(font)

        draw_bar("RAM ", s.get("ram_pct", 0), 100, QColor(80, 140, 255))
        draw_bar("R/W↑", s.get("disk_r_mbs", 0), 200, QColor(60, 220, 120), " MB/s")
        draw_bar("R/W↓", s.get("disk_w_mbs", 0), 200, QColor(255, 160, 40), " MB/s")

        painter.end()


# ── Glass Overlay Window ─────────────────────────────────────────────

_ANNOTATION_TYPE = dict  # {"type": str, "text": str, "x": float, "y": float, "born": float}

# Windows API constants for click-through
_GWL_EXSTYLE = -20
_WS_EX_LAYERED = 0x80000
_WS_EX_TRANSPARENT = 0x20


class GlassOverlay(QWidget):
    """Transparent always-on-top overlay for holographic desktop annotations.

    Jarvis can draw text labels, highlight boxes, and arrows on this window
    while it remains invisible to mouse clicks (pass-through mode by default).
    Toggle with show()/hide() or the 'show overlay'/'hide overlay' voice command.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        # Cover primary screen
        screen: Optional[QScreen] = QApplication.primaryScreen()
        if screen is not None:
            self.setGeometry(screen.geometry())
        else:
            self.setGeometry(0, 0, 1920, 1080)

        self._annotations: List[_ANNOTATION_TYPE] = []
        self._lock = threading.Lock()
        self._phase = 0.0

        # Make window click-through on Windows
        self._set_clickthrough(True)

        # Animation + cleanup timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(50)

    def _set_clickthrough(self, enabled: bool) -> None:
        """Set or clear Windows WS_EX_TRANSPARENT so clicks pass through."""
        try:
            hwnd = int(self.winId())
            style = ctypes.windll.user32.GetWindowLongW(hwnd, _GWL_EXSTYLE)
            if enabled:
                style |= (_WS_EX_TRANSPARENT | _WS_EX_LAYERED)
            else:
                style &= ~_WS_EX_TRANSPARENT
            ctypes.windll.user32.SetWindowLongW(hwnd, _GWL_EXSTYLE, style)
        except Exception:
            pass  # Non-Windows or no window handle yet

    def add_text(self, text: str, x: float = 0.5, y: float = 0.3,
                 color: Tuple[int, int, int] = (80, 230, 255), ttl: float = 0) -> None:
        """Add a text annotation at (x, y) in screen-fraction coordinates."""
        from . import config as _cfg
        with self._lock:
            self._annotations.append({
                "type": "text",
                "text": text,
                "x": x,
                "y": y,
                "color": color,
                "born": time.time(),
                "ttl": ttl or _cfg.OVERLAY_ANNOTATION_TTL,
            })
        self.update()

    def add_box(self, x1: float, y1: float, x2: float, y2: float,
                color: Tuple[int, int, int] = (30, 200, 255), ttl: float = 0) -> None:
        from . import config as _cfg
        with self._lock:
            self._annotations.append({
                "type": "box",
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "color": color,
                "born": time.time(),
                "ttl": ttl or _cfg.OVERLAY_ANNOTATION_TTL,
            })
        self.update()

    def add_posture_alert(self, state: str) -> None:
        """Show a subtle posture reminder at bottom-centre."""
        msgs = {
            "hunched": "SIT UP",
            "tilted": "LEVEL YOUR SHOULDERS",
        }
        msg = msgs.get(state, "CHECK POSTURE")
        self.add_text(msg, x=0.5, y=0.92, color=(255, 120, 60), ttl=8.0)

    def clear(self) -> None:
        with self._lock:
            self._annotations.clear()
        self.update()

    def _tick(self) -> None:
        now = time.time()
        self._phase += 0.07
        with self._lock:
            self._annotations = [
                a for a in self._annotations
                if now - a["born"] < a["ttl"]
            ]
        self.update()

    def paintEvent(self, event):  # noqa: N802
        now = time.time()
        with self._lock:
            annotations = list(self._annotations)

        if not annotations:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        for ann in annotations:
            age = now - ann["born"]
            ttl = ann["ttl"]
            fade = max(0.0, min(1.0, 1.0 - (age / ttl) ** 3))  # cubic fade
            pulse = 0.75 + 0.25 * math.sin(self._phase + age)
            alpha = int(255 * fade * pulse)

            cr, cg, cb = ann.get("color", (80, 230, 255))

            if ann["type"] == "text":
                x_px = int(ann["x"] * w)
                y_px = int(ann["y"] * h)
                text = ann.get("text", "")

                # Glow shadow
                for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
                    painter.setPen(QColor(cr // 2, cg // 2, cb // 2, alpha // 4))
                    painter.setFont(QFont("Consolas", 18, QFont.Weight.Bold))
                    painter.drawText(x_px + dx * 2, y_px + dy * 2, text)

                painter.setPen(QColor(cr, cg, cb, alpha))
                painter.setFont(QFont("Consolas", 18, QFont.Weight.Bold))
                painter.drawText(x_px, y_px, text)

                # Underline accent
                fm = painter.fontMetrics()
                tw = fm.horizontalAdvance(text)
                painter.setPen(QPen(QColor(cr, cg, cb, alpha // 2), 1))
                painter.drawLine(x_px, y_px + 3, x_px + tw, y_px + 3)

            elif ann["type"] == "box":
                x1 = int(ann["x1"] * w)
                y1 = int(ann["y1"] * h)
                x2 = int(ann["x2"] * w)
                y2 = int(ann["y2"] * h)
                painter.setPen(QPen(QColor(cr, cg, cb, alpha), 2))
                painter.setBrush(QBrush(QColor(cr, cg, cb, int(alpha * 0.08))))
                painter.drawRect(x1, y1, x2 - x1, y2 - y1)

                # Corner accents
                L = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
                painter.setPen(QPen(QColor(cr, cg, cb, alpha), 3))
                for (px, py, dx, dy) in [
                    (x1, y1, 1, 1), (x2, y1, -1, 1),
                    (x1, y2, 1, -1), (x2, y2, -1, -1),
                ]:
                    painter.drawLine(px, py, px + dx * L, py)
                    painter.drawLine(px, py, px, py + dy * L)

        painter.end()


# ── Arc Reactor / Core Orb Widget ────────────────────────────────

class ArcReactorWidget(QWidget):
    """Central JARVIS arc reactor orb — glows, pulses, and shows state."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(220, 220)
        self._state = "idle"  # idle | listening | thinking | speaking
        self._audio_level = 0.0
        self._tick = 0.0
        self._ring_angle = 0.0

        # Animation timer
        self._timer = QTimer()
        self._timer.timeout.connect(self._animate)
        self._timer.start(30)  # ~33 fps

    def set_state(self, state: str):
        self._state = state
        self.update()

    def set_audio_level(self, level: float):
        self._audio_level = min(level * 6, 1.0)
        self.update()

    def _animate(self):
        self._tick += 0.04
        self._ring_angle += 1.2
        if self._ring_angle >= 360:
            self._ring_angle -= 360
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        cx, cy = self.width() / 2, self.height() / 2
        max_r = min(cx, cy) - 4

        # ── State-based colors ────────────────────────────────
        if self._state == "listening":
            core_color = QColor(255, 80, 80)       # Red when listening
            glow_color = QColor(255, 60, 60, 60)
            ring_color = QColor(255, 100, 100, 180)
        elif self._state == "thinking":
            core_color = QColor(255, 180, 50)       # Amber when thinking
            glow_color = QColor(255, 160, 30, 50)
            ring_color = QColor(255, 200, 80, 180)
        elif self._state == "speaking":
            core_color = QColor(55, 211, 178)       # Teal when speaking
            glow_color = QColor(55, 211, 178, 60)
            ring_color = QColor(80, 230, 200, 180)
        else:
            core_color = QColor(30, 140, 255)       # Blue when idle
            glow_color = QColor(30, 130, 255, 40)
            ring_color = QColor(60, 160, 255, 150)

        # ── Outer glow ────────────────────────────────────────
        pulse = 0.5 + 0.5 * math.sin(self._tick * 2)
        glow_r = max_r * (0.85 + 0.15 * pulse)
        grad = QRadialGradient(cx, cy, glow_r)
        grad.setColorAt(0, QColor(glow_color.red(), glow_color.green(), glow_color.blue(), int(80 * pulse)))
        grad.setColorAt(0.6, QColor(glow_color.red(), glow_color.green(), glow_color.blue(), 20))
        grad.setColorAt(1, QColor(0, 0, 0, 0))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(grad))
        painter.drawEllipse(QPointF(cx, cy), glow_r, glow_r)

        # ── Outer ring (rotating) ─────────────────────────────
        pen = QPen(ring_color, 1.5)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(cx, cy), max_r * 0.88, max_r * 0.88)

        # ── Rotating arc segments ─────────────────────────────
        pen = QPen(ring_color, 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        rect = QRectF(cx - max_r * 0.82, cy - max_r * 0.82, max_r * 1.64, max_r * 1.64)
        for i in range(3):
            start_angle = int((self._ring_angle + i * 120) * 16)
            span = int(50 * 16)
            painter.drawArc(rect, start_angle, span)

        # ── Counter-rotating inner arcs ───────────────────────
        pen = QPen(QColor(ring_color.red(), ring_color.green(), ring_color.blue(), 100), 1.5,
                   Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        rect2 = QRectF(cx - max_r * 0.68, cy - max_r * 0.68, max_r * 1.36, max_r * 1.36)
        for i in range(4):
            start_angle = int((-self._ring_angle * 1.5 + i * 90) * 16)
            span = int(35 * 16)
            painter.drawArc(rect2, start_angle, span)

        # ── Inner circle ──────────────────────────────────────
        inner_r = max_r * 0.45
        # React to audio level
        if self._state in ("listening", "speaking"):
            inner_r += self._audio_level * max_r * 0.12

        inner_grad = QRadialGradient(cx, cy - inner_r * 0.2, inner_r * 1.2)
        inner_grad.setColorAt(0, QColor(255, 255, 255, 90))
        inner_grad.setColorAt(0.3, core_color)
        inner_grad.setColorAt(0.8, QColor(core_color.red() // 2, core_color.green() // 2, core_color.blue() // 2))
        inner_grad.setColorAt(1, QColor(0, 0, 0, 200))

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(inner_grad))
        painter.drawEllipse(QPointF(cx, cy), inner_r, inner_r)

        # ── Center dot ────────────────────────────────────────
        dot_r = 6 + 3 * pulse
        painter.setBrush(QColor(255, 255, 255, 200))
        painter.drawEllipse(QPointF(cx, cy), dot_r, dot_r)

        # ── Tick marks ────────────────────────────────────────
        pen = QPen(QColor(ring_color.red(), ring_color.green(), ring_color.blue(), 80), 1)
        painter.setPen(pen)
        for i in range(12):
            angle = math.radians(i * 30 + self._ring_angle * 0.3)
            r1 = max_r * 0.92
            r2 = max_r * 0.96
            x1 = cx + r1 * math.cos(angle)
            y1 = cy + r1 * math.sin(angle)
            x2 = cx + r2 * math.cos(angle)
            y2 = cy + r2 * math.sin(angle)
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        painter.end()


# ── HUD Chat Display ─────────────────────────────────────────────

class HUDChatDisplay(QTextEdit):
    """Holographic-style minimal chat display."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(QFont("Consolas", 11))
        self.setStyleSheet("""
            QTextEdit {
                background: transparent;
                color: #b0d0f0;
                border: none;
                padding: 8px;
                selection-background-color: rgba(30, 140, 255, 0.3);
            }
            QScrollBar:vertical {
                background: transparent;
                width: 4px;
            }
            QScrollBar::handle:vertical {
                background: rgba(30, 140, 255, 0.3);
                border-radius: 2px;
                min-height: 20px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        self._current_role = None

    def add_user_message(self, text: str) -> None:
        self._current_role = "user"
        self.append(f'<div style="margin: 6px 0;">'
                    f'<span style="color: #ffffff; font-weight: bold; font-size: 12px;">SIR ›</span> '
                    f'<span style="color: #d0e0f5;">{text}</span></div>')
        self._scroll()

    def add_assistant_token(self, token: str) -> None:
        if self._current_role != "assistant":
            self._current_role = "assistant"
            self.append(f'<span style="color: #30a0ff; font-weight: bold; font-size: 12px;">JARVIS ›</span> ')
        cursor = self.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(token)
        self.setTextCursor(cursor)
        self._scroll()

    def add_assistant_done(self) -> None:
        self._current_role = None
        self.append("")
        self._scroll()

    def add_system_message(self, text: str) -> None:
        self._current_role = None
        self.append(f'<div style="text-align: center; margin: 3px 0;">'
                    f'<span style="color: #304060; font-size: 9px;">◆ {text}</span></div>')
        self._scroll()

    def add_error_message(self, text: str) -> None:
        self._current_role = None
        self.append(f'<span style="color: #ff5555; font-size: 11px;">⚠ {text}</span>')
        self._scroll()

    def _scroll(self):
        sb = self.verticalScrollBar()
        sb.setValue(sb.maximum())


# ── Audio Wave Bar ────────────────────────────────────────────────

class AudioWaveBar(QWidget):
    """JARVIS-style full audio visualiser — reacts to real mic/TTS levels."""

    _NUM_BARS = 120

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(80)
        self.setMinimumWidth(300)

        self._levels: list[float] = [0.0] * self._NUM_BARS
        self._peaks: list[float] = [0.0] * self._NUM_BARS   # falling peak dots
        self._state = "idle"
        self._phase = 0.0
        self._raw_level = 0.0   # latest audio level from mic
        self._idle_breath = 0.0

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(30)   # ~33 fps

    def set_level(self, level: float) -> None:
        """Called every audio frame with the current mic level (0–1)."""
        self._raw_level = min(level * 12.0, 1.0)
        # Shift history left and push new level
        self._levels.pop(0)
        self._levels.append(self._raw_level)
        self.update()

    def set_state(self, state: str) -> None:
        self._state = state
        self.update()

    def _tick(self) -> None:
        self._phase += 0.18
        self._idle_breath += 0.04

        if self._state == "speaking":
            # Multi-frequency organic voice waveform — never flat
            new_levels = []
            for i in range(self._NUM_BARS):
                t = self._phase
                fi = i / self._NUM_BARS
                v = (
                    0.18 * abs(math.sin(t * 2.1 + fi * math.pi * 3.2)) +
                    0.22 * abs(math.sin(t * 3.7 + fi * math.pi * 5.1)) +
                    0.18 * abs(math.sin(t * 6.3 + fi * math.pi * 7.4)) +
                    0.12 * abs(math.sin(t * 11.1 + fi * math.pi * 12.6)) +
                    0.10 * abs(math.sin(t * 17.9 + fi * math.pi * 2.3))
                )
                # Taper the edges slightly
                edge = math.sin(fi * math.pi)
                new_levels.append(min(1.0, v * (0.55 + 0.45 * edge)))
            self._levels = new_levels

        elif self._state == "idle":
            # Gentle idle breath — centre beat, quiet edges
            new_levels = []
            for i in range(self._NUM_BARS):
                fi = i / self._NUM_BARS
                edge = math.sin(fi * math.pi)
                v = 0.03 + 0.07 * abs(math.sin(self._idle_breath * 0.8 + fi * 2.5)) * edge
                new_levels.append(v)
            self._levels = new_levels

        # Update falling peaks
        for i, lv in enumerate(self._levels):
            if lv > self._peaks[i]:
                self._peaks[i] = lv
            else:
                self._peaks[i] = max(0.0, self._peaks[i] - 0.025)

        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        center_y = h // 2
        count = len(self._levels)

        # Dark translucent background
        painter.fillRect(self.rect(), QColor(3, 6, 14, 200))

        bar_slot = w / count
        bar_w = max(1.5, bar_slot * 0.62)

        for i, level in enumerate(self._levels):
            x = bar_slot * i + bar_slot * 0.5
            bar_h = max(1.5, level * center_y * 0.92)

            # ── Colour based on state ─────────────────────────────
            if self._state == "listening":
                # Red spectrum, brighter at peaks
                brightness = 0.55 + 0.45 * level
                r = int(255 * brightness)
                g = int(60 * brightness * (1 - level * 0.5))
                b = int(50 * brightness)
            elif self._state == "speaking":
                # Teal/cyan spectrum
                brightness = 0.5 + 0.5 * level
                r = int(40 * brightness)
                g = int(220 * brightness)
                b = int(185 * brightness)
            else:
                # Calm blue
                brightness = 0.4 + 0.6 * level
                r = int(20 * brightness)
                g = int(120 * brightness)
                b = int(255 * brightness)

            # Centre bars brighter
            center_factor = 1.0 - 0.3 * abs(i - count / 2) / (count / 2)
            alpha = int(min(255, (140 + 115 * level) * center_factor))

            # Draw bar (vertical line with round cap)
            pen = QPen(
                QColor(r, g, b, alpha),
                bar_w,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
            )
            painter.setPen(pen)
            painter.drawLine(QPointF(x, center_y - bar_h), QPointF(x, center_y + bar_h))

            # ── Falling peak dot ──────────────────────────────────
            pk = self._peaks[i]
            if pk > 0.05:
                pk_h = pk * center_y * 0.92
                dot_alpha = int(min(255, alpha * 1.4))
                painter.setPen(QPen(QColor(r, g, b, dot_alpha), bar_w * 1.1,
                                    Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                painter.drawPoint(QPointF(x, center_y - pk_h))
                painter.drawPoint(QPointF(x, center_y + pk_h))

        # Centre line glow
        if self._state != "idle":
            pulse = 0.4 + 0.6 * abs(math.sin(self._phase * 1.5))
            if self._state == "listening":
                line_col = QColor(255, 60, 40, int(50 * pulse))
            else:
                line_col = QColor(55, 211, 178, int(40 * pulse))
            painter.setPen(QPen(line_col, 1))
            painter.drawLine(0, center_y, w, center_y)

        painter.end()


class HoloSketchPad(QWidget):
    """Interactive neon sketch pad for visual interaction with Jarvis."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(320, 170)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._strokes: list[list[QPointF]] = []
        self._current: list[QPointF] = []
        self._phase = 0.0

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)

    def clear(self) -> None:
        self._strokes.clear()
        self._current.clear()
        self.update()

    def get_summary(self) -> str:
        points = sum(len(s) for s in self._strokes)
        if not self._strokes or points == 0:
            return "No sketch present."

        xs = [p.x() for s in self._strokes for p in s]
        ys = [p.y() for s in self._strokes for p in s]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        return (
            f"User drew a holographic sketch with {len(self._strokes)} stroke(s), "
            f"{points} points, and approximate bounds {int(width)}x{int(height)} pixels. "
            "Respond briefly and ask if they want edits or interpretation."
        )

    def _tick(self):
        self._phase += 0.09
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._current = [event.position()]
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._current and (event.buttons() & Qt.MouseButton.LeftButton):
            self._current.append(event.position())
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._current:
            self._strokes.append(self._current[:])
            self._current.clear()
            self.update()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        painter.fillRect(self.rect(), QColor(5, 10, 22, 220))

        # Holographic grid.
        grid_pen = QPen(QColor(40, 120, 255, 45), 1)
        painter.setPen(grid_pen)
        step = 20
        for x in range(0, w, step):
            painter.drawLine(x, 0, x, h)
        for y in range(0, h, step):
            painter.drawLine(0, y, w, y)

        # Scan beam.
        beam_y = int((0.5 + 0.5 * math.sin(self._phase)) * (h - 1))
        beam_pen = QPen(QColor(80, 220, 255, 90), 2)
        painter.setPen(beam_pen)
        painter.drawLine(0, beam_y, w, beam_y)

        # Draw persisted strokes.
        for idx, stroke in enumerate(self._strokes):
            if len(stroke) < 2:
                continue
            alpha = 120 + int(80 * (0.5 + 0.5 * math.sin(self._phase + idx * 0.6)))
            pen = QPen(QColor(90, 230, 255, alpha), 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            for i in range(1, len(stroke)):
                painter.drawLine(stroke[i - 1], stroke[i])

        # Draw live stroke.
        if len(self._current) >= 2:
            pen = QPen(QColor(145, 255, 255, 230), 2.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            for i in range(1, len(self._current)):
                painter.drawLine(self._current[i - 1], self._current[i])

        painter.setPen(QPen(QColor(70, 180, 255, 140), 1))
        painter.drawRect(0, 0, w - 1, h - 1)
        painter.end()


# ── Main JARVIS Window ────────────────────────────────────────────

class JarvisWindow(QMainWindow):
    """Iron Man JARVIS HUD interface."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("A-EYE  ·  J.A.R.V.I.S.")
        self.setMinimumSize(1400, 860)
        self.resize(1680, 980)

        self._ui_phase = 0.0          # drives all ambient animations
        self._current_state = "idle"   # tracks state for animation colouring

        # Signal bridge
        self.bridge = SignalBridge()
        self.bridge.status_signal.connect(self._on_status)
        self.bridge.user_text_signal.connect(self._on_user_text)
        self.bridge.assistant_token_signal.connect(self._on_assistant_token)
        self.bridge.assistant_done_signal.connect(self._on_assistant_done)
        self.bridge.error_signal.connect(self._on_error)
        self.bridge.listening_state_signal.connect(self._on_listening_state)
        self.bridge.audio_level_signal.connect(self._on_audio_level)
        self.bridge.speaking_signal.connect(self._on_speaking)
        self.bridge.posture_signal.connect(self._on_posture)
        self.bridge.gaze_signal.connect(self._on_gaze)

        # Vitals
        from . import config as _cfg
        self._vitals = SystemVitals(poll_interval=_cfg.VITALS_POLL_INTERVAL)
        if _cfg.VITALS_ENABLED:
            self._vitals.start()

        self._build_ui()
        self._setup_tray()

        # Glass overlay
        self._overlay: Optional[GlassOverlay] = None
        if _cfg.GLASS_OVERLAY_ENABLED:
            self._overlay = GlassOverlay()

        # Camera timer
        self._cam_timer = QTimer()
        self._cam_timer.timeout.connect(self._update_camera)
        self._cam_timer.start(33)

        # Gaze poll timer
        self._gaze_timer = QTimer()
        self._gaze_timer.timeout.connect(self._poll_gaze)
        self._gaze_timer.start(500)  # 2 Hz is plenty for gaze region

        # Time-reactive background — update every 5 minutes
        self._bg_hour = -1
        self._update_bg_for_time()
        self._bg_timer = QTimer()
        self._bg_timer.timeout.connect(self._update_bg_for_time)
        self._bg_timer.start(300_000)  # 5 min

        # Posture state
        self._last_posture_alert_time = 0.0

        # Ambient UI animation timer — drives title pulse, status dot blink etc.
        self._ui_anim_timer = QTimer(self)
        self._ui_anim_timer.timeout.connect(self._tick_ui_animation)
        self._ui_anim_timer.start(50)   # 20 fps ambient

        self.engine: Optional[JarvisEngine] = None

    def set_engine(self, engine: JarvisEngine):
        self.engine = engine

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        central.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:0.5, y2:1,
                    stop:0 #020508, stop:0.5 #040a14, stop:1 #020406);
                color: #b0d0f0;
                font-family: 'Segoe UI', 'Consolas', monospace;
            }
        """)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(20, 12, 20, 12)
        main_layout.setSpacing(0)

        # ── Top HUD bar ───────────────────────────────────────
        top_bar = QHBoxLayout()
        top_bar.setSpacing(20)

        # Left: animated title
        self.sys_label = QLabel("A - E Y E    ·    J.A.R.V.I.S.")
        self.sys_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Light))
        self.sys_label.setStyleSheet("color: rgba(30, 140, 255, 0.65); letter-spacing: 5px;")
        top_bar.addWidget(self.sys_label)

        # Centre: build info strip
        build_label = QLabel("MARK  IV  ·  LOCAL  ·  PRIVATE  ·  v2.0")
        build_label.setFont(QFont("Consolas", 7))
        build_label.setStyleSheet("color: rgba(30, 140, 255, 0.18); letter-spacing: 4px;")
        build_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top_bar.addWidget(build_label)

        top_bar.addStretch()

        # Right: animated status dot + label
        self.status_dot = QLabel("●")
        self.status_dot.setFont(QFont("Consolas", 10))
        self.status_dot.setStyleSheet("color: #204060;")
        top_bar.addWidget(self.status_dot)

        self.status_label = QLabel("INITIALIZING")
        self.status_label.setFont(QFont("Consolas", 9))
        self.status_label.setStyleSheet("color: #305080; letter-spacing: 2px;")
        top_bar.addWidget(self.status_label)

        main_layout.addLayout(top_bar)

        # ── Thin separator line ───────────────────────────────
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("background: rgba(30, 140, 255, 0.15); max-height: 1px; margin: 6px 0;")
        main_layout.addWidget(line)

        # ── Center content ────────────────────────────────────
        content = QHBoxLayout()
        content.setSpacing(20)

        # Left column: Camera + detections
        left_col = QVBoxLayout()
        left_col.setSpacing(8)

        cam_label_header = QLabel("◈ VISUAL FEED")
        cam_label_header.setFont(QFont("Consolas", 8))
        cam_label_header.setStyleSheet("color: rgba(30, 140, 255, 0.4); letter-spacing: 3px;")
        left_col.addWidget(cam_label_header)

        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setFixedSize(640, 400)
        self.camera_label.setStyleSheet("""
            QLabel {
                background: #030610;
                border: 1px solid rgba(30, 140, 255, 0.2);
                border-radius: 4px;
            }
        """)
        self.camera_label.setText("CAMERA INITIALIZING")
        self.camera_label.setFont(QFont("Consolas", 8))
        left_col.addWidget(self.camera_label)

        self.detection_label = QLabel("")
        self.detection_label.setFont(QFont("Consolas", 8))
        self.detection_label.setStyleSheet("color: #205540;")
        self.detection_label.setWordWrap(True)
        left_col.addWidget(self.detection_label)

        self.describe_btn = QPushButton("◈ ANALYZE SCENE")
        self.describe_btn.setFont(QFont("Consolas", 9))
        self.describe_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.describe_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: rgba(30, 140, 255, 0.5);
                border: 1px solid rgba(30, 140, 255, 0.2);
                border-radius: 3px;
                padding: 6px 12px;
                letter-spacing: 2px;
            }
            QPushButton:hover {
                border-color: rgba(30, 140, 255, 0.6);
                color: rgba(30, 140, 255, 0.9);
                background: rgba(30, 140, 255, 0.05);
            }
        """)
        self.describe_btn.clicked.connect(self._on_describe)
        left_col.addWidget(self.describe_btn)

        gesture_header = QLabel("◈ GESTURE CONTROL")
        gesture_header.setFont(QFont("Consolas", 8))
        gesture_header.setStyleSheet("color: rgba(90, 220, 255, 0.45); letter-spacing: 3px;")
        left_col.addWidget(gesture_header)

        gesture_row = QHBoxLayout()
        self.airdraw_toggle_btn = QPushButton("AIRDRAW ON")
        self.airdraw_toggle_btn.setFont(QFont("Consolas", 8))
        self.airdraw_toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.airdraw_toggle_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: rgba(80, 230, 255, 0.9);
                border: 1px solid rgba(80, 230, 255, 0.35);
                border-radius: 3px;
                padding: 4px 8px;
                letter-spacing: 1px;
            }
            QPushButton:hover {
                border-color: rgba(120, 245, 255, 0.8);
                background: rgba(80, 230, 255, 0.08);
            }
        """)
        self.airdraw_toggle_btn.clicked.connect(self._on_toggle_airdraw)
        gesture_row.addWidget(self.airdraw_toggle_btn)

        self.count_fingers_btn = QPushButton("COUNT FINGERS")
        self.count_fingers_btn.setFont(QFont("Consolas", 8))
        self.count_fingers_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.count_fingers_btn.setStyleSheet(self.airdraw_toggle_btn.styleSheet())
        self.count_fingers_btn.clicked.connect(self._on_count_fingers)
        gesture_row.addWidget(self.count_fingers_btn)
        left_col.addLayout(gesture_row)

        self.analyze_airdraw_btn = QPushButton("WHAT DID I DRAW?")
        self.analyze_airdraw_btn.setFont(QFont("Consolas", 8))
        self.analyze_airdraw_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.analyze_airdraw_btn.setStyleSheet(self.airdraw_toggle_btn.styleSheet())
        self.analyze_airdraw_btn.clicked.connect(self._on_analyze_airdraw)
        left_col.addWidget(self.analyze_airdraw_btn)

        self.gesture_status = QLabel("Gesture mode active. Raise one finger to draw in air.")
        self.gesture_status.setWordWrap(True)
        self.gesture_status.setFont(QFont("Consolas", 8))
        self.gesture_status.setStyleSheet("color: rgba(120, 235, 255, 0.65);")
        left_col.addWidget(self.gesture_status)

        sketch_header = QLabel("◈ HOLO SKETCH")
        sketch_header.setFont(QFont("Consolas", 8))
        sketch_header.setStyleSheet("color: rgba(90, 220, 255, 0.45); letter-spacing: 3px;")
        left_col.addWidget(sketch_header)

        self.sketch_pad = HoloSketchPad()
        self.sketch_pad.setStyleSheet("border: 1px solid rgba(90, 220, 255, 0.25); border-radius: 4px;")
        left_col.addWidget(self.sketch_pad)

        sketch_actions = QHBoxLayout()
        self.clear_sketch_btn = QPushButton("CLEAR")
        self.clear_sketch_btn.setFont(QFont("Consolas", 8))
        self.clear_sketch_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clear_sketch_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: rgba(255, 120, 120, 0.75);
                border: 1px solid rgba(255, 120, 120, 0.35);
                border-radius: 3px;
                padding: 4px 8px;
                letter-spacing: 1px;
            }
            QPushButton:hover {
                color: rgba(255, 150, 150, 1.0);
                border-color: rgba(255, 140, 140, 0.75);
            }
        """)
        self.clear_sketch_btn.clicked.connect(lambda: self.sketch_pad.clear())
        sketch_actions.addWidget(self.clear_sketch_btn)

        self.send_sketch_btn = QPushButton("ASK JARVIS")
        self.send_sketch_btn.setFont(QFont("Consolas", 8))
        self.send_sketch_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.send_sketch_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: rgba(90, 220, 255, 0.85);
                border: 1px solid rgba(90, 220, 255, 0.35);
                border-radius: 3px;
                padding: 4px 8px;
                letter-spacing: 1px;
            }
            QPushButton:hover {
                color: rgba(140, 240, 255, 1.0);
                border-color: rgba(120, 235, 255, 0.8);
                background: rgba(90, 220, 255, 0.08);
            }
        """)
        self.send_sketch_btn.clicked.connect(self._on_send_sketch)
        sketch_actions.addWidget(self.send_sketch_btn)
        left_col.addLayout(sketch_actions)

        left_col.addStretch()
        content.addLayout(left_col)

        # Center column: Arc reactor + waveform
        center_col = QVBoxLayout()
        center_col.setSpacing(10)
        center_col.setAlignment(Qt.AlignmentFlag.AlignCenter)

        center_col.addStretch()

        # State label above orb
        self.state_label = QLabel("S T A N D B Y")
        self.state_label.setFont(QFont("Consolas", 10))
        self.state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.state_label.setStyleSheet("color: rgba(30, 140, 255, 0.4); letter-spacing: 4px;")
        center_col.addWidget(self.state_label)

        # Arc reactor
        self.reactor = ArcReactorWidget()
        center_col.addWidget(self.reactor, alignment=Qt.AlignmentFlag.AlignCenter)

        # Waveform under orb
        self.waveform = AudioWaveBar()
        center_col.addWidget(self.waveform)

        # ── System Vitals ─────────────────────────────────────────
        vitals_header = QLabel("◈ SYSTEM NERVES")
        vitals_header.setFont(QFont("Consolas", 8))
        vitals_header.setStyleSheet("color: rgba(80, 200, 255, 0.45); letter-spacing: 3px;")
        center_col.addWidget(vitals_header)

        self.vitals_widget = SystemVitalsWidget(self._vitals)
        center_col.addWidget(self.vitals_widget)

        center_col.addStretch()
        content.addLayout(center_col)

        # Right column: Chat log
        right_col = QVBoxLayout()
        right_col.setSpacing(6)

        chat_header = QLabel("◈ COMMUNICATIONS")
        chat_header.setFont(QFont("Consolas", 8))
        chat_header.setStyleSheet("color: rgba(30, 140, 255, 0.4); letter-spacing: 3px;")
        right_col.addWidget(chat_header)

        self.chat_display = HUDChatDisplay()
        self.chat_display.setStyleSheet(self.chat_display.styleSheet() + """
            QTextEdit {
                background: rgba(8, 18, 38, 0.35);
                border: 1px solid rgba(70, 170, 255, 0.18);
                border-radius: 6px;
            }
        """)
        right_col.addWidget(self.chat_display, stretch=1)

        content.addLayout(right_col)

        # Set column proportions
        content.setStretch(0, 2)  # camera
        content.setStretch(1, 2)  # reactor
        content.setStretch(2, 3)  # chat

        main_layout.addLayout(content, stretch=1)

        # ── Bottom: separator + input ─────────────────────────
        line2 = QFrame()
        line2.setFrameShape(QFrame.Shape.HLine)
        line2.setStyleSheet("background: rgba(30, 140, 255, 0.15); max-height: 1px; margin: 6px 0;")
        main_layout.addWidget(line2)

        # Input row
        input_layout = QHBoxLayout()
        input_layout.setSpacing(10)

        self.text_input = QLineEdit()
        self.text_input.setFont(QFont("Consolas", 12))
        self.text_input.setPlaceholderText("Type command … or just speak")
        self.text_input.setStyleSheet("""
            QLineEdit {
                background: rgba(10, 20, 40, 0.8);
                color: #c0d8f0;
                border: 1px solid rgba(30, 140, 255, 0.2);
                border-radius: 4px;
                padding: 10px 16px;
                letter-spacing: 1px;
            }
            QLineEdit:focus {
                border-color: rgba(30, 140, 255, 0.5);
            }
        """)
        self.text_input.returnPressed.connect(self._on_text_submit)
        input_layout.addWidget(self.text_input, stretch=1)

        self.send_btn = QPushButton("▶")
        self.send_btn.setFixedSize(44, 44)
        self.send_btn.setFont(QFont("Segoe UI", 14))
        self.send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.send_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: 1px solid rgba(30, 140, 255, 0.3);
                border-radius: 22px;
                color: rgba(30, 140, 255, 0.6);
            }
            QPushButton:hover {
                border-color: rgba(30, 140, 255, 0.7);
                color: rgba(30, 140, 255, 1.0);
                background: rgba(30, 140, 255, 0.08);
            }
        """)
        self.send_btn.clicked.connect(self._on_text_submit)
        input_layout.addWidget(self.send_btn)

        main_layout.addLayout(input_layout)

        # ── Bottom status ─────────────────────────────────────
        bottom = QHBoxLayout()
        self.mode_label = QLabel("◈ ALWAYS-ON VOICE  ·  LOCAL AI  ·  PRIVATE")
        self.mode_label.setFont(QFont("Consolas", 7))
        self.mode_label.setStyleSheet("color: rgba(30, 140, 255, 0.2); letter-spacing: 3px;")
        bottom.addWidget(self.mode_label)
        bottom.addStretch()
        self.engine_label = QLabel("")
        self.engine_label.setFont(QFont("Consolas", 7))
        self.engine_label.setStyleSheet("color: rgba(30, 140, 255, 0.2); letter-spacing: 2px;")
        bottom.addWidget(self.engine_label)
        main_layout.addLayout(bottom)

    def _setup_tray(self):
        try:
            self.tray_icon = QSystemTrayIcon(self)
            px = QPixmap(32, 32)
            px.fill(QColor(30, 140, 255))
            self.tray_icon.setIcon(QIcon(px))
            self.tray_icon.setToolTip("A-EYE JARVIS")
            menu = QMenu()
            show = QAction("Show", self)
            show.triggered.connect(lambda: (self.showNormal(), self.activateWindow()))
            menu.addAction(show)
            quit_a = QAction("Quit", self)
            quit_a.triggered.connect(self._quit)
            menu.addAction(quit_a)
            self.tray_icon.setContextMenu(menu)
            self.tray_icon.activated.connect(
                lambda r: (self.showNormal(), self.activateWindow())
                if r == QSystemTrayIcon.ActivationReason.DoubleClick else None
            )
            self.tray_icon.show()
        except Exception:
            self.tray_icon = None

    def _quit(self):
        if self.engine:
            self.engine.stop()
        QApplication.quit()

    def closeEvent(self, event):
        if self.tray_icon:
            self.hide()
            event.ignore()
        else:
            if self.engine:
                self.engine.stop()
            event.accept()

    # ── Handlers ──────────────────────────────────────────────

    def _on_text_submit(self):
        text = self.text_input.text().strip()
        if not text or self.engine is None:
            return
        self.text_input.clear()
        threading.Thread(target=self.engine.process_text, args=(text,), daemon=True).start()

    def _on_describe(self):
        if self.engine is None:
            return
        def _do():
            desc = self.engine.describe_scene()
            self.bridge.assistant_token_signal.emit(f"\n🔍 {desc}")
            self.bridge.assistant_done_signal.emit(desc)
        threading.Thread(target=_do, daemon=True).start()

    def _on_toggle_airdraw(self):
        if self.engine is None:
            return
        enabled = self.airdraw_toggle_btn.text().strip().endswith("ON")
        new_enabled = not enabled
        self.engine.set_airdraw_enabled(new_enabled)
        self.airdraw_toggle_btn.setText("AIRDRAW ON" if new_enabled else "AIRDRAW OFF")
        self.gesture_status.setText(
            "Gesture mode active. Raise one finger to draw in air." if new_enabled else "Gesture drawing paused."
        )

    def _on_count_fingers(self):
        if self.engine is None:
            return
        count = self.engine.get_finger_count()
        if count is None:
            msg = "I cannot confidently detect your fingers yet."
        else:
            msg = f"I can see approximately {count} finger{'s' if count != 1 else ''}, sir."
        self.chat_display.add_system_message(msg)
        self.gesture_status.setText(msg)

    def _on_analyze_airdraw(self):
        if self.engine is None:
            return
        summary = self.engine.summarize_airdraw()
        self.chat_display.add_system_message(summary)
        self.gesture_status.setText(summary)

    def _on_send_sketch(self):
        if self.engine is None:
            return
        summary = self.sketch_pad.get_summary()
        if summary == "No sketch present.":
            self.chat_display.add_system_message("Draw something on the holo pad first.")
            return

        prompt = (
            "I created a holographic sketch for you. "
            + summary
            + " Infer what it could represent, ask one clarifying question, and suggest one refinement."
        )
        threading.Thread(target=self.engine.process_text, args=(prompt,), daemon=True).start()

    def _on_status(self, text: str):
        self.status_label.setText(text.upper())
        if "online" in text.lower():
            self.status_label.setStyleSheet("color: rgba(30, 180, 100, 0.75); letter-spacing: 2px;")
            self.status_dot.setStyleSheet("color: rgba(30, 200, 100, 0.9);")
        elif "thinking" in text.lower() or "transcrib" in text.lower():
            self.status_label.setStyleSheet("color: rgba(255, 180, 50, 0.8); letter-spacing: 2px;")
            self.status_dot.setStyleSheet("color: rgba(255, 180, 50, 0.9);")
        elif "speaking" in text.lower() or "holographic" in text.lower():
            self.status_label.setStyleSheet("color: rgba(55, 211, 178, 0.8); letter-spacing: 2px;")
            self.status_dot.setStyleSheet("color: rgba(55, 211, 178, 0.9);")
        elif "error" in text.lower() or "alert" in text.lower():
            self.status_label.setStyleSheet("color: rgba(255, 100, 60, 0.9); letter-spacing: 2px;")
            self.status_dot.setStyleSheet("color: rgba(255, 80, 40, 1.0);")
        else:
            self.status_label.setStyleSheet("color: rgba(30, 140, 255, 0.55); letter-spacing: 2px;")
            self.status_dot.setStyleSheet("color: rgba(30, 100, 200, 0.5);")

    def _on_user_text(self, text: str):
        self.chat_display.add_user_message(text)

    def _on_assistant_token(self, token: str):
        self.chat_display.add_assistant_token(token)

    def _on_assistant_done(self, text: str):
        self.chat_display.add_assistant_done()

    def _on_error(self, text: str):
        self.chat_display.add_error_message(text)

    def _on_listening_state(self, state: str):
        self.reactor.set_state(state)
        self.waveform.set_state(state)
        if state == "listening":
            self._current_state = "listening"
            self.state_label.setText("L I S T E N I N G")
            self.state_label.setStyleSheet("color: rgba(255, 80, 80, 0.9); letter-spacing: 4px;")
        elif state == "processing":
            self._current_state = "thinking"
            self.state_label.setText("P R O C E S S I N G")
            self.state_label.setStyleSheet("color: rgba(255, 180, 50, 0.8); letter-spacing: 4px;")
            self.reactor.set_state("thinking")
        else:
            self._current_state = "idle"
            self.state_label.setText("S T A N D B Y")
            self.state_label.setStyleSheet("color: rgba(30, 140, 255, 0.4); letter-spacing: 4px;")

    def _on_audio_level(self, level: float):
        self.reactor.set_audio_level(level)
        self.waveform.set_level(level)

    def _on_speaking(self, active: bool):
        if active:
            self._current_state = "speaking"
            self.reactor.set_state("speaking")
            self.waveform.set_state("speaking")
            self.state_label.setText("S P E A K I N G")
            self.state_label.setStyleSheet("color: rgba(55, 211, 178, 0.9); letter-spacing: 4px;")
            self.mode_label.setText("◈ HOLOGRAPHIC LINK ACTIVE  ·  VOICE SYNTHESIS LIVE")
            self.mode_label.setStyleSheet("color: rgba(80, 230, 200, 0.42); letter-spacing: 3px;")
        else:
            self._current_state = "idle"
            self.reactor.set_state("idle")
            self.waveform.set_state("idle")
            self.state_label.setText("S T A N D B Y")
            self.state_label.setStyleSheet("color: rgba(30, 140, 255, 0.4); letter-spacing: 4px;")
            self.mode_label.setText("◈ ALWAYS-ON VOICE  ·  LOCAL AI  ·  PRIVATE")
            self.mode_label.setStyleSheet("color: rgba(30, 140, 255, 0.2); letter-spacing: 3px;")

    # ── Ambient UI animation ──────────────────────────────────────

    def _tick_ui_animation(self) -> None:
        """Heartbeat timer: keeps the title and status dot feeling alive."""
        self._ui_phase += 0.08
        pulse = 0.5 + 0.5 * math.sin(self._ui_phase)

        # Pulse the title colour subtly
        state = self._current_state
        if state == "listening":
            r, g, b = 255, 80, 80
        elif state == "thinking":
            r, g, b = 255, 180, 50
        elif state == "speaking":
            r, g, b = 55, 211, 178
        else:
            r, g, b = 30, 140, 255

        title_alpha = int(0.45 + 0.25 * pulse * (1.5 if state != "idle" else 1.0))
        title_alpha = min(255, int((0.38 + 0.30 * pulse) * 255))
        self.sys_label.setStyleSheet(
            f"color: rgba({r}, {g}, {b}, {title_alpha}); letter-spacing: 5px;"
        )

        # Blink status dot in active states
        if state in ("listening", "thinking", "speaking"):
            dot_alpha = int((0.5 + 0.5 * pulse) * 255)
            if state == "listening":
                self.status_dot.setStyleSheet(f"color: rgba(255, 80, 80, {dot_alpha});")
            elif state == "thinking":
                self.status_dot.setStyleSheet(f"color: rgba(255, 180, 50, {dot_alpha});")
            else:
                self.status_dot.setStyleSheet(f"color: rgba(55, 211, 178, {dot_alpha});")

    # ── Time-reactive background ──────────────────────────────────

    def _update_bg_for_time(self) -> None:
        hour = datetime.now(_TZ_PKT).hour
        if hour == self._bg_hour:
            return
        self._bg_hour = hour
        from . import config as _cfg
        if not _cfg.TIME_REACTIVE_BG:
            return
        css = build_bg_stylesheet(hour)
        self.centralWidget().setStyleSheet(css)

    # ── Posture handling ──────────────────────────────────────────

    def _on_posture(self, state: str) -> None:
        from . import config as _cfg
        now = time.time()
        if state == "good":
            return
        if now - self._last_posture_alert_time < _cfg.POSTURE_ALERT_COOLDOWN:
            return
        self._last_posture_alert_time = now

        if self._overlay is not None:
            self._overlay.add_posture_alert(state)

        # Brief status flash
        old = self.status_label.text()
        old_style = self.status_label.styleSheet()
        old_dot = self.status_dot.styleSheet()
        if state == "hunched":
            self.status_label.setText("POSTURE ALERT — SIT UP")
            self.status_label.setStyleSheet("color: rgba(255, 140, 40, 0.95); letter-spacing: 2px;")
        else:
            self.status_label.setText("POSTURE ALERT — LEVEL SHOULDERS")
            self.status_label.setStyleSheet("color: rgba(255, 140, 40, 0.95); letter-spacing: 2px;")
        self.status_dot.setStyleSheet("color: rgba(255, 120, 30, 1.0);")
        QTimer.singleShot(4000, lambda: (
            self.status_label.setText(old),
            self.status_label.setStyleSheet(old_style),
            self.status_dot.setStyleSheet(old_dot),
        ))

    # ── Gaze handling ─────────────────────────────────────────────

    def _poll_gaze(self) -> None:
        if self.engine is None or self.engine.vision is None:
            return
        region = self.engine.vision.get_gaze_region()
        # "bot_right" or "right" → user looking at vitals area
        gaze_at_vitals = "right" in region or "bot" in region
        self.vitals_widget.set_gaze_active(gaze_at_vitals)

    def _on_gaze(self, region: str) -> None:
        pass  # placeholder for future gaze-driven actions

    # ── Overlay control ───────────────────────────────────────────

    def show_overlay(self) -> None:
        if self._overlay is not None:
            self._overlay.show()

    def hide_overlay(self) -> None:
        if self._overlay is not None:
            self._overlay.hide()

    def toggle_overlay(self) -> None:
        if self._overlay is None:
            return
        if self._overlay.isVisible():
            self._overlay.hide()
        else:
            self._overlay.show()

    def annotate_overlay(self, text: str, x: float = 0.5, y: float = 0.3) -> None:
        if self._overlay is not None:
            self._overlay.add_text(text, x=x, y=y)

    def clear_overlay(self) -> None:
        if self._overlay is not None:
            self._overlay.clear()

    def _update_camera(self):
        if self.engine is None:
            return
        try:
            frame = self.engine.get_camera_frame()
            if frame is None:
                return

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Mirror horizontally so the feed feels like a selfie camera
            # (detection happens on the original unflipped frame)
            rgb = cv2.flip(rgb, 1)

            h, w, ch = rgb.shape

            lw = self.camera_label.width() - 2
            lh = self.camera_label.height() - 2
            if lw > 10 and lh > 10:
                scale = min(lw / w, lh / h)
                nw, nh = int(w * scale), int(h * scale)
                rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
                h, w, ch = rgb.shape

            # Ensure contiguous array for QImage
            rgb = np.ascontiguousarray(rgb)
            bpl = ch * w
            img = QImage(rgb.data, w, h, bpl, QImage.Format.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(img))

            if self.engine.vision:
                dets = self.engine.vision.get_detections()
                if dets:
                    # Filter out boring labels for display
                    from . import config
                    interesting = [d for d in dets if d not in config.YOLO_IGNORE_LABELS]
                    if interesting:
                        self.detection_label.setText(f"◈ {', '.join(interesting).upper()}")
                    else:
                        self.detection_label.setText("")
        except Exception:
            pass  # Never crash the GUI timer


def create_and_run_gui(engine: JarvisEngine) -> None:
    """Create and run the JARVIS HUD."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("A-EYE JARVIS")
    app.setFont(QFont("Segoe UI", 10))

    window = JarvisWindow()
    window.set_engine(engine)

    # Wire engine callbacks → bridge signals
    engine.on_status = window.bridge.status_signal.emit
    engine.on_user_text = window.bridge.user_text_signal.emit
    engine.on_assistant_token = window.bridge.assistant_token_signal.emit
    engine.on_assistant_done = window.bridge.assistant_done_signal.emit
    engine.on_error = window.bridge.error_signal.emit
    engine.on_listening_state = window.bridge.listening_state_signal.emit
    engine.on_audio_level = window.bridge.audio_level_signal.emit
    engine.on_speaking = window.bridge.speaking_signal.emit
    engine.on_posture = window.bridge.posture_signal.emit

    engine.start()

    window.chat_display.add_system_message("J.A.R.V.I.S. systems online")
    window.chat_display.add_system_message("Always-on voice active — just speak, sir")

    window.showMaximized()
    sys.exit(app.exec())
