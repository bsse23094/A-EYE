"""System vitals — real-time CPU, memory, disk I/O, and Lahore time feed."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional

# Pakistan Standard Time = UTC+5
_TZ_PKT = timezone(timedelta(hours=5))


class SystemVitals:
    """Background poller for system metrics. All reads are thread-safe snapshots."""

    def __init__(self, poll_interval: float = 1.0) -> None:
        self._interval = poll_interval
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._cpu_pct: float = 0.0
        self._cpu_temp: Optional[float] = None
        self._ram_pct: float = 0.0
        self._ram_used_gb: float = 0.0
        self._ram_total_gb: float = 0.0
        self._disk_r_mbs: float = 0.0
        self._disk_w_mbs: float = 0.0

        # Disk delta tracking
        self._prev_r: int = 0
        self._prev_w: int = 0
        self._prev_ts: float = time.time()

        self._available = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _loop(self) -> None:
        try:
            import psutil  # type: ignore
        except ImportError:
            print("[Vitals] psutil not installed — run: pip install psutil")
            return

        self._available = True

        # Prime disk counters so first delta is not garbage
        try:
            dc = psutil.disk_io_counters()
            if dc:
                self._prev_r, self._prev_w = dc.read_bytes, dc.write_bytes
                self._prev_ts = time.time()
        except Exception:
            pass

        # Warm up the non-blocking cpu_percent
        try:
            psutil.cpu_percent(interval=None)
        except Exception:
            pass

        while self._running:
            try:
                cpu = psutil.cpu_percent(interval=None)
                vm = psutil.virtual_memory()

                # CPU temperature — platform-specific, graceful fallback
                cpu_temp: Optional[float] = None
                try:
                    temps = psutil.sensors_temperatures()
                    if temps:
                        for key in ("coretemp", "cpu_thermal", "k10temp", "acpitz"):
                            if key in temps and temps[key]:
                                cpu_temp = temps[key][0].current
                                break
                except Exception:
                    pass

                # Disk I/O bandwidth (MB/s) since last poll
                r_mbs = w_mbs = 0.0
                try:
                    dc = psutil.disk_io_counters()
                    if dc:
                        now = time.time()
                        dt = max(0.001, now - self._prev_ts)
                        r_mbs = (dc.read_bytes - self._prev_r) / (1024 ** 2) / dt
                        w_mbs = (dc.write_bytes - self._prev_w) / (1024 ** 2) / dt
                        self._prev_r, self._prev_w = dc.read_bytes, dc.write_bytes
                        self._prev_ts = now
                except Exception:
                    pass

                with self._lock:
                    self._cpu_pct = cpu
                    self._cpu_temp = cpu_temp
                    self._ram_pct = vm.percent
                    self._ram_used_gb = vm.used / (1024 ** 3)
                    self._ram_total_gb = vm.total / (1024 ** 3)
                    self._disk_r_mbs = max(0.0, r_mbs)
                    self._disk_w_mbs = max(0.0, w_mbs)

            except Exception:
                pass

            time.sleep(self._interval)

    def get(self) -> Dict[str, Any]:
        """Return a thread-safe snapshot including Lahore time."""
        with self._lock:
            snap: Dict[str, Any] = {
                "cpu_pct": self._cpu_pct,
                "cpu_temp": self._cpu_temp,
                "ram_pct": self._ram_pct,
                "ram_used_gb": self._ram_used_gb,
                "ram_total_gb": self._ram_total_gb,
                "disk_r_mbs": self._disk_r_mbs,
                "disk_w_mbs": self._disk_w_mbs,
                "available": self._available,
            }

        now_pkt = datetime.now(_TZ_PKT)
        snap["lahore_time"] = now_pkt.strftime("%H:%M:%S")
        snap["lahore_hour"] = now_pkt.hour
        snap["lahore_minute"] = now_pkt.minute
        return snap

    @property
    def is_available(self) -> bool:
        return self._available


def get_time_period(hour: int) -> str:
    """Map Lahore hour to a named time period for reactive background."""
    if 5 <= hour < 7:
        return "dawn"
    elif 7 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 20:
        return "evening"
    elif 20 <= hour < 22:
        return "dusk"
    else:
        return "night"


# Background gradient stops per time period
BG_GRADIENTS: Dict[str, str] = {
    "dawn": "stop:0 #0a0614, stop:0.5 #160a24, stop:1 #080412",
    "morning": "stop:0 #020508, stop:0.5 #040a14, stop:1 #020406",
    "afternoon": "stop:0 #030608, stop:0.5 #060c18, stop:1 #020508",
    "evening": "stop:0 #0c0604, stop:0.5 #1a0c04, stop:1 #0a0402",
    "dusk": "stop:0 #0a0408, stop:0.5 #120610, stop:1 #080208",
    "night": "stop:0 #010204, stop:0.5 #020408, stop:1 #010204",
}


def build_bg_stylesheet(hour: int) -> str:
    period = get_time_period(hour)
    stops = BG_GRADIENTS[period]
    return (
        f"QWidget {{ background: qlineargradient(x1:0, y1:0, x2:0.5, y2:1, {stops}); "
        f"color: #b0d0f0; font-family: 'Segoe UI', 'Consolas', monospace; }}"
    )
