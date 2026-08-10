"""
Attendance Logger — writes a CSV row whenever a recognized (live) face is seen.

One CSV file per day: attendance/attendance_YYYY-MM-DD.csv
Per-person cooldown prevents duplicate entries within a configurable window.

Usage:
    from attendance_logger import AttendanceLogger
    logger = AttendanceLogger(cooldown_seconds=60)
    logged = logger.try_log("prattyan", confidence=0.82)
"""

import csv
import os
import time
from collections import defaultdict
from datetime import datetime


class AttendanceLogger:
    """
    Thread-safe (single-process) CSV attendance logger with per-person cooldown.

    Columns: Date | Time | Name | Confidence | Status
    """

    def __init__(self, log_dir: str = "attendance", cooldown_seconds: float = 60.0):
        """
        Args:
            log_dir:           Directory to write daily CSV files into.
            cooldown_seconds:  Minimum seconds between two log entries for the
                               same person. Prevents duplicate logging.
        """
        self.log_dir = log_dir
        self.cooldown_seconds = cooldown_seconds
        self._last_logged: dict[str, float] = defaultdict(float)

        os.makedirs(log_dir, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def try_log(self, name: str, confidence: float) -> bool:
        """
        Attempt to log an attendance entry for *name*.

        Returns True if a new entry was written (cooldown elapsed),
        False if the entry was suppressed (still within cooldown window).
        """
        now = time.monotonic()
        if now - self._last_logged[name] < self.cooldown_seconds:
            return False

        self._last_logged[name] = now
        dt = datetime.now()

        path = self._today_path(dt)
        is_new_file = not os.path.exists(path)

        with open(path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if is_new_file:
                writer.writerow(["Date", "Time", "Name", "Confidence", "Status"])
            writer.writerow([
                dt.strftime("%Y-%m-%d"),
                dt.strftime("%H:%M:%S"),
                name,
                f"{confidence:.4f}",
                "PRESENT",
            ])

        print(f"[ATTENDANCE] {name:20s} logged at {dt.strftime('%H:%M:%S')}  "
              f"(conf={confidence:.2f})  → {path}")
        return True

    def seconds_until_next(self, name: str) -> float:
        """Remaining cooldown seconds for *name* (0 if ready to log)."""
        elapsed = time.monotonic() - self._last_logged[name]
        return max(0.0, self.cooldown_seconds - elapsed)

    @property
    def today_log_path(self) -> str:
        """Absolute path of today's CSV file (may not exist yet)."""
        return self._today_path(datetime.now())

    # ── Internal ──────────────────────────────────────────────────────────────

    def _today_path(self, dt: datetime) -> str:
        filename = f"attendance_{dt.strftime('%Y-%m-%d')}.csv"
        return os.path.join(self.log_dir, filename)
