"""Timer and reminder tools.

- Timers live in memory only: nothing is written to disk, and they end when J.A.R.V.I.S quits.
- At most 10 at a time, each from 1 second to 24 hours.
- When one finishes, the app plays an alarm, says a fixed line ("Sir, your timer is done.") and
  shows the label on screen. The label is your own words, so it is never sent to the voice service.
- A background thread checks every quarter of a second; firing never blocks the gate.
"""
from __future__ import annotations

import itertools
import threading
import time
from dataclasses import dataclass
from typing import Callable

from ..gate.models import Param, Risk, Tool, ToolError

MAX_TIMERS = 10
MAX_SECONDS = 24 * 3600
LABEL_PATTERN = r"[A-Za-z0-9 ,.'!?&()\-]{0,60}"


def describe_seconds(seconds: int) -> str:
    """300 -> "5 minutes", 5400 -> "1 hour 30 minutes"."""
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    parts = [f"{n} {word}{'' if n == 1 else 's'}" for n, word in ((hours, "hour"), (minutes, "minute"), (secs, "second")) if n]
    return " ".join(parts) or "0 seconds"


@dataclass
class _Timer:
    number: int
    due: float
    seconds: int
    label: str


class TimerService:
    def __init__(self, on_fire: Callable[[str], None] | None = None,
                 clock: Callable[[], float] = time.monotonic, start: bool = True):
        self.on_fire = on_fire                  # told the label of each timer that finishes
        self._clock = clock
        self._timers: list[_Timer] = []
        self._numbers = itertools.count(1)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        if start:
            self._thread = threading.Thread(target=self._run, name="jarvis-timers", daemon=True)
            self._thread.start()

    # ---- used by the tools --------------------------------------------------
    def add(self, seconds: int, label: str) -> str:
        with self._lock:
            if len(self._timers) >= MAX_TIMERS:
                raise ToolError("You already have 10 timers. Cancel one first.")
            self._timers.append(_Timer(next(self._numbers), self._clock() + seconds, seconds, label.strip()))
        kind = "Reminder" if label.strip() else "Timer"
        return f"{kind} set for {describe_seconds(seconds)}."

    def describe(self) -> str:
        now = self._clock()
        with self._lock:
            timers = sorted(self._timers, key=lambda t: t.due)
        if not timers:
            return "You have no timers."
        rows = [f"{describe_seconds(max(1, round(t.due - now)))} left" + (f" for {t.label}" if t.label else "")
                for t in timers]
        return "Timers: " + "; ".join(rows) + "."

    def cancel(self, label: str) -> str:
        wanted = label.strip().lower()
        with self._lock:
            if not self._timers:
                raise ToolError("You have no timers to cancel.")
            if wanted in ("all", "everything", "all timers"):
                count = len(self._timers)
                self._timers.clear()
                return f"Cancelled all {count} timers." if count > 1 else "Cancelled your timer."
            if wanted:
                matches = [t for t in self._timers if wanted in t.label.lower()]
            else:
                matches = list(self._timers) if len(self._timers) == 1 else []
            if not matches:
                raise ToolError("I'm not sure which timer you mean. Say its name, or say all.")
            if len(matches) > 1:
                raise ToolError("More than one timer matches. Say its full name, or say all.")
            self._timers.remove(matches[0])
        return "Cancelled that timer."

    # ---- firing --------------------------------------------------------------
    def check(self) -> list[str]:
        """Remove and return the labels of timers that are due ("" for a plain timer)."""
        now = self._clock()
        with self._lock:
            due = sorted((t for t in self._timers if t.due <= now), key=lambda t: t.due)
            self._timers = [t for t in self._timers if t.due > now]
        return [t.label for t in due]

    def _run(self) -> None:
        while not self._stop.wait(0.25):
            for label in self.check():
                callback = self.on_fire
                if callback is not None:
                    try:
                        callback(label)
                    except Exception:
                        pass                    # a display problem must never stop the other timers

    def stop(self) -> None:
        self._stop.set()


def make_timer_tools(service: TimerService) -> list[Tool]:
    def set_timer(seconds: int, label: str) -> str:
        return service.add(seconds, label)

    def list_timers() -> str:
        return service.describe()

    def cancel_timer(label: str) -> str:
        return service.cancel(label)

    label_p = Param(str, max_len=60, pattern=LABEL_PATTERN)
    return [
        Tool("set_timer", "Start a timer or reminder. seconds is the total time in seconds (for example "
             "5 minutes = 300, 1 hour = 3600). label says what it is for, or is empty.", Risk.LOW,
             {"seconds": Param(int, min_value=1, max_value=MAX_SECONDS), "label": label_p}, set_timer),
        Tool("list_timers", "Say which timers are running and how long is left.", Risk.LOW, {}, list_timers,
             private_output=True),
        Tool("cancel_timer", "Cancel a timer by its label, or 'all'. Use an empty label if there is only one.",
             Risk.LOW, {"label": label_p}, cancel_timer),
    ]
