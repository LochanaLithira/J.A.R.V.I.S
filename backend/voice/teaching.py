"""Teaching J.A.R.V.I.S your voice from the screen, one sample at a time, at your own pace.

    start  -> listening pauses, the model loads, "Sample 1 of 20: press Record"
    record -> ~2 s getting ready, a chime, you say "hey jarvis": got it, or try again
    redo   -> forget the last sample and take it again
    cancel -> everything collected so far is thrown away, nothing is saved
    (after 20: compare with other voices, set the bar, save, switch the check on)

The screen can only press these buttons; it never receives audio or voice numbers, just
progress ("sample 7 got it"). Between presses only the voice numbers of the samples taken so
far are held in memory, never sound. If nothing happens for IDLE_SECONDS the session cancels
itself and listening resumes, so unfinished samples are never left lying around.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable

import numpy as np

from .enroll import SAMPLE_MIN_MATCH, SAMPLES, TIPS
from .voiceprint import (CHECK_SECONDS, Voiceprint, VoiceprintError, calibrate, normalize, similarity)

IDLE_SECONDS = 300


class TeachingSession:
    """One teaching (or test) run. All methods return at once; the work happens on a thread and
    progress arrives through `on_update(dict)`.

    The real microphone, models and saving are passed in, so the tests drive it with fakes:
      prepare()            -> (embed, listen) : load the model; listen(stop, on_ready) gives one
                              wake moment (list of slices) or None
      voices()             -> [(name, clip)] other voices for comparison
      save(voiceprint)     -> store it and switch the voice check on
      pause_listening(bool)-> close (True) or reopen (False) the main microphone
    """

    def __init__(self, prepare: Callable[[], tuple[Callable, Callable]], voices: Callable[[], list],
                 save: Callable[[Voiceprint], None], pause_listening: Callable[[bool], None],
                 on_update: Callable[[dict], None], clock: Callable[[], float] = time.monotonic,
                 idle_seconds: float = IDLE_SECONDS, current: Voiceprint | None = None):
        self._prepare, self._voices, self._save = prepare, voices, save
        self._pause, self._update, self._clock = pause_listening, on_update, clock
        self._idle = idle_seconds
        self._current = current                  # your saved voiceprint, for the test button
        self._lock = threading.Lock()
        self._stop = threading.Event()           # set by cancel: stops a recording at once
        self._busy = False
        self._mode: str | None = None            # "teach" or "test"
        self._embed = self._listen = None
        self._kept: list[np.ndarray] = []
        self._last_action = clock()
        self.phase = "idle"

    # ---- buttons (any thread) ----------------------------------------------------------
    def start(self, mode: str = "teach") -> bool:
        with self._lock:
            if self._mode is not None or mode not in ("teach", "test"):
                return False
            if mode == "test" and self._current is None:
                self._send("result", message="There's no voiceprint to test yet. Teach J.A.R.V.I.S your voice first.")
                return False
            self._mode, self._kept, self._busy = mode, [], True
            self._stop.clear()
            self._last_action = self._clock()
        threading.Thread(target=self._begin, name="jarvis-teach", daemon=True).start()
        return True

    def record(self) -> bool:
        with self._lock:
            if self._mode is None or self._busy or self.phase != "waiting":
                return False
            self._busy = True
            self._last_action = self._clock()
        threading.Thread(target=self._take_sample, name="jarvis-teach-sample", daemon=True).start()
        return True

    def redo(self) -> bool:
        with self._lock:
            if self._mode != "teach" or self._busy or not self._kept:
                return False
            self._kept.pop()
            self._last_action = self._clock()
        self._waiting("Took that one off. Record it again.")
        return True

    def cancel(self) -> None:
        with self._lock:
            if self._mode is None:
                return
            self._stop.set()
            busy = self._busy
        if not busy:
            self._finish("cancelled", message="Cancelled. Nothing was saved.")

    def check_idle(self) -> None:
        """Called now and then: an abandoned session cancels itself."""
        with self._lock:
            idle = self._mode is not None and not self._busy and self._clock() - self._last_action > self._idle
        if idle:
            self._stop.set()
            self._finish("cancelled", message="Teaching was left for a while, so it was cancelled. Nothing was saved.")

    @property
    def active(self) -> bool:
        return self._mode is not None

    # ---- the work (on a thread) ------------------------------------------------------------
    def _send(self, phase: str, **data: Any) -> None:
        self.phase = phase
        update = {"phase": phase, "mode": self._mode, "of": SAMPLES, "done": len(self._kept), **data}
        try:
            self._update(update)
        except Exception:
            pass

    def _waiting(self, message: str = "", outcome: str = "") -> None:
        tip = TIPS[(len(self._kept) // 4) % len(TIPS)]
        self._send("waiting", tip=tip, message=message, outcome=outcome,
                   can_redo=self._mode == "teach" and bool(self._kept))

    def _begin(self) -> None:
        self._pause(True)
        self._send("preparing", message="Getting ready...")
        try:
            self._embed, self._listen = self._prepare()
        except VoiceprintError as exc:
            self._finish("failed", message=str(exc))
            return
        except Exception:
            self._finish("failed", message="The microphone or the voice model couldn't start.")
            return
        with self._lock:
            self._busy = False
        if self._stop.is_set():
            self._finish("cancelled", message="Cancelled. Nothing was saved.")
            return
        self._waiting("Press Record, then say \"hey jarvis\" after the chime.")

    def _take_sample(self) -> None:
        self._send("getting_ready", message="Getting ready...")
        try:
            moment = self._listen(self._stop, lambda: self._send("say_it", message="Say \"hey jarvis\" now."))
        except Exception:
            moment = None
        if self._stop.is_set():
            self._finish("cancelled", message="Cancelled. Nothing was saved.")
            return
        outcome, message = self._use(moment)
        with self._lock:
            self._busy = False
            self._last_action = self._clock()
        if self._mode == "test":
            self._finish("result", **message)
            return
        if len(self._kept) >= SAMPLES:
            self._compare()
            return
        self._waiting(message, outcome)

    def _use(self, moment: list | None) -> tuple[str, Any]:
        """Turn one wake moment into numbers (the sound is dropped here) and judge it."""
        if not moment:
            return "missed", "I didn't hear \"hey jarvis\". Try again." if self._mode == "teach" else \
                {"message": "I didn't hear \"hey jarvis\". Press Test to try again."}
        audio = np.concatenate([np.asarray(c, dtype=np.int16).reshape(-1) for c in moment])
        moment = None
        try:
            numbers = self._embed(audio[-int(CHECK_SECONDS * 16000):])
        except VoiceprintError:
            numbers = None
        finally:
            audio = None
        if numbers is None:
            return "missed", "That was too short. Try again." if self._mode == "teach" else \
                {"message": "That was too short. Press Test to try again."}
        if self._mode == "test":
            score = similarity(numbers, np.array(self._current.centroid))
            ok = score >= self._current.threshold
            return "result", {"score": round(score, 3), "bar": round(self._current.threshold, 3), "recognised": ok,
                              "message": "That's you." if ok else "Not recognised as you."}
        if len(self._kept) >= 5 and similarity(numbers, normalize(np.mean(self._kept, axis=0))) < SAMPLE_MIN_MATCH:
            return "odd", "That didn't sound like the others (another voice or a noise?). Try again."
        self._kept.append(numbers)
        return "got_it", "Got it."

    def _compare(self) -> None:
        self._send("comparing", message="Comparing your voice with other voices...")
        others = []
        try:
            for _name, clip in self._voices():
                try:
                    others.append(self._embed(clip[: int(CHECK_SECONDS * 16000)]))
                except VoiceprintError:
                    continue
            result = calibrate(self._kept, others)
            if result.voiceprint is not None:
                self._save(result.voiceprint)
                self._current = result.voiceprint
        except VoiceprintError as exc:
            self._finish("failed", message=str(exc))
            return
        except Exception:
            self._finish("failed", message="Saving the voiceprint failed, so nothing was changed.")
            return
        self._finish("result", saved=result.voiceprint is not None, message=result.message,
                     yours=[round(result.yours[0], 3), round(result.yours[1], 3)],
                     others=round(result.others, 3),
                     bar=round(result.voiceprint.threshold, 3) if result.voiceprint else None)

    def _finish(self, phase: str, **data: Any) -> None:
        with self._lock:
            if self._mode is None:
                return
            self._kept = []                       # voice numbers are not kept after a session
            self._embed = self._listen = None
            self._busy = False
        self._send(phase, **data)
        with self._lock:
            self._mode = None
            self.phase = "idle"
        try:
            self._pause(False)
        except Exception:
            pass
