"""J.A.R.V.I.S main program: runs quietly in the background with a tray icon.

    pythonw -m backend.main     no console window (normal use)
    python  -m backend.main     the same, but errors also print to this console

All the time it: waits for "hey jarvis" or a double clap (on this laptop), chimes, hears one
command (only your speech goes to Groq), lets the AI suggest an action, lets the gate decide,
and speaks the reply. The dev console's `listen` command runs this same loop.

Safety rules this file adds:
- Only one copy runs at a time. Two would fight over the microphone, and two writers would
  break the audit log's tamper-evident chain.
- The microphone is CLOSED while you pause J.A.R.V.I.S, while Windows is locked, and while the
  laptop sleeps, so nobody can command your laptop while you're away from it.
- Nothing crashes the program: a watchdog restarts the listening loop after an unexpected error
  and the tray icon shows when something is wrong. A microphone that stops sending sound (common
  after sleep) is reopened.
- Runs at below-normal priority, so it never slows your other work.
- The error log lives outside the workspace, is capped in size, and holds error types and fixed
  messages only: never your words, replies or audio.
"""
from __future__ import annotations

import logging
import logging.handlers
import queue
import sys
import threading
import time
from enum import Enum
from typing import Any, Callable

from . import config
from .brain import Brain, BrainError, GeminiBrain, Turn
from .credentials import CredentialError, get_secret
from .flow import Flow, FlowError
from .gate.gate import Gate
from .gate.models import Result
from .tools.timers import TimerService
from .voice.listen import CommandRecorder, GroqTranscriber, Heard, SileroVAD, SpeechInput
from .voice.keepalive import warm_all
from .voice.speak import FishSpeaker, Voice
from .voice.triggers import (ALARM_CHIME, ClapDetector, Microphone, TriggerError, Triggers, WakeEvent,
                             WakeWordDetector, pick_microphone, play_chime, play_close_chime, play_done_chime)

log = logging.getLogger("jarvis")

MIC_IDLE_SECONDS = 5.0           # a microphone silent at driver level for this long is reopened
TIMER_DONE_LINE = "Sir, your timer is done."
RETRY_SECONDS = 5.0              # wait before retrying after a problem (a missing mic, for example)


class Status(str, Enum):
    STARTING = "starting"
    LISTENING = "listening"
    WORKING = "working"
    PAUSED = "paused"
    LOCKED = "locked"
    ASLEEP = "asleep"
    PROBLEM = "problem"
    STOPPED = "stopped"


# ---------------------------------------------------------------------------
# Building the parts (shared with the dev console)
# ---------------------------------------------------------------------------

def make_brain_getter() -> Callable[[], Brain]:
    """Builds the Gemini brain on first use, so startup is fast and works offline."""
    cache: list[Brain] = []

    def get() -> Brain:
        if not cache:
            key = get_secret("gemini_api_key")
            if not key:
                raise BrainError("No Gemini key is stored. Run: python -m backend.credentials set gemini_api_key")
            cache.append(GeminiBrain(key, config.GEMINI_MODELS, config.GEMINI_TIMEOUT_SECONDS))
        return cache[0]

    return get


def make_flow_getter(gate: Gate, get_brain: Callable[[], Brain]) -> Callable[[], Flow]:
    cache: list[Flow] = []

    def get() -> Flow:
        if not cache:
            cache.append(Flow(get_brain(), gate))
        return cache[0]

    return get


def make_voice(report: Callable[[str], None] = print) -> Voice:
    """Builds the speaker. No network call happens here; the key is read only when speaking."""
    try:
        speaker = FishSpeaker(lambda: get_secret("fish_api_key"), voice_id=config.FISH_VOICE_ID)
    except ValueError as exc:               # a bad voice ID in config.py
        report(f"Voice is off: {exc}")
        return Voice(None, report)
    return Voice(speaker, report)


def make_speech(audit: Any, on_recorded: Callable[[], None] | None = None) -> SpeechInput:
    """Silero VAD (local) to find your speech, Groq Whisper to turn it into text."""
    recorder = CommandRecorder(SileroVAD(config.MODELS_DIR), max_seconds=config.LISTEN_MAX_SECONDS,
                               no_speech_seconds=config.LISTEN_NO_SPEECH_SECONDS,
                               end_silence=config.LISTEN_END_SILENCE_SECONDS)
    transcriber = GroqTranscriber(lambda: get_secret("stt_api_key"), config.STT_MODEL, config.STT_LANGUAGE)
    return SpeechInput(recorder, transcriber, audit, on_recorded=on_recorded)


_NOTHING_HEARD = {
    "no speech": "I didn't hear anything.",
    "too short": "That was too short to understand.",
    "stopped": "Stopped.",
}


def describe_heard(heard: Heard) -> str:
    if heard.error:
        return f"NO: {heard.error}"
    if not heard.text:
        return _NOTHING_HEARD.get(heard.reason, "I couldn't make out any words.")
    return f'Heard: "{heard.text}"'


_STOP_PHRASES = {"stop", "cancel", "never mind", "nevermind", "nothing", "no thanks", "no thank you",
                 "thats all", "thats it", "thank you", "thanks", "nothing else", "forget it"}


def is_stop_phrase(text: str) -> bool:
    """True for "stop", "cancel", "that's all" and the like, which end a follow-up."""
    plain = " ".join("".join(ch for ch in text.lower() if ch.isalpha() or ch == " ").split())
    plain = plain.removeprefix("jarvis ").removesuffix(" jarvis").removesuffix(" sir")
    return plain in _STOP_PHRASES


class NearMisses:
    """Spots wake-word score bumps that rose but stayed under the threshold (console tuning aid)."""

    def __init__(self, threshold: float):
        self._threshold, self._peak, self._quiet = threshold, 0.0, 0

    def update(self, score: float, woke: bool) -> float | None:
        if woke:
            self._peak, self._quiet = 0.0, 0
            return None
        if score > 0.10:
            self._peak, self._quiet = max(self._peak, score), 0
            return None
        if self._peak > 0:
            self._quiet += 1
            if self._quiet >= 3:
                peak, self._peak, self._quiet = self._peak, 0.0, 0
                return peak if peak <= self._threshold else None
        return None


# ---------------------------------------------------------------------------
# The listening loop
# ---------------------------------------------------------------------------

class Assistant:
    """Listen -> wake -> hear -> act -> speak, forever, until stopped.

    Control methods (pause, resume, set_locked, set_asleep, stop) are safe to call from any
    thread, for example from the tray icon. Any of them interrupts the microphone at once.

    `out` gets console lines (they include your words, so it is only used by the dev console;
    the background app leaves it unset). `show_text` puts a reply on screen when it is private
    or could not be spoken. `on_status` and `on_notice` feed the tray icon."""

    def __init__(self, triggers: Triggers, speech: SpeechInput | None,
                 get_flow: Callable[[], Flow] | None = None, voice: Voice | None = None,
                 device: int | None = None, *, mic_factory: Callable[..., Any] = Microphone,
                 chime: Callable[[], None] = play_chime, show_text: Callable[[str], Any] | None = None,
                 on_status: Callable[[Status, str], None] | None = None,
                 on_notice: Callable[[str], None] | None = None, out: Callable[[str], None] | None = None,
                 warm: Callable[[], None] | None = None,
                 follow_up_on_question: bool = True, follow_up_always: bool = False,
                 follow_up_seconds: float = 5.0, follow_up_max_turns: int = 3,
                 close_chime: Callable[[], None] = play_close_chime,
                 alarm: Callable[[], None] | None = None,
                 retry_seconds: float = RETRY_SECONDS, mic_idle_seconds: float = MIC_IDLE_SECONDS):
        self._triggers, self._speech, self._get_flow, self._voice = triggers, speech, get_flow, voice
        self._warm = warm                     # reconnect to the cloud services while you speak
        self._follow_on_question, self._follow_always = follow_up_on_question, follow_up_always
        self._follow_seconds, self._follow_max = follow_up_seconds, follow_up_max_turns
        self._close_chime = close_chime
        self._alarm = alarm or _play_alarm_and_wait
        self._announcements: queue.Queue[str] = queue.Queue()   # finished timers, from the timer thread
        self._device, self._mic_factory, self._chime = device, mic_factory, chime
        self._show_text, self._on_status, self._on_notice, self._out = show_text, on_status, on_notice, out
        self._retry, self._mic_idle = retry_seconds, mic_idle_seconds
        self._paused = self._locked = self._asleep = False
        self._stop = threading.Event()
        self._interrupt = threading.Event()   # set whenever the microphone must be let go at once
        self._thread: threading.Thread | None = None
        self._near = NearMisses(getattr(triggers, "_threshold", 0.25)) if out else None
        self.status, self.problem, self.crashes = Status.STARTING, "", 0

    # ---- control (any thread) ------------------------------------------------
    def _change(self, attr: str, value: bool) -> None:
        setattr(self, attr, value)
        self._interrupt.set()

    def pause(self) -> None:
        self._change("_paused", True)

    def resume(self) -> None:
        self._change("_paused", False)

    def toggle_pause(self) -> None:
        self._change("_paused", not self._paused)

    @property
    def paused(self) -> bool:
        return self._paused

    def set_locked(self, locked: bool) -> None:
        """Windows was locked (True) or unlocked (False)."""
        self._change("_locked", locked)

    def set_asleep(self, asleep: bool) -> None:
        """The laptop is going to sleep (True) or has woken up (False). Waking also reopens the mic."""
        self._change("_asleep", asleep)

    def announce(self, label: str) -> None:
        """A timer finished (called from the timer thread). Handled by the loop within ~0.1 s."""
        self._announcements.put(label if isinstance(label, str) else "")

    def _announce_pending(self, mic: Any = None) -> None:
        while True:
            try:
                label = self._announcements.get_nowait()
            except queue.Empty:
                return
            log.info("timer finished")                     # the label is your words: never logged
            self._alarm()
            spoken = False
            if self._voice is not None:
                self._triggers.set_muted(True)
                try:
                    spoken = self._voice.say(Result(True, TIMER_DONE_LINE), wait=True)
                finally:
                    self._triggers.set_muted(False)
            text = f"Timer done: {label}" if label else "Your timer is done."
            self._say_out(f"TIMER: {text}")
            if self._show_text is not None and (label or not spoken):
                self._show_text(text)                     # the label is shown, never spoken
            if mic is not None:
                mic.drain()

    def listening_allowed(self) -> bool:
        return not (self._paused or self._locked or self._asleep or self._stop.is_set())

    def start(self) -> None:
        self._thread = threading.Thread(target=self.run, name="jarvis-listen", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 15.0) -> None:
        self._stop.set()
        self._interrupt.set()
        if self._thread is not None:
            self._thread.join(timeout)

    # ---- the loop ---------------------------------------------------------------
    def _set_status(self, status: Status, problem: str = "") -> None:
        if (status, problem) == (self.status, self.problem):
            return
        self.status, self.problem = status, problem
        if problem:
            log.warning("status %s: %s", status.value, problem)
        else:
            log.info("status %s", status.value)
        if self._on_status is not None:
            try:
                self._on_status(status, problem)
            except Exception as exc:
                log.error("status display failed: %s", type(exc).__name__)

    def _notice(self, message: str) -> None:
        log.warning("notice: %s", message)       # messages here are fixed, safe texts
        if self._on_notice is not None:
            try:
                self._on_notice(message)
            except Exception as exc:
                log.error("notice display failed: %s", type(exc).__name__)

    def _say_out(self, line: str) -> None:
        if self._out is not None:
            self._out(line)

    def run(self) -> None:
        """The watchdog: run the loop, and restart it after any unexpected error."""
        backoff = 1.0
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self._loop()
            except Exception as exc:                 # a bug or a surprise: never let it end listening
                self.crashes += 1
                log.error("listening loop crashed (%s), restarting", type(exc).__name__)
                self._set_status(Status.PROBLEM, "Something went wrong, so listening was restarted.")
                if time.monotonic() - started > 60:
                    backoff = 1.0
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 60.0)
        self._set_status(Status.STOPPED)

    def _idle_status(self) -> Status:
        if self._paused:
            return Status.PAUSED
        if self._locked:
            return Status.LOCKED
        return Status.ASLEEP

    def _loop(self) -> None:
        while not self._stop.is_set():
            if not self.listening_allowed():
                self._set_status(self._idle_status())   # the microphone is closed here
                self._announce_pending()                # timers still ring while paused
                self._interrupt.wait(0.5)
                self._interrupt.clear()
                continue
            self._interrupt.clear()
            try:
                with self._mic_factory(self._device) as mic:
                    self._triggers.reset()
                    self._set_status(Status.LISTENING)
                    self._listen(mic)
            except TriggerError as exc:              # no microphone, or it stopped sending sound
                self._set_status(Status.PROBLEM, str(exc))
                self._interrupt.wait(self._retry)

    def _listen(self, mic: Any) -> None:
        for chunk in mic.chunks(self._interrupt, idle_timeout=self._mic_idle):
            if not self.listening_allowed():
                return
            if not self._announcements.empty():
                self._announce_pending(mic)
                self._triggers.reset()
            event = self._triggers.process(chunk)
            if event is None:
                if self._near is not None:
                    missed = self._near.update(self._triggers.last_score, False)
                    if missed is not None:
                        self._say_out(f"  (almost: score reached {missed:.2f})")
                continue
            self._handle(event, mic)
            if not self.listening_allowed():
                return
            mic.drain()                          # drop anything heard while busy
            self._triggers.reset()               # and start the wake-up fresh
            self._set_status(Status.LISTENING)

    def _handle(self, event: WakeEvent, mic: Any) -> None:
        self._set_status(Status.WORKING)
        self._chime()
        if self._warm is not None:
            try:
                self._warm()
            except Exception as exc:
                log.error("warm-up failed: %s", type(exc).__name__)
        detail = f" (score {event.score:.2f})" if event.score is not None else ""
        self._say_out(f"WAKE: {event.source}{detail}")
        t0 = time.monotonic()
        if self._speech is None:
            heard = Heard(error="Speech input is not set up.")
        else:
            heard = self._speech.hear(mic.chunks(self._interrupt, idle_timeout=self._mic_idle))
        heard_s = time.monotonic() - t0
        self._say_out(describe_heard(heard))
        if heard.error:
            self._notice(heard.error)
        elif heard.text:
            act_s, voice_s, result = self._act(heard.text)
            # durations only, never words: shows where the time goes
            log.info("timing: listening and hearing %.1fs, AI and action %.1fs, voice %.1fs",
                     heard_s, act_s, voice_s)
            self._follow_up(mic, heard.text, result)

    # ---- follow-ups: answering without the wake word -----------------------------
    def _wants_follow_up(self, result: Any) -> bool:
        if result is None:
            return False
        if self._follow_always:
            return True
        return self._follow_on_question and getattr(result, "question", False) is True

    def _follow_up(self, mic: Any, said: str, result: Any) -> None:
        """After a question (or after every reply, if that setting is on), chime and listen once
        more without "hey jarvis". Silence, "stop"/"cancel", or the turn limit end it."""
        turns = 0
        while self._wants_follow_up(result) and self._speech is not None and self.listening_allowed():
            if turns >= self._follow_max:
                self._close_chime()
                self._say_out("(say 'hey jarvis' to start again)")
                return
            turns += 1
            history = (Turn(said, result.message),) if getattr(result, "question", False) else ()
            mic.drain()                          # never record J.A.R.V.I.S's own voice
            self._chime()
            self._say_out("(listening for your answer...)")
            heard = self._speech.hear(mic.chunks(self._interrupt, idle_timeout=self._mic_idle),
                                      no_speech_seconds=self._follow_seconds)
            self._say_out(describe_heard(heard))
            if heard.error:
                self._notice(heard.error)
                return
            if not heard.text or is_stop_phrase(heard.text):
                self._close_chime()
                return
            _, _, result = self._act(heard.text, history)
            said = heard.text

    def _act(self, text: str, history: tuple[Turn, ...] = ()) -> tuple[float, float, Any]:
        """Run a heard command like say. Returns (seconds for AI and action, seconds speaking,
        the result, or None if nothing ran)."""
        t0 = time.monotonic()
        if self._get_flow is None:
            self._say_out("NO: The AI is not set up.")
            self._notice("The AI is not set up.")
            return 0.0, 0.0, None
        try:
            flow = self._get_flow()
            result = flow.run(text, history) if history else flow.run(text)
        except (BrainError, CredentialError, FlowError) as exc:
            self._say_out(f"NO: {exc}")
            self._notice(str(exc))
            return time.monotonic() - t0, 0.0, None
        t1 = time.monotonic()
        self._say_out(("OK: " if result.ok else "NO: ") + result.message)
        spoken = False
        if self._voice is not None:
            self._triggers.set_muted(True)       # J.A.R.V.I.S's own voice must never wake it
            try:
                spoken = self._voice.say(result, wait=True)
            finally:
                self._triggers.set_muted(False)
        if self._show_text is not None and (getattr(result, "private", True) or not spoken):
            self._show_text(result.message)      # "it's on your screen" must be true
        return t1 - t0, time.monotonic() - t1, result


def build_assistant(audit: Any, get_flow: Callable[[], Flow] | None, voice: Voice | None, *,
                    out: Callable[[str], None] | None = None, show_text: Callable[[str], Any] | None = None,
                    on_status: Callable[[Status, str], None] | None = None,
                    on_notice: Callable[[str], None] | None = None) -> Assistant:
    """Everything the loop needs, from config.py. Raises TriggerError if the models or the
    microphone named in config.py are missing."""
    wake = WakeWordDetector(config.MODELS_DIR)
    clap = None
    if config.CLAP_ENABLED:
        on_clap = (lambda level: out(f"  (clap heard, level {level:.0f})")) if out else None
        clap = ClapDetector(config.CLAP_MIN_LEVEL, on_clap=on_clap)
    triggers = Triggers(wake, clap, config.WAKE_THRESHOLD, config.WAKE_PATIENCE,
                        config.WAKE_COOLDOWN_SECONDS, audit=audit)
    return Assistant(triggers, make_speech(audit, on_recorded=play_done_chime), get_flow, voice,
                     pick_microphone(config.MIC_DEVICE), show_text=show_text, on_status=on_status,
                     on_notice=on_notice, out=out, warm=warm_all,
                     follow_up_on_question=config.FOLLOW_UP_ON_QUESTION, follow_up_always=config.FOLLOW_UP_ALWAYS,
                     follow_up_seconds=config.FOLLOW_UP_SECONDS, follow_up_max_turns=config.FOLLOW_UP_MAX_TURNS)


# ---------------------------------------------------------------------------
# The background app
# ---------------------------------------------------------------------------

_KEYS = (("gemini_api_key", "the AI (Gemini)"), ("stt_api_key", "hearing commands (Groq)"),
         ("fish_api_key", "the voice (Fish Audio)"))


def startup_checks(get: Callable[[str], str | None] = get_secret) -> list[str]:
    """Plain-language list of what still needs setting up. Never reveals a key."""
    problems = []
    for name, purpose in _KEYS:
        try:
            present = bool(get(name))
        except CredentialError:
            return ["The Windows Credential Manager is not available, so no keys can be read."]
        if not present:
            problems.append(f"No key for {purpose}. Run: python -m backend.credentials set {name}")
    return problems


def setup_logging(path=None, console: bool = True) -> logging.Handler:
    """A small rotating error log (1 MB, 2 old copies) outside the workspace."""
    path = path or config.APP_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(path, maxBytes=1_000_000, backupCount=2, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    if console and sys.stderr is not None:
        log.addHandler(logging.StreamHandler(sys.stderr))
    return handler


def acquire_single_instance(name: str = "Local\\JARVIS-assistant") -> Any:
    """Return a handle that must be kept for as long as the program runs, or None if another
    copy already holds it. Windows releases it automatically when the program ends."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
    kernel32.CreateMutexW.restype = wintypes.HANDLE
    handle = kernel32.CreateMutexW(None, False, name)
    if not handle:
        return None
    if ctypes.get_last_error() == 183:          # ERROR_ALREADY_EXISTS
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle(handle)
        return None
    return handle


def lower_priority() -> bool:
    """Below-normal priority: J.A.R.V.I.S gives way to everything you are doing."""
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32")
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    return bool(kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x00004000))


def _play_alarm_and_wait() -> None:
    """Play the timer alarm to the end (the voice that follows would otherwise cut it off)."""
    from .voice.speak import WinsoundPlayer

    WinsoundPlayer().play(ALARM_CHIME)


def _prepare(get_flow: Callable[[], Flow]) -> None:
    """Load the AI library and build the flow now, so your first command doesn't wait ~2 s for it.
    No network call is made. A missing key is reported by startup_checks instead."""
    try:
        get_flow()
    except Exception as exc:
        log.info("AI not prepared at startup: %s", type(exc).__name__)


def _message_box(text: str) -> None:
    import ctypes

    ctypes.windll.user32.MessageBoxW(None, text, "J.A.R.V.I.S", 0x40)   # information icon


def main() -> int:
    from .registry import build_gate
    from .ui.confirm import popup_confirm, show_message
    from .ui.tray import Tray

    setup_logging()
    instance = acquire_single_instance()
    if instance is None:
        _message_box("J.A.R.V.I.S is already running. Look for its icon next to the clock.")
        return 1
    lower_priority()
    log.info("starting")

    timers = TimerService()
    gate = build_gate(confirm=popup_confirm, timers=timers)
    voice = make_voice(report=log.warning)
    get_flow = make_flow_getter(gate, make_brain_getter())
    threading.Thread(target=_prepare, args=(get_flow,), name="jarvis-prepare", daemon=True).start()

    holder: dict[str, Assistant] = {}
    tray = Tray(on_toggle_pause=lambda: holder["a"].toggle_pause() if "a" in holder else None,
                is_paused=lambda: holder["a"].paused if "a" in holder else False,
                on_lock=lambda locked: holder["a"].set_locked(locked) if "a" in holder else None,
                on_sleep=lambda asleep: holder["a"].set_asleep(asleep) if "a" in holder else None)
    try:
        holder["a"] = build_assistant(gate.audit, get_flow, voice, show_text=show_message,
                                      on_status=tray.set_status, on_notice=lambda m: tray.notify("J.A.R.V.I.S", m))
    except TriggerError as exc:
        log.warning("cannot listen: %s", exc)
        tray.set_status(Status.PROBLEM, str(exc))
        tray.notify("J.A.R.V.I.S can't listen yet", str(exc))
    if "a" in holder:
        timers.on_fire = holder["a"].announce
    else:                                # no listening loop: still ring, label on screen only
        timers.on_fire = lambda label: (tray.notify("J.A.R.V.I.S", "Your timer is done."),
                                        show_message(f"Timer done: {label}") if label else None)
    problems = startup_checks()
    if problems:
        tray.notify("J.A.R.V.I.S needs setting up", "\n".join(problems))
    if "a" in holder:
        holder["a"].start()
    try:
        tray.run()                       # the Windows message loop; returns when you choose Quit
    finally:
        timers.stop()
        if "a" in holder:
            holder["a"].stop()
        voice.finish()
        log.info("stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
