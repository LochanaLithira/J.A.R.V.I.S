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
import math
import queue
import sys
import threading
import time
from collections import deque
from enum import Enum
from typing import Any, Callable

from . import config
from .brain import Brain, BrainError, GeminiBrain, Turn
from .credentials import CredentialError, get_secret
from .flow import Flow, FlowError
from .gate.gate import Gate
from .gate.models import Result
from .tools.timers import TimerService
from .voice.listen import CommandRecorder, GroqTranscriber, Heard, SileroVAD, SpeechInput, VoiceCheck
from .voice.keepalive import warm_all
from .voice.speak import FishSpeaker, Voice
from .voice.style import Style
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
            cache.append(GeminiBrain(key, config.GEMINI_MODELS, config.GEMINI_TIMEOUT_SECONDS,
                                     address=config.USER_ADDRESS))
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
                               end_silence=config.LISTEN_END_SILENCE_SECONDS,
                               min_speech=config.LISTEN_MIN_SPEECH_SECONDS,
                               start_prob=config.LISTEN_START_CONFIDENCE, start_frames=3)
    transcriber = GroqTranscriber(lambda: get_secret("stt_api_key"), config.STT_MODEL, config.STT_LANGUAGE)
    return SpeechInput(recorder, transcriber, audit, on_recorded=on_recorded,
                       unfinished_extra_seconds=config.LISTEN_UNFINISHED_EXTRA_SECONDS)


_NOTHING_HEARD = {
    "no speech": "I didn't hear anything.",
    "too short": "That was too short to understand.",
    "noise": "That sounded like background noise, so I ignored it.",
    "stopped": "Stopped.",
}


def describe_heard(heard: Heard) -> str:
    if heard.error:
        return f"NO: {heard.error}"
    if not heard.text:
        return _NOTHING_HEARD.get(heard.reason, "I couldn't make out any words.")
    return f'Heard: "{heard.text}"'


_STOP_PHRASES = {"stop", "cancel", "never mind", "nevermind", "nothing", "no thanks", "no thank you",
                 "thats all", "thats it", "thank you", "thanks", "nothing else", "forget it",
                 "goodbye", "bye", "bye bye", "good bye", "that will be all", "thats all for now", "thats all thanks",
                 "thats all thank you", "go to sleep", "ok thanks", "okay thanks", "ok thank you", "okay thank you",
                 "no thats all", "no thats it", "thanks thats all", "thank you thats all", "see you", "see you later",
                 "leave the conversation", "end the conversation", "stop listening"}


def is_stop_phrase(text: str) -> bool:
    """True for "that's all", "goodbye", "thanks, Jarvis", "stop" and the like, which end a
    conversation (and are never sent to the AI as a command)."""
    plain = " ".join("".join(ch for ch in text.lower() if ch.isalpha() or ch == " ").split())
    for prefix in ("hey jarvis ", "jarvis ", "ok ", "okay "):
        if plain.startswith(prefix) and plain[len(prefix):] in _STOP_PHRASES:
            return True
    plain = plain.removeprefix("jarvis ").removesuffix(" jarvis").removesuffix(" sir")
    return plain in _STOP_PHRASES


def sound_level(chunk: Any) -> float:
    """0 to 1 loudness of an audio slice, shaped so normal speech sits around the middle."""
    import numpy as np

    samples = np.asarray(chunk, dtype=np.float64)
    if samples.size == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(samples * samples)))
    return min(1.0, (rms / 1800.0) ** 0.7)


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
                 follow_up_on_question: bool = True,
                 follow_up_seconds: float = 5.0, follow_up_max_turns: int = 3,
                 conversation: bool = False, conversation_seconds: float = 8.0,
                 conversation_max_turns: int = 10, conversation_max_seconds: float = 180.0,
                 memory_turns: int = 6,
                 close_chime: Callable[[], None] = play_close_chime,
                 alarm: Callable[[], None] | None = None,
                 on_event: Callable[[str, dict], None] | None = None,
                 on_level: Callable[[float], None] | None = None,
                 style: Style | None = None,
                 retry_seconds: float = RETRY_SECONDS, mic_idle_seconds: float = MIC_IDLE_SECONDS):
        self._triggers, self._speech, self._get_flow, self._voice = triggers, speech, get_flow, voice
        self._warm = warm                     # reconnect to the cloud services while you speak
        self._follow_on_question = follow_up_on_question
        self._follow_seconds, self._follow_max = follow_up_seconds, follow_up_max_turns
        self._conversation = conversation      # keep listening after every reply, until goodbye or silence
        self._conv_seconds, self._conv_max_turns = conversation_seconds, conversation_max_turns
        self._conv_max_seconds = conversation_max_seconds
        self._memory_turns = max(1, min(6, memory_turns))
        self._close_chime = close_chime
        self._alarm = alarm or _play_alarm_and_wait
        self._announcements: queue.Queue[str] = queue.Queue()   # finished timers, from the timer thread
        self._on_event, self._on_level = on_event, on_level    # feed the on-demand screen
        self._style = style                    # how J.A.R.V.I.S phrases what it says
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
        self._announcements.put(("timer", label if isinstance(label, str) else ""))

    def say_line(self, text: str) -> None:
        """Speak a fixed line of our own (the startup greeting), safely from any thread: the loop
        mutes the wake-up while it speaks and drops what the microphone heard meanwhile."""
        if isinstance(text, str) and text.strip():
            self._announcements.put(("say", text.strip()))

    def _speak_line(self, text: str) -> bool:
        if self._voice is None:
            return False
        self._triggers.set_muted(True)
        try:
            return self._voice.say(Result(True, text), wait=True)
        finally:
            self._triggers.set_muted(False)

    def _announce_pending(self, mic: Any = None) -> None:
        while True:
            try:
                kind, label = self._announcements.get_nowait()
            except queue.Empty:
                return
            if kind == "say":
                self._emit("said", text=label)
                self._say_out(f"J.A.R.V.I.S: {label}")
                self._speak_line(label)
                if mic is not None:
                    mic.drain()
                continue
            log.info("timer finished")                     # the label is your words: never logged
            self._emit("timer", label=label)
            self._alarm()
            line = self._style.timer_line(bool(label)) if self._style is not None else TIMER_DONE_LINE
            spoken = self._speak_line(line)
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

    def _emit(self, kind: str, **data: Any) -> None:
        """Tell the screen what just happened (it may include your words: the screen is local)."""
        if self._on_event is not None:
            try:
                self._on_event(kind, data)
            except Exception as exc:
                log.error("screen event failed: %s", type(exc).__name__)

    def _level(self, chunk: Any) -> None:
        if self._on_level is not None:
            try:
                self._on_level(sound_level(chunk))
            except Exception:
                pass

    def _metered(self, chunks: Any) -> Any:
        """Pass microphone slices through, reporting the sound level to the screen on the way."""
        for chunk in chunks:
            self._level(chunk)
            yield chunk

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
            self._level(chunk)
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
        self._emit("wake", source=event.source, score=event.score)
        t0 = time.monotonic()
        if self._speech is None:
            heard = Heard(error="Speech input is not set up.")
        else:
            self._emit("hearing", follow_up=False)
            heard = self._speech.hear(self._metered(mic.chunks(self._interrupt, idle_timeout=self._mic_idle)))
        heard_s = time.monotonic() - t0
        self._say_out(describe_heard(heard))
        self._emit_heard(heard)
        if heard.error:
            self._notice(heard.error)
        elif heard.text and is_stop_phrase(heard.text):
            self._goodbye()                      # "hey jarvis... never mind"
        elif heard.text:
            act_s, voice_s, result = self._act(heard.text)
            # durations only, never words: shows where the time goes
            log.info("timing: listening and hearing %.1fs, AI and action %.1fs, voice %.1fs",
                     heard_s, act_s, voice_s)
            self._converse(mic, heard.text, result)

    # ---- conversation: carrying on without the wake word -----------------------------
    def _keeps_going(self, result: Any) -> bool:
        if result is None:
            return False
        if self._conversation:
            return True
        return self._follow_on_question and getattr(result, "question", False) is True

    @staticmethod
    def _turn(said: str, result: Any) -> Turn | None:
        """What to remember of one exchange: the AI's own words or proposed action, never a tool's
        output. Fails closed: a result that says neither gives nothing to remember."""
        if result is None:
            return None
        action = getattr(result, "action", None)
        if isinstance(action, dict):
            return Turn(said, action=action)
        if getattr(result, "ai_reply", False) is True or getattr(result, "question", False) is True:
            return Turn(said, reply=result.message)
        return None

    def _goodbye(self) -> None:
        """Close the conversation politely: a short farewell, then the closing tone."""
        if self._style is not None and self._voice is not None:
            line = self._style.farewell()
            self._emit("said", text=line)
            self._say_out(f"J.A.R.V.I.S: {line}")
            self._speak_line(line)
        self._close_chime()

    def _converse(self, mic: Any, said: str, result: Any) -> None:
        """Carry on listening without "hey jarvis": after every reply in conversation mode, or
        after a question otherwise. It ends on a goodbye ("that's all", "thanks, Jarvis"), on
        silence, at the turn or time limit, or when listening is paused, locked or asleep.
        The last few exchanges are remembered for this conversation only."""
        conv = self._conversation
        window = self._conv_seconds if conv else self._follow_seconds
        max_turns = self._conv_max_turns if conv else self._follow_max
        max_seconds = self._conv_max_seconds if conv else math.inf
        memory: deque = deque(maxlen=self._memory_turns)
        first = self._turn(said, result)
        if first is not None:
            memory.append(first)
        started, turns = time.monotonic(), 0
        while self._keeps_going(result) and self._speech is not None and self.listening_allowed():
            if turns >= max_turns or time.monotonic() - started >= max_seconds:
                self._close_chime()
                self._say_out("(say 'hey jarvis' to start again)")
                return
            turns += 1
            mic.drain()                          # never record J.A.R.V.I.S's own voice
            self._chime()
            self._say_out("(listening...)" if conv else "(listening for your answer...)")
            self._emit("hearing", follow_up=True, conversation=conv)
            heard = self._speech.hear(self._metered(mic.chunks(self._interrupt, idle_timeout=self._mic_idle)),
                                      no_speech_seconds=window)
            self._say_out(describe_heard(heard))
            self._emit_heard(heard)
            if heard.error:
                self._notice(heard.error)
                return
            if not heard.text:
                self._close_chime()              # silence: back to waiting for "hey jarvis"
                return
            if is_stop_phrase(heard.text):
                self._goodbye()
                return
            _, _, result = self._act(heard.text, tuple(memory))
            turn = self._turn(heard.text, result)
            if turn is not None:
                memory.append(turn)

    def _emit_heard(self, heard: Heard) -> None:
        if heard.error:
            self._emit("hear_error", message=heard.error)
        elif heard.text:
            self._emit("heard", text=heard.text)
        else:
            self._emit("nothing", reason=heard.reason)

    def _act(self, text: str, history: tuple[Turn, ...] = ()) -> tuple[float, float, Any]:
        """Run a heard command like say. Returns (seconds for AI and action, seconds speaking,
        the result, or None if nothing ran)."""
        t0 = time.monotonic()
        if self._get_flow is None:
            self._say_out("NO: The AI is not set up.")
            self._notice("The AI is not set up.")
            return 0.0, 0.0, None
        self._emit("thinking")
        try:
            flow = self._get_flow()
            result = flow.run(text, history) if history else flow.run(text)
        except (BrainError, CredentialError, FlowError) as exc:
            self._say_out(f"NO: {exc}")
            self._notice(str(exc))
            self._emit("reply", text=str(exc), ok=False, private=False)
            return time.monotonic() - t0, 0.0, None
        t1 = time.monotonic()
        self._say_out(("OK: " if result.ok else "NO: ") + result.message)
        private = getattr(result, "private", True) is not False
        # How J.A.R.V.I.S says it. Built from spoken_text(), so a private result only ever becomes
        # a fixed "it's on your screen" line; the real content stays on screen.
        to_speak = Result(result.ok, self._style.line(result)) if self._style is not None else result
        shown = result.message if private or self._style is None else to_speak.message
        self._emit("reply", text=shown, ok=bool(result.ok), private=private)
        spoken = False
        if self._voice is not None:
            self._triggers.set_muted(True)       # J.A.R.V.I.S's own voice must never wake it
            self._emit("speaking")
            try:
                spoken = self._voice.say(to_speak, wait=True)
            finally:
                self._triggers.set_muted(False)
                self._emit("spoke")
        if self._show_text is not None and (getattr(result, "private", True) or not spoken):
            self._show_text(result.message)      # "it's on your screen" must be true
        return t1 - t0, time.monotonic() - t1, result


def build_assistant(audit: Any, get_flow: Callable[[], Flow] | None, voice: Voice | None, *,
                    out: Callable[[str], None] | None = None, show_text: Callable[[str], Any] | None = None,
                    on_status: Callable[[Status, str], None] | None = None,
                    on_notice: Callable[[str], None] | None = None,
                    on_event: Callable[[str, dict], None] | None = None,
                    on_level: Callable[[float], None] | None = None,
                    style: Style | None = None) -> Assistant:
    """Everything the loop needs, from config.py. Raises TriggerError if the models or the
    microphone named in config.py are missing."""
    wake = WakeWordDetector(config.MODELS_DIR)
    clap = None
    if config.CLAP_ENABLED:
        on_clap = (lambda level: out(f"  (clap heard, level {level:.0f})")) if out else None
        clap = ClapDetector(config.CLAP_MIN_LEVEL, on_clap=on_clap)
    voice_check = (VoiceCheck(SileroVAD(config.MODELS_DIR), config.WAKE_MIN_SPEECH_SECONDS,
                              config.WAKE_SPEECH_CONFIDENCE) if config.WAKE_SPEECH_CHECK else None)
    triggers = Triggers(wake, clap, config.WAKE_THRESHOLD, config.WAKE_PATIENCE,
                        config.WAKE_COOLDOWN_SECONDS, audit=audit, voice_check=voice_check)
    return Assistant(triggers, make_speech(audit, on_recorded=play_done_chime), get_flow, voice,
                     pick_microphone(config.MIC_DEVICE), show_text=show_text, on_status=on_status,
                     on_notice=on_notice, out=out, warm=warm_all,
                     follow_up_on_question=config.FOLLOW_UP_ON_QUESTION,
                     follow_up_seconds=config.FOLLOW_UP_SECONDS, follow_up_max_turns=config.FOLLOW_UP_MAX_TURNS,
                     conversation=config.CONVERSATION_MODE, conversation_seconds=config.CONVERSATION_SECONDS,
                     conversation_max_turns=config.CONVERSATION_MAX_TURNS,
                     conversation_max_seconds=config.CONVERSATION_MAX_SECONDS,
                     memory_turns=config.CONVERSATION_MEMORY_TURNS,
                     on_event=on_event, on_level=on_level, style=style or Style(config.USER_ADDRESS))


# ---------------------------------------------------------------------------
# The background app
# ---------------------------------------------------------------------------

_KEYS = (("gemini_api_key", "the AI (Gemini)"), ("stt_api_key", "hearing commands (Groq)"),
         ("fish_api_key", "the voice (Fish Audio)"))


def key_status(get: Callable[[str], str | None] = get_secret) -> dict[str, bool]:
    """Which keys are stored (True/False). Never reveals a key."""
    status = {}
    for name, _ in _KEYS:
        try:
            status[name] = bool(get(name))
        except CredentialError:
            status[name] = False
    return status


_HUD_STATE = {Status.STARTING: "starting", Status.LISTENING: "listening", Status.WORKING: "working",
              Status.PAUSED: "paused", Status.LOCKED: "locked", Status.ASLEEP: "asleep",
              Status.PROBLEM: "problem", Status.STOPPED: "paused"}
_STATUS_ACTIVITY = {Status.PAUSED: ("listening paused", "warn"), Status.LOCKED: ("Windows locked: microphone off", "warn"),
                    Status.ASLEEP: ("asleep: microphone off", "warn")}
_NOTHING_ACTIVITY = {"no speech": "heard nothing", "too short": "too short to understand", "stopped": "stopped",
                     "noise": "ignored background noise"}


def screen_feed(screen: Any) -> tuple[Callable[[Status, str], None], Callable[[str, dict], None]]:
    """Turn the loop's status changes and events into updates for the on-demand screen."""

    def on_status(status: Status, problem: str) -> None:
        screen.set_state(_HUD_STATE.get(status, "listening"), problem)
        if status is Status.PROBLEM and problem:
            screen.add_activity(problem, "bad")
        elif status in _STATUS_ACTIVITY:
            screen.add_activity(*_STATUS_ACTIVITY[status])

    def on_event(kind: str, data: dict) -> None:
        if kind == "wake":
            score = data.get("score")
            screen.add_activity(f"wake · {data.get('source', '')}" + (f" {score:.2f}" if score is not None else ""))
        elif kind == "hearing":
            detail = ("in conversation" if data.get("conversation") else "your answer") if data.get("follow_up") \
                else "go ahead"
            screen.set_state("hearing", detail)
        elif kind == "heard":
            screen.add_message("you", data.get("text", ""))
        elif kind == "nothing":
            screen.add_activity(_NOTHING_ACTIVITY.get(data.get("reason", ""), "heard nothing"), "warn")
        elif kind == "hear_error":
            screen.add_message("system", data.get("message", ""))
            screen.add_activity("couldn't hear the command", "bad")
        elif kind == "thinking":
            screen.set_state("working", "thinking")
        elif kind == "reply":
            screen.add_message("jarvis", data.get("text", ""), private=bool(data.get("private")))
            screen.add_activity("done" if data.get("ok") else "not done", "ok" if data.get("ok") else "warn")
            screen.refresh_timers()
        elif kind == "speaking":
            screen.set_state("speaking")
        elif kind == "said":
            screen.add_message("jarvis", data.get("text", ""))
        elif kind == "timer":
            label = data.get("label", "")
            screen.add_message("system", f"Timer done: {label}" if label else "Your timer is done.")
            screen.add_activity("timer finished")
            screen.refresh_timers()

    return on_status, on_event


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

    from .ui.window import ScreenLink

    holder: dict[str, Assistant] = {}

    def toggle_pause() -> None:
        if "a" in holder:
            holder["a"].toggle_pause()

    def systems() -> list[dict]:
        keys = key_status()
        a = holder.get("a")
        return [{"name": "Microphone", "ok": a is not None and a.status is not Status.PROBLEM},
                {"name": "Wake word", "ok": a is not None},
                {"name": "Gemini", "ok": keys.get("gemini_api_key", False)},
                {"name": "Groq", "ok": keys.get("stt_api_key", False)},
                {"name": "Fish Audio", "ok": keys.get("fish_api_key", False)},
                {"name": "Audit chain intact", "ok": gate.audit.verify()[0]}]

    screen = ScreenLink(on_command=lambda cmd: toggle_pause() if cmd == "toggle_pause" else None,
                        get_systems=systems, get_timers=timers.snapshot)
    feed_status, feed_event = screen_feed(screen)
    tray = Tray(on_toggle_pause=toggle_pause,
                is_paused=lambda: holder["a"].paused if "a" in holder else False,
                on_lock=lambda locked: holder["a"].set_locked(locked) if "a" in holder else None,
                on_sleep=lambda asleep: holder["a"].set_asleep(asleep) if "a" in holder else None,
                on_open=screen.open)

    def on_status(status: Status, problem: str) -> None:
        tray.set_status(status, problem)
        feed_status(status, problem)

    def show_text(text: str) -> None:          # the screen already shows it when it is open
        if not screen.is_open:
            show_message(text)

    try:
        holder["a"] = build_assistant(gate.audit, get_flow, voice, show_text=show_text, on_status=on_status,
                                      on_notice=lambda m: tray.notify("J.A.R.V.I.S", m),
                                      on_event=feed_event, on_level=screen.set_level)
    except TriggerError as exc:
        log.warning("cannot listen: %s", exc)
        on_status(Status.PROBLEM, str(exc))
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
        if config.GREETING_ON_START:
            holder["a"].say_line(Style(config.USER_ADDRESS).greeting())
    try:
        tray.run()                       # the Windows message loop; returns when you choose Quit
    finally:
        timers.stop()
        screen.close()
        if "a" in holder:
            holder["a"].stop()
        voice.finish()
        log.info("stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
