"""Dev console to exercise the gate, the AI, voice output, wake-up triggers and speech input by hand.

Run from the repo root:  python -m backend.cli
"""
from __future__ import annotations

import json
import threading
from typing import Any, Callable

from . import config
from .brain import Brain, BrainError, GeminiBrain
from .credentials import CredentialError, get_secret
from .flow import Flow, FlowError
from .gate.audit import AuditLog
from .gate.gate import Gate
from .gate.models import Result
from .registry import build_gate
from .ui.confirm import popup_confirm
from .voice.listen import CommandRecorder, GroqTranscriber, Heard, SileroVAD, SpeechInput
from .voice.speak import FishSpeaker, Speaker, spoken_text
from .voice.triggers import (ClapDetector, Microphone, TriggerError, Triggers, WakeWordDetector,
                             pick_microphone, play_chime)

MAX_LINE = 10_000

HELP = """Commands:
  tools                      list the tools and their risk level
  <tool> {json arguments}    run a tool through the gate
  ask <sentence>             DRY RUN: sends your sentence to Google's Gemini AI and shows
                             what it suggests. Nothing is executed.
  say <sentence>             REAL RUN: the AI suggests an action, the gate checks it, and it
                             runs only if allowed. Also sends your sentence to Google.
  voice on | voice off       speak the replies to say commands (on by default)
  hear                       DRY RUN: record one spoken command now and show the text.
                             Nothing is executed. Your speech is sent to Groq.
  listen                     REAL RUN, hands free: wait for "hey jarvis" or a double clap,
                             chime, then hear your command and run it like say (Ctrl+C stops).
                             Waiting for the wake-up happens on this laptop; only the command
                             you speak after the chime is sent to Groq.
  log                        show the last 5 audit entries
  verify                     check that the audit log was not tampered with
  help, quit

Examples:
  list_notes
  create_note {"title": "ideas", "text": "first note"}
  read_note {"title": "ideas"}
  open_app {"name": "notepad"}
  ask open notepad
  say open notepad
  listen      then say "hey jarvis", wait for the chime, and say "open notepad"

Privacy: on the free Gemini tier, Google may review your sentences and use them to
improve its products. Never put anything private in an ask.
Spoken commands go to Groq, which does not keep them by default (turn on Zero Data
Retention in the Groq console to be sure). Silence and anything before the chime stay here.
Spoken replies go to Fish Audio, which may keep them. Private results (note text, note
titles) are never sent: you hear a fixed line and read the real answer on screen.
"""


class Voice:
    """Speaks console replies in the background, so the prompt never waits for Fish Audio."""

    def __init__(self, speaker: Speaker | None, report: Callable[[str], None] = print,
                 background: bool = True):
        self._speaker = speaker
        self._report = report
        self._background = background
        self._thread: threading.Thread | None = None
        self.enabled = speaker is not None

    def set(self, on: bool) -> str:
        if self._speaker is None:
            return "Voice is not set up."
        self.enabled = on
        if not on:
            self._speaker.stop()
        return "Voice on." if on else "Voice off."

    def say(self, result: Result, wait: bool = False) -> None:
        """Speak a result. With wait=True, return only when speaking has finished (used while
        listening, so the microphone can be muted for exactly as long as J.A.R.V.I.S talks)."""
        if not self.enabled or self._speaker is None:
            return
        speaker, text = self._speaker, spoken_text(result)   # private results never leave the laptop

        def work() -> None:
            if not speaker.speak(text, wait=True):
                self._report(f"(voice: {speaker.last_error})")

        if self._background and not wait:
            self._thread = threading.Thread(target=work, daemon=True)
            self._thread.start()
        else:
            work()

    def finish(self, timeout: float = 30.0) -> None:
        """Let the last reply finish speaking before the console closes."""
        if self._thread is not None:
            self._thread.join(timeout)

def _recent_log(audit: AuditLog, n: int = 5) -> str:
    if not audit.path.exists():
        return "(log is empty)"
    rows = []
    for raw in audit.path.read_text(encoding="utf-8").splitlines()[-n:]:
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            rows.append("(unreadable line)")
            continue
        rows.append(f"{entry.get('event', '?'):<12} {entry.get('tool', '')} {entry.get('reason', '')}".rstrip())
    return "\n".join(rows) or "(log is empty)"


def _ask(gate: Gate, get_brain: Callable[[], Brain] | None, sentence: str) -> str:
    """Dry run: show what the AI suggests, and run nothing."""
    if get_brain is None:
        return "NO: The AI is not set up."
    try:
        reply = get_brain().propose(sentence, gate.tool_specs())
    except (BrainError, CredentialError) as exc:
        return f"NO: {exc}"
    if reply.proposal is not None:
        args = json.dumps(reply.proposal.args, sort_keys=True)
        return f"AI suggests: {reply.proposal.tool} {args}   (dry run, nothing was run)"
    return f"AI says: {reply.text}"


def _say(get_flow: Callable[[], Flow] | None, sentence: str, voice: Voice | None = None) -> str:
    """Real run: brain -> gate -> tool. The gate still decides everything."""
    if get_flow is None:
        return "NO: The AI is not set up."
    try:
        result = get_flow().run(sentence)
    except (BrainError, CredentialError, FlowError) as exc:
        return f"NO: {exc}"
    if voice is not None:
        voice.say(result)
    return ("OK: " if result.ok else "NO: ") + result.message


_NOTHING_HEARD = {
    "no speech": "I didn't hear anything.",
    "too short": "That was too short to understand.",
    "stopped": "Stopped.",
}


def _describe_heard(heard: Heard) -> str:
    if heard.error:
        return f"NO: {heard.error}"
    if not heard.text:
        return _NOTHING_HEARD.get(heard.reason, "I couldn't make out any words.")
    return f'Heard: "{heard.text}"'


def _hear(get_hearer: Callable[[], Callable[[], Heard]] | None) -> str:
    """Dry run: record one spoken command and show the text. Nothing runs."""
    if get_hearer is None:
        return "NO: Speech input is not set up."
    try:
        heard = get_hearer()()
    except TriggerError as exc:
        return f"NO: {exc}"
    except KeyboardInterrupt:
        return "Stopped."
    line = _describe_heard(heard)
    return line + ("   (dry run, nothing was run)" if heard.text else "")


def _listen(get_listener: Callable[[], Callable[[], None]] | None) -> str:
    """Run the wake-up triggers until Ctrl+C."""
    if get_listener is None:
        return "NO: Listening is not set up."
    try:
        get_listener()()
    except TriggerError as exc:
        return f"NO: {exc}"
    except KeyboardInterrupt:
        pass
    return "Stopped listening."


def handle_line(gate: Gate, audit: AuditLog, line: str,
                get_brain: Callable[[], Brain] | None = None,
                get_flow: Callable[[], Flow] | None = None,
                voice: Voice | None = None,
                get_listener: Callable[[], Callable[[], None]] | None = None,
                get_hearer: Callable[[], Callable[[], Heard]] | None = None) -> str | None:
    """Run one console line. Returns text to print, or None to quit."""
    if len(line) > MAX_LINE:
        return "Input too long."
    line = line.strip()
    if not line:
        return ""
    if line in ("quit", "exit"):
        return None
    if line == "help":
        return HELP
    if line == "tools":
        return "\n".join(f"- {s['name']} [{s['risk']}]: {s['description']}" for s in gate.tool_specs())
    if line == "log":
        return _recent_log(audit)
    if line == "verify":
        ok, bad_line = audit.verify()
        return "Audit log OK" if ok else f"Audit log TAMPERED (first bad line: {bad_line})"
    if line in ("voice on", "voice off"):
        return voice.set(line == "voice on") if voice is not None else "Voice is not set up."
    if line == "listen":
        return _listen(get_listener)
    if line == "hear":
        return _hear(get_hearer)

    name, _, rest = line.partition(" ")
    if name == "ask":
        return _ask(gate, get_brain, rest)
    if name == "say":
        return _say(get_flow, rest, voice)
    try:
        args = json.loads(rest) if rest.strip() else {}
    except json.JSONDecodeError:
        return 'Arguments must be JSON, for example {"title": "ideas"}'
    result = gate.execute(name, args)
    return ("OK: " if result.ok else "NO: ") + result.message


def _make_brain_getter() -> Callable[[], Brain]:
    """Builds the Gemini brain on first use, so the console starts fast and offline."""
    cache: list[Brain] = []

    def get() -> Brain:
        if not cache:
            key = get_secret("gemini_api_key")
            if not key:
                raise BrainError("No Gemini key is stored. Run: python -m backend.credentials set gemini_api_key")
            cache.append(GeminiBrain(key, config.GEMINI_MODELS, config.GEMINI_TIMEOUT_SECONDS))
        return cache[0]

    return get


def _make_flow_getter(gate: Gate, get_brain: Callable[[], Brain]) -> Callable[[], Flow]:
    cache: list[Flow] = []

    def get() -> Flow:
        if not cache:
            cache.append(Flow(get_brain(), gate))
        return cache[0]

    return get


def _make_voice() -> Voice:
    """Builds the speaker. No network call happens here, and the key is read only when speaking."""
    try:
        speaker = FishSpeaker(lambda: get_secret("fish_api_key"), voice_id=config.FISH_VOICE_ID)
    except ValueError as exc:               # a bad voice ID in config.py
        print(f"Voice is off: {exc}")
        return Voice(None)
    return Voice(speaker)


class _NearMisses:
    """Spots score bumps that rose but stayed under the threshold, to help tuning."""

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


def _make_speech(audit: AuditLog) -> SpeechInput:
    """Silero VAD (local) to find your speech, Groq Whisper to turn it into text."""
    recorder = CommandRecorder(SileroVAD(config.MODELS_DIR), max_seconds=config.LISTEN_MAX_SECONDS,
                               no_speech_seconds=config.LISTEN_NO_SPEECH_SECONDS,
                               end_silence=config.LISTEN_END_SILENCE_SECONDS)
    transcriber = GroqTranscriber(lambda: get_secret("stt_api_key"), config.STT_MODEL, config.STT_LANGUAGE)
    return SpeechInput(recorder, transcriber, audit)


def _make_hearer(audit: AuditLog, out: Callable[[str], None] = print) -> Callable[[], Callable[[], Heard]]:
    """Builds the one-shot recorder for the hear command on first use."""

    def get() -> Callable[[], Heard]:
        speech = _make_speech(audit)
        device = pick_microphone(config.MIC_DEVICE)

        def hear() -> Heard:
            with Microphone(device) as mic:
                play_chime()
                out("Speak now (after the chime)...")
                return speech.hear(mic.chunks())

        return hear

    return get


def _act(text: str, get_flow: Callable[[], Flow] | None, voice: Voice | None, triggers: Triggers,
         out: Callable[[str], None]) -> None:
    """Run a heard command exactly like say, then speak the reply with the wake-up muted."""
    if get_flow is None:
        out("NO: The AI is not set up.")
        return
    try:
        result = get_flow().run(text)
    except (BrainError, CredentialError, FlowError) as exc:
        out(f"NO: {exc}")
        return
    out(("OK: " if result.ok else "NO: ") + result.message)
    if voice is not None:
        triggers.set_muted(True)            # J.A.R.V.I.S's own voice must never wake it
        try:
            voice.say(result, wait=True)
        finally:
            triggers.set_muted(False)


def _make_listener(audit: AuditLog, out: Callable[[str], None] = print,
                   get_flow: Callable[[], Flow] | None = None,
                   voice: Voice | None = None) -> Callable[[], Callable[[], None]]:
    """Builds the live listener on first use. The models are checked when listening starts."""

    def get() -> Callable[[], None]:
        wake = WakeWordDetector(config.MODELS_DIR)
        clap = (ClapDetector(config.CLAP_MIN_LEVEL, on_clap=lambda level: out(f"  (clap heard, level {level:.0f})"))
                if config.CLAP_ENABLED else None)
        triggers = Triggers(wake, clap, config.WAKE_THRESHOLD, config.WAKE_PATIENCE,
                            config.WAKE_COOLDOWN_SECONDS, audit=audit)
        speech = _make_speech(audit)
        device = pick_microphone(config.MIC_DEVICE)

        def listen() -> None:
            near = _NearMisses(config.WAKE_THRESHOLD)
            with Microphone(device) as mic:
                out(f"Listening on {mic.name}. Say 'hey jarvis' or clap twice, wait for the chime, "
                    "then say your command. Ctrl+C to stop.")
                for chunk in mic.chunks():
                    event = triggers.process(chunk)
                    if event is not None:
                        play_chime()
                        detail = f" (score {event.score:.2f})" if event.score is not None else ""
                        out(f"WAKE: {event.source}{detail}")
                        heard = speech.hear(mic.chunks())
                        out(_describe_heard(heard))
                        if heard.text:
                            _act(heard.text, get_flow, voice, triggers, out)
                        mic.drain()                 # drop anything heard while busy
                        triggers.reset()            # and start the wake-up fresh
                        continue
                    missed = near.update(triggers.last_score, False)
                    if missed is not None:
                        out(f"  (almost: score reached {missed:.2f})")

        return listen

    return get


def main() -> None:
    gate = build_gate(confirm=popup_confirm)
    audit = gate.audit                      # one shared log, so the chain stays unbroken
    get_brain = _make_brain_getter()
    get_flow = _make_flow_getter(gate, get_brain)
    voice = _make_voice()
    get_listener = _make_listener(audit, get_flow=get_flow, voice=voice)
    get_hearer = _make_hearer(audit)
    print("J.A.R.V.I.S dev console. Type help for commands.")
    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            break
        output = handle_line(gate, audit, line, get_brain, get_flow, voice, get_listener, get_hearer)
        if output is None:
            break
        if output:
            print(output)
    voice.finish()


if __name__ == "__main__":
    main()