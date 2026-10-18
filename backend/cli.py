"""Text-only dev console to exercise the gate by hand (no voice yet).

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
from .voice.speak import FishSpeaker, Speaker, spoken_text

MAX_LINE = 10_000

HELP = """Commands:
  tools                      list the tools and their risk level
  <tool> {json arguments}    run a tool through the gate
  ask <sentence>             DRY RUN: sends your sentence to Google's Gemini AI and shows
                             what it suggests. Nothing is executed.
  say <sentence>             REAL RUN: the AI suggests an action, the gate checks it, and it
                             runs only if allowed. Also sends your sentence to Google.
  voice on | voice off       speak the replies to say commands (on by default)
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

Privacy: on the free Gemini tier, Google may review your sentences and use them to
improve its products. Never put anything private in an ask.
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

    def say(self, result: Result) -> None:
        if not self.enabled or self._speaker is None:
            return
        speaker, text = self._speaker, spoken_text(result)   # private results never leave the laptop

        def work() -> None:
            if not speaker.speak(text, wait=True):
                self._report(f"(voice: {speaker.last_error})")

        if self._background:
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


def handle_line(gate: Gate, audit: AuditLog, line: str,
                get_brain: Callable[[], Brain] | None = None,
                get_flow: Callable[[], Flow] | None = None,
                voice: Voice | None = None) -> str | None:
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


def main() -> None:
    gate = build_gate(confirm=popup_confirm)
    audit = gate.audit                      # one shared log, so the chain stays unbroken
    get_brain = _make_brain_getter()
    get_flow = _make_flow_getter(gate, get_brain)
    voice = _make_voice()
    print("J.A.R.V.I.S dev console. Type help for commands.")
    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            break
        output = handle_line(gate, audit, line, get_brain, get_flow, voice)
        if output is None:
            break
        if output:
            print(output)
    voice.finish()


if __name__ == "__main__":
    main()