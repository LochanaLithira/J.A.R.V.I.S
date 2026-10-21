"""Dev console to exercise the gate, the AI, voice output, wake-up triggers and speech input by hand.

Run from the repo root:  python -m backend.cli
(The everyday background app is:  pythonw -m backend.main)
"""
from __future__ import annotations

import json
from typing import Callable

from . import config
from .brain import Brain, BrainError
from .credentials import CredentialError
from .flow import Flow, FlowError
from .gate.audit import AuditLog
from .gate.gate import Gate
from .gate.models import Result
from .main import (TIMER_DONE_LINE, Status, build_assistant, describe_heard, make_brain_getter, make_flow_getter,
                   make_speech, make_voice)
from .registry import build_gate
from .tools.timers import TimerService
from .ui.confirm import popup_confirm
from .voice.listen import Heard
from .voice.speak import Voice
from .voice.style import Style
from .voice.triggers import Microphone, TriggerError, pick_microphone, play_chime

MAX_LINE = 10_000
_STYLE = Style(config.USER_ADDRESS)          # how spoken replies are phrased (the printout stays literal)

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
  list_files {"folder": "documents"}
  find_files {"name": "budget"}
  set_timer {"seconds": 300, "label": "tea"}
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
        voice.say(Result(result.ok, _STYLE.line(result)))   # private results stay a fixed line
    return ("OK: " if result.ok else "NO: ") + result.message


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
    line = describe_heard(heard)
    return line + ("   (dry run, nothing was run)" if heard.text else "")


def _listen(get_listener: Callable[[], Callable[[], None]] | None) -> str:
    """Run the hands-free loop until Ctrl+C."""
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


def _make_hearer(audit: AuditLog, out: Callable[[str], None] = print) -> Callable[[], Callable[[], Heard]]:
    """Builds the one-shot recorder for the hear command on first use."""

    def get() -> Callable[[], Heard]:
        speech = make_speech(audit)
        device = pick_microphone(config.MIC_DEVICE)

        def hear() -> Heard:
            with Microphone(device) as mic:
                play_chime()
                out("Speak now (after the chime)...")
                return speech.hear(mic.chunks())

        return hear

    return get


def _make_listener(audit: AuditLog, out: Callable[[str], None] = print,
                   get_flow: Callable[[], Flow] | None = None,
                   voice: Voice | None = None,
                   timers: TimerService | None = None) -> Callable[[], Callable[[], None]]:
    """The same loop as the background app, printing to this console instead of a tray icon.
    While it runs, finished timers are announced by the loop (alarm and voice, label printed)."""

    def get() -> Callable[[], None]:
        greeted = []

        def on_status(status: Status, problem: str) -> None:
            if problem:
                out(f"({status.value}: {problem})")
            elif status is Status.LISTENING and not greeted:
                greeted.append(True)
                out("Listening. Say 'hey jarvis' or clap twice, wait for the chime, then say your command. "
                    "Ctrl+C to stop.")

        assistant = build_assistant(audit, get_flow, voice, out=out, on_status=on_status)

        def listen() -> None:                     # runs here until Ctrl+C
            previous = timers.on_fire if timers is not None else None
            if timers is not None:
                timers.on_fire = assistant.announce
            try:
                assistant.run()
            finally:
                if timers is not None:
                    timers.on_fire = previous

        return listen

    return get


def _console_timer_done(voice: Voice | None) -> Callable[[str], None]:
    """Outside `listen`: print the timer (with its label) and say the fixed line."""
    def done(label: str) -> None:
        print(f"\nTIMER: {'Timer done: ' + label if label else 'Your timer is done.'}")
        if voice is not None:
            voice.say(Result(True, _STYLE.timer_line(bool(label))))

    return done


def main() -> None:
    timers = TimerService()
    gate = build_gate(confirm=popup_confirm, timers=timers)
    audit = gate.audit                      # one shared log, so the chain stays unbroken
    get_brain = make_brain_getter()
    get_flow = make_flow_getter(gate, get_brain)
    voice = make_voice()
    timers.on_fire = _console_timer_done(voice)
    get_listener = _make_listener(audit, get_flow=get_flow, voice=voice, timers=timers)
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
