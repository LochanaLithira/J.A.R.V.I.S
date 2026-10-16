"""Text-only dev console to exercise the gate by hand (no voice yet).

Run from the repo root:  python -m backend.cli
"""
from __future__ import annotations

import json
from typing import Any, Callable

from . import config
from .brain import Brain, BrainError, GeminiBrain
from .credentials import CredentialError, get_secret
from .flow import Flow, FlowError
from .gate.audit import AuditLog
from .gate.gate import Gate
from .registry import build_gate
from .ui.confirm import popup_confirm

MAX_LINE = 10_000

HELP = """Commands:
  tools                      list the tools and their risk level
  <tool> {json arguments}    run a tool through the gate
  ask <sentence>             DRY RUN: sends your sentence to Google's Gemini AI and shows
                             what it suggests. Nothing is executed.
  say <sentence>             REAL RUN: the AI suggests an action, the gate checks it, and it
                             runs only if allowed. Also sends your sentence to Google.
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


def _say(get_flow: Callable[[], Flow] | None, sentence: str) -> str:
    """Real run: brain -> gate -> tool. The gate still decides everything."""
    if get_flow is None:
        return "NO: The AI is not set up."
    try:
        result = get_flow().run(sentence)
    except (BrainError, CredentialError, FlowError) as exc:
        return f"NO: {exc}"
    return ("OK: " if result.ok else "NO: ") + result.message


def handle_line(gate: Gate, audit: AuditLog, line: str,
                get_brain: Callable[[], Brain] | None = None,
                get_flow: Callable[[], Flow] | None = None) -> str | None:
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

    name, _, rest = line.partition(" ")
    if name == "ask":
        return _ask(gate, get_brain, rest)
    if name == "say":
        return _say(get_flow, rest)
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


def main() -> None:
    gate = build_gate(confirm=popup_confirm)
    audit = gate.audit                      # one shared log, so the chain stays unbroken
    get_brain = _make_brain_getter()
    get_flow = _make_flow_getter(gate, get_brain)
    print("J.A.R.V.I.S dev console. Type help for commands.")
    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            break
        output = handle_line(gate, audit, line, get_brain, get_flow)
        if output is None:
            break
        if output:
            print(output)


if __name__ == "__main__":
    main()