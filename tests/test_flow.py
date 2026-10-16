import ast
import json
from pathlib import Path

import pytest

from backend import flow as flow_module
from backend.brain import BrainError, BrainReply, Proposal
from backend.flow import Flow, FlowError
from backend.gate.audit import AuditLog
from backend.gate.gate import Gate
from backend.gate.models import Param, Risk, Tool
from backend.tools.apps import make_app_tools
from backend.tools.notes import make_note_tools

ALLOWED = {"notepad": r"C:\Windows\System32\notepad.exe"}


class ScriptedBrain:
    """Stands in for Gemini: returns what we tell it to, and records what it was asked."""

    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.calls = reply, error, []

    def propose(self, command, tool_specs):
        self.calls.append((command, [s["name"] for s in tool_specs]))
        if self.error:
            raise self.error
        return self.reply


def action(tool, **args):
    return BrainReply(proposal=Proposal(tool, args))


def make_gate(tmp_path, launcher=None, extra=()):
    tools = [*make_note_tools(tmp_path / "ws"),
             *make_app_tools(ALLOWED, launcher or (lambda *a, **k: None)), *extra]
    return Gate(tools, AuditLog(tmp_path / "log.jsonl"), max_calls=1000)


def events(tmp_path):
    return [json.loads(line)["event"] for line in (tmp_path / "log.jsonl").read_text().splitlines()]


def test_a_suggested_action_runs_through_the_gate(tmp_path):
    launched = []
    gate = make_gate(tmp_path, launcher=lambda *a, **k: launched.append(a))
    result = Flow(ScriptedBrain(action("open_app", name="notepad")), gate).run("open notepad")
    assert (result.ok, result.message) == (True, "Opening notepad.") and len(launched) == 1
    assert events(tmp_path) == ["command", "proposed", "ran"]
    assert gate.audit.verify() == (True, None)


def test_the_brain_is_given_the_tool_list_and_the_cleaned_command(tmp_path):
    brain = ScriptedBrain(action("list_notes"))
    Flow(brain, make_gate(tmp_path)).run("  list   my notes ")
    assert brain.calls == [("list my notes", ["create_note", "read_note", "list_notes", "open_app"])]


def test_a_plain_reply_is_spoken_and_nothing_runs(tmp_path):
    launched = []
    gate = make_gate(tmp_path, launcher=lambda *a, **k: launched.append(a))
    result = Flow(ScriptedBrain(BrainReply(text="Which app?")), gate).run("do it")
    assert (result.ok, result.message) == (True, "Which app?") and launched == []
    assert events(tmp_path) == ["command", "said"]


def test_an_invented_tool_is_refused_by_the_gate_and_logged(tmp_path):
    result = Flow(ScriptedBrain(action("format_disk")), make_gate(tmp_path)).run("wipe it")
    assert not result.ok and events(tmp_path) == ["command", "proposed", "denied"]


def test_bad_arguments_from_the_brain_write_nothing(tmp_path):
    brain = ScriptedBrain(action("create_note", title="..\\evil", text="x"))
    assert not Flow(brain, make_gate(tmp_path)).run("make a note").ok
    created = {p.name for p in tmp_path.rglob("*") if p.is_file()}
    assert created <= {"log.jsonl"}


def test_a_high_risk_tool_suggested_by_the_brain_is_not_run_without_confirmation(tmp_path):
    called = []
    danger = Tool("danger", "test", Risk.HIGH, {"x": Param(str)}, lambda x: called.append(x) or "boom")
    gate = make_gate(tmp_path, extra=[danger])
    assert not Flow(ScriptedBrain(action("danger", x="y")), gate).run("do the thing").ok
    assert called == [] and "denied" in events(tmp_path)


def test_a_brain_error_becomes_a_safe_message_and_nothing_runs(tmp_path):
    launched = []
    gate = make_gate(tmp_path, launcher=lambda *a, **k: launched.append(a))
    result = Flow(ScriptedBrain(error=BrainError("The AI service is busy.")), gate).run("open notepad")
    assert (result.ok, result.message) == (False, "The AI service is busy.") and launched == []
    assert events(tmp_path) == ["command", "brain_error"]


def test_an_unexpected_crash_shows_and_logs_no_details(tmp_path):
    brain = ScriptedBrain(error=RuntimeError("leaked SECRET-KEY-123"))
    result = Flow(brain, make_gate(tmp_path)).run("open notepad")
    assert (result.ok, result.message) == (False, "Something went wrong.")
    assert "SECRET" not in (tmp_path / "log.jsonl").read_text()


def test_a_brain_returning_the_wrong_type_is_refused(tmp_path):
    result = Flow(ScriptedBrain("open notepad"), make_gate(tmp_path)).run("open notepad")
    assert not result.ok and "unusable" in result.message


@pytest.mark.parametrize("command", ["", "   ", None, 5, "x" * 501])
def test_unusable_commands_never_reach_the_brain(tmp_path, command):
    brain = ScriptedBrain(action("list_notes"))
    assert not Flow(brain, make_gate(tmp_path)).run(command).ok
    assert brain.calls == []


def test_ai_calls_are_rate_limited_and_recover(tmp_path):
    now = [0.0]
    brain = ScriptedBrain(action("list_notes"))
    flow = Flow(brain, make_gate(tmp_path), max_ai_calls=2, window_seconds=60, clock=lambda: now[0])
    assert flow.run("a").ok and flow.run("b").ok
    assert not flow.run("c").ok and len(brain.calls) == 2
    now[0] = 61.0
    assert flow.run("d").ok and len(brain.calls) == 3


@pytest.mark.parametrize("name", ["LANGSMITH_TRACING", "LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING", "LANGCHAIN_TRACING_V2"])
def test_flow_refuses_to_start_while_tracing_is_on(tmp_path, monkeypatch, name):
    monkeypatch.setenv(name, "true")
    with pytest.raises(FlowError):
        Flow(ScriptedBrain(), make_gate(tmp_path))


@pytest.mark.parametrize("value", ["", "false", "0"])
def test_flow_starts_when_tracing_is_off(tmp_path, monkeypatch, value):
    monkeypatch.setenv("LANGSMITH_TRACING", value)
    assert Flow(ScriptedBrain(), make_gate(tmp_path)) is not None


def test_one_unbroken_audit_chain_across_many_runs(tmp_path):
    gate = make_gate(tmp_path)
    flow = Flow(ScriptedBrain(action("list_notes")), gate)
    for _ in range(3):
        flow.run("list notes")
    flow._brain = ScriptedBrain(error=BrainError("busy"))
    flow.run("again")
    assert gate.audit.verify() == (True, None)


# ---- the architecture rules, enforced ----------------------------------------

def test_flow_never_imports_tools():
    tree = ast.parse(Path(flow_module.__file__).read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(("." * node.level) + (node.module or ""))
    assert not [m for m in imported if "tools" in m], imported


def test_the_only_way_into_the_guard_node_is_from_the_brain_node(tmp_path):
    drawn = Flow(ScriptedBrain(), make_gate(tmp_path))._graph.get_graph()
    into_guard = {edge.source for edge in drawn.edges if edge.target == "guard"}
    assert into_guard == {"think"}
    assert {n for n in drawn.nodes} == {"__start__", "__end__", "think", "guard"}