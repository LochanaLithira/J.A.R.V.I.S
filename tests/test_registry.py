import pytest

from backend import registry
from backend.gate.audit import AuditLog
from backend.tools.notes import make_note_tools


def test_registered_tools_are_exactly_the_approved_set(tmp_path):
    gate = registry.build_gate(tmp_path / "ws", tmp_path / "log.jsonl")
    names = {s["name"] for s in gate.tool_specs()}
    assert names == {"create_note", "read_note", "list_notes", "open_app"}


def test_no_high_risk_tools_yet(tmp_path):
    gate = registry.build_gate(tmp_path / "ws", tmp_path / "log.jsonl")
    assert all(s["risk"] != "high" for s in gate.tool_specs())


def test_overridden_paths_are_used(tmp_path):
    ws, log = tmp_path / "ws", tmp_path / "audit" / "log.jsonl"
    gate = registry.build_gate(ws, log)
    assert gate.execute("create_note", {"title": "a", "text": "b"}).ok
    assert (ws / "notes" / "a.txt").read_text() == "b"
    assert AuditLog(log).verify() == (True, None)


def test_duplicate_tool_names_are_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "make_app_tools", lambda allowed: make_note_tools(tmp_path))
    with pytest.raises(ValueError):
        registry.build_gate(tmp_path / "ws", tmp_path / "log.jsonl")