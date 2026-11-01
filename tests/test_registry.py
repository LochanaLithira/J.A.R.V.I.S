import pytest

from backend import registry
from backend.gate.audit import AuditLog
from backend.tools.notes import make_note_tools


FILE_TOOLS = {"list_files", "find_files", "open_file", "open_folder", "move_file", "rename_file", "delete_file"}
TIMER_TOOLS = {"set_timer", "list_timers", "cancel_timer"}


def test_registered_tools_are_exactly_the_approved_set(tmp_path):
    gate = registry.build_gate(tmp_path / "ws", tmp_path / "log.jsonl")
    names = {s["name"] for s in gate.tool_specs()}
    assert names == {"create_note", "read_note", "list_notes", "open_app"} | FILE_TOOLS


def test_timer_tools_only_when_a_timer_service_is_given(tmp_path):
    from backend.tools.timers import TimerService

    service = TimerService(start=False)
    gate = registry.build_gate(tmp_path / "ws", tmp_path / "log.jsonl", timers=service)
    assert TIMER_TOOLS <= {s["name"] for s in gate.tool_specs()}


def test_only_delete_needs_a_keypress(tmp_path):
    gate = registry.build_gate(tmp_path / "ws", tmp_path / "log.jsonl")
    assert {s["name"] for s in gate.tool_specs() if s["risk"] == "high"} == {"delete_file"}


def test_file_tools_only_see_the_configured_folders(tmp_path):
    gate = registry.build_gate(tmp_path / "ws", tmp_path / "log.jsonl")
    spec = next(s for s in gate.tool_specs() if s["name"] == "list_files")
    assert spec["params"]["folder"]["choices"] == ["workspace"]       # the test workspace, nothing real


def test_no_folders_means_no_file_tools(tmp_path):
    gate = registry.build_gate(tmp_path / "ws", tmp_path / "log.jsonl", folders={})
    assert not FILE_TOOLS & {s["name"] for s in gate.tool_specs()}


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