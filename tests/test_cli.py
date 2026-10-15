import pytest

from backend import registry
from backend.cli import handle_line
from backend.gate.audit import AuditLog


@pytest.fixture
def console(tmp_path):
    log = tmp_path / "log.jsonl"
    gate = registry.build_gate(tmp_path / "ws", log)
    return gate, AuditLog(log)


def run(console, line):
    return handle_line(*console, line)


@pytest.mark.parametrize("word", ["quit", "exit"])
def test_quit_returns_none(console, word):
    assert run(console, word) is None


def test_blank_line_prints_nothing(console):
    assert run(console, "   ") == ""


def test_tools_lists_everything(console):
    out = run(console, "tools")
    for name in ("create_note", "read_note", "list_notes", "open_app"):
        assert name in out


def test_note_roundtrip(console):
    assert run(console, 'create_note {"title": "ideas", "text": "hello"}') == "OK: Created note ideas."
    assert run(console, 'read_note {"title": "ideas"}') == "OK: hello"


def test_unknown_tool_refused(console):
    assert run(console, "format_disk {}").startswith("NO:")


def test_bad_json_explained(console):
    assert run(console, "read_note {title: ideas}").startswith("Arguments must be JSON")


def test_json_that_is_not_an_object_is_refused(console):
    assert run(console, "read_note [1, 2]").startswith("NO:")


def test_traversal_refused_and_nothing_written(console, tmp_path):
    assert run(console, 'create_note {"title": "..\\\\evil", "text": "x"}').startswith("NO:")
    created = {p.name for p in tmp_path.rglob("*") if p.is_file()}
    assert created <= {"log.jsonl"}


def test_verify_ok_then_detects_tampering(console, tmp_path):
    run(console, "list_notes")
    assert run(console, "verify") == "Audit log OK"
    log = tmp_path / "log.jsonl"
    log.write_text(log.read_text().replace("proposed", "hacked", 1))
    assert run(console, "verify").startswith("Audit log TAMPERED")


def test_log_command_shows_recent_events(console):
    run(console, "list_notes")
    assert "proposed" in run(console, "log")


def test_overlong_input_refused(console):
    assert run(console, "list_notes " + "x" * 10_001) == "Input too long."