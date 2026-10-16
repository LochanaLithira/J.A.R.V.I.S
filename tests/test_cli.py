import pytest

from backend import registry
from backend.brain import BrainError, BrainReply, Proposal
from backend.cli import handle_line
from backend.credentials import CredentialError
from backend.flow import Flow, FlowError
from backend.gate.audit import AuditLog
from backend.gate.models import Result


@pytest.fixture
def console(tmp_path):
    log = tmp_path / "log.jsonl"
    gate = registry.build_gate(tmp_path / "ws", log)
    return gate, AuditLog(log)


def run(console, line, get_brain=None, get_flow=None):
    return handle_line(*console, line, get_brain, get_flow)


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


# ---- ask: the AI dry run -----------------------------------------------------

class FakeBrain:
    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.seen = reply, error, []

    def propose(self, command, tool_specs):
        self.seen.append((command, [s["name"] for s in tool_specs]))
        if self.error:
            raise self.error
        return self.reply


def test_ask_without_a_brain_says_so(console):
    assert run(console, "ask open notepad") == "NO: The AI is not set up."


def test_ask_shows_the_suggestion_and_runs_nothing(console, tmp_path):
    brain = FakeBrain(BrainReply(proposal=Proposal("create_note", {"title": "x", "text": "y"})))
    out = run(console, "ask make a note called x", lambda: brain)
    assert out == 'AI suggests: create_note {"text": "y", "title": "x"}   (dry run, nothing was run)'
    assert not (tmp_path / "ws" / "notes" / "x.txt").exists()          # the tool did NOT run
    log = tmp_path / "log.jsonl"
    assert not log.exists() or "ran" not in log.read_text()            # and the gate never ran it
    assert brain.seen == [("make a note called x", ["create_note", "read_note", "list_notes", "open_app"])]


def test_ask_shows_a_spoken_reply(console):
    out = run(console, "ask hello", lambda: FakeBrain(BrainReply(text="Which app?")))
    assert out == "AI says: Which app?"


@pytest.mark.parametrize("error", [BrainError("The AI service is busy."), CredentialError("No key.")])
def test_ask_reports_failures_safely(console, error):
    out = run(console, "ask open notepad", lambda: FakeBrain(error=error))
    assert out == f"NO: {error}"


def test_ask_with_an_empty_sentence_is_refused(console):
    class Real:
        def propose(self, command, tool_specs):
            from backend.brain import clean_command
            clean_command(command)

    assert run(console, "ask", lambda: Real()).startswith("NO:")


def test_help_mentions_the_privacy_warning(console):
    assert "Google may review" in run(console, "help")

# ---- say: the real run ---------------------------------------------------------

class FakeFlow:
    def __init__(self, result):
        self.result, self.seen = result, []

    def run(self, sentence):
        self.seen.append(sentence)
        return self.result


def test_say_without_a_flow_says_so(console):
    assert run(console, "say open notepad") == "NO: The AI is not set up."


@pytest.mark.parametrize("result, expected", [
    (Result(True, "Opening notepad."), "OK: Opening notepad."),
    (Result(False, "That action is not allowed."), "NO: That action is not allowed."),
])
def test_say_prints_what_the_flow_returns(console, result, expected):
    assert run(console, "say open notepad", get_flow=lambda: FakeFlow(result)) == expected


def test_say_passes_the_sentence_to_the_flow(console):
    flow = FakeFlow(Result(True, "ok"))
    run(console, "say make a note called x", get_flow=lambda: flow)
    assert flow.seen == ["make a note called x"]


@pytest.mark.parametrize("error", [BrainError("busy"), CredentialError("No key."), FlowError("Tracing is on.")])
def test_say_reports_setup_failures_safely(console, error):
    def broken():
        raise error

    assert run(console, "say open notepad", get_flow=broken) == f"NO: {error}"


def test_help_mentions_say(console):
    assert "say <sentence>" in run(console, "help")


def test_say_end_to_end_with_the_real_flow_and_gate(console, tmp_path):
    gate, _ = console
    brain = FakeBrain(BrainReply(proposal=Proposal("create_note", {"title": "x", "text": "y"})))
    out = run(console, "say make a note", get_flow=lambda: Flow(brain, gate))
    assert out == "OK: Created note x."
    assert (tmp_path / "ws" / "notes" / "x.txt").read_text() == "y"