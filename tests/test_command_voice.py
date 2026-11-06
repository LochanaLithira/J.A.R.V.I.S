"""The second voice check: the command itself must be in your voice before anything that changes
things, or shows your own data, can run. The verdict comes from the voice check, never the AI."""
import json
import threading
import time

import numpy as np
import pytest

from backend.brain import BrainReply, Proposal
from backend.flow import Flow
from backend.gate.audit import AuditLog
from backend.gate.gate import Gate, needs_your_voice
from backend.gate.models import Param, Risk, Tool
from backend.voice.listen import Heard, Recording, SpeechInput


def tool(name, risk=Risk.LOW, private=False, ran=None):
    return Tool(name, "t", risk, {}, lambda: (ran.append(name) if ran is not None else None) or "done",
                private_output=private)


def events(log):
    return [json.loads(line) for line in log.path.read_text().splitlines()]


@pytest.fixture
def gate(tmp_path):
    ran = []
    tools = [tool("volume", ran=ran), tool("close_app", Risk.MEDIUM, ran=ran),
             tool("delete_file", Risk.HIGH, ran=ran), tool("read_note", private=True, ran=ran)]
    g = Gate(tools, AuditLog(tmp_path / "log.jsonl"), confirm=lambda n, a: True, max_calls=1000)
    g.ran = ran
    return g


def test_which_tools_need_your_voice():
    assert not needs_your_voice(tool("volume"))
    assert needs_your_voice(tool("close_app", Risk.MEDIUM))
    assert needs_your_voice(tool("delete_file", Risk.HIGH))
    assert needs_your_voice(tool("read_note", private=True))      # shows your own data


@pytest.mark.parametrize("name", ["volume", "close_app", "delete_file", "read_note"])
def test_your_voice_or_no_check_runs_as_before(gate, name):
    assert gate.execute(name, {}).ok
    assert gate.execute(name, {}, voice="you").ok


@pytest.mark.parametrize("voice", ["not you", "unsure", "YOU", "", "anything else"])
@pytest.mark.parametrize("name", ["close_app", "delete_file", "read_note"])
def test_another_voice_cannot_change_things_or_read_your_data(gate, name, voice):
    r = gate.execute(name, {}, voice=voice)
    assert not r.ok and gate.ran == []
    denied = [e for e in events(gate.audit) if e["event"] == "denied"]
    assert denied[-1]["reason"] in ("not_your_voice", "voice_unsure")


def test_another_voice_is_refused_before_the_keypress_window(tmp_path):
    asked = []
    g = Gate([tool("delete_file", Risk.HIGH)], AuditLog(tmp_path / "l.jsonl"),
             confirm=lambda n, a: asked.append(n) or True)
    assert not g.execute("delete_file", {}, voice="not you").ok and asked == []


def test_low_risk_actions_still_work_for_anyone(gate):
    assert gate.execute("volume", {}, voice="not you").ok


def test_the_refusal_says_why_in_plain_words(gate):
    assert "didn't sound like you" in gate.execute("close_app", {}, voice="not you").message
    assert "longer" in gate.execute("close_app", {}, voice="unsure").message


class Brain:
    def __init__(self, tool_name):
        self.tool_name = tool_name

    def propose(self, command, specs, history=()):
        return BrainReply(proposal=Proposal(self.tool_name, {}))


def test_the_flow_hands_the_verdict_to_the_gate(gate):
    flow = Flow(Brain("close_app"), gate)
    assert not flow.run("close notepad", voice="not you").ok
    assert flow.run("close notepad", voice="you").ok
    assert flow.run("close notepad").ok                           # no voice check: as before


# ---- the speech input runs the check on the command, in memory --------------------------

class Recorder:
    max_seconds = 15.0

    def __init__(self, audio):
        self.audio = audio

    def record(self, chunks, **kw):
        return Recording(self.audio, "done", 1.0)


class Transcriber:
    def __init__(self, text="close notepad", delay=0.0):
        self.text, self.delay = text, delay

    def transcribe(self, audio):
        time.sleep(self.delay)
        return self.text


class Audit:
    def __init__(self):
        self.entries = []

    def record(self, event, **fields):
        self.entries.append((event, fields))


AUDIO = np.zeros(16000 * 2, np.int16)


@pytest.mark.parametrize("verdict", ["you", "not you", "unsure"])
def test_the_verdict_comes_back_with_the_words(verdict):
    audit = Audit()
    speech = SpeechInput(Recorder(AUDIO), Transcriber(), audit, voice_judge=lambda a: (verdict, 1.25))
    heard = speech.hear([])
    assert heard == Heard(text="close notepad", reason="done", voice=verdict)
    assert ("command_voice", {"voice": verdict, "voice_score": 1.25}) in audit.entries


def test_without_a_voiceprint_there_is_no_verdict():
    assert SpeechInput(Recorder(AUDIO), Transcriber()).hear([]).voice is None


def test_a_broken_check_never_counts_as_you():
    def boom(audio):
        raise RuntimeError("model crashed")

    assert SpeechInput(Recorder(AUDIO), Transcriber(), voice_judge=boom).hear([]).voice == "unsure"


def test_an_odd_verdict_never_counts_as_you():
    assert SpeechInput(Recorder(AUDIO), Transcriber(), voice_judge=lambda a: ("me!", 9.0)).hear([]).voice == "unsure"


def test_the_check_runs_while_the_words_are_being_turned_into_text():
    started = threading.Event()

    def judge(audio):
        started.set()
        return "you", 1.0

    class SlowTranscriber(Transcriber):
        def transcribe(self, audio):
            assert started.wait(2), "the voice check should already be running"
            return "close notepad"

    assert SpeechInput(Recorder(AUDIO), SlowTranscriber(), voice_judge=judge).hear([]).voice == "you"


def test_the_audit_holds_numbers_only():
    audit = Audit()
    SpeechInput(Recorder(AUDIO), Transcriber(), audit, voice_judge=lambda a: ("you", 2.0)).hear([])
    for event, fields in audit.entries:
        assert all(isinstance(v, (str, int, float, bool)) for v in fields.values())
