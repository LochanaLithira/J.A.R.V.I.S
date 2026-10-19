import io
import json
import wave

import httpx
import numpy as np
import pytest

from backend import cli, config, registry
from backend.gate.audit import AuditLog
from backend.gate.models import Result
from backend.voice import listen as ls
from backend.voice.listen import (API_URL, CommandRecorder, GroqTranscriber, Heard, SileroVAD, SpeechError,
                                  SpeechInput, clean_text, wav_bytes)
from backend.voice.triggers import CHUNK, TriggerError

KEY = "gsk_test_key_1234567890"
SR = 16000


# ---- helpers ------------------------------------------------------------------------

def slices(audio):
    audio = np.concatenate([np.asarray(audio, np.int16), np.zeros((-len(audio)) % CHUNK, np.int16)])
    return [audio[i:i + CHUNK] for i in range(0, len(audio), CHUNK)]


def tone(seconds, level=6000):
    t = np.arange(int(SR * seconds)) / SR
    return (np.sin(2 * np.pi * 220 * t) * level).astype(np.int16)


def quiet(seconds):
    return np.zeros(int(SR * seconds), np.int16)


class LoudVAD:
    """Stands in for Silero: loud frames are 'speech'."""

    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1

    def __call__(self, frame):
        return 0.95 if np.abs(frame.astype(np.int32)).max() > 1000 else 0.02


def record(audio, **kw):
    used = []

    def feed():
        for c in slices(audio):
            used.append(c)
            yield c

    rec = CommandRecorder(LoudVAD(), **kw).record(feed())
    return rec, len(used) * CHUNK / SR


# ---- finding the command --------------------------------------------------------------

def test_speech_then_silence_is_trimmed_and_ends():
    rec, consumed = record(np.concatenate([quiet(1.0), tone(1.5), quiet(3.0)]))
    assert rec.reason == "done" and rec.audio is not None
    assert 1.5 + 0.3 <= len(rec.audio) / SR <= 1.5 + 0.3 + 0.2 + 0.1       # speech + a little either side
    assert consumed < 1.0 + 1.5 + 0.8 + 0.2                                   # stopped soon after you did


def test_nothing_said_sends_nothing():
    rec, consumed = record(quiet(10.0), no_speech_seconds=5.0)
    assert rec.audio is None and rec.reason == "no speech"
    assert consumed <= 5.1


def test_long_speech_is_capped():
    rec, consumed = record(tone(20.0), max_seconds=15.0)
    assert rec.reason == "time limit" and len(rec.audio) / SR <= 15.0 and consumed <= 15.1


def test_a_short_blip_is_not_a_command():
    rec, _ = record(np.concatenate([quiet(0.5), tone(0.1), quiet(2.0)]))
    assert rec.audio is None and rec.reason == "too short"


def test_the_chime_at_the_start_is_ignored():
    rec, _ = record(np.concatenate([tone(0.15), quiet(6.0)]))
    assert rec.audio is None and rec.reason == "no speech"


def test_stopping_early():
    rec, _ = record(quiet(1.0))
    assert rec.audio is None and rec.reason == "stopped"
    rec, _ = record(np.concatenate([quiet(0.5), tone(1.0)]))
    assert rec.audio is not None and rec.reason == "stopped"


def test_vad_memory_is_reset_for_each_command():
    vad = LoudVAD()
    r = CommandRecorder(vad)
    r.record(slices(quiet(0.2)))
    r.record(slices(quiet(0.2)))
    assert vad.resets == 2


def test_real_silero_finds_a_spoken_command(models_dir, spoken):
    vad = SileroVAD(models_dir)
    assert max(vad(f) for f in np.split(quiet(1.024), 32)) < 0.2
    rec = CommandRecorder(vad).record(slices(np.concatenate([spoken["command"], quiet(2.0)])))
    assert rec.reason == "done" and rec.audio is not None
    assert 0.5 < len(rec.audio) / SR < len(spoken["command"]) / SR


def test_wav_is_16khz_mono():
    with wave.open(io.BytesIO(wav_bytes(tone(0.5)))) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()) == (16000, 1, 2, 8000)


# ---- Groq ---------------------------------------------------------------------------

class FakeTransport:
    def __init__(self, status=200, body=b'{"text": " Open notepad. "}', error=None):
        self.status, self.body, self.error, self.calls = status, body, error, []

    def __call__(self, url, headers, data, audio, timeout):
        self.calls.append((url, dict(headers), dict(data), audio, timeout))
        if self.error:
            raise self.error
        return self.status, self.body


def groq(transport=None, key=KEY, **kw):
    return GroqTranscriber(lambda: key, transport=transport or FakeTransport(), **kw)


def test_upload_goes_to_groq_with_the_right_fields():
    t = FakeTransport()
    assert groq(t).transcribe(tone(1.0)) == "Open notepad."
    url, headers, data, audio, timeout = t.calls[0]
    assert url == API_URL == "https://api.groq.com/openai/v1/audio/transcriptions"
    assert headers == {"Authorization": f"Bearer {KEY}"}
    assert data == {"model": "whisper-large-v3-turbo", "language": "en", "response_format": "json", "temperature": "0"}
    assert audio[:4] == b"RIFF" and timeout == ls.REQUEST_TIMEOUT_SECONDS


def test_key_is_never_kept_on_the_object():
    g = groq()
    g.transcribe(tone(0.5))
    assert KEY not in repr(vars(g))


@pytest.mark.parametrize("key", [None, ""])
def test_missing_key_sends_nothing(key):
    t = FakeTransport()
    with pytest.raises(SpeechError, match="stt_api_key"):
        groq(t, key=key).transcribe(tone(0.5))
    assert t.calls == []


def test_key_lookup_error_hides_details():
    def boom():
        raise RuntimeError("vault at C:\\secret")

    with pytest.raises(SpeechError) as exc:
        GroqTranscriber(boom, transport=FakeTransport()).transcribe(tone(0.5))
    assert "secret" not in str(exc.value)


@pytest.mark.parametrize("status, words", [(401, "key"), (413, "too long"), (429, "free limit"),
                                           (503, "error 503"), (418, "418")])
def test_http_errors_explained(status, words):
    with pytest.raises(SpeechError, match=words):
        groq(FakeTransport(status=status, body=b"")).transcribe(tone(0.5))


def test_network_error_hides_details():
    with pytest.raises(SpeechError) as exc:
        groq(FakeTransport(error=OSError("TLS failure at 10.9.8.7"))).transcribe(tone(0.5))
    assert str(exc.value) == "I couldn't reach Groq."


@pytest.mark.parametrize("body", [b"not json", b"[1, 2]", b'{"text": ' * 5000], ids=["not-json", "list", "deep"])
def test_strange_answers_refused(body):
    with pytest.raises(SpeechError):
        groq(FakeTransport(body=body)).transcribe(tone(0.5))


@pytest.mark.parametrize("kw", [{"model": "../evil"}, {"model": "Whisper Large"}, {"language": "english"}])
def test_bad_settings_refused(kw):
    with pytest.raises(ValueError):
        groq(**kw)


@pytest.mark.parametrize("raw, clean", [
    ("  Open\nnotepad\x07 ", "Open notepad"),
    ("Thanks for watching!", ""),
    ("Subtitles by the Amara.org community", ""),
    ("...", ""),
    (None, ""),
    ("x" * 900, "x" * 500),
    ("Thank you, Jarvis.", "Thank you, Jarvis."),
])
def test_clean_text(raw, clean):
    assert clean_text(raw) == clean


def _factory(handler):
    return lambda **kw: httpx.Client(transport=httpx.MockTransport(handler), **kw)


def test_real_transport_sends_multipart_and_reads_reply():
    seen = {}

    def handler(request):
        seen["body"] = request.read()
        seen["type"] = request.headers["content-type"]
        return httpx.Response(200, content=b'{"text": "hello"}')

    status, body = ls._httpx_transport(API_URL, {"Authorization": "Bearer k"}, {"model": "m"}, b"RIFFdata", 5,
                                       _factory(handler))
    assert (status, json.loads(body)) == (200, {"text": "hello"})
    assert seen["type"].startswith("multipart/form-data")
    assert b'name="file"; filename="command.wav"' in seen["body"] and b"RIFFdata" in seen["body"]


def test_real_transport_never_follows_redirects():
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(307, headers={"Location": "https://evil.example/steal"})

    status, _ = ls._httpx_transport(API_URL, {"Authorization": "Bearer k"}, {}, b"x", 5, _factory(handler))
    assert status == 307 and seen == [API_URL]


def test_real_transport_caps_the_reply(monkeypatch):
    monkeypatch.setattr(ls, "MAX_REPLY_BYTES", 100)
    with pytest.raises(SpeechError):
        ls._httpx_transport(API_URL, {}, {}, b"x", 5, _factory(lambda r: httpx.Response(200, content=b"x" * 500)))


# ---- one command, start to finish ---------------------------------------------------------

def speech_input(audit, transport=None):
    return SpeechInput(CommandRecorder(LoudVAD()), groq(transport), audit)


def test_heard_text_and_audit_without_words(audit):
    heard = speech_input(audit).hear(slices(np.concatenate([quiet(0.5), tone(1.0), quiet(1.5)])))
    assert heard == Heard(text="Open notepad.", reason="done")
    log = audit.path.read_text()
    assert '"event":"speech_sent"' in log and '"ok":true' in log and "notepad" not in log
    assert audit.verify()[0]


def test_silence_is_never_uploaded(audit):
    t = FakeTransport()
    heard = speech_input(audit, t).hear(slices(quiet(6.0)))
    assert heard == Heard(reason="no speech") and t.calls == []
    assert '"event":"speech_skipped"' in audit.path.read_text()


def test_upload_failure_is_reported(audit):
    heard = speech_input(audit, FakeTransport(status=429, body=b"")).hear(
        slices(np.concatenate([tone(1.0), quiet(1.5)])))
    assert heard.text == "" and "free limit" in heard.error
    assert '"ok":false' in audit.path.read_text()


# ---- console --------------------------------------------------------------------------

@pytest.fixture
def console(tmp_path):
    log = tmp_path / "log.jsonl"
    return registry.build_gate(tmp_path / "ws", log), AuditLog(log)


def test_hear_without_setup(console):
    assert cli.handle_line(*console, "hear") == "NO: Speech input is not set up."


@pytest.mark.parametrize("heard, expected", [
    (Heard(text="open notepad", reason="done"), 'Heard: "open notepad"   (dry run, nothing was run)'),
    (Heard(reason="no speech"), "I didn't hear anything."),
    (Heard(reason="too short"), "That was too short to understand."),
    (Heard(reason="done"), "I couldn't make out any words."),
    (Heard(reason="done", error="Groq's free limit is used up for now. Try again later."),
     "NO: Groq's free limit is used up for now. Try again later."),
])
def test_hear_shows_what_was_heard_and_runs_nothing(console, tmp_path, heard, expected):
    assert cli.handle_line(*console, "hear", get_hearer=lambda: (lambda: heard)) == expected
    assert not (tmp_path / "ws" / "notes").exists() or not any((tmp_path / "ws" / "notes").iterdir())


def test_hear_reports_setup_errors(console):
    def broken():
        raise TriggerError("I couldn't open the microphone.")

    assert cli.handle_line(*console, "hear", get_hearer=broken) == "NO: I couldn't open the microphone."


def test_help_mentions_hear_and_groq(console):
    text = cli.handle_line(*console, "help")
    assert "hear" in text and "Groq" in text


class FakeWake:
    def __init__(self, scores):
        self.scores = iter(scores)

    def feed(self, chunk):
        return next(self.scores, 0.0)


class FakeVoice:
    def __init__(self, triggers_holder):
        self.said, self.holder = [], triggers_holder

    def say(self, result, wait=False):
        self.said.append((result.message, wait, self.holder["t"]._muted))


def test_listen_hears_runs_and_speaks_with_wake_muted(monkeypatch, tmp_path):
    audio = slices(np.zeros(CHUNK * 4))
    drained = []

    class Mic:
        name = "Test Mic"

        def __init__(self, device):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def chunks(self):
            yield from audio
            raise KeyboardInterrupt

        def drain(self):
            drained.append(1)
            return 0

    holder = {}
    real_triggers = cli.Triggers

    def make_triggers(*a, **k):
        holder["t"] = real_triggers(*a, **k)
        return holder["t"]

    class Speech:
        def hear(self, chunks):
            return Heard(text="open notepad", reason="done")

    class Flow:
        def __init__(self):
            self.seen = []

        def run(self, text):
            self.seen.append(text)
            return Result(True, "Opening notepad.")

    flow = Flow()
    monkeypatch.setattr(cli, "WakeWordDetector", lambda d: FakeWake([0.0, 0.9]))
    monkeypatch.setattr(cli, "ClapDetector", lambda *a, **k: None)
    monkeypatch.setattr(cli, "Triggers", make_triggers)
    monkeypatch.setattr(cli, "pick_microphone", lambda name: None)
    monkeypatch.setattr(cli, "Microphone", Mic)
    monkeypatch.setattr(cli, "play_chime", lambda: None)
    monkeypatch.setattr(cli, "_make_speech", lambda audit: Speech())
    lines = []
    voice = FakeVoice(holder)
    listen = cli._make_listener(AuditLog(tmp_path / "log.jsonl"), out=lines.append,
                                get_flow=lambda: flow, voice=voice)()
    with pytest.raises(KeyboardInterrupt):
        listen()
    assert flow.seen == ["open notepad"]
    assert 'Heard: "open notepad"' in lines and "OK: Opening notepad." in lines
    assert voice.said == [("Opening notepad.", True, True)]        # spoken, waited for, wake-up muted
    assert holder["t"]._muted is False and drained == [1]


def test_listen_without_ai_says_so(monkeypatch, tmp_path):
    lines = []
    t = cli.Triggers()
    cli._act("open notepad", None, None, t, lines.append)
    assert lines == ["NO: The AI is not set up."]
