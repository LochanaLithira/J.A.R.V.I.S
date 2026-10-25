import struct
import threading

import httpx
import pytest

from backend import registry
from backend.cli import Voice, handle_line
from backend.gate.audit import AuditLog
from backend.gate.models import Result
from backend.voice import speak
from backend.voice.speak import (FREE_MODEL, PRIVATE_REPLY, FishSpeaker, clean_for_speech, normalize_wav,
                                 spoken_text)

KEY = "test-key-1234567890"


def make_wav(data=b"\x01\x00" * 32, riff_size=None, data_size=None, rate=44100, channels=1, bits=16,
             audio_format=1, extra_chunk=b""):
    align = channels * bits // 8
    fmt = struct.pack("<HHIIHH", audio_format, channels, rate, rate * align, align, bits)
    body = (b"WAVE" + b"fmt " + struct.pack("<I", 16) + fmt + extra_chunk
            + b"data" + struct.pack("<I", len(data) if data_size is None else data_size) + data)
    return b"RIFF" + struct.pack("<I", len(body) if riff_size is None else riff_size) + body


WAV = make_wav()


class FakePlayer:
    def __init__(self, ok=True):
        self.ok, self.played, self.stops = ok, [], 0

    def play(self, audio):
        self.played.append(audio)
        return self.ok

    def stop(self):
        self.stops += 1


class FakeTransport:
    def __init__(self, status=200, body=WAV, error=None, during=None):
        self.status, self.body, self.error, self.during, self.calls = status, body, error, during, []

    def __call__(self, url, headers, payload, timeout):
        self.calls.append((url, dict(headers), dict(payload), timeout))
        if self.during:
            self.during()
        if self.error:
            raise self.error
        return self.status, self.body


def make(transport=None, player=None, key=KEY, **kw):
    return FishSpeaker(lambda: key, transport=transport or FakeTransport(), player=player or FakePlayer(), **kw)


# ---- what may be said -----------------------------------------------------------

def test_clean_for_speech_strips_markup_controls_and_clips():
    assert clean_for_speech("hi <b>there</b>\n[pause] &amp;") == "hi b there /b pause amp;"
    assert clean_for_speech("a\u202eb\u200bc\x07d") == "a b c d"
    assert len(clean_for_speech("x" * 1000)) == speak.MAX_SPOKEN_CHARS
    assert clean_for_speech(None) == "" and clean_for_speech("  \n ") == ""


def test_public_result_is_spoken_as_is():
    assert spoken_text(Result(True, "Opening notepad.")) == "Opening notepad."


@pytest.mark.parametrize("result", [
    Result(True, "my secret note", private=True),
    type("R", (), {"message": "no private attribute"})(),
    type("R", (), {"message": "truthy but not False", "private": 0})(),
    type("R", (), {"message": None, "private": False})(),
    "a bare string",
    None,
])
def test_anything_not_explicitly_public_gets_the_fixed_line(result):
    assert spoken_text(result) == PRIVATE_REPLY


def test_gate_marks_note_contents_and_titles_private(tmp_path):
    gate = registry.build_gate(tmp_path / "ws", tmp_path / "log.jsonl")
    created = gate.execute("create_note", {"title": "diary", "text": "dear diary"})
    assert created.ok and not created.private
    assert gate.execute("read_note", {"title": "diary"}).private
    assert gate.execute("list_notes", {}).private
    assert not gate.execute("format_disk", {}).private     # fixed refusal text is safe to say


# ---- the request to Fish --------------------------------------------------------

def test_request_always_uses_the_free_model_and_fixed_url():
    transport = FakeTransport()
    assert make(transport, voice_id="a" * 32).speak("Hello <sir>")
    url, headers, payload, _ = transport.calls[0]
    assert url == "https://api.fish.audio/v1/tts"
    assert headers["model"] == FREE_MODEL == "s2.1-pro-free"
    assert headers["Authorization"] == f"Bearer {KEY}"
    assert payload == {"text": "Hello sir", "format": "wav", "latency": "balanced", "reference_id": "a" * 32}


def test_default_voice_sends_no_reference_id():
    transport = FakeTransport()
    make(transport).speak("hi")
    assert "reference_id" not in transport.calls[0][2]


def test_key_is_never_kept_on_the_object():
    s = make()
    s.speak("hi")
    assert KEY not in repr(vars(s))


@pytest.mark.parametrize("voice_id", ["A" * 32, "a" * 31, "g" * 32, "../" + "a" * 29])
def test_bad_voice_id_refused(voice_id):
    with pytest.raises(ValueError):
        make(voice_id=voice_id)


def test_bad_latency_refused():
    with pytest.raises(ValueError):
        make(latency="fast")


# ---- failures are quiet and explained ----------------------------------------------

def test_empty_text_sends_nothing():
    transport = FakeTransport()
    s = make(transport)
    assert not s.speak("  <> ") and transport.calls == []
    assert s.last_error == "There was nothing to say."


@pytest.mark.parametrize("key", [None, ""])
def test_missing_key_explained_and_nothing_sent(key):
    transport = FakeTransport()
    s = make(transport, key=key)
    assert not s.speak("hi") and transport.calls == []
    assert "fish_api_key" in s.last_error


def test_key_lookup_error_hides_details():
    def boom():
        raise RuntimeError("vault path C:\\secret")

    s = FishSpeaker(boom, transport=FakeTransport(), player=FakePlayer())
    assert not s.speak("hi") and "secret" not in s.last_error


def test_network_error_hides_details():
    s = make(FakeTransport(error=OSError("TLS failure at 10.0.0.1")))
    assert not s.speak("hi")
    assert s.last_error == "I couldn't reach Fish Audio."


def test_oversized_audio_ignored():
    s = make(FakeTransport(error=speak._TooLarge()))
    assert not s.speak("hi") and "more audio" in s.last_error


@pytest.mark.parametrize("status, words", [(401, "key"), (402, "free model"), (429, "rate"),
                                           (503, "error 503"), (418, "418")])
def test_http_errors_explained(status, words):
    player = FakePlayer()
    s = make(FakeTransport(status=status, body=b""), player)
    assert not s.speak("hi") and words in s.last_error and player.played == []


@pytest.mark.parametrize("body", [b"", b"not audio" * 10, b"RIFF" + b"\x00" * 4 + b"AVI " + b"\x00" * 64, WAV[:40]])
def test_non_wav_audio_never_played(body):
    player = FakePlayer()
    s = make(FakeTransport(body=body), player)
    assert not s.speak("hi") and player.played == []


# ---- the WAV header is rebuilt, never trusted -----------------------------------------

def test_fish_streamed_header_with_fake_sizes_is_fixed():
    # What Fish really sends: placeholder sizes of about 4 GB, which crashed PlaySound.
    data = b"\x10\x00" * 100
    fixed = normalize_wav(make_wav(data, riff_size=4294967076, data_size=4294967040))
    assert fixed == make_wav(data)
    assert struct.unpack("<I", fixed[4:8])[0] == len(fixed) - 8
    assert struct.unpack("<I", fixed[40:44])[0] == len(data)


def test_streamed_audio_is_played_with_the_fixed_header():
    player = FakePlayer()
    data = b"\x10\x00" * 100
    assert make(FakeTransport(body=make_wav(data, riff_size=0xFFFFFFFF, data_size=0xFFFFFFFF)), player).speak("hi")
    assert player.played == [make_wav(data)]


def test_a_correct_wav_is_unchanged():
    assert normalize_wav(WAV) == WAV


def test_partial_last_frame_is_trimmed():
    fixed = normalize_wav(make_wav(b"\x01\x00\x02", data_size=3))
    assert fixed == make_wav(b"\x01\x00")


def test_extra_chunks_before_the_audio_are_dropped():
    info = b"LIST" + struct.pack("<I", 4) + b"INFO"
    assert normalize_wav(make_wav(extra_chunk=info)) == WAV


@pytest.mark.parametrize("wav", [
    make_wav(audio_format=3),                          # float, not plain PCM
    make_wav(channels=6),
    make_wav(bits=32),
    make_wav(rate=192000),
    make_wav(data=b""),                                # no audio at all
    make_wav(extra_chunk=b"LIST" + struct.pack("<I", 4_000_000) + b"INFO"),   # chunk claims too much
    WAV.replace(b"fmt ", b"junk"),                     # audio with no format chunk
])
def test_unusual_or_broken_audio_refused(wav):
    assert normalize_wav(wav) is None


def test_mismatched_byte_rate_refused():
    wav = bytearray(WAV)
    wav[28:32] = struct.pack("<I", 12345)
    assert normalize_wav(bytes(wav)) is None


def test_playback_failure_explained():
    s = make(player=FakePlayer(ok=False))
    assert not s.speak("hi") and "play" in s.last_error


# ---- interruption ----------------------------------------------------------------

def test_stop_during_fetch_drops_the_reply():
    player = FakePlayer()
    holder = {}
    s = make(FakeTransport(during=lambda: holder["s"].stop()), player)
    holder["s"] = s
    assert not s.speak("hi") and player.played == []


def test_successful_speech_plays_the_audio():
    player = FakePlayer()
    assert make(player=player).speak("hi") and player.played == [WAV]


# ---- the real HTTP transport, against a fake server ---------------------------------

def _factory(handler):
    return lambda **kw: httpx.Client(transport=httpx.MockTransport(handler), **kw)


def test_transport_returns_audio():
    status, body = speak._httpx_transport("https://x/tts", {}, {"text": "hi"}, 5,
                                          _factory(lambda r: httpx.Response(200, content=WAV)))
    assert (status, body) == (200, WAV)


def test_transport_never_follows_redirects():
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(302, headers={"Location": "https://evil.example/steal"})

    status, body = speak._httpx_transport("https://x/tts", {"Authorization": "Bearer k"}, {}, 5, _factory(handler))
    assert status == 302 and body == b"" and seen == ["https://x/tts"]


def test_transport_caps_the_audio_size(monkeypatch):
    monkeypatch.setattr(speak, "MAX_AUDIO_BYTES", 100)
    with pytest.raises(speak._TooLarge):
        speak._httpx_transport("https://x/tts", {}, {}, 5, _factory(lambda r: httpx.Response(200, content=b"x" * 500)))


# ---- the console voice -------------------------------------------------------------

class FakeSpeaker:
    def __init__(self, ok=True):
        self.ok, self.said, self.stops, self.last_error = ok, [], 0, ""

    def speak(self, text, wait=True):
        self.said.append(text)
        if not self.ok:
            self.last_error = "I couldn't reach Fish Audio."
        return self.ok

    def stop(self):
        self.stops += 1


def test_voice_speaks_public_and_hides_private():
    speaker = FakeSpeaker()
    voice = Voice(speaker, background=False)
    voice.say(Result(True, "Opening notepad."))
    voice.say(Result(True, "dear diary", private=True))
    assert speaker.said == ["Opening notepad.", PRIVATE_REPLY]


def test_voice_failure_is_reported_not_raised():
    reports = []
    Voice(FakeSpeaker(ok=False), report=reports.append, background=False).say(Result(True, "hi"))
    assert reports == ["(voice: I couldn't reach Fish Audio.)"]


def test_voice_off_stops_and_stays_quiet():
    speaker = FakeSpeaker()
    voice = Voice(speaker, background=False)
    assert voice.set(False) == "Voice off." and speaker.stops == 1
    voice.say(Result(True, "hi"))
    assert speaker.said == []
    assert voice.set(True) == "Voice on."


def test_voice_without_a_speaker():
    assert Voice(None).set(True) == "Voice is not set up."
    Voice(None).finish()                               # nothing to wait for


def test_finish_waits_for_the_last_reply():
    release = threading.Event()

    class SlowSpeaker(FakeSpeaker):
        def speak(self, text, wait=True):
            release.wait(5)
            return super().speak(text, wait)

    speaker = SlowSpeaker()
    voice = Voice(speaker)
    voice.say(Result(True, "goodbye"))
    threading.Timer(0.05, release.set).start()
    voice.finish(timeout=5)
    assert speaker.said == ["goodbye"]


def test_console_say_speaks_the_result(tmp_path):
    log = tmp_path / "log.jsonl"
    gate = registry.build_gate(tmp_path / "ws", log)
    speaker = FakeSpeaker()

    class Flow:
        def run(self, sentence):
            return Result(True, "secret text", private=True)

    out = handle_line(gate, AuditLog(log), "say read my note", get_flow=lambda: Flow(),
                      voice=Voice(speaker, background=False))
    assert out == "OK: secret text" and len(speaker.said) == 1
    assert "screen" in speaker.said[0] and "secret" not in speaker.said[0]     # a fixed line, never the content


def test_console_voice_commands(tmp_path):
    log = tmp_path / "log.jsonl"
    console = (registry.build_gate(tmp_path / "ws", log), AuditLog(log))
    assert handle_line(*console, "voice off") == "Voice is not set up."
    assert handle_line(*console, "voice off", voice=Voice(FakeSpeaker())) == "Voice off."
