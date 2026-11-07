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


# ---- streamed replies: played while they arrive ------------------------------------------------

from contextlib import contextmanager

from backend.voice.speak import WavStream


class FakeStream:
    """Stands in for Fish: hands over the reply in pieces, and records the request."""

    def __init__(self, body=None, status=200, piece=7, error=None, during=None):
        self.body, self.status, self.piece, self.error, self.during = body, status, piece, error, during
        self.calls = []

    @contextmanager
    def __call__(self, url, headers, payload, timeout):
        self.calls.append((url, dict(headers), dict(payload)))
        if self.error:
            raise self.error

        def pieces():
            for i in range(0, len(self.body), self.piece):
                if self.during and i == self.piece * 2:
                    self.during()
                yield self.body[i:i + self.piece]

        yield self.status, (pieces() if self.status == 200 else iter(()))


class FakeOut:
    opened = []

    def __init__(self, channels, rate, bits):
        self.format, self.written, self.finished, self.aborted = (channels, rate, bits), bytearray(), False, False
        FakeOut.opened.append(self)

    def write(self, data):
        self.written += data

    def finish(self):
        self.finished = True

    def abort(self):
        self.aborted = True


@pytest.fixture(autouse=False)
def outs():
    FakeOut.opened = []
    return FakeOut.opened


def streamer(stream, key=KEY, **kw):
    return FishSpeaker(lambda: key, stream=stream, audio_out=FakeOut, **kw)


SOUND = bytes(range(256)) * 400                     # ~1.2 s of 16-bit mono at 44.1 kHz


def test_a_streamed_reply_plays_all_of_its_sound(outs):
    stream = FakeStream(make_wav(SOUND), piece=999)
    assert streamer(stream).speak("Opening Spotify.")
    out, = outs
    assert out.format == (1, 44100, 16) and bytes(out.written) == SOUND and out.finished and not out.aborted


def test_streaming_starts_before_the_reply_has_fully_arrived(outs):
    import time

    body, last_piece, first_play = make_wav(SOUND * 3), [0.0], []

    @contextmanager
    def stream(url, headers, payload, timeout):
        def pieces():                               # arrives faster than it plays, like a good day
            for i in range(0, len(body), 8192):
                time.sleep(0.01)
                last_piece[0] = time.monotonic()
                yield body[i:i + 8192]
        yield 200, pieces()

    class Out(FakeOut):
        def write(self, data):
            first_play.append(time.monotonic())
            super().write(data)

    assert FishSpeaker(lambda: KEY, stream=stream, audio_out=Out).speak("A long reply.")
    assert first_play[0] < last_piece[0]            # the first sound played mid-download


def test_streamed_request_is_the_same_safe_request(outs):
    stream = FakeStream(make_wav(SOUND))
    streamer(stream, voice_id="a" * 32).speak("Hello <sir>")
    (url, headers, payload), = stream.calls
    assert url == speak.API_URL and headers["model"] == FREE_MODEL
    assert payload == {"text": "Hello sir", "format": "wav", "latency": "balanced", "reference_id": "a" * 32}


def test_a_short_reply_still_plays(outs):
    assert streamer(FakeStream(make_wav(b"\x01\x00" * 100))).speak("Sir?")
    assert bytes(outs[0].written) == b"\x01\x00" * 100


@pytest.mark.parametrize("body", [
    b"not audio at all" * 10,
    make_wav(SOUND, audio_format=3),                 # compressed / float
    make_wav(SOUND, rate=96000),
    make_wav(SOUND, bits=12),
    b"RIFF\x00\x00\x00\x00WAVE" + b"junk" + struct.pack("<I", 10_000_000) + b"x" * 100,
], ids=["not-audio", "compressed", "too-fast", "12-bit", "huge-chunk"])
def test_odd_audio_is_refused_and_never_played(outs, body):
    s = streamer(FakeStream(body))
    assert not s.speak("hi") and "playable" in s.last_error
    assert all(not o.written for o in outs)


def test_streamed_errors_are_explained_without_details(outs):
    s = streamer(FakeStream(status=429))
    assert not s.speak("hi") and "rate limiting" in s.last_error and outs == []
    s = streamer(FakeStream(error=httpx.ConnectError("C:\\secret")))
    assert not s.speak("hi") and s.last_error == "I couldn't reach Fish Audio." and outs == []


def test_too_much_streamed_audio_is_cut_off(outs, monkeypatch):
    monkeypatch.setattr(speak, "MAX_AUDIO_BYTES", 50_000)
    s = streamer(FakeStream(make_wav(SOUND * 3), piece=4096))
    assert not s.speak("hi") and "more audio" in s.last_error
    assert all(o.aborted for o in outs)


def test_stop_cuts_a_streamed_reply_off(outs):
    holder = {}
    s = streamer(FakeStream(make_wav(SOUND * 3), piece=4096, during=lambda: holder["s"].stop()))
    holder["s"] = s
    assert not s.speak("A long reply.")
    assert all(o.aborted or not o.written for o in outs)


def test_no_key_means_no_streamed_request(outs):
    stream = FakeStream(make_wav(SOUND))
    s = FishSpeaker(lambda: None, stream=stream, audio_out=FakeOut)
    assert not s.speak("hi") and stream.calls == [] and "No Fish Audio key" in s.last_error


def test_a_speaker_that_cannot_open_is_explained(outs):
    def broken(*a):
        raise OSError("no device")

    s = FishSpeaker(lambda: KEY, stream=FakeStream(make_wav(SOUND)), audio_out=broken)
    assert not s.speak("hi") and s.last_error == "Windows could not play the audio."


def test_nothing_of_a_streamed_reply_is_kept(outs):
    s = streamer(FakeStream(make_wav(SOUND)))
    s.speak("Opening Spotify.")
    assert not any(isinstance(v, (bytes, bytearray)) and v for v in vars(s).values())


def test_the_app_streams_by_default():
    assert FishSpeaker(lambda: KEY)._stream is speak._httpx_stream


# ---- several sentences at once ------------------------------------------------------------------

import time

from backend.voice.speak import split_sentences


@pytest.mark.parametrize("text, parts", [
    ("Opening Spotify, sir.", ["Opening Spotify, sir."]),
    ("Battery at sixty percent. The processor is at twelve percent.",
     ["Battery at sixty percent.", "The processor is at twelve percent."]),
    ("Yes. Sir? Opening Word now.", ["Yes. Sir? Opening Word now."]),          # tiny bits join the next
    ("One thing here. Two things here. Three things here. Four things here.",
     ["One thing here.", "Two things here.", "Three things here. Four things here."]),
    ("Version 1.5 is ready, sir.", ["Version 1.5 is ready, sir."]),           # no split inside numbers
])
def test_sentences_are_split_sensibly(text, parts):
    assert split_sentences(text) == parts


class MultiStream:
    """Fish for several sentences: each request gets its own WAV, made from the sentence's text."""

    def __init__(self, fail_on=None):
        self.calls, self.fail_on, self.lock = [], fail_on, threading.Lock()

    @staticmethod
    def sound_for(text):
        return (text.encode() * 400)[:len(text) * 400 // 2 * 2]

    @contextmanager
    def __call__(self, url, headers, payload, timeout):
        with self.lock:
            self.calls.append(payload["text"])
        if payload["text"] == self.fail_on:
            yield 503, iter(())
            return
        body = make_wav(self.sound_for(payload["text"]))

        def pieces():
            for i in range(0, len(body), 1000):
                time.sleep(0.02)
                yield body[i:i + 1000]
        yield 200, pieces()


def test_each_sentence_is_requested_and_played_in_order(outs):
    stream = MultiStream()
    text = "Battery at sixty percent. The processor is at twelve percent."
    assert streamer(stream).speak(text)
    first, second = split_sentences(text)
    assert sorted(stream.calls) == sorted([first, second])
    out, = outs                                   # one speaker stream for the whole reply
    assert bytes(out.written) == MultiStream.sound_for(first) + MultiStream.sound_for(second)


def test_a_failing_sentence_stops_the_reply_and_says_why(outs):
    text = "Battery at sixty percent. The processor is at twelve percent."
    s = streamer(MultiStream(fail_on="The processor is at twelve percent."))
    assert not s.speak(text) and "Fish Audio is having trouble" in s.last_error


def test_fish_speed_is_remembered_as_a_number_only(outs):
    s = streamer(MultiStream())
    before = s._speed
    s.speak("Battery at sixty percent. The processor is at twelve percent.")
    assert isinstance(s._speed, float) and s._speed != before and 0 < s._speed <= 3.0


def test_wav_stream_reads_a_header_split_into_tiny_pieces():
    w, body, sound = WavStream(), make_wav(SOUND[:1000]), bytearray()
    for i in range(len(body)):
        sound += w.feed(body[i:i + 1])
    assert w.format == (1, 44100, 16) and bytes(sound) == SOUND[:1000]


# ---- when a streamed reply may start --------------------------------------------------------------

from backend.voice.speak import ready_to_play


def test_fast_arrival_starts_after_a_short_look():
    assert ready_to_play(buffered=0.7, elapsed=0.6, expected=4.0)
    assert not ready_to_play(buffered=0.1, elapsed=0.05, expected=4.0)       # always a small head start


def test_an_opening_burst_is_not_trusted():
    assert not ready_to_play(buffered=0.4, elapsed=0.05, expected=2.0)       # quick start, may stall next


def test_slow_arrival_waits_for_a_bigger_head_start():
    # sound arriving at 0.4x real time (as measured on a slow day): a 4 s reply needs most of it first
    assert not ready_to_play(buffered=1.0, elapsed=2.5, expected=4.0)
    assert ready_to_play(buffered=3.0, elapsed=7.5, expected=4.0)


def test_a_slow_start_never_plays_a_gap_on_the_measured_profile():
    """Simulate the slow day measured live: 3.9 s of sound arriving over 10 s after a 0.8 s wait."""
    rate, total, step = 0.39, 3.9, 0.05
    t, start = 0.0, None
    while start is None:
        t += step
        if ready_to_play(min(total, rate * t), t, expected=72 * 0.075):
            start = t
    for k in range(1, int(total / step)):          # while playing, sound must already be there
        played = k * step
        assert rate * (start + played) >= min(total, played) - 1e-9
