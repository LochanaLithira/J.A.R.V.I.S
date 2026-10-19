import threading

import httpx
import numpy as np
import pytest

from backend.voice import keepalive
from backend.voice.keepalive import KeepAlive
from backend.voice.listen import CommandRecorder, GroqTranscriber, SpeechInput
from backend.voice.speak import FishSpeaker, normalize_wav
from backend.voice.triggers import CHIME, CHUNK, DONE_CHIME


def test_only_https_hosts():
    with pytest.raises(ValueError):
        KeepAlive("http://api.groq.com")


def test_fixed_hosts():
    assert keepalive.GROQ.base_url == "https://api.groq.com"
    assert keepalive.FISH.base_url == "https://api.fish.audio"


def test_one_shared_client_that_never_follows_redirects():
    k = KeepAlive("https://example.test")
    try:
        c = k.client()
        assert c is k.client() and c.follow_redirects is False
    finally:
        k.close()


def test_warm_up_sends_no_key_and_no_data():
    seen = []
    done = threading.Event()

    def handler(request):
        seen.append((request.method, str(request.url), dict(request.headers), request.read()))
        done.set()
        return httpx.Response(404)

    k = KeepAlive("https://api.groq.com")
    k._client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    try:
        k.warm()
        assert done.wait(3)
    finally:
        k.close()
    method, url, headers, body = seen[0]
    assert method == "HEAD" and url == "https://api.groq.com" and body == b""
    assert not any(h.lower() == "authorization" for h in headers)


def test_warm_up_failure_is_silent():
    k = KeepAlive("https://unreachable.test")
    k._client = httpx.Client(transport=httpx.MockTransport(lambda r: (_ for _ in ()).throw(httpx.ConnectError("x"))))
    try:
        k.warm()                                            # must not raise anywhere
    finally:
        k.close()


# ---- repeated replies play from memory ---------------------------------------------------

WAV = normalize_wav(DONE_CHIME)


class Player:
    def __init__(self):
        self.played = []

    def play(self, audio):
        self.played.append(audio)
        return True

    def stop(self):
        pass


def test_repeated_reply_is_not_fetched_again():
    calls = []

    def transport(url, headers, payload, timeout):
        calls.append(payload["text"])
        return 200, WAV

    player = Player()
    s = FishSpeaker(lambda: "fish-key-123456", transport=transport, player=player)
    assert s.speak("Opening notepad.") and s.speak("Opening notepad.") and s.speak("Cancelled.")
    assert calls == ["Opening notepad.", "Cancelled."] and len(player.played) == 3


def test_reply_cache_is_bounded(monkeypatch):
    from backend.voice import speak

    monkeypatch.setattr(speak, "CACHE_REPLIES", 2)
    s = FishSpeaker(lambda: "fish-key-123456", transport=lambda *a: (200, WAV), player=Player())
    for word in ("one", "two", "three"):
        s.speak(word)
    assert list(s._cache) == ["two", "three"]


def test_failed_fetches_are_not_cached():
    s = FishSpeaker(lambda: "fish-key-123456", transport=lambda *a: (503, b""), player=Player())
    assert not s.speak("hello") and s._cache == {}


# ---- the "got it" tone ---------------------------------------------------------------------

class LoudVAD:
    def __call__(self, frame):
        return 0.95 if np.abs(frame.astype(np.int32)).max() > 1000 else 0.02


def slices(audio):
    audio = np.concatenate([audio.astype(np.int16), np.zeros((-len(audio)) % CHUNK, np.int16)])
    return [audio[i:i + CHUNK] for i in range(0, len(audio), CHUNK)]


def test_got_it_tone_plays_before_the_upload():
    order = []

    class Groq:
        def transcribe(self, audio):
            order.append("upload")
            return "open notepad"

    tone = (np.sin(np.arange(16000) / 16000 * 2 * np.pi * 220) * 6000)
    speech = SpeechInput(CommandRecorder(LoudVAD()), Groq(), on_recorded=lambda: order.append("tone"))
    speech.hear(slices(np.concatenate([np.zeros(8000), tone, np.zeros(24000)])))
    assert order == ["tone", "upload"]


def test_no_tone_when_nothing_was_said():
    order = []
    speech = SpeechInput(CommandRecorder(LoudVAD()), None, on_recorded=lambda: order.append("tone"))
    speech.hear(slices(np.zeros(16000 * 6)))
    assert order == []


def test_got_it_tone_is_a_short_clean_wav_and_differs_from_the_wake_chime():
    assert normalize_wav(DONE_CHIME) == DONE_CHIME and DONE_CHIME != CHIME
