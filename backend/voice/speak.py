"""Voice output through Fish Audio's FREE text-to-speech model.

Rules this file enforces:
- Only the free model name is ever sent. It is fixed in this file. If the model header were
  missing or wrong, Fish could bill the request at the paid rate, so no setting can change it.
- Only the cleaned reply text leaves the laptop (control and markup characters removed,
  300 characters at most). Fish may keep free-tier requests, so a reply marked private (note
  text, file names...) is never sent: spoken_text() swaps it for a fixed line. Callers must pass
  every gate Result through spoken_text() before calling speak().
- The API key is read from the Windows Credential Manager for each request and is never
  stored on the object, logged, or put in an error message.
- Requests use HTTPS with timeouts, never follow redirects (so the key cannot be forwarded
  elsewhere), and cap the size of the reply. The audio must be plain PCM WAV, and its header is
  rebuilt from the bytes that arrived (Fish's streamed header holds fake sizes that crash PlaySound).
- Replies are streamed: they start playing while Fish is still making them.
  * Each sentence (up to 3) is requested at the same time, so later sentences are being made
    while the first one plays. Measured: a two-sentence reply fully arrives ~40% sooner.
  * A sentence starts once the arrival speed shows it can play to its end without a gap
    (ready_to_play). If Fish falls behind, J.A.R.V.I.S waits between sentences, which sounds
    like a natural pause, rather than stopping mid-word.
  * Fish's speed swings a lot from hour to hour, so the speed of the last reply is remembered
    (one number, never any audio) to pick a safe head start before the new one can be measured.
  * Sound plays through sounddevice (already used for the microphone) and is checked as
    strictly as a whole file.
- No audio is kept anywhere: nothing is written to disk and no reply is cached in memory. Each
  piece of sound is dropped once it has been played. Requests reuse one kept-alive connection
  (keepalive.py), which saves over a second each.
- Failure is quiet and explained: speak() returns False and last_error says why in plain words.
  If Fish is unreachable, nothing else breaks and the text reply still shows.
"""
from __future__ import annotations

import queue
import re
import struct
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Protocol

API_URL = "https://api.fish.audio/v1/tts"
FREE_MODEL = "s2.1-pro-free"          # never change this to a paid model name
MAX_SPOKEN_CHARS = 300
MAX_AUDIO_BYTES = 8_000_000
REQUEST_TIMEOUT_SECONDS = 15.0
STREAM_MIN_SECONDS = 0.3              # always buffer at least this much sound before playing
STREAM_MARGIN_SECONDS = 0.4           # extra head start, as the arrival speed wobbles
STREAM_JUDGE_SECONDS = 0.6            # watch a sentence arrive this long before trusting its speed
SECONDS_PER_CHAR = 0.075              # a generous guess of how long speech lasts, per character
MAX_SENTENCES = 3                     # requests made at once for one reply
START_SPEED = 0.8                     # assumed speed before any reply has been measured


def ready_to_play(buffered: float, elapsed: float, expected: float, recent_speed: float = START_SPEED) -> bool:
    """Can a sentence start now and play to its end without a gap?

    buffered: seconds of sound received so far; elapsed: seconds since its first sound arrived;
    expected: a generous guess of the sentence's length; recent_speed: how fast the last reply
    arrived (seconds of sound per second). Fish often sends a quick burst and then slows down, so
    until it has been watched for a moment, the last reply's speed is trusted instead."""
    if buffered < STREAM_MIN_SECONDS:
        return False
    if buffered >= expected:
        return True
    rate = recent_speed if elapsed < STREAM_JUDGE_SECONDS else buffered / elapsed
    if rate >= 1.2:
        return True
    still_to_come = expected - buffered
    # While playing, sound keeps arriving at `rate`; the buffer runs out unless it covers the gap.
    return buffered >= max(0.0, 1 - rate) * still_to_come / max(rate, 0.05) + STREAM_MARGIN_SECONDS


def split_sentences(text: str, max_parts: int = MAX_SENTENCES, min_chars: int = 12) -> list[str]:
    """'Done, sir. The volume is at 30 percent.' -> two parts. Tiny ones join the next, and
    anything past max_parts stays with the last part."""
    parts: list[str] = []
    for piece in re.split(r"(?<=[.!?])\s+", text.strip()):
        if not piece:
            continue
        if parts and (len(parts[-1]) < min_chars or len(parts) >= max_parts):
            parts[-1] += " " + piece
        else:
            parts.append(piece)
    return parts

PRIVATE_REPLY = "Done, sir. It's on your screen."

_VOICE_ID_RE = re.compile(r"[0-9a-f]{32}")
_STRIP_CHARS = set("<>[]&")            # markup and control-tag syntax a reply must not use


def spoken_text(result: Any) -> str:
    """Choose what may be said out loud for a gate Result.

    FAIL CLOSED: unless the result is explicitly marked private=False and its message is text,
    the fixed PRIVATE_REPLY is spoken instead, so the real message never leaves the laptop."""
    message = getattr(result, "message", None)
    if getattr(result, "private", True) is not False or not isinstance(message, str):
        return PRIVATE_REPLY
    return message


class Speaker(Protocol):
    last_error: str

    def speak(self, text: str, wait: bool = True) -> bool: ...
    def stop(self) -> None: ...


def clean_for_speech(text: Any) -> str:
    """Make a reply safe and short enough to say out loud. Returns "" if nothing is left."""
    if not isinstance(text, str):
        return ""
    kept = []
    for ch in text:
        if ch in _STRIP_CHARS or not ch.isprintable():
            kept.append(" ")              # markup, control, invisible and direction characters, odd spaces
        else:
            kept.append(ch)
    return " ".join("".join(kept).split())[:MAX_SPOKEN_CHARS].strip()


def normalize_wav(body: Any) -> bytes | None:
    """Return a clean, self-consistent PCM WAV built from Fish's audio, or None if it isn't one.

    Fish streams its audio, so its header carries placeholder sizes (about 4 GB). Windows'
    PlaySound trusts those sizes and reads far past the real audio, which crashes Python.
    So the header is never trusted: the format is checked strictly, and a fresh header is
    written with the sizes of the bytes that actually arrived."""
    if not isinstance(body, bytes) or len(body) < 44 or body[:4] != b"RIFF" or body[8:12] != b"WAVE":
        return None
    fmt = None
    pos = 12
    while pos + 8 <= len(body):
        chunk_id = body[pos:pos + 4]
        size = struct.unpack("<I", body[pos + 4:pos + 8])[0]
        start = pos + 8
        if chunk_id == b"fmt ":
            if size < 16 or start + 16 > len(body):
                return None
            fmt = struct.unpack("<HHIIHH", body[start:start + 16])
        elif chunk_id == b"data":
            if fmt is None or not _plain_pcm(fmt):
                return None
            audio_format, channels, rate, byte_rate, align, bits = fmt
            data = body[start:]
            data = data[:len(data) - len(data) % align]       # whole sample frames only
            if not data:
                return None
            header = (b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE"
                      + b"fmt " + struct.pack("<IHHIIHH", 16, 1, channels, rate, byte_rate, align, bits)
                      + b"data" + struct.pack("<I", len(data)))
            return header + data
        if start + size > len(body):          # a chunk before the audio claims more than arrived
            return None
        pos = start + size + (size & 1)
    return None


def _plain_pcm(fmt: tuple) -> bool:
    """True only for ordinary uncompressed sound: 1 or 2 channels, 8/16/24-bit, 8-48 kHz."""
    audio_format, channels, rate, byte_rate, align, bits = fmt
    return (audio_format == 1 and channels in (1, 2) and bits in (8, 16, 24) and 8000 <= rate <= 48000
            and align == channels * bits // 8 and byte_rate == rate * align)


class WavStream:
    """Reads Fish's WAV as it arrives. The header is checked as strictly as normalize_wav() does
    (its sizes are never trusted); after it, feed() hands back whole frames of sound."""

    MAX_HEADER = 65536                     # anything before the sound must be small

    def __init__(self):
        self._buf = bytearray()
        self.format: tuple[int, int, int] | None = None     # (channels, rate, bits)
        self.byte_rate = 0
        self.align = 0

    def feed(self, chunk: bytes) -> bytes:
        """Add arrived bytes; return the sound that can be played now. ValueError if it isn't plain WAV."""
        self._buf += chunk
        if self.format is None and not self._read_header():
            return b""
        usable = len(self._buf) - len(self._buf) % self.align
        out = bytes(self._buf[:usable])
        del self._buf[:usable]
        return out

    def _read_header(self) -> bool:
        b = self._buf
        if len(b) >= 12 and (b[:4] != b"RIFF" or b[8:12] != b"WAVE"):
            raise ValueError("not a WAV")
        fmt, pos = None, 12
        while len(b) >= 12 and pos + 8 <= len(b):
            chunk_id = bytes(b[pos:pos + 4])
            size = struct.unpack("<I", b[pos + 4:pos + 8])[0]
            start = pos + 8
            if chunk_id == b"fmt ":
                if size < 16:
                    raise ValueError("bad format chunk")
                if start + 16 > len(b):
                    return False
                fmt = struct.unpack("<HHIIHH", bytes(b[start:start + 16]))
            elif chunk_id == b"data":
                if fmt is None or not _plain_pcm(fmt):
                    raise ValueError("not plain sound")
                self.format, self.byte_rate, self.align = (fmt[1], fmt[2], fmt[5]), fmt[3], fmt[4]
                del b[:start]
                return True
            if size > self.MAX_HEADER:
                raise ValueError("chunk too large")
            if start + size > len(b):
                return False
            pos = start + size + (size & 1)
        if len(b) > self.MAX_HEADER:
            raise ValueError("no sound found")
        return False


class SoundDeviceOut:
    """Plays raw sound through the default speakers as it is written."""

    def __init__(self, channels: int, rate: int, bits: int):
        import sounddevice as sd

        dtype = {8: "uint8", 16: "int16", 24: "int24"}[bits]
        self._stream = sd.RawOutputStream(samplerate=rate, channels=channels, dtype=dtype)
        self._stream.start()

    def write(self, data: bytes) -> None:
        self._stream.write(data)

    def finish(self) -> None:
        self._stream.stop()                # waits for the buffered sound to play out
        self._stream.close()

    def abort(self) -> None:
        try:
            self._stream.abort()
            self._stream.close()
        except Exception:
            pass


class _TooLarge(Exception):
    pass


class _Stopped(Exception):
    pass


@contextmanager
def _httpx_stream(url: str, headers: dict[str, str], payload: dict[str, Any],
                  timeout: float) -> Iterator[tuple[int, Iterator[bytes]]]:
    """Like _httpx_transport, but hands over the audio piece by piece as it arrives."""
    import httpx
    from .keepalive import FISH

    limits = httpx.Timeout(timeout, connect=min(5.0, timeout))
    with FISH.client().stream("POST", url, headers=headers, json=payload, timeout=limits,
                              follow_redirects=False) as response:
        yield response.status_code, (response.iter_bytes() if response.status_code == 200 else iter(()))


def _httpx_transport(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float,
                     client_factory: Callable[..., Any] | None = None) -> tuple[int, bytes]:
    """Send one request over the kept-alive Fish connection (a fresh client only when one is
    supplied, for tests). Returns (status code, body); the body is empty unless the status is 200."""
    import httpx  # imported here, so nothing loads until the first time something is spoken

    limits = httpx.Timeout(timeout, connect=min(5.0, timeout))

    def send(client: Any) -> tuple[int, bytes]:
        with client.stream("POST", url, headers=headers, json=payload, timeout=limits,
                           follow_redirects=False) as response:
            if response.status_code != 200:
                return response.status_code, b""
            body = bytearray()
            for chunk in response.iter_bytes():
                body += chunk
                if len(body) > MAX_AUDIO_BYTES:
                    raise _TooLarge()
            return 200, bytes(body)

    if client_factory is not None:
        with client_factory(follow_redirects=False) as client:
            return send(client)
    from .keepalive import FISH

    return send(FISH.client())


class WinsoundPlayer:
    """Plays WAV bytes from memory with Windows' built-in sound API."""

    def play(self, audio: bytes) -> bool:
        try:
            import winsound

            winsound.PlaySound(audio, winsound.SND_MEMORY | winsound.SND_NODEFAULT)   # blocks until done
            return True
        except Exception:
            return False

    def stop(self) -> None:
        try:
            import winsound

            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass


_STATUS_REASONS = {
    401: "Fish Audio rejected the key. Check it with: python -m backend.credentials status fish_api_key",
    402: "Fish Audio says this account has no credit for the request. We only use the free model.",
    403: "Fish Audio refused the request (the key may not be allowed to do this).",
    422: "Fish Audio did not accept the request (check the voice ID in config.py).",
    429: "Fish Audio is rate limiting you. Try again in a moment.",
}


def _explain_status(status: int) -> str:
    if status in _STATUS_REASONS:
        return _STATUS_REASONS[status]
    if status >= 500:
        return f"Fish Audio is having trouble (error {status})."
    return f"Fish Audio returned an unexpected answer ({status})."


class FishSpeaker:
    def __init__(self, get_key: Callable[[], str | None], voice_id: str = "", latency: str = "balanced",
                 timeout_seconds: float = REQUEST_TIMEOUT_SECONDS,
                 transport: Callable[..., tuple[int, bytes]] | None = None, player: Any = None,
                 stream: Callable[..., Any] | None = None, audio_out: Callable[..., Any] | None = None,
                 clock: Callable[[], float] = time.monotonic):
        """Replies are streamed unless a whole-reply `transport` or a `player` is supplied (as the
        older tests do). `stream` and `audio_out` replace the network and the speakers in tests."""
        if voice_id and not _VOICE_ID_RE.fullmatch(voice_id):
            raise ValueError("voice_id must be 32 lowercase hex characters (or empty for the default voice)")
        if latency not in ("normal", "balanced"):
            raise ValueError('latency must be "normal" or "balanced"')
        self._get_key = get_key
        self._voice_id = voice_id
        self._latency = latency
        self._timeout = timeout_seconds
        streaming = stream is not None or (transport is None and player is None)
        self._transport = transport or _httpx_transport
        self._player = player or WinsoundPlayer()
        self._stream = (stream or _httpx_stream) if streaming else None
        self._audio_out = audio_out or SoundDeviceOut
        self._clock = clock
        self._speed = START_SPEED               # how fast Fish's last reply arrived (not audio)
        self._lock = threading.Lock()
        self._ticket = 0                      # bumped by every new reply and by stop()
        self.last_error = ""

    # ---- talking to Fish ---------------------------------------------------
    def _request(self, text: str) -> tuple[dict[str, str], dict[str, Any]] | None:
        """Headers and body for one request, or None (with last_error set) if there is no key."""
        try:
            key = self._get_key()
        except Exception:                     # e.g. a CredentialError: say nothing about why
            key = None
        if not key:
            self.last_error = "No Fish Audio key is stored. Run: python -m backend.credentials set fish_api_key"
            return None
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "model": FREE_MODEL}
        payload: dict[str, Any] = {"text": text, "format": "wav", "latency": self._latency}
        if self._voice_id:
            payload["reference_id"] = self._voice_id
        return headers, payload

    def _fetch(self, text: str) -> bytes | None:
        request = self._request(text)
        if request is None:
            return None
        headers, payload = request
        key = None
        try:
            status, body = self._transport(API_URL, headers, payload, self._timeout)
        except _TooLarge:
            self.last_error = "Fish Audio sent far more audio than expected, so it was ignored."
            return None
        except Exception:                     # no internet, timeout, TLS problem: never show details
            self.last_error = "I couldn't reach Fish Audio."
            return None
        finally:
            headers = None
        if status != 200:
            self.last_error = _explain_status(status)
            return None
        audio = normalize_wav(body)
        if audio is None:
            self.last_error = "Fish Audio did not send playable audio."
        return audio

    # ---- speaking ------------------------------------------------------------
    def stop(self) -> None:
        """Cut off any speech in progress, and drop any reply that is still being fetched."""
        with self._lock:
            self._ticket += 1
        self._player.stop()

    def _current(self, ticket: int) -> bool:
        with self._lock:
            return ticket == self._ticket

    def _pull(self, text: str, feed: "queue.Queue", ticket: int, budget: list[int]) -> None:
        """Fetch one sentence and pass its sound to the player as it arrives. Runs in its own
        thread; every message is (kind, value) with kind "format", "sound", "done" or "error"."""
        request = self._request(text)
        if request is None:
            feed.put(("error", self.last_error))
            return
        headers, payload = request
        wav, first, sound = WavStream(), None, 0
        try:
            with self._stream(API_URL, headers, payload, self._timeout) as (status, pieces):
                headers = None
                if status != 200:
                    feed.put(("error", _explain_status(status)))
                    return
                for piece in pieces:
                    if not self._current(ticket):
                        return
                    with self._lock:
                        budget[0] -= len(piece)
                        too_much = budget[0] < 0
                    if too_much:
                        feed.put(("error", "Fish Audio sent far more audio than expected, so it was ignored."))
                        return
                    had_format = wav.format is not None
                    pcm = wav.feed(piece)
                    if wav.format is not None and not had_format:
                        feed.put(("format", (*wav.format, wav.byte_rate, wav.align)))
                    if pcm:
                        first = first or self._clock()
                        sound += len(pcm)
                        feed.put(("sound", pcm))
            seconds = sound / wav.byte_rate if wav.byte_rate else 0.0
            feed.put(("done", (seconds, self._clock() - first if first else 0.0)))
        except ValueError:
            feed.put(("error", "Fish Audio did not send playable audio."))
        except Exception:                       # no internet, timeout, TLS problem: never show details
            feed.put(("error", "I couldn't reach Fish Audio."))
        finally:
            headers = None

    def _speak_streamed(self, text: str, ticket: int) -> bool:
        """Fetch every sentence at once and play them in order as their sound arrives. stop() or
        a newer reply cuts it off within ~0.1 s. Sound is dropped as soon as it has been handed to
        the speakers: nothing is kept."""
        parts = split_sentences(text)
        if self._request(parts[0]) is None:     # no key: make no request at all
            return False
        budget = [MAX_AUDIO_BYTES]              # shared by all sentences of this reply
        feeds = [queue.Queue() for _ in parts]
        for part, feed in zip(parts, feeds):
            threading.Thread(target=self._pull, args=(part, feed, ticket, budget), daemon=True).start()
        out, fmt, speeds, finished = None, None, [], False

        def play(sound: bytes) -> None:
            step = max(fmt[4], fmt[3] // 10 // fmt[4] * fmt[4])                  # ~0.1 s at a time
            for i in range(0, len(sound), step):
                if not self._current(ticket):
                    raise _Stopped()
                out.write(sound[i:i + step])

        def open_speakers() -> None:
            nonlocal out
            if out is None:
                try:
                    out = self._audio_out(*fmt[:3])
                except Exception:
                    self.last_error = "Windows could not play the audio."
                    raise _Stopped() from None

        def next_message(feed: "queue.Queue") -> tuple[str, Any]:
            deadline = self._timeout + 5
            while True:                         # wake up regularly so stop() is noticed quickly
                if not self._current(ticket):
                    raise _Stopped()
                try:
                    return feed.get(timeout=0.1)
                except queue.Empty:
                    deadline -= 0.1
                    if deadline <= 0:
                        self.last_error = "I couldn't reach Fish Audio."
                        raise _Stopped() from None

        try:
            for part, feed in zip(parts, feeds):
                expected = max(0.6, len(part) * SECONDS_PER_CHAR)
                waiting, first, playing = bytearray(), None, False
                while True:
                    kind, value = next_message(feed)
                    if kind == "error":
                        self.last_error = value
                        raise _Stopped()
                    if kind == "format":
                        if fmt is not None and value != fmt:
                            self.last_error = "Fish Audio did not send playable audio."
                            raise _Stopped()
                        fmt = value
                        continue
                    if kind == "done":
                        speeds.append(value)
                        break
                    if playing:
                        play(value)
                        continue
                    waiting += value
                    first = first or self._clock()
                    if ready_to_play(len(waiting) / fmt[3], self._clock() - first, expected, self._speed):
                        open_speakers()
                        play(bytes(waiting))
                        waiting.clear()
                        playing = True
                if waiting:                     # a sentence shorter than the head start
                    open_speakers()
                    play(bytes(waiting))
                    waiting.clear()
            if out is None:
                self.last_error = "Fish Audio did not send playable audio."
                return False
            out.finish()
            out = None
            finished = True
        except _Stopped:
            return False
        except Exception:                       # a sound device problem: no details
            self.last_error = self.last_error or "Windows could not play the audio."
            return False
        finally:
            if out is not None:
                out.abort()
            if not finished:                    # tell the other sentences' downloads to stop
                with self._lock:
                    if self._ticket == ticket:
                        self._ticket += 1
        sound, took = sum(s for s, _ in speeds), sum(t for _, t in speeds)
        if took > 0.2:                          # remember how fast Fish was (a number, not audio)
            self._speed = 0.5 * self._speed + 0.5 * min(3.0, sound / took)
        return True

    def _play(self, audio: bytes, ticket: int) -> bool:
        with self._lock:
            if ticket != self._ticket:
                return False                  # something newer arrived, or stop() was called
        ok = self._player.play(audio)
        if not ok:
            self.last_error = "Windows could not play the audio."
        return ok

    def speak(self, text: str, wait: bool = True) -> bool:
        """Say the text. Returns True if the audio was fetched and playback started (and, if wait,
        finished). On any failure it returns False and last_error says why."""
        cleaned = clean_for_speech(text)
        if not cleaned:
            self.last_error = "There was nothing to say."
            return False
        self.last_error = ""
        with self._lock:
            self._ticket += 1                 # a new reply replaces the old one
            ticket = self._ticket
        self._player.stop()
        if self._stream is not None:
            if not wait:
                threading.Thread(target=self._speak_streamed, args=(cleaned, ticket), daemon=True).start()
                return True
            return self._speak_streamed(cleaned, ticket)
        audio = self._fetch(cleaned)
        if audio is None:
            return False
        if not wait:
            threading.Thread(target=self._play, args=(audio, ticket), daemon=True).start()
            return True
        return self._play(audio, ticket)


class Voice:
    """Speaks gate results, always through spoken_text(), so private results never leave the laptop.

    By default it speaks in the background, so a console prompt never waits for Fish Audio. With
    wait=True it returns only when speaking has finished, so the wake-up can stay muted for
    exactly as long as J.A.R.V.I.S talks."""

    def __init__(self, speaker: Speaker | None, report: Callable[[str], None] = print,
                 background: bool = True):
        self._speaker = speaker
        self._report = report
        self._background = background
        self._thread: threading.Thread | None = None
        self.enabled = speaker is not None

    def set(self, on: bool) -> str:
        if self._speaker is None:
            return "Voice is not set up."
        self.enabled = on
        if not on:
            self._speaker.stop()
        return "Voice on." if on else "Voice off."

    def say(self, result: Any, wait: bool = False) -> bool:
        """Speak a result. Returns True if it was spoken (or handed to the background), and
        False if voice is off or speaking failed, so the caller can show the reply instead."""
        if not self.enabled or self._speaker is None:
            return False
        speaker, text = self._speaker, spoken_text(result)   # private results never leave the laptop

        def work() -> bool:
            ok = speaker.speak(text, wait=True)
            if not ok:
                self._report(f"(voice: {speaker.last_error})")
            return ok

        if self._background and not wait:
            self._thread = threading.Thread(target=work, daemon=True)
            self._thread.start()
            return True
        return work()

    def finish(self, timeout: float = 30.0) -> None:
        """Let the last reply finish speaking before the program closes."""
        if self._thread is not None:
            self._thread.join(timeout)
