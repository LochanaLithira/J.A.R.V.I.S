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
- Audio plays from memory with Python's built-in winsound. Nothing is written to disk.
- Failure is quiet and explained: speak() returns False and last_error says why in plain words.
  If Fish is unreachable, nothing else breaks and the text reply still shows.
"""
from __future__ import annotations

import re
import struct
import threading
from typing import Any, Callable, Protocol

API_URL = "https://api.fish.audio/v1/tts"
FREE_MODEL = "s2.1-pro-free"          # never change this to a paid model name
MAX_SPOKEN_CHARS = 300
MAX_AUDIO_BYTES = 8_000_000
REQUEST_TIMEOUT_SECONDS = 15.0

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
            if fmt is None:
                return None
            audio_format, channels, rate, byte_rate, align, bits = fmt
            if (audio_format != 1 or channels not in (1, 2) or bits not in (8, 16, 24)
                    or not 8000 <= rate <= 48000 or align != channels * bits // 8
                    or byte_rate != rate * align):
                return None
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


class _TooLarge(Exception):
    pass


def _httpx_transport(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float,
                     client_factory: Callable[..., Any] | None = None) -> tuple[int, bytes]:
    """Send one request. Returns (status code, body); the body is empty unless the status is 200."""
    import httpx  # imported here, so nothing loads until the first time something is spoken

    factory = client_factory or httpx.Client
    with factory(timeout=httpx.Timeout(timeout, connect=min(5.0, timeout)), follow_redirects=False) as client:
        with client.stream("POST", url, headers=headers, json=payload) as response:
            if response.status_code != 200:
                return response.status_code, b""
            body = bytearray()
            for chunk in response.iter_bytes():
                body += chunk
                if len(body) > MAX_AUDIO_BYTES:
                    raise _TooLarge()
            return 200, bytes(body)


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
                 transport: Callable[..., tuple[int, bytes]] | None = None, player: Any = None):
        if voice_id and not _VOICE_ID_RE.fullmatch(voice_id):
            raise ValueError("voice_id must be 32 lowercase hex characters (or empty for the default voice)")
        if latency not in ("normal", "balanced"):
            raise ValueError('latency must be "normal" or "balanced"')
        self._get_key = get_key
        self._voice_id = voice_id
        self._latency = latency
        self._timeout = timeout_seconds
        self._transport = transport or _httpx_transport
        self._player = player or WinsoundPlayer()
        self._lock = threading.Lock()
        self._ticket = 0                      # bumped by every new reply and by stop()
        self.last_error = ""

    # ---- talking to Fish ---------------------------------------------------
    def _fetch(self, text: str) -> bytes | None:
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
        try:
            status, body = self._transport(API_URL, headers, payload, self._timeout)
        except _TooLarge:
            self.last_error = "Fish Audio sent far more audio than expected, so it was ignored."
            return None
        except Exception:                     # no internet, timeout, TLS problem: never show details
            self.last_error = "I couldn't reach Fish Audio."
            return None
        finally:
            key = None
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
        audio = self._fetch(cleaned)
        if audio is None:
            return False
        if not wait:
            threading.Thread(target=self._play, args=(audio, ticket), daemon=True).start()
            return True
        return self._play(audio, ticket)