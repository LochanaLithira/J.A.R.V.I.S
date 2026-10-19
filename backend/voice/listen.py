"""Speech input: records one spoken command after a wake-up, then turns it into text with Groq's
Whisper (free tier, which resets daily).

Rules this file enforces:
- Audio leaves the laptop only AFTER a wake-up (or the console's `hear` command), and only the
  part where you actually spoke. Silero VAD runs locally to find where your speech starts and
  ends; silence before and after is trimmed, and if you say nothing, nothing is sent at all.
- Hard limits: at most 15 s per command, and 5 s to start speaking.
- The upload goes over HTTPS to one fixed Groq URL, never follows redirects (so the key cannot
  be forwarded elsewhere), has timeouts, and caps the size of the reply. It reuses one kept-alive
  connection (see keepalive.py), which saves over a second per command. The audio is built in
  memory and never written to disk.
- The Groq key is read from the Windows Credential Manager for each request. It is never stored
  on an object, logged, or put in a message.
- Groq does not keep inputs by default. For the strictest setting, turn on Zero Data Retention
  in the Groq console (Settings > Data Controls).
- Failures are quiet and explained in plain words; nothing here can run a tool.

Set the key once:   python -m backend.credentials set stt_api_key
"""
from __future__ import annotations

import io
import json
import re
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .keepalive import GROQ
from .triggers import SAMPLE_RATE, verify_models

API_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
VAD_FRAME = 512                     # 32 ms: the frame size Silero VAD expects at 16 kHz
MAX_TEXT_CHARS = 500
MAX_REPLY_BYTES = 64_000
REQUEST_TIMEOUT_SECONDS = 15.0

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
# Whisper sometimes "hears" these stock phrases in noise. VAD makes it rare; this catches the rest.
_HALLUCINATIONS = {"thanks for watching", "thank you for watching", "subtitles by the amaraorg community",
                   "please subscribe", "like and subscribe"}


class SpeechError(Exception):
    """Speech input failed in an expected way. The message is safe to show."""


# ---------------------------------------------------------------------------
# Finding where speech starts and ends (local, Silero VAD)
# ---------------------------------------------------------------------------

class SileroVAD:
    """Gives the chance (0 to 1) that a 32 ms frame of 16 kHz audio contains speech."""

    def __init__(self, models_dir: Path):
        path = verify_models(models_dir)["silero_vad.onnx"]
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3
        self._session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        x = (np.asarray(frame, dtype=np.float32) / 32768.0)[None, :]
        out, self._h, self._c = self._session.run(None, {"input": x, "sr": self._sr, "h": self._h, "c": self._c})
        return float(out.reshape(-1)[0])


@dataclass(frozen=True)
class Recording:
    audio: np.ndarray | None        # the trimmed speech, or None when there is nothing to send
    reason: str                     # "done", "time limit", "no speech", "too short", "stopped"
    speech_seconds: float = 0.0


class CommandRecorder:
    """Reads microphone slices until you finish one command, and returns just the speech.

    Speech starts when two frames in a row look like speech (chance >= 0.5). It ends after
    `end_silence` seconds that look like silence (chance < 0.35). The first `skip_start` seconds
    are not used to detect a start, because the wake-up chime plays then."""

    def __init__(self, vad: Callable[[np.ndarray], float], max_seconds: float = 15.0,
                 no_speech_seconds: float = 5.0, end_silence: float = 0.8, min_speech: float = 0.25,
                 skip_start: float = 0.2, keep_before: float = 0.3, keep_after: float = 0.2):
        self._vad = vad
        self.max_seconds, self.no_speech_seconds, self.end_silence = max_seconds, no_speech_seconds, end_silence
        self.min_speech, self.skip_start = min_speech, skip_start
        self.keep_before, self.keep_after = keep_before, keep_after

    def record(self, chunks: Iterable[np.ndarray], no_speech_seconds: float | None = None) -> Recording:
        """`no_speech_seconds` overrides the usual wait for you to start (shorter for follow-ups)."""
        no_speech = self.no_speech_seconds if no_speech_seconds is None else min(no_speech_seconds,
                                                                                 self.max_seconds)
        if hasattr(self._vad, "reset"):
            self._vad.reset()
        frame_s = VAD_FRAME / SAMPLE_RATE
        parts: list[np.ndarray] = []
        pending = np.zeros(0, dtype=np.int16)
        n = 0                                   # frames looked at
        start = last_speech = None              # frame numbers
        run = quiet = speech_frames = 0
        reason = "stopped"
        for chunk in chunks:
            chunk = np.asarray(chunk, dtype=np.int16)
            parts.append(chunk)
            pending = np.concatenate([pending, chunk])
            finished = False
            while len(pending) >= VAD_FRAME:
                frame, pending = pending[:VAD_FRAME], pending[VAD_FRAME:]
                p = self._vad(frame)
                t = n * frame_s
                if start is None:
                    run = run + 1 if (p >= 0.5 and t >= self.skip_start) else 0
                    if run >= 2:
                        start, last_speech, speech_frames, quiet = n - 1, n, 2, 0
                    elif t + frame_s >= no_speech:
                        return Recording(None, "no speech")
                else:
                    if p >= 0.35:
                        last_speech, quiet = n, 0
                        speech_frames += 1
                    else:
                        quiet += 1
                        if quiet * frame_s >= self.end_silence:
                            reason, finished = "done", True
                n += 1
                if not finished and n * frame_s >= self.max_seconds:
                    reason, finished = "time limit", True
                if finished:
                    break
            if finished:
                break
        if start is None:
            return Recording(None, "no speech" if reason != "stopped" else "stopped")
        if speech_frames * frame_s < self.min_speech:
            return Recording(None, "too short")
        audio = np.concatenate(parts)
        a = max(0, int((start * frame_s - self.keep_before) * SAMPLE_RATE))
        b = min(len(audio), int(((last_speech + 1) * frame_s + self.keep_after) * SAMPLE_RATE),
                int(self.max_seconds * SAMPLE_RATE))        # the time limit is absolute
        a = min(a, b)
        return Recording(audio[a:b], reason, round(speech_frames * frame_s, 2))


def wav_bytes(audio: np.ndarray) -> bytes:
    """16 kHz mono 16-bit WAV, built in memory."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(np.asarray(audio, dtype="<i2").tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Turning speech into text (Groq Whisper)
# ---------------------------------------------------------------------------

def _httpx_transport(url: str, headers: dict[str, str], data: dict[str, str], audio: bytes,
                     timeout: float, client_factory: Callable[..., Any] | None = None) -> tuple[int, bytes]:
    """Send one upload over the kept-alive Groq connection (a fresh client only when one is
    supplied, for tests). Returns (status code, body); the body is capped in size."""
    import httpx

    files = {"file": ("command.wav", audio, "audio/wav")}
    limits = httpx.Timeout(timeout, connect=min(5.0, timeout))

    def send(client: Any) -> tuple[int, bytes]:
        with client.stream("POST", url, headers=headers, data=data, files=files, timeout=limits,
                           follow_redirects=False) as response:
            body = bytearray()
            for part in response.iter_bytes():
                body += part
                if len(body) > MAX_REPLY_BYTES:
                    raise SpeechError("Groq sent far more text than expected, so it was ignored.")
            return response.status_code, bytes(body)

    if client_factory is not None:
        with client_factory(follow_redirects=False) as client:
            return send(client)
    return send(GROQ.client())


_STATUS_REASONS = {
    400: "Groq didn't accept the audio.",
    401: "Groq rejected the key. Check it with: python -m backend.credentials status stt_api_key",
    403: "Groq refused the request (the key may not be allowed to do this).",
    413: "That command was too long for Groq.",
    429: "Groq's free limit is used up for now. Try again later.",
}


def _explain_status(status: int) -> str:
    if status in _STATUS_REASONS:
        return _STATUS_REASONS[status]
    if status >= 500:
        return f"Groq is having trouble (error {status}). Try again later."
    return f"Groq returned an unexpected answer ({status})."


def clean_text(text: Any) -> str:
    """Tidy Whisper's text, drop stock phrases it invents in noise, and cap the length."""
    if not isinstance(text, str):
        return ""
    text = " ".join(_CONTROL_RE.sub("", text).split())[:MAX_TEXT_CHARS]
    plain = re.sub(r"[^a-z ]", "", text.lower()).strip()
    return "" if plain in _HALLUCINATIONS or not plain else text


class GroqTranscriber:
    def __init__(self, get_key: Callable[[], str | None], model: str = "whisper-large-v3-turbo",
                 language: str = "en", timeout_seconds: float = REQUEST_TIMEOUT_SECONDS,
                 transport: Callable[..., tuple[int, bytes]] | None = None):
        if not re.fullmatch(r"[a-z0-9][a-z0-9.\-]{1,60}", model):
            raise ValueError("model must be a plain model name")
        if not re.fullmatch(r"[a-z]{2}", language):
            raise ValueError("language must be a two-letter code such as 'en'")
        self._get_key = get_key
        self._model, self._language, self._timeout = model, language, timeout_seconds
        self._transport = transport or _httpx_transport

    def transcribe(self, audio: np.ndarray) -> str:
        try:
            key = self._get_key()
        except Exception:                      # e.g. a CredentialError: say nothing about why
            key = None
        if not key:
            raise SpeechError("No Groq key is stored. Run: python -m backend.credentials set stt_api_key")
        headers = {"Authorization": f"Bearer {key}"}
        data = {"model": self._model, "language": self._language, "response_format": "json", "temperature": "0"}
        try:
            status, body = self._transport(API_URL, headers, data, wav_bytes(audio), self._timeout)
        except SpeechError:
            raise
        except Exception:                      # no internet, timeout, TLS problem: never show details
            raise SpeechError("I couldn't reach Groq.") from None
        finally:
            key = headers = None
        if status != 200:
            raise SpeechError(_explain_status(status))
        try:
            reply = json.loads(body)
        except (ValueError, RecursionError):
            raise SpeechError("Groq's answer wasn't understandable.") from None
        if not isinstance(reply, dict):
            raise SpeechError("Groq's answer wasn't understandable.")
        return clean_text(reply.get("text"))


# ---------------------------------------------------------------------------
# One command, start to finish
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Heard:
    text: str = ""                    # what you said ("" if nothing usable)
    reason: str = ""                  # why recording stopped
    error: str = ""                   # a safe message if something failed


class SpeechInput:
    """Record one command, send only the speech, return the text. Every upload is audited
    (when and how many seconds left the laptop), but the words themselves are not logged here."""

    def __init__(self, recorder: CommandRecorder, transcriber: GroqTranscriber, audit: Any = None,
                 on_recorded: Callable[[], None] | None = None):
        self._recorder, self._transcriber, self._audit = recorder, transcriber, audit
        self.on_recorded = on_recorded          # called the moment you stop speaking (the "got it" tone)

    def _record(self, event: str, **fields: Any) -> None:
        if self._audit is not None:
            self._audit.record(event, **fields)

    def hear(self, chunks: Iterable[np.ndarray], no_speech_seconds: float | None = None) -> Heard:
        rec = (self._recorder.record(chunks) if no_speech_seconds is None
               else self._recorder.record(chunks, no_speech_seconds=no_speech_seconds))
        if rec.audio is None:
            self._record("speech_skipped", reason=rec.reason)
            return Heard(reason=rec.reason)
        seconds = round(len(rec.audio) / SAMPLE_RATE, 2)
        if self.on_recorded is not None:
            try:
                self.on_recorded()
            except Exception:
                pass
        try:
            text = self._transcriber.transcribe(rec.audio)
        except SpeechError as exc:
            self._record("speech_sent", seconds=seconds, ok=False)
            return Heard(reason=rec.reason, error=str(exc))
        self._record("speech_sent", seconds=seconds, ok=True)
        return Heard(text=text, reason=rec.reason)
