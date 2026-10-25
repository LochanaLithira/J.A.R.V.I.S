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


class VoiceCheck:
    """Was there really a human voice in this audio? Used when the wake word fires, so a fan, a
    door or a knock that happens to score like "hey jarvis" is ignored. Runs on this laptop."""

    def __init__(self, vad: Callable[[np.ndarray], float], min_speech_seconds: float = 0.3,
                 speech_prob: float = 0.5):
        self._vad, self.min_speech, self.speech_prob = vad, min_speech_seconds, speech_prob

    def speech_seconds(self, chunks: Iterable[np.ndarray]) -> float:
        audio = np.concatenate([np.asarray(c, dtype=np.int16) for c in chunks]) if chunks else np.zeros(0, np.int16)
        if hasattr(self._vad, "reset"):
            self._vad.reset()
        frames = 0
        for i in range(0, len(audio) - VAD_FRAME + 1, VAD_FRAME):
            if self._vad(audio[i:i + VAD_FRAME]) >= self.speech_prob:
                frames += 1
        return frames * VAD_FRAME / SAMPLE_RATE

    def __call__(self, chunks: Iterable[np.ndarray]) -> bool:
        return self.speech_seconds(list(chunks)) >= self.min_speech


@dataclass(frozen=True)
class Recording:
    audio: np.ndarray | None        # the trimmed speech, or None when there is nothing to send
    reason: str                     # "done", "time limit", "no speech", "too short", "stopped"
    speech_seconds: float = 0.0


class CommandRecorder:
    """Reads microphone slices until you finish one command, and returns just the speech.

    Speech starts when `start_frames` frames in a row (about 0.1 s) clearly look like speech
    (chance >= `start_prob`), so a short bang or a burst of noise doesn't count. It ends after
    `end_silence` seconds that look like silence (chance < 0.35). Less than `min_speech` seconds of
    speech in total is not a command and is never sent. The first `skip_start` seconds are not
    used to detect a start, because the wake-up chime plays then."""

    def __init__(self, vad: Callable[[np.ndarray], float], max_seconds: float = 15.0,
                 no_speech_seconds: float = 5.0, end_silence: float = 0.8, min_speech: float = 0.25,
                 skip_start: float = 0.2, keep_before: float = 0.3, keep_after: float = 0.2,
                 start_prob: float = 0.5, start_frames: int = 2):
        self.start_prob, self.start_frames = start_prob, max(1, start_frames)
        self._vad = vad
        self.max_seconds, self.no_speech_seconds, self.end_silence = max_seconds, no_speech_seconds, end_silence
        self.min_speech, self.skip_start = min_speech, skip_start
        self.keep_before, self.keep_after = keep_before, keep_after

    def record(self, chunks: Iterable[np.ndarray], no_speech_seconds: float | None = None,
               max_seconds: float | None = None, skip_start: float | None = None) -> Recording:
        """`no_speech_seconds` overrides the usual wait for you to start (conversations, follow-ups),
        `max_seconds` lowers the length limit, and `skip_start` the chime allowance at the start."""
        limit = self.max_seconds if max_seconds is None else max(0.5, min(max_seconds, self.max_seconds))
        no_speech = self.no_speech_seconds if no_speech_seconds is None else no_speech_seconds
        no_speech = min(no_speech, limit)
        skip = self.skip_start if skip_start is None else skip_start
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
                    run = run + 1 if (p >= self.start_prob and t >= skip) else 0
                    if run >= self.start_frames:
                        start, last_speech, speech_frames, quiet = n - run + 1, n, run, 0
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
                if not finished and n * frame_s >= limit:
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
                int(limit * SAMPLE_RATE))                   # the time limit is absolute
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


# Whisper's own signs that it is guessing: very low average confidence, or very repetitive text
# (a classic sign of words invented from noise). These are Whisper's usual cut-offs. Groq currently
# reports no_speech_prob as 0, so that check only takes effect if Groq starts filling it in.
MIN_AVG_LOGPROB = -1.0
MAX_COMPRESSION_RATIO = 2.4


def _segment_is_noise(segment: dict) -> bool:
    def num(key: str, default: float) -> float:
        value = segment.get(key, default)
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else default

    no_speech, logprob, ratio = num("no_speech_prob", 0.0), num("avg_logprob", 0.0), num("compression_ratio", 1.0)
    return logprob < MIN_AVG_LOGPROB or ratio > MAX_COMPRESSION_RATIO or no_speech > 0.9 \
        or (no_speech > 0.6 and logprob < -0.7)


def trusted_text(reply: dict) -> str:
    """The transcript without the parts Whisper was probably inventing. Without segment details
    (a plain answer), the whole text is used as before."""
    segments = reply.get("segments")
    if not isinstance(segments, list) or not segments:
        return reply.get("text") if isinstance(reply.get("text"), str) else ""
    kept = [s.get("text", "") for s in segments if isinstance(s, dict) and not _segment_is_noise(s)]
    return " ".join(t.strip() for t in kept if isinstance(t, str) and t.strip())


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
        data = {"model": self._model, "language": self._language, "response_format": "verbose_json",
                "temperature": "0"}
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
        return clean_text(trusted_text(reply))


# ---------------------------------------------------------------------------
# One command, start to finish
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Heard:
    text: str = ""                    # what you said ("" if nothing usable)
    reason: str = ""                  # why recording stopped
    error: str = ""                   # a safe message if something failed


# Words a sentence doesn't end on: "open the...", "make a note called...", "um...".
_UNFINISHED_ENDINGS = {
    "and", "or", "but", "so", "because", "then", "to", "the", "a", "an", "my", "your", "our", "his", "her",
    "their", "this", "that", "these", "those", "of", "for", "with", "in", "on", "at", "from", "into", "about",
    "called", "named", "um", "uh", "er", "erm", "hmm", "like",
}


def sounds_unfinished(text: str) -> bool:
    """True when the words clearly stop mid-sentence (Whisper often still adds a full stop)."""
    t = (text or "").strip()
    if not t or t.endswith(("?", "!")):
        return False
    if t.endswith(("...", "…", "-", "—", ",")):
        return True
    return re.sub(r"[^a-z']", "", t.split()[-1].lower()) in _UNFINISHED_ENDINGS


def _join(first: str, second: str) -> str:
    return first.rstrip(" .,…-—") + " " + second.strip()


class SpeechInput:
    """Record one command, send only the speech, return the text. Every upload is audited
    (when and how many seconds left the laptop), but the words themselves are not logged here.

    With `unfinished_extra_seconds`, a command that clearly stops mid-sentence ("open the...")
    keeps listening that much longer for the rest, and the two parts are joined."""

    def __init__(self, recorder: CommandRecorder, transcriber: GroqTranscriber, audit: Any = None,
                 on_recorded: Callable[[], None] | None = None, unfinished_extra_seconds: float = 0.0):
        self._recorder, self._transcriber, self._audit = recorder, transcriber, audit
        self.on_recorded = on_recorded          # the "got it" tone: you can stop talking
        self.unfinished_extra = max(0.0, unfinished_extra_seconds)

    def _record(self, event: str, **fields: Any) -> None:
        if self._audit is not None:
            self._audit.record(event, **fields)

    def _tone(self) -> None:
        if self.on_recorded is not None:
            try:
                self.on_recorded()
            except Exception:
                pass

    def _upload(self, rec: Recording) -> tuple[str, str]:
        """(text, error). Each upload is audited with its length."""
        seconds = round(len(rec.audio) / SAMPLE_RATE, 2)
        try:
            text = self._transcriber.transcribe(rec.audio)
        except SpeechError as exc:
            self._record("speech_sent", seconds=seconds, ok=False)
            return "", str(exc)
        self._record("speech_sent", seconds=seconds, ok=True)
        return text, ""

    def hear(self, chunks: Iterable[np.ndarray], no_speech_seconds: float | None = None) -> Heard:
        stream = iter(chunks)                   # the same stream continues if the sentence wasn't finished
        rec = (self._recorder.record(stream) if no_speech_seconds is None
               else self._recorder.record(stream, no_speech_seconds=no_speech_seconds))
        if rec.audio is None:
            self._record("speech_skipped", reason=rec.reason)
            return Heard(reason=rec.reason)
        early = self.unfinished_extra <= 0      # without the check, the tone comes before the upload
        if early:
            self._tone()
        text, error = self._upload(rec)
        if error:
            return Heard(reason=rec.reason, error=error)
        if not text:                            # only noise, or words Whisper was guessing: run nothing
            if not early:
                self._tone()
            return Heard(reason="noise")
        if not early and rec.reason == "done" and sounds_unfinished(text):
            remaining = self._recorder.max_seconds - len(rec.audio) / SAMPLE_RATE
            if remaining >= 1.0:
                more = self._recorder.record(stream, no_speech_seconds=self.unfinished_extra,
                                             max_seconds=remaining, skip_start=0.0)
                if more.audio is not None:
                    rest, _ = self._upload(more)
                    if rest:
                        text = _join(text, rest)
        if not early:
            self._tone()
        return Heard(text=text, reason=rec.reason)
