"""Wake-up triggers: "hey jarvis" and a double clap, detected entirely on this laptop.

Rules this file enforces:
- No audio ever leaves the laptop from here. Microphone sound is kept in memory in 80 ms slices,
  checked, and thrown away. Nothing is written to disk.
- Waking up gives no permission. It only means "start listening"; every action still goes
  through the gate, and high-risk ones still need a keypress.
- The model files (three for the wake word, plus Silero VAD used by listen.py) come from the
  openWakeWord v0.5.1 release. Their SHA-256 fingerprints are fixed in this file and checked
  every time they load, so a changed file is refused.
- The models live outside the workspace (config.MODELS_DIR), so no tool can ever write them.
- Light on the laptop: onnxruntime uses one thread, and during silence the models are skipped.

Fetch the models once:   python -m backend.voice.triggers download
Check them:              python -m backend.voice.triggers status
"""
from __future__ import annotations

import hashlib
import math
import os
import queue
import struct
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

SAMPLE_RATE = 16000
CHUNK = 1280                      # 80 ms: one step of the wake-word pipeline
MEL_CONTEXT = 480                 # extra samples the melspectrogram needs from the previous slice
MEL_WINDOW = 76                   # melspectrogram frames per embedding
EMBEDDINGS = 16                   # embeddings the wake model looks at (about 1.3 s)

GATE_LEVEL = 200                  # a slice quieter than this (peak) counts as silence
GATE_AFTER_CHUNKS = 12            # skip the models only after ~1 s of continuous silence

_RELEASE = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/"
MODEL_FILES = {
    "melspectrogram.onnx": "ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f",
    "embedding_model.onnx": "70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f",
    "hey_jarvis_v0.1.onnx": "94a13cfe60075b132f6a472e7e462e8123ee70861bc3fb58434a73712ee0d2cb",
    # Silero VAD v4 (same release): tells speech from silence, so recording stops when you stop.
    "silero_vad.onnx": "a35ebf52fd3ce5f1469b2a36158dba761bc47b973ea3382b3186ca15b1f5af28",
}
MAX_MODEL_BYTES = 5_000_000


class TriggerError(Exception):
    """Something needed for listening is missing or wrong. The message is safe to show."""


# ---------------------------------------------------------------------------
# Model files: fetched once, checked on every load
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_models(models_dir: Path) -> dict[str, Path]:
    """Return the path of each model file, or raise TriggerError if any is missing or changed."""
    found = {}
    for name, digest in MODEL_FILES.items():
        path = Path(models_dir) / name
        if not path.is_file():
            raise TriggerError("The wake-word models are not downloaded. Run: python -m backend.voice.triggers download")
        if path.stat().st_size > MAX_MODEL_BYTES or _sha256(path) != digest:
            raise TriggerError(f"The model file {name} has changed, so it was refused. "
                               "Run: python -m backend.voice.triggers download")
        found[name] = path
    return found


def _fetch(url: str) -> bytes:
    """Download one file over HTTPS, with a timeout and a size cap. GitHub answers release links
    with a redirect to its file host, so redirects are followed; the fingerprint check that
    follows is what makes the content trustworthy."""
    import httpx

    with httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0), follow_redirects=True) as client:
        with client.stream("GET", url) as response:
            if response.status_code != 200 or response.url.scheme != "https":
                raise TriggerError(f"The download failed (status {response.status_code}).")
            body = bytearray()
            for part in response.iter_bytes():
                body += part
                if len(body) > MAX_MODEL_BYTES:
                    raise TriggerError("The download was far bigger than expected, so it was stopped.")
            return bytes(body)


def download_models(models_dir: Path, fetch: Callable[[str], bytes] = _fetch) -> list[str]:
    """Fetch any missing or changed model file. A file is saved only if its fingerprint matches.
    Returns the names that were downloaded."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    fetched = []
    for name, digest in MODEL_FILES.items():
        path = models_dir / name
        if path.is_file() and path.stat().st_size <= MAX_MODEL_BYTES and _sha256(path) == digest:
            continue
        try:
            data = fetch(_RELEASE + name)
        except TriggerError:
            raise
        except Exception:
            raise TriggerError("I couldn't download the models. Check the internet connection.") from None
        if not isinstance(data, bytes) or len(data) > MAX_MODEL_BYTES or hashlib.sha256(data).hexdigest() != digest:
            raise TriggerError(f"The downloaded {name} did not match its fingerprint, so it was not saved.")
        tmp = path.with_suffix(".part")
        tmp.write_bytes(data)
        os.replace(tmp, path)          # all or nothing: never a half-written model
        fetched.append(name)
    return fetched


# ---------------------------------------------------------------------------
# "Hey jarvis": openWakeWord's three models, run directly with onnxruntime
# ---------------------------------------------------------------------------

class WakeWordDetector:
    """Feed it 80 ms slices of 16 kHz audio; it returns a score from 0 to 1 for "hey jarvis".

    Pipeline per slice: sound -> 8 melspectrogram frames -> 1 embedding (from the last 76 frames)
    -> score (from the last 16 embeddings). After about a second of silence the models are
    skipped and the last real background frames are repeated instead, which costs almost nothing."""

    def __init__(self, models_dir: Path, gate_level: int | None = GATE_LEVEL):
        paths = verify_models(models_dir)
        import onnxruntime as ort      # imported here, so nothing heavy loads until listening starts

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3

        def load(name: str):
            return ort.InferenceSession(str(paths[name]), opts, providers=["CPUExecutionProvider"])

        self._mel = load("melspectrogram.onnx")
        self._emb = load("embedding_model.onnx")
        self._wake = load("hey_jarvis_v0.1.onnx")
        self.gate_level = gate_level
        self.skipped = 0               # slices where the models were skipped (for tests and tuning)
        self.reset()

    def reset(self) -> None:
        self._tail = np.zeros(MEL_CONTEXT, dtype=np.float32)
        self._frames: deque = deque([np.ones(32, dtype=np.float32)] * MEL_WINDOW, maxlen=MEL_WINDOW)
        self._embeddings: deque = deque(maxlen=EMBEDDINGS)
        self._quiet = 0
        self._last_frames: np.ndarray | None = None
        self._last_embedding: np.ndarray | None = None

    def feed(self, chunk: np.ndarray) -> float:
        chunk = np.asarray(chunk)
        if chunk.dtype != np.int16 or chunk.shape != (CHUNK,):
            raise ValueError("feed() needs exactly 1280 int16 samples (80 ms at 16 kHz)")
        x = np.concatenate([self._tail, chunk.astype(np.float32)])
        self._tail = x[-MEL_CONTEXT:]

        quiet = self.gate_level is not None and int(np.abs(chunk.astype(np.int32)).max()) < self.gate_level
        self._quiet = self._quiet + 1 if quiet else 0
        if self._quiet > GATE_AFTER_CHUNKS and self._last_embedding is not None:
            self._frames.extend(self._last_frames)
            self._embeddings.append(self._last_embedding)
            self.skipped += 1
            return 0.0

        frames = self._mel.run(None, {"input": x[None, :]})[0].reshape(-1, 32) / 10.0 + 2.0
        self._frames.extend(frames)
        window = np.asarray(self._frames, dtype=np.float32)[None, :, :, None]
        embedding = self._emb.run(None, {"input_1": window})[0].reshape(96)
        self._embeddings.append(embedding)
        self._last_frames, self._last_embedding = frames, embedding
        if len(self._embeddings) < EMBEDDINGS:
            return 0.0
        stack = np.asarray(self._embeddings, dtype=np.float32)[None, :, :]
        return float(self._wake.run(None, {"x.1": stack})[0].reshape(-1)[0])


# ---------------------------------------------------------------------------
# Double clap: no AI, just the shape of the sound
# ---------------------------------------------------------------------------

class ClapDetector:
    """Spots two sharp, short, similarly loud sounds 0.2 to 0.8 s apart, followed by quiet.

    Works on 10 ms frames. Shape matters more than loudness: on this laptop's microphone a clap
    jumps 13-30x above the frame before and fades within about 0.1 s, while speech of the same
    loudness rises only 2-3x and lasts 0.25 s or more. So a clap must start at least 8x louder
    than the quietest of the previous 30 ms (and well above the background) and fade within 0.15 s. A long loud
    sound (speech, music, a door rumbling) cancels any clap in progress, and so does a third
    sharp sound right after the second one (knocking, music beats)."""

    FRAME = 160                        # 10 ms

    def __init__(self, min_level: float = 600, gap: tuple[float, float] = (0.2, 0.8),
                 max_length: float = 0.15, quiet_after: float = 0.3, max_level_ratio: float = 3.0,
                 onset_ratio: float = 8.0, on_clap: Callable[[float], None] | None = None):
        self.min_level = float(min_level)
        self.gap = gap
        self.max_length = max_length
        self.quiet_after = quiet_after
        self.max_level_ratio = max_level_ratio
        self.onset_ratio = onset_ratio
        self.on_clap = on_clap             # told the level of every single clap (for tuning)
        self.reset()

    def reset(self) -> None:
        """Forget everything heard so far (used after J.A.R.V.I.S stops listening or speaking)."""
        self._n = 0                        # frames seen; time = n * 10 ms
        self._recent: deque = deque([0.0] * 3, maxlen=3)   # the last 30 ms, to judge how sudden a start is
        self._background = 50.0
        self._in_sound = False
        self._start = 0.0
        self._peak = 0.0
        self._last_clap: tuple[float, float] | None = None
        self._pending: float | None = None  # time a double clap finished, waiting for quiet

    def feed(self, chunk: np.ndarray) -> bool:
        """Return True once, when a double clap has been confirmed."""
        samples = np.asarray(chunk, dtype=np.float64)
        confirmed = False
        for i in range(0, len(samples) - self.FRAME + 1, self.FRAME):
            level = math.sqrt(float(np.mean(samples[i:i + self.FRAME] ** 2)))
            t = self._n * 0.01
            self._n += 1
            if self._in_sound:
                self._peak = max(self._peak, level)
                if level < self._peak * 0.25:
                    self._in_sound = False
                    if t - self._start <= self.max_length:
                        self._clap(self._start, self._peak)
                    else:                      # a long sound: speech or music, never a clap
                        self._last_clap, self._pending = None, None
            elif (level >= max(self.min_level, self._background * 8)
                  and level >= self.onset_ratio * max(min(self._recent), 1.0)):
                # Compared with the quietest of the last 30 ms, not just the frame before: a real
                # clap can take two or three frames to reach full loudness.
                self._in_sound, self._start, self._peak = True, t, level
                if self._pending is not None:  # a third sharp sound straight after: not a double clap
                    self._pending, self._last_clap = None, None
            else:
                self._background = 0.99 * self._background + 0.01 * level
            self._recent.append(level)
            if self._pending is not None and not self._in_sound and t - self._pending >= self.quiet_after:
                self._pending = None
                confirmed = True
        return confirmed

    def _clap(self, t: float, level: float) -> None:
        if self.on_clap is not None:
            self.on_clap(level)
        if self._last_clap is not None:
            last_t, last_level = self._last_clap
            gap = t - last_t
            ratio = max(level, last_level) / max(min(level, last_level), 1.0)
            if self.gap[0] <= gap <= self.gap[1] and ratio <= self.max_level_ratio:
                self._pending, self._last_clap = t, None
                return
        self._last_clap = (t, level)


# ---------------------------------------------------------------------------
# Deciding when to wake
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WakeEvent:
    source: str                        # "hey jarvis" or "double clap"
    score: float | None = None


class Triggers:
    """Combines both detectors and applies the wake rules: threshold, patience, cooldown, and
    silence while J.A.R.V.I.S itself is talking (so its own voice can never wake it)."""

    MUTE_TAIL_SECONDS = 1.0            # stay deaf this long after speech ends (echo, room reverb)

    def __init__(self, wake: WakeWordDetector | None = None, clap: ClapDetector | None = None,
                 threshold: float = 0.25, patience: int = 1, cooldown_seconds: float = 1.5,
                 audit: Any = None, clock: Callable[[], float] = time.monotonic):
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if not isinstance(patience, int) or patience < 1:
            raise ValueError("patience must be a whole number of at least 1")
        self._wake, self._clap = wake, clap
        self._threshold, self._patience, self._cooldown = threshold, patience, cooldown_seconds
        self._audit, self._clock = audit, clock
        self._above = 0
        self._last_wake = -math.inf
        self._muted = False
        self._muted_until = -math.inf
        self.last_score = 0.0

    def reset(self) -> None:
        """Start fresh after a gap in the audio (while a command was recorded or a reply spoken),
        so sound from before the gap can never be joined to sound after it."""
        for detector in (self._wake, self._clap):
            if detector is not None and hasattr(detector, "reset"):
                detector.reset()
        self._above = 0

    def set_muted(self, muted: bool) -> None:
        """Call with True while J.A.R.V.I.S is speaking, and False when it stops."""
        self._muted = muted
        if not muted:
            self._muted_until = self._clock() + self.MUTE_TAIL_SECONDS

    def process(self, chunk: np.ndarray) -> WakeEvent | None:
        score = self._wake.feed(chunk) if self._wake is not None else 0.0
        clapped = self._clap.feed(chunk) if self._clap is not None else False
        self.last_score = score
        self._above = self._above + 1 if score > self._threshold else 0

        now = self._clock()
        if self._muted or now < self._muted_until:
            self._above = 0
            return None
        if self._above >= self._patience:
            event = WakeEvent("hey jarvis", round(score, 3))
        elif clapped:
            event = WakeEvent("double clap")
        else:
            return None
        if now - self._last_wake < self._cooldown:
            return None
        self._last_wake = now
        self._above = 0
        if self._audit is not None:
            self._audit.record("wake", source=event.source, score=event.score)
        return event


# ---------------------------------------------------------------------------
# Microphone
# ---------------------------------------------------------------------------

def pick_microphone(name: str | None, devices: list[dict] | None = None,
                    hostapis: list[dict] | None = None) -> int | None:
    """None means the Windows default. Otherwise the first input whose name contains `name`,
    preferring Windows' MME audio system, which converts any microphone to 16 kHz for us."""
    if name is None:
        return None
    if devices is None or hostapis is None:
        import sounddevice as sd

        devices, hostapis = list(sd.query_devices()), list(sd.query_hostapis())
    matches = [i for i, d in enumerate(devices)
               if d["max_input_channels"] > 0 and name.lower() in d["name"].lower()]
    if not matches:
        raise TriggerError(f"No microphone name contains {name!r}. Check MIC_DEVICE in config.py.")
    matches.sort(key=lambda i: hostapis[devices[i]["hostapi"]]["name"] != "MME")
    return matches[0]


class Microphone:
    """Streams 80 ms slices from the microphone. The sound callback never blocks: if the
    listener falls behind, the oldest slices are dropped."""

    def __init__(self, device: int | None = None, max_queue: int = 50, stream_factory: Any = None):
        self._device = device
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._stream_factory = stream_factory
        self._stream: Any = None
        self.dropped = 0
        self.name = "microphone"

    def _callback(self, data, frames, time_info, status) -> None:
        chunk = np.array(data[:, 0], dtype=np.int16, copy=True)
        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self.dropped += 1
            self._queue.put_nowait(chunk)

    def __enter__(self) -> "Microphone":
        try:
            if self._stream_factory is None:
                import sounddevice as sd

                self.name = sd.query_devices(self._device, kind="input")["name"]
                factory = sd.InputStream
            else:
                factory = self._stream_factory
            self._stream = factory(samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=CHUNK,
                                   device=self._device, callback=self._callback)
            self._stream.start()
        except TriggerError:
            raise
        except Exception:
            raise TriggerError("I couldn't open the microphone. Check that it's connected and that "
                               "Windows lets desktop apps use it.") from None
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None

    def drain(self) -> int:
        """Throw away slices that piled up (e.g. J.A.R.V.I.S's own voice). Returns how many."""
        dropped = 0
        while True:
            try:
                self._queue.get_nowait()
                dropped += 1
            except queue.Empty:
                return dropped

    def chunks(self, stop: threading.Event | None = None) -> Iterator[np.ndarray]:
        while stop is None or not stop.is_set():
            try:
                yield self._queue.get(timeout=0.25)
            except queue.Empty:
                continue


# ---------------------------------------------------------------------------
# The listening chime
# ---------------------------------------------------------------------------

def chime_wav(rate: int = 22050) -> bytes:
    """A short rising two-note chime, as WAV bytes made in memory."""
    parts = []
    for freq, seconds in ((880.0, 0.07), (1320.0, 0.10)):
        t = np.arange(int(rate * seconds)) / rate
        tone = np.sin(2 * np.pi * freq * t)
        fade = np.minimum(1.0, np.minimum(t, seconds - t) / 0.008)     # soft edges, no clicks
        parts.append(tone * fade)
    pcm = (np.concatenate(parts) * 0.3 * 32767).astype("<i2").tobytes()
    header = (b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE"
              + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
              + b"data" + struct.pack("<I", len(pcm)))
    return header + pcm


CHIME = chime_wav()


def play_chime(player: Any = None) -> None:
    """Play the chime without waiting for it."""
    if player is None:
        from .speak import WinsoundPlayer

        player = WinsoundPlayer()
    threading.Thread(target=player.play, args=(CHIME,), daemon=True).start()


# ---------------------------------------------------------------------------
# Command line: download / status
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    from .. import config

    argv = sys.argv[1:] if argv is None else argv
    if argv not in (["download"], ["status"]):
        print("usage: python -m backend.voice.triggers [download|status]")
        return 2
    try:
        if argv == ["download"]:
            fetched = download_models(config.MODELS_DIR)
            print(f"Downloaded and checked: {', '.join(fetched)}" if fetched else "Already downloaded and checked.")
        verify_models(config.MODELS_DIR)
        print(f"Wake-word models OK in {config.MODELS_DIR}")
    except TriggerError as exc:
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
