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


def _fetch(url: str, max_bytes: int = MAX_MODEL_BYTES) -> bytes:
    """Download one file over HTTPS, with a timeout and a size cap. GitHub and Hugging Face answer
    with a redirect to their file hosts, so redirects are followed; the fingerprint check that
    follows is what makes the content trustworthy."""
    import httpx

    if not url.startswith("https://"):
        raise TriggerError("Only HTTPS downloads are allowed.")
    with httpx.Client(timeout=httpx.Timeout(60.0, connect=10.0), follow_redirects=True) as client:
        with client.stream("GET", url) as response:
            if response.status_code != 200 or response.url.scheme != "https":
                raise TriggerError(f"The download failed (status {response.status_code}).")
            body = bytearray()
            for part in response.iter_bytes():
                body += part
                if len(body) > max_bytes:
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
    weak: bool = False                 # a weak "hey jarvis" that passed the voice check with room to spare


class Triggers:
    """Combines both detectors and applies the wake rules: threshold, patience, cooldown, and
    silence while J.A.R.V.I.S itself is talking (so its own voice can never wake it).

    With `voice_check`, a "hey jarvis" only counts if the last ~2 seconds really contained a
    human voice; a fan, a door or a knock that happens to score like the wake word is ignored.
    That audio is kept in memory only, and only for those 2 seconds."""

    MUTE_TAIL_SECONDS = 1.0            # stay deaf this long after speech ends (echo, room reverb)
    RECENT_CHUNKS = 30                 # about 2.4 seconds of audio, in memory only, for the voice check
    REJECT_PAUSE_SECONDS = 0.3         # after a rejected wake-up, don't check again for a moment
    TRY_FLOOR = 0.05                   # a "hey jarvis" score from here up is a try (near miss or wake-up)
    TRY_END_CHUNKS = 3                 # this many lower slices in a row (~0.24 s) end a try
    TRIES_NOTED_SECONDS = 15.0         # near misses this recent are noted with the next wake-up

    def __init__(self, wake: WakeWordDetector | None = None, clap: ClapDetector | None = None,
                 threshold: float = 0.25, patience: int = 1, cooldown_seconds: float = 1.5,
                 audit: Any = None, clock: Callable[[], float] = time.monotonic,
                 voice_check: Callable[[list], bool] | None = None,
                 weak_threshold: float | None = None, weak_extra: float = 0.05,
                 second_chance_seconds: float = 0.0, second_chance_threshold: float | None = None):
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if not isinstance(patience, int) or patience < 1:
            raise ValueError("patience must be a whole number of at least 1")
        for name, value in (("weak_threshold", weak_threshold), ("second_chance_threshold", second_chance_threshold)):
            if value is not None and not 0.0 < value < threshold:
                raise ValueError(f"{name} must be between 0 and the threshold")
        if not 0.0 <= weak_extra < 1.0:
            raise ValueError("weak_extra must be between 0 and 1")
        self._wake, self._clap = wake, clap
        self._threshold, self._patience, self._cooldown = threshold, patience, cooldown_seconds
        self._weak, self._weak_extra = weak_threshold, weak_extra
        self._second_seconds, self._second_threshold = second_chance_seconds, second_chance_threshold
        self._audit, self._clock = audit, clock
        self._voice_check = voice_check
        self._recent: deque = deque(maxlen=self.RECENT_CHUNKS)
        self._above = 0
        self._last_wake = -math.inf
        self._last_reject = -math.inf
        self._muted = False
        self._muted_until = -math.inf
        self._clear_try()
        self._tries: deque = deque(maxlen=10)       # recent near misses: (time, score, what stopped it)
        self._second_until = -math.inf              # a second chance is open until then
        self._test: dict | None = None              # the wake-up self-test, while it runs
        self.last_score = 0.0
        self.rejected = 0                  # wake-ups dropped because no voice was heard

    def reset(self) -> None:
        """Start fresh after a gap in the audio (while a command was recorded or a reply spoken),
        so sound from before the gap can never be joined to sound after it."""
        for detector in (self._wake, self._clap):
            if detector is not None and hasattr(detector, "reset"):
                detector.reset()
        self._above = 0
        self._recent.clear()
        self._clear_try()

    def recent(self, seconds: float = 1.6) -> list:
        """The last few seconds of sound (memory only). Used when you say the wake word and your
        command in one breath: the wake detector needs a moment to be sure, and words spoken in
        that moment would otherwise be lost."""
        count = max(0, min(len(self._recent), int(round(seconds * SAMPLE_RATE / CHUNK))))
        return list(self._recent)[-count:] if count else []

    def set_voice_check(self, check: Callable[[list], bool] | None) -> None:
        """Replace the wake-up check (used when your voice check is switched on or off)."""
        self._voice_check = check

    def set_muted(self, muted: bool) -> None:
        """Call with True while J.A.R.V.I.S is speaking, and False when it stops."""
        self._muted = muted
        if not muted:
            self._muted_until = self._clock() + self.MUTE_TAIL_SECONDS

    def process(self, chunk: np.ndarray) -> WakeEvent | None:
        score = self._wake.feed(chunk) if self._wake is not None else 0.0
        clapped = self._clap.feed(chunk) if self._clap is not None else False
        self.last_score = score
        self._recent.append(chunk)              # the last ~2 s, in memory only

        now = self._clock()
        if self._muted or now < self._muted_until:
            self._above = 0
            self._clear_try()
            return None
        ended = self._track_try(score)
        if self._test is not None:              # the self-test only measures: nothing wakes
            if ended is not None:
                peak, _handled, at_peak, at_cross = ended
                # judged on the sound a real wake-up would be judged on: when the score first crossed
                # the line (a strong try), else at its highest score (a weak try)
                self._test_try(peak, at_cross if (peak > self._threshold and at_cross) else at_peak)
            if self._test is not None and (self._test["n"] >= self._test["tries"] or now > self._test["until"]):
                self.stop_test()
            return None
        self._above = self._above + 1 if score > self._threshold else 0
        if self._above >= self._patience:
            self._try_handled = True             # decided now; the end of this try needs no second look
            event = WakeEvent("hey jarvis", round(score, 3))
        elif clapped:
            event = WakeEvent("double clap")
        elif ended is not None and not ended[1]:
            event = self._weak_try(ended[0], now)
            if event is None:
                return None
            # a weak try is decided once the score has fallen again, so it is judged on the sound kept
            # from its highest moment (by then the last 2 s have moved past the start of "hey")
            return self._decide(event, now, ended[2])
        else:
            return None
        return self._decide(event, now)

    # ---- tries: one rise and fall of the "hey jarvis" score -------------------------------
    def _clear_try(self) -> None:
        self._try_peak, self._try_quiet, self._try_handled = 0.0, 0, False
        self._try_audio, self._cross_audio = [], None

    def _track_try(self, score: float) -> tuple[float, bool, list, list | None] | None:
        """Follow the score; when a try ends, return (its highest score, whether it was already
        decided, the sound at its highest score, the sound when it first crossed the line). The
        sound is the same last ~2 s already kept in memory, and it is dropped with the try."""
        if score >= self.TRY_FLOOR:
            if score > self._try_peak:
                self._try_peak, self._try_audio = score, list(self._recent)
            if score > self._threshold and self._cross_audio is None:
                self._cross_audio = list(self._recent)
            self._try_quiet = 0
            return None
        if self._try_peak <= 0.0:
            return None
        self._try_quiet += 1
        if self._try_quiet < self.TRY_END_CHUNKS:
            return None
        ended = (self._try_peak, self._try_handled, self._try_audio, self._cross_audio)
        self._clear_try()
        return ended

    def _weak_needed(self, now: float) -> float | None:
        """The lowest score that may still wake J.A.R.V.I.S, or None when weak wake-ups are off.
        Weak wake-ups need your voice check, which then asks for a higher voice score."""
        if self._weak is None or not getattr(self._voice_check, "has_speaker", False):
            return None
        if self._second_threshold is not None and now < self._second_until:
            return min(self._weak, self._second_threshold)
        return self._weak

    def _weak_try(self, peak: float, now: float) -> WakeEvent | None:
        needed = self._weak_needed(now)
        if needed is not None and peak >= needed:
            return WakeEvent("hey jarvis", round(peak, 3), weak=True)
        self._note_try(now, peak, "low")
        if self._second_threshold is not None and peak >= self._second_threshold:
            self._open_second_chance(now)
        return None

    def _note_try(self, now: float, score: float | None, stopped: str) -> None:
        self._tries.append((now, score, stopped))

    def _open_second_chance(self, now: float) -> None:
        if self._second_seconds > 0:
            self._second_until = now + self._second_seconds

    def _decide(self, event: WakeEvent, now: float, chunks: list | None = None) -> WakeEvent | None:
        if now - self._last_wake < self._cooldown:
            return None
        if event.source == "hey jarvis" and self._voice_check is not None:
            if now - self._last_reject < self.REJECT_PAUSE_SECONDS:
                self._above = 0
                return None
            extra = self._weak_extra if event.weak else 0.0
            sound = list(chunks) if chunks else list(self._recent)
            try:
                voiced = bool(self._voice_check(sound, extra=extra) if extra else self._voice_check(sound))
            except Exception:
                voiced = not event.weak          # a broken check must not make J.A.R.V.I.S deaf (strong only)
            if not voiced:
                self._last_reject, self._above = now, 0
                self.rejected += 1
                reason = getattr(self._voice_check, "failed", None) or "no voice"
                self._note_try(now, event.score, reason)
                self._open_second_chance(now)
                if self._audit is not None:
                    extra_fields = {"weak": True} if event.weak else {}
                    voice_score = getattr(self._voice_check, "score", None)
                    if isinstance(voice_score, float):
                        extra_fields["voice_score"] = round(voice_score, 3)   # a number, never sound
                    self._audit.record("wake_rejected", source=event.source, score=event.score,
                                       reason=reason, **extra_fields)
                return None
        self._last_wake = now
        self._above = 0
        tries = [[round(now - t, 1), s, why] for t, s, why in self._tries if now - t <= self.TRIES_NOTED_SECONDS]
        self._tries.clear()
        if self._audit is not None:
            extra_fields = {"weak": True} if event.weak else {}
            voice_score = getattr(self._voice_check, "score", None) if event.source == "hey jarvis" else None
            if isinstance(voice_score, float):
                extra_fields["voice_score"] = round(voice_score, 3)          # a number, never sound
            if tries and event.source == "hey jarvis":
                extra_fields["tries_before"] = tries   # near misses just before: seconds ago, score, why
            self._audit.record("wake", source=event.source, score=event.score, **extra_fields)
        return event

    # ---- the wake-up self-test ------------------------------------------------------------------
    def start_test(self, on_try: Callable[[dict], None], on_done: Callable[[dict], None],
                   tries: int = 10, seconds: float = 90.0) -> None:
        """Measure your "hey jarvis": each try is scored by every check and reported (numbers only),
        and nothing wakes until the test ends (after `tries` tries or `seconds`)."""
        self._clear_try()
        self._test = {"on_try": on_try, "on_done": on_done, "tries": tries, "until": self._clock() + seconds,
                      "n": 0, "verdicts": []}

    @property
    def testing(self) -> bool:
        return self._test is not None

    def stop_test(self) -> None:
        test, self._test = self._test, None
        if test is None:
            return
        verdicts = test["verdicts"]
        passed = sum(1 for v in verdicts if v.startswith("passed"))
        stops: dict[str, int] = {}
        for v in verdicts:
            if not v.startswith("passed"):
                stops[v] = stops.get(v, 0) + 1
        try:
            test["on_done"]({"count": len(verdicts), "passed": passed, "stops": stops})
        except Exception:
            pass

    def _test_try(self, peak: float, sound: list | None = None) -> None:
        test = self._test
        if test is None:
            return
        check = self._voice_check
        sound = list(sound) if sound else list(self._recent)
        diag = check.diagnose(sound) if hasattr(check, "diagnose") else {}
        weak_needed = self._weak if (self._weak is not None and getattr(check, "has_speaker", False)) else None
        if peak > self._threshold:
            wake_ok, extra = True, 0.0
        elif weak_needed is not None and peak >= weak_needed:
            wake_ok, extra = True, self._weak_extra
        else:
            wake_ok, extra = False, 0.0
        speech, speech_needed = diag.get("speech"), diag.get("speech_needed")
        voice, bar = diag.get("voice"), diag.get("bar")
        voice_needed = round(bar + extra, 3) if bar is not None else None
        if not wake_ok:
            verdict = "hey jarvis score too low"
        elif speech is not None and speech_needed is not None and speech < speech_needed:
            verdict = "no voice heard"
        elif voice_needed is not None and (voice is None or voice < voice_needed):
            verdict = "not your voice"
        else:
            verdict = "passed (weak)" if extra else "passed"
        test["n"] += 1
        test["verdicts"].append(verdict)
        result = {"n": test["n"], "of": test["tries"], "wake": round(peak, 3), "wake_needed": self._threshold,
                  "weak_needed": weak_needed, "speech": speech, "speech_needed": speech_needed,
                  "voice": voice, "voice_needed": voice_needed, "verdict": verdict}
        try:
            test["on_try"](result)
        except Exception:
            pass


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


MAX_MICROPHONES = 16


def list_microphones(devices: list[dict] | None = None, hostapis: list[dict] | None = None) -> list[str]:
    """The names of the microphones you can pick on the screen (at most 16), as Windows' MME audio
    system names them: pick_microphone finds each one again by that name."""
    if devices is None or hostapis is None:
        import sounddevice as sd

        devices, hostapis = list(sd.query_devices()), list(sd.query_hostapis())
    names: list[str] = []
    for d in devices:
        name = str(d.get("name", "")).strip()
        if (d.get("max_input_channels", 0) > 0 and hostapis[d["hostapi"]]["name"] == "MME" and name
                and "sound mapper" not in name.lower() and name.isprintable() and name not in names):
            names.append(name[:200])
    return names[:MAX_MICROPHONES]


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

    def chunks(self, stop: threading.Event | None = None,
               idle_timeout: float | None = None) -> Iterator[np.ndarray]:
        """Yield slices until `stop` is set. With `idle_timeout`, a microphone that goes silent
        at the driver level (typical after the laptop wakes from sleep) raises TriggerError, so
        the caller can reopen it."""
        last = time.monotonic()
        while stop is None or not stop.is_set():
            try:
                chunk = self._queue.get(timeout=0.25)
            except queue.Empty:
                if idle_timeout is not None and time.monotonic() - last > idle_timeout:
                    raise TriggerError("The microphone stopped sending sound.")
                continue
            last = time.monotonic()
            yield chunk


# ---------------------------------------------------------------------------
# The listening chime
# ---------------------------------------------------------------------------

def chime_wav(rate: int = 22050, notes: tuple[tuple[float, float], ...] = ((880.0, 0.07), (1320.0, 0.10))) -> bytes:
    """A short two-note chime, as WAV bytes made in memory. Rising = listening, falling = got it."""
    parts = []
    for freq, seconds in notes:
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
DONE_CHIME = chime_wav(notes=((1320.0, 0.06), (990.0, 0.08)))   # "got it": you can stop talking


def play_chime(player: Any = None, sound: bytes = CHIME) -> None:
    """Play a chime without waiting for it."""
    if player is None:
        from .speak import WinsoundPlayer

        player = WinsoundPlayer()
    threading.Thread(target=player.play, args=(sound,), daemon=True).start()


CLOSE_CHIME = chime_wav(notes=((660.0, 0.09),))   # one low note: "I've stopped listening"
ALARM_CHIME = chime_wav(notes=((988.0, 0.12), (0.0, 0.08)) * 3)   # three beeps: a timer has finished


def play_alarm(player: Any = None) -> None:
    play_chime(player, ALARM_CHIME)


def play_done_chime(player: Any = None) -> None:
    play_chime(player, DONE_CHIME)


def play_close_chime(player: Any = None) -> None:
    play_chime(player, CLOSE_CHIME)


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
