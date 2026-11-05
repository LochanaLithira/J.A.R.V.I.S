"""Your voice check: J.A.R.V.I.S answers "hey jarvis" only when it sounds like you.

How it works:
- A speaker-recognition model (WeSpeaker ResNet34, trained on VoxCeleb; CC-BY-4.0) turns about
  1.5 seconds of speech into 256 numbers that describe the voice, not the words.
- Teaching it (enroll.py) averages 20 of your "hey jarvis" into one voiceprint and picks a bar.
- When the wake word fires, the last 1.5 s is turned into numbers the same way and compared with
  your voiceprint (cosine similarity). Above the bar: it's you.

Privacy and safety rules this file enforces:
- No audio is ever stored. Sound is turned into numbers in memory and dropped.
- The voiceprint file holds only numbers (no audio), lives outside the workspace, and is
  never sent anywhere. Its SHA-256 is kept in Windows Credential Manager, so an edited or
  swapped file is refused.
- The model's fingerprint is fixed here and checked on every load, like the wake-word models.
- Runs only when the wake word fires (a few times an hour at most), on one thread, ~50 ms.

Not protected against: a recording of your own voice played back, or a convincing voice clone.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .triggers import SAMPLE_RATE, TriggerError, _fetch

SPEAKER_MODEL = "wespeaker_en_voxceleb_resnet34.onnx"
SPEAKER_MODEL_SHA256 = "5ef208a9da1453335308a6b6f4e6dfbd7e183a38b604de0a57664f45d257fe94"
SPEAKER_MODEL_URL = ("https://huggingface.co/csukuangfj/speaker-embedding-models/resolve/main/"
                     + SPEAKER_MODEL)
MAX_SPEAKER_MODEL_BYTES = 40_000_000
VOICEPRINT_VERSION = 1
CHECK_SECONDS = 1.5               # how much sound before the wake-up is compared
EMBEDDING_SIZE = 256
DIGEST_NAME = "voiceprint_digest"  # where the voiceprint's fingerprint lives in Credential Manager


class VoiceprintError(Exception):
    """Something about the voice check is missing or wrong. The message is safe to show."""


# ---------------------------------------------------------------------------
# Audio preparation: Kaldi-style log mel filterbank, as the model was trained with
# (80 bands, 25 ms frames every 10 ms, Hamming window, no dither, mean removed per band)
# ---------------------------------------------------------------------------

_FRAME, _SHIFT, _FFT, _BANDS = 400, 160, 512, 80
_PREEMPH, _LOW_HZ = 0.97, 20.0


def _mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 1127.0 * np.log(1.0 + np.asarray(hz) / 700.0)


def _mel_banks() -> np.ndarray:
    """(80, 256) triangular filters over the FFT bins, built exactly as Kaldi does."""
    bins = _FFT // 2
    width = SAMPLE_RATE / _FFT
    low, high = _mel(_LOW_HZ), _mel(SAMPLE_RATE / 2)
    delta = (high - low) / (_BANDS + 1)
    freqs_mel = _mel(np.arange(bins) * width)
    banks = np.zeros((_BANDS, bins), dtype=np.float64)
    for b in range(_BANDS):
        left, center, right = low + b * delta, low + (b + 1) * delta, low + (b + 2) * delta
        up = (freqs_mel - left) / (center - left)
        down = (right - freqs_mel) / (right - center)
        inside = (freqs_mel > left) & (freqs_mel < right)
        banks[b] = np.where(inside, np.where(freqs_mel <= center, up, down), 0.0)
    return banks


_BANKS = _mel_banks()
_WINDOW = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(_FRAME) / (_FRAME - 1))


def fbank(samples: np.ndarray) -> np.ndarray:
    """16 kHz int16 sound -> (frames, 80) features with the mean of each band removed."""
    x = np.asarray(samples, dtype=np.float64).reshape(-1)    # int16 scale, as the model expects
    if len(x) < _FRAME:
        return np.zeros((0, _BANDS), dtype=np.float32)
    count = 1 + (len(x) - _FRAME) // _SHIFT
    idx = np.arange(_FRAME)[None, :] + _SHIFT * np.arange(count)[:, None]
    frames = x[idx]
    frames = frames - frames.mean(axis=1, keepdims=True)                     # remove DC offset
    frames = np.concatenate([frames[:, :1] * (1 - _PREEMPH),
                             frames[:, 1:] - _PREEMPH * frames[:, :-1]], axis=1)
    frames = frames * _WINDOW
    power = np.abs(np.fft.rfft(frames, n=_FFT)) ** 2
    feats = np.log(np.maximum(power[:, :_FFT // 2] @ _BANKS.T, np.finfo(np.float32).eps))
    feats = feats - feats.mean(axis=0, keepdims=True)                        # CMN
    return feats.astype(np.float32)


# ---------------------------------------------------------------------------
# The model: fetched once, checked on every load
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def speaker_model_path(models_dir: Path) -> Path:
    """The model's path, or VoiceprintError if it is missing or changed."""
    path = Path(models_dir) / SPEAKER_MODEL
    if not path.is_file():
        raise VoiceprintError("The voice-check model is not downloaded. Run: python -m backend.voice.enroll download")
    if path.stat().st_size > MAX_SPEAKER_MODEL_BYTES or _sha256(path) != SPEAKER_MODEL_SHA256:
        raise VoiceprintError("The voice-check model has changed, so it was refused. "
                              "Run: python -m backend.voice.enroll download")
    return path


def download_speaker_model(models_dir: Path, fetch: Callable[..., bytes] = _fetch) -> bool:
    """Fetch the model if it is missing or changed; saved only if its fingerprint matches.
    Returns True if it was downloaded."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / SPEAKER_MODEL
    if path.is_file() and path.stat().st_size <= MAX_SPEAKER_MODEL_BYTES and _sha256(path) == SPEAKER_MODEL_SHA256:
        return False
    try:
        data = fetch(SPEAKER_MODEL_URL, max_bytes=MAX_SPEAKER_MODEL_BYTES)
    except TriggerError as exc:
        raise VoiceprintError(str(exc)) from None
    except Exception:
        raise VoiceprintError("I couldn't download the voice-check model. Check the internet connection.") from None
    if (not isinstance(data, bytes) or len(data) > MAX_SPEAKER_MODEL_BYTES
            or hashlib.sha256(data).hexdigest() != SPEAKER_MODEL_SHA256):
        raise VoiceprintError("The downloaded voice-check model did not match its fingerprint, so it was not saved.")
    tmp = path.with_suffix(".part")
    tmp.write_bytes(data)
    os.replace(tmp, path)              # all or nothing: never a half-written model
    return True


class SpeakerModel:
    """Turns speech into 256 numbers describing the voice."""

    def __init__(self, models_dir: Path):
        path = speaker_model_path(models_dir)
        import onnxruntime as ort      # imported here, so nothing heavy loads until it is needed

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(str(path), sess_options=options,
                                             providers=["CPUExecutionProvider"])

    def embed(self, samples: np.ndarray) -> np.ndarray:
        """Unit-length voice numbers for 16 kHz int16 sound (at least ~0.5 s)."""
        feats = fbank(samples)
        if len(feats) < 50:
            raise VoiceprintError("Too little sound to recognise a voice.")
        out = self._session.run(["embs"], {"feats": feats[None, :, :]})[0].reshape(-1)
        return normalize(out)


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(v))
    if not math.isfinite(n) or n == 0.0:
        raise VoiceprintError("The voice numbers were not usable.")
    return v / n


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(normalize(a), normalize(b)))


# ---------------------------------------------------------------------------
# The voiceprint file: numbers only, fingerprint kept in Credential Manager
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Voiceprint:
    centroid: tuple[float, ...]        # your average voice numbers (unit length)
    threshold: float                   # the bar a wake-up must reach
    samples: int
    created: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M"))
    model: str = SPEAKER_MODEL_SHA256
    version: int = VOICEPRINT_VERSION

    def to_json(self) -> str:
        return json.dumps({"version": self.version, "model": self.model, "created": self.created,
                           "samples": self.samples, "threshold": round(self.threshold, 4),
                           "centroid": [round(x, 6) for x in self.centroid]}, indent=1)

    @classmethod
    def from_json(cls, text: str) -> "Voiceprint":
        try:
            data = json.loads(text)
            centroid = tuple(float(x) for x in data["centroid"])
            threshold, samples = float(data["threshold"]), int(data["samples"])
            version, model, created = int(data["version"]), str(data["model"]), str(data["created"])
        except (ValueError, KeyError, TypeError):
            raise VoiceprintError("The voiceprint file is damaged. Teach J.A.R.V.I.S your voice again.") from None
        if version != VOICEPRINT_VERSION or model != SPEAKER_MODEL_SHA256:
            raise VoiceprintError("The voiceprint was made for a different model. Teach J.A.R.V.I.S your voice again.")
        if (len(centroid) != EMBEDDING_SIZE or not all(math.isfinite(x) for x in centroid)
                or not 0.0 < threshold < 1.0 or not 1 <= samples <= 100):
            raise VoiceprintError("The voiceprint file is damaged. Teach J.A.R.V.I.S your voice again.")
        return cls(tuple(normalize(np.array(centroid))), threshold, samples, created, model, version)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save_voiceprint(vp: Voiceprint, path: Path, set_secret: Callable[[str, str], None]) -> None:
    """Write the file (all or nothing) and keep its fingerprint in Credential Manager."""
    text = vp.to_json()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_bytes(text.encode("utf-8"))         # exact bytes: no newline changes on Windows
    set_secret(DIGEST_NAME, _digest(text))
    os.replace(tmp, path)


def load_voiceprint(path: Path, get_secret: Callable[[str], str | None]) -> Voiceprint | None:
    """Your voiceprint, None if you haven't taught one, or VoiceprintError if it was changed."""
    path = Path(path)
    if not path.is_file():
        return None
    raw = path.read_bytes()
    if len(raw) > 100_000:
        raise VoiceprintError("The voiceprint file is damaged. Teach J.A.R.V.I.S your voice again.")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = ""
    if not text:
        raise VoiceprintError("The voiceprint file is damaged. Teach J.A.R.V.I.S your voice again.")
    try:
        stored = get_secret(DIGEST_NAME)
    except Exception:
        stored = None
    if not stored or stored != _digest(text):
        raise VoiceprintError("The voiceprint file was changed outside J.A.R.V.I.S, so it was refused. "
                              "Teach J.A.R.V.I.S your voice again.")
    return Voiceprint.from_json(text)


def delete_voiceprint(path: Path, delete_secret: Callable[[str], bool]) -> bool:
    """Forget your voiceprint. Returns True if there was one."""
    path = Path(path)
    existed = path.is_file()
    if existed:
        path.unlink()
    try:
        delete_secret(DIGEST_NAME)
    except Exception:
        pass
    return existed


# ---------------------------------------------------------------------------
# The check that runs when the wake word fires
# ---------------------------------------------------------------------------

def _last_seconds(chunks: Iterable[np.ndarray], seconds: float) -> np.ndarray:
    chunks = list(chunks)
    if not chunks:
        return np.zeros(0, np.int16)
    audio = np.concatenate([np.asarray(c, dtype=np.int16).reshape(-1) for c in chunks])
    return audio[-int(seconds * SAMPLE_RATE):]


class SpeakerCheck:
    """Callable for Triggers' voice check: True when the last 1.5 s sounds like you."""

    def __init__(self, model: SpeakerModel, voiceprint: Voiceprint, on_score: Callable[[float], None] | None = None):
        self._model, self._vp = model, voiceprint
        self._centroid = np.array(voiceprint.centroid)
        self.on_score = on_score            # told each score (a number), for the log and screen
        self.last_score: float | None = None

    def score(self, chunks: Iterable[np.ndarray]) -> float:
        return similarity(self._model.embed(_last_seconds(chunks, CHECK_SECONDS)), self._centroid)

    def __call__(self, chunks: Iterable[np.ndarray]) -> bool:
        try:
            s = self.score(chunks)
        except VoiceprintError:                 # too little sound to recognise anyone: not you
            self.last_score = None
            return False
        self.last_score = s
        if self.on_score is not None:
            try:
                self.on_score(s)
            except Exception:
                pass
        return s >= self._vp.threshold


class AllChecks:
    """Several wake-up checks in a row, cheapest first; every one must pass. `failed` names the
    one that said no (for the audit log), and `score` is your voice score when it ran."""

    def __init__(self, checks: list[tuple[str, Callable[[list], bool]]]):
        self._checks = checks
        self.failed: str | None = None
        self.score: float | None = None

    def __call__(self, chunks: list) -> bool:
        self.failed, self.score = None, None
        for name, check in self._checks:
            ok = check(chunks)
            if isinstance(check, (SpeakerCheck, StrictCheck)):
                self.score = check.last_score
            if not ok:
                self.failed = name
                return False
        return True


class StrictCheck:
    """Wraps your voice check so an error counts as "not you" (VOICE_CHECK_STRICT), instead of
    letting the wake-up through."""

    def __init__(self, check: Callable[[list], bool] | None):
        self._check = check
        self.last_score: float | None = None

    def __call__(self, chunks: list) -> bool:
        if self._check is None:
            return False                    # the voice check couldn't be set up: refuse
        try:
            ok = bool(self._check(chunks))
        except Exception:
            ok = False
        self.last_score = getattr(self._check, "last_score", None)
        return ok


# ---------------------------------------------------------------------------
# Choosing the bar from your samples and some other voices
# ---------------------------------------------------------------------------

MIN_GAP = 0.12                    # your voice must clear other voices by at least this much
BAR_LIMITS = (0.25, 0.75)


@dataclass(frozen=True)
class Calibration:
    voiceprint: Voiceprint | None     # None when your voice and other voices are too alike
    yours: tuple[float, float]        # (lowest typical, highest) similarity of your own samples
    others: float                     # highest similarity of any other voice
    message: str


def calibrate(yours: list[np.ndarray], others: list[np.ndarray]) -> Calibration:
    """Average your samples into a voiceprint and pick a bar between you and other voices.

    Each of your samples is compared with the average of the OTHER samples (leave one out), so the
    numbers show how you'll score on a new day, not how well a sample matches itself."""
    if len(yours) < 5:
        raise VoiceprintError("Not enough samples of your voice.")
    vecs = np.array([normalize(v) for v in yours])
    centroid = normalize(vecs.mean(axis=0))
    own = []
    for i in range(len(vecs)):
        rest = normalize(np.delete(vecs, i, axis=0).mean(axis=0))
        own.append(float(vecs[i] @ rest))
    own_low = float(np.percentile(own, 10))       # one odd sample shouldn't decide the bar
    own_high = max(own)
    other_high = max((float(normalize(o) @ centroid) for o in others), default=0.0)
    gap = own_low - other_high
    if gap < MIN_GAP:
        return Calibration(None, (own_low, own_high), other_high,
                           "Your voice and other voices came out too alike, so nothing was saved. "
                           "Try again somewhere quieter, a little closer to the microphone.")
    bar = min(max(other_high + gap * 0.45, BAR_LIMITS[0]), BAR_LIMITS[1])
    vp = Voiceprint(tuple(centroid), bar, len(yours))
    return Calibration(vp, (own_low, own_high), other_high,
                       f"Your voice scored {own_low:.2f}-{own_high:.2f}, other voices at most "
                       f"{other_high:.2f}. The bar is {bar:.2f}.")
