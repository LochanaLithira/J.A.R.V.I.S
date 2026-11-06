"""Your voice check: J.A.R.V.I.S answers "hey jarvis" only when it sounds like you.

How it works:
- A speaker-recognition model (WeSpeaker ResNet34, trained on VoxCeleb; CC-BY-4.0) turns speech
  into 256 numbers that describe the voice, not the words. Silero VAD first keeps only the part
  where someone is speaking.
- Teaching it (enroll.py / teaching.py) averages 20 of your "hey jarvis", plus noisy copies of
  each made in memory, into one voiceprint. The bar goes just above the closest of ~600 real
  people's voices (shipped as numbers in data/other_voices.npy) and Windows' own voices.
- When the wake word fires, the speech in the last 2 s is turned into numbers the same way and
  compared with your voiceprint (cosine similarity). Above the bar: it's you. Each spoken command
  is checked the same way (`judge`), and the gate lets only your voice change things.
- Why this design: see the test results next to BAR_ABOVE_CLOSEST below. Other scoring tricks
  (cohort score normalisation, matching against single samples, a trained "you vs others"
  classifier) were tried on the same test and did not do better.

Privacy and safety rules this file enforces:
- No audio is ever stored. Sound is turned into numbers in memory and dropped.
- The voiceprint file holds only numbers (no audio), lives outside the workspace, and is
  never sent anywhere. Its SHA-256 is kept in Windows Credential Manager, so an edited or
  swapped file is refused.
- The model's fingerprint is fixed here and checked on every load, like the wake-word models.
- Runs only when the wake word fires and for each spoken command, on one thread, ~0.2-0.3 s
  (for a command, while the words are being turned into text, so it adds no wait).

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
VOICEPRINT_VERSION = 2            # 2: speech-only scoring and a bar set against real people
CHECK_SECONDS = 2.0               # how much sound before the wake-up is looked at (speech kept)
COMMAND_SECONDS = 4.0             # at most this much of a spoken command is checked (tested at 4 s)
COHORT_FILE = "data/other_voices.npy"
COHORT_SHA256 = "76f3c3d69dc17f7d1bec87a79dd9e761cd04333ee0686b55cc50c7791612e9e4"
COHORT_ROWS = (100, 2000)
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


def load_cohort(path: Path | None = None) -> np.ndarray:
    """The voice numbers of real people (LibriSpeech, CC BY 4.0) that ship with J.A.R.V.I.S and set
    your bar. Checked against its fixed fingerprint, so a swapped file can't lower your bar."""
    path = Path(path) if path is not None else Path(__file__).resolve().parent / COHORT_FILE
    try:
        raw = path.read_bytes()
    except OSError:
        raise VoiceprintError("The comparison voices are missing. Reinstall J.A.R.V.I.S.") from None
    if hashlib.sha256(raw).hexdigest() != COHORT_SHA256:
        raise VoiceprintError("The comparison voices have changed, so they were refused. Reinstall J.A.R.V.I.S.")
    import io

    data = np.load(io.BytesIO(raw), allow_pickle=False)
    if (data.ndim != 2 or data.shape[1] != EMBEDDING_SIZE or not COHORT_ROWS[0] <= len(data) <= COHORT_ROWS[1]
            or not np.isfinite(data).all()):
        raise VoiceprintError("The comparison voices are damaged. Reinstall J.A.R.V.I.S.")
    data = data.astype(np.float64)
    return data / np.linalg.norm(data, axis=1, keepdims=True)


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
        if version != VOICEPRINT_VERSION:
            raise VoiceprintError("The voice check was upgraded. Teach J.A.R.V.I.S your voice again (Settings).")
        if model != SPEAKER_MODEL_SHA256:
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

VAD_FRAME = 512                       # 32 ms: the frame size Silero VAD expects at 16 kHz
SPEECH_PROB = 0.5
MIN_SPEECH_SAMPLES = int(0.6 * SAMPLE_RATE)


def speech_only(audio: np.ndarray, vad: Callable[[np.ndarray], float] | None) -> np.ndarray:
    """Keep just the 32 ms frames that sound like speech (plus one frame either side), so silence
    and room noise around the words don't blur the voice numbers. If too little is left, the
    audio is returned as it was; the caller decides whether that's enough."""
    audio = np.asarray(audio, dtype=np.int16).reshape(-1)
    if vad is None or len(audio) < VAD_FRAME:
        return audio
    if hasattr(vad, "reset"):
        vad.reset()
    count = len(audio) // VAD_FRAME
    speech = np.array([vad(audio[i * VAD_FRAME:(i + 1) * VAD_FRAME]) >= SPEECH_PROB for i in range(count)])
    keep = speech.copy()
    keep[1:] |= speech[:-1]
    keep[:-1] |= speech[1:]
    if keep.sum() * VAD_FRAME < MIN_SPEECH_SAMPLES:
        return audio
    frames = audio[:count * VAD_FRAME].reshape(count, VAD_FRAME)
    return frames[keep].reshape(-1)


def _mix(audio: np.ndarray, other: np.ndarray, snr_db: float) -> np.ndarray:
    a = audio.astype(np.float64)
    pa, po = float((a ** 2).mean()) + 1e-9, float((other ** 2).mean()) + 1e-9
    return np.clip(a + other * np.sqrt(pa / (po * 10 ** (snr_db / 10))), -32768, 32767).astype(np.int16)


def noisy_copy(audio: np.ndarray, kind: str, rng: np.random.Generator,
               chatter: list[np.ndarray] | None = None) -> np.ndarray:
    """A copy of a sample as it might sound on a worse day, made in memory only:
    "noise" (room hiss), "chatter" (other voices behind you) or "phone" (a thin microphone)."""
    audio = np.asarray(audio, dtype=np.int16).reshape(-1)
    n = len(audio)
    if kind == "noise":
        pink = np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) / np.sqrt(np.arange(n // 2 + 1) + 1.0), n)
        return _mix(audio, pink, rng.uniform(6, 12))
    if kind == "chatter" and chatter:
        mixed = np.zeros(n)
        for i in rng.choice(len(chatter), size=min(3, len(chatter)), replace=False):
            c = np.asarray(chatter[i], dtype=np.float64).reshape(-1)
            c = np.resize(c, n) if len(c) else np.zeros(n)
            mixed += np.roll(c, int(rng.integers(n)))
        return _mix(audio, mixed, rng.uniform(6, 12))
    if kind == "phone":
        spectrum = np.fft.rfft(audio.astype(np.float64))
        hz = np.fft.rfftfreq(n, 1 / SAMPLE_RATE)
        spectrum[(hz < 300) | (hz > 3400)] = 0
        return _mix(np.fft.irfft(spectrum, n), rng.standard_normal(n), 20)
    return audio.copy()


def _last_seconds(chunks: Iterable[np.ndarray], seconds: float) -> np.ndarray:
    chunks = list(chunks)
    if not chunks:
        return np.zeros(0, np.int16)
    audio = np.concatenate([np.asarray(c, dtype=np.int16).reshape(-1) for c in chunks])
    return audio[-int(seconds * SAMPLE_RATE):]


def voice_numbers(model: SpeakerModel, audio: np.ndarray, vad: Callable[[np.ndarray], float] | None,
                  seconds: float = CHECK_SECONDS, last: bool = True) -> np.ndarray:
    """Voice numbers for the speech in `audio` (the last or first `seconds` of it). Used the same
    way when teaching, at each wake-up and for commands, so the scores line up."""
    audio = np.asarray(audio, dtype=np.int16).reshape(-1)
    n = int(seconds * SAMPLE_RATE)
    return model.embed(speech_only(audio[-n:] if last else audio[:n], vad))


class SpeakerCheck:
    """Callable for Triggers' voice check: True when the speech just before the wake-up sounds
    like you. `judge` checks a whole spoken command the same way."""

    def __init__(self, model: SpeakerModel, voiceprint: Voiceprint, on_score: Callable[[float], None] | None = None,
                 vad: Callable[[np.ndarray], float] | None = None):
        import threading

        self._model, self._vp, self._vad = model, voiceprint, vad
        self._centroid = np.array(voiceprint.centroid)
        self._lock = threading.Lock()       # the wake-up and a command never share the models at once
        self.on_score = on_score            # told each score (a number), for the log and screen
        self.last_score: float | None = None

    def score(self, chunks: Iterable[np.ndarray]) -> float:
        with self._lock:
            return similarity(voice_numbers(self._model, _last_seconds(chunks, CHECK_SECONDS), self._vad),
                              self._centroid)

    def judge(self, audio: np.ndarray) -> tuple[str, float | None]:
        """Was this spoken command in your voice? ("you" / "not you" / "unsure", score)"""
        audio = np.asarray(audio, dtype=np.int16).reshape(-1)
        with self._lock:
            speech = speech_only(audio[:int(COMMAND_SECONDS * SAMPLE_RATE)], self._vad)
            if len(speech) < MIN_SPEECH_SAMPLES:
                return "unsure", None
            try:
                s = similarity(self._model.embed(speech), self._centroid)
            except VoiceprintError:
                return "unsure", None
        return ("you" if s >= self._vp.threshold else "not you"), s

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


def command_judge(check: object) -> Callable[[np.ndarray], tuple[str, float | None]] | None:
    """From the running wake-up checks, the function that checks a spoken command:
    audio -> ("you" / "not you" / "unsure", score). None when no voiceprint is in use."""
    for _name, c in getattr(check, "_checks", [("", check)]):
        if isinstance(c, (SpeakerCheck, StrictCheck)):
            return c.judge
    return None


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

    def judge(self, audio: np.ndarray) -> tuple[str, float | None]:
        if self._check is None:
            return "not you", None              # the voice check couldn't be set up: refuse
        try:
            return self._check.judge(audio)
        except Exception:
            return "not you", None


# ---------------------------------------------------------------------------
# Choosing the bar from your samples and some other voices
# ---------------------------------------------------------------------------

# Tested on 73 LibriSpeech speakers J.A.R.V.I.S never saw, with noise, chatter and phone-quality
# sound added, and your voiceprint built with noisy copies: a bar just above the closest of ~600
# other voices let in 0.14% of strangers (2.3% of the three who sounded most like you), and
# refused you about 1% of the time when quiet, ~10% in noise; 4 s commands: 0.2%.
# A looser bar (the closest 0.5% get in) refused you less but let in 10% of close-sounding people.
# The old rule (well inside the gap, no copies) refused you about 1 in 4 times in noise.
# The shipped voices were recorded on other microphones; someone speaking into YOUR microphone
# shares its sound and scores higher. So the bar also goes a quarter of the way from the closest
# voice up to your own typical score: on the test that let in 0.02% of strangers and 0.5% of the
# three closest-sounding, for ~11% refusals in noise (vs ~9%), 0.4% for 4 s commands.
BAR_ABOVE_CLOSEST = 0.01          # the bar sits at least this far above the closest other voice
BAR_GAP_SHARE = 0.25              # ...and at least this share of the way from it up to you
MIN_MARGIN = 0.03                 # your own samples must clear the bar by at least this much
BAR_LIMITS = (0.25, 0.75)
MIN_OTHERS = 50


@dataclass(frozen=True)
class Calibration:
    voiceprint: Voiceprint | None     # None when your voice and other voices are too alike
    yours: tuple[float, float]        # (lowest typical, highest) similarity of your own samples
    others: float                     # highest similarity of any other voice
    message: str
    bar: float | None = None


def calibrate(yours: list[np.ndarray], others: list[np.ndarray], copies: list[np.ndarray] = ()) -> Calibration:
    """Average your samples into a voiceprint and set the bar from real people's voices.

    `others` are other people's voice numbers (the shipped set plus Windows' voices); the bar goes
    above the closest of them, a quarter of the way up to your own typical score. `copies` are
    noisy versions of your samples, made
    in memory, which go into the average so it knows how you sound on a worse day.
    Each of your samples is compared with the average of the OTHER samples (leave one out), so the
    numbers show how you'll score on a new day, not how well a sample matches itself."""
    if len(yours) < 5:
        raise VoiceprintError("Not enough samples of your voice.")
    if len(others) < MIN_OTHERS:
        raise VoiceprintError("There weren't enough other voices to compare with.")
    vecs = np.array([normalize(v) for v in yours])
    extra = np.array([normalize(v) for v in copies]) if len(copies) else np.zeros((0, vecs.shape[1]))
    centroid = normalize(np.vstack([vecs, extra]).mean(axis=0))
    own = []
    for i in range(len(vecs)):
        rest = normalize(np.vstack([np.delete(vecs, i, axis=0), extra]).mean(axis=0))
        own.append(float(vecs[i] @ rest))
    own_low = float(np.percentile(own, 10))       # one odd sample shouldn't decide anything
    own_high = max(own)
    scores = np.array([float(normalize(o) @ centroid) for o in others])
    other_high = float(scores.max())
    bar = max(other_high + BAR_ABOVE_CLOSEST, other_high + BAR_GAP_SHARE * (own_low - other_high),
              BAR_LIMITS[0])                                    # the limit only ever raises it
    if bar > BAR_LIMITS[1] or own_low < bar + MIN_MARGIN:
        return Calibration(None, (own_low, own_high), other_high,
                           "Your voice and other people's voices came out too alike, so nothing was saved. "
                           "Try again somewhere quieter, a little closer to the microphone.")
    vp = Voiceprint(tuple(centroid), bar, len(yours))
    return Calibration(vp, (own_low, own_high), other_high,
                       f"Your voice scored {own_low:.2f}-{own_high:.2f}. Out of {len(scores)} other "
                       f"people's voices, the closest reached {other_high:.2f}; the bar is {bar:.2f}.", bar)
