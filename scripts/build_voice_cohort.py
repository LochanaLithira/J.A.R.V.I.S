"""Build backend/voice/data/other_voices.npy: the voice numbers of real people that your voice is
compared with when J.A.R.V.I.S sets your bar. Run once by the developer; the result ships with the
project and its SHA-256 is fixed in backend/voice/voiceprint.py.

Source: LibriSpeech dev-clean and dev-other (CC BY 4.0, https://www.openslr.org/12), 73 speakers.
Only numbers are written (8 per speaker, 256 each); no audio goes into the project. Some samples
get room noise, background chatter or a phone-quality filter, so the set covers everyday sound.

    python scripts/build_voice_cohort.py dev-clean.tar.gz dev-other.tar.gz

The archives are read in memory and never unpacked. Needs `soundfile` (not a project dependency).
"""
from __future__ import annotations

import io
import sys
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import config                                       # noqa: E402
from backend.voice.voiceprint import (COHORT_FILE, SpeakerModel, noisy_copy,  # noqa: E402
                                      speech_only)

UTTERANCES = 8                     # per speaker, one 2 s clip from each
SECONDS = 2.0


def clips(archive: Path) -> dict[str, list[np.ndarray]]:
    import soundfile as sf

    out: dict[str, list[np.ndarray]] = {}
    with tarfile.open(archive, "r:gz") as tf:
        for m in tf:
            if not (m.isfile() and m.name.endswith(".flac")):
                continue
            speaker = m.name.split("/")[-3]
            got = out.setdefault(speaker, [])
            if len(got) < UTTERANCES:
                audio, rate = sf.read(io.BytesIO(tf.extractfile(m).read()), dtype="int16")
                if rate == 16000 and len(audio) >= SECONDS * 16000 + 8000:
                    got.append(audio[8000:8000 + int(SECONDS * 16000)])      # skip the first 0.5 s
    return out


def main(archives: list[str]) -> int:
    model = SpeakerModel(config.MODELS_DIR)
    rng = np.random.default_rng(12)
    speakers: dict[str, list[np.ndarray]] = {}
    for a in archives:
        speakers.update(clips(Path(a)))
    chatter = [c for cs in speakers.values() for c in cs[:1]]
    rows = []
    for name in sorted(speakers):
        for i, clip in enumerate(speakers[name]):
            kind = ("clean", "noise", "chatter", "phone")[i % 4]
            others = [c for c in chatter if c is not clip]
            x = clip if kind == "clean" else noisy_copy(clip, kind, rng, others)
            rows.append(model.embed(speech_only(x, None)))
    data = np.array(rows, dtype=np.float32)
    out = ROOT / "backend" / "voice" / COHORT_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, data, allow_pickle=False)
    import hashlib
    print(f"{len(speakers)} speakers, {len(data)} rows -> {out}\nSHA-256 {hashlib.sha256(out.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
