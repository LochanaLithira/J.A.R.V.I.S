"""Your voice check: the voiceprint file, the bar, the checks, and teaching. No real audio is used
or kept: the tests use made-up voice numbers, and Windows' synthetic voices for the real model."""
import hashlib
import json
import sys

import numpy as np
import pytest

from backend import config, main
from backend.voice import enroll, voiceprint as vpm
from backend.voice.triggers import CHUNK, Triggers
from backend.voice.voiceprint import (AllChecks, SpeakerCheck, StrictCheck, Voiceprint, VoiceprintError,
                                      calibrate, delete_voiceprint, fbank, load_voiceprint, normalize,
                                      save_voiceprint)

rng = np.random.default_rng(7)


def unit(v):
    return normalize(np.asarray(v, dtype=float))


ME = unit(rng.normal(size=256))


def like_me(spread=0.25):
    return unit(ME + rng.normal(scale=spread / 16, size=256))


def someone_else():
    return unit(rng.normal(size=256))


class Secrets:
    def __init__(self):
        self.store = {}

    def get(self, name):
        return self.store.get(name)

    def set(self, name, value):
        self.store[name] = value

    def delete(self, name):
        return self.store.pop(name, None) is not None


def a_voiceprint(threshold=0.5):
    return Voiceprint(tuple(ME), threshold, 20)


# ---- audio preparation ------------------------------------------------------------------------

def test_fbank_shape_and_mean_removed():
    sound = (rng.normal(0, 2000, 16000)).astype(np.int16)
    feats = fbank(sound)
    assert feats.shape == (1 + (16000 - 400) // 160, 80) and feats.dtype == np.float32
    assert np.allclose(feats.mean(axis=0), 0, atol=1e-4)


def test_fbank_of_too_little_sound_is_empty():
    assert fbank(np.zeros(100, np.int16)).shape == (0, 80)


def test_mel_filters_cover_the_band():
    banks = vpm._mel_banks()
    assert banks.shape == (80, 256) and (banks >= 0).all() and (banks.sum(axis=1) > 0).all()


# ---- the voiceprint file -----------------------------------------------------------------------

def test_save_and_load_round_trip(tmp_path):
    s, path = Secrets(), tmp_path / "vp.json"
    save_voiceprint(a_voiceprint(), path, s.set)
    loaded = load_voiceprint(path, s.get)
    assert np.allclose(loaded.centroid, ME, atol=1e-5) and loaded.threshold == 0.5 and loaded.samples == 20
    assert s.store["voiceprint_digest"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_file_holds_numbers_only(tmp_path):
    s, path = Secrets(), tmp_path / "vp.json"
    save_voiceprint(a_voiceprint(), path, s.set)
    data = json.loads(path.read_text())
    assert set(data) == {"version", "model", "created", "samples", "threshold", "centroid"}
    assert len(data["centroid"]) == 256 and all(isinstance(x, float) for x in data["centroid"])
    assert path.stat().st_size < 10_000


def test_no_voiceprint_is_not_an_error(tmp_path):
    assert load_voiceprint(tmp_path / "none.json", Secrets().get) is None


@pytest.mark.parametrize("change", [
    lambda t: t.replace('"threshold": 0.5', '"threshold": 0.05'),      # someone lowers the bar
    lambda t: t + " ",
])
def test_an_edited_voiceprint_is_refused(tmp_path, change):
    s, path = Secrets(), tmp_path / "vp.json"
    save_voiceprint(a_voiceprint(), path, s.set)
    path.write_text(change(path.read_text()))
    with pytest.raises(VoiceprintError, match="changed outside"):
        load_voiceprint(path, s.get)


def test_a_swapped_voiceprint_is_refused(tmp_path):
    mine, theirs = Secrets(), Secrets()
    save_voiceprint(a_voiceprint(), tmp_path / "mine.json", mine.set)
    save_voiceprint(Voiceprint(tuple(someone_else()), 0.3, 20), tmp_path / "theirs.json", theirs.set)
    (tmp_path / "mine.json").write_bytes((tmp_path / "theirs.json").read_bytes())
    with pytest.raises(VoiceprintError):
        load_voiceprint(tmp_path / "mine.json", mine.get)


def test_a_missing_fingerprint_is_refused(tmp_path):
    s, path = Secrets(), tmp_path / "vp.json"
    save_voiceprint(a_voiceprint(), path, s.set)
    s.store.clear()
    with pytest.raises(VoiceprintError):
        load_voiceprint(path, s.get)


@pytest.mark.parametrize("data", [
    {"centroid": [0.1] * 10},
    {"centroid": [float("nan")] * 256},
    {"threshold": 1.5},
    {"threshold": 0},
    {"model": "0" * 64},
    {"version": 99},
    {"samples": 0},
])
def test_damaged_contents_are_refused(data):
    good = json.loads(a_voiceprint().to_json())
    good.update(data)
    with pytest.raises(VoiceprintError):
        Voiceprint.from_json(json.dumps(good))


def test_delete_forgets_file_and_fingerprint(tmp_path):
    s, path = Secrets(), tmp_path / "vp.json"
    save_voiceprint(a_voiceprint(), path, s.set)
    assert delete_voiceprint(path, s.delete) and not path.exists() and s.store == {}
    assert not delete_voiceprint(path, s.delete)


# ---- choosing the bar ----------------------------------------------------------------------------

def test_calibration_puts_the_bar_between_you_and_others():
    result = calibrate([like_me() for _ in range(20)], [someone_else() for _ in range(12)])
    vp = result.voiceprint
    assert vp is not None and result.others < vp.threshold < result.yours[0]
    assert vpm.similarity(like_me(), np.array(vp.centroid)) >= vp.threshold
    assert all(vpm.similarity(someone_else(), np.array(vp.centroid)) < vp.threshold for _ in range(50))


def test_too_alike_saves_nothing():
    others = [like_me() for _ in range(5)]                       # "other voices" sound just like you
    result = calibrate([like_me() for _ in range(20)], others)
    assert result.voiceprint is None and "too alike" in result.message


def test_calibration_needs_enough_samples():
    with pytest.raises(VoiceprintError):
        calibrate([like_me() for _ in range(3)], [])


def test_the_bar_stays_within_limits():
    result = calibrate([like_me(spread=0.01) for _ in range(20)], [])
    assert vpm.BAR_LIMITS[0] <= result.voiceprint.threshold <= vpm.BAR_LIMITS[1]


# ---- the checks ------------------------------------------------------------------------------------

class FakeModel:
    def __init__(self, voice):
        self.voice, self.lengths = voice, []

    def embed(self, samples):
        self.lengths.append(len(samples))
        if len(samples) < 8000:
            raise VoiceprintError("Too little sound to recognise a voice.")
        return self.voice() if callable(self.voice) else self.voice


def chunks(seconds=2.0):
    return [np.zeros(CHUNK, np.int16) for _ in range(int(seconds * 16000 / CHUNK))]


def test_your_voice_passes_and_others_do_not():
    assert SpeakerCheck(FakeModel(like_me), a_voiceprint())(chunks())
    assert not SpeakerCheck(FakeModel(someone_else), a_voiceprint())(chunks())


def test_only_the_last_moment_is_compared():
    model = FakeModel(like_me)
    SpeakerCheck(model, a_voiceprint())(chunks(2.0))
    assert model.lengths == [int(vpm.CHECK_SECONDS * 16000)]


def test_too_little_sound_counts_as_not_you():
    check = SpeakerCheck(FakeModel(like_me), a_voiceprint())
    assert not check(chunks(0.3)) and check.last_score is None


def test_scores_are_reported_as_numbers():
    seen = []
    check = SpeakerCheck(FakeModel(like_me), a_voiceprint(), on_score=seen.append)
    check(chunks())
    assert len(seen) == 1 and isinstance(seen[0], float) and check.last_score == seen[0]


def test_all_checks_names_the_one_that_failed():
    combined = AllChecks([("no voice", lambda c: True), ("not your voice", SpeakerCheck(FakeModel(someone_else), a_voiceprint()))])
    assert not combined(chunks()) and combined.failed == "not your voice" and isinstance(combined.score, float)
    quiet = AllChecks([("no voice", lambda c: False), ("not your voice", SpeakerCheck(FakeModel(like_me), a_voiceprint()))])
    assert not quiet(chunks()) and quiet.failed == "no voice" and quiet.score is None   # cheapest first


def test_strict_check_refuses_on_error_or_when_missing():
    def broken(c):
        raise RuntimeError("model crashed")

    assert not StrictCheck(broken)(chunks()) and not StrictCheck(None)(chunks())
    assert StrictCheck(SpeakerCheck(FakeModel(like_me), a_voiceprint()))(chunks())


class Audit:
    def __init__(self):
        self.rows = []

    def record(self, event, **fields):
        self.rows.append((event, fields))


class LoudWake:
    """Scores like "hey jarvis" once ~1.6 s of sound has passed, as the real detector needs."""

    def __init__(self):
        self.n = 0

    def feed(self, chunk):
        self.n += 1
        return 0.9 if self.n >= 20 else 0.0

    def reset(self):
        self.n = 0


def test_a_wake_from_another_voice_is_ignored_and_audited_with_numbers_only():
    audit = Audit()
    check = AllChecks([("not your voice", SpeakerCheck(FakeModel(someone_else), a_voiceprint()))])
    t = Triggers(LoudWake(), None, audit=audit, voice_check=check)
    events = [t.process(c) for c in chunks(2.0)]
    assert not any(events)
    event, fields = audit.rows[0]
    assert event == "wake_rejected" and fields["reason"] == "not your voice"
    assert isinstance(fields["voice_score"], float) and set(fields) == {"source", "score", "reason", "voice_score"}


def test_your_wake_goes_through():
    check = AllChecks([("not your voice", SpeakerCheck(FakeModel(like_me), a_voiceprint()))])
    t = Triggers(LoudWake(), None, voice_check=check)
    assert any(t.process(c) for c in chunks(2.0))


# ---- teaching -----------------------------------------------------------------------------------------

def wake_moments(*voices):
    """Fake listener: each call returns the sound of one wake moment, or None."""
    it = iter(voices)
    return lambda: next(it, None)


def test_collect_keeps_twenty_and_retakes_odd_ones():
    sounds = [chunks(1.7)] * 30 + [None]
    voices = iter([like_me() for _ in range(6)] + [someone_else()] + [like_me() for _ in range(30)])
    model = FakeModel(lambda: next(voices))
    said = []
    kept = enroll.collect_samples(wake_moments(*sounds), model.embed, said.append)
    assert len(kept) == 20 and any("didn't sound like the others" in s for s in said)
    assert all(n <= int(vpm.CHECK_SECONDS * 16000) for n in model.lengths)


def test_collect_gives_up_after_too_many_tries():
    kept = enroll.collect_samples(lambda: None, FakeModel(like_me).embed, lambda s: None, max_tries=5)
    assert kept == []


def test_teach_saves_only_a_good_voiceprint():
    saved = []
    others = lambda: [("David", np.zeros(32000, np.int16))] * 6
    model = FakeModel(like_me)
    model_other = FakeModel(someone_else)

    class Mixed:                               # your samples sound like you; Windows' voices don't
        def __init__(self):
            self.calls = 0

        def embed(self, samples):
            self.calls += 1
            return model.embed(samples) if self.calls <= 20 else model_other.embed(samples)

    result = enroll.teach(wake_moments(*[chunks(1.7)] * 20), Mixed(), voices=others,
                          say=lambda s: None, save=saved.append)
    assert result.voiceprint is not None and saved == [result.voiceprint]


def test_teach_with_too_few_samples_saves_nothing():
    saved = []
    with pytest.raises(VoiceprintError):
        enroll.teach(wake_moments(*[chunks(1.7)] * 5), FakeModel(like_me), voices=lambda: [],
                     say=lambda s: None, save=saved.append)
    assert saved == []


# ---- starting up -----------------------------------------------------------------------------------------

def test_no_voiceprint_means_no_check():
    assert main.make_speaker_check(get=Secrets().get) == (None, "")


def test_a_taught_voiceprint_turns_the_check_on():
    s = Secrets()
    save_voiceprint(a_voiceprint(), config.VOICEPRINT, s.set)
    check, problem = main.make_speaker_check(get=s.get, model_factory=lambda d: FakeModel(like_me))
    assert isinstance(check, SpeakerCheck) and problem == ""


def test_voice_check_off_in_config(monkeypatch):
    s = Secrets()
    save_voiceprint(a_voiceprint(), config.VOICEPRINT, s.set)
    monkeypatch.setattr(config, "VOICE_CHECK", False)
    assert main.make_speaker_check(get=s.get, model_factory=lambda d: FakeModel(like_me)) == (None, "")


def test_a_changed_voiceprint_is_reported_and_falls_back(monkeypatch):
    s = Secrets()
    save_voiceprint(a_voiceprint(), config.VOICEPRINT, s.set)
    config.VOICEPRINT.write_text(config.VOICEPRINT.read_text().replace("0.5", "0.1"))
    check, problem = main.make_speaker_check(get=s.get, model_factory=lambda d: FakeModel(like_me))
    assert check is None and "changed outside" in problem and "answers any voice" in problem


def test_strict_mode_refuses_every_wake_when_broken(monkeypatch):
    monkeypatch.setattr(config, "VOICE_CHECK_STRICT", True)
    s = Secrets()
    save_voiceprint(a_voiceprint(), config.VOICEPRINT, s.set)

    def no_model(d):
        raise VoiceprintError("The voice-check model is not downloaded.")

    check, problem = main.make_speaker_check(get=s.get, model_factory=no_model)
    assert isinstance(check, StrictCheck) and not check(chunks()) and "ignored" in problem


def test_defaults():
    assert config.VOICE_CHECK is True and config.VOICE_CHECK_STRICT is False
    assert config.WORKSPACE not in config.VOICEPRINT.parents
    from backend.credentials import KNOWN_NAMES
    assert vpm.DIGEST_NAME in KNOWN_NAMES


# ---- the model download --------------------------------------------------------------------------------

def test_a_wrong_download_is_never_saved(tmp_path):
    with pytest.raises(VoiceprintError, match="fingerprint"):
        vpm.download_speaker_model(tmp_path, fetch=lambda url, max_bytes: b"evil model")
    assert list(tmp_path.iterdir()) == []


def test_a_good_download_is_saved(tmp_path, monkeypatch):
    data = b"pretend model"
    monkeypatch.setattr(vpm, "SPEAKER_MODEL_SHA256", hashlib.sha256(data).hexdigest())
    seen = []
    assert vpm.download_speaker_model(tmp_path, fetch=lambda url, max_bytes: seen.append(url) or data)
    assert (tmp_path / vpm.SPEAKER_MODEL).read_bytes() == data and seen[0].startswith("https://")
    assert not vpm.download_speaker_model(tmp_path, fetch=lambda url, max_bytes: pytest.fail("fetched again"))


def test_a_changed_model_file_is_refused(tmp_path):
    (tmp_path / vpm.SPEAKER_MODEL).write_bytes(b"tampered")
    with pytest.raises(VoiceprintError, match="changed"):
        vpm.speaker_model_path(tmp_path)


# ---- the real model, with Windows' synthetic voices (made in memory) --------------------------------------

@pytest.fixture(scope="module")
def real_model():
    try:
        return vpm.SpeakerModel(config.MODELS_DIR)
    except VoiceprintError:
        pytest.skip("voice-check model not downloaded (python -m backend.voice.enroll download)")


@pytest.mark.skipif(sys.platform != "win32", reason="needs Windows text-to-speech")
def test_real_model_tells_two_voices_apart(real_model):
    from backend.voice.other_voices import other_voices

    clips = {}
    for name, clip in other_voices():
        clips.setdefault(name, []).append(real_model.embed(clip[: int(vpm.CHECK_SECONDS * 16000)]))
    if len(clips) < 2:
        pytest.skip("fewer than two Windows voices installed")
    (a, va), (b, vb) = list(clips.items())[:2]
    same = np.mean([vpm.similarity(x, y) for i, x in enumerate(va) for y in va[i + 1:]])
    other = np.mean([vpm.similarity(x, y) for x in va for y in vb])
    assert same > other + 0.1, (a, b, same, other)


# ---- listening for each sample (real wake-word models, Windows' synthetic voice) ----------------------

class FakeMic:
    """Serves prepared sound in 80 ms slices, like the microphone."""

    def __init__(self, audio):
        self.slices = [audio[i:i + CHUNK] for i in range(0, len(audio) - CHUNK + 1, CHUNK)]

    def drain(self):
        return 0

    def chunks(self, stop=None, idle_timeout=None):
        yield from self.slices


def test_each_sample_waits_for_the_detector_then_hears_hey_jarvis(models_dir, spoken):
    from backend.voice.triggers import WakeWordDetector

    chimes = []
    quiet = np.zeros(enroll.WARM_UP_CHUNKS * CHUNK, np.int16)
    mic = FakeMic(np.concatenate([quiet, spoken["wake"], np.zeros(16000, np.int16)]))
    triggers = Triggers(WakeWordDetector(models_dir), None, config.WAKE_THRESHOLD, config.WAKE_PATIENCE,
                        cooldown_seconds=0.5)
    moment = enroll._wake_listener(mic, triggers, chime=lambda: chimes.append(1))()
    assert chimes == [1] and moment and len(np.concatenate(moment)) >= int(vpm.CHECK_SECONDS * 16000)


def test_speaking_during_the_warm_up_does_not_count(models_dir, spoken):
    from backend.voice.triggers import WakeWordDetector

    mic = FakeMic(np.concatenate([spoken["wake"], np.zeros((enroll.LISTEN_CHUNKS + 30) * CHUNK, np.int16)]))
    triggers = Triggers(WakeWordDetector(models_dir), None, config.WAKE_THRESHOLD, config.WAKE_PATIENCE)
    assert enroll._wake_listener(mic, triggers, chime=lambda: None)() is None
